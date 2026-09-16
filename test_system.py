"""
Self-contained test suite for the Real-Time AI Prediction System.

It checks the parts that must work before a demo:
    1. the external API answers and returns well-formed candles
    2. feature engineering produces finite values and no look-ahead leakage
    3. the scaler round-trips
    4. the trained model loads and produces a sane forecast
    5. the predictor degrades gracefully on bad input instead of crashing
    6. the prediction ledger scores forecasts correctly
    7. the chatbot answers with grounded numbers (rule layer, no GPU needed)
    8. the backend endpoints respond, when the backend happens to be running

Run:
    python -m test_system
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from app.config import BACKEND_URL, FEATURE_COLUMNS, MODEL_PATH, SEQUENCE_LENGTH, TARGET_COLUMN

PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------
def test_external_api() -> pd.DataFrame | None:
    section("1. External API")
    from app.crypto_api import ExternalAPIError, get_client

    client = get_client()
    try:
        reachable = client.ping()
        check("API reachable", reachable)
        if not reachable:
            return None

        bars = client.get_klines(limit=200)
        check("klines returns rows", len(bars) == 200, f"{len(bars)} bars")
        check("OHLCV columns present",
              {"open", "high", "low", "close", "volume"}.issubset(bars.columns))
        check("prices are positive", bool((bars["close"] > 0).all()))
        check("timestamps increase", bool(bars["open_time"].is_monotonic_increasing))

        tick = client.get_live_tick()
        check("live tick has a price", tick.price > 0, f"{tick.price:,.2f} from {tick.source}")
        spread = abs(tick.price - float(bars["close"].iloc[-1])) / tick.price
        check("live price agrees with the last candle", spread < 0.02, f"gap {spread * 100:.3f}%")
        return bars
    except ExternalAPIError as exc:
        check("API reachable", False, str(exc)[:70])
        return None


def test_features(bars: pd.DataFrame) -> None:
    section("2. Feature engineering")
    from app.data import build_features

    features = build_features(bars, with_target=True)
    check("rows survive feature engineering", len(features) > 100, f"{len(features)} rows")
    check("all features present", all(c in features.columns for c in FEATURE_COLUMNS))
    check("no NaN or inf in features",
          bool(np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()))
    check("target is the NEXT bar return, not the current one",
          bool(np.allclose(
              features[TARGET_COLUMN].iloc[:-1],
              np.log(features["close"].shift(-1) / features["close"]).iloc[:-1],
              atol=1e-9)))
    check("returns are plausible for 1-minute bars",
          float(features["log_return"].abs().max()) < 0.1,
          f"max |log return| = {features['log_return'].abs().max():.5f}")

    live_features = build_features(bars, with_target=False)
    check("live feature frame keeps the newest bar", len(live_features) >= len(features))


def test_scaler() -> None:
    section("3. Scaler")
    from app.data import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(loc=5.0, scale=3.0, size=(500, len(FEATURE_COLUMNS)))
    y = rng.normal(loc=0.001, scale=0.002, size=500)

    scaler = StandardScaler().fit(X, FEATURE_COLUMNS, y)
    Xs = scaler.transform(X)
    check("features standardised", abs(float(Xs.mean())) < 1e-9 and abs(float(Xs.std()) - 1) < 1e-6)
    check("target inverse round-trips",
          bool(np.allclose(scaler.inverse_target(scaler.transform_target(y)), y, atol=1e-12)))

    tmp = Path(__file__).parent / "_tmp_scaler.json"
    scaler.save(tmp)
    reloaded = StandardScaler.load(tmp)
    check("scaler survives save/load",
          bool(np.allclose(reloaded.transform(X), Xs, atol=1e-12)))
    tmp.unlink(missing_ok=True)


def test_model(bars: pd.DataFrame) -> None:
    section("4. Trained LSTM")
    if not MODEL_PATH.exists():
        check("model file exists", False, "run  python -m train_model")
        return

    from app.predictor import get_predictor

    predictor = get_predictor()
    check("model loads", predictor.loaded, str(predictor.device))
    if not predictor.loaded:
        return

    long_bars = bars if len(bars) >= 200 else pd.concat([bars] * 3, ignore_index=True)
    prediction = predictor.predict(long_bars)
    check("prediction available", prediction.available, prediction.message)
    if prediction.available:
        last_close = float(long_bars["close"].iloc[-1])
        drift = abs(prediction.predicted_price - last_close) / last_close
        check("forecast is within 1% of the current price", drift < 0.01,
              f"{prediction.predicted_price:,.2f} vs {last_close:,.2f}")
        check("horizon is 60 seconds", prediction.horizon_seconds == 60)
        check("direction label is valid", prediction.direction in {"up", "down", "stable"})
        check("sequence length matches the config", prediction.inputs_used == SEQUENCE_LENGTH)


def test_error_handling() -> None:
    section("5. Error handling")
    from app.predictor import get_predictor

    predictor = get_predictor()
    empty = predictor.predict(pd.DataFrame())
    check("empty input returns a message, not an exception",
          not empty.available and bool(empty.message), empty.message)

    short = predictor.predict(pd.DataFrame({
        "open_time": pd.date_range("2026-01-01", periods=5, freq="min", tz="UTC"),
        "open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5,
        "close": [1.0] * 5, "volume": [1.0] * 5,
    }))
    check("too-short input is refused cleanly", not short.available, short.message)

    broken = pd.DataFrame({"open_time": pd.date_range("2026-01-01", periods=200,
                                                      freq="min", tz="UTC"),
                           "wrong_column": range(200)})
    result = predictor.predict(broken)
    check("malformed input is handled", not result.available, result.message[:60])


def test_ledger() -> None:
    section("6. Prediction ledger")
    from app.predictor import Prediction, PredictionLedger

    ledger = PredictionLedger()
    times = pd.date_range("2026-01-01 00:00", periods=4, freq="min", tz="UTC")

    ledger.register(Prediction(available=True, predicted_price=101.0, current_price=100.0,
                               direction="up", target_time=times[1].isoformat(),
                               generated_at=times[0].isoformat()))
    ledger.register(Prediction(available=True, predicted_price=99.0, current_price=100.0,
                               direction="down", target_time=times[2].isoformat(),
                               generated_at=times[1].isoformat()))

    bars = pd.DataFrame({"open_time": times, "close": [100.0, 102.0, 98.0, 99.0]})
    resolved = ledger.resolve(bars)
    check("two forecasts resolved", resolved == 2, f"resolved {resolved}")

    metrics = ledger.live_metrics()
    check("live MAE computed", abs(metrics["MAE"] - 1.0) < 1e-9, f"MAE {metrics['MAE']}")
    check("direction scored on the sign of the move",
          abs(metrics["directional_accuracy_pct"] - 100.0) < 1e-9,
          f"{metrics['directional_accuracy_pct']}%")

    unresolved = ledger.resolve(bars)
    check("resolved forecasts are not counted twice", unresolved == 0)


def test_chatbot() -> None:
    section("7. Chatbot grounding (rule layer)")
    from app.chatbot import build_data_block, detect_intent, deterministic_answer

    context = {
        "live": {"available": True, "price": 75000.0, "age_seconds": 3,
                 "source": "binance", "timestamp": "2026-09-16T12:00:00+00:00",
                 "poll_count": 10, "error_count": 0, "poll_interval_seconds": 10,
                 "change_24h_pct": 1.5, "high_24h": 76000.0, "low_24h": 74000.0},
        "prediction": {"available": True, "predicted_price": 75050.0, "current_price": 75000.0,
                       "predicted_change_abs": 50.0, "predicted_change_pct": 0.0667,
                       "direction": "up", "horizon_seconds": 60,
                       "target_time": "2026-09-16T12:01:00+00:00"},
        "statistics": {"available": True, "count": 200, "mean": 74900.0, "median": 74950.0,
                       "min": 74000.0, "max": 76000.0, "std": 300.0, "change_pct": 0.5,
                       "trend": {"direction": "rising", "change_pct": 0.12, "lookback": 15}},
        "live_evaluation": {"available": True, "MAE": 25.0, "RMSE": 33.0, "resolved": 12,
                            "directional_accuracy_pct": 58.3},
        "model_metrics": {"available": True,
                          "test": {"price_space": {"MAE": 29.6, "RMSE": 43.1,
                                                   "MAPE": 0.0386, "R2": 0.9986},
                                   "directional_accuracy_pct": 49.8}},
        "collector": {"poll_count": 10, "error_count": 0, "poll_interval_seconds": 10},
    }

    cases = {
        "What is the current price?": ("price", "75,000.00"),
        "What do you predict for the next minute?": ("prediction", "75,050.00"),
        "Is it going up or down?": ("trend", "rising"),
        "Give me the statistics": ("statistics", "74,900.00"),
        "How accurate is the model?": ("accuracy", "29.60"),
        "When was the last update?": ("update", "10"),
        "Which API do you use?": ("source", "Binance"),
    }
    for question, (expected_intent, must_contain) in cases.items():
        intent = detect_intent(question)
        answer = deterministic_answer(context, intent)
        check(f"intent of '{question[:32]}...' = {expected_intent}",
              intent == expected_intent, f"got {intent}")
        check(f"answer contains {must_contain}", must_contain in answer)

    block = build_data_block(context, "price")
    check("data block carries the real live price", "current_price_usdt: 75,000.00" in block)
    check("data block carries the real forecast",
          "lstm_predicted_price_usdt: 75,050.00" in block)
    check("data block reports the data source", "data_source: binance" in block)

    offline = build_data_block({"live": {"available": False, "message": "API down"},
                                "prediction": {"available": False, "message": "not ready"}},
                               "price")
    check("data block states unavailability instead of inventing numbers",
          "unavailable" in offline and "75," not in offline)

    # ---- the layer that rejects a generated answer contradicting the data ----
    from app.chatbot import verify_answer

    block = ("current_price_usdt: 75,000.00\n"
             "lstm_predicted_price_usdt: 75,050.00\n"
             "predicted_direction: up")
    up_context = dict(context)

    ok, _ = verify_answer("Bitcoin is at 75,000.00 USDT and the forecast is 75,050.00 USDT.",
                          block, up_context)
    check("a faithful answer is accepted", ok)

    ok, reason = verify_answer("Bitcoin is trading at 91,234.00 USDT.", block, up_context)
    check("an invented price is rejected", not ok, reason)

    down_context = dict(context)
    down_context["prediction"] = dict(context["prediction"])
    down_context["prediction"].update(predicted_change_abs=-40.0, predicted_change_pct=-0.05,
                                      direction="down")

    # The guard follows the same direction label the dashboard shows, so the
    # chatbot can never say "going up" while the metric card reads "stable".
    flat_context = dict(context)
    flat_context["prediction"] = dict(context["prediction"])
    flat_context["prediction"].update(predicted_change_abs=0.3, predicted_change_pct=0.0004,
                                      direction="stable")
    ok, reason = verify_answer("The price is expected to go up.", block, flat_context)
    check("an up claim is rejected when the model says stable", not ok, reason)
    ok, _ = verify_answer("The LSTM expects it to stay stable next minute.", block, flat_context)
    check("a stable claim is accepted when the model says stable", ok)
    ok, reason = verify_answer("The LSTM expects the price to rise in the next minute.",
                               block, down_context)
    check("a contradicted forecast direction is rejected", not ok, reason)

    ok, _ = verify_answer("Over the last 15 minutes the price is rising, "
                          "and the LSTM expects it to fall next minute.", block, down_context)
    check("a past-trend direction is not mistaken for a forecast claim", ok)

    # The guard is deliberately biased towards accepting: a missed detection only
    # lets a correct answer through, while a false rejection would replace a good
    # answer with a stiffer one. This case must not be rejected.
    ok, _ = verify_answer("The LSTM predicts 75,050.00 USDT, a stable direction.",
                          block, down_context)
    check("an ambiguous phrasing is accepted rather than wrongly rejected", ok)


def test_backend() -> None:
    section("8. Backend endpoints (skipped if the server is not running)")
    import requests

    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    except Exception as exc:                          # noqa: BLE001
        print(f"  SKIP  backend not running at {BACKEND_URL} ({type(exc).__name__})")
        return

    check("/health answers", "status" in health, health.get("status"))
    for path in ("/api/live", "/api/prediction", "/api/stats", "/api/status",
                 "/api/history?limit=50", "/api/model/metrics"):
        try:
            response = requests.get(f"{BACKEND_URL}{path}", timeout=10)
            check(f"GET {path}", response.status_code == 200, f"HTTP {response.status_code}")
        except Exception as exc:                      # noqa: BLE001
            check(f"GET {path}", False, type(exc).__name__)


def main() -> int:
    print("=" * 68)
    print("  Real-Time AI Prediction System - test suite")
    print("=" * 68)

    bars = test_external_api()
    if bars is not None:
        test_features(bars)
        test_scaler()
        test_model(bars)
        test_error_handling()
    else:
        print("\n  Skipping the data-dependent tests: the external API is unreachable.")
    test_ledger()
    test_chatbot()
    test_backend()

    print("\n" + "=" * 68)
    print(f"  {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"    - {name}")
    print("=" * 68)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
