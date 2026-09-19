"""
Inference service: loads the trained LSTM once and turns the live buffer of
candles into a forecast of the price 60 seconds ahead.

The predictor never raises on a bad input: it returns a structured object with
`available = False` and a human-readable message, so the backend and the
dashboard can show a warning instead of crashing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from app.config import (
    FEATURE_COLUMNS,
    FORECAST_HORIZON,
    INTERVAL_SECONDS,
    METRICS_PATH,
    MODEL_PATH,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    SYMBOL,
)
from app.data import StandardScaler, build_features
from app.model import PriceLSTM, load_model, pick_device
from app.utils import get_logger, load_json

log = get_logger("predictor")


@dataclass
class Prediction:
    """One forecast produced by the LSTM."""

    available: bool
    message: str = ""
    symbol: str = SYMBOL
    predicted_price: Optional[float] = None
    current_price: Optional[float] = None
    predicted_change_abs: Optional[float] = None
    predicted_change_pct: Optional[float] = None
    predicted_log_return: Optional[float] = None
    direction: Optional[str] = None
    horizon_seconds: int = INTERVAL_SECONDS * FORECAST_HORIZON
    target_time: Optional[str] = None
    generated_at: Optional[str] = None
    base_bar_time: Optional[str] = None
    inputs_used: int = 0
    model_device: str = "cpu"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LSTMPredictor:
    """Thread-safe-enough wrapper (single writer) around the trained network."""

    def __init__(self, device_preference: str = "auto") -> None:
        self.model: Optional[PriceLSTM] = None
        self.scaler: Optional[StandardScaler] = None
        self.device = pick_device(device_preference)
        self.metadata: Dict[str, Any] = {}
        self.load_error: Optional[str] = None
        self.loaded = False
        self._load()

    # ------------------------------------------------------------- loading
    def _load(self) -> None:
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            self.load_error = (
                "Trained model not found. Run:  python train_model.py"
            )
            log.warning(self.load_error)
            return
        try:
            self.model = load_model(MODEL_PATH, self.device)
            self.scaler = StandardScaler.load(SCALER_PATH)
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            self.metadata = checkpoint.get("extra", {})
            self.loaded = True
            log.info("LSTM loaded on %s (%s)", self.device, MODEL_PATH.name)
        except Exception as exc:                      # noqa: BLE001
            self.load_error = f"Could not load the model: {exc}"
            log.error(self.load_error)

    def reload(self) -> bool:
        """Pick up a freshly trained model without restarting the server."""
        self.model = None
        self.scaler = None
        self.loaded = False
        self.load_error = None
        self._load()
        return self.loaded

    # ---------------------------------------------------------- inference
    def predict(self, bars: pd.DataFrame, live_price: Optional[float] = None) -> Prediction:
        """
        Forecast the close price one bar (60 s) ahead.

        `bars` must contain at least SEQUENCE_LENGTH + ~65 rows of OHLCV data so
        that the rolling features (60-bar volume window) are all defined.
        """
        if not self.loaded or self.model is None or self.scaler is None:
            return Prediction(available=False,
                              message=self.load_error or "Model is not loaded")

        if bars is None or bars.empty:
            return Prediction(available=False, message="No market data available yet")

        required_columns = {"open_time", "open", "high", "low", "close", "volume"}
        missing = required_columns - set(bars.columns)
        if missing:
            return Prediction(
                available=False,
                message=f"Market data is missing the columns {sorted(missing)}",
            )

        needed = SEQUENCE_LENGTH + 65
        if len(bars) < needed:
            return Prediction(
                available=False,
                message=f"Collecting data: {len(bars)}/{needed} bars needed for a forecast",
                inputs_used=len(bars),
            )

        try:
            features = build_features(bars, with_target=False)
            if len(features) < SEQUENCE_LENGTH:
                return Prediction(
                    available=False,
                    message=f"Only {len(features)} usable rows after feature engineering, "
                            f"{SEQUENCE_LENGTH} required",
                )

            window = features.tail(SEQUENCE_LENGTH)
            X = window[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            if not np.isfinite(X).all():
                return Prediction(available=False,
                                  message="Market data contains gaps; skipping this forecast")

            X_scaled = self.scaler.transform(X).astype(np.float32)[None, :, :]

            with torch.no_grad():
                tensor = torch.from_numpy(X_scaled).to(self.device)
                scaled_output = float(self.model(tensor).cpu().numpy().ravel()[0])

            log_return = float(self.scaler.inverse_target(np.array([scaled_output]))[0])
            # Guard against an absurd extrapolation (more than 5% in one minute).
            log_return = float(np.clip(log_return, -0.05, 0.05))

            last_bar = window.iloc[-1]
            base_price = float(live_price) if live_price else float(last_bar["close"])
            predicted_price = base_price * float(np.exp(log_return))
            change_abs = predicted_price - base_price
            change_pct = (change_abs / base_price * 100.0) if base_price else 0.0

            if change_pct > 0.01:
                direction = "up"
            elif change_pct < -0.01:
                direction = "down"
            else:
                direction = "stable"

            # The window ends on bar T; the model predicts the close of bar T+1.
            # target_time is therefore the open_time of that next bar, which is
            # exactly the key the PredictionLedger uses to score the forecast.
            base_time = pd.to_datetime(last_bar["open_time"], utc=True).to_pydatetime()
            target_time = base_time + timedelta(seconds=INTERVAL_SECONDS * FORECAST_HORIZON)

            return Prediction(
                available=True,
                message="ok",
                predicted_price=predicted_price,
                current_price=base_price,
                predicted_change_abs=change_abs,
                predicted_change_pct=change_pct,
                predicted_log_return=log_return,
                direction=direction,
                target_time=target_time.isoformat(timespec="seconds"),
                generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                base_bar_time=base_time.isoformat(timespec="seconds"),
                inputs_used=SEQUENCE_LENGTH,
                model_device=str(self.device),
                extras={"sequence_length": SEQUENCE_LENGTH,
                        "features": FEATURE_COLUMNS},
            )

        except Exception as exc:                      # noqa: BLE001 - never crash the server
            log.exception("Prediction failed")
            return Prediction(available=False, message=f"Prediction error: {exc}")

    # ------------------------------------------------------------ metrics
    @staticmethod
    def offline_metrics() -> Dict[str, Any]:
        """The evaluation report produced by scripts/train_lstm.py."""
        payload = load_json(METRICS_PATH)
        if payload is None:
            return {"available": False,
                    "message": "No metrics file. Train the model first."}
        payload["available"] = True
        return payload


class PredictionLedger:
    """
    Keeps every forecast until the matching real price arrives, then scores it.

    This is what powers the live 'actual vs predicted' chart: at time t the
    model predicts the close of bar t+1; one minute later the real close is
    known and the pair becomes a resolved record.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self.maxlen = maxlen
        self._pending: Dict[pd.Timestamp, Dict[str, Any]] = {}
        self._resolved: List[Dict[str, Any]] = []

    def register(self, prediction: Prediction) -> None:
        if not prediction.available or prediction.target_time is None:
            return
        key = pd.Timestamp(prediction.target_time).tz_convert("UTC").floor("min")
        # Keep the first forecast made for each target bar (the earliest call).
        self._pending.setdefault(key, {
            "target_time": key.isoformat(),
            "predicted_price": prediction.predicted_price,
            "base_price": prediction.current_price,
            "predicted_direction": prediction.direction,
            "predicted_at": prediction.generated_at,
        })

    def resolve(self, bars: pd.DataFrame) -> int:
        """Match pending forecasts against closed bars. Returns how many resolved."""
        if bars.empty or not self._pending:
            return 0

        closed = bars.iloc[:-1] if len(bars) > 1 else bars       # last bar is still open
        actual = {pd.Timestamp(t).tz_convert("UTC").floor("min"): float(c)
                  for t, c in zip(closed["open_time"], closed["close"])}

        resolved_now = 0
        for key in sorted(k for k in self._pending if k in actual):
            record = self._pending.pop(key)
            actual_price = actual[key]
            predicted = float(record["predicted_price"])
            base = float(record["base_price"])

            # Direction is scored on the SIGN of the move, the same definition
            # used offline. Comparing the display label ("stable") against a
            # strict up/down label would understate the model unfairly.
            predicted_sign = float(np.sign(predicted - base))
            actual_sign = float(np.sign(actual_price - base))

            record.update({
                "actual_price": actual_price,
                "error": predicted - actual_price,
                "abs_error": abs(predicted - actual_price),
                "pct_error": abs(predicted - actual_price) / actual_price * 100.0,
                "actual_direction": ("up" if actual_sign > 0
                                     else "down" if actual_sign < 0 else "flat"),
                "predicted_sign": predicted_sign,
                "actual_sign": actual_sign,
                "direction_correct": bool(predicted_sign == actual_sign),
            })
            self._resolved.append(record)
            resolved_now += 1

        if len(self._resolved) > self.maxlen:
            self._resolved = self._resolved[-self.maxlen:]

        # Drop stale pending forecasts (older than 30 minutes).
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=30)
        for key in [k for k in self._pending if k < cutoff]:
            self._pending.pop(key, None)

        return resolved_now

    @property
    def resolved(self) -> List[Dict[str, Any]]:
        return list(self._resolved)

    def live_metrics(self) -> Dict[str, Any]:
        """MAE / RMSE / MAPE / direction accuracy over the live session."""
        if not self._resolved:
            return {"available": False, "resolved": 0, "pending": len(self._pending),
                    "message": "No forecast has matured yet - wait about one minute."}

        errors = np.array([r["error"] for r in self._resolved], dtype=np.float64)
        actual = np.array([r["actual_price"] for r in self._resolved], dtype=np.float64)
        correct = np.array([r["direction_correct"] for r in self._resolved], dtype=bool)

        return {
            "available": True,
            "resolved": int(len(self._resolved)),
            "pending": int(len(self._pending)),
            "MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MAPE": float(np.mean(np.abs(errors / actual)) * 100.0),
            "bias": float(np.mean(errors)),
            "directional_accuracy_pct": float(np.mean(correct) * 100.0),
        }


_predictor: Optional[LSTMPredictor] = None


def get_predictor() -> LSTMPredictor:
    global _predictor
    if _predictor is None:
        _predictor = LSTMPredictor()
    return _predictor
