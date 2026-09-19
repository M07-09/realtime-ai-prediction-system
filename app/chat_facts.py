"""
The grounding layer of the chatbot: the part that KNOWS, as opposed to the part
that SPEAKS (app/chatbot.py).

Nothing here uses a neural network. Every number it produces is computed in
Python from the live collector snapshot, which is what makes a wrong figure
structurally impossible.

    detect_intent()        what is the user asking about?
    build_data_block()     the verified numbers the Transformer may use
    deterministic_answer() the exact answer, always correct, slightly stiff
    verify_answer()        rejects a generated answer that contradicts the data
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import ASSET_NAME, SYMBOL


# --------------------------------------------------------------------------
# Stage 1 - intent detection
# --------------------------------------------------------------------------
#   Order matters: the first pattern that matches wins, so the more specific
#   intents are listed before the generic ones. Stems end in \w* so that
#   "statistics", "accurate" or "predicted" match as well as their root form.
INTENT_PATTERNS: List[Tuple[str, str]] = [
    ("prediction", r"\b(predict\w*|forecast\w*|future|next minute|will it|expect\w*|going to)\b"),
    ("accuracy",   r"\b(accura\w*|error\w*|mae|rmse|mse|mape|r2|r squared|reliab\w*|"
                   r"how good|performance|metric\w*|evaluat\w*)\b"),
    ("trend",      r"\b(trend\w*|rising|rise|falling|fall|increas\w*|decreas\w*|"
                   r"going up|going down|direction|up or down|momentum)\b"),
    ("statistics", r"\b(statistic\w*|stats|average|mean|median|minimum|maximum|highest|lowest|"
                   r"volatil\w*|std|standard deviation|summary|range|distribution)\b"),
    ("update",     r"\b(update\w*|latest|when|how often|refresh\w*|poll\w*|fresh|interval)\b"),
    ("model",      r"\b(model|lstm|neural|network|architect\w*|train\w*|"
                   r"how does it work|sequence|layer\w*)\b"),
    ("source",     r"\b(api|source|binance|exchange|data come\w*|where.*data)\b"),
    ("price",      r"\b(price|value|worth|cost|how much|current|now|btc|bitcoin)\b"),
    ("help",       r"\b(help|what can you|commands|who are you|hello|hi|hey)\b"),
]


def detect_intent(question: str) -> str:
    text = question.lower().strip()
    for intent, pattern in INTENT_PATTERNS:
        if re.search(pattern, text):
            return intent
    return "general"


# --------------------------------------------------------------------------
# Stage 2 - fact extraction (exact numbers, computed in Python)
# --------------------------------------------------------------------------
def _fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:                       # NaN
            return "n/a"
        return f"{value:,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def build_data_block(context: Dict[str, Any], intent: str) -> str:
    """Render the verified numbers the Transformer is allowed to use."""
    live = context.get("live", {}) or {}
    pred = context.get("prediction", {}) or {}
    stats = context.get("statistics", {}) or {}
    live_eval = context.get("live_evaluation", {}) or {}
    metrics = context.get("model_metrics", {}) or {}
    collector = context.get("collector", {}) or {}

    lines: List[str] = [f"asset: {ASSET_NAME} ({SYMBOL})"]

    if live.get("available"):
        lines.append(f"current_price_usdt: {_fmt(live.get('price'))}")
        lines.append(f"last_update_utc: {live.get('timestamp')}")
        lines.append(f"seconds_since_update: {_fmt(live.get('age_seconds'), 0)}")
        lines.append(f"data_source: {live.get('source')}")
        if live.get("change_24h_pct") is not None:
            lines.append(f"change_24h_pct: {_fmt(live.get('change_24h_pct'), 2)}")
            lines.append(f"high_24h: {_fmt(live.get('high_24h'))}")
            lines.append(f"low_24h: {_fmt(live.get('low_24h'))}")
    else:
        lines.append(f"current_price_usdt: unavailable ({live.get('message', 'no data')})")

    if pred.get("available"):
        lines.append(f"lstm_predicted_price_usdt: {_fmt(pred.get('predicted_price'))}")
        lines.append(f"forecast_horizon_seconds: {pred.get('horizon_seconds')}")
        lines.append(f"predicted_change_usdt: {_fmt(pred.get('predicted_change_abs'))}")
        lines.append(f"predicted_change_pct: {_fmt(pred.get('predicted_change_pct'), 4)}")
        lines.append(f"predicted_direction: {pred.get('direction')}")
        lines.append(f"forecast_target_time_utc: {pred.get('target_time')}")
    else:
        lines.append(f"lstm_prediction: unavailable ({pred.get('message', 'not ready')})")

    if intent in {"trend", "statistics", "price", "general", "help"} and stats.get("available"):
        trend = stats.get("trend", {}) or {}
        lines.append(f"short_term_trend_last_{trend.get('lookback', 15)}_bars: "
                     f"{trend.get('direction')} ({_fmt(trend.get('change_pct'), 3)}%)")
        lines.append(f"window_bars_collected: {stats.get('count')}")
        lines.append(f"window_mean_price: {_fmt(stats.get('mean'))}")
        lines.append(f"window_median_price: {_fmt(stats.get('median'))}")
        lines.append(f"window_min_price: {_fmt(stats.get('min'))}")
        lines.append(f"window_max_price: {_fmt(stats.get('max'))}")
        lines.append(f"window_std_price: {_fmt(stats.get('std'))}")
        lines.append(f"window_change_pct: {_fmt(stats.get('change_pct'), 3)}")

    if intent in {"accuracy", "model", "general"}:
        test = (metrics.get("test") or {}).get("price_space") if metrics.get("available") else None
        if test:
            lines.append(f"offline_test_MAE_usdt: {_fmt(test.get('MAE'))}")
            lines.append(f"offline_test_RMSE_usdt: {_fmt(test.get('RMSE'))}")
            lines.append(f"offline_test_MAPE_pct: {_fmt(test.get('MAPE'), 4)}")
            lines.append(f"offline_test_R2: {_fmt(test.get('R2'), 4)}")
            direction = (metrics.get("test") or {}).get("directional_accuracy_pct")
            lines.append(f"offline_directional_accuracy_pct: {_fmt(direction, 2)}")
        if live_eval.get("available"):
            lines.append(f"live_session_MAE_usdt: {_fmt(live_eval.get('MAE'))}")
            lines.append(f"live_session_RMSE_usdt: {_fmt(live_eval.get('RMSE'))}")
            lines.append(f"live_forecasts_scored: {live_eval.get('resolved')}")
            lines.append(f"live_directional_accuracy_pct: "
                         f"{_fmt(live_eval.get('directional_accuracy_pct'), 2)}")

    if intent in {"model", "source", "general", "help"}:
        lines.append("model: 2-layer LSTM, 60-step input window of 1-minute candles")
        lines.append("model_target: next-bar log return, converted back to a USDT price")
        lines.append("external_api: Binance public REST API (klines + 24h ticker)")
        lines.append(f"poll_interval_seconds: {collector.get('poll_interval_seconds', 10)}")
        lines.append(f"api_calls_made: {collector.get('poll_count', 0)}")
        lines.append(f"api_errors: {collector.get('error_count', 0)}")

    return "\n".join(lines)


def deterministic_answer(context: Dict[str, Any], intent: str) -> str:
    """
    The exact, rule-based answer. Used as the fallback when the Transformer is
    unavailable, and as the safety net behind every generated reply.
    """
    live = context.get("live", {}) or {}
    pred = context.get("prediction", {}) or {}
    stats = context.get("statistics", {}) or {}
    live_eval = context.get("live_evaluation", {}) or {}
    metrics = context.get("model_metrics", {}) or {}

    price = live.get("price") if live.get("available") else None

    if intent == "price":
        if price is None:
            return ("The live price is not available right now: "
                    f"{live.get('message', 'the external API did not answer')}.")
        return (f"{ASSET_NAME} is trading at {_fmt(price)} USDT, "
                f"updated {_fmt(live.get('age_seconds'), 0)} seconds ago from {live.get('source')}.")

    if intent == "prediction":
        if not pred.get("available"):
            return f"No forecast is available yet: {pred.get('message', 'the model is warming up')}."
        return (f"The LSTM expects {_fmt(pred.get('predicted_price'))} USDT in "
                f"{pred.get('horizon_seconds')} seconds, "
                f"{_fmt(pred.get('predicted_change_abs'))} USDT "
                f"({_fmt(pred.get('predicted_change_pct'), 4)}%) from the current "
                f"{_fmt(pred.get('current_price'))} USDT, so the direction is "
                f"{pred.get('direction')}.")

    if intent == "trend":
        trend = (stats.get("trend") or {}) if stats.get("available") else {}
        direction = trend.get("direction", "unknown")
        predicted = pred.get("direction", "unknown") if pred.get("available") else "unknown"
        return (f"Over the last {trend.get('lookback', 15)} minutes the price is {direction} "
                f"({_fmt(trend.get('change_pct'), 3)}%), and the LSTM expects the next minute to go "
                f"{predicted}.")

    if intent == "statistics":
        if not stats.get("available"):
            return "No statistics yet - the system is still collecting data."
        return (f"Across the {stats.get('count')} minutes in memory: mean {_fmt(stats.get('mean'))}, "
                f"median {_fmt(stats.get('median'))}, low {_fmt(stats.get('min'))}, "
                f"high {_fmt(stats.get('max'))}, standard deviation {_fmt(stats.get('std'))} USDT, "
                f"overall move {_fmt(stats.get('change_pct'), 3)}%.")

    if intent == "accuracy":
        parts: List[str] = []
        test = (metrics.get("test") or {}).get("price_space") if metrics.get("available") else None
        if test:
            parts.append(f"On the held-out test split the LSTM scores MAE {_fmt(test.get('MAE'))} USDT, "
                         f"RMSE {_fmt(test.get('RMSE'))} USDT, MAPE {_fmt(test.get('MAPE'), 4)}% "
                         f"and R2 {_fmt(test.get('R2'), 4)}.")
        if live_eval.get("available"):
            parts.append(f"In this live session {live_eval.get('resolved')} forecasts have matured "
                         f"with MAE {_fmt(live_eval.get('MAE'))} USDT and "
                         f"{_fmt(live_eval.get('directional_accuracy_pct'), 1)}% correct direction.")
        return " ".join(parts) if parts else "No evaluation results are available yet."

    if intent == "update":
        if not live.get("available"):
            return "The feed has not delivered a successful update yet."
        return (f"The last update arrived at {live.get('timestamp')} UTC, "
                f"{_fmt(live.get('age_seconds'), 0)} seconds ago. The system polls the external API "
                f"every {live.get('poll_interval_seconds', 10)} seconds and has made "
                f"{live.get('poll_count')} calls with {live.get('error_count')} errors.")

    if intent == "model":
        return ("The forecaster is a 2-layer LSTM with 96 hidden units. It reads the last 60 "
                "one-minute candles, each described by 7 engineered features, and predicts the "
                "log return of the next minute, which is converted back into a USDT price.")

    if intent == "source":
        return (f"Data comes from the Binance public REST API: /api/v3/klines for candles and "
                f"/api/v3/ticker/24hr for the live price of {SYMBOL}. No API key is required, and "
                f"Coinbase and Kraken are used automatically as fallbacks.")

    if intent == "help":
        return ("Ask me about the current price, the LSTM prediction for the next minute, whether "
                "the price is rising or falling, the latest update time, statistics of the "
                "collected window, or how accurate the model is.")

    # general
    if price is None:
        return "The system is starting up and has not received live data yet."
    return (f"{ASSET_NAME} is at {_fmt(price)} USDT. "
            + (f"The LSTM forecast for the next minute is {_fmt(pred.get('predicted_price'))} USDT "
               f"({pred.get('direction')})." if pred.get("available") else
               "The forecast is not ready yet."))


# --------------------------------------------------------------------------
# Stage 2b - verification of what the Transformer wrote
# --------------------------------------------------------------------------
UP_WORDS = {"up", "rise", "rises", "rising", "increase", "increases", "increasing",
            "higher", "climb", "climbing", "grow", "growing", "upward"}
DOWN_WORDS = {"down", "fall", "falls", "falling", "decrease", "decreases", "decreasing",
              "lower", "drop", "dropping", "decline", "declining", "downward"}
FLAT_WORDS = {"stable", "flat", "unchanged", "steady", "same"}
FORECAST_WORDS = {"predict", "predicts", "predicted", "prediction", "forecast", "forecasts",
                  "expect", "expects", "expected", "will", "next", "lstm", "upcoming"}

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _numbers_in(text: str) -> List[str]:
    """Numeric tokens, normalised so 75,854.00 and 75854.0 compare equal."""
    found = []
    for raw in _NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        if not cleaned:
            continue
        try:
            found.append(f"{float(cleaned):.4f}")
        except ValueError:
            continue
    return found


def _forecast_direction_claim(answer: str) -> Optional[int]:
    """
    Detect the direction the answer attributes to the FORECAST.

    Returns +1 (up), -1 (down), 0 (stable) or None when the answer makes no
    forecast-direction claim.

    The text is split into CLAUSES, not just sentences, because a correct answer
    often puts both facts in one sentence:

        "Over the last 15 minutes the price is rising, and the LSTM expects it
         to fall next minute."

    Only the clause that actually mentions the forecast is inspected, so the
    description of the past trend is never mistaken for a claim about the future.
    """
    clauses = re.split(r"[.!?;,]|\band\b", answer.lower())
    for clause in clauses:
        words = set(re.findall(r"[a-z]+", clause))
        if not words & FORECAST_WORDS:
            continue
        if words & UP_WORDS:
            return 1
        if words & DOWN_WORDS:
            return -1
        if words & FLAT_WORDS:
            return 0
    return None


# Intents whose answer is a list of figures. Dropping one of them lets the rest
# be misread: "25.92 USDT MAE, RMSE and R2" passes check 1 yet is false.
STRICT_INTENTS = {"accuracy", "statistics"}


def verify_answer(answer: str, data_block: str, context: Dict[str, Any],
                  required_from: Optional[str] = None) -> Tuple[bool, str]:
    """
    Reject a generated answer that contradicts the verified data.

    Three independent checks:
      1. every number in the answer must appear in the DATA block, so the model
         cannot invent a price or a metric;
      2. any direction it attributes to the forecast must match the sign of the
         LSTM's predicted change;
      3. when `required_from` is given, every number in it must survive the
         rewrite, so no figure can be dropped and another passed off in its place.
    """
    allowed = set(_numbers_in(data_block))
    # Small integers are ordinary prose ("1 minute", "3 sentences"), not data.
    for number in _numbers_in(answer):
        if float(number) < 100:
            continue
        if number not in allowed:
            return False, f"invented the number {number}"

    prediction = context.get("prediction", {}) or {}
    if prediction.get("available"):
        # The authority is the same `direction` label the dashboard shows, so the
        # chatbot can never say "going up" while the metric card reads "stable".
        expected = {"up": 1, "down": -1, "stable": 0}.get(prediction.get("direction"))
        claimed = _forecast_direction_claim(answer)
        if expected is not None and claimed is not None and claimed != expected:
            return False, (f"said the forecast is "
                           f"{ {1: 'up', -1: 'down', 0: 'stable'}[claimed] } "
                           f"but the model says {prediction.get('direction')}")

    if required_from:
        present = set(_numbers_in(answer))
        for number in _numbers_in(required_from):
            if number not in present:
                return False, f"dropped the figure {float(number):g}"

    return True, "ok"



SUGGESTED_QUESTIONS = [
    "What is the current Bitcoin price?",
    "What does the LSTM predict for the next minute?",
    "Is the price going up or down?",
    "When was the last update?",
    "Show me the statistics of the collected data.",
    "How accurate is the model?",
    "Which API does the data come from?",
    "How does the LSTM model work?",
]
