"""
Streamlit dashboard - the user interface of the system.

It never touches the external API or the model directly: everything comes from
the FastAPI backend, which keeps the architecture clean

    External API -> FastAPI -> processing -> LSTM -> prediction -> Streamlit

Auto-update
-----------
The live panels live inside an st.fragment that re-runs every 10 seconds, so
the numbers, the charts and the forecast refresh on their own while the chat
conversation on the right-hand side is preserved.

Run:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests
import streamlit as st

from app.charts import (actual_vs_predicted_chart, error_chart, money, price_chart,
                        render_table, tick_chart)
from app.config import ASSET_NAME, BACKEND_URL, MODELS_DIR, POLL_INTERVAL_SECONDS, SYMBOL

st.set_page_config(
    page_title="Real-Time AI Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
  div[data-testid="stMetricValue"] {font-size: 1.7rem;}
  .status-ok   {color:#12864a; font-weight:600;}
  .status-warn {color:#b45309; font-weight:600;}
  .status-bad  {color:#b91c1c; font-weight:600;}
  .small-note  {color:#6b7280; font-size:0.82rem;}
  .fact-box {background:rgba(128,128,128,0.12); border:1px solid rgba(128,128,128,0.32);
             border-radius:8px; color:inherit;
             padding:0.6rem 0.8rem; font-family:ui-monospace,monospace; font-size:0.78rem;
             white-space:pre-wrap;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Backend access - every call degrades to a message instead of an exception
# --------------------------------------------------------------------------
def api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 12) -> Dict[str, Any]:
    try:
        response = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=timeout)
        response.raise_for_status()
        return {"ok": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "connection",
                "message": f"Cannot reach the backend at {BACKEND_URL}. "
                           "Start it with:  python -m app.main"}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "timeout", "message": "The backend did not answer in time."}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "error": "http", "message": f"Backend returned {exc.response.status_code}."}
    except Exception as exc:                          # noqa: BLE001
        return {"ok": False, "error": "unknown", "message": str(exc)}


def api_post(path: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
    try:
        response = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return {"ok": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "message": f"Cannot reach the backend at {BACKEND_URL}."}
    except requests.exceptions.Timeout:
        return {"ok": False, "message": "The chatbot took too long to answer. Try again."}
    except Exception as exc:                          # noqa: BLE001
        return {"ok": False, "message": str(exc)}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.header("System")
        health = api_get("/health", timeout=8)

        if not health["ok"]:
            st.error(health["message"])
            st.caption("The dashboard keeps running and will reconnect automatically.")
            return

        data = health["data"]
        status = data["status"]
        if status == "healthy":
            st.success(f"Backend: {status}")
        else:
            st.warning(f"Backend: {status}")

        st.metric("API calls made", data["poll_count"])
        col_a, col_b = st.columns(2)
        col_a.metric("Errors", data["error_count"])
        col_b.metric("Uptime", f"{data['uptime_seconds'] / 60:.1f} min")

        st.markdown(
            f"- External API: {'reachable' if data['external_api_reachable'] else 'unreachable'}\n"
            f"- Collector: {'running' if data['collector_running'] else 'stopped'}\n"
            f"- LSTM model: {'loaded' if data['model_loaded'] else 'not loaded'}\n"
            f"- Transformer: {'loaded' if data['chatbot_loaded'] else 'loads on first question'}"
        )
        if data.get("last_error"):
            st.warning(f"Last error: {data['last_error']}")

        st.divider()
        st.header("Architecture")
        st.code(
            "Binance API\n"
            "     |\n"
            "  FastAPI  (polls every 10 s)\n"
            "     |\n"
            "  Processing (features)\n"
            "     |\n"
            "   LSTM  ->  prediction\n"
            "     |\n"
            "Streamlit + Transformer chat",
            language="text",
        )

        st.divider()
        if st.button("Force refresh now", width="stretch"):
            result = api_post("/api/refresh", {}, timeout=30)
            if result["ok"]:
                st.success("Polled the external API")
            else:
                st.error(result["message"])

        st.caption(f"Backend: {BACKEND_URL}")


# --------------------------------------------------------------------------
# Live section (auto-refreshing fragment)
# --------------------------------------------------------------------------
@st.fragment(run_every=POLL_INTERVAL_SECONDS)
def live_section() -> None:
    result = api_get("/api/status", {"history_limit": 240})

    if not result["ok"]:
        st.error(f"⚠️ {result['message']}")
        st.info("This panel retries every 10 seconds on its own.")
        return

    status = result["data"]
    live = status.get("live", {})
    prediction = status.get("prediction", {})
    stats = status.get("statistics", {})
    collector = status.get("collector", {})

    # ---------------------------------------------------------- headline
    if not live.get("available"):
        st.error(f"⚠️ No live data: {live.get('message', 'the external API did not answer')}")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(f"{ASSET_NAME} price",
                  money(live.get("price")),
                  f"{live.get('tick_change', 0):+,.2f} since last poll")

        if prediction.get("available"):
            arrow = {"up": "▲", "down": "▼", "stable": "■"}.get(prediction.get("direction"), "")
            c2.metric("LSTM forecast (+60 s)",
                      money(prediction.get("predicted_price")),
                      f"{arrow} {prediction.get('predicted_change_abs', 0):+,.2f} "
                      f"({prediction.get('predicted_change_pct', 0):+.4f}%)")
        else:
            c2.metric("LSTM forecast (+60 s)", "n/a")
            c2.caption(prediction.get("message", ""))

        c3.metric("24h change", f"{live.get('change_24h_pct', 0):+.2f}%",
                  f"H {money(live.get('high_24h'), 0)} / L {money(live.get('low_24h'), 0)}")

        trend = (stats.get("trend") or {}) if stats.get("available") else {}
        c4.metric(f"Trend ({trend.get('lookback', 15)} min)",
                  str(trend.get("direction", "unknown")).capitalize(),
                  f"{trend.get('change_pct', 0):+.3f}%")

        age = live.get("age_seconds", 0) or 0
        freshness = "fresh" if age < 25 else "stale"
        c5.metric("Last update", f"{age:.0f} s ago", freshness)

        st.caption(
            f"Source: {live.get('source')} · bars in memory: {live.get('buffer_bars')} · "
            f"API calls: {collector.get('poll_count')} (errors: {collector.get('error_count')}) · "
            f"server time {status.get('server_time')} UTC · auto-refresh every "
            f"{POLL_INTERVAL_SECONDS} s"
        )

    if collector.get("consecutive_errors", 0) > 0:
        st.warning(f"The external API has failed {collector['consecutive_errors']} time(s) in a row: "
                   f"{collector.get('last_error')}. Showing the last known values.")

    # ------------------------------------------------------------- charts
    history = status.get("history", [])
    if history:
        st.plotly_chart(price_chart(history, prediction), width="stretch",
                        key=f"price_{collector.get('poll_count')}")
    else:
        st.info("Waiting for the first candles ...")

    left, right = st.columns([3, 2])

    resolved = status.get("resolved_predictions", [])
    live_eval = status.get("live_evaluation", {})
    with left:
        if resolved:
            st.plotly_chart(actual_vs_predicted_chart(resolved), width="stretch",
                            key=f"avp_{collector.get('poll_count')}")
            st.plotly_chart(error_chart(resolved), width="stretch",
                            key=f"err_{collector.get('poll_count')}")
        else:
            st.info(f"⏳ {live_eval.get('message', 'Scoring starts once a forecast matures.')}")
            ticks = status.get("ticks", [])
            if ticks:
                st.plotly_chart(tick_chart(ticks), width="stretch",
                                key=f"tick_{collector.get('poll_count')}")

    with right:
        st.subheader("Live scoring")
        if live_eval.get("available"):
            m1, m2 = st.columns(2)
            m1.metric("MAE", money(live_eval.get("MAE")))
            m2.metric("RMSE", money(live_eval.get("RMSE")))
            m3, m4 = st.columns(2)
            m3.metric("MAPE", f"{live_eval.get('MAPE', 0):.4f}%")
            m4.metric("Direction", f"{live_eval.get('directional_accuracy_pct', 0):.1f}%")
            st.caption(f"{live_eval.get('resolved')} forecasts scored, "
                       f"{live_eval.get('pending')} waiting to mature.")
        else:
            st.caption(live_eval.get("message", "Waiting for the first matured forecast."))

        st.subheader("Collected window")
        if stats.get("available"):
            render_table(pd.DataFrame({
                "metric": ["bars", "mean", "median", "min", "max",
                           "std dev", "change %", "1-min volatility %"],
                "value": [
                    f"{stats.get('count')}",
                    money(stats.get("mean")), money(stats.get("median")),
                    money(stats.get("min")), money(stats.get("max")),
                    money(stats.get("std")),
                    f"{stats.get('change_pct', 0):+.3f}%",
                    f"{stats.get('volatility_pct', 0):.4f}%",
                ],
            }))
        else:
            st.caption("No statistics yet.")

    if history:
        with st.expander("Raw data received from the external API (latest bars)"):
            table = pd.DataFrame(history).tail(25)[
                ["open_time", "open", "high", "low", "close", "volume", "trades"]
            ].iloc[::-1]
            table["open_time"] = table["open_time"].str.replace("+00:00", " UTC", regex=False)
            render_table(table, height=420)


# --------------------------------------------------------------------------
# Model evaluation section
# --------------------------------------------------------------------------
def model_section() -> None:
    result = api_get("/api/model/metrics")
    if not result["ok"]:
        st.error(result["message"])
        return

    metrics = result["data"]
    if not metrics.get("available"):
        st.warning(metrics.get("message", "No metrics available."))
        st.code("python -m train_model", language="bash")
        return

    test = metrics.get("test", {})
    price = test.get("price_space", {})
    naive = test.get("naive_baseline_price_space", {})
    data_info = metrics.get("data", {})
    training = metrics.get("training", {})

    st.subheader("Offline evaluation on the held-out test split")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", money(price.get("MAE")))
    c2.metric("RMSE", money(price.get("RMSE")))
    c3.metric("MSE", f"{price.get('MSE', 0):,.1f}")
    c4.metric("MAPE", f"{price.get('MAPE', 0):.4f}%")
    c5.metric("R²", f"{price.get('R2', 0):.4f}")

    st.caption(
        f"Test samples: {test.get('samples'):,} · directional accuracy "
        f"{test.get('directional_accuracy_pct', 0):.2f}% · naive 'no-change' baseline RMSE "
        f"{money(naive.get('RMSE'))} · trained on {data_info.get('total_bars'):,} candles "
        f"({data_info.get('period_start', '')[:16]} → {data_info.get('period_end', '')[:16]} UTC) "
        f"in {training.get('training_seconds', 0):.1f}s on {training.get('device')}."
    )

    with st.expander("Validation split and full metric report"):
        val = (metrics.get("validation") or {}).get("price_space", {})
        def fmt(value, digits=4):
            return "n/a" if value is None else f"{value:,.{digits}f}"

        render_table(pd.DataFrame({
            "metric": ["MAE (USDT)", "MSE", "RMSE (USDT)", "MAPE %", "R²"],
            "test": [fmt(price.get("MAE"), 2), fmt(price.get("MSE"), 1),
                     fmt(price.get("RMSE"), 2), fmt(price.get("MAPE"), 4),
                     fmt(price.get("R2"), 4)],
            "validation": [fmt(val.get("MAE"), 2), fmt(val.get("MSE"), 1),
                           fmt(val.get("RMSE"), 2), fmt(val.get("MAPE"), 4),
                           fmt(val.get("R2"), 4)],
            "naive baseline (test)": [fmt(naive.get("MAE"), 2), fmt(naive.get("MSE"), 1),
                                      fmt(naive.get("RMSE"), 2), fmt(naive.get("MAPE"), 4),
                                      fmt(naive.get("R2"), 4)],
        }))
        st.json(metrics, expanded=False)

    st.subheader("Training and evaluation plots")
    plots = [
        ("1_historical_prices.png", "Historical data used for training"),
        ("2_loss_curves.png", "Training and validation loss"),
        ("3_actual_vs_predicted.png", "Actual vs predicted price (test split)"),
        ("4_prediction_scatter.png", "Predicted vs actual scatter"),
        ("5_error_distribution.png", "Error distribution"),
    ]
    available = [(MODELS_DIR / "plots" / name, caption)
                 for name, caption in plots if (MODELS_DIR / "plots" / name).exists()]
    if not available:
        st.info("No plots found. Run the training script to generate them.")
        return
    for index in range(0, len(available), 2):
        cols = st.columns(2)
        for col, (path, caption) in zip(cols, available[index:index + 2]):
            col.image(str(path), caption=caption, width="stretch")


# --------------------------------------------------------------------------
# Chatbot section
# --------------------------------------------------------------------------
def chatbot_section() -> None:
    st.subheader("Transformer chatbot")
    st.caption("The chatbot reads the same live snapshot as the dashboard. The numbers are "
               "computed in Python and handed to the Transformer, which writes the answer.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    suggestions = api_get("/api/chat/suggestions", timeout=8)
    if suggestions["ok"]:
        options = suggestions["data"].get("suggestions", [])
        st.write("Try one of these:")
        for row_start in range(0, len(options), 2):
            cols = st.columns(2)
            for col, question in zip(cols, options[row_start:row_start + 2]):
                if col.button(question, key=f"sugg_{question}", width="stretch"):
                    st.session_state.pending_question = question

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                st.caption(turn["meta"])
            if turn.get("facts"):
                with st.expander("Verified data given to the Transformer"):
                    st.markdown(f"<div class='fact-box'>{turn['facts']}</div>",
                                unsafe_allow_html=True)

    typed = st.chat_input("Ask about the price, the forecast, the trend or the model ...")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("The Transformer is reading the live data ..."):
            payload = {
                "question": question,
                "history": [{"role": t["role"], "content": t["content"]}
                            for t in st.session_state.chat_history[-8:-1]],
            }
            result = api_post("/api/chat", payload, timeout=240)

        if not result["ok"]:
            message = f"⚠️ {result['message']}"
            st.error(message)
            st.session_state.chat_history.append({"role": "assistant", "content": message})
            return

        data = result["data"]
        answer = data.get("answer", "No answer returned.")
        meta = (f"intent: {data.get('intent')} · engine: {data.get('engine')} · "
                f"{data.get('timestamp')} UTC")
        st.markdown(answer)
        st.caption(meta)
        with st.expander("Verified data given to the Transformer"):
            st.markdown(f"<div class='fact-box'>{data.get('grounded_facts', '')}</div>",
                        unsafe_allow_html=True)

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer,
            "meta": meta, "facts": data.get("grounded_facts", ""),
        })

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------
def main() -> None:
    st.title("📈 Real-Time AI Prediction System")
    st.markdown(
        f"**{ASSET_NAME} ({SYMBOL})** · live data from the Binance public API every "
        f"**{POLL_INTERVAL_SECONDS} seconds** · **LSTM** forecast for the next **60 seconds** · "
        f"**Transformer** chatbot grounded in the live numbers."
    )

    render_sidebar()

    tab_live, tab_model, tab_chat, tab_about = st.tabs(
        ["🔴 Live monitor", "🧠 Model evaluation", "💬 Chatbot", "ℹ️ About"]
    )
    with tab_live:
        live_section()
    with tab_model:
        model_section()
    with tab_chat:
        chatbot_section()
    with tab_about:
        st.markdown(f"""
### What this system does

1. **External API** - the backend calls the Binance public REST API
   (`/api/v3/klines` and `/api/v3/ticker/24hr`) for **{SYMBOL}**. No API key is needed.
2. **Real time** - a background task polls that API every **{POLL_INTERVAL_SECONDS} seconds**.
   Every response is merged into a rolling buffer of 1-minute candles.
3. **Data processing** - each candle is turned into 7 stationary features:
   log return, two moving-average ratios, rolling volatility, RSI, intrabar range
   and a volume z-score.
4. **LSTM** - a 2-layer LSTM (96 hidden units) reads the last **60 candles** and predicts the
   **log return of the next minute**, which is converted back to a price:
   `predicted_price = current_price × exp(predicted_log_return)`.
5. **Prediction** - every forecast is stored. One minute later the real close price arrives and
   the forecast is scored, which produces the live MAE, RMSE and directional accuracy you see
   on the Live monitor tab.
6. **Transformer chatbot** - the exact numbers are computed in Python and passed to a
   Transformer language model, which phrases the answer. The model is never asked to
   remember or invent a price.

### Error handling

If the external API is unreachable the backend retries with exponential back-off, moves to a
Binance mirror, and finally falls back to Coinbase and Kraken for the live price. Every
endpoint returns a structured message rather than a traceback, and the dashboard shows a
warning while keeping the last known values on screen.

### Documentation

The interactive API documentation is available at
[{BACKEND_URL}/docs]({BACKEND_URL}/docs).
        """)


if __name__ == "__main__":
    main()
