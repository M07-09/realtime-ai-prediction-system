"""
Streamlit dashboard - the user interface of the system.

Everything on the page comes from the FastAPI backend; the dashboard never
touches the external API or the model directly:

    External API -> FastAPI -> processing -> LSTM -> prediction -> Streamlit

The live panels sit inside an st.fragment that re-runs every 10 seconds, so
prices, charts and the forecast refresh on their own while the chat
conversation is preserved.

Run:
    python -m streamlit run streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests
import streamlit as st

from app.charts import (actual_vs_predicted_chart, candlestick_chart, error_chart,
                        money, render_table, tick_chart)
from app.config import ASSET_NAME, BACKEND_URL, MODELS_DIR, POLL_INTERVAL_SECONDS, SYMBOL

st.set_page_config(page_title="Real-Time AI Prediction", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
  header[data-testid="stHeader"] {background: transparent;}
  h1, h2, h3 {letter-spacing: -0.01em;}

  /* header strip */
  .hdr {display:flex; align-items:center; gap:.7rem; flex-wrap:wrap;
        padding:.2rem 0 .9rem 0; border-bottom:1px solid #1f2731; margin-bottom:1rem;}
  .hdr-t {font-size:1.35rem; font-weight:700; color:#e6edf3;}
  .pill {font-size:.72rem; font-weight:600; letter-spacing:.04em; text-transform:uppercase;
         color:#9aa4b2; background:#151b23; border:1px solid #1f2731;
         border-radius:999px; padding:.18rem .6rem;}
  .dot {width:.55rem; height:.55rem; border-radius:50%; display:inline-block;
        background:#26a69a; box-shadow:0 0 0 3px rgba(38,166,154,.25);}
  .dot.off {background:#ef5350; box-shadow:0 0 0 3px rgba(239,83,80,.25);}
  .hdr-r {margin-left:auto; font-size:.78rem; color:#6b7684;}

  /* kpi cards */
  .kpi {background:#151b23; border:1px solid #1f2731; border-radius:10px;
        padding:.8rem 1rem; min-height:92px;}
  .kpi-l {font-size:.68rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
          color:#6b7684; margin-bottom:.3rem; white-space:nowrap; overflow:hidden;
          text-overflow:ellipsis;}
  .kpi-v {font-size:clamp(1.05rem, 1.6vw, 1.5rem); font-weight:700; color:#e6edf3;
          line-height:1.15; font-variant-numeric: tabular-nums; white-space:nowrap;}
  .kpi-d {font-size:.74rem; margin-top:.35rem; color:#9aa4b2; white-space:nowrap;
          overflow:hidden; text-overflow:ellipsis;}
  .up {color:#26a69a;} .down {color:#ef5350;} .flat {color:#9aa4b2;}

  /* panels */
  .panel-t {font-size:.78rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase;
            color:#6b7684; margin:.2rem 0 .5rem 0;}
  .note {font-size:.78rem; color:#6b7684;}

  /* tables */
  .tbl-wrap {overflow-x:auto; border:1px solid #1f2731; border-radius:10px; background:#151b23;}
  .tbl {width:100%; border-collapse:collapse; font-size:.8rem; color:#e6edf3;
        font-variant-numeric: tabular-nums;}
  .tbl th {text-align:left; padding:.5rem .7rem; color:#6b7684; font-weight:600;
           font-size:.7rem; letter-spacing:.05em; text-transform:uppercase;
           border-bottom:1px solid #1f2731; position:sticky; top:0; background:#151b23;}
  .tbl td {padding:.42rem .7rem; border-top:1px solid #1a222c;}
  .tbl tr:hover td {background:#1a222c;}

  .fact {background:#0f1419; border:1px solid #1f2731; border-radius:8px;
         padding:.6rem .8rem; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
         font-size:.74rem; color:#9aa4b2; white-space:pre-wrap;}

  div[data-testid="stTabs"] button {font-weight:600;}
  section[data-testid="stSidebar"] {border-right:1px solid #1f2731;}
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
        return {"ok": False, "message": f"Cannot reach the backend at {BACKEND_URL}. "
                                        "Start it with:  python -m app.main"}
    except requests.exceptions.Timeout:
        return {"ok": False, "message": "The backend did not answer in time."}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "message": f"Backend returned {exc.response.status_code}."}
    except Exception as exc:                          # noqa: BLE001
        return {"ok": False, "message": str(exc)}


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
# Small HTML building blocks
# --------------------------------------------------------------------------
def kpi(col, label: str, value: str, delta: str = "", tone: str = "flat") -> None:
    col.markdown(
        f'<div class="kpi"><div class="kpi-l">{label}</div>'
        f'<div class="kpi-v">{value}</div>'
        f'<div class="kpi-d {tone}">{delta}</div></div>',
        unsafe_allow_html=True,
    )


def panel_title(text: str) -> None:
    st.markdown(f'<div class="panel-t">{text}</div>', unsafe_allow_html=True)


def tone_of(value: float, eps: float = 1e-9) -> str:
    return "up" if value > eps else "down" if value < -eps else "flat"


def header(live: Dict[str, Any]) -> None:
    online = bool(live.get("available"))
    age = live.get("age_seconds")
    right = f"updated {age:.0f} s ago · {live.get('source', '')}" if online and age is not None \
        else "waiting for the backend"
    st.markdown(
        f'<div class="hdr"><span class="dot{"" if online else " off"}"></span>'
        f'<span class="hdr-t">Real-Time AI Prediction</span>'
        f'<span class="pill">{SYMBOL}</span><span class="pill">Binance</span>'
        f'<span class="pill">LSTM · +60 s</span><span class="pill">refresh {POLL_INTERVAL_SECONDS} s</span>'
        f'<span class="hdr-r">{right}</span></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("#### System")
        health = api_get("/health", timeout=8)
        if not health["ok"]:
            st.error(health["message"])
            st.caption("The page keeps running and reconnects automatically.")
            return

        d = health["data"]
        (st.success if d["status"] == "healthy" else st.warning)(d["status"].capitalize())

        c1, c2 = st.columns(2)
        c1.metric("API calls", d["poll_count"])
        c2.metric("Errors", d["error_count"])
        st.caption(f"Uptime {d['uptime_seconds'] / 60:.0f} min")

        rows = [
            ("External API", "reachable" if d["external_api_reachable"] else "unreachable"),
            ("Collector", "running" if d["collector_running"] else "stopped"),
            ("LSTM", "loaded" if d["model_loaded"] else "not loaded"),
            ("Transformer", "loaded" if d["chatbot_loaded"] else "loads on first question"),
        ]
        st.markdown("\n".join(f"- {k}: **{v}**" for k, v in rows))
        if d.get("last_error"):
            st.warning(d["last_error"])

        st.divider()
        if st.button("Poll now", width="stretch"):
            r = api_post("/api/refresh", {}, timeout=30)
            (st.success if r["ok"] else st.error)("Polled" if r["ok"] else r["message"])
        st.caption(f"Backend {BACKEND_URL}")


# --------------------------------------------------------------------------
# Live tab (auto-refreshing fragment)
# --------------------------------------------------------------------------
# Chart range -> (candle interval, number of candles). Longer ranges use bigger
# candles, the way exchanges do, so a year is 365 candles rather than 525,600.
TIMEFRAMES = {"1H": ("1m", 60), "4H": ("1m", 240), "1D": ("5m", 288),
              "1W": ("1h", 168), "1M": ("4h", 180), "1Y": ("1d", 365)}


@st.fragment(run_every=POLL_INTERVAL_SECONDS)
def live_section() -> None:
    result = api_get("/api/status", {"history_limit": 240})
    if not result["ok"]:
        header({})
        st.error(result["message"])
        st.caption("This panel retries every 10 seconds on its own.")
        return

    s = result["data"]
    live, pred = s.get("live", {}), s.get("prediction", {})
    stats, coll = s.get("statistics", {}), s.get("collector", {})
    key = coll.get("poll_count", 0)

    header(live)

    if not live.get("available"):
        st.error(f"No live data: {live.get('message', 'the external API did not answer')}")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        tick = live.get("tick_change", 0) or 0
        kpi(c1, f"{ASSET_NAME} price", money(live.get("price")),
            f"{tick:+,.2f} since last poll", tone_of(tick))

        if pred.get("available"):
            ch = pred.get("predicted_change_abs", 0) or 0
            kpi(c2, "Forecast · +60 s", money(pred.get("predicted_price")),
                f"{ch:+,.2f} ({pred.get('predicted_change_pct', 0):+.4f}%) · {pred.get('direction')}",
                tone_of(ch, 0.01))
        else:
            kpi(c2, "Forecast · +60 s", "—", pred.get("message", ""))

        d24 = live.get("change_24h_pct", 0) or 0
        kpi(c3, "24h change", f"{d24:+.2f}%",
            f"H {money(live.get('high_24h'), 0)} · L {money(live.get('low_24h'), 0)}", tone_of(d24))

        trend = (stats.get("trend") or {}) if stats.get("available") else {}
        tc = trend.get("change_pct", 0) or 0
        kpi(c4, f"Trend · {trend.get('lookback', 15)} min",
            str(trend.get("direction", "—")).capitalize(), f"{tc:+.3f}%", tone_of(tc))

        age = live.get("age_seconds", 0) or 0
        kpi(c5, "Data freshness", f"{age:.0f} s", "fresh" if age < 25 else "stale",
            "up" if age < 25 else "down")

    if coll.get("consecutive_errors", 0) > 0:
        st.warning(f"External API failed {coll['consecutive_errors']} time(s) in a row: "
                   f"{coll.get('last_error')}. Showing the last known values.")

    st.write("")
    history = s.get("history", [])
    frame = st.segmented_control("Timeframe", list(TIMEFRAMES), default="4H", required=True,
                                 key="timeframe", label_visibility="collapsed")
    interval, limit = TIMEFRAMES[frame]
    if interval == "1m" and len(history) >= limit:   # the live buffer already holds these
        bars, problem = history[-limit:], ""
    else:                                            # fetched on demand, cached 30 s
        r = api_get("/api/candles", {"interval": interval, "limit": limit}, timeout=20)
        bars = r["data"].get("bars", []) if r["ok"] else []
        problem = "" if r["ok"] else r["message"]
    if bars:
        st.plotly_chart(candlestick_chart(bars), width="stretch", key=f"px_{frame}_{key}",
                        config={"displayModeBar": False, "scrollZoom": True})
        st.markdown(f'<div class="note">{len(bars)} candles · {interval} each</div>',
                    unsafe_allow_html=True)
    else:
        st.info(problem or "Waiting for the first candles ...")

    left, right = st.columns([3, 2], gap="large")
    resolved, ev = s.get("resolved_predictions", []), s.get("live_evaluation", {})

    with left:
        panel_title("Forecast vs reality · live session")
        if resolved:
            st.plotly_chart(actual_vs_predicted_chart(resolved, pred), width="stretch",
                            key=f"avp_{key}", config={"displayModeBar": False})
            panel_title("Error per forecast · green = direction correct")
            st.plotly_chart(error_chart(resolved), width="stretch", key=f"err_{key}",
                            config={"displayModeBar": False})
        else:
            st.markdown(f'<div class="note">{ev.get("message", "Scoring starts once a forecast matures.")}</div>',
                        unsafe_allow_html=True)
            ticks = s.get("ticks", [])
            if ticks:
                panel_title(f"Real-time stream · one point per API call ({POLL_INTERVAL_SECONDS} s)")
                st.plotly_chart(tick_chart(ticks, pred), width="stretch", key=f"tick_{key}",
                                config={"displayModeBar": False})

    with right:
        panel_title("Live scoring")
        if ev.get("available"):
            a, b = st.columns(2)
            kpi(a, "MAE", money(ev.get("MAE")))
            kpi(b, "RMSE", money(ev.get("RMSE")))
            a, b = st.columns(2)
            kpi(a, "MAPE", f"{ev.get('MAPE', 0):.4f}%")
            acc = ev.get("directional_accuracy_pct", 0) or 0
            kpi(b, "Direction", f"{acc:.1f}%", f"{ev.get('resolved')} scored · {ev.get('pending')} pending")
        else:
            st.markdown(f'<div class="note">{ev.get("message", "")}</div>', unsafe_allow_html=True)

        st.write("")
        panel_title("Collected window")
        if stats.get("available"):
            render_table(pd.DataFrame({
                "metric": ["Candles", "Mean", "Median", "Low", "High", "Std dev",
                           "Change", "1-min volatility"],
                "value": [f"{stats.get('count')}", money(stats.get("mean")),
                          money(stats.get("median")), money(stats.get("min")),
                          money(stats.get("max")), money(stats.get("std")),
                          f"{stats.get('change_pct', 0):+.3f}%",
                          f"{stats.get('volatility_pct', 0):.4f}%"],
            }))

    if history:
        with st.expander("Latest candles received from the external API"):
            t = pd.DataFrame(history).tail(25)[
                ["open_time", "open", "high", "low", "close", "volume", "trades"]].iloc[::-1]
            t["open_time"] = t["open_time"].str.replace("+00:00", " UTC", regex=False)
            render_table(t, height=420)


# --------------------------------------------------------------------------
# Model tab
# --------------------------------------------------------------------------
def model_section() -> None:
    result = api_get("/api/model/metrics")
    if not result["ok"]:
        st.error(result["message"])
        return
    m = result["data"]
    if not m.get("available"):
        st.warning(m.get("message", "No metrics available."))
        st.code("python train_model.py", language="bash")
        return

    test = m.get("test", {})
    price, naive = test.get("price_space", {}), test.get("naive_baseline_price_space", {})
    info, tr = m.get("data", {}), m.get("training", {})

    panel_title("Held-out test split · price space")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "MAE", money(price.get("MAE")))
    kpi(c2, "RMSE", money(price.get("RMSE")))
    kpi(c3, "MSE", f"{price.get('MSE', 0):,.1f}")
    kpi(c4, "MAPE", f"{price.get('MAPE', 0):.4f}%")
    kpi(c5, "R²", f"{price.get('R2', 0):.4f}")
    st.markdown(
        f'<div class="note" style="margin-top:.6rem">{test.get("samples", 0):,} test sequences · '
        f'directional accuracy {test.get("directional_accuracy_pct", 0):.2f}% · '
        f'naive "no-change" RMSE {money(naive.get("RMSE"))} · trained on '
        f'{info.get("total_bars", 0):,} candles in {tr.get("training_seconds", 0):.1f} s '
        f'on {tr.get("device")} · {tr.get("parameters", 0):,} parameters</div>',
        unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([2, 3], gap="large")
    with left:
        panel_title("Test vs validation vs naive baseline")
        val = (m.get("validation") or {}).get("price_space", {})

        def f(v, d=4):
            return "n/a" if v is None else f"{v:,.{d}f}"

        render_table(pd.DataFrame({
            "metric": ["MAE (USDT)", "MSE", "RMSE (USDT)", "MAPE %", "R²"],
            "test": [f(price.get("MAE"), 2), f(price.get("MSE"), 1), f(price.get("RMSE"), 2),
                     f(price.get("MAPE")), f(price.get("R2"))],
            "validation": [f(val.get("MAE"), 2), f(val.get("MSE"), 1), f(val.get("RMSE"), 2),
                           f(val.get("MAPE")), f(val.get("R2"))],
            "naive": [f(naive.get("MAE"), 2), f(naive.get("MSE"), 1), f(naive.get("RMSE"), 2),
                      f(naive.get("MAPE")), f(naive.get("R2"))],
        }))
        with st.expander("Full metrics.json"):
            st.json(m, expanded=False)
    with right:
        panel_title("Model")
        render_table(pd.DataFrame({
            "setting": ["Input", "Architecture", "Target", "Data", "Split", "Training"],
            "value": [
                f"last {m.get('sequence_length')} candles × {len(m.get('features', []))} features",
                "LSTM 2 × 96 → Linear 32 → 1",
                "next-candle log return → price = close × exp(r)",
                f"{info.get('period_start', '')[:16]} → {info.get('period_end', '')[:16]} UTC",
                f"{info.get('train_sequences', 0):,} / {info.get('val_sequences', 0):,} / "
                f"{info.get('test_sequences', 0):,} sequences",
                f"{tr.get('epochs_run')} epochs · early stopping · Adam",
            ],
        }))

    st.write("")
    panel_title("Training and evaluation plots")
    plots = [("3_actual_vs_predicted.png", "Actual vs predicted · test split"),
             ("2_loss_curves.png", "Training and validation loss"),
             ("4_prediction_scatter.png", "Predicted vs actual"),
             ("5_error_distribution.png", "Error distribution"),
             ("1_historical_prices.png", "Historical data used for training")]
    found = [(MODELS_DIR / "plots" / n, c) for n, c in plots if (MODELS_DIR / "plots" / n).exists()]
    if not found:
        st.info("No plots found. Run python train_model.py to generate them.")
        return
    for i in range(0, len(found), 2):
        for col, (p, cap) in zip(st.columns(2), found[i:i + 2]):
            col.image(str(p), caption=cap, width="stretch")


# --------------------------------------------------------------------------
# Chat tab
# --------------------------------------------------------------------------
def chatbot_section() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("pending_question", None)

    panel_title("Transformer chatbot · grounded in the live snapshot")
    sugg = api_get("/api/chat/suggestions", timeout=8)
    if sugg["ok"]:
        options = sugg["data"].get("suggestions", [])
        for i in range(0, len(options), 4):
            for col, q in zip(st.columns(4), options[i:i + 4]):
                if col.button(q, key=f"sugg_{q}", width="stretch"):
                    st.session_state.pending_question = q

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("meta"):
                st.caption(turn["meta"])
            if turn.get("facts"):
                with st.expander("Verified data given to the Transformer"):
                    st.markdown(f'<div class="fact">{turn["facts"]}</div>', unsafe_allow_html=True)

    typed = st.chat_input("Ask about the price, the forecast, the trend or the model ...")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Reading the live data ..."):
            r = api_post("/api/chat", {
                "question": question,
                "history": [{"role": t["role"], "content": t["content"]}
                            for t in st.session_state.chat_history[-8:-1]],
            }, timeout=240)
        if not r["ok"]:
            st.error(r["message"])
            st.session_state.chat_history.append({"role": "assistant", "content": r["message"]})
            return
        d = r["data"]
        answer = d.get("answer", "No answer returned.")
        meta = f"{d.get('intent')} · {d.get('engine')} · {d.get('timestamp')} UTC"
        st.markdown(answer)
        st.caption(meta)
        with st.expander("Verified data given to the Transformer"):
            st.markdown(f'<div class="fact">{d.get("grounded_facts", "")}</div>',
                        unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": answer,
                                              "meta": meta, "facts": d.get("grounded_facts", "")})

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------
def about_section() -> None:
    st.markdown(f"""
**Pipeline** · Binance public API → FastAPI (polls every {POLL_INTERVAL_SECONDS} s) → 7 stationary
features → LSTM → forecast 60 s ahead → this dashboard + a Transformer chatbot.

**Model** · a 2-layer LSTM (96 units) reads the last 60 one-minute candles and predicts the log
return of the next candle, converted back to a price with `close × exp(r)`. Every forecast is
stored and scored one minute later against the real close, which produces the live MAE, RMSE
and directional accuracy on the first tab.

**Chatbot** · the exact numbers are computed in Python and handed to a Transformer that only
phrases the answer; anything it says that contradicts the data is rejected.

**Errors** · retries with back-off, four Binance mirrors, Coinbase and Kraken as fallbacks, and a
structured message instead of a traceback at every layer.

API docs: [{BACKEND_URL}/docs]({BACKEND_URL}/docs)
""")


def main() -> None:
    render_sidebar()
    tab_live, tab_model, tab_chat, tab_about = st.tabs(["Live", "Model", "Chat", "About"])
    with tab_live:
        live_section()
    with tab_model:
        model_section()
    with tab_chat:
        chatbot_section()
    with tab_about:
        about_section()


if __name__ == "__main__":
    main()
