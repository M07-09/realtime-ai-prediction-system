"""
Every chart and table the dashboard draws.

Kept apart from streamlit_app.py so the page file stays about layout and the
plotting details stay reviewable on their own.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.config import POLL_INTERVAL_SECONDS

__all__ = ["render_table", "money", "price_chart", "actual_vs_predicted_chart",
           "error_chart", "tick_chart"]


def render_table(df: pd.DataFrame, height: Optional[int] = None) -> None:
    """
    Render a DataFrame as a plain HTML table.

    st.dataframe / st.table serialise through pyarrow, which some locked-down
    Windows installations block with an Application Control policy. Rendering
    the HTML ourselves keeps the dashboard working everywhere and adds no
    dependency.
    """
    # NOTE: the markup below must start at column 0. Markdown turns any line
    # indented by four or more spaces into a code block, which would print the
    # raw HTML instead of rendering it.
    style = f"max-height:{height}px; overflow-y:auto;" if height else ""
    html = df.to_html(index=False, border=0, justify="left",
                      classes="rtaps-table", float_format=lambda v: f"{v:,.2f}")
    block = (
        "<style>\n"
        f".rtaps-wrap {{ {style} overflow-x:auto; "
        "border:1px solid rgba(128,128,128,0.32); border-radius:8px; }\n"
        ".rtaps-table { width:100%; border-collapse:collapse; font-size:0.82rem; "
        "color:inherit; background:transparent; }\n"
        ".rtaps-table th { background:rgba(128,128,128,0.16); text-align:left; "
        "padding:0.45rem 0.6rem; position:sticky; top:0; font-weight:600; }\n"
        ".rtaps-table td { padding:0.38rem 0.6rem; "
        "border-top:1px solid rgba(128,128,128,0.22); }\n"
        ".rtaps-table tr:hover td { background:rgba(128,128,128,0.10); }\n"
        "</style>\n"
        f'<div class="rtaps-wrap">{html}</div>'
    )
    st.markdown(block, unsafe_allow_html=True)


def money(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
        return f"${value:,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def price_chart(bars: List[Dict[str, Any]], prediction: Dict[str, Any]) -> go.Figure:
    """Candle history with the LSTM forecast plotted one minute into the future."""
    df = pd.DataFrame(bars)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
        vertical_spacing=0.06,
    )
    fig.add_trace(
        go.Candlestick(
            x=df["open_time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="BTC/USDT",
            increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["open_time"], y=df["close"].rolling(15).mean(),
                   name="15-min average", line=dict(color="#2563eb", width=1.3)),
        row=1, col=1,
    )

    if prediction.get("available"):
        target = pd.to_datetime(prediction["target_time"], utc=True)
        last_time = df["open_time"].iloc[-1]
        last_close = float(df["close"].iloc[-1])
        predicted = float(prediction["predicted_price"])
        colour = "#16a34a" if predicted >= last_close else "#dc2626"
        fig.add_trace(
            go.Scatter(
                x=[last_time, target], y=[last_close, predicted],
                name="LSTM forecast", mode="lines+markers",
                line=dict(color=colour, width=2.4, dash="dot"),
                marker=dict(size=[6, 13], symbol=["circle", "star"], color=colour),
            ),
            row=1, col=1,
        )

    colours = ["#16a34a" if c >= o else "#dc2626" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(x=df["open_time"], y=df["volume"], name="volume",
               marker_color=colours, opacity=0.55),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_layout(
        height=540, margin=dict(l=10, r=10, t=54, b=10),
        xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        hovermode="x unified",
    )
    return fig


def actual_vs_predicted_chart(records: List[Dict[str, Any]]) -> go.Figure:
    """Live scoring: each matured forecast against the price that really occurred."""
    df = pd.DataFrame(records)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["target_time"], y=df["actual_price"],
                             name="actual price", mode="lines+markers",
                             line=dict(color="#111827", width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df["target_time"], y=df["predicted_price"],
                             name="LSTM prediction", mode="lines+markers",
                             line=dict(color="#dc2626", width=2, dash="dash"),
                             marker=dict(size=5, symbol="x")))
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=30, b=10),
        title="Actual vs predicted (forecasts made 1 minute earlier, live session)",
        yaxis_title="USDT", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def error_chart(records: List[Dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(records)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    colours = ["#16a34a" if ok else "#dc2626" for ok in df["direction_correct"]]
    fig = go.Figure(go.Bar(x=df["target_time"], y=df["error"], marker_color=colours,
                           name="prediction error"))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=10),
        title="Forecast error per minute (green = direction called correctly)",
        yaxis_title="predicted - actual (USDT)", showlegend=False,
    )
    return fig


def tick_chart(ticks: List[Dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(ticks)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    fig = go.Figure(go.Scatter(x=df["timestamp"], y=df["price"], mode="lines+markers",
                               line=dict(color="#7c3aed", width=1.6), marker=dict(size=4),
                               name="10-second tick"))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=10),
        title=f"Real-time stream - one point per API call ({POLL_INTERVAL_SECONDS} s)",
        yaxis_title="USDT", showlegend=False,
    )
    return fig
