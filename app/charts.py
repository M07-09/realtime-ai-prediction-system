"""
Every chart and table the dashboard draws.

All figures share one dark, trading-terminal style so the page reads as a
single product rather than a collection of default Plotly charts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

__all__ = ["render_table", "money", "candlestick_chart",
           "actual_vs_predicted_chart", "error_chart", "tick_chart"]

# Palette shared with the page CSS in streamlit_app.py
BG = "#0f1419"
PANEL = "#151b23"
GRID = "#1f2731"
TEXT = "#9aa4b2"
UP = "#26a69a"
DOWN = "#ef5350"
ACCENT = "#4f8cff"
ACCENT_2 = "#f5a524"


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", color=TEXT, size=11),
        margin=dict(l=8, r=56, t=12, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1c2430", bordercolor=GRID, font=dict(color="#e6edf3")),
        showlegend=False,
        dragmode="pan",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False,
                     showspikes=True, spikecolor=TEXT, spikethickness=1,
                     spikedash="dot", spikemode="across")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, side="right",
                     showspikes=True, spikecolor=TEXT, spikethickness=1,
                     spikedash="dot", spikemode="across")
    return fig


def render_table(df: pd.DataFrame, height: Optional[int] = None) -> None:
    """
    Render a DataFrame as plain HTML.

    st.dataframe serialises through pyarrow, which some locked-down Windows
    installations block with an Application Control policy. Rendering HTML
    ourselves keeps the dashboard working everywhere.
    """
    # The markup must start at column 0: an indented line becomes a code block.
    style = f"max-height:{height}px; overflow-y:auto;" if height else ""
    html = df.to_html(index=False, border=0, justify="left",
                      classes="tbl", float_format=lambda v: f"{v:,.2f}")
    st.markdown(
        f'<div class="tbl-wrap" style="{style}">{html}</div>',
        unsafe_allow_html=True,
    )


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
def candlestick_chart(bars: List[Dict[str, Any]]) -> go.Figure:
    """Price candles with a volume pane underneath, nothing else."""
    df = pd.DataFrame(bars)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.78, 0.22], vertical_spacing=0.03)
    fig.add_trace(
        go.Candlestick(
            x=df["open_time"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="BTC/USDT",
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
            whiskerwidth=0.4,
        ),
        row=1, col=1,
    )
    colours = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(x=df["open_time"], y=df["volume"], name="Volume",
               marker_color=colours, opacity=0.45, hovertemplate="%{y:,.2f}<extra></extra>"),
        row=2, col=1,
    )

    _base_layout(fig, 560)
    # Plotly stretches each candle to fill its x-slot, so 60 candles come out fat
    # and 6 come out huge. Candlesticks obey the box layout gap, so widen the gap
    # as the count drops to keep the body near 8 px on a ~1200 px plot.
    slot_px = 1200 / max(len(df), 1)
    fig.update_layout(xaxis_rangeslider_visible=False,
                      boxgap=min(0.9, max(0.2, 1 - 8 / slot_px)))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(tickformat=".2s", row=2, col=1, showgrid=False)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    return fig


def _add_latest_forecast(fig: go.Figure, latest: Optional[Dict[str, Any]],
                         anchor_x: Any, anchor_y: float) -> None:
    """Draw the forecast that has not matured yet, one minute ahead of the last point."""
    if not latest or not latest.get("available"):
        return
    target = pd.to_datetime(latest["target_time"], utc=True)
    predicted = float(latest["predicted_price"])
    fig.add_trace(go.Scatter(
        x=[anchor_x, target], y=[anchor_y, predicted], name="Next forecast",
        mode="lines+markers", line=dict(color=ACCENT_2, width=1.4, dash="dot"),
        marker=dict(size=[0, 9], color=ACCENT_2, line=dict(color=BG, width=1.5)),
        hovertemplate="next forecast %{y:$,.2f}<extra></extra>",
    ))


def actual_vs_predicted_chart(records: List[Dict[str, Any]],
                              latest: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Matured forecasts against the real price, plus the forecast still pending."""
    df = pd.DataFrame(records)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["target_time"], y=df["actual_price"], name="Actual",
                             mode="lines+markers", line=dict(color="#e6edf3", width=1.8),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=df["target_time"], y=df["predicted_price"], name="LSTM",
                             mode="lines+markers",
                             line=dict(color=ACCENT_2, width=1.8, dash="dot"),
                             marker=dict(size=5, symbol="diamond")))
    _add_latest_forecast(fig, latest, df["target_time"].iloc[-1],
                         float(df["actual_price"].iloc[-1]))
    _base_layout(fig, 300)
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def error_chart(records: List[Dict[str, Any]]) -> go.Figure:
    df = pd.DataFrame(records)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    colours = [UP if ok else DOWN for ok in df["direction_correct"]]
    fig = go.Figure(go.Bar(x=df["target_time"], y=df["error"], marker_color=colours,
                           hovertemplate="%{y:+,.2f} USDT<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=TEXT, width=1))
    _base_layout(fig, 220)
    fig.update_yaxes(tickprefix="$", tickformat="+,.0f")
    return fig


def tick_chart(ticks: List[Dict[str, Any]],
               latest: Optional[Dict[str, Any]] = None) -> go.Figure:
    df = pd.DataFrame(ticks)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    fig = go.Figure(go.Scatter(x=df["timestamp"], y=df["price"], mode="lines+markers",
                               name="Live price", line=dict(color=ACCENT, width=1.6),
                               marker=dict(size=4)))
    _add_latest_forecast(fig, latest, df["timestamp"].iloc[-1], float(df["price"].iloc[-1]))
    _base_layout(fig, 300)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig
