"""
Generate the architecture diagram used in the README.

    python make_diagram.py     ->  models/plots/0_architecture.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from app.config import PLOT_DIR

INK = "#0f172a"
MUTED = "#64748b"

BOXES = [
    # x, y, w, h, title, subtitle, fill, edge
    (0.4, 5.05, 2.5, 1.25, "External API", "Binance public REST\n/klines · /ticker/24hr\nno API key",
     "#eff6ff", "#3b82f6"),
    (3.5, 5.05, 2.5, 1.25, "FastAPI backend", "polls every 10 s\n15 endpoints\nretries · fallbacks",
     "#f0fdf4", "#22c55e"),
    (6.6, 5.05, 2.5, 1.25, "Data processing", "7 stationary features\nrolling 720-candle buffer\nz-score scaler",
     "#fefce8", "#eab308"),
    (6.6, 2.95, 2.5, 1.25, "LSTM model", "2 layers × 96 units\n60-candle window\n117,953 parameters",
     "#fdf2f8", "#ec4899"),
    (3.5, 2.95, 2.5, 1.25, "Prediction", "price in +60 s\nscored 1 min later\nlive MAE / RMSE",
     "#f5f3ff", "#8b5cf6"),
    (0.4, 2.95, 2.5, 1.25, "Streamlit dashboard", "4 tabs · 7 charts\nauto-refresh 10 s",
     "#ecfeff", "#06b6d4"),
    (3.5, 0.85, 2.5, 1.25, "Transformer chatbot", "Qwen2.5-0.5B\ngrounded in live data\nanswers verified",
     "#fff7ed", "#f97316"),
]

ARROWS = [
    ((2.9, 5.68), (3.5, 5.68)),      # API   -> FastAPI
    ((6.0, 5.68), (6.6, 5.68)),      # FastAPI -> processing
    ((7.85, 5.05), (7.85, 4.20)),    # processing -> LSTM
    ((6.6, 3.58), (6.0, 3.58)),      # LSTM -> prediction
    ((3.5, 3.58), (2.9, 3.58)),      # prediction -> dashboard
    ((4.75, 2.95), (4.75, 2.10)),    # prediction -> chatbot
]

# routed with a right angle so it reaches the dashboard box instead of
# stopping in empty space below it
ELBOW = ((3.5, 1.48), (1.65, 2.95))


def main() -> int:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0.2, 7.2)
    ax.axis("off")

    ax.text(0.4, 6.85, "Real-Time AI Prediction System",
            fontsize=19, fontweight="bold", color=INK, va="bottom")
    ax.text(0.4, 6.55, "Bitcoin price 60 seconds ahead, from a live public API",
            fontsize=11, color=MUTED, va="bottom")

    for x, y, w, h, title, subtitle, fill, edge in BOXES:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.8, facecolor=fill, edgecolor=edge))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 0.27, subtitle, ha="center", va="center",
                fontsize=8.4, color=MUTED, linespacing=1.5)

    for start, end in ARROWS:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=17,
            linewidth=1.7, color="#334155", shrinkA=2, shrinkB=2))

    ax.add_patch(FancyArrowPatch(
        *ELBOW, arrowstyle="-|>", mutation_scale=17, linewidth=1.7,
        color="#334155", shrinkA=2, shrinkB=2,
        connectionstyle="angle,angleA=180,angleB=90,rad=8"))

    ax.text(9.35, 1.48, "every 10 seconds the\nwhole loop runs again",
            ha="right", va="center", fontsize=9.2, color=MUTED,
            style="italic", linespacing=1.6)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "0_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
