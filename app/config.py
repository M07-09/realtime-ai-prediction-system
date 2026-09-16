"""
Central configuration for the Real-Time Crypto AI Prediction System.

Every tunable constant of the project lives here so that the rest of the
code never hard-codes a symbol, an interval, a path or a URL.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"        # raw + live market data
MODELS_DIR = PROJECT_ROOT / "models"    # everything the training run produces
PLOT_DIR = MODELS_DIR / "plots"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, MODELS_DIR, PLOT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HISTORY_CSV = DATA_DIR / "btcusdt_1m_history.csv"
LIVE_LOG_CSV = DATA_DIR / "live_stream_log.csv"

MODEL_PATH = MODELS_DIR / "lstm_model.pt"
SCALER_PATH = MODELS_DIR / "scaler.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
TRAIN_HISTORY_PATH = MODELS_DIR / "training_history.json"

# --------------------------------------------------------------------------
# External API  (Binance public REST API - no API key required)
# --------------------------------------------------------------------------
SYMBOL = os.getenv("RTAPS_SYMBOL", "BTCUSDT")
ASSET_NAME = "Bitcoin"
QUOTE_CURRENCY = "USDT"

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_KLINES_ENDPOINT = "/api/v3/klines"
BINANCE_TICKER_ENDPOINT = "/api/v3/ticker/24hr"
BINANCE_PRICE_ENDPOINT = "/api/v3/ticker/price"
BINANCE_PING_ENDPOINT = "/api/v3/ping"

# Mirror hosts used automatically when the primary host is blocked / down.
BINANCE_MIRRORS = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api-gcp.binance.com",
    "https://data-api.binance.vision",
]

# Independent exchanges used as a last-resort fallback for the live price.
COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
KRAKEN_TICKER_URL = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"

HTTP_TIMEOUT = 6           # seconds for a single HTTP request
HTTP_MAX_RETRIES = 2       # attempts per host before moving to the next one
HTTP_BACKOFF = 1.5         # exponential back-off factor between attempts
HTTP_DEADLINE = 8.0        # hard cap on one logical request, so that retrying
                           # across mirrors can never overrun the 10 s poll cycle

# --------------------------------------------------------------------------
# Time-series definition
# --------------------------------------------------------------------------
INTERVAL = "1m"            # Binance kline interval used for training
INTERVAL_SECONDS = 60      # length of one bar in seconds
SEQUENCE_LENGTH = 60       # the LSTM looks at the last 60 one-minute bars
FORECAST_HORIZON = 1       # it predicts 1 bar ahead  ->  60 seconds into the future
HISTORY_BARS = 20_000      # number of 1-minute bars downloaded for training (~14 days)

# Feature columns produced by app.data_processing.build_features()
FEATURE_COLUMNS = [
    "log_return",
    "ma_ratio_5",
    "ma_ratio_15",
    "volatility_15",
    "rsi_14",
    "range_pct",
    "volume_z",
]
TARGET_COLUMN = "target_log_return"

# --------------------------------------------------------------------------
# LSTM hyper-parameters
# --------------------------------------------------------------------------
HIDDEN_SIZE = 96
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128
MAX_EPOCHS = 80
EARLY_STOPPING_PATIENCE = 10
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15           # test split = 1 - TRAIN_SPLIT - VAL_SPLIT = 0.15
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Real-time behaviour
# --------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 10     # the backend calls the external API every 10 s
LIVE_BUFFER_SIZE = 720         # 720 one-minute bars kept in memory (12 hours)
LIVE_TICK_BUFFER = 360         # 360 ten-second ticks kept in memory (1 hour)
PREDICTION_LEDGER_SIZE = 500   # resolved prediction-vs-actual pairs kept in memory

# --------------------------------------------------------------------------
# Backend / dashboard wiring
# --------------------------------------------------------------------------
BACKEND_HOST = os.getenv("RTAPS_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("RTAPS_PORT", "8000"))
BACKEND_URL = os.getenv("RTAPS_BACKEND_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}")

# --------------------------------------------------------------------------
# Transformer chatbot
# --------------------------------------------------------------------------
CHATBOT_MODEL = os.getenv("RTAPS_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CHATBOT_MAX_NEW_TOKENS = 180
CHATBOT_TEMPERATURE = 0.0   # greedy decoding: a data assistant must be reproducible
CHATBOT_TOP_P = 0.9
CHATBOT_DEVICE = os.getenv("RTAPS_CHAT_DEVICE", "auto")   # "auto" | "cuda" | "cpu"
CHATBOT_HISTORY_TURNS = 4
