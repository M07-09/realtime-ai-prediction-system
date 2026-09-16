"""
Data processing: feature engineering, scaling, sequence building and the
in-memory buffers that hold the real-time stream.

Modelling choice
----------------
The LSTM does **not** regress the raw price. A network fed with raw prices
simply learns to copy the last value, which produces a beautiful R2 and a
useless model. Instead the target is the **next-bar log return**

        y_t = ln( close_{t+1} / close_t )

and the predicted price is reconstructed afterwards as

        price_hat_{t+1} = close_t * exp( y_hat_t )

This keeps the series stationary, makes the learning problem honest, and still
lets us report MAE / RMSE / MAPE / R2 in plain US-dollar price space.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_COLUMNS,
    LIVE_BUFFER_SIZE,
    LIVE_TICK_BUFFER,
    SEQUENCE_LENGTH,
    TARGET_COLUMN,
)
from app.utils import get_logger

log = get_logger("data")


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index, scaled to 0-100."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def build_features(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """
    Turn raw OHLCV candles into the model's feature matrix.

    Features (all stationary / scale-free by construction):
        log_return     one-bar log return of the close price
        ma_ratio_5     close / 5-bar moving average  - 1
        ma_ratio_15    close / 15-bar moving average - 1
        volatility_15  rolling std of log returns over 15 bars
        rsi_14         momentum oscillator, rescaled to 0-1
        range_pct      (high - low) / close, the bar's intrabar range
        volume_z       volume z-score over a 60-bar rolling window
    """
    if df.empty:
        raise ValueError("Cannot build features from an empty DataFrame")

    out = df.copy().reset_index(drop=True)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    close = out["close"].astype(float)

    out["log_return"] = np.log(close / close.shift(1))
    out["ma_ratio_5"] = close / close.rolling(5).mean() - 1.0
    out["ma_ratio_15"] = close / close.rolling(15).mean() - 1.0
    out["volatility_15"] = out["log_return"].rolling(15).std()
    out["rsi_14"] = rsi(close, 14) / 100.0
    out["range_pct"] = (out["high"] - out["low"]) / close

    vol = out["volume"].astype(float)
    vol_mean = vol.rolling(60, min_periods=10).mean()
    vol_std = vol.rolling(60, min_periods=10).std().replace(0.0, np.nan)
    out["volume_z"] = ((vol - vol_mean) / vol_std).clip(-5.0, 5.0)

    if with_target:
        # Next-bar log return - this is what the LSTM learns to predict.
        out[TARGET_COLUMN] = np.log(close.shift(-1) / close)

    out = out.replace([np.inf, -np.inf], np.nan)
    subset = FEATURE_COLUMNS + ([TARGET_COLUMN] if with_target else [])
    out = out.dropna(subset=subset).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Scaling  (standardisation implemented with NumPy - no scikit-learn needed)
# --------------------------------------------------------------------------
class StandardScaler:
    """Z-score scaler fitted on the training split only (no data leakage)."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.columns: List[str] = []
        self.target_mean_: float = 0.0
        self.target_std_: float = 1.0

    def fit(self, X: np.ndarray, columns: List[str],
            y: Optional[np.ndarray] = None) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        self.columns = list(columns)
        if y is not None:
            self.target_mean_ = float(np.mean(y))
            target_std = float(np.std(y))
            self.target_std_ = target_std if target_std > 1e-12 else 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted")
        return (X - self.mean_) / self.std_

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean_) / self.target_std_

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return y_scaled * self.target_std_ + self.target_mean_

    # -------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Nothing to save: scaler has not been fitted")
        payload = {
            "columns": self.columns,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "target_mean": self.target_mean_,
            "target_std": self.target_std_,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        log.info("Scaler saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "StandardScaler":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        scaler = cls()
        scaler.columns = payload["columns"]
        scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        scaler.std_ = np.asarray(payload["std"], dtype=np.float64)
        scaler.target_mean_ = float(payload["target_mean"])
        scaler.target_std_ = float(payload["target_std"])
        return scaler


# --------------------------------------------------------------------------
# Sequence building
# --------------------------------------------------------------------------
def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window of length seq_len over the feature matrix.

    Returns
    -------
    X_seq : (n_samples, seq_len, n_features)
    y_seq : (n_samples,)                      target of the bar after the window
    idx   : (n_samples,)                      index of the last bar in each window
    """
    if len(X) <= seq_len:
        raise ValueError(f"Need more than {seq_len} rows to build sequences, got {len(X)}")

    n_samples = len(X) - seq_len + 1
    X_seq = np.empty((n_samples, seq_len, X.shape[1]), dtype=np.float32)
    y_seq = np.empty(n_samples, dtype=np.float32)
    idx = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        end = i + seq_len
        X_seq[i] = X[i:end]
        y_seq[i] = y[end - 1]
        idx[i] = end - 1

    return X_seq, y_seq, idx


def chronological_split(n: int, train_ratio: float, val_ratio: float) -> Tuple[slice, slice, slice]:
    """Time-ordered split: the test set is always the most recent data."""
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


# --------------------------------------------------------------------------
# Descriptive statistics used by the dashboard and the chatbot
# --------------------------------------------------------------------------
def describe_series(df: pd.DataFrame, price_col: str = "close") -> Dict[str, float]:
    """Basic statistics over the collected data."""
    if df.empty or price_col not in df.columns:
        return {}

    prices = df[price_col].astype(float)
    returns = np.log(prices / prices.shift(1)).dropna()
    first, last = float(prices.iloc[0]), float(prices.iloc[-1])

    stats: Dict[str, float] = {
        "count": int(len(prices)),
        "current": last,
        "mean": float(prices.mean()),
        "median": float(prices.median()),
        "min": float(prices.min()),
        "max": float(prices.max()),
        "std": float(prices.std()) if len(prices) > 1 else 0.0,
        "first": first,
        "change_abs": last - first,
        "change_pct": ((last - first) / first * 100.0) if first else 0.0,
        "range": float(prices.max() - prices.min()),
    }
    if len(returns) > 1:
        # Annualised volatility of 1-minute returns: sqrt(60 * 24 * 365) bars.
        stats["volatility_pct"] = float(returns.std() * 100.0)
        stats["volatility_annualised_pct"] = float(returns.std() * np.sqrt(525_600) * 100.0)
    return stats


def trend_label(df: pd.DataFrame, lookback: int = 15, price_col: str = "close") -> Dict[str, float | str]:
    """Classify the short-term trend of the collected series."""
    if df.empty or len(df) < 2:
        return {"direction": "unknown", "change_pct": 0.0, "lookback": 0}

    prices = df[price_col].astype(float)
    window = min(lookback, len(prices) - 1)
    start, end = float(prices.iloc[-1 - window]), float(prices.iloc[-1])
    change_pct = ((end - start) / start * 100.0) if start else 0.0

    if change_pct > 0.05:
        direction = "rising"
    elif change_pct < -0.05:
        direction = "falling"
    else:
        direction = "flat"

    return {
        "direction": direction,
        "change_pct": change_pct,
        "change_abs": end - start,
        "lookback": window,
    }


# --------------------------------------------------------------------------
# Real-time buffers
# --------------------------------------------------------------------------
class RollingBuffer:
    """Fixed-size, time-ordered store of the most recent 1-minute bars."""

    def __init__(self, maxlen: int = LIVE_BUFFER_SIZE) -> None:
        self.maxlen = maxlen
        self._frame = pd.DataFrame()

    def seed(self, df: pd.DataFrame) -> None:
        """Fill the buffer with a block of historical bars at start-up."""
        self._frame = df.tail(self.maxlen).copy().reset_index(drop=True)

    def upsert(self, df: pd.DataFrame) -> int:
        """
        Merge freshly downloaded bars into the buffer.

        The most recent bar is still open, so its close price keeps changing:
        matching rows are replaced rather than appended. Returns the number of
        genuinely new bars.
        """
        if df.empty:
            return 0
        if self._frame.empty:
            self.seed(df)
            return len(self._frame)

        known = set(self._frame["open_time"])
        new_bars = int(sum(1 for t in df["open_time"] if t not in known))

        merged = pd.concat([self._frame, df], ignore_index=True)
        merged = merged.drop_duplicates("open_time", keep="last")
        merged = merged.sort_values("open_time").reset_index(drop=True)
        self._frame = merged.tail(self.maxlen).reset_index(drop=True)
        return new_bars

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def __len__(self) -> int:
        return len(self._frame)

    def is_ready(self, min_bars: int = SEQUENCE_LENGTH + 65) -> bool:
        """True when there are enough bars to build one full model input."""
        return len(self._frame) >= min_bars


class TickBuffer:
    """Every 10-second observation received from the external API."""

    def __init__(self, maxlen: int = LIVE_TICK_BUFFER) -> None:
        self._ticks: Deque[dict] = deque(maxlen=maxlen)

    def append(self, tick: dict) -> None:
        self._ticks.append(tick)

    def to_list(self) -> List[dict]:
        return list(self._ticks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_list())

    def __len__(self) -> int:
        return len(self._ticks)
