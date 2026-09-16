"""
Communication layer with the external public API.

Primary source
--------------
Binance Public REST API (https://api.binance.com) - free, no API key, no account.

    GET /api/v3/klines        historical OHLCV candles  (symbol, interval, limit)
    GET /api/v3/ticker/24hr   live price + 24h statistics (symbol)
    GET /api/v3/ping          connectivity test

Resilience
----------
1.  Every request is retried HTTP_MAX_RETRIES times with exponential back-off.
2.  If a Binance host fails, the next mirror in BINANCE_MIRRORS is tried.
3.  If every Binance host fails, the live price falls back to Coinbase and then
    to Kraken, so the dashboard keeps working during a regional outage.
4.  All failures raise ExternalAPIError, which the callers turn into a clean
    message instead of a crash.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from app.config import (
    BINANCE_KLINES_ENDPOINT,
    BINANCE_MIRRORS,
    BINANCE_PING_ENDPOINT,
    BINANCE_TICKER_ENDPOINT,
    COINBASE_TICKER_URL,
    HTTP_BACKOFF,
    HTTP_DEADLINE,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT,
    INTERVAL,
    KRAKEN_TICKER_URL,
    SYMBOL,
)
from app.utils import get_logger

log = get_logger("crypto_api", "api.log")

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_base", "taker_quote", "ignore",
]


class ExternalAPIError(RuntimeError):
    """Raised when no data source could satisfy a request."""


@dataclass
class LiveTick:
    """One real-time observation coming from the external API."""

    symbol: str
    price: float
    timestamp: datetime
    source: str
    change_24h_pct: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.astimezone(timezone.utc).isoformat(timespec="seconds")
        return payload


class BinanceClient:
    """Thin, dependency-free wrapper around the endpoints this project needs."""

    def __init__(self, symbol: str = SYMBOL, timeout: int = HTTP_TIMEOUT) -> None:
        self.symbol = symbol
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RealTimeCryptoAI/1.0"})
        self._host_index = 0
        self.last_source: str = "unknown"

    # ---------------------------------------------------------------- core
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        deadline: Optional[float] = HTTP_DEADLINE,
    ) -> Any:
        """
        GET endpoint trying every mirror, with retries and back-off.

        `deadline` bounds the total wall-clock time spent across all hosts and
        attempts. Without it, a full outage would make one logical request take
        longer than the 10-second poll interval and the real-time loop would
        fall behind.
        """
        errors: List[str] = []
        hosts = BINANCE_MIRRORS[self._host_index:] + BINANCE_MIRRORS[: self._host_index]
        started = time.monotonic()

        def out_of_time() -> bool:
            return deadline is not None and (time.monotonic() - started) >= deadline

        for host in hosts:
            if out_of_time():
                errors.append("deadline reached, remaining mirrors skipped")
                break
            url = f"{host}{endpoint}"
            for attempt in range(1, HTTP_MAX_RETRIES + 1):
                remaining = self.timeout
                if deadline is not None:
                    remaining = min(remaining, max(1.0, deadline - (time.monotonic() - started)))
                try:
                    response = self.session.get(url, params=params, timeout=remaining)
                    if response.status_code == 429:
                        wait = HTTP_BACKOFF ** attempt
                        log.warning("Rate limited by %s, waiting %.1fs", host, wait)
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    # Remember the host that worked so the next call starts there.
                    self._host_index = BINANCE_MIRRORS.index(host) if host in BINANCE_MIRRORS else 0
                    self.last_source = "binance:" + host.split("//")[-1]
                    return response.json()
                except requests.RequestException as exc:
                    errors.append(f"{host} attempt {attempt}: {type(exc).__name__}")
                    if attempt < HTTP_MAX_RETRIES and not out_of_time():
                        time.sleep(HTTP_BACKOFF ** attempt)
                if out_of_time():
                    break

        detail = " ; ".join(errors[-4:])
        raise ExternalAPIError(f"All Binance hosts failed for {endpoint}. Details: {detail}")

    # ------------------------------------------------------------ requests
    def ping(self) -> bool:
        """True when the external API answers, False otherwise (never raises)."""
        try:
            self._request(BINANCE_PING_ENDPOINT)
            return True
        except ExternalAPIError as exc:
            log.error("Ping failed: %s", exc)
            return False

    def get_klines(
        self,
        interval: str = INTERVAL,
        limit: int = 1000,
        end_time_ms: Optional[int] = None,
        start_time_ms: Optional[int] = None,
    ) -> pd.DataFrame:
        """Download OHLCV candles and return them as a tidy DataFrame."""
        params: Dict[str, Any] = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(int(limit), 1000),
        }
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)

        raw = self._request(BINANCE_KLINES_ENDPOINT, params)
        if not isinstance(raw, list) or not raw:
            raise ExternalAPIError(f"Empty kline payload for {self.symbol} {interval}")
        return self._klines_to_frame(raw)

    @staticmethod
    def _klines_to_frame(raw: List[List[Any]]) -> pd.DataFrame:
        df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
        numeric = ["open", "high", "low", "close", "volume", "quote_volume", "trades"]
        df[numeric] = df[numeric].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        df = df[["open_time", "close_time", "open", "high", "low", "close", "volume", "trades"]]
        return df.sort_values("open_time").reset_index(drop=True)

    def fetch_history(self, total_bars: int, interval: str = INTERVAL) -> pd.DataFrame:
        """
        Page backwards through /klines until total_bars candles are collected.
        Binance returns at most 1000 candles per call.
        """
        frames: List[pd.DataFrame] = []
        collected = 0
        end_time_ms: Optional[int] = None
        page = 0

        while collected < total_bars:
            page += 1
            batch_size = min(1000, total_bars - collected)
            chunk = self.get_klines(interval=interval, limit=batch_size, end_time_ms=end_time_ms)
            if chunk.empty:
                break
            frames.append(chunk)
            collected += len(chunk)
            oldest_ms = int(chunk["open_time"].iloc[0].timestamp() * 1000)
            end_time_ms = oldest_ms - 1
            log.info("History page %d: +%d bars (total %d/%d)",
                     page, len(chunk), collected, total_bars)
            time.sleep(0.25)                      # stay comfortably inside rate limits
            if len(chunk) < batch_size:           # reached the beginning of the series
                break

        if not frames:
            raise ExternalAPIError("Could not download any historical data")

        history = pd.concat(frames, ignore_index=True)
        history = (history.drop_duplicates("open_time")
                          .sort_values("open_time")
                          .reset_index(drop=True))
        log.info("Downloaded %d unique %s bars for %s", len(history), interval, self.symbol)
        return history

    # ---------------------------------------------------------- live price
    def get_live_tick(self) -> LiveTick:
        """Current price + 24h statistics, with cross-exchange fallback."""
        try:
            data = self._request(BINANCE_TICKER_ENDPOINT, {"symbol": self.symbol})
            return LiveTick(
                symbol=self.symbol,
                price=float(data["lastPrice"]),
                timestamp=datetime.now(timezone.utc),
                source=self.last_source,
                change_24h_pct=float(data.get("priceChangePercent", "nan")),
                high_24h=float(data.get("highPrice", "nan")),
                low_24h=float(data.get("lowPrice", "nan")),
                volume_24h=float(data.get("volume", "nan")),
                bid=float(data.get("bidPrice", "nan")),
                ask=float(data.get("askPrice", "nan")),
            )
        except (ExternalAPIError, KeyError, ValueError, TypeError) as exc:
            log.warning("Binance live price unavailable (%s) - trying fallbacks", exc)

        for name, getter in (("coinbase", self._coinbase_tick), ("kraken", self._kraken_tick)):
            try:
                tick = getter()
                log.info("Live price served by fallback source: %s", name)
                return tick
            except Exception as exc:              # noqa: BLE001 - a fallback must not raise
                log.warning("Fallback %s failed: %s", name, exc)

        raise ExternalAPIError(
            "Live price unavailable: Binance, Coinbase and Kraken all failed. "
            "Check your internet connection."
        )

    def _coinbase_tick(self) -> LiveTick:
        response = self.session.get(COINBASE_TICKER_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        return LiveTick(
            symbol=self.symbol,
            price=float(data["price"]),
            timestamp=datetime.now(timezone.utc),
            source="coinbase (fallback)",
            volume_24h=float(data.get("volume", "nan")),
            bid=float(data.get("bid", "nan")),
            ask=float(data.get("ask", "nan")),
        )

    def _kraken_tick(self) -> LiveTick:
        response = self.session.get(KRAKEN_TICKER_URL, timeout=5)
        response.raise_for_status()
        result = response.json()["result"]
        pair = next(iter(result.values()))
        return LiveTick(
            symbol=self.symbol,
            price=float(pair["c"][0]),
            timestamp=datetime.now(timezone.utc),
            source="kraken (fallback)",
            high_24h=float(pair["h"][1]),
            low_24h=float(pair["l"][1]),
            volume_24h=float(pair["v"][1]),
            bid=float(pair["b"][0]),
            ask=float(pair["a"][0]),
        )


_client: Optional[BinanceClient] = None


def get_client() -> BinanceClient:
    """Process-wide singleton so the HTTP session (and its pool) is reused."""
    global _client
    if _client is None:
        _client = BinanceClient()
    return _client
