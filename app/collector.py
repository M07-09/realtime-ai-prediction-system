"""
The real-time engine.

A background asyncio task calls the external API every POLL_INTERVAL_SECONDS
(10 s by default). On every cycle it:

    1. requests the live ticker                       (external API)
    2. requests the most recent candles               (external API)
    3. merges them into the rolling buffer            (data processing)
    4. runs the LSTM                                  (prediction)
    5. scores forecasts whose target minute has closed (live evaluation)
    6. publishes an immutable snapshot that the API endpoints read

Nothing here ever raises out of the loop: a failed cycle is recorded in the
snapshot as an error, the consecutive-failure counter goes up, and the loop
simply tries again 10 seconds later.
"""
from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config import (
    INTERVAL,
    LIVE_LOG_CSV,
    POLL_INTERVAL_SECONDS,
    PREDICTION_LEDGER_SIZE,
    SEQUENCE_LENGTH,
    SYMBOL,
)
from app.data import RollingBuffer, TickBuffer, describe_series, trend_label
from app.crypto_api import ExternalAPIError, LiveTick, get_client
from app.predictor import Prediction, PredictionLedger, get_predictor
from app.utils import get_logger

log = get_logger("collector", "collector.log")

WARMUP_BARS = SEQUENCE_LENGTH + 120        # enough rows for every rolling feature


class DataCollector:
    """Owns the live state of the whole system."""

    def __init__(self, poll_seconds: int = POLL_INTERVAL_SECONDS) -> None:
        self.poll_seconds = poll_seconds
        self.client = get_client()
        self.predictor = get_predictor()

        self.bars = RollingBuffer()
        self.ticks = TickBuffer()
        self.ledger = PredictionLedger(PREDICTION_LEDGER_SIZE)

        self.latest_tick: Optional[LiveTick] = None
        self.latest_prediction: Optional[Prediction] = None
        self.previous_price: Optional[float] = None

        self.poll_count = 0
        self.success_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error: Optional[str] = None
        self.last_success_at: Optional[datetime] = None
        self.started_at: Optional[datetime] = None
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------- start-up
    async def warmup(self) -> None:
        """Seed the buffer with recent history so predictions start immediately."""
        try:
            frame = await asyncio.to_thread(
                self.client.get_klines, INTERVAL, WARMUP_BARS
            )
            self.bars.seed(frame)
            log.info("Warm-up complete: %d bars in the buffer", len(self.bars))
        except ExternalAPIError as exc:
            self.last_error = f"Warm-up failed: {exc}"
            log.error(self.last_error)

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.started_at = datetime.now(timezone.utc)
        await self.warmup()
        self._task = asyncio.create_task(self._loop(), name="collector-loop")
        log.info("Collector started - polling every %d seconds", self.poll_seconds)

    async def stop(self) -> None:
        self.running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):     # noqa: BLE001
                pass
        log.info("Collector stopped after %d polls (%d errors)", self.poll_count, self.error_count)

    # ------------------------------------------------------------- the loop
    async def _loop(self) -> None:
        while self.running:
            cycle_started = asyncio.get_event_loop().time()
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:                        # noqa: BLE001
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error = f"Unexpected error in poll cycle: {exc}"
                log.exception("Poll cycle crashed - the loop keeps running")

            elapsed = asyncio.get_event_loop().time() - cycle_started
            await asyncio.sleep(max(0.5, self.poll_seconds - elapsed))

    async def poll_once(self) -> Dict[str, Any]:
        """
        One complete cycle: fetch -> process -> predict -> score.

        The two external calls are independent and degrade separately. If only
        the ticker fails, the close of the newest candle is used as the live
        price; if only the candles fail, the previous buffer is reused. The
        cycle is lost only when both calls fail.
        """
        async with self._lock:
            self.poll_count += 1
            tick: Optional[LiveTick] = None
            recent: Optional[pd.DataFrame] = None
            problems: List[str] = []

            try:
                tick = await asyncio.to_thread(self.client.get_live_tick)
            except ExternalAPIError as exc:
                problems.append(f"live ticker: {exc}")

            try:
                recent = await asyncio.to_thread(self.client.get_klines, INTERVAL, 5)
            except ExternalAPIError as exc:
                problems.append(f"candles: {exc}")

            if tick is None and (recent is None or recent.empty):
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error = " | ".join(problems) or "unknown external API failure"
                log.warning("Poll %d failed: %s", self.poll_count, self.last_error)
                return {"ok": False, "error": self.last_error}

            if tick is None:
                # Partial degradation: derive the price from the newest candle.
                last = recent.iloc[-1]
                tick = LiveTick(
                    symbol=SYMBOL,
                    price=float(last["close"]),
                    timestamp=datetime.now(timezone.utc),
                    source="binance klines (ticker unavailable)",
                )
                log.warning("Poll %d: ticker unavailable, using the latest candle close",
                            self.poll_count)

            if recent is None or recent.empty:
                recent = pd.DataFrame()
                log.warning("Poll %d: candles unavailable, reusing the existing buffer",
                            self.poll_count)

            self.previous_price = self.latest_tick.price if self.latest_tick else None
            self.latest_tick = tick
            new_bars = self.bars.upsert(recent)

            self.ticks.append({
                "timestamp": tick.timestamp.isoformat(timespec="seconds"),
                "price": tick.price,
                "source": tick.source,
            })
            self._append_live_log(tick)

            prediction = self.predictor.predict(self.bars.frame, live_price=tick.price)
            self.latest_prediction = prediction
            self.ledger.register(prediction)
            self.ledger.resolve(self.bars.frame)

            self.success_count += 1
            self.consecutive_errors = 0
            self.last_success_at = tick.timestamp
            if problems:
                # The cycle still produced a value, but say what degraded.
                self.error_count += 1
                self.last_error = "partial: " + " | ".join(p[:120] for p in problems)
            else:
                self.last_error = None

            if new_bars:
                log.info("Poll %d | price %.2f | %d new bar(s) | forecast %s",
                         self.poll_count, tick.price, new_bars,
                         f"{prediction.predicted_price:.2f}" if prediction.available else "n/a")

            return {"ok": True, "new_bars": new_bars, "price": tick.price}

    @staticmethod
    def _append_live_log(tick: LiveTick) -> None:
        """Append every observation to data/live_stream_log.csv (audit trail)."""
        try:
            is_new = not LIVE_LOG_CSV.exists()
            with open(LIVE_LOG_CSV, "a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if is_new:
                    writer.writerow(["timestamp", "symbol", "price", "source"])
                writer.writerow([tick.timestamp.isoformat(timespec="seconds"),
                                 tick.symbol, tick.price, tick.source])
        except OSError as exc:
            log.warning("Could not write the live log: %s", exc)

    # ------------------------------------------------------------ snapshots
    def live_snapshot(self) -> Dict[str, Any]:
        """Current value plus the health of the data feed."""
        if self.latest_tick is None:
            return {
                "available": False,
                "message": self.last_error or "Waiting for the first successful poll ...",
                "symbol": SYMBOL,
                "poll_count": self.poll_count,
            }

        tick = self.latest_tick.to_dict()
        tick.update({
            "available": True,
            "previous_price": self.previous_price,
            "tick_change": (self.latest_tick.price - self.previous_price)
                           if self.previous_price else 0.0,
            "poll_count": self.poll_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "buffer_bars": len(self.bars),
            "ticks_recorded": len(self.ticks),
            "poll_interval_seconds": self.poll_seconds,
            "age_seconds": (datetime.now(timezone.utc) - self.latest_tick.timestamp).total_seconds(),
        })
        return tick

    def history_records(self, limit: int = 240) -> List[Dict[str, Any]]:
        frame = self.bars.frame
        if frame.empty:
            return []
        subset = frame.tail(limit).copy()
        subset["open_time"] = subset["open_time"].astype(str)
        subset["close_time"] = subset["close_time"].astype(str)
        return subset.to_dict(orient="records")

    def statistics(self) -> Dict[str, Any]:
        frame = self.bars.frame
        if frame.empty:
            return {"available": False, "message": "No data collected yet"}

        stats = describe_series(frame)
        stats["available"] = True
        stats["symbol"] = SYMBOL
        stats["interval"] = INTERVAL
        stats["trend"] = trend_label(frame, lookback=15)
        stats["trend_60"] = trend_label(frame, lookback=60)
        stats["window_start"] = str(frame["open_time"].iloc[0])
        stats["window_end"] = str(frame["open_time"].iloc[-1])
        if self.latest_tick is not None:
            stats["live_price"] = self.latest_tick.price
            stats["source"] = self.latest_tick.source
        return stats

    def prediction_snapshot(self) -> Dict[str, Any]:
        if self.latest_prediction is None:
            return {"available": False,
                    "message": "No prediction yet - the first one appears within 10 seconds."}
        return self.latest_prediction.to_dict()

    def full_status(self, history_limit: int = 240) -> Dict[str, Any]:
        """Everything the dashboard needs, in a single round-trip."""
        return {
            "server_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "collector": {
                "running": self.running,
                "started_at": self.started_at.isoformat(timespec="seconds") if self.started_at else None,
                "poll_interval_seconds": self.poll_seconds,
                "poll_count": self.poll_count,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "consecutive_errors": self.consecutive_errors,
                "last_error": self.last_error,
                "last_success_at": self.last_success_at.isoformat(timespec="seconds")
                                   if self.last_success_at else None,
                "buffer_bars": len(self.bars),
            },
            "live": self.live_snapshot(),
            "prediction": self.prediction_snapshot(),
            "statistics": self.statistics(),
            "live_evaluation": self.ledger.live_metrics(),
            "history": self.history_records(history_limit),
            "ticks": self.ticks.to_list(),
            "resolved_predictions": self.ledger.resolved[-120:],
            "model": {
                "loaded": self.predictor.loaded,
                "error": self.predictor.load_error,
                "device": str(self.predictor.device),
                "metadata": self.predictor.metadata,
            },
        }

    def chatbot_context(self) -> Dict[str, Any]:
        """A compact, factual snapshot handed to the Transformer chatbot."""
        return {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "live": self.live_snapshot(),
            "prediction": self.prediction_snapshot(),
            "statistics": self.statistics(),
            "live_evaluation": self.ledger.live_metrics(),
            "model_metrics": self.predictor.offline_metrics(),
            "collector": {
                "poll_count": self.poll_count,
                "error_count": self.error_count,
                "poll_interval_seconds": self.poll_seconds,
                "last_error": self.last_error,
            },
        }


_collector: Optional[DataCollector] = None


def get_collector() -> DataCollector:
    global _collector
    if _collector is None:
        _collector = DataCollector()
    return _collector
