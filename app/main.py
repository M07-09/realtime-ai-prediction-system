"""
FastAPI backend - the middle tier of the architecture.

    External API  ->  FastAPI  ->  Data processing  ->  LSTM  ->  Prediction
                                                                     |
                                              Streamlit + Transformer chatbot

The background collector starts with the application (lifespan) and keeps
polling the external API every 10 seconds. Every endpoint reads the collector's
latest snapshot, so an HTTP request never blocks on a network call.

Run:
    uvicorn app.main:app --host 127.0.0.1 --port 8000
    python -m app.main
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.chatbot import SUGGESTED_QUESTIONS, get_chatbot
from app.collector import get_collector
from app.config import (
    ASSET_NAME,
    BACKEND_HOST,
    BACKEND_PORT,
    INTERVAL,
    POLL_INTERVAL_SECONDS,
    SYMBOL,
)
from app.crypto_api import ExternalAPIError, get_client
from app.predictor import get_predictor
from app.utils import get_logger

log = get_logger("backend", "backend.log")
START_TIME = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start the 10-second polling loop with the server; stop it on shutdown."""
    collector = get_collector()
    log.info("Starting the real-time collector ...")
    await collector.start()
    try:
        yield
    finally:
        log.info("Shutting down the collector ...")
        await collector.stop()


app = FastAPI(
    title="Real-Time Crypto AI Prediction API",
    description=(
        "Backend of the Real-Time AI Prediction System. It polls the Binance public "
        "API every 10 seconds, processes the data, runs an LSTM forecast for the next "
        "minute and serves a Transformer chatbot grounded in those numbers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Error handling: no traceback ever reaches the client
# --------------------------------------------------------------------------
@app.exception_handler(ExternalAPIError)
async def external_api_error_handler(_, exc: ExternalAPIError):
    log.warning("External API error: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"available": False, "error": "external_api_unavailable", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"available": False, "error": type(exc).__name__,
                 "message": "The server hit an unexpected error but is still running."},
    )


# --------------------------------------------------------------------------
# Meta
# --------------------------------------------------------------------------
@app.get("/", tags=["meta"])
async def root() -> Dict[str, Any]:
    return {
        "name": "Real-Time Crypto AI Prediction API",
        "asset": f"{ASSET_NAME} ({SYMBOL})",
        "external_api": "Binance public REST API",
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "candle_interval": INTERVAL,
        "forecast_horizon_seconds": 60,
        "docs": "/docs",
        "endpoints": [
            "/health", "/api/live", "/api/history", "/api/prediction",
            "/api/predictions/history", "/api/stats", "/api/model/metrics",
            "/api/status", "/api/chat", "/api/chat/suggestions", "/api/refresh",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health() -> HealthResponse:
    collector = get_collector()
    predictor = get_predictor()
    chatbot = get_chatbot()

    reachable = collector.consecutive_errors == 0 and collector.success_count > 0
    if collector.poll_count == 0:                     # nothing polled yet - test directly
        reachable = get_client().ping()

    if reachable and predictor.loaded:
        status = "healthy"
    elif reachable:
        status = "degraded: model not loaded"
    else:
        status = "degraded: external API unreachable"

    return HealthResponse(
        status=status,
        external_api_reachable=reachable,
        collector_running=collector.running,
        model_loaded=predictor.loaded,
        chatbot_loaded=chatbot.loaded,
        poll_count=collector.poll_count,
        error_count=collector.error_count,
        consecutive_errors=collector.consecutive_errors,
        last_error=collector.last_error,
        uptime_seconds=time.time() - START_TIME,
        server_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------
# Live data
# --------------------------------------------------------------------------
@app.get("/api/live", tags=["data"])
async def live() -> Dict[str, Any]:
    """Most recent observation from the external API (refreshed every 10 s)."""
    return get_collector().live_snapshot()


@app.get("/api/history", tags=["data"])
async def history(limit: int = Query(240, ge=10, le=1000)) -> Dict[str, Any]:
    """The rolling window of 1-minute candles held in memory."""
    collector = get_collector()
    records = collector.history_records(limit)
    if not records:
        return {"available": False, "message": "No historical data collected yet", "bars": []}
    return {
        "available": True,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "count": len(records),
        "bars": records,
    }


@app.get("/api/ticks", tags=["data"])
async def ticks() -> Dict[str, Any]:
    """Every 10-second observation of this session (the real-time stream)."""
    collector = get_collector()
    data = collector.ticks.to_list()
    return {"available": bool(data), "count": len(data), "ticks": data}


@app.get("/api/stats", tags=["data"])
async def stats() -> Dict[str, Any]:
    """Descriptive statistics and trend of the collected window."""
    return get_collector().statistics()


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
@app.get("/api/prediction", tags=["prediction"])
async def prediction() -> Dict[str, Any]:
    """The latest LSTM forecast for the price 60 seconds ahead."""
    return get_collector().prediction_snapshot()


@app.get("/api/predictions/history", tags=["prediction"])
async def prediction_history(limit: int = Query(120, ge=1, le=500)) -> Dict[str, Any]:
    """Forecasts whose target minute has closed, paired with the real price."""
    collector = get_collector()
    resolved = collector.ledger.resolved[-limit:]
    return {
        "available": bool(resolved),
        "count": len(resolved),
        "records": resolved,
        "live_evaluation": collector.ledger.live_metrics(),
    }


@app.get("/api/model/metrics", tags=["prediction"])
async def model_metrics() -> Dict[str, Any]:
    """Offline evaluation produced by scripts/train_lstm.py."""
    return get_predictor().offline_metrics()


@app.post("/api/model/reload", tags=["prediction"])
async def reload_model() -> Dict[str, Any]:
    """Load a freshly trained checkpoint without restarting the server."""
    ok = get_predictor().reload()
    return {"reloaded": ok, "error": get_predictor().load_error}


# --------------------------------------------------------------------------
# One-call snapshot for the dashboard
# --------------------------------------------------------------------------
@app.get("/api/status", tags=["data"])
async def status(history_limit: int = Query(240, ge=10, le=1000)) -> Dict[str, Any]:
    """Live value, history, forecast, statistics and evaluation in one response."""
    return get_collector().full_status(history_limit)


@app.post("/api/refresh", tags=["data"])
async def refresh() -> Dict[str, Any]:
    """Force an immediate poll instead of waiting for the next 10-second tick."""
    result = await get_collector().poll_once()
    return result


# --------------------------------------------------------------------------
# Transformer chatbot
# --------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse, tags=["chatbot"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Ask the Transformer chatbot a question about the live data."""
    import asyncio

    collector = get_collector()
    chatbot = get_chatbot()
    context = collector.chatbot_context()
    history = [{"role": m.role, "content": m.content} for m in request.history]

    try:
        result = await asyncio.to_thread(chatbot.answer, request.question, context, history)
    except Exception as exc:                          # noqa: BLE001
        log.exception("Chatbot failed")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {exc}") from exc

    return ChatResponse(**result)


@app.get("/api/chat/suggestions", tags=["chatbot"])
async def chat_suggestions() -> Dict[str, Any]:
    chatbot = get_chatbot()
    return {
        "suggestions": SUGGESTED_QUESTIONS,
        "model": chatbot.model_name,
        "loaded": chatbot.loaded,
        "error": chatbot.load_error,
    }


@app.post("/api/chat/warmup", tags=["chatbot"])
async def chat_warmup() -> Dict[str, Any]:
    """Pre-load the Transformer so the first question is not slow."""
    import asyncio

    chatbot = get_chatbot()
    ok = await asyncio.to_thread(chatbot.load)
    return {"loaded": ok, "model": chatbot.model_name,
            "device": chatbot.device, "error": chatbot.load_error}


def run() -> None:
    import uvicorn
    uvicorn.run("app.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=False)


if __name__ == "__main__":
    run()
