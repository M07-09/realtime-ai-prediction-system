"""Pydantic models describing the request and response payloads of the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          description="Question about the live data or the forecast")
    history: List[ChatMessage] = Field(default_factory=list,
                                       description="Previous turns of the conversation")


class ChatResponse(BaseModel):
    answer: str
    intent: str
    engine: str
    deterministic_answer: Optional[str] = None
    grounded_facts: Optional[str] = None
    model_loaded: bool = False
    model_error: Optional[str] = None
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    external_api_reachable: bool
    collector_running: bool
    model_loaded: bool
    chatbot_loaded: bool
    poll_count: int
    error_count: int
    consecutive_errors: int
    last_error: Optional[str] = None
    uptime_seconds: float
    server_time: str


class GenericResponse(BaseModel):
    """Envelope used by the data endpoints so the client can always check 'available'."""
    available: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
