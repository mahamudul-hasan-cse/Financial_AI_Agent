"""Pydantic schemas for the Financial AI Agent API.

This package re-exports every public model so that existing imports
like ``from schemas import ChatRequest`` continue to work unchanged.

Modules:
  • ``schemas.nlp``        — NLP entities, sentiment, metadata
  • ``schemas.chat``       — Chat request / response + MAX_MESSAGE_LENGTH
  • ``schemas.streaming``  — SSE event models
  • ``schemas.common``     — Health, stats, session, error, tool responses
"""

# ── nlp.py ───────────────────────────────────────────────────────────
from schemas.nlp import (
    Entity,
    IntentMethod,
    NLPMetadataSchema,
    NLPStatusDetail,
    SentimentLabel,
    SentimentResult,
)

# ── chat.py ──────────────────────────────────────────────────────────
from schemas.chat import (
    MAX_MESSAGE_LENGTH,
    ChatRequest,
    ChatResponse,
    ExportMessage,
    ExportRequest,
)

# ── streaming.py ─────────────────────────────────────────────────────
from schemas.streaming import (
    StreamEvent,
    StreamEventChunk,
    StreamEventDone,
    StreamEventError,
    StreamEventNLPMeta,
    StreamEventType,
)

# ── common.py ────────────────────────────────────────────────────────
from schemas.common import (
    ErrorResponse,
    HealthResponse,
    SessionClearResponse,
    StatsResponse,
    ToolResponse,
)

# ── watchlist.py ──────────────────────────────────────────────────────
from schemas.watchlist import (
    WatchlistAddRequest,
    WatchlistItem,
    WatchlistRemoveRequest,
    WatchlistResponse,
)

__all__ = [
    # nlp
    "Entity",
    "IntentMethod",
    "NLPMetadataSchema",
    "NLPStatusDetail",
    "SentimentLabel",
    "SentimentResult",
    # chat
    "MAX_MESSAGE_LENGTH",
    "ChatRequest",
    "ChatResponse",
    "ExportMessage",
    "ExportRequest",
    # streaming
    "StreamEvent",
    "StreamEventChunk",
    "StreamEventDone",
    "StreamEventError",
    "StreamEventNLPMeta",
    "StreamEventType",
    # common
    "ErrorResponse",
    "HealthResponse",
    "SessionClearResponse",
    "StatsResponse",
    "ToolResponse",
    # watchlist
    "WatchlistAddRequest",
    "WatchlistItem",
    "WatchlistRemoveRequest",
    "WatchlistResponse",
]
