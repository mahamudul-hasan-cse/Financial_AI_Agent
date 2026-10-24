"""Pydantic schemas for the Financial AI Agent API.

This package re-exports every public model so that existing imports
like ``from schemas import ChatRequest`` continue to work unchanged.

Modules:
  • ``schemas.nlp``        — NLP entities, sentiment, metadata
  • ``schemas.chat``       — Chat request / response + MAX_MESSAGE_LENGTH
  • ``schemas.streaming``  — SSE event models
  • ``schemas.common``     — Health, stats, session, error, tool responses
"""

from schemas.alerts import (
    AlertCheckItem,
    AlertCheckResponse,
    AlertCreateRequest,
    AlertListResponse,
    AlertResponse,
    AlertUpdateRequest,
)

# ── chat.py ──────────────────────────────────────────────────────────
from schemas.chat import (
    MAX_MESSAGE_LENGTH,
    ChatRequest,
    ChatResponse,
    ExportMessage,
    ExportRequest,
)

# ── common.py ────────────────────────────────────────────────────────
from schemas.common import (
    ErrorResponse,
    FallbackModel,
    FallbackStatus,
    HealthResponse,
    LastLLMUsage,
    SessionClearResponse,
    StatsResponse,
    ToolResponse,
)

# ── nlp.py ───────────────────────────────────────────────────────────
from schemas.nlp import (
    Entity,
    IntentMethod,
    NLPMetadataSchema,
    NLPStatusDetail,
    SentimentLabel,
    SentimentResult,
)
from schemas.research import (
    CompanySnapshotResponse,
    ComparisonResponse,
    ComparisonRow,
    ExplainabilityItem,
    ResearchMetric,
    ResearchResponse,
    ResearchSignal,
    ResearchSource,
    SnapshotHeadline,
    SnapshotPoint,
)
from schemas.sessions import (
    SessionDetail,
    SessionListResponse,
    SessionMessage,
    SessionRenameRequest,
    SessionSummary,
)

# ── streaming.py ─────────────────────────────────────────────────────
from schemas.streaming import (
    StreamEvent,
    StreamEventChunk,
    StreamEventDone,
    StreamEventError,
    StreamEventNLPMeta,
    StreamEventResearch,
    StreamEventType,
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
    "StreamEventResearch",
    "StreamEventType",
    # common
    "ErrorResponse",
    "HealthResponse",
    "FallbackModel",
    "FallbackStatus",
    "LastLLMUsage",
    "SessionClearResponse",
    "StatsResponse",
    "ToolResponse",
    # watchlist
    "WatchlistAddRequest",
    "WatchlistItem",
    "WatchlistRemoveRequest",
    "WatchlistResponse",
    # research
    "CompanySnapshotResponse",
    "ComparisonResponse",
    "ComparisonRow",
    "ExplainabilityItem",
    "ResearchMetric",
    "ResearchResponse",
    "ResearchSignal",
    "ResearchSource",
    "SnapshotHeadline",
    "SnapshotPoint",
    # sessions
    "SessionDetail",
    "SessionListResponse",
    "SessionMessage",
    "SessionRenameRequest",
    "SessionSummary",
    # alerts
    "AlertCheckItem",
    "AlertCheckResponse",
    "AlertCreateRequest",
    "AlertListResponse",
    "AlertResponse",
    "AlertUpdateRequest",
]
