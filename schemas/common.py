"""Common / shared response schemas.

Contains endpoint response models that don't belong to the chat domain:
health checks, analytics stats, session management, error responses,
and the generic tool-output wrapper.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from schemas.nlp import NLPStatusDetail


# ── Health ───────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    status: str = "ok"
    groq_configured: bool = False
    active_sessions: int = 0
    total_messages: int = 0
    nlp_status: NLPStatusDetail = Field(default_factory=NLPStatusDetail)


# ── Stats ────────────────────────────────────────────────────────────
class StatsResponse(BaseModel):
    """Response body for ``GET /api/stats``."""

    total_messages: int = 0
    active_sessions: int = 0
    total_sessions_created: int = 0
    intent_distribution: dict[str, int] = Field(default_factory=dict)
    avg_response_time_ms: float = 0.0
    classifier_usage: dict[str, int] = Field(default_factory=dict)


# ── Session ──────────────────────────────────────────────────────────
class SessionClearResponse(BaseModel):
    """Response body for ``DELETE /api/chat/{session_id}``."""

    status: str = "cleared"


# ── Error ────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standard error body returned by HTTP 4xx / 5xx responses.

    Matches the shape FastAPI uses for ``HTTPException`` by default so
    existing frontend error-handling continues to work.
    """

    detail: str = Field(..., description="Human-readable error message.")


# ── Tool ─────────────────────────────────────────────────────────────
class ToolResponse(BaseModel):
    """Generic wrapper returned by agent tools (chart, excel, RAG).

    Tools may set ``file_path`` when they produce a downloadable artefact,
    or populate ``data`` with structured payload.
    """

    success: bool = True
    message: str = Field(..., description="Human-readable result summary.")
    file_path: str | None = Field(
        default=None,
        description="Relative path to an output file (chart image, Excel sheet, …).",
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured payload.",
    )
