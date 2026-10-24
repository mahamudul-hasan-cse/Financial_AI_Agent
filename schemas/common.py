"""Common / shared response schemas.

Contains endpoint response models that don't belong to the chat domain:
health checks, analytics stats, session management, error responses,
and the generic tool-output wrapper.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from schemas.nlp import NLPStatusDetail

# Session ids: letters, digits, "_" and "-" (UUIDs and "sess_<uuid>" match).
# No dots or slashes, so a session id can never form a path that escapes a
# directory.  Shared by every request schema and path parameter.
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
SESSION_ID_MAX_LENGTH = 64


# ── Health ───────────────────────────────────────────────────────────
class FallbackModel(BaseModel):
    """One step of the backup chain."""

    provider: str
    model: str


class FallbackStatus(BaseModel):
    """Backup models used when the primary LLM is rate-limited or unavailable."""

    provider: str = Field(
        default="none",
        description='Backup provider(s), e.g. "openrouter", "openrouter,groq"; "none" = disabled.',
    )
    configured: bool = False
    models: list[str] = Field(default_factory=list, description="Backup models, tried in order.")
    chain: list[FallbackModel] = Field(
        default_factory=list, description="Backup provider/model pairs, tried in order."
    )
    detail: str | None = None


class LastLLMUsage(BaseModel):
    """Provider/model that produced the most recent answer in this process."""

    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    at: str | None = Field(default=None, description="ISO-8601 UTC time of that answer.")


class HealthResponse(BaseModel):
    """Response body for ``GET /api/health``."""

    status: str = Field(default="ok", description='"ok", or "degraded" when no LLM is usable.')
    llm_provider: str = Field(default="groq", description="Primary provider.")
    llm_model: str | None = Field(default=None, description="Primary model id.")
    llm_configured: bool = Field(default=False, description="True when the primary is usable.")
    detail: str | None = Field(default=None, description="Why the service is degraded.")
    groq_configured: bool = False
    openrouter_configured: bool = False
    fallback: FallbackStatus = Field(default_factory=FallbackStatus)
    last_llm: LastLLMUsage | None = Field(
        default=None, description="Who answered last (null until the first answer)."
    )
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
