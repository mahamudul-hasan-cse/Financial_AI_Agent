"""Chat request / response schemas.

Defines the Pydantic models that FastAPI uses to validate the
``POST /api/chat`` and ``POST /api/chat/stream`` endpoints.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from schemas.common import SESSION_ID_PATTERN
from schemas.nlp import NLPMetadataSchema
from schemas.research import ResearchResponse

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# ── Configuration constant (imported by api.py too) ──────────────────
MAX_MESSAGE_LENGTH: int = 2_000


# ── Request ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    """Request body for ``POST /api/chat`` and ``POST /api/chat/stream``."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The user's message to the financial agent.",
    )
    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="Optional session ID for conversation continuity ([A-Za-z0-9_-]{1,64}).",
    )

    @field_validator("message")
    @classmethod
    def strip_html_tags(cls, v: str) -> str:
        """Remove any HTML tags from the message to prevent injection."""
        return _HTML_TAG_RE.sub("", v)


# ── Response ─────────────────────────────────────────────────────────
class ChatResponse(BaseModel):
    """Response body for ``POST /api/chat``."""

    response: str = Field(..., description="Agent textual reply.")
    session_id: str = Field(..., description="Session ID for follow-up calls.")
    session_title: str | None = Field(
        default=None,
        description="Human-friendly saved session title.",
    )
    nlp_metadata: NLPMetadataSchema | None = Field(
        default=None,
        description="NLP analysis of the user query.",
    )
    structured_response: ResearchResponse | None = Field(
        default=None,
        description="Structured investor answer card rendered by the frontend.",
    )
    llm_provider: str | None = Field(
        default=None,
        description='LLM provider that answered: "openrouter", "groq" or "ollama".',
    )
    llm_model: str | None = Field(default=None, description="Model id that answered.")
    llm_fallback: bool = Field(
        default=False,
        description="True when the primary model was busy/down and a backup model answered.",
    )


# ── Export ──────────────────────────────────────────────────────────
class ExportMessage(BaseModel):
    """A single message in an export request."""

    role: str = Field(..., max_length=20, description="Message role: 'user' or 'agent'.")
    content: str = Field(..., max_length=50_000, description="Message text content.")


class ExportRequest(BaseModel):
    """Request body for ``POST /api/chat/export``."""

    session_id: str = Field(
        default="unknown",
        pattern=SESSION_ID_PATTERN,
        description="Session ID ([A-Za-z0-9_-]{1,64}); must belong to the caller.",
    )
    messages: list[ExportMessage] | None = Field(
        default=None,
        max_length=500,
        description="Messages to export. Falls back to server-side session if omitted.",
    )
