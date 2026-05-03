"""Chat request / response schemas.

Defines the Pydantic models that FastAPI uses to validate the
``POST /api/chat`` and ``POST /api/chat/stream`` endpoints.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

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
        description="Optional session ID for conversation continuity.",
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


# ── Export ──────────────────────────────────────────────────────────
class ExportMessage(BaseModel):
    """A single message in an export request."""

    role: str = Field(..., description="Message role: 'user' or 'agent'.")
    content: str = Field(..., description="Message text content.")


class ExportRequest(BaseModel):
    """Request body for ``POST /api/chat/export``."""

    session_id: str = Field(
        default="unknown",
        description="Session ID (used for filename generation).",
    )
    messages: list[ExportMessage] | None = Field(
        default=None,
        description="Messages to export. Falls back to server-side session if omitted.",
    )
