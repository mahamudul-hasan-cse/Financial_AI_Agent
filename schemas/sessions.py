"""Saved research session schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from schemas.chat import ExportMessage
from schemas.research import ResearchResponse
from schemas.nlp import NLPMetadataSchema


class SessionMessage(BaseModel):
    """Persisted chat message returned from the saved session API."""

    role: str
    content: str
    timestamp: float | int
    nlp_metadata: NLPMetadataSchema | None = None
    structured_response: ResearchResponse | None = None


class SessionSummary(BaseModel):
    """Compact session item used in the sidebar."""

    session_id: str
    title: str
    preview: str = ""
    updated_at: float | int
    message_count: int = 0


class SessionDetail(BaseModel):
    """Full saved research session."""

    session_id: str
    title: str
    updated_at: float | int
    messages: list[SessionMessage] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    """Saved sessions response."""

    sessions: list[SessionSummary] = Field(default_factory=list)


class SessionRenameRequest(BaseModel):
    """Rename request for a saved session."""

    title: str = Field(..., min_length=1, max_length=120)

