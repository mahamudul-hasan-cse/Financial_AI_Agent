"""Server-Sent Events (SSE) streaming schemas.

Defines the discriminated union of SSE event types emitted by the
``POST /api/chat/stream`` endpoint.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel

from schemas.nlp import NLPMetadataSchema


class StreamEventType(str, Enum):
    """Discriminator for SSE event payloads."""

    nlp_meta = "nlp_meta"
    chunk = "chunk"
    error = "error"
    done = "done"


class StreamEventNLPMeta(BaseModel):
    """First SSE event — carries the NLP metadata."""

    event: Literal[StreamEventType.nlp_meta] = StreamEventType.nlp_meta
    data: NLPMetadataSchema


class StreamEventChunk(BaseModel):
    """Intermediate SSE event — a text chunk from the agent."""

    event: Literal[StreamEventType.chunk] = StreamEventType.chunk
    data: str


class StreamEventError(BaseModel):
    """SSE event emitted when the agent fails mid-stream."""

    event: Literal[StreamEventType.error] = StreamEventType.error
    data: str


class StreamEventDone(BaseModel):
    """Terminal SSE event — signals end of stream."""

    event: Literal[StreamEventType.done] = StreamEventType.done
    data: None = None


# Discriminated union of all SSE event types.
StreamEvent = StreamEventNLPMeta | StreamEventChunk | StreamEventError | StreamEventDone
