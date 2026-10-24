"""Chat endpoints: non-streaming, SSE streaming, and session clearing.

SSE protocol (``POST /api/chat/stream``), one ``data:`` line per event:

* ``[NLP_META]{json}``      – NLP analysis of the question (always first)
* ``"<json string>"``       – a chunk of answer text (JSON-encoded so
  newlines survive SSE framing)
* ``[ERROR] <message>``     – generation failed (stream still ends with DONE)
* ``[FINAL]"<json string>"`` – the cleaned, persisted answer; clients should
  replace the streamed text with it
* ``[RESEARCH_DATA]{json}`` – structured research card
* ``[DONE]``                – end of stream (always last)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Generator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import persistence
from backend.app.routers.common import ClientId, SessionIdPath
from backend.app.services import chat as chat_service
from backend.app.services import sessions as session_service
from backend.app.state import record_response_time
from research_service import build_structured_response
from schemas import ChatRequest, ChatResponse, SessionClearResponse

logger = logging.getLogger("api.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])

_STREAM_ERROR = "[ERROR] Unable to generate a response right now. Please try again in a moment."
_MAX_STREAM_ATTEMPTS = 2


def _sse(payload: str) -> str:
    return f"data: {payload}\n\n"


def _ready_agent():
    """Return the agent service, or raise 503 if no LLM provider is configured."""
    service = chat_service.agent()
    try:
        service.ensure_ready()
    except chat_service.AgentNotConfiguredError as exc:
        logger.warning("Chat rejected: LLM not configured (%s)", exc)
        raise HTTPException(
            status_code=503,
            detail="The AI model is not configured on this server. See /api/health.",
        ) from exc
    return service


@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, client_id: ClientId) -> ChatResponse:
    """Non-streaming chat: NLP preprocessing → agent → structured response."""
    session_id, history = session_service.get_or_create_session(req.session_id, client_id)
    session_service.check_rate_limit(session_id, request)
    service = _ready_agent()
    logger.info("Chat request [%s]: %.80s", session_id, req.message)

    nlp_meta = chat_service.process_nlp_and_track(req.message, session_id)
    sentiment_ctx = chat_service.get_news_sentiment_context(nlp_meta)
    prompt = chat_service.build_prompt(
        req.message, history, intent=nlp_meta.intent, sentiment_context=sentiment_ctx
    )

    t0 = time.time()
    try:
        content: str = service.run(prompt, session_id=session_id)
    except chat_service.AgentNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail="The AI model is not configured.") from exc
    except Exception as exc:
        logger.exception("Agent error for session %s", session_id)
        raise HTTPException(
            status_code=502,
            detail="An internal error occurred while processing your request. Please try again.",
        ) from exc
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    record_response_time(elapsed_ms)

    content = chat_service.append_disclaimer(chat_service.clean_response(str(content)))
    structured = build_structured_response(req.message, content, nlp_meta)
    chat_service.persist_exchange(session_id, history, req.message, content, nlp_meta, structured)

    stored = persistence.get_session_summary(session_id)
    logger.info("Chat response [%s]: %d chars in %.0fms", session_id, len(content), elapsed_ms)
    return ChatResponse(
        response=content,
        session_id=session_id,
        session_title=stored["title"] if stored else None,
        nlp_metadata=nlp_meta,
        structured_response=structured,
    )


@router.post("/stream")
def chat_stream(req: ChatRequest, request: Request, client_id: ClientId) -> StreamingResponse:
    """Streaming chat endpoint (Server-Sent Events) — see module docstring."""
    session_id, history = session_service.get_or_create_session(req.session_id, client_id)
    session_service.check_rate_limit(session_id, request)
    service = _ready_agent()
    logger.info("Stream request [%s]: %.80s", session_id, req.message)

    nlp_meta = chat_service.process_nlp_and_track(req.message, session_id)

    def generate() -> Generator[str, None, None]:
        yield _sse(f"[NLP_META]{nlp_meta.model_dump_json()}")

        # Headline sentiment is fetched after NLP_META so the client gets
        # immediate feedback while the (cached) news lookup runs.
        sentiment_ctx = chat_service.get_news_sentiment_context(nlp_meta)
        prompt = chat_service.build_prompt(
            req.message, history, intent=nlp_meta.intent, sentiment_context=sentiment_ctx
        )

        collected: list[str] = []
        errored = False
        t0 = time.time()
        try:
            for attempt in range(1, _MAX_STREAM_ATTEMPTS + 1):
                errored = False
                for text in service.stream(prompt, session_id=session_id):
                    if text.startswith("[ERROR]"):
                        errored = True
                        break
                    collected.append(text)
                    yield _sse(json.dumps(text))
                # Retry only if nothing reached the client yet — retrying after
                # partial output would duplicate text.
                if not (errored and not collected):
                    break
                logger.warning("Stream attempt %d failed before output; retrying", attempt)
            if not collected and attempt > 1:
                # The retry produced nothing at all — report the original failure.
                errored = True
        except Exception:
            logger.exception("Unexpected error in stream generator for session %s", session_id)
            errored = True

        record_response_time(round((time.time() - t0) * 1000, 1))

        if errored:
            yield _sse(_STREAM_ERROR)
        elif collected:
            final = chat_service.append_disclaimer(chat_service.clean_response("".join(collected)))
            structured = build_structured_response(req.message, final, nlp_meta)
            chat_service.persist_exchange(
                session_id, history, req.message, final, nlp_meta, structured
            )
            yield _sse(f"[FINAL]{json.dumps(final)}")
            yield _sse(f"[RESEARCH_DATA]{json.dumps(structured)}")

        yield _sse("[DONE]")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/{session_id}", response_model=SessionClearResponse)
def clear_session(session_id: SessionIdPath, client_id: ClientId) -> SessionClearResponse:
    """Delete one of the caller's sessions (idempotent for unknown ids)."""
    session_service.delete_owned_session(session_id, client_id)
    logger.info("Cleared session %s", session_id)
    return SessionClearResponse(status="cleared")
