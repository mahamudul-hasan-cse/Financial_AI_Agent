"""Financial AI Agent API.

FastAPI backend that exposes the financial AI agent via REST endpoints.
Supports both streaming (SSE) and non-streaming chat, with in-memory
session management, rate limiting, and automatic session cleanup.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Generator, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

import cache
from config import settings
from financial_agent import agent_service
from market_service import build_comparison, build_watchlist_items, get_market_context
from nlp_pipeline import run_nlp_pipeline, NLPMetadata
from persistence import (
    add_watchlist_symbol,
    append_message,
    create_alert,
    delete_alert,
    delete_session,
    get_alert,
    get_session,
    list_alerts,
    list_sessions as list_saved_sessions,
    list_watchlist,
    mark_alert_triggered,
    prune_old_sessions,
    rename_session,
    remove_watchlist_symbol,
    upsert_session,
    update_alert,
)
from research_service import build_structured_response
from schemas import (
    AlertCheckItem,
    AlertCheckResponse,
    AlertCreateRequest,
    AlertListResponse,
    AlertResponse,
    AlertUpdateRequest,
    ChatRequest,
    ChatResponse,
    CompanySnapshotResponse,
    ComparisonResponse,
    ErrorResponse,
    ExportRequest,
    HealthResponse,
    NLPMetadataSchema,
    NLPStatusDetail,
    SessionClearResponse,
    SessionDetail,
    SessionListResponse,
    SessionRenameRequest,
    SessionSummary,
    StatsResponse,
    WatchlistAddRequest,
    WatchlistItem,
    WatchlistRemoveRequest,
    WatchlistResponse,
    MAX_MESSAGE_LENGTH,
)

load_dotenv(override=True)

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ── Configuration ────────────────────────────────────────────────────
AGENT_TIMEOUT: int = settings.agent_timeout
# MAX_MESSAGE_LENGTH is imported from schemas
MAX_SESSIONS: int = settings.max_sessions
SESSION_TTL: int = settings.session_ttl
RATE_LIMIT_WINDOW: int = settings.rate_limit_window
RATE_LIMIT_MAX: int = settings.rate_limit_max
API_KEY: str | None = settings.api_key
IP_RATE_LIMIT_MAX: int = settings.ip_rate_limit_max

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Financial AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# GZip compression for responses over 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Request logging middleware ────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, response time, and client IP.

    Never logs request or response bodies to avoid exposing sensitive data.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        client_ip = request.client.host if request.client else "unknown"
        t0 = time.time()
        response = await call_next(request)
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s %d %.1fms client=%s request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            client_ip,
            request_id,
        )
        return response


app.add_middleware(RequestLoggingMiddleware)


# ── API key validation middleware ─────────────────────────────────────

# Paths that never require authentication
_AUTH_EXEMPT_PATHS: set[str] = {"/api/health", "/docs", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require ``X-API-Key`` header on all ``/api/`` endpoints except health.

    Disabled when the ``API_KEY`` env-var is unset (dev mode).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if API_KEY and request.url.path.startswith("/api/"):
            if request.url.path not in _AUTH_EXEMPT_PATHS:
                provided = request.headers.get("X-API-Key", "")
                if provided != API_KEY:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing API key."},
                    )
        return await call_next(request)


app.add_middleware(APIKeyMiddleware)


# ── Global exception handler ─────────────────────────────────────────
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception, log the full traceback, and return
    a clean JSON error — never leaking stack traces to the client."""
    import traceback
    print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# ── Typed in-memory stores ────────────────────────────────────────────


class SessionData(TypedDict):
    """Shape of a single session stored in the session dict."""

    messages: list[dict[str, str]]
    last_active: float
    watchlist: list[str]


# Maps session_id -> session data
sessions: dict[str, SessionData] = {}

# Maps session_id -> list of request timestamps (sliding window)
_rate_limits: dict[str, list[float]] = defaultdict(list)
# Maps client IP -> list of request timestamps (per-IP sliding window)
_ip_rate_limits: dict[str, list[float]] = defaultdict(list)


class _StatsStore(TypedDict):
    """Shape of the in-memory analytics counters."""

    total_messages: int
    total_sessions_created: int
    intent_counts: defaultdict[str, int]
    response_times_ms: list[float]
    classifier_usage: defaultdict[str, int]


_stats: _StatsStore = {
    "total_messages": 0,
    "total_sessions_created": 0,
    "intent_counts": defaultdict(int),
    "response_times_ms": [],
    "classifier_usage": defaultdict(int),
}


# ── Models (imported from schemas.py) ────────────────────────────────
# ChatRequest, ChatResponse, HealthResponse, StatsResponse,
# SessionClearResponse, ErrorResponse, NLPMetadataSchema, NLPStatusDetail
# are all defined in schemas.py and imported at the top of this file.


# ── Helpers ──────────────────────────────────────────────────────────
def clean_response(text: str) -> str:
    raw = text
    # Strip markdown bold
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove section label prefixes
    labels = [
        "Quick Take:", "Key Data:", "What It Means:",
        "Risk:", "Next Steps:", "Summary:", "Overview:",
        "Analysis:", "Verdict:", "Bottom Line:", "Insight:",
        "Context:", "Recommendation:"
    ]
    for label in labels:
        text = text.replace(label, "")
    # Collapse 3+ blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)
    cleaned = text.strip()
    print(f"[CLEAN] response length before={len(raw)} after={len(cleaned)}", flush=True)
    return cleaned


def _cleanup_expired_sessions() -> int:
    """Remove sessions that have been inactive for longer than SESSION_TTL.

    Returns the number of sessions removed.
    """
    removed = set(prune_old_sessions(MAX_SESSIONS, SESSION_TTL))
    now = time.time()
    expired_runtime = [
        sid for sid, data in sessions.items()
        if now - data["last_active"] > SESSION_TTL
    ]
    for sid in expired_runtime:
        removed.add(sid)
    for sid in list(removed):
        sessions.pop(sid, None)
        _rate_limits.pop(sid, None)

    if len(sessions) >= MAX_SESSIONS:
        overflow = len(sessions) - MAX_SESSIONS + 1
        oldest = sorted(sessions.items(), key=lambda item: item[1]["last_active"])[:overflow]
        for sid, _ in oldest:
            removed.add(sid)
            sessions.pop(sid, None)
            _rate_limits.pop(sid, None)
            delete_session(sid)

    if removed:
        logger.info("Cleaned up %d expired/purged sessions", len(removed))
    return len(removed)


def _load_persisted_session(session_id: str) -> SessionData | None:
    """Load a session from SQLite into the in-memory compatibility cache."""

    stored = get_session(session_id)
    if stored is None:
        return None
    watchlist = list_watchlist(session_id)
    payload: SessionData = {
        "messages": [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in stored["messages"]
        ],
        "last_active": time.time(),
        "watchlist": watchlist,
    }
    sessions[session_id] = payload
    return payload


def _get_or_create_session(session_id: str | None) -> tuple[str, list[dict[str, str]]]:
    """Retrieve an existing session or create a new one.

    Runs session cleanup and enforces MAX_SESSIONS limit.
    Returns (session_id, message_history).
    """
    _cleanup_expired_sessions()

    if session_id and session_id in sessions:
        sessions[session_id]["last_active"] = time.time()
        upsert_session(session_id)
        return session_id, sessions[session_id]["messages"]

    if session_id:
        loaded = _load_persisted_session(session_id)
        if loaded is not None:
            return session_id, loaded["messages"]

    new_id = session_id or str(uuid.uuid4())
    upsert_session(new_id)
    sessions[new_id] = {"messages": [], "last_active": time.time(), "watchlist": []}
    _stats["total_sessions_created"] += 1
    _cleanup_expired_sessions()
    return new_id, sessions[new_id]["messages"]


def _sliding_window_check(
    store: dict[str, list[float]], key: str, max_requests: int
) -> float | None:
    """Check a sliding-window rate limit for *key*.

    Returns ``None`` if the request is allowed, or the number of seconds
    until the next slot opens (for the ``Retry-After`` header).
    """
    now = time.time()
    store[key] = [t for t in store[key] if now - t < RATE_LIMIT_WINDOW]
    if len(store[key]) >= max_requests:
        oldest = store[key][0]
        return math.ceil(RATE_LIMIT_WINDOW - (now - oldest))
    store[key].append(now)
    return None


def _check_rate_limit(session_id: str, request: Request | None = None) -> None:
    """Enforce per-session AND per-IP sliding-window rate limiting.

    Raises HTTPException(429) with a ``Retry-After`` header if either
    limit is exceeded.
    """
    # Per-session check
    retry_after = _sliding_window_check(_rate_limits, session_id, RATE_LIMIT_MAX)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX} requests per minute.",
            headers={"Retry-After": str(retry_after)},
        )

    # Per-IP check
    if request and request.client:
        client_ip = request.client.host
        retry_after = _sliding_window_check(
            _ip_rate_limits, client_ip, IP_RATE_LIMIT_MAX
        )
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail=f"IP rate limit exceeded. Maximum {IP_RATE_LIMIT_MAX} requests per minute.",
                headers={"Retry-After": str(retry_after)},
            )


def _build_context(history: list[dict[str, str]]) -> str:
    """Build a conversation context string from the last 10 messages."""
    if not history:
        return ""
    lines: list[str] = []
    for msg in history[-10:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _build_prompt(
    user_message: str,
    history: list[dict[str, str]],
    intent: str = "general",
    sentiment_context: str | None = None,
) -> str:
    """Construct the full prompt including intent directive, conversation history,
    and optional VADER news-sentiment context."""
    from backend.app.agent.prompts import get_intent_instruction
    format_directive = get_intent_instruction(intent)

    parts: list[str] = [f"[INTENT: {intent.upper()}]\n{format_directive}"]

    if sentiment_context:
        parts.append(f"[MARKET SENTIMENT]\n{sentiment_context}")

    context = _build_context(history)
    if context:
        parts.append(f"Previous conversation:\n{context}")

    parts.append(f"User: {user_message}")
    return "\n\n".join(parts)


def _run_agent(prompt: str) -> str:
    """Execute the agent synchronously and return the response text.

    Delegates to ``agent_service.run()`` which handles retries internally.
    """
    return agent_service.run(prompt)


_SENTIMENT_INTENTS: frozenset[str] = frozenset({"stock_price", "recommendation", "news"})


def _get_news_sentiment_context(nlp_meta: "NLPMetadataSchema") -> str | None:
    """For stock_price / recommendation / news intents, fetch recent headlines
    and compute a VADER aggregate sentiment score to inject as context.

    Returns a one-line string like "Market Sentiment: Bullish (+0.42)" or None
    if the intent doesn't warrant sentiment, no ticker was found, or VADER
    is unavailable.
    """
    if nlp_meta.intent not in _SENTIMENT_INTENTS:
        return None

    from nlp_pipeline import VADER_AVAILABLE, analyze_sentiment
    if not VADER_AVAILABLE:
        return None

    # Extract the first TICKER entity (Entity objects use attribute access)
    ticker: str | None = None
    for ent in nlp_meta.entities:
        label = ent.label if hasattr(ent, "label") else ent.get("label", "")
        t = ent.ticker if hasattr(ent, "ticker") else ent.get("ticker")
        if label == "TICKER" and t:
            ticker = t
            break
    if not ticker:
        return None

    try:
        import yfinance as yf
        news_items = yf.Ticker(ticker).news or []
        headlines: list[str] = []
        for item in news_items[:5]:
            title = (
                item.get("title")
                or (item.get("content") or {}).get("title", "")
            )
            if title:
                headlines.append(title)
        if not headlines:
            return None

        scores = [analyze_sentiment(h)["compound"] for h in headlines]
        avg = sum(scores) / len(scores)
        if avg >= 0.1:
            label = "Bullish"
        elif avg <= -0.1:
            label = "Bearish"
        else:
            label = "Neutral"
        result = f"Market Sentiment: {label} ({avg:+.2f}) based on {len(headlines)} recent headlines"
        logger.info("VADER sentiment for %s: %s (avg=%.2f)", ticker, label, avg)
        return result
    except Exception:
        import traceback
        print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
        logger.debug("News sentiment fetch failed for %s", ticker, exc_info=True)
        return None


def _append_disclaimer(content: str) -> str:
    """Append the educational disclaimer to a non-empty financial response."""
    if not content.strip():
        return content
    ts = time.strftime("%Y-%m-%d %H:%M UTC")
    disclaimer = f"\n\n*Data as of {ts}. For educational purposes only — not financial advice.*"
    if disclaimer.strip() in content:
        return content
    return content + disclaimer


# ── NLP + analytics shared helper ────────────────────────────────────


def _default_nlp_metadata() -> NLPMetadataSchema:
    """Return safe default NLP metadata when the pipeline fails."""
    return NLPMetadataSchema.model_validate(
        {
            "entities": [],
            "intent": "general",
            "intent_confidence": 0.0,
            "intent_method": "fallback",
            "sentiment": {"label": "unavailable", "score": 0.0, "compound": 0.0},
            "spacy_available": False,
            "vader_available": False,
        }
    )


def _process_nlp_and_track(message: str, session_id: str) -> NLPMetadataSchema:
    """Run the NLP pipeline on *message* and update analytics counters.

    Centralises the NLP-preprocessing + stats-tracking + logging that
    both the streaming and non-streaming chat endpoints need.

    Returns a validated ``NLPMetadataSchema`` Pydantic model built from
    the raw ``NLPMetadata`` TypedDict produced by the pipeline.  If the
    pipeline raises, logs the error and returns safe defaults so that
    chat still works.
    """
    try:
        nlp_meta: NLPMetadata = run_nlp_pipeline(message)
    except Exception:
        import traceback
        print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
        logger.exception("NLP pipeline failed for session %s — using defaults", session_id)
        _stats["total_messages"] += 1
        _stats["intent_counts"]["general"] += 1
        _stats["classifier_usage"]["fallback"] += 1
        return _default_nlp_metadata()

    _stats["total_messages"] += 1
    _stats["intent_counts"][nlp_meta["intent"]] += 1
    _stats["classifier_usage"][nlp_meta.get("intent_method", "keyword")] += 1
    logger.info(
        "NLP [%s] intent=%s (%.0f%% %s) entities=%d sentiment=%s",
        session_id,
        nlp_meta["intent"],
        nlp_meta.get("intent_confidence", 0) * 100,
        nlp_meta.get("intent_method", "keyword"),
        len(nlp_meta["entities"]),
        nlp_meta["sentiment"]["label"],
    )
    return NLPMetadataSchema.model_validate(dict(nlp_meta))


def _record_response_time(elapsed_ms: float) -> None:
    """Append a response time and keep only the most recent 100 entries."""
    times = _stats["response_times_ms"]
    times.append(elapsed_ms)
    if len(times) > 100:
        _stats["response_times_ms"] = times[-100:]


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint — returns API status and NLP pipeline availability."""
    from nlp_pipeline import SPACY_AVAILABLE, VADER_AVAILABLE
    groq_key = os.getenv("GROQ_API_KEY", "")
    return HealthResponse(
        status="ok",
        groq_configured=bool(groq_key),
        active_sessions=len(sessions),
        total_messages=_stats["total_messages"],
        nlp_status=NLPStatusDetail(
            spacy_ner=SPACY_AVAILABLE,
            vader_sentiment=VADER_AVAILABLE,
        ),
    )


@app.get("/api/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Analytics endpoint — returns usage statistics for the current server session."""
    times = _stats["response_times_ms"]
    avg_ms = round(sum(times) / len(times), 1) if times else 0.0
    return StatsResponse(
        total_messages=_stats["total_messages"],
        active_sessions=len(sessions),
        total_sessions_created=_stats["total_sessions_created"],
        intent_distribution=dict(_stats["intent_counts"]),
        avg_response_time_ms=avg_ms,
        classifier_usage=dict(_stats["classifier_usage"]),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """Non-streaming chat endpoint.

    Runs the NLP preprocessing pipeline on the user message, then sends the
    enriched prompt to the financial agent and returns the complete response.
    """
    session_id, history = _get_or_create_session(req.session_id)
    _check_rate_limit(session_id, request)
    logger.info("Chat request [%s]: %.80s", session_id, req.message)

    nlp_meta = _process_nlp_and_track(req.message, session_id)
    intent = nlp_meta.intent
    from financial_agent import LLM_PROVIDER as _provider
    print(f"[REQUEST] intent={intent} provider={_provider}", flush=True)
    sentiment_ctx = _get_news_sentiment_context(nlp_meta)
    full_prompt = _build_prompt(req.message, history, intent=intent, sentiment_context=sentiment_ctx)

    t0 = time.time()
    try:
        content: str = _run_agent(full_prompt)
    except Exception:
        import traceback
        print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
        logger.exception("Agent error for session %s", session_id)
        raise HTTPException(
            status_code=502,
            detail="An internal error occurred while processing your request. Please try again.",
        )
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    _record_response_time(elapsed_ms)

    content = clean_response(content)
    content = _append_disclaimer(content)
    structured = build_structured_response(req.message, content, nlp_meta)
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": content})
    append_message(
        session_id,
        role="user",
        content=req.message,
        nlp_metadata=nlp_meta.model_dump(),
    )
    append_message(
        session_id,
        role="assistant",
        content=content,
        structured_response=structured,
    )
    stored = get_session(session_id)

    logger.info("Chat response [%s]: %d chars in %.0fms", session_id, len(content), elapsed_ms)
    return ChatResponse(
        response=content,
        session_id=session_id,
        session_title=stored["title"] if stored else None,
        nlp_metadata=nlp_meta,
        structured_response=structured,
    )


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Streaming chat endpoint (Server-Sent Events).

    Emits NLP preprocessing metadata as the first SSE event (prefixed with
    [NLP_META]), followed by agent response chunks, ending with [DONE].
    """
    session_id, history = _get_or_create_session(req.session_id)
    _check_rate_limit(session_id, request)
    logger.info("Stream request [%s]: %.80s", session_id, req.message)

    nlp_meta = _process_nlp_and_track(req.message, session_id)
    intent = nlp_meta.intent
    from financial_agent import LLM_PROVIDER as _provider
    print(f"[REQUEST] intent={intent} provider={_provider}", flush=True)
    sentiment_ctx = _get_news_sentiment_context(nlp_meta)
    full_prompt = _build_prompt(req.message, history, intent=intent, sentiment_context=sentiment_ctx)

    def generate() -> Generator[str, None, None]:
        import traceback as _tb
        try:
            yield f"data: [NLP_META]{nlp_meta.model_dump_json()}\n\n"
        except Exception:
            print(f"[ERROR DETAIL] {_tb.format_exc()}", flush=True)
            yield "data: [ERROR] Failed to encode NLP metadata.\n\n"
            return

        collected: list[str] = []
        t0 = time.time()
        had_error = False
        try:
            for text in agent_service.stream(full_prompt):
                if text.startswith("[ERROR]"):
                    had_error = True
                    # Retry once before surfacing the error
                    retry_collected: list[str] = []
                    retry_ok = False
                    try:
                        for chunk in agent_service.stream(full_prompt):
                            if chunk.startswith("[ERROR]"):
                                break
                            retry_collected.append(chunk)
                            yield f"data: {json.dumps(chunk)}\n\n"
                        if retry_collected:
                            retry_ok = True
                            collected = retry_collected
                    except Exception:
                        print(f"[ERROR DETAIL] {_tb.format_exc()}", flush=True)
                    if not retry_ok:
                        friendly = (
                            "[ERROR] Unable to retrieve data right now. "
                            "Please try again in a moment."
                        )
                        yield f"data: {friendly}\n\n"
                    break
                collected.append(text)
                # JSON-encode each chunk so embedded newlines survive SSE framing
                yield f"data: {json.dumps(text)}\n\n"
        except Exception:
            print(f"[ERROR DETAIL] {_tb.format_exc()}", flush=True)
            logger.exception("Unexpected error in stream generator for session %s", session_id)
            yield f"data: [ERROR] Something went wrong. Please try again.\n\n"

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        _record_response_time(elapsed_ms)

        full_response = "".join(collected)
        if full_response:
            full_response = clean_response(full_response)
            full_response = _append_disclaimer(full_response)
            structured = build_structured_response(req.message, full_response, nlp_meta)
            history.append({"role": "user", "content": req.message})
            history.append({"role": "assistant", "content": full_response})
            append_message(
                session_id,
                role="user",
                content=req.message,
                nlp_metadata=nlp_meta.model_dump(),
            )
            append_message(
                session_id,
                role="assistant",
                content=full_response,
                structured_response=structured,
            )
            yield f"data: [RESEARCH_DATA]{json.dumps(structured)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/chat/{session_id}", response_model=SessionClearResponse)
def clear_session(session_id: str) -> SessionClearResponse:
    """Clear conversation history for a session."""
    sessions.pop(session_id, None)
    _rate_limits.pop(session_id, None)
    delete_session(session_id)
    logger.info("Cleared session %s", session_id)
    return SessionClearResponse(status="cleared")


# ── Watchlist endpoints ────────────────────────────────────────────────

@app.post("/api/watchlist", response_model=WatchlistResponse)
def add_to_watchlist(req: WatchlistAddRequest) -> WatchlistResponse:
    """Add a ticker symbol to the session's watchlist."""
    session_id, _ = _get_or_create_session(req.session_id)
    symbol = req.symbol.strip().upper()
    wl: list[str] = sessions[session_id].setdefault("watchlist", [])
    if symbol not in wl:
        wl.append(symbol)
        add_watchlist_symbol(session_id, symbol)
    logger.info("Watchlist [%s]: added %s → %s", session_id, symbol, wl)
    return _build_watchlist_response(session_id, wl)


@app.get("/api/watchlist/{session_id}", response_model=WatchlistResponse)
def get_watchlist(session_id: str) -> WatchlistResponse:
    """Return current prices for all tickers in the session's watchlist."""
    if session_id not in sessions:
        loaded = _load_persisted_session(session_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    wl: list[str] = list_watchlist(session_id)
    sessions[session_id]["watchlist"] = wl
    sessions[session_id]["last_active"] = time.time()
    return _build_watchlist_response(session_id, wl)


@app.delete("/api/watchlist", response_model=WatchlistResponse)
def remove_from_watchlist(req: WatchlistRemoveRequest) -> WatchlistResponse:
    """Remove a ticker symbol from the session's watchlist."""
    if req.session_id not in sessions:
        loaded = _load_persisted_session(req.session_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Session not found.")
    symbol = req.symbol.strip().upper()
    wl: list[str] = sessions[req.session_id].get("watchlist", [])
    if symbol in wl:
        wl.remove(symbol)
    remove_watchlist_symbol(req.session_id, symbol)
    logger.info("Watchlist [%s]: removed %s → %s", req.session_id, symbol, wl)
    return _build_watchlist_response(req.session_id, wl)


def _build_watchlist_response(session_id: str, symbols: list[str]) -> WatchlistResponse:
    """Fetch live prices for watchlist symbols and return a response."""
    items = [WatchlistItem.model_validate(item) for item in build_watchlist_items(symbols)]
    return WatchlistResponse(session_id=session_id, watchlist=items)


# ── Chat export as PDF ────────────────────────────────────────────────

@app.get("/api/sessions", response_model=SessionListResponse)
def get_saved_sessions() -> SessionListResponse:
    """List saved research sessions for the sidebar."""

    rows = list_saved_sessions()
    return SessionListResponse(
        sessions=[
            SessionSummary(
                session_id=row["session_id"],
                title=row["title"],
                preview=row["preview"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]
    )


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
def get_saved_session(session_id: str) -> SessionDetail:
    """Return a full saved research session."""

    payload = get_session(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionDetail.model_validate(payload)


@app.patch("/api/sessions/{session_id}", response_model=SessionSummary)
def rename_saved_session(session_id: str, req: SessionRenameRequest) -> SessionSummary:
    """Rename a saved session."""

    if not rename_session(session_id, req.title):
        raise HTTPException(status_code=404, detail="Session not found.")
    payload = get_session(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    preview = payload["messages"][-1]["content"][:140] if payload["messages"] else ""
    return SessionSummary(
        session_id=session_id,
        title=req.title,
        preview=preview,
        updated_at=payload["updated_at"],
        message_count=len(payload["messages"]),
    )


@app.get("/api/snapshot/{symbol}", response_model=CompanySnapshotResponse)
def company_snapshot(symbol: str) -> CompanySnapshotResponse:
    """Return a dedicated company snapshot page payload."""

    return CompanySnapshotResponse.model_validate(get_market_context(symbol.strip().upper()))


@app.get("/api/compare", response_model=ComparisonResponse)
def compare_symbols(symbols: str = Query(..., description="Comma-separated stock symbols.")) -> ComparisonResponse:
    """Return a structured comparison table for 2-4 symbols."""

    chosen = [part.strip().upper() for part in symbols.split(",") if part.strip()]
    if len(chosen) < 2:
        raise HTTPException(status_code=422, detail="Provide at least two symbols to compare.")
    return ComparisonResponse.model_validate(build_comparison(chosen[:4]))


@app.get("/api/alerts/{session_id}", response_model=AlertListResponse)
def get_session_alerts(session_id: str) -> AlertListResponse:
    """Return all alerts for a given session."""

    return AlertListResponse(
        session_id=session_id,
        alerts=[AlertResponse.model_validate(item) for item in list_alerts(session_id)],
    )


@app.post("/api/alerts", response_model=AlertResponse)
def create_session_alert(req: AlertCreateRequest) -> AlertResponse:
    """Create a simple threshold, digest, or breaking-news alert."""

    alert_type = req.alert_type.strip().lower()
    if alert_type not in {"price_above", "price_below", "daily_digest", "breaking_news"}:
        raise HTTPException(status_code=422, detail="Unsupported alert type.")
    if alert_type.startswith("price_") and req.threshold is None:
        raise HTTPException(status_code=422, detail="Price alerts require a threshold.")
    label = req.label or f"{req.symbol.upper()} {alert_type.replace('_', ' ')}"
    created = create_alert(
        req.session_id,
        req.symbol,
        alert_type,
        threshold=req.threshold,
        label=label,
    )
    return AlertResponse.model_validate(created)


@app.patch("/api/alerts/item/{alert_id}", response_model=AlertResponse)
def update_session_alert(alert_id: int, req: AlertUpdateRequest) -> AlertResponse:
    """Rename or enable/disable an alert."""

    try:
        payload = update_alert(alert_id, enabled=req.enabled, label=req.label)
    except KeyError:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return AlertResponse.model_validate(payload)


@app.delete("/api/alerts/item/{alert_id}", response_model=SessionClearResponse)
def delete_session_alert(alert_id: int) -> SessionClearResponse:
    """Delete one alert by id."""

    delete_alert(alert_id)
    return SessionClearResponse(status="cleared")


@app.get("/api/alerts/{session_id}/check", response_model=AlertCheckResponse)
def check_session_alerts(session_id: str) -> AlertCheckResponse:
    """Evaluate current alerts against live watch data."""

    items: list[AlertCheckItem] = []
    for alert in list_alerts(session_id):
        triggered = False
        reason = None
        latest_price = None
        try:
            context = get_market_context(alert["symbol"])
            latest_price = context["price"]
            if alert["enabled"] and latest_price is not None:
                if alert["alert_type"] == "price_above" and alert["threshold"] is not None:
                    triggered = latest_price >= alert["threshold"]
                    reason = f"Current price ${latest_price:.2f} is above {alert['threshold']:.2f}."
                elif alert["alert_type"] == "price_below" and alert["threshold"] is not None:
                    triggered = latest_price <= alert["threshold"]
                    reason = f"Current price ${latest_price:.2f} is below {alert['threshold']:.2f}."
                elif alert["alert_type"] == "daily_digest":
                    triggered = True
                    reason = "Daily digest is available."
                elif alert["alert_type"] == "breaking_news":
                    triggered = bool(context["headlines"])
                    reason = "Recent headlines are available." if triggered else "No fresh headline signal right now."
            if triggered:
                mark_alert_triggered(alert["id"], last_price=latest_price)
                alert = get_alert(alert["id"])
        except Exception:
            reason = "Could not evaluate this alert right now."
        items.append(
            AlertCheckItem(
                alert=AlertResponse.model_validate(alert),
                triggered=triggered,
                reason=reason,
            )
        )
    return AlertCheckResponse(session_id=session_id, items=items)


@app.post("/api/chat/export")
def export_chat_pdf(req: ExportRequest) -> FileResponse:
    """Export the full conversation as a formatted PDF report.

    Accepts messages directly from the frontend (preferred) or falls back
    to the server-side session history.

    Args:
        req: Export request with session_id and optional messages list.

    Returns:
        FileResponse containing the generated PDF.

    Raises:
        HTTPException: 400 if no messages available, 501 if reportlab missing.
    """
    session_id: str = req.session_id

    # Prefer messages sent from the frontend (works for history chats too)
    history: list[dict[str, str]] | None = None
    if req.messages:
        history = [m.model_dump() for m in req.messages]
    if not history and session_id in sessions:
        history = sessions[session_id]["messages"]
    if not history:
        stored = get_session(session_id)
        if stored:
            history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in stored["messages"]
            ]
    if not history:
        raise HTTPException(status_code=400, detail="No messages to export.")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires reportlab. Install with: pip install reportlab",
        )

    pdf_path = OUTPUTS / f"chat_export_{session_id[:8]}_{int(time.time())}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChatTitle", parent=styles["Title"], fontSize=16,
        textColor=HexColor("#4f46e5"), spaceAfter=12,
    )
    user_style = ParagraphStyle(
        "UserMsg", parent=styles["Normal"], fontSize=10,
        textColor=HexColor("#1e3a5f"), leftIndent=10, spaceAfter=4,
    )
    agent_style = ParagraphStyle(
        "AgentMsg", parent=styles["Normal"], fontSize=10,
        textColor=HexColor("#1a1d2e"), leftIndent=10, spaceAfter=4,
    )
    role_style = ParagraphStyle(
        "RoleLabel", parent=styles["Normal"], fontSize=10,
        textColor=HexColor("#4f46e5"), bold=True, spaceAfter=2,
    )

    elements: list = []
    elements.append(Paragraph("Financial AI Agent — Chat Export", title_style))
    elements.append(Paragraph(
        f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')} | Session: {session_id[:12]}…",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 12))

    for msg in history:
        role = "You" if msg["role"] == "user" else "Agent"
        style = user_style if msg["role"] == "user" else agent_style
        elements.append(Paragraph(f"<b>{role}:</b>", role_style))
        # Escape XML-unsafe chars for reportlab
        safe_content = (
            msg["content"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        # Truncate very long messages
        if len(safe_content) > 3000:
            safe_content = safe_content[:3000] + "… [truncated]"
        elements.append(Paragraph(safe_content, style))

        # Inline chart images if referenced
        chart_matches = re.findall(r'([a-zA-Z0-9_.-]+\.png)', msg["content"])
        for chart_file in chart_matches:
            chart_path = OUTPUTS / "charts" / chart_file
            if chart_path.exists():
                try:
                    elements.append(Spacer(1, 6))
                    elements.append(RLImage(str(chart_path), width=400, height=200))
                except Exception:
                    pass

        elements.append(Spacer(1, 8))

    doc.build(elements)
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"chat_export_{session_id[:8]}.pdf",
    )


# ── Serve generated outputs (charts, sheets) with path traversal protection ──
OUTPUTS = pathlib.Path(__file__).parent / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
(OUTPUTS / "charts").mkdir(parents=True, exist_ok=True)
(OUTPUTS / "sheets").mkdir(parents=True, exist_ok=True)

# Regex for safe filenames: alphanumeric, underscores, hyphens, dots only
_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")


@app.get("/outputs/{subdir}/{filename}")
def serve_output_file(subdir: str, filename: str) -> FileResponse:
    """Serve generated chart/sheet files with path traversal protection.

    Rejects any path containing ``..``, ``/``, ``\\``, or other unsafe chars.
    Only ``charts/`` and ``sheets/`` subdirectories are allowed.
    """
    if subdir not in ("charts", "sheets"):
        raise HTTPException(status_code=404, detail="Not found.")
    if not _SAFE_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    file_path = (OUTPUTS / subdir / filename).resolve()
    # Ensure resolved path is still within OUTPUTS
    if not str(file_path).startswith(str(OUTPUTS.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(file_path))

# ── Production: serve React build from frontend/dist ─────────────────
DIST = pathlib.Path(__file__).parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        """Serve the React SPA for all non-API routes."""
        return FileResponse(str(DIST / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)
