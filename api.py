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
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from financial_agent import agent_service
from nlp_pipeline import run_nlp_pipeline, NLPMetadata
from schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ExportRequest,
    HealthResponse,
    NLPMetadataSchema,
    NLPStatusDetail,
    SessionClearResponse,
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
AGENT_TIMEOUT: int = 120  # seconds
# MAX_MESSAGE_LENGTH is imported from schemas
MAX_SESSIONS: int = 100
SESSION_TTL: int = 1800  # 30 minutes in seconds
RATE_LIMIT_WINDOW: int = 60  # seconds
RATE_LIMIT_MAX: int = 10  # requests per window
API_KEY: str | None = os.getenv("API_KEY")  # None = auth disabled
IP_RATE_LIMIT_MAX: int = 30  # requests per window per IP

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Financial AI Agent API")

cors_origins_raw: str = os.getenv("CORS_ORIGINS", "http://localhost:5173")
cors_origins: list[str] = [o.strip() for o in cors_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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
        client_ip = request.client.host if request.client else "unknown"
        t0 = time.time()
        response = await call_next(request)
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            "%s %s %d %.1fms client=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            client_ip,
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
def _cleanup_expired_sessions() -> int:
    """Remove sessions that have been inactive for longer than SESSION_TTL.

    Returns the number of sessions removed.
    """
    now = time.time()
    expired = [
        sid for sid, data in sessions.items()
        if now - data["last_active"] > SESSION_TTL
    ]
    for sid in expired:
        del sessions[sid]
        _rate_limits.pop(sid, None)
    if expired:
        logger.info("Cleaned up %d expired sessions", len(expired))
    return len(expired)


def _get_or_create_session(session_id: str | None) -> tuple[str, list[dict[str, str]]]:
    """Retrieve an existing session or create a new one.

    Runs session cleanup and enforces MAX_SESSIONS limit.
    Returns (session_id, message_history).
    """
    _cleanup_expired_sessions()

    if session_id and session_id in sessions:
        sessions[session_id]["last_active"] = time.time()
        return session_id, sessions[session_id]["messages"]

    # Enforce max sessions limit
    if len(sessions) >= MAX_SESSIONS:
        # Evict the oldest session
        oldest_sid = min(sessions, key=lambda s: sessions[s]["last_active"])
        del sessions[oldest_sid]
        _rate_limits.pop(oldest_sid, None)
        logger.info("Evicted oldest session %s (max sessions reached)", oldest_sid)

    new_id = session_id or str(uuid.uuid4())
    sessions[new_id] = {"messages": [], "last_active": time.time(), "watchlist": []}
    _stats["total_sessions_created"] += 1
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


def _build_prompt(user_message: str, history: list[dict[str, str]]) -> str:
    """Construct the full prompt including conversation history."""
    context = _build_context(history)
    if context:
        return (
            f"Previous conversation:\n{context}\n\n"
            f"User's new message: {user_message}\n\n"
            "Respond to the user's new message, taking the conversation history into account."
        )
    return user_message


def _run_agent(prompt: str) -> str:
    """Execute the agent synchronously and return the response text.

    Delegates to ``agent_service.run()`` which handles retries internally.
    """
    return agent_service.run(prompt)


# ── NLP + analytics shared helper ────────────────────────────────────


def _process_nlp_and_track(message: str, session_id: str) -> NLPMetadataSchema:
    """Run the NLP pipeline on *message* and update analytics counters.

    Centralises the NLP-preprocessing + stats-tracking + logging that
    both the streaming and non-streaming chat endpoints need.

    Returns a validated ``NLPMetadataSchema`` Pydantic model built from
    the raw ``NLPMetadata`` TypedDict produced by the pipeline.
    """
    nlp_meta: NLPMetadata = run_nlp_pipeline(message)
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
    full_prompt = _build_prompt(req.message, history)

    t0 = time.time()
    try:
        content: str = _run_agent(full_prompt)
    except Exception:
        logger.exception("Agent error for session %s", session_id)
        raise HTTPException(
            status_code=502,
            detail="An internal error occurred while processing your request.",
        )
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    _record_response_time(elapsed_ms)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": content})

    logger.info("Chat response [%s]: %d chars in %.0fms", session_id, len(content), elapsed_ms)
    return ChatResponse(response=content, session_id=session_id, nlp_metadata=nlp_meta)


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
    full_prompt = _build_prompt(req.message, history)

    def generate() -> Generator[str, None, None]:
        # Emit NLP metadata as the first event so the frontend can display it
        yield f"data: [NLP_META]{nlp_meta.model_dump_json()}\n\n"

        collected: list[str] = []
        t0 = time.time()
        for text in agent_service.stream(full_prompt):
            if text.startswith("[ERROR]"):
                yield f"data: {text}\n\n"
                break
            collected.append(text)
            # JSON-encode each chunk so embedded newlines survive SSE framing
            yield f"data: {json.dumps(text)}\n\n"

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        _record_response_time(elapsed_ms)

        full_response = "".join(collected)
        if full_response:
            history.append({"role": "user", "content": req.message})
            history.append({"role": "assistant", "content": full_response})

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/chat/{session_id}", response_model=SessionClearResponse)
def clear_session(session_id: str) -> SessionClearResponse:
    """Clear conversation history for a session."""
    sessions.pop(session_id, None)
    _rate_limits.pop(session_id, None)
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
    logger.info("Watchlist [%s]: added %s → %s", session_id, symbol, wl)
    return _build_watchlist_response(session_id, wl)


@app.get("/api/watchlist/{session_id}", response_model=WatchlistResponse)
def get_watchlist(session_id: str) -> WatchlistResponse:
    """Return current prices for all tickers in the session's watchlist."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    wl: list[str] = sessions[session_id].get("watchlist", [])
    sessions[session_id]["last_active"] = time.time()
    return _build_watchlist_response(session_id, wl)


@app.delete("/api/watchlist", response_model=WatchlistResponse)
def remove_from_watchlist(req: WatchlistRemoveRequest) -> WatchlistResponse:
    """Remove a ticker symbol from the session's watchlist."""
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    symbol = req.symbol.strip().upper()
    wl: list[str] = sessions[req.session_id].get("watchlist", [])
    if symbol in wl:
        wl.remove(symbol)
    logger.info("Watchlist [%s]: removed %s → %s", req.session_id, symbol, wl)
    return _build_watchlist_response(req.session_id, wl)


def _build_watchlist_response(session_id: str, symbols: list[str]) -> WatchlistResponse:
    """Fetch live prices for watchlist symbols and return a response."""
    import cache

    items: list[WatchlistItem] = []
    for sym in symbols:
        try:
            info = cache.cached_fast_info(sym)
            price = getattr(info, "last_price", None) or getattr(info, "lastPrice", None)
            change = getattr(info, "regular_market_previous_close", None)
            pct = None
            if price and change and change > 0:
                pct = round(((price - change) / change) * 100, 2)
            items.append(WatchlistItem(
                symbol=sym,
                price=round(float(price), 2) if price else None,
                change_pct=pct,
            ))
        except Exception:
            items.append(WatchlistItem(symbol=sym, price=None, change_pct=None))
    return WatchlistResponse(session_id=session_id, watchlist=items)


# ── Chat export as PDF ────────────────────────────────────────────────

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
        import re as _re
        chart_matches = _re.findall(r'([a-zA-Z0-9_.-]+\.png)', msg["content"])
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

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
