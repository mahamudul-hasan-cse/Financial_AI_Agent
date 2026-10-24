"""Session lifecycle, ownership checks and rate limiting."""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Any

from fastapi import HTTPException, Request

import persistence
from backend.app import state
from backend.app.security import client_ip
from backend.app.state import SessionData, _ip_rate_limits, _rate_limits, _stats, sessions

logger = logging.getLogger("api.sessions")

SESSION_NOT_FOUND = "Session not found."


def _not_found() -> HTTPException:
    # 404 (not 403) for sessions owned by someone else, so ids can't be probed.
    return HTTPException(status_code=404, detail=SESSION_NOT_FOUND)


def _forget(session_id: str) -> None:
    sessions.pop(session_id, None)
    _rate_limits.pop(session_id, None)


def cleanup_expired_sessions() -> int:
    """Remove sessions inactive for longer than SESSION_TTL and enforce MAX_SESSIONS.

    Returns the number of sessions removed.
    """
    removed = set(persistence.prune_old_sessions(state.MAX_SESSIONS, state.SESSION_TTL))
    now = time.time()
    removed.update(
        sid for sid, data in sessions.items() if now - data["last_active"] > state.SESSION_TTL
    )
    for sid in removed:
        _forget(sid)

    if len(sessions) >= state.MAX_SESSIONS:
        overflow = len(sessions) - state.MAX_SESSIONS + 1
        oldest = sorted(sessions.items(), key=lambda item: item[1]["last_active"])[:overflow]
        for sid, _ in oldest:
            removed.add(sid)
            _forget(sid)
            persistence.delete_session(sid)

    if removed:
        logger.info("Cleaned up %d expired/purged sessions", len(removed))
    return len(removed)


def _load_persisted_session(session_id: str) -> SessionData | None:
    """Load a session from SQLite into the in-memory cache."""
    stored = persistence.get_session(session_id)
    if stored is None:
        return None
    payload: SessionData = {
        "messages": [
            {"role": message["role"], "content": message["content"]}
            for message in stored["messages"]
        ],
        "last_active": time.time(),
        "watchlist": persistence.list_watchlist(session_id),
        "client_id": stored.get("client_id"),
    }
    sessions[session_id] = payload
    return payload


def _owner_of(session_id: str) -> tuple[bool, str | None]:
    cached = sessions.get(session_id)
    if cached is not None and "client_id" in cached:
        return True, cached["client_id"]
    return persistence.get_session_owner(session_id)


def require_owned_session(session_id: str, client_id: str) -> SessionData:
    """Return the cached session if it exists and belongs to *client_id*, else 404."""
    exists, owner = _owner_of(session_id)
    if not exists or owner != client_id:
        raise _not_found()
    data = sessions.get(session_id) or _load_persisted_session(session_id)
    if data is None:
        raise _not_found()
    data["last_active"] = time.time()
    return data


def assert_not_foreign(session_id: str, client_id: str) -> None:
    """404 if *session_id* exists but belongs to another client (missing is fine)."""
    exists, owner = _owner_of(session_id)
    if exists and owner != client_id:
        raise _not_found()


def get_or_create_session(
    session_id: str | None, client_id: str
) -> tuple[str, list[dict[str, str]]]:
    """Retrieve an owned session or create a new one for *client_id*.

    Returns ``(session_id, message_history)``.  Raises 404 when the id
    belongs to a different client.
    """
    cleanup_expired_sessions()

    if session_id:
        exists, owner = _owner_of(session_id)
        if exists:
            if owner != client_id:
                raise _not_found()
            data = sessions.get(session_id) or _load_persisted_session(session_id)
            if data is not None:
                data["last_active"] = time.time()
                persistence.upsert_session(session_id)
                return session_id, data["messages"]
        elif session_id in sessions:
            # Cached without ownership info (legacy/test fixtures) — adopt it.
            data = sessions[session_id]
            data["last_active"] = time.time()
            data["client_id"] = client_id
            persistence.upsert_session(session_id, client_id=client_id)
            return session_id, data["messages"]

    new_id = session_id or str(uuid.uuid4())
    persistence.upsert_session(new_id, client_id=client_id)
    sessions[new_id] = {
        "messages": [],
        "last_active": time.time(),
        "watchlist": [],
        "client_id": client_id,
    }
    _stats["total_sessions_created"] += 1
    return new_id, sessions[new_id]["messages"]


def delete_owned_session(session_id: str, client_id: str) -> None:
    """Delete a session if owned by *client_id* (idempotent for unknown ids)."""
    assert_not_foreign(session_id, client_id)
    _forget(session_id)
    persistence.delete_session(session_id)


def list_client_sessions(client_id: str) -> list[dict[str, Any]]:
    return persistence.list_sessions(client_id)


# ── Rate limiting ────────────────────────────────────────────────────

def _sliding_window_check(
    store: dict[str, list[float]], key: str, max_requests: int
) -> float | None:
    """Return ``None`` if allowed, else seconds until the next slot opens."""
    now = time.time()
    store[key] = [t for t in store[key] if now - t < state.RATE_LIMIT_WINDOW]
    if len(store[key]) >= max_requests:
        oldest = store[key][0]
        return math.ceil(state.RATE_LIMIT_WINDOW - (now - oldest))
    store[key].append(now)
    return None


def check_rate_limit(session_id: str, request: Request | None = None) -> None:
    """Enforce per-session AND per-IP sliding-window rate limiting (429 on breach)."""
    retry_after = _sliding_window_check(_rate_limits, session_id, state.RATE_LIMIT_MAX)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {state.RATE_LIMIT_MAX} requests per minute.",
            headers={"Retry-After": str(retry_after)},
        )

    if request is not None:
        retry_after = _sliding_window_check(
            _ip_rate_limits, client_ip(request), state.IP_RATE_LIMIT_MAX
        )
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"IP rate limit exceeded. Maximum {state.IP_RATE_LIMIT_MAX} "
                    "requests per minute."
                ),
                headers={"Retry-After": str(retry_after)},
            )
