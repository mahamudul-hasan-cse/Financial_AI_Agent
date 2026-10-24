"""Process-local runtime state shared by the API routers.

Everything here lives in memory, which is why the server must run with a
single worker.  SQLite (``persistence.py``) is the durable store; the
``sessions`` dict is a hot cache of recent conversation history.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import TypedDict

from config import settings

# ── Tunables (re-exported by api.py for backward compatibility) ──────
AGENT_TIMEOUT: int = settings.agent_timeout
MAX_SESSIONS: int = settings.max_sessions
SESSION_TTL: int = settings.session_ttl
RATE_LIMIT_WINDOW: int = settings.rate_limit_window
RATE_LIMIT_MAX: int = settings.rate_limit_max
IP_RATE_LIMIT_MAX: int = settings.ip_rate_limit_max
RATE_LIMIT_PER_MINUTE: int = settings.rate_limit_per_minute
API_KEY: str | None = settings.api_key


class SessionData(TypedDict, total=False):
    """Shape of a single session stored in the session dict."""

    messages: list[dict[str, str]]
    last_active: float
    watchlist: list[str]
    client_id: str | None


# Maps session_id -> session data
sessions: dict[str, SessionData] = {}

# Maps session_id -> list of request timestamps (sliding window)
_rate_limits: dict[str, list[float]] = defaultdict(list)
# Maps client IP -> list of request timestamps (per-IP sliding window)
_ip_rate_limits: dict[str, list[float]] = defaultdict(list)
# Maps "client:<X-Client-Id>" (or "ip:<addr>") -> timestamps, for chat endpoints
_client_rate_limits: dict[str, list[float]] = defaultdict(list)
# Guards the three rate-limit stores (sync endpoints run in a thread pool)
rate_limit_lock = threading.Lock()


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


def record_response_time(elapsed_ms: float) -> None:
    """Append a response time and keep only the most recent 100 entries."""
    times = _stats["response_times_ms"]
    times.append(elapsed_ms)
    if len(times) > 100:
        _stats["response_times_ms"] = times[-100:]
