"""Financial AI Agent API — entry-point.

The application lives in ``backend/app`` (routers, services, middleware);
this module keeps the historical ``uvicorn api:app`` entry-point and
re-exports the runtime state that tests and scripts inspect.

Run locally::

    python api.py            # honours HOST, PORT, FORWARDED_ALLOW_IPS
"""

from __future__ import annotations

from backend.app.main import app  # noqa: F401
from backend.app.state import (  # noqa: F401
    AGENT_TIMEOUT,
    API_KEY,
    IP_RATE_LIMIT_MAX,
    MAX_SESSIONS,
    RATE_LIMIT_MAX,
    RATE_LIMIT_WINDOW,
    SESSION_TTL,
    _ip_rate_limits,
    _rate_limits,
    _stats,
    sessions,
)
from config import settings
from schemas import MAX_MESSAGE_LENGTH  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        proxy_headers=True,
        forwarded_allow_ips=settings.forwarded_allow_ips,
        log_level=settings.log_level.lower(),
    )
