"""Request identity helpers: session-id / client-id validation and client IP.

Sessions are owned by an *anonymous client id* — a random token the web UI
generates once per browser and sends as the ``X-Client-Id`` header.  It is
not authentication (anyone holding the token can act as that browser), but
it stops users from listing or reading each other's sessions.
"""

from __future__ import annotations

import re

from fastapi import Header, HTTPException, Request

from schemas.common import SESSION_ID_PATTERN

# SESSION_ID_PATTERN (schemas.common): letters, digits, "_" and "-" only.
CLIENT_ID_PATTERN = r"^[A-Za-z0-9_-]{16,128}$"

_SESSION_ID_RE = re.compile(SESSION_ID_PATTERN)
_CLIENT_ID_RE = re.compile(CLIENT_ID_PATTERN)


def is_valid_session_id(value: str | None) -> bool:
    return bool(value) and bool(_SESSION_ID_RE.fullmatch(value))


def get_client_id(
    x_client_id: str | None = Header(
        default=None,
        alias="X-Client-Id",
        description="Anonymous per-browser id that owns the caller's sessions.",
    ),
) -> str:
    """FastAPI dependency returning the validated ``X-Client-Id`` header."""
    if not x_client_id or not _CLIENT_ID_RE.fullmatch(x_client_id):
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid X-Client-Id header (16-128 chars of A-Z, a-z, 0-9, _ or -).",
        )
    return x_client_id


def client_ip(request: Request) -> str:
    """Best-effort client IP.

    Behind a reverse proxy, run uvicorn with ``--proxy-headers`` and
    ``--forwarded-allow-ips`` (``FORWARDED_ALLOW_IPS``) so ``request.client``
    already reflects ``X-Forwarded-For`` from trusted proxies only.
    """
    return request.client.host if request.client else "unknown"
