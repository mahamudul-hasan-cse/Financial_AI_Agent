"""HTTP middleware: CORS, gzip, request logging and optional API-key auth."""

from __future__ import annotations

import hmac
import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware

from backend.app import state
from backend.app.security import client_ip
from config import settings

logger = logging.getLogger("api")

# Paths that never require authentication
_AUTH_EXEMPT_PATHS: set[str] = {"/api/health"}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, response time, and client IP.

    Never logs request or response bodies to avoid exposing sensitive data.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        t0 = time.time()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s %d %.1fms client=%s request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            (time.time() - t0) * 1000,
            client_ip(request),
            request_id,
        )
        return response


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Require the ``X-API-Key`` header on ``/api/`` endpoints except health.

    Disabled when the ``API_KEY`` env-var is unset (dev mode).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        api_key = state.API_KEY
        if (
            api_key
            and request.method != "OPTIONS"
            and request.url.path.startswith("/api/")
            and request.url.path not in _AUTH_EXEMPT_PATHS
        ):
            provided = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(provided.encode(), api_key.encode()):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )
        return await call_next(request)


def install_middleware(app: FastAPI) -> None:
    """Register all middleware on *app* (outermost last)."""
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(APIKeyMiddleware)
    # CORS outermost so even 401/429 responses carry CORS headers.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Client-Id", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
