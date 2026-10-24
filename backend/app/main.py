"""FastAPI application factory.

``api.py`` at the repository root re-exports :data:`app` so the
``uvicorn api:app`` entry-point keeps working.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.app.middleware import install_middleware
from backend.app.routers import alerts, chat, export, files, health, market, sessions, watchlist
from config import settings

logger = logging.getLogger("api")


def configure_logging() -> None:
    """Configure root logging once (uvicorn may already have handlers)."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Financial AI Agent API",
        description="NLP-enriched financial research assistant (Groq/Ollama + agno).",
        version="1.1.0",
    )
    install_middleware(app)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log the traceback and return a generic JSON error (no stack traces leak)."""
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred. Please try again later."},
        )

    for module in (health, chat, export, sessions, watchlist, alerts, market, files):
        app.include_router(module.router)
    files.install_spa(app)
    return app


app = create_app()
