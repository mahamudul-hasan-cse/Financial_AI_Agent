"""Static files: generated outputs and the production React build."""

from __future__ import annotations

import re

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import settings

router = APIRouter(tags=["files"])

OUTPUTS = settings.outputs_dir
for _sub in ("charts", "sheets"):
    (OUTPUTS / _sub).mkdir(parents=True, exist_ok=True)

DIST = settings.app_dir / "frontend" / "dist"

# Safe filenames: alphanumeric, underscores, hyphens, dots only (no "..").
_SAFE_FILENAME_RE = re.compile(r"^(?!.*\.\.)[A-Za-z0-9_.-]{1,200}$")


@router.get("/outputs/{subdir}/{filename}")
def serve_output_file(subdir: str, filename: str) -> FileResponse:
    """Serve generated chart/sheet files with path traversal protection."""
    if subdir not in ("charts", "sheets"):
        raise HTTPException(status_code=404, detail="Not found.")
    if not _SAFE_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    base = (OUTPUTS / subdir).resolve()
    file_path = (base / filename).resolve()
    if file_path.parent != base:
        raise HTTPException(status_code=400, detail="Invalid path.")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(file_path))


_NON_SPA_PREFIXES = ("/api/", "/outputs/", "/docs", "/redoc", "/openapi.json")


def install_spa(app: FastAPI) -> None:
    """Serve the React build for unknown non-API GET paths.

    Implemented as a 404 handler (not a catch-all route) so that unknown
    ``/api/*`` paths keep FastAPI's JSON 404 and wrong methods keep 405.
    """
    index = DIST / "index.html"
    if (DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.exception_handler(StarletteHTTPException)
    async def spa_or_http_error(request: Request, exc: StarletteHTTPException) -> Response:
        path = request.url.path
        if (
            exc.status_code == 404
            and request.method in ("GET", "HEAD")
            and index.is_file()
            and path != "/api"
            and not path.startswith(_NON_SPA_PREFIXES)
        ):
            candidate = (DIST / path.lstrip("/")).resolve()
            if candidate.parent == DIST.resolve() and candidate.is_file():
                return FileResponse(str(candidate))  # favicon.svg etc. at the dist root
            return FileResponse(str(index))
        return await http_exception_handler(request, exc)
