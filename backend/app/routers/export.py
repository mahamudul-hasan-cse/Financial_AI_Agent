"""Chat transcript export as PDF."""

from __future__ import annotations

import logging
import shutil

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

import persistence
from backend.app.routers.common import ClientId
from backend.app.services import export as export_service
from backend.app.services import sessions as session_service
from schemas import ExportRequest

logger = logging.getLogger("api.export")

router = APIRouter(prefix="/api/chat", tags=["export"])


@router.post("/export")
def export_chat_pdf(req: ExportRequest, client_id: ClientId) -> FileResponse:
    """Export a conversation as a PDF.

    Uses the messages sent by the client when present, otherwise the
    caller's stored session.  400 if there is nothing to export, 404 if the
    session belongs to someone else, 501 if reportlab is not installed.
    """
    session_service.assert_not_foreign(req.session_id, client_id)

    history: list[dict[str, str]] | None = None
    if req.messages:
        history = [m.model_dump() for m in req.messages]
    else:
        stored = persistence.get_session(req.session_id)
        if stored:
            history = [{"role": m["role"], "content": m["content"]} for m in stored["messages"]]
    if not history:
        raise HTTPException(status_code=400, detail="No messages to export.")

    try:
        pdf_path = export_service.render_chat_pdf(history, req.session_id)
    except export_service.ReportlabMissingError as exc:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires reportlab. Install with: pip install reportlab",
        ) from exc

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=export_service.download_filename(req.session_id),
        background=BackgroundTask(shutil.rmtree, pdf_path.parent, ignore_errors=True),
    )
