"""Saved research sessions (sidebar) — scoped to the caller's client id."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

import persistence
from backend.app.routers.common import ClientId, SessionIdPath
from backend.app.services import sessions as session_service
from schemas import (
    SessionDetail,
    SessionListResponse,
    SessionRenameRequest,
    SessionSummary,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
def list_sessions(client_id: ClientId) -> SessionListResponse:
    """List the caller's saved sessions, most recent first."""
    rows = session_service.list_client_sessions(client_id)
    return SessionListResponse(sessions=[SessionSummary.model_validate(row) for row in rows])


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: SessionIdPath, client_id: ClientId) -> SessionDetail:
    """Return one of the caller's sessions with all messages."""
    session_service.require_owned_session(session_id, client_id)
    payload = persistence.get_session(session_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=session_service.SESSION_NOT_FOUND)
    return SessionDetail.model_validate(payload)


@router.patch("/{session_id}", response_model=SessionSummary)
def rename_session(
    session_id: SessionIdPath, req: SessionRenameRequest, client_id: ClientId
) -> SessionSummary:
    """Rename one of the caller's sessions."""
    session_service.require_owned_session(session_id, client_id)
    if not persistence.rename_session(session_id, req.title):
        raise HTTPException(status_code=404, detail=session_service.SESSION_NOT_FOUND)
    summary = persistence.get_session_summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=session_service.SESSION_NOT_FOUND)
    return SessionSummary.model_validate(summary)
