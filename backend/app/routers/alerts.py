"""Per-session price/news alerts (API-only; no bundled UI)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, HTTPException

import persistence
from backend.app.routers.common import ClientId, SessionIdPath
from backend.app.services import sessions as session_service
from market_service import get_market_context
from schemas import (
    AlertCheckItem,
    AlertCheckResponse,
    AlertCreateRequest,
    AlertListResponse,
    AlertResponse,
    AlertUpdateRequest,
    SessionClearResponse,
)

logger = logging.getLogger("api.alerts")

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

_ALERT_TYPES = {"price_above", "price_below", "daily_digest", "breaking_news"}


def _owned_alert(alert_id: int, client_id: str) -> dict[str, Any]:
    """Return the alert if its session belongs to *client_id*, else 404."""
    try:
        alert = persistence.get_alert(alert_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Alert not found.") from None
    exists, owner = persistence.get_session_owner(alert["session_id"])
    if not exists or owner != client_id:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return alert


@router.get("/{session_id}", response_model=AlertListResponse)
def list_alerts(session_id: SessionIdPath, client_id: ClientId) -> AlertListResponse:
    """Return all alerts for one of the caller's sessions."""
    session_service.require_owned_session(session_id, client_id)
    return AlertListResponse(
        session_id=session_id,
        alerts=[AlertResponse.model_validate(item) for item in persistence.list_alerts(session_id)],
    )


@router.post("", response_model=AlertResponse)
def create_alert(req: AlertCreateRequest, client_id: ClientId) -> AlertResponse:
    """Create a threshold, digest, or breaking-news alert."""
    alert_type = req.alert_type.strip().lower()
    if alert_type not in _ALERT_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported alert type.")
    if alert_type.startswith("price_") and req.threshold is None:
        raise HTTPException(status_code=422, detail="Price alerts require a threshold.")
    session_id, _ = session_service.get_or_create_session(req.session_id, client_id)
    label = req.label or f"{req.symbol.upper()} {alert_type.replace('_', ' ')}"
    created = persistence.create_alert(
        session_id, req.symbol, alert_type, threshold=req.threshold, label=label
    )
    return AlertResponse.model_validate(created)


@router.patch("/item/{alert_id}", response_model=AlertResponse)
def update_alert(alert_id: int, req: AlertUpdateRequest, client_id: ClientId) -> AlertResponse:
    """Rename or enable/disable one of the caller's alerts."""
    _owned_alert(alert_id, client_id)
    payload = persistence.update_alert(alert_id, enabled=req.enabled, label=req.label)
    return AlertResponse.model_validate(payload)


@router.delete("/item/{alert_id}", response_model=SessionClearResponse)
def delete_alert(alert_id: int, client_id: ClientId) -> SessionClearResponse:
    """Delete one of the caller's alerts."""
    _owned_alert(alert_id, client_id)
    persistence.delete_alert(alert_id)
    return SessionClearResponse(status="cleared")


def _evaluate(alert: dict[str, Any]) -> AlertCheckItem:
    triggered = False
    reason = None
    try:
        context = get_market_context(alert["symbol"])
        latest_price = context["price"]
        if alert["enabled"] and latest_price is not None:
            threshold = alert["threshold"]
            if alert["alert_type"] == "price_above" and threshold is not None:
                triggered = latest_price >= threshold
                reason = f"Current price ${latest_price:.2f} vs threshold {threshold:.2f}."
            elif alert["alert_type"] == "price_below" and threshold is not None:
                triggered = latest_price <= threshold
                reason = f"Current price ${latest_price:.2f} vs threshold {threshold:.2f}."
            elif alert["alert_type"] == "daily_digest":
                triggered = True
                reason = "Daily digest is available."
            elif alert["alert_type"] == "breaking_news":
                triggered = bool(context["headlines"])
                reason = (
                    "Recent headlines are available."
                    if triggered
                    else "No fresh headline signal right now."
                )
        if triggered:
            persistence.mark_alert_triggered(alert["id"], last_price=latest_price)
            alert = persistence.get_alert(alert["id"])
    except Exception:
        logger.warning("Could not evaluate alert %s", alert.get("id"), exc_info=True)
        reason = "Could not evaluate this alert right now."
    return AlertCheckItem(alert=AlertResponse.model_validate(alert), triggered=triggered, reason=reason)


@router.get("/{session_id}/check", response_model=AlertCheckResponse)
def check_alerts(session_id: SessionIdPath, client_id: ClientId) -> AlertCheckResponse:
    """Evaluate the session's alerts against live market data (symbols fetched in parallel)."""
    session_service.require_owned_session(session_id, client_id)
    alerts = persistence.list_alerts(session_id)
    if not alerts:
        return AlertCheckResponse(session_id=session_id, items=[])
    with ThreadPoolExecutor(max_workers=min(8, len(alerts))) as pool:
        items = list(pool.map(_evaluate, alerts))
    return AlertCheckResponse(session_id=session_id, items=items)
