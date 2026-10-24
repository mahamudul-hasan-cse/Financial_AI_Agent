"""Per-session watchlist endpoints (API-only; no bundled UI)."""

from __future__ import annotations

import logging

from fastapi import APIRouter

import persistence
from backend.app.routers.common import ClientId, SessionIdPath
from backend.app.services import sessions as session_service
from market_service import build_watchlist_items
from schemas import (
    WatchlistAddRequest,
    WatchlistItem,
    WatchlistRemoveRequest,
    WatchlistResponse,
)

logger = logging.getLogger("api.watchlist")

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])


def _build_response(session_id: str, symbols: list[str]) -> WatchlistResponse:
    items = [WatchlistItem.model_validate(item) for item in build_watchlist_items(symbols)]
    return WatchlistResponse(session_id=session_id, watchlist=items)


@router.post("", response_model=WatchlistResponse)
def add_to_watchlist(req: WatchlistAddRequest, client_id: ClientId) -> WatchlistResponse:
    """Add a ticker to the session's watchlist (creates the session if new)."""
    session_id, _ = session_service.get_or_create_session(req.session_id, client_id)
    symbol = req.symbol.strip().upper()
    persistence.add_watchlist_symbol(session_id, symbol)
    symbols = persistence.list_watchlist(session_id)
    session_service.sessions[session_id]["watchlist"] = symbols
    logger.info("Watchlist [%s]: added %s", session_id, symbol)
    return _build_response(session_id, symbols)


@router.get("/{session_id}", response_model=WatchlistResponse)
def get_watchlist(session_id: SessionIdPath, client_id: ClientId) -> WatchlistResponse:
    """Return current prices for all tickers in the session's watchlist."""
    data = session_service.require_owned_session(session_id, client_id)
    symbols = persistence.list_watchlist(session_id)
    data["watchlist"] = symbols
    return _build_response(session_id, symbols)


@router.delete("", response_model=WatchlistResponse)
def remove_from_watchlist(req: WatchlistRemoveRequest, client_id: ClientId) -> WatchlistResponse:
    """Remove a ticker from the session's watchlist."""
    data = session_service.require_owned_session(req.session_id, client_id)
    symbol = req.symbol.strip().upper()
    persistence.remove_watchlist_symbol(req.session_id, symbol)
    symbols = persistence.list_watchlist(req.session_id)
    data["watchlist"] = symbols
    logger.info("Watchlist [%s]: removed %s", req.session_id, symbol)
    return _build_response(req.session_id, symbols)
