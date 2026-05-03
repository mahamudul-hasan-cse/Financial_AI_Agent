"""Watchlist request / response schemas.

Defines the Pydantic models for the portfolio watchlist feature:
add/remove tickers, and return live prices for watched symbols.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WatchlistAddRequest(BaseModel):
    """Request body for ``POST /api/watchlist``."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol to add (e.g. AAPL, GOOGL).",
    )
    session_id: str = Field(..., description="Session ID that owns the watchlist.")


class WatchlistRemoveRequest(BaseModel):
    """Request body for ``DELETE /api/watchlist``."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol to remove.",
    )
    session_id: str = Field(..., description="Session ID that owns the watchlist.")


class WatchlistItem(BaseModel):
    """A single ticker in the watchlist with its current price."""

    symbol: str
    price: float | None = Field(default=None, description="Current price (None if fetch failed).")
    change_pct: float | None = Field(default=None, description="Percentage change today.")
    signal: str | None = Field(
        default=None,
        description="Short intelligence signal for the watchlist card.",
    )


class WatchlistResponse(BaseModel):
    """Response body for ``GET /api/watchlist/{session_id}``."""

    session_id: str
    watchlist: list[WatchlistItem] = Field(default_factory=list)
