"""Market data endpoints: company snapshot and multi-symbol comparison."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Query

from market_service import build_comparison, get_market_context
from schemas import CompanySnapshotResponse, ComparisonResponse

logger = logging.getLogger("api.market")

router = APIRouter(prefix="/api", tags=["market"])

_SYMBOL_RE = re.compile(r"^[A-Z0-9.^=-]{1,10}$")


def _symbol(raw: str) -> str:
    symbol = raw.strip().upper()
    if not _SYMBOL_RE.fullmatch(symbol):
        raise HTTPException(status_code=422, detail=f"Invalid ticker symbol: {raw[:12]!r}")
    return symbol


@router.get("/snapshot/{symbol}", response_model=CompanySnapshotResponse)
def company_snapshot(symbol: str) -> CompanySnapshotResponse:
    """Company snapshot: price, change, fundamentals, trend and headlines."""
    ticker = _symbol(symbol)
    try:
        return CompanySnapshotResponse.model_validate(get_market_context(ticker))
    except Exception as exc:
        logger.warning("Snapshot failed for %s", ticker, exc_info=True)
        raise HTTPException(
            status_code=502, detail=f"Market data unavailable for {ticker}."
        ) from exc


@router.get("/compare", response_model=ComparisonResponse)
def compare_symbols(
    symbols: str = Query(..., max_length=100, description="Comma-separated stock symbols."),
) -> ComparisonResponse:
    """Structured comparison table for 2-4 symbols."""
    chosen = [_symbol(part) for part in symbols.split(",") if part.strip()]
    if len(chosen) < 2:
        raise HTTPException(status_code=422, detail="Provide at least two symbols to compare.")
    try:
        return ComparisonResponse.model_validate(build_comparison(chosen[:4]))
    except Exception as exc:
        logger.warning("Comparison failed for %s", chosen, exc_info=True)
        raise HTTPException(status_code=502, detail="Market data unavailable right now.") from exc
