"""Compact, cached Yahoo Finance tools for the agent.

Drop-in replacement for agno's ``YFinanceTools`` with the same function
names, tuned for latency:

* Every result is cached per ticker + function (``cache.get_or_set``, TTL
  ``CACHE_TTL_SECONDS``), and price/fundamentals share one ``Ticker.info``.
* Outputs are short single-line JSON instead of pretty-printed dumps: news is
  reduced to title/publisher/date, analyst recommendations to the latest
  month's counts plus the price target.  Fewer tokens per tool result means
  faster follow-up model calls.
* ``get_stock_fundamentals`` already includes the current price, so one call
  answers most stock questions.

Errors are returned as short strings (and never cached) so the model can
explain the failure instead of the run crashing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable

os.environ.setdefault("AGNO_TELEMETRY", "false")

from agno.tools import Toolkit  # noqa: E402

import cache  # noqa: E402

logger = logging.getLogger("financial_agent.tools")

MAX_NEWS_STORIES = 5
_MAX_TITLE_CHARS = 160


def _num(value: Any) -> float | None:
    try:
        return None if value in (None, "") else round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _compact(data: dict[str, Any]) -> str:
    """Single-line JSON without null fields."""
    return json.dumps({k: v for k, v in data.items() if v not in (None, "")}, separators=(",", ":"))


def _clean_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


class MarketDataTools(Toolkit):
    """Price, fundamentals, analyst recommendations and news for a ticker."""

    def __init__(self) -> None:
        super().__init__(name="market_data_tools")
        self.register(self.get_current_stock_price)
        self.register(self.get_stock_fundamentals)
        self.register(self.get_analyst_recommendations)
        self.register(self.get_company_news)

    @staticmethod
    def _cached(function: str, symbol: str, build: Callable[[str], str], *args: Any) -> str:
        sym = _clean_symbol(symbol)
        try:
            return cache.get_or_set(
                cache.make_key(f"tool:{function}", sym, *args), lambda: build(sym)
            )
        except Exception as exc:
            logger.warning("%s(%s) failed: %s", function, sym, exc)
            return f"Error: could not fetch {function.replace('_', ' ')} for {sym} ({type(exc).__name__})."

    # ── Tools ────────────────────────────────────────────────────────

    def get_current_stock_price(self, symbol: str) -> str:
        """Get the latest price and daily change for a stock symbol.

        Args:
            symbol: Ticker symbol, e.g. AAPL.
        """
        return self._cached("current_price", symbol, _price)

    def get_stock_fundamentals(self, symbol: str) -> str:
        """Get price, daily change, valuation (P/E, P/B, market cap), EPS, dividend yield,
        beta, margins, revenue growth and 52-week range for a stock. Includes the current
        price, so a separate price lookup is not needed.

        Args:
            symbol: Ticker symbol, e.g. AAPL.
        """
        return self._cached("fundamentals", symbol, _fundamentals)

    def get_analyst_recommendations(self, symbol: str) -> str:
        """Get the latest analyst rating counts (strong buy … strong sell) and price targets.

        Args:
            symbol: Ticker symbol, e.g. AAPL.
        """
        return self._cached("analyst_recommendations", symbol, _recommendations)

    def get_company_news(self, symbol: str, num_stories: int = 3) -> str:
        """Get recent news headlines (title, publisher, date) for a stock.

        Args:
            symbol: Ticker symbol, e.g. AAPL.
            num_stories: Number of headlines (1-5, default 3).
        """
        try:
            n = max(1, min(int(num_stories), MAX_NEWS_STORIES))
        except (TypeError, ValueError):
            n = 3
        return self._cached("company_news", symbol, lambda sym: _news(sym, n), n)


# ── Builders (run on cache miss; may raise) ──────────────────────────


def _price_fields(info: dict[str, Any]) -> dict[str, Any]:
    price = _num(info.get("regularMarketPrice")) or _num(info.get("currentPrice"))
    if price is None:
        raise LookupError("no price data")
    prev = _num(info.get("regularMarketPreviousClose")) or _num(info.get("previousClose"))
    change_pct = round((price - prev) / prev * 100, 2) if prev else None
    return {
        "price": price,
        "currency": info.get("currency"),
        "change_pct": change_pct,
        "day_low": _num(info.get("dayLow")),
        "day_high": _num(info.get("dayHigh")),
    }


def _price(symbol: str) -> str:
    return _compact({"symbol": symbol, **_price_fields(cache.cached_ticker_info(symbol))})


def _fundamentals(symbol: str) -> str:
    info = cache.cached_ticker_info(symbol)
    return _compact(
        {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            **_price_fields(info),
            "market_cap": _num(info.get("marketCap")),
            "pe_trailing": _num(info.get("trailingPE")),
            "pe_forward": _num(info.get("forwardPE")),
            "pb": _num(info.get("priceToBook")),
            "eps": _num(info.get("trailingEps")),
            "dividend_yield": _num(info.get("dividendYield")),
            "beta": _num(info.get("beta")),
            "profit_margin": _num(info.get("profitMargins")),
            "revenue_growth": _num(info.get("revenueGrowth")),
            "debt_to_equity": _num(info.get("debtToEquity")),
            "52w_low": _num(info.get("fiftyTwoWeekLow")),
            "52w_high": _num(info.get("fiftyTwoWeekHigh")),
        }
    )


def _recommendations(symbol: str) -> str:
    result: dict[str, Any] = {"symbol": symbol}
    table = cache.cached_recommendations(symbol)
    if table is not None and len(table):
        # yfinance returns one row per month ("0m", "-1m", …); keep the latest.
        latest = table.iloc[0].to_dict()
        result.update(
            {
                k: int(latest[k])
                for k in ("strongBuy", "buy", "hold", "sell", "strongSell")
                if k in latest and latest[k] == latest[k]  # skip NaN
            }
        )
    info = cache.cached_ticker_info(symbol)
    result.update(
        {
            "consensus": info.get("recommendationKey"),
            "analysts": info.get("numberOfAnalystOpinions"),
            "target_mean": _num(info.get("targetMeanPrice")),
            "target_low": _num(info.get("targetLowPrice")),
            "target_high": _num(info.get("targetHighPrice")),
        }
    )
    if len(result) == 1:
        raise LookupError("no analyst data")
    return _compact(result)


def _date(value: Any) -> str | None:
    """ISO date from an ISO string or an epoch timestamp."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc).date().isoformat()
    return str(value)[:10] if value else None


def _news(symbol: str, limit: int) -> str:
    stories = []
    for item in cache.cached_ticker_news(symbol)[:limit]:
        content = item.get("content") or {}
        title = (item.get("title") or content.get("title") or "").strip()
        if not title:
            continue
        stories.append(
            {
                "title": title[:_MAX_TITLE_CHARS],
                "publisher": item.get("publisher")
                or (content.get("provider") or {}).get("displayName"),
                "date": _date(content.get("pubDate") or item.get("providerPublishTime")),
            }
        )
    if not stories:
        return f"No recent news found for {symbol}."
    return json.dumps([{k: v for k, v in s.items() if v} for s in stories], separators=(",", ":"))
