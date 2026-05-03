"""Financial data service helpers used by the API layer."""

from __future__ import annotations

from typing import Any

import cache


def _safe_float(value: Any) -> float | None:
    """Convert a market value to float when possible."""

    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any) -> str:
    """Pretty-print a large finance number."""

    num = _safe_float(value)
    if num is None:
        return "n/a"
    abs_num = abs(num)
    if abs_num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    if abs_num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    if abs_num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    if abs_num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return f"{num:.2f}" if not float(num).is_integer() else f"{int(num)}"


def extract_symbols_from_entities(entities: list[Any]) -> list[str]:
    """Return deduplicated ticker symbols from NLP entities."""

    seen: set[str] = set()
    result: list[str] = []
    for ent in entities:
        ticker = ent.ticker if hasattr(ent, "ticker") else ent.get("ticker")
        label = ent.label if hasattr(ent, "label") else ent.get("label", "")
        if label == "TICKER" and ticker and ticker not in seen:
            seen.add(ticker)
            result.append(str(ticker).upper())
    return result


def get_recent_headlines(symbol: str, limit: int = 5) -> list[dict[str, Any]]:
    """Fetch recent headlines from Yahoo Finance when available."""

    import yfinance as yf

    headlines: list[dict[str, Any]] = []
    try:
        for item in (yf.Ticker(symbol).news or [])[:limit]:
            title = item.get("title") or (item.get("content") or {}).get("title", "")
            url = item.get("link") or (item.get("content") or {}).get("canonicalUrl", {}).get("url")
            publisher = item.get("publisher") or (item.get("content") or {}).get("provider", {}).get("displayName")
            if title:
                headlines.append({"title": title, "url": url, "publisher": publisher})
    except Exception:
        return []
    return headlines


def get_market_context(symbol: str) -> dict[str, Any]:
    """Fetch company info, fast price data, trend history, and headlines."""

    info = cache.cached_ticker_info(symbol)
    history = cache.cached_ticker_history(symbol, period="1mo")
    fast_info = cache.cached_fast_info(symbol)

    price = (
        _safe_float(info.get("currentPrice"))
        or _safe_float(getattr(fast_info, "last_price", None))
        or _safe_float(getattr(fast_info, "lastPrice", None))
    )
    prev_close = (
        _safe_float(info.get("regularMarketPreviousClose"))
        or _safe_float(getattr(fast_info, "previous_close", None))
        or _safe_float(getattr(fast_info, "regular_market_previous_close", None))
    )
    change_pct = None
    if price is not None and prev_close not in (None, 0):
        change_pct = round(((price - prev_close) / prev_close) * 100, 2)

    chart = []
    if history is not None and not history.empty:
        tail = history.tail(10)
        for idx, row in tail.iterrows():
            close = _safe_float(row.get("Close"))
            if close is not None:
                chart.append({"label": idx.strftime("%b %d"), "value": close})
        start_close = _safe_float(history.iloc[0].get("Close"))
        end_close = _safe_float(history.iloc[-1].get("Close"))
        week_change_pct = None
        if start_close not in (None, 0) and end_close is not None:
            week_change_pct = round(((end_close - start_close) / start_close) * 100, 2)
    else:
        week_change_pct = None

    return {
        "symbol": symbol.upper(),
        "company_name": info.get("shortName") or info.get("longName") or symbol.upper(),
        "price": price,
        "change_pct": change_pct,
        "week_change_pct": week_change_pct,
        "recommendation": info.get("recommendationKey"),
        "analyst_view": info.get("recommendationMean"),
        "sentiment_label": "bullish" if (change_pct or 0) > 1 else "bearish" if (change_pct or 0) < -1 else "neutral",
        "fundamentals": [
            {"label": "Market Cap", "value": _format_number(info.get("marketCap"))},
            {"label": "P/E", "value": _format_number(info.get("trailingPE"))},
            {"label": "Forward P/E", "value": _format_number(info.get("forwardPE"))},
            {"label": "Revenue", "value": _format_number(info.get("totalRevenue"))},
            {"label": "Profit Margin", "value": f"{(_safe_float(info.get('profitMargins')) or 0) * 100:.1f}%"},
            {"label": "52W Range", "value": f"{_safe_float(info.get('fiftyTwoWeekLow')) or 0:.2f} - {_safe_float(info.get('fiftyTwoWeekHigh')) or 0:.2f}"},
        ],
        "headlines": get_recent_headlines(symbol),
        "chart": chart,
        "sources": [
            {"label": "Yahoo Finance Quote", "kind": "market_data"},
            {"label": "Yahoo Finance News", "kind": "news"},
        ],
    }


def build_comparison(symbols: list[str]) -> dict[str, Any]:
    """Build a compact comparison payload for 2-4 stocks."""

    clean_symbols = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
    snapshots = [get_market_context(symbol) for symbol in clean_symbols[:4]]
    rows = []
    metrics = [
        ("Current Price", lambda snap: f"${snap['price']:.2f}" if snap["price"] is not None else "n/a"),
        ("1M Trend", lambda snap: f"{snap['week_change_pct']:+.2f}%" if snap["week_change_pct"] is not None else "n/a"),
        ("Market Cap", lambda snap: next((m["value"] for m in snap["fundamentals"] if m["label"] == "Market Cap"), "n/a")),
        ("P/E", lambda snap: next((m["value"] for m in snap["fundamentals"] if m["label"] == "P/E"), "n/a")),
        ("Forward P/E", lambda snap: next((m["value"] for m in snap["fundamentals"] if m["label"] == "Forward P/E"), "n/a")),
        ("Revenue", lambda snap: next((m["value"] for m in snap["fundamentals"] if m["label"] == "Revenue"), "n/a")),
        ("Sentiment", lambda snap: snap["sentiment_label"] or "neutral"),
    ]
    for label, getter in metrics:
        rows.append(
            {
                "metric": label,
                "values": {snap["symbol"]: getter(snap) for snap in snapshots},
            }
        )

    winner = None
    best_score = None
    for snap in snapshots:
        score = (snap["week_change_pct"] or 0) - (snap["change_pct"] or 0) * 0.25
        if best_score is None or score > best_score:
            best_score = score
            winner = snap["symbol"]

    recommendation = (
        f"{winner} currently leads the comparison on momentum and overall health signals."
        if winner
        else "Use the comparison table to decide which symbol deserves deeper research."
    )

    return {
        "symbols": [snap["symbol"] for snap in snapshots],
        "headline": "Compare valuation, trend, scale, and sentiment at a glance.",
        "winner": winner,
        "recommendation": recommendation,
        "rows": rows,
        "sources": [{"label": "Yahoo Finance", "kind": "market_data"}],
    }


def build_watchlist_items(symbols: list[str]) -> list[dict[str, Any]]:
    """Return watchlist price cards with lightweight intelligence."""

    items = []
    for sym in symbols:
        context = get_market_context(sym)
        item = {
            "symbol": context["symbol"],
            "price": context["price"],
            "change_pct": context["change_pct"],
            "signal": None,
        }
        headline = context["headlines"][0]["title"] if context["headlines"] else None
        if headline:
            item["signal"] = headline
        elif context["week_change_pct"] is not None:
            item["signal"] = f"1M trend {context['week_change_pct']:+.2f}%"
        items.append(item)
    return items

