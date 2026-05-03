"""Simple in-memory TTL cache for stock data.

Caches yfinance responses (ticker info, history, fast_info) to avoid
hammering the Yahoo Finance API with repeated identical requests.

Design:
  - Thread-safe via a simple dict + timestamp check (GIL protects reads/writes).
  - Default TTL: 5 minutes — fresh enough for intraday use, avoids API rate limits.
  - Falls back gracefully: if Redis or another backend is added later, this
    module's interface stays the same.

Academic relevance:
  - Demonstrates caching as an optimization technique (memoization pattern).
  - Shows TTL-based cache invalidation for time-sensitive financial data.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

# Default time-to-live in seconds
DEFAULT_TTL: int = 300  # 5 minutes

# Maximum cache entries before LRU eviction kicks in
MAX_CACHE_SIZE: int = 1000

# Internal cache store: key -> (value, expiry_timestamp)
# OrderedDict preserves insertion/access order for LRU eviction
_cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()


def get(key: str) -> Any | None:
    """Retrieve a cached value if it exists and hasn't expired.

    Args:
        key: Cache key string.

    Returns:
        The cached value, or ``None`` if missing or expired.
    """
    entry = _cache.get(key)
    if entry is None:
        return None
    value, expiry = entry
    if time.time() > expiry:
        _cache.pop(key, None)
        return None
    # Move to end on access (most-recently used)
    _cache.move_to_end(key)
    return value


def set(key: str, value: Any, ttl: int = DEFAULT_TTL) -> None:
    """Store a value in the cache with a TTL.

    Enforces MAX_CACHE_SIZE via LRU eviction: when the cache is full,
    the oldest entry (least-recently inserted) is evicted first.

    Args:
        key: Cache key string.
        value: Value to cache.
        ttl: Time-to-live in seconds (default 300 = 5 minutes).
    """
    # If key already exists, remove it so it moves to the end (most recent)
    if key in _cache:
        _cache.move_to_end(key)
    _cache[key] = (value, time.time() + ttl)
    # Evict oldest entries if cache exceeds max size
    while len(_cache) > MAX_CACHE_SIZE:
        _cache.popitem(last=False)


def invalidate(key: str) -> None:
    """Remove a specific key from the cache."""
    _cache.pop(key, None)


def clear() -> None:
    """Clear the entire cache."""
    _cache.clear()


def stats() -> dict[str, int]:
    """Return basic cache statistics."""
    now = time.time()
    total = len(_cache)
    alive = sum(1 for _, (_, exp) in _cache.items() if now <= exp)
    return {"total_entries": total, "alive_entries": alive, "expired_entries": total - alive}


def cached_ticker_info(symbol: str) -> dict:
    """Fetch and cache yfinance Ticker.info for a symbol.

    Returns cached data if available and fresh, otherwise fetches from
    Yahoo Finance and caches the result for DEFAULT_TTL seconds.
    """
    import yfinance as yf

    key = f"ticker_info:{symbol.upper()}"
    cached = get(key)
    if cached is not None:
        return cached

    ticker = yf.Ticker(symbol)
    info = ticker.info
    set(key, info)
    return info


def cached_ticker_history(symbol: str, period: str = "1mo") -> "pd.DataFrame":
    """Fetch and cache yfinance Ticker.history() for a symbol + period.

    Returns cached DataFrame if available and fresh, otherwise fetches
    from Yahoo Finance and caches the result for DEFAULT_TTL seconds.
    """
    import yfinance as yf

    key = f"ticker_history:{symbol.upper()}:{period}"
    cached = get(key)
    if cached is not None:
        return cached

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    set(key, hist)
    return hist


def cached_fast_info(symbol: str) -> Any:
    """Fetch and cache yfinance Ticker.fast_info for a symbol.

    Returns cached data if available and fresh, otherwise fetches
    from Yahoo Finance and caches the result for DEFAULT_TTL seconds.
    """
    import yfinance as yf

    key = f"fast_info:{symbol.upper()}"
    cached = get(key)
    if cached is not None:
        return cached

    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    set(key, info)
    return info
