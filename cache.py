"""Thread-safe in-memory TTL cache for market data and tool results.

Caches yfinance responses (ticker info, history, news, recommendations) and
agent tool outputs so repeated questions about the same ticker don't hammer
Yahoo Finance or DuckDuckGo, and so one chat turn that needs both a price and
fundamentals only fetches ``Ticker.info`` once.

Design:
  - One ``RLock`` guards the store; reads, writes and LRU eviction are atomic.
  - :func:`get_or_set` is *single-flight*: concurrent callers asking for the
    same missing key wait for one fetch instead of all hitting the network.
  - Default TTL comes from ``CACHE_TTL_SECONDS`` (60s). ``0`` disables caching.
  - Keys are ``"<function>:<TICKER>[:<args>]"`` — see :func:`make_key`.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from config import settings

if TYPE_CHECKING:
    import pandas as pd

T = TypeVar("T")

# Default time-to-live in seconds (CACHE_TTL_SECONDS); read at call time so
# tests can patch it.
DEFAULT_TTL: int = settings.cache_ttl_seconds

# Maximum cache entries before LRU eviction kicks in
MAX_CACHE_SIZE: int = 1000

# Internal cache store: key -> (value, expiry_timestamp), in LRU order
_cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
_lock = threading.RLock()
# Per-key locks for single-flight fetches in get_or_set()
_key_locks: dict[str, threading.Lock] = {}

_MISSING = object()


def make_key(function: str, symbol: str, *args: Any) -> str:
    """Build a cache key from a function name, ticker and extra arguments."""
    parts = [function, symbol.strip().upper(), *(str(a) for a in args)]
    return ":".join(parts)


def _lookup(key: str) -> Any:
    entry = _cache.get(key)
    if entry is None:
        return _MISSING
    value, expiry = entry
    if time.monotonic() > expiry:
        _cache.pop(key, None)
        return _MISSING
    _cache.move_to_end(key)
    return value


def get(key: str) -> Any | None:
    """Return the cached value, or ``None`` if missing or expired."""
    with _lock:
        value = _lookup(key)
    return None if value is _MISSING else value


def set(key: str, value: Any, ttl: int | None = None) -> None:
    """Store *value* for *ttl* seconds (default ``DEFAULT_TTL``; ``<= 0`` skips)."""
    ttl = DEFAULT_TTL if ttl is None else ttl
    if ttl <= 0:
        return
    with _lock:
        _cache[key] = (value, time.monotonic() + ttl)
        _cache.move_to_end(key)
        while len(_cache) > MAX_CACHE_SIZE:
            _cache.popitem(last=False)


def get_or_set(key: str, factory: Callable[[], T], ttl: int | None = None) -> T:
    """Return the cached value for *key*, computing it with *factory* on a miss.

    Only one thread runs *factory* per key at a time; others wait and reuse
    its result.  Exceptions from *factory* propagate and nothing is cached.
    """
    with _lock:
        value = _lookup(key)
        if value is not _MISSING:
            return value
        key_lock = _key_locks.setdefault(key, threading.Lock())

    with key_lock:
        with _lock:
            value = _lookup(key)
        if value is not _MISSING:
            return value
        try:
            value = factory()
            set(key, value, ttl)
            return value
        finally:
            with _lock:
                _key_locks.pop(key, None)


def invalidate(key: str) -> None:
    """Remove a specific key from the cache."""
    with _lock:
        _cache.pop(key, None)


def clear() -> None:
    """Clear the entire cache."""
    with _lock:
        _cache.clear()


def stats() -> dict[str, int]:
    """Return basic cache statistics."""
    now = time.monotonic()
    with _lock:
        total = len(_cache)
        alive = sum(1 for _, exp in _cache.values() if now <= exp)
    return {"total_entries": total, "alive_entries": alive, "expired_entries": total - alive}


# ── yfinance helpers ─────────────────────────────────────────────────

def _ticker(symbol: str) -> Any:
    import yfinance as yf

    return yf.Ticker(symbol.strip().upper())


def cached_ticker_info(symbol: str) -> dict:
    """``yfinance.Ticker(symbol).info``, cached per ticker."""
    return get_or_set(make_key("ticker_info", symbol), lambda: _ticker(symbol).info or {})


def cached_ticker_history(symbol: str, period: str = "1mo") -> "pd.DataFrame":
    """``yfinance.Ticker(symbol).history(period)``, cached per ticker + period."""
    return get_or_set(
        make_key("ticker_history", symbol, period),
        lambda: _ticker(symbol).history(period=period),
    )


def cached_fast_info(symbol: str) -> Any:
    """``yfinance.Ticker(symbol).fast_info``, cached per ticker."""
    return get_or_set(make_key("fast_info", symbol), lambda: _ticker(symbol).fast_info)


def cached_ticker_news(symbol: str) -> list[dict[str, Any]]:
    """``yfinance.Ticker(symbol).news``, cached per ticker."""
    return get_or_set(make_key("ticker_news", symbol), lambda: list(_ticker(symbol).news or []))


def cached_recommendations(symbol: str) -> "pd.DataFrame | None":
    """``yfinance.Ticker(symbol).recommendations``, cached per ticker."""
    return get_or_set(
        make_key("recommendations", symbol), lambda: _ticker(symbol).recommendations
    )
