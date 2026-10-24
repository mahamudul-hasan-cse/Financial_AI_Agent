"""Toolkit registry for the Financial AI Agent.

Centralises toolkit construction with **graceful degradation**: each
toolkit factory is wrapped in its own try/except so a missing optional
dependency (e.g. ``matplotlib``, ``openpyxl``) only disables that
single toolkit instead of crashing the whole agent.

Public API
----------
- ``build_tools()`` – instantiate all registered toolkits.
- ``TOOL_FACTORIES`` – ordered list of ``(name, factory)`` tuples; safe
  to inspect or extend before calling ``build_tools()``.
- ``compact_tool_output`` – agno tool hook: caches read-only search tools
  and truncates oversized results so follow-up model calls stay small.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

os.environ.setdefault("AGNO_TELEMETRY", "false")

from agno.tools import Toolkit  # noqa: E402
from agno.tools.duckduckgo import DuckDuckGoTools  # noqa: E402

import cache  # noqa: E402
from config import settings  # noqa: E402
from tools.chart_tools import ChartTools  # noqa: E402
from tools.excel_tools import ExcelTools  # noqa: E402
from tools.market_tools import MarketDataTools  # noqa: E402
from tools.rag_tools import RAGTools  # noqa: E402

logger = logging.getLogger("financial_agent")

__all__ = [
    "build_tools",
    "compact_tool_output",
    "truncate_output",
    "TOOL_FACTORIES",
    "SEARCH_MAX_RESULTS",
]

# Web/news search results per call (each result carries a body snippet).
SEARCH_MAX_RESULTS = 3

# Read-only tools whose results may be served from the TTL cache.  Chart and
# Excel tools write files, and market tools cache themselves.
_CACHEABLE_TOOLS = frozenset({"duckduckgo_search", "duckduckgo_news", "retrieve_financial_context"})

# ── Factory definitions ──────────────────────────────────────────────
#
# Each entry is ``(human_readable_name, zero-arg callable → Toolkit)``.
# The name is used solely for logging; it has no functional impact.

TOOL_FACTORIES: list[tuple[str, Callable[[], Toolkit]]] = [
    ("DuckDuckGo", lambda: DuckDuckGoTools(fixed_max_results=SEARCH_MAX_RESULTS)),
    ("MarketData", lambda: MarketDataTools()),
    ("Charts", lambda: ChartTools()),
    ("Excel", lambda: ExcelTools()),
    ("RAG", lambda: RAGTools()),
]


# ── Tool hook ────────────────────────────────────────────────────────

def truncate_output(text: str, limit: int | None = None) -> str:
    """Cut *text* to *limit* characters (default ``TOOL_OUTPUT_MAX_CHARS``)."""
    limit = limit or settings.tool_output_max_chars
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …[truncated]"


def compact_tool_output(
    function_name: str, function_call: Callable[..., Any], arguments: dict[str, Any]
) -> Any:
    """agno ``tool_hooks`` entry wrapping every tool call.

    * Search/knowledge tools are cached per function + arguments, so the
      same query in a follow-up question skips the network.
    * String results longer than ``TOOL_OUTPUT_MAX_CHARS`` are truncated —
      tool output is re-sent to the model on every subsequent round.
    """

    def call() -> Any:
        result = function_call(**arguments)
        return truncate_output(result) if isinstance(result, str) else result

    if function_name not in _CACHEABLE_TOOLS:
        return call()
    key = cache.make_key(
        f"tool:{function_name}", "", json.dumps(arguments, sort_keys=True, default=str)
    )
    return cache.get_or_set(key, call)


# ── Public builder ───────────────────────────────────────────────────

def build_tools(
    factories: list[tuple[str, Callable[[], Toolkit]]] | None = None,
) -> list[Toolkit]:
    """Instantiate all toolkits, skipping any that fail.

    Parameters
    ----------
    factories:
        Override the default ``TOOL_FACTORIES`` (useful for testing).
        When *None*, the module-level ``TOOL_FACTORIES`` list is used.
    """
    entries = factories if factories is not None else TOOL_FACTORIES
    tools: list[Toolkit] = []
    for name, factory in entries:
        try:
            tools.append(factory())
            logger.debug("Loaded toolkit: %s", name)
        except Exception:
            logger.warning(
                "Failed to load toolkit '%s' — skipping.", name, exc_info=True,
            )
    return tools
