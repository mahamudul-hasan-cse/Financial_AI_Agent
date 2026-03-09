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
"""

from __future__ import annotations

import logging
from typing import Callable

from agno.tools import Toolkit
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools

from tools.chart_tools import ChartTools
from tools.excel_tools import ExcelTools
from tools.rag_tools import RAGTools

logger = logging.getLogger("financial_agent")

__all__ = ["build_tools", "TOOL_FACTORIES"]

# ── Factory definitions ──────────────────────────────────────────────
#
# Each entry is ``(human_readable_name, zero-arg callable → Toolkit)``.
# The name is used solely for logging; it has no functional impact.

TOOL_FACTORIES: list[tuple[str, Callable[[], Toolkit]]] = [
    ("DuckDuckGo", lambda: DuckDuckGoTools()),
    ("YFinance", lambda: YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True,
    )),
    ("Charts", lambda: ChartTools()),
    ("Excel", lambda: ExcelTools()),
    ("RAG", lambda: RAGTools()),
]


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
