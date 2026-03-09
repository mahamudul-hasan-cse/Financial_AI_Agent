"""Backward-compatible shim — delegates to ``backend.app.agent``.

Every public name that previously lived here is re-exported so that
existing consumers (``api.py``, tests, CLI) continue to work with
**zero import changes**.

Canonical location: ``backend/app/agent/``
"""

from __future__ import annotations

# ── Re-exports from the new package ──────────────────────────────────
from backend.app.agent.prompts import (              # noqa: F401
    TOOL_CATALOGUE,
    BEHAVIOURAL_RULES,
    AGENT_INSTRUCTIONS,
)
from backend.app.agent.tool_registry import (        # noqa: F401
    build_tools as _build_tools,
    build_tools,
    TOOL_FACTORIES,
)
from backend.app.agent.financial_agent import (      # noqa: F401
    FinancialAgentService,
    agent_service,
    agent,
    create_agent,
    main,
    DEFAULT_MODEL_ID,
    _interactive_loop,
)

__all__ = [
    "FinancialAgentService",
    "agent_service",
    "agent",
    "create_agent",
    "main",
    "AGENT_INSTRUCTIONS",
    "TOOL_CATALOGUE",
    "BEHAVIOURAL_RULES",
    "_build_tools",
    "build_tools",
    "TOOL_FACTORIES",
    "DEFAULT_MODEL_ID",
    "_interactive_loop",
]


if __name__ == "__main__":
    main()
