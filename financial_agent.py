"""Backward-compatible shim — delegates to ``backend.app.agent``.

Every public name that previously lived here is re-exported so that
existing consumers (tests, CLI, notebooks) keep working.  ``agent`` is
resolved lazily so importing this module never requires GROQ_API_KEY.

Canonical location: ``backend/app/agent/``
"""

from __future__ import annotations

from typing import Any

from backend.app.agent import financial_agent as _impl
from backend.app.agent.financial_agent import (  # noqa: F401
    DEFAULT_MODEL_ID,
    DEFAULT_OLLAMA_MODEL,
    LLM_PROVIDER,
    AgentNotConfiguredError,
    FinancialAgentService,
    _interactive_loop,
    agent_service,
    create_agent,
    get_agent_service,
    llm_status,
    main,
)
from backend.app.agent.prompts import (  # noqa: F401
    AGENT_INSTRUCTIONS,
    BEHAVIOURAL_RULES,
    TOOL_CATALOGUE,
)
from backend.app.agent.tool_registry import TOOL_FACTORIES, build_tools  # noqa: F401
from backend.app.agent.tool_registry import build_tools as _build_tools  # noqa: F401

__all__ = [
    "FinancialAgentService",
    "AgentNotConfiguredError",
    "agent_service",
    "agent",
    "create_agent",
    "get_agent_service",
    "llm_status",
    "main",
    "AGENT_INSTRUCTIONS",
    "TOOL_CATALOGUE",
    "BEHAVIOURAL_RULES",
    "build_tools",
    "TOOL_FACTORIES",
    "DEFAULT_MODEL_ID",
    "DEFAULT_OLLAMA_MODEL",
    "LLM_PROVIDER",
]


def __getattr__(name: str) -> Any:
    if name == "agent":
        return _impl.agent_service.agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
