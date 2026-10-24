"""backend.app.agent — Financial AI Agent package.

Re-exports the public API so consumers can do::

    from backend.app.agent import agent_service, create_agent

``agent`` is resolved lazily (see ``financial_agent.__getattr__``) so the
package imports cleanly without an LLM API key.
"""

import os
from typing import Any

# Opt out of agno telemetry before agno is first imported (AGNO_TELEMETRY=true
# in the environment still opts back in).
os.environ.setdefault("AGNO_TELEMETRY", "false")

from backend.app.agent import financial_agent as _impl  # noqa: E402
from backend.app.agent.financial_agent import (  # noqa: E402, F401
    AGENT_INSTRUCTIONS,
    DEFAULT_MODEL_ID,
    AgentNotConfiguredError,
    FinancialAgentService,
    agent_service,
    create_agent,
    get_agent_service,
    llm_status,
    main,
)
from backend.app.agent.prompts import BEHAVIOURAL_RULES, TOOL_CATALOGUE  # noqa: E402, F401
from backend.app.agent.tool_registry import TOOL_FACTORIES, build_tools  # noqa: E402, F401

__all__ = [  # noqa: F822
    "FinancialAgentService",
    "AgentNotConfiguredError",
    "agent_service",
    "agent",  # resolved lazily via __getattr__
    "create_agent",
    "get_agent_service",
    "llm_status",
    "main",
    "DEFAULT_MODEL_ID",
    "AGENT_INSTRUCTIONS",
    "TOOL_CATALOGUE",
    "BEHAVIOURAL_RULES",
    "build_tools",
    "TOOL_FACTORIES",
]


def __getattr__(name: str) -> Any:
    if name == "agent":
        return _impl.agent_service.agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
