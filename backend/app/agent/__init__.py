"""backend.app.agent — Financial AI Agent package.

Re-exports the public API so consumers can do::

    from backend.app.agent import agent_service, agent, create_agent
"""

from backend.app.agent.financial_agent import (  # noqa: F401
    FinancialAgentService,
    agent_service,
    agent,
    create_agent,
    main,
    DEFAULT_MODEL_ID,
    AGENT_INSTRUCTIONS,
)
from backend.app.agent.prompts import (          # noqa: F401
    TOOL_CATALOGUE,
    BEHAVIOURAL_RULES,
)
from backend.app.agent.tool_registry import (    # noqa: F401
    build_tools,
    TOOL_FACTORIES,
)

__all__ = [
    "FinancialAgentService",
    "agent_service",
    "agent",
    "create_agent",
    "main",
    "DEFAULT_MODEL_ID",
    "AGENT_INSTRUCTIONS",
    "TOOL_CATALOGUE",
    "BEHAVIOURAL_RULES",
    "build_tools",
    "TOOL_FACTORIES",
]
