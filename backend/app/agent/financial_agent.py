"""Financial AI Agent — service class and CLI.

Provides a thin ``FinancialAgentService`` that wraps the raw ``agno``
``Agent`` with retry logic, clean return types, and structured error
handling so the API layer doesn't need to know agent internals.

Public API
----------
- ``FinancialAgentService`` – the main service class.
- ``agent_service``         – ready-to-use singleton instance.
- ``agent``                 – raw agno Agent (backward-compat; prefer the service).
- ``create_agent()``        – factory for a raw Agent with overrides.
- ``main()``                – CLI entry-point.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Generator, Iterator, Sequence

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

load_dotenv(override=True)

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools import Toolkit

from backend.app.agent.prompts import AGENT_INSTRUCTIONS
from backend.app.agent.tool_registry import build_tools

logger = logging.getLogger("financial_agent")

__all__ = [
    "FinancialAgentService",
    "agent_service",
    "agent",
    "create_agent",
    "main",
    "AGENT_INSTRUCTIONS",
    "DEFAULT_MODEL_ID",
]

# ── Model configuration ──────────────────────────────────────────────
DEFAULT_MODEL_ID: str = "meta-llama/llama-4-scout-17b-16e-instruct"
_DEFAULT_MAX_RETRIES: int = 3

# Regex to strip agno tool-call metadata lines from agent responses
# Matches lines like: "get_current_stock_price(symbol=AAPL) completed in 1.23s."
# Also matches: "Running: func(args)" and "┃ Running: ..." style lines
_TOOL_META_RE = re.compile(
    r"^"
    r"(?:"
    r"\s*\w+\(.*?\)\s+completed\s+in\s+[\d.]+s\.?"
    r"|\s*[┃│|]*\s*Running:\s+\w+\(.*?\)"
    r"|\s*[┃│|]*\s*\w+\(.*?\)\s*$"
    r")\s*$",
    re.MULTILINE,
)


def _strip_tool_metadata(text: str) -> str:
    """Remove internal tool execution metadata lines from agent output.

    Args:
        text: Raw agent response that may contain tool-call logs.

    Returns:
        Cleaned text with metadata lines removed and excess blank lines collapsed.
    """
    cleaned = _TOOL_META_RE.sub("", text)
    # Also strip inline metadata that isn't on its own line
    cleaned = re.sub(r"\w+\([^)]*\)\s+completed\s+in\s+[\d.]+s\.?\s*", "", cleaned)
    # Collapse runs of 3+ newlines down to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ── Agent factory (unchanged public contract) ────────────────────────

def create_agent(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    api_key: str | None = None,
    tools: Sequence[Toolkit] | None = None,
    instructions: list[str] | None = None,
) -> Agent:
    """Create and return a configured :class:`Agent`.

    Parameters
    ----------
    model_id:
        Groq model identifier.  Defaults to ``DEFAULT_MODEL_ID``.
    api_key:
        Groq API key.  Falls back to the ``GROQ_API_KEY`` env-var.
    tools:
        Override the default toolkit list (useful for testing).
    instructions:
        Override the default prompt instructions.

    Raises
    ------
    RuntimeError
        If no Groq API key is available.
    """
    resolved_key: str | None = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    return Agent(
        model=Groq(id=model_id, api_key=resolved_key),
        tools=list(tools) if tools is not None else build_tools(),
        instructions=instructions if instructions is not None else AGENT_INSTRUCTIONS,
        show_tool_calls=False,
        markdown=True,
        num_history_responses=3,
    )


# ── Service class ────────────────────────────────────────────────────

class FinancialAgentService:
    """High-level wrapper around the agno Agent.

    Provides two execution methods that the API layer should call:

    * :meth:`run`    – synchronous, returns ``str``.
    * :meth:`stream` – yields ``str`` chunks for SSE.

    Both methods include retry logic and structured exception handling
    so the caller never has to deal with raw agno internals.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._agent: Agent = agent if agent is not None else create_agent()
        self._max_retries: int = max_retries

    # -- Expose underlying agent for consumers that need it ------------

    @property
    def agent(self) -> Agent:
        """The underlying agno ``Agent`` instance."""
        return self._agent

    # -- Synchronous execution -----------------------------------------

    def run(self, prompt: str) -> str:
        """Execute the agent and return the full response text.

        Retries up to ``max_retries`` times on transient failures.

        Returns
        -------
        str
            The agent's response content.

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._agent.run(prompt, stream=False)
                raw: str = (
                    result.content if hasattr(result, "content") else str(result)
                )
                content = _strip_tool_metadata(raw)
                logger.info("Agent run succeeded on attempt %d", attempt)
                return content
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Agent attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    str(exc)[:200],
                )
        raise RuntimeError(
            f"Agent failed after {self._max_retries} attempts"
        ) from last_error

    # -- Streaming execution -------------------------------------------

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Execute the agent in streaming mode, yielding text chunks.

        Unlike :meth:`run`, streaming does **not** retry — if the
        underlying stream errors mid-flight, an ``"[ERROR]"`` sentinel
        chunk is yielded and the generator ends.

        Yields
        ------
        str
            Individual text chunks from the agent.  The final yield may
            be an ``"[ERROR] ..."`` string if the stream failed.
        """
        try:
            run_stream = self._agent.run(prompt, stream=True)
            for chunk in run_stream:
                text: str = ""
                if hasattr(chunk, "content") and chunk.content:
                    text = chunk.content
                elif isinstance(chunk, str):
                    text = chunk
                if text:
                    yield text
        except Exception:
            logger.exception("Stream error")
            yield "[ERROR] An error occurred while generating the response."


# ── Module-level singletons (backward-compatible) ────────────────────
agent_service: FinancialAgentService = FinancialAgentService()
agent: Agent = agent_service.agent


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run the agent interactively from the command line."""
    parser = argparse.ArgumentParser(
        description="Financial AI Agent — ask questions about stocks, news, and markets.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="A single query to run (if omitted, starts interactive mode).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream the response token-by-token.",
    )
    args = parser.parse_args()

    if args.query:
        agent.print_response(args.query, stream=args.stream)
    else:
        _interactive_loop(stream=args.stream)


def _interactive_loop(*, stream: bool = False) -> None:
    """REPL loop for interactive CLI usage."""
    print("Financial AI Agent (type 'exit' to quit)")
    print("-" * 45)
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not query or query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        try:
            agent.print_response(query, stream=stream)
        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
