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
import re
import time
from typing import Any, Generator, Sequence

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools import Toolkit

from backend.app.agent.prompts import AGENT_INSTRUCTIONS
from backend.app.agent.tool_registry import build_tools
from config import (
    DEFAULT_GROQ_MODEL,
    get_groq_api_key,
    get_llm_provider,
    settings,
)

logger = logging.getLogger("financial_agent")

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
    "DEFAULT_MODEL_ID",
    "DEFAULT_OLLAMA_MODEL",
    "LLM_PROVIDER",
]

# ── Model configuration ──────────────────────────────────────────────
# All values come from config.py (env vars: LLM_PROVIDER, GROQ_MODEL,
# OLLAMA_MODEL, OLLAMA_HOST).
DEFAULT_MODEL_ID: str = DEFAULT_GROQ_MODEL
DEFAULT_OLLAMA_MODEL: str = settings.ollama_model
_DEFAULT_MAX_RETRIES: int = 3

# Provider selected at import time ("groq" or "ollama"). ``create_agent``
# re-reads the environment on every call so tests and CLIs can switch it.
LLM_PROVIDER: str = settings.llm_provider
_OLLAMA_MODEL: str = settings.ollama_model
_OLLAMA_HOST: str = settings.ollama_host


class AgentNotConfiguredError(RuntimeError):
    """Raised when no usable LLM provider is configured (e.g. missing GROQ_API_KEY)."""


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


def _strip_tool_metadata_streaming(text: str) -> str:
    """Remove tool metadata while preserving natural in-progress spacing."""

    cleaned = _TOOL_META_RE.sub("", text)
    cleaned = re.sub(r"\w+\([^)]*\)\s+completed\s+in\s+[\d.]+s\.?\s*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ── Agent factory ────────────────────────────────────────────────────

def _build_groq_model(model_id: str, api_key: str | None) -> Groq:
    resolved_key: str | None = api_key or get_groq_api_key()
    if not resolved_key:
        raise AgentNotConfiguredError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set LLM_PROVIDER=ollama to use a local model."
        )
    return Groq(id=model_id, api_key=resolved_key)


def _build_ollama_model(model_id: str, host: str | None = None) -> "Any":
    """Create an agno Ollama model, verifying connectivity first.

    Raises ``RuntimeError`` if Ollama is not reachable so the caller can
    fall back to Groq.
    """
    resolved_host = (host or _OLLAMA_HOST).rstrip("/")
    try:
        import httpx

        resp = httpx.get(f"{resolved_host}/api/tags", timeout=3.0)
        resp.raise_for_status()
        from agno.models.ollama import Ollama

        return Ollama(id=model_id, host=resolved_host)
    except Exception as exc:
        logger.debug("Ollama connectivity check failed", exc_info=True)
        raise RuntimeError(f"Ollama unavailable ({type(exc).__name__}: {exc})") from exc


def _resolve_model(
    *,
    provider: str,
    model_id: str,
    api_key: str | None,
) -> Any:
    """Return an agno model instance for *provider*.

    ``ollama`` falls back to Groq when Ollama is unreachable *and* a Groq
    key is available; otherwise the original error is raised.
    """
    if provider == "ollama":
        ollama_model_id = model_id if model_id != DEFAULT_MODEL_ID else settings.ollama_model
        try:
            t0 = time.monotonic()
            model = _build_ollama_model(ollama_model_id, settings.ollama_host)
            logger.info(
                "[LLM] Using Ollama model=%s host=%s (connectivity check: %.1fms)",
                ollama_model_id,
                settings.ollama_host,
                (time.monotonic() - t0) * 1000,
            )
            return model
        except RuntimeError as exc:
            if not (api_key or get_groq_api_key()):
                raise AgentNotConfiguredError(str(exc)) from exc
            logger.warning(
                "[LLM] %s — falling back to Groq model=%s", exc, DEFAULT_MODEL_ID
            )
            model_id = DEFAULT_MODEL_ID

    model = _build_groq_model(model_id, api_key)
    logger.info("[LLM] Using Groq model=%s", model_id)
    return model


def create_agent(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    api_key: str | None = None,
    tools: Sequence[Toolkit] | None = None,
    instructions: list[str] | None = None,
    provider: str | None = None,
    model: Any | None = None,
    session_id: str | None = None,
) -> Agent:
    """Create and return a configured :class:`Agent`.

    Parameters
    ----------
    model_id:
        Model identifier (Groq default: ``DEFAULT_MODEL_ID``; when the
        provider is Ollama the default becomes ``OLLAMA_MODEL``).
    api_key:
        Groq API key.  Falls back to the ``GROQ_API_KEY`` env-var.
    tools:
        Override the default toolkit list (useful for testing).
    instructions:
        Override the default prompt instructions.
    provider:
        ``"groq"`` or ``"ollama"``. Defaults to the ``LLM_PROVIDER`` env-var.
    model:
        A pre-built agno model to reuse (skips provider resolution).
    session_id:
        Session the agent run belongs to.  agno's own chat history is
        disabled: the API injects the (per-session) history into the prompt
        itself, so nothing is shared between sessions or users.

    Raises
    ------
    AgentNotConfiguredError
        If no Groq API key is available when Groq is (or falls back to) the provider.
    """
    if model is None:
        resolved_provider = (provider or get_llm_provider()).lower().strip()
        model = _resolve_model(provider=resolved_provider, model_id=model_id, api_key=api_key)

    return Agent(
        model=model,
        tools=list(tools) if tools is not None else build_tools(),
        instructions=instructions if instructions is not None else AGENT_INSTRUCTIONS,
        show_tool_calls=False,
        markdown=True,
        add_history_to_messages=False,
        session_id=session_id,
    )


# ── Service class ────────────────────────────────────────────────────

class FinancialAgentService:
    """High-level wrapper around the agno Agent.

    Provides two execution methods that the API layer should call:

    * :meth:`run`    – synchronous, returns ``str``.
    * :meth:`stream` – yields ``str`` chunks for SSE.

    The underlying agent is created lazily, so importing this module never
    requires an API key.  Unless a pre-built agent is injected, every call
    runs on a *fresh* ``Agent`` that shares the (stateless) model client and
    toolkits but has its own run state and memory, so concurrent requests
    from different sessions can't see each other's conversation.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        self._injected: bool = agent is not None
        self._agent: Agent | None = agent
        self._max_retries: int = max_retries
        self.init_error: str | None = None

    # -- Expose underlying agent for consumers that need it ------------

    @property
    def agent(self) -> Agent:
        """The template agno ``Agent`` (built on first access)."""
        if self._agent is None:
            try:
                self._agent = create_agent()
                self.init_error = None
            except Exception as exc:
                self.init_error = str(exc)
                raise
        return self._agent

    @property
    def provider(self) -> str:
        """Name of the model class in use, or the configured provider."""
        if self._agent is not None:
            return type(self._agent.model).__name__
        return get_llm_provider()

    def _agent_for(self, session_id: str | None) -> Agent:
        """Return an agent instance that is safe to use for *session_id*."""
        if self._injected:
            return self.agent
        template = self.agent
        return create_agent(
            model=template.model,
            tools=template.tools,
            instructions=template.instructions,
            session_id=session_id,
        )

    @staticmethod
    def _run_kwargs(stream: bool, session_id: str | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"stream": stream}
        if session_id:
            kwargs["session_id"] = session_id
        return kwargs

    # -- Synchronous execution -----------------------------------------

    def run(self, prompt: str, session_id: str | None = None) -> str:
        """Execute the agent and return the full response text.

        Retries up to ``max_retries`` times on transient failures.

        Raises
        ------
        AgentNotConfiguredError
            If no LLM provider is configured (not retried).
        RuntimeError
            If all retry attempts are exhausted.
        """
        agent = self._agent_for(session_id)
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                t0 = time.monotonic()
                result = agent.run(prompt, **self._run_kwargs(False, session_id))
                elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                raw: str = (
                    result.content if hasattr(result, "content") else str(result)
                )
                content = _strip_tool_metadata(raw)
                logger.info("Agent run succeeded on attempt %d", attempt)
                logger.info(
                    "[LLM latency] provider=%s attempt=%d elapsed=%.1fms",
                    type(agent.model).__name__,
                    attempt,
                    elapsed_ms,
                )
                return content
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Agent attempt %d/%d failed: %s",
                    attempt,
                    self._max_retries,
                    str(exc)[:200],
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
        raise RuntimeError(
            f"Agent failed after {self._max_retries} attempts"
        ) from last_error

    # -- Streaming execution -------------------------------------------

    def stream(self, prompt: str, session_id: str | None = None) -> Generator[str, None, None]:
        """Execute the agent in streaming mode, yielding text chunks.

        Unlike :meth:`run`, streaming does **not** retry — if the
        underlying stream errors mid-flight, an ``"[ERROR]"`` sentinel
        chunk is yielded and the generator ends.

        Raises
        ------
        AgentNotConfiguredError
            Before the first chunk, if no LLM provider is configured.
        """
        agent = self._agent_for(session_id)
        t0 = time.monotonic()
        first_chunk = True
        raw_buffer = ""
        emitted_clean = ""
        try:
            run_stream = agent.run(prompt, **self._run_kwargs(True, session_id))
            for chunk in run_stream:
                text: str = ""
                if hasattr(chunk, "content") and chunk.content:
                    text = chunk.content
                elif isinstance(chunk, str):
                    text = chunk
                if text:
                    if first_chunk:
                        logger.info(
                            "[LLM latency] provider=%s TTFT=%.1fms (stream)",
                            type(agent.model).__name__,
                            (time.monotonic() - t0) * 1000,
                        )
                        first_chunk = False
                    raw_buffer += text
                    cleaned = _strip_tool_metadata_streaming(raw_buffer)
                    if cleaned.startswith(emitted_clean):
                        delta = cleaned[len(emitted_clean):]
                    else:
                        delta = cleaned
                    if delta:
                        emitted_clean = cleaned
                        yield delta
        except Exception:
            logger.exception("Stream error")
            yield "[ERROR] An error occurred while generating the response."


def llm_status() -> dict[str, Any]:
    """Describe the LLM configuration without building the agent.

    Returns ``{"provider": str, "configured": bool, "detail": str | None}``.
    """
    provider = get_llm_provider()
    if agent_service.init_error:
        return {"provider": provider, "configured": False, "detail": agent_service.init_error}
    if provider == "groq" and not get_groq_api_key():
        return {"provider": provider, "configured": False, "detail": "GROQ_API_KEY is not set."}
    return {"provider": provider, "configured": True, "detail": None}


# ── Module-level singletons (backward-compatible) ────────────────────
agent_service: FinancialAgentService = FinancialAgentService()


def get_agent_service() -> FinancialAgentService:
    """Return the process-wide agent service (agent is built lazily)."""
    return agent_service


def __getattr__(name: str) -> Any:
    # ``agent`` is resolved lazily so importing this module (and api.py)
    # works without GROQ_API_KEY.
    if name == "agent":
        return agent_service.agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _cli_agent() -> Agent:
    patched = globals().get("agent")
    return patched if patched is not None else agent_service.agent


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
        _cli_agent().print_response(args.query, stream=args.stream)
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
            _cli_agent().print_response(query, stream=stream)
        except Exception as e:
            logger.error("Agent error: %s", e, exc_info=True)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
