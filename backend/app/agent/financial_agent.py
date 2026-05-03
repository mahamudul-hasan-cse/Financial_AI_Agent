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
import time
from typing import Any, Generator, Sequence

from dotenv import load_dotenv

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
    "LLM_PROVIDER",
]

# ── Model configuration ──────────────────────────────────────────────
DEFAULT_MODEL_ID: str = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_OLLAMA_MODEL: str = "qwen2.5:7b"
_DEFAULT_MAX_RETRIES: int = 3

# Read from env — "ollama" or "groq" (default: groq)
LLM_PROVIDER: str = "groq"
_OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
_OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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


# ── Agent factory (unchanged public contract) ────────────────────────

def _build_groq_model(model_id: str, api_key: str | None) -> Groq:
    resolved_key: str | None = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(id=model_id, api_key=resolved_key)


def _build_ollama_model(model_id: str) -> "Any":
    """Try to create an agno Ollama model, verifying connectivity first.

    Returns the model object, or raises ``RuntimeError`` if Ollama is not
    reachable so the caller can fall back to Groq.
    """
    try:
        import httpx
        resp = httpx.get(f"{_OLLAMA_HOST}/api/tags", timeout=3.0)
        resp.raise_for_status()
        from agno.models.ollama import Ollama
        return Ollama(id=model_id, host=_OLLAMA_HOST)
    except Exception as exc:
        import traceback
        print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
        raise RuntimeError(f"Ollama unavailable ({type(exc).__name__}: {exc})") from exc


def create_agent(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    api_key: str | None = None,
    tools: Sequence[Toolkit] | None = None,
    instructions: list[str] | None = None,
    provider: str | None = None,
) -> Agent:
    """Create and return a configured :class:`Agent`.

    Parameters
    ----------
    model_id:
        Model identifier (Groq default: ``DEFAULT_MODEL_ID``,
        Ollama default: ``DEFAULT_OLLAMA_MODEL``).
    api_key:
        Groq API key.  Falls back to the ``GROQ_API_KEY`` env-var.
    tools:
        Override the default toolkit list (useful for testing).
    instructions:
        Override the default prompt instructions.
    provider:
        ``"groq"`` or ``"ollama"``. Defaults to ``"groq"`` so the
        factory matches the project's documented behavior.

    Raises
    ------
    RuntimeError
        If no Groq API key is available when using Groq.
    """
    resolved_provider = (provider or LLM_PROVIDER).lower().strip()
    model: Any

    if resolved_provider == "ollama":
        ollama_model_id = model_id if model_id != DEFAULT_MODEL_ID else _OLLAMA_MODEL
        try:
            t0 = time.monotonic()
            model = _build_ollama_model(ollama_model_id)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            print(f"[LLM latency] provider=Ollama model={ollama_model_id} elapsed={latency_ms:.1f}ms", flush=True)
            logger.info(
                "[LLM] Using Ollama model=%s host=%s (connectivity check: %.1fms)",
                ollama_model_id, _OLLAMA_HOST, latency_ms,
            )
        except RuntimeError as exc:
            import traceback
            print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
            logger.warning("[LLM] Ollama unavailable (%s) — falling back to Groq model=%s", exc, model_id)
            model = _build_groq_model(model_id, api_key)
            logger.info("[LLM] Fallback: Using Groq model=%s", model_id)
    else:
        model = _build_groq_model(model_id, api_key)
        logger.info("[LLM] Using Groq model=%s", model_id)

    return Agent(
        model=model,
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
                t0 = time.monotonic()
                result = self._agent.run(prompt, stream=False)
                elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                raw: str = (
                    result.content if hasattr(result, "content") else str(result)
                )
                content = _strip_tool_metadata(raw)
                provider = type(self._agent.model).__name__
                logger.info("Agent run succeeded on attempt %d", attempt)
                logger.info(
                    "[LLM latency] provider=%s attempt=%d elapsed=%.1fms",
                    provider, attempt, elapsed_ms,
                )
                print(f"[LLM latency] provider={provider} elapsed={elapsed_ms:.1f}ms", flush=True)
                return content
            except Exception as exc:
                import traceback
                print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
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
        t0 = time.monotonic()
        first_chunk = True
        raw_buffer = ""
        emitted_clean = ""
        try:
            run_stream = self._agent.run(prompt, stream=True)
            for chunk in run_stream:
                text: str = ""
                if hasattr(chunk, "content") and chunk.content:
                    text = chunk.content
                elif isinstance(chunk, str):
                    text = chunk
                if text:
                    if first_chunk:
                        ttft_ms = round((time.monotonic() - t0) * 1000, 1)
                        provider = type(self._agent.model).__name__
                        logger.info(
                            "[LLM latency] provider=%s TTFT=%.1fms (stream)",
                            provider, ttft_ms,
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
            import traceback
            print(f"[ERROR DETAIL] {traceback.format_exc()}", flush=True)
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
