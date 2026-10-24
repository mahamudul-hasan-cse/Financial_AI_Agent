"""Financial AI Agent — service class and CLI.

Provides a thin ``FinancialAgentService`` that wraps the raw ``agno``
``Agent`` with retry logic, clean return types, and structured error
handling so the API layer doesn't need to know agent internals.

Retries (``LLM_MAX_RETRIES``, default 2) are handled here rather than in the
Groq SDK so streaming clients can be told about them: :meth:`stream` yields
``"[STATUS] Retrying in Ns…"`` sentinels while it waits, honouring Groq's
``Retry-After`` on 429s, and a final ``"[ERROR] …"`` when it gives up.

The primary provider is ``LLM_PROVIDER`` (default ``openrouter`` when
``OPENROUTER_API_KEY`` is set, else ``groq``).  Before any of that waiting, a
primary failure that looks like overload (429, 5xx, timeout, connection
error) is retried *immediately* on the backup chain, once per request:

* OpenRouter primary → ``OPENROUTER_FALLBACK_MODELS``, then Groq
  (``GROQ_MODEL``) when ``GROQ_API_KEY`` is set;
* Groq primary → ``OPENROUTER_MODEL`` then ``OPENROUTER_FALLBACK_MODELS``.

``FALLBACK_PROVIDER=none`` disables the chain.  Streams announce the switch
with ``"[STATUS] Primary model busy — switched to backup model"``.  Which
provider/model answered is returned on :class:`AgentReply` (``run``), as a
final ``"[MODEL_INFO] {json}"`` chunk (``stream``) and in
:attr:`FinancialAgentService.last_used`.

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
import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Generator, Iterator, Sequence

os.environ.setdefault("AGNO_TELEMETRY", "false")

import httpx  # noqa: E402
from agno.agent import Agent  # noqa: E402
from agno.models.groq import Groq  # noqa: E402
from agno.tools import Toolkit  # noqa: E402

from backend.app.agent.prompts import AGENT_INSTRUCTIONS  # noqa: E402
from backend.app.agent.tool_registry import build_tools, compact_tool_output  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_GROQ_MODEL,
    get_fallback_provider,
    get_groq_api_key,
    get_groq_model,
    get_llm_provider,
    get_openrouter_api_key,
    get_openrouter_models,
    settings,
)

logger = logging.getLogger("financial_agent")

__all__ = [  # noqa: F822
    "FinancialAgentService",
    "AgentReply",
    "AgentNotConfiguredError",
    "LLMRateLimitError",
    "STATUS_PREFIX",
    "ERROR_PREFIX",
    "MODEL_INFO_PREFIX",
    "agent_service",
    "agent",  # resolved lazily via __getattr__
    "create_agent",
    "fallback_chain",
    "fallback_status",
    "get_agent_service",
    "llm_status",
    "main",
    "primary_model_id",
    "AGENT_INSTRUCTIONS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_OLLAMA_MODEL",
    "LLM_PROVIDER",
]

# ── Model configuration ──────────────────────────────────────────────
# All values come from config.py (env vars: LLM_PROVIDER, GROQ_MODEL,
# OLLAMA_MODEL, OLLAMA_HOST).  DEFAULT_MODEL_ID is the built-in default;
# create_agent() uses GROQ_MODEL when set.
DEFAULT_MODEL_ID: str = DEFAULT_GROQ_MODEL
DEFAULT_OLLAMA_MODEL: str = settings.ollama_model
# Retries *after* the first attempt (LLM_MAX_RETRIES).
_DEFAULT_MAX_RETRIES: int = settings.llm_max_retries

# Sentinel prefixes yielded by FinancialAgentService.stream().
STATUS_PREFIX = "[STATUS]"
ERROR_PREFIX = "[ERROR]"
MODEL_INFO_PREFIX = "[MODEL_INFO]"
FALLBACK_STATUS_MESSAGE = "Primary model busy — switched to backup model"
NEXT_FALLBACK_STATUS_MESSAGE = "Backup model busy — trying another…"
GENERIC_ERROR_MESSAGE = "Unable to generate a response right now. Please try again in a moment."
RATE_LIMIT_ERROR_MESSAGE = (
    "The AI service is busy (rate limit reached). Please wait a minute and try again."
)

# Provider selected at import time ("openrouter", "groq" or "ollama"). ``create_agent``
# re-reads the environment on every call so tests and CLIs can switch it.
LLM_PROVIDER: str = settings.llm_provider
_OLLAMA_MODEL: str = settings.ollama_model
_OLLAMA_HOST: str = settings.ollama_host


class AgentNotConfiguredError(RuntimeError):
    """Raised when no usable LLM provider is configured (e.g. a missing API key)."""


class LLMRateLimitError(RuntimeError):
    """Raised by :meth:`FinancialAgentService.run` when retries end on a rate limit."""


class AgentReply(str):
    """Answer text from :meth:`FinancialAgentService.run`, plus who produced it."""

    provider: str | None
    model: str | None
    fallback: bool

    def __new__(
        cls,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        fallback: bool = False,
    ) -> "AgentReply":
        reply = super().__new__(cls, text)
        reply.provider = provider
        reply.model = model
        reply.fallback = fallback
        return reply


# ── Retry helpers ────────────────────────────────────────────────────

_RETRY_IN_RE = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE)


def _sleep(seconds: float) -> None:
    """Indirection so tests can skip real back-off waits."""
    time.sleep(seconds)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < 6:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def is_rate_limit_error(exc: BaseException) -> bool:
    """True for HTTP 429 / rate-limit errors from the provider (directly or via agno)."""
    for err in _exception_chain(exc):
        status = getattr(err, "status_code", None)
        if status is None:
            status = getattr(getattr(err, "response", None), "status_code", None)
        if status == 429 or type(err).__name__ in ("RateLimitError", "ModelRateLimitError"):
            return True
        text = str(err).lower()
        if "rate limit" in text or "rate_limit" in text or "too many requests" in text:
            return True
    return False


def is_fallback_error(exc: BaseException) -> bool:
    """True when the provider is busy or down: 429, 5xx, timeout or connection error.

    These are worth retrying on a *different* provider right away; 4xx
    errors such as a bad request would fail there too.
    """
    if is_rate_limit_error(exc):
        return True
    for err in _exception_chain(exc):
        if isinstance(err, (httpx.TransportError, ConnectionError, TimeoutError)):
            return True
        if type(err).__name__ in (
            "APIConnectionError", "APITimeoutError", "InternalServerError", "ServiceUnavailableError"
        ):
            return True
        status = getattr(err, "status_code", None)
        if status is None:
            status = getattr(getattr(err, "response", None), "status_code", None)
        if isinstance(status, int) and status >= 500:
            return True
        text = str(err).lower()
        if "unavailable" in text or "over capacity" in text or "overloaded" in text:
            return True
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server-suggested wait: ``Retry-After`` header or Groq's "try again in 7.5s"."""
    for err in _exception_chain(exc):
        headers = getattr(getattr(err, "response", None), "headers", None)
        if headers is not None:
            try:
                value = headers.get("retry-after")
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                pass
        match = _RETRY_IN_RE.search(str(err))
        if match:
            return int(match.group(1) or 0) * 60 + float(match.group(2))
    return None


def retry_delay(exc: BaseException, attempt: int) -> int:
    """Whole seconds to wait before retry number *attempt* (1-based)."""
    if is_rate_limit_error(exc):
        delay = retry_after_seconds(exc) or 2.0 ** attempt
    else:
        delay = 1.0
    return max(1, min(math.ceil(delay), settings.llm_retry_max_delay))


def retry_status_message(exc: BaseException, delay: int) -> str:
    if is_rate_limit_error(exc):
        return f"Rate limit reached — retrying in {delay}s…"
    return f"Temporary error — retrying in {delay}s…"


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

_http_client: httpx.Client | None = None
_http_client_lock = threading.Lock()


def _groq_http_client() -> httpx.Client:
    """Process-wide HTTP client for Groq (keeps connections warm between turns).

    Passing our own client also avoids noisy ``SyncHttpxClientWrapper``
    ``'_state'`` AttributeErrors: agno's memory deep-copies the model on every
    Agent, and the Groq SDK's *default* wrapper client has a ``__del__`` that
    breaks on half-copied instances.  A plain ``httpx.Client`` has none.
    """
    global _http_client
    with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
                follow_redirects=True,
            )
        return _http_client


def _build_groq_model(model_id: str, api_key: str | None) -> Groq:
    resolved_key: str | None = api_key or get_groq_api_key()
    if not resolved_key:
        raise AgentNotConfiguredError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or set LLM_PROVIDER=ollama to use a local model."
        )
    # max_retries=0: the SDK would otherwise retry 429s silently (with its own
    # back-off); FinancialAgentService retries instead and reports progress.
    return Groq(
        id=model_id,
        api_key=resolved_key,
        max_retries=0,
        http_client=_groq_http_client(),
    )


def _build_openrouter_model(model_id: str) -> Any:
    """Create an agno OpenRouter model (primary or backup) sharing the HTTP client."""
    api_key = get_openrouter_api_key()
    if not api_key:
        raise AgentNotConfiguredError(
            "OPENROUTER_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=groq "
            "(with GROQ_API_KEY) or LLM_PROVIDER=ollama."
        )
    from agno.models.openrouter import OpenRouter  # needs the `openai` package

    return OpenRouter(
        id=model_id,
        api_key=api_key,
        max_retries=0,
        # agno's OpenRouter default (1024) cuts longer analyses short.
        max_tokens=4096,
        # Upstreams can hang ~120s; give up fast and fall through to a backup.
        timeout=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30")),
        http_client=_groq_http_client(),
    )


def _provider_of(model: Any) -> str:
    """``"groq"``, ``"openrouter"``, ``"ollama"``… for an agno model (isinstance covers mocks)."""
    if isinstance(model, Groq):
        return "groq"
    try:
        from agno.models.openrouter import OpenRouter
    except ImportError:  # pragma: no cover - `openai` package missing
        OpenRouter = None  # type: ignore[assignment]  # noqa: N806
    if OpenRouter is not None and isinstance(model, OpenRouter):
        return "openrouter"
    return type(model).__name__.lower()


def _model_info(model: Any) -> dict[str, Any]:
    """``{"provider", "model", "fallback": False}`` for the primary agno model."""
    model_id = getattr(model, "id", None)
    return {
        "provider": _provider_of(model),
        "model": model_id if isinstance(model_id, str) else None,
        "fallback": False,
    }


def primary_model_id(provider: str | None = None) -> str:
    """Model id the primary *provider* (default: ``LLM_PROVIDER``) runs."""
    provider = provider or get_llm_provider()
    if provider == "openrouter":
        return get_openrouter_models()[0]
    if provider == "ollama":
        return settings.ollama_model
    return get_groq_model()


def fallback_chain(primary: str | None = None) -> list[tuple[str, str]]:
    """``(provider, model_id)`` backups, in order, for the *primary* provider.

    Empty when ``FALLBACK_PROVIDER=none``, for Ollama, or when the needed
    keys are missing.  No network calls.
    """
    primary = primary or get_llm_provider()
    if get_fallback_provider() == "none":
        return []
    chain: list[tuple[str, str]] = []
    if primary == "openrouter":
        chain += [("openrouter", m) for m in get_openrouter_models()[1:]]
        if get_groq_api_key():
            chain.append(("groq", get_groq_model()))
    elif primary == "groq" and get_openrouter_api_key():
        chain += [("openrouter", m) for m in get_openrouter_models()]
    return chain


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
    model_id: str | None,
    api_key: str | None,
) -> Any:
    """Return an agno model instance for *provider* (*model_id* None = its default).

    ``ollama`` falls back to Groq when Ollama is unreachable *and* a Groq
    key is available; otherwise the original error is raised.
    """
    if provider == "openrouter":
        model_id = primary_model_id("openrouter") if model_id is None else model_id
        model = _build_openrouter_model(model_id)
        logger.info("[LLM] Using OpenRouter model=%s", model_id)
        return model
    if provider == "ollama":
        ollama_model_id = settings.ollama_model if model_id is None else model_id
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
            model_id = get_groq_model()
            logger.warning("[LLM] %s — falling back to Groq model=%s", exc, model_id)

    model_id = get_groq_model() if model_id is None else model_id
    model = _build_groq_model(model_id, api_key)
    logger.info("[LLM] Using Groq model=%s", model_id)
    return model


def create_agent(
    *,
    model_id: str | None = None,
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
        Model identifier (default: the provider's model env-var —
        ``OPENROUTER_MODEL``, ``GROQ_MODEL`` or ``OLLAMA_MODEL``).
    api_key:
        Groq API key.  Falls back to the ``GROQ_API_KEY`` env-var.
    tools:
        Override the default toolkit list (useful for testing).
    instructions:
        Override the default prompt instructions.
    provider:
        ``"openrouter"``, ``"groq"`` or ``"ollama"``. Defaults to ``LLM_PROVIDER``
        (``openrouter`` when ``OPENROUTER_API_KEY`` is set, else ``groq``).
    model:
        A pre-built agno model to reuse (skips provider resolution).
    session_id:
        Session the agent run belongs to.  agno's own chat history is
        disabled: the API injects the (per-session) history into the prompt
        itself, so nothing is shared between sessions or users.

    Raises
    ------
    AgentNotConfiguredError
        If the chosen provider's API key is missing (Groq also when Ollama falls back to it).
    """
    if model is None:
        resolved_provider = (provider or get_llm_provider()).lower().strip()
        model = _resolve_model(
            provider=resolved_provider,
            model_id=model_id,
            api_key=api_key,
        )

    return Agent(
        model=model,
        tools=list(tools) if tools is not None else build_tools(),
        instructions=instructions if instructions is not None else AGENT_INSTRUCTIONS,
        show_tool_calls=False,
        markdown=True,
        add_history_to_messages=False,
        session_id=session_id,
        # Each tool round is another full model call: cap them, and keep tool
        # results compact (and cached) via the hook.
        tool_call_limit=settings.agent_tool_call_limit,
        tool_hooks=[compact_tool_output],
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
        max_retries: int | None = None,
        fallback_factory: Any | None = None,
    ) -> None:
        self._injected: bool = agent is not None
        self._agent: Agent | None = agent
        # Retries after the first attempt; total attempts = max_retries + 1.
        self._max_retries: int = max(
            0, _DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        )
        # (provider, model_id, session_id) -> Agent for a backup; tests inject one.
        self._fallback_factory = fallback_factory
        self.init_error: str | None = None
        self.last_used: dict[str, Any] | None = None

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

    def ensure_ready(self) -> None:
        """Build the template agent now; raises ``AgentNotConfiguredError`` if impossible."""
        if not self._injected:
            _ = self.agent

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

    # -- Fallback chain ------------------------------------------------

    @staticmethod
    def _fallback_models(agent: Agent) -> list[tuple[str, str]]:
        """``(provider, model_id)`` backups for this request (see :func:`fallback_chain`)."""
        primary = _model_info(agent.model)
        chain = fallback_chain(primary["provider"])
        # Never "fall back" to the very model that just failed.
        return [(p, m) for p, m in chain if (p, m) != (primary["provider"], primary["model"])]

    def _fallback_agent(self, provider: str, model_id: str, session_id: str | None) -> Agent:
        """An agent on backup *provider*/*model_id* with the primary's tools and instructions."""
        if self._fallback_factory is not None:
            return self._fallback_factory(provider, model_id, session_id)
        template = self.agent
        model = (
            _build_groq_model(model_id, None)
            if provider == "groq"
            else _build_openrouter_model(model_id)
        )
        return create_agent(
            model=model,
            tools=template.tools,
            instructions=template.instructions,
            session_id=session_id,
        )

    def _record(self, info: dict[str, Any]) -> None:
        self.last_used = {**info, "at": datetime.now(timezone.utc).isoformat()}

    # -- Synchronous execution -----------------------------------------

    def _run_once(self, agent: Agent, prompt: str, session_id: str | None, attempt: int) -> str:
        t0 = time.monotonic()
        result = agent.run(prompt, **self._run_kwargs(False, session_id))
        raw: str = result.content if hasattr(result, "content") else str(result)
        logger.info(
            "[LLM latency] provider=%s attempt=%d elapsed=%.1fms",
            type(agent.model).__name__,
            attempt,
            (time.monotonic() - t0) * 1000,
        )
        return _strip_tool_metadata(raw)

    def _run_fallback(
        self, models: list[tuple[str, str]], prompt: str, session_id: str | None
    ) -> AgentReply | None:
        """Try each backup model in order; ``None`` if all of them fail."""
        for provider, model_id in models:
            try:
                fb_agent = self._fallback_agent(provider, model_id, session_id)
                content = self._run_once(fb_agent, prompt, session_id, 1)
            except Exception as exc:
                logger.warning(
                    "[LLM] Backup %s model %s failed: %s", provider, model_id, str(exc)[:200]
                )
                continue
            info = {"provider": provider, "model": model_id, "fallback": True}
            self._record(info)
            logger.info("[LLM] Answered by backup %s model=%s", info["provider"], info["model"])
            return AgentReply(content, **info)
        return None

    def run(self, prompt: str, session_id: str | None = None) -> AgentReply:
        """Execute the agent and return the full response text.

        On a primary 429/5xx/timeout/connection error the request first goes
        to the backup chain (once, without waiting); if every backup fails
        too, the primary is retried up to ``max_retries`` times (rate limits
        wait for ``Retry-After``, capped; other errors wait 1s).

        Returns an :class:`AgentReply` (a ``str``) carrying the provider/model.

        Raises
        ------
        AgentNotConfiguredError
            If no LLM provider is configured (not retried).
        LLMRateLimitError
            If the last attempt failed on a rate limit.
        RuntimeError
            If all retry attempts are exhausted for another reason.
        """
        agent = self._agent_for(session_id)
        fallback_models = self._fallback_models(agent)
        attempts = self._max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                content = self._run_once(agent, prompt, session_id, attempt)
                logger.info("Agent run succeeded on attempt %d", attempt)
                info = _model_info(agent.model)
                self._record(info)
                return AgentReply(content, **info)
            except AgentNotConfiguredError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Agent attempt %d/%d failed: %s",
                    attempt,
                    attempts,
                    str(exc)[:200],
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
                if fallback_models and is_fallback_error(exc):
                    # Backups are tried once per request, before any back-off.
                    models, fallback_models = fallback_models, []
                    logger.warning("[LLM] Primary model unavailable — switching to backup models")
                    reply = self._run_fallback(models, prompt, session_id)
                    if reply is not None:
                        return reply
                if attempt < attempts:
                    _sleep(retry_delay(exc, attempt))
        if last_error is not None and is_rate_limit_error(last_error):
            raise LLMRateLimitError(RATE_LIMIT_ERROR_MESSAGE) from last_error
        raise RuntimeError(f"Agent failed after {attempts} attempts") from last_error

    # -- Streaming execution -------------------------------------------

    def _stream_once(
        self, agent: Agent, prompt: str, session_id: str | None, attempt: int
    ) -> Iterator[str]:
        """Yield cleaned text deltas from one streamed run; raises on failure."""
        t0 = time.monotonic()
        raw_buffer = ""
        emitted_clean = ""
        for chunk in agent.run(prompt, **self._run_kwargs(True, session_id)):
            if getattr(chunk, "event", None) == "RunError":
                raise RuntimeError(getattr(chunk, "content", None) or "agent run error")
            text: str = ""
            if hasattr(chunk, "content") and chunk.content:
                text = chunk.content
            elif isinstance(chunk, str):
                text = chunk
            if not text:
                continue
            if not raw_buffer:
                logger.info(
                    "[LLM latency] provider=%s attempt=%d TTFT=%.1fms (stream)",
                    type(agent.model).__name__,
                    attempt,
                    (time.monotonic() - t0) * 1000,
                )
            raw_buffer += text
            cleaned = _strip_tool_metadata_streaming(raw_buffer)
            if cleaned.startswith(emitted_clean):
                delta = cleaned[len(emitted_clean):]
            else:
                delta = cleaned
            if delta:
                emitted_clean = cleaned
                yield delta

    def _model_info_chunk(self, info: dict[str, Any]) -> str:
        self._record(info)
        return f"{MODEL_INFO_PREFIX} {json.dumps(info)}"

    def stream(self, prompt: str, session_id: str | None = None) -> Generator[str, None, None]:
        """Execute the agent in streaming mode, yielding text chunks.

        Besides answer text, three kinds of sentinel chunks may be yielded:

        * ``"[STATUS] <message>"`` – a failure happened *before* any text was
          produced: either the request moved to a backup model
          (``FALLBACK_STATUS_MESSAGE``) or it will be retried after the
          stated wait (at most ``max_retries`` times).
        * ``"[ERROR] <message>"`` – generation failed for good; the message is
          safe to show to users.  Always the last chunk.
        * ``"[MODEL_INFO] {json}"`` – after a successful answer: which
          ``provider``/``model`` produced it and whether it was a ``fallback``.
          Always the last chunk.

        Failures after partial output are not retried (that would duplicate
        text); they end with ``[ERROR]``.

        Raises
        ------
        AgentNotConfiguredError
            Before the first chunk, if no LLM provider is configured.
        """
        agent = self._agent_for(session_id)
        fallback_models = self._fallback_models(agent)
        attempts = self._max_retries + 1
        emitted_any = False
        for attempt in range(1, attempts + 1):
            try:
                for delta in self._stream_once(agent, prompt, session_id, attempt):
                    emitted_any = True
                    yield delta
                yield self._model_info_chunk(_model_info(agent.model))
                return
            except Exception as exc:
                rate_limited = is_rate_limit_error(exc)
                if not emitted_any and fallback_models and is_fallback_error(exc):
                    # Backups are tried once per request, before any back-off.
                    models, fallback_models = fallback_models, []
                    logger.warning(
                        "Stream attempt %d failed (%s); switching to backup models",
                        attempt,
                        str(exc)[:200],
                    )
                    yield f"{STATUS_PREFIX} {FALLBACK_STATUS_MESSAGE}"
                    for index, (provider, model_id) in enumerate(models):
                        try:
                            fb_agent = self._fallback_agent(provider, model_id, session_id)
                            for delta in self._stream_once(fb_agent, prompt, session_id, 1):
                                emitted_any = True
                                yield delta
                        except Exception as fb_exc:
                            logger.warning(
                                "[LLM] Backup %s model %s failed: %s",
                                provider,
                                model_id,
                                str(fb_exc)[:200],
                            )
                            if emitted_any:
                                yield f"{ERROR_PREFIX} {GENERIC_ERROR_MESSAGE}"
                                return
                            if index + 1 < len(models):
                                yield f"{STATUS_PREFIX} {NEXT_FALLBACK_STATUS_MESSAGE}"
                            continue
                        yield self._model_info_chunk(
                            {"provider": provider, "model": model_id, "fallback": True}
                        )
                        return
                    # Every backup failed: carry on with the primary's retry path.
                if emitted_any or attempt >= attempts:
                    logger.exception("Stream error (attempt %d/%d)", attempt, attempts)
                    message = RATE_LIMIT_ERROR_MESSAGE if rate_limited else GENERIC_ERROR_MESSAGE
                    yield f"{ERROR_PREFIX} {message}"
                    return
                delay = retry_delay(exc, attempt)
                logger.warning(
                    "Stream attempt %d/%d failed (%s); retrying in %ds",
                    attempt,
                    attempts,
                    str(exc)[:200],
                    delay,
                )
                yield f"{STATUS_PREFIX} {retry_status_message(exc, delay)}"
                _sleep(delay)


def fallback_status() -> dict[str, Any]:
    """Describe the backup chain for the primary provider (no network calls).

    ``chain`` lists ``{"provider", "model"}`` in the order they are tried;
    ``provider``/``models`` summarise it (``provider`` is ``"none"`` when
    the chain is empty or disabled).
    """
    primary = get_llm_provider()
    if get_fallback_provider() == "none":
        return {"provider": "none", "configured": False, "models": [], "chain": [], "detail": None}
    chain = fallback_chain(primary)
    if not chain:
        detail = None
        if primary == "groq" and not get_openrouter_api_key():
            detail = "OPENROUTER_API_KEY is not set."
        return {
            "provider": "openrouter" if detail else "none",
            "configured": False,
            "models": [],
            "chain": [],
            "detail": detail,
        }
    providers = list(dict.fromkeys(p for p, _ in chain))
    return {
        "provider": ",".join(providers),
        "configured": True,
        "models": [m for _, m in chain],
        "chain": [{"provider": p, "model": m} for p, m in chain],
        "detail": None,
    }


_PROVIDER_KEYS = {"openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY"}


def llm_status() -> dict[str, Any]:
    """Describe the primary LLM configuration without building the agent.

    Returns ``{"provider", "model", "configured": bool, "detail": str | None}``.
    """
    provider = get_llm_provider()
    status: dict[str, Any] = {
        "provider": provider,
        "model": primary_model_id(provider),
        "configured": True,
        "detail": None,
    }
    if agent_service.init_error:
        return {**status, "configured": False, "detail": agent_service.init_error}
    key_name = _PROVIDER_KEYS.get(provider)
    if key_name and not os.getenv(key_name):
        return {**status, "configured": False, "detail": f"{key_name} is not set."}
    return status


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
