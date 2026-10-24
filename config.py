"""Runtime configuration helpers.

All environment variables the application reads are resolved here so the
rest of the code base has one place to look.  ``.env`` is loaded with
``override=False`` so real environment variables (Docker, PaaS dashboards,
CI) always win over values in a local ``.env`` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)

# agno phones home with usage telemetry unless told otherwise. Default it off
# here (after .env is loaded, so an explicit AGNO_TELEMETRY=true still wins);
# backend.app.agent and tools/ repeat this before their own agno imports.
os.environ.setdefault("AGNO_TELEMETRY", "false")

APP_DIR = Path(__file__).resolve().parent

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# Groq models this app is tuned for (GROQ_MODEL may name any Groq model).
# gpt-oss-20b is the fastest; qwen3.8-27b sits between the two.
KNOWN_GROQ_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUPPORTED_LLM_PROVIDERS = ("openrouter", "groq", "ollama")
SUPPORTED_FALLBACK_PROVIDERS = ("openrouter", "none")
# OpenRouter models (paid; tool calling tested). OPENROUTER_MODEL is the
# primary when LLM_PROVIDER=openrouter and the first backup when it is groq.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b"
DEFAULT_OPENROUTER_FALLBACK_MODELS = "google/gemini-2.5-flash"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    """Integer env-var; falls back to *default* when unset or not a number."""
    try:
        return max(minimum, int(_env(name, str(default))))
    except ValueError:
        return default


def default_llm_provider() -> str:
    """``openrouter`` when ``OPENROUTER_API_KEY`` is set, else ``groq``."""
    return "openrouter" if get_openrouter_api_key() else "groq"


def get_llm_provider() -> str:
    """Return the primary LLM provider (``openrouter``, ``groq`` or ``ollama``).

    ``LLM_PROVIDER`` picks it; when unset (or unknown) it defaults to
    ``openrouter`` if ``OPENROUTER_API_KEY`` is set and ``groq`` otherwise.
    """
    provider = _env("LLM_PROVIDER", "").lower()
    return provider if provider in SUPPORTED_LLM_PROVIDERS else default_llm_provider()


def get_groq_model() -> str:
    """Return the Groq model id (``GROQ_MODEL``, default ``DEFAULT_GROQ_MODEL``)."""
    return _env("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def get_groq_api_key() -> str | None:
    """Return the Groq API key, or ``None`` when unset/empty."""
    return os.getenv("GROQ_API_KEY") or None


def get_openrouter_api_key() -> str | None:
    """Return the OpenRouter API key, or ``None`` when unset/empty."""
    return os.getenv("OPENROUTER_API_KEY") or None


def get_fallback_provider() -> str:
    """Return whether backup models are used when the primary is busy or down.

    ``FALLBACK_PROVIDER`` defaults to ``openrouter`` when ``OPENROUTER_API_KEY``
    is set and ``none`` otherwise; ``none`` (or any unknown value) disables the
    whole backup chain (OpenRouter backups and, for an OpenRouter primary, Groq).
    """
    default = "openrouter" if get_openrouter_api_key() else "none"
    provider = _env("FALLBACK_PROVIDER", default).lower()
    return provider if provider in SUPPORTED_FALLBACK_PROVIDERS else "none"


def get_openrouter_models() -> list[str]:
    """OpenRouter model ids to try, in order: ``OPENROUTER_MODEL`` first, then
    ``OPENROUTER_FALLBACK_MODELS`` (comma-separated; empty string = none)."""
    models = [_env("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)]
    extra = os.getenv("OPENROUTER_FALLBACK_MODELS")
    if extra is None:
        extra = DEFAULT_OPENROUTER_FALLBACK_MODELS
    for model in extra.split(","):
        model = model.strip()
        if model and model not in models:
            models.append(model)
    return models


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_dir: Path
    outputs_dir: Path
    sqlite_path: Path
    api_key: str | None
    cors_origins: list[str]
    agent_timeout: int
    rate_limit_window: int
    rate_limit_max: int
    ip_rate_limit_max: int
    rate_limit_per_minute: int
    session_ttl: int
    max_sessions: int
    llm_provider: str
    groq_model: str
    ollama_model: str
    ollama_host: str
    spacy_model: str
    embedding_model: str
    host: str
    port: int
    forwarded_allow_ips: str
    log_level: str
    llm_max_retries: int
    llm_retry_max_delay: int
    agent_tool_call_limit: int
    tool_output_max_chars: int
    cache_ttl_seconds: int

    @property
    def primary_llm_provider(self) -> str:
        """Backward-compatible alias for :attr:`llm_provider`."""
        return self.llm_provider


def load_settings() -> Settings:
    """Load validated settings for the API process."""

    outputs_dir = Path(_env("OUTPUTS_DIR", str(APP_DIR / "outputs")))
    # DATABASE_PATH is the documented name; SQLITE_PATH is kept for compatibility.
    db_path = (
        os.getenv("DATABASE_PATH")
        or os.getenv("SQLITE_PATH")
        or str(APP_DIR / "data" / "financial_agent.db")
    )
    cors_origins_raw = _env("CORS_ORIGINS", "http://localhost:5173")
    cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]

    return Settings(
        app_dir=APP_DIR,
        outputs_dir=outputs_dir,
        sqlite_path=Path(db_path),
        api_key=os.getenv("API_KEY") or None,
        cors_origins=cors_origins,
        agent_timeout=int(_env("AGENT_TIMEOUT", "120")),
        rate_limit_window=int(_env("RATE_LIMIT_WINDOW", "60")),
        rate_limit_max=int(_env("RATE_LIMIT_MAX", "10")),
        ip_rate_limit_max=int(_env("IP_RATE_LIMIT_MAX", "30")),
        rate_limit_per_minute=_env_int("RATE_LIMIT_PER_MINUTE", 20),
        session_ttl=int(_env("SESSION_TTL", "1800")),
        max_sessions=int(_env("MAX_SESSIONS", "100")),
        llm_provider=get_llm_provider(),
        groq_model=get_groq_model(),
        ollama_model=_env("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        ollama_host=_env("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/"),
        spacy_model=_env("SPACY_MODEL", DEFAULT_SPACY_MODEL),
        embedding_model=_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "8001")),
        forwarded_allow_ips=_env("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        llm_retry_max_delay=_env_int("LLM_RETRY_MAX_DELAY", 20, minimum=1),
        agent_tool_call_limit=_env_int("AGENT_TOOL_CALL_LIMIT", 3, minimum=1),
        tool_output_max_chars=_env_int("TOOL_OUTPUT_MAX_CHARS", 2000, minimum=200),
        cache_ttl_seconds=_env_int("CACHE_TTL_SECONDS", 60),
    )


settings = load_settings()
