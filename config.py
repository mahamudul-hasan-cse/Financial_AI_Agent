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

APP_DIR = Path(__file__).resolve().parent

DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SUPPORTED_LLM_PROVIDERS = ("groq", "ollama")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def get_llm_provider() -> str:
    """Return the configured LLM provider (``groq`` or ``ollama``).

    Unknown values fall back to ``groq``.
    """
    provider = _env("LLM_PROVIDER", "groq").lower()
    return provider if provider in SUPPORTED_LLM_PROVIDERS else "groq"


def get_groq_api_key() -> str | None:
    """Return the Groq API key, or ``None`` when unset/empty."""
    return os.getenv("GROQ_API_KEY") or None


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

    @property
    def primary_llm_provider(self) -> str:
        """Backward-compatible alias for :attr:`llm_provider`."""
        return self.llm_provider


def load_settings() -> Settings:
    """Load validated settings for the API process."""

    outputs_dir = Path(_env("OUTPUTS_DIR", str(APP_DIR / "outputs")))
    # DATABASE_PATH is the documented name; SQLITE_PATH is kept for compatibility.
    db_path = os.getenv("DATABASE_PATH") or os.getenv("SQLITE_PATH") or str(
        APP_DIR / "data" / "financial_agent.db"
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
        session_ttl=int(_env("SESSION_TTL", "1800")),
        max_sessions=int(_env("MAX_SESSIONS", "100")),
        llm_provider=get_llm_provider(),
        groq_model=_env("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        ollama_model=_env("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        ollama_host=_env("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/"),
        spacy_model=_env("SPACY_MODEL", DEFAULT_SPACY_MODEL),
        embedding_model=_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        host=_env("HOST", "0.0.0.0"),
        port=int(_env("PORT", "8001")),
        forwarded_allow_ips=_env("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )


settings = load_settings()
