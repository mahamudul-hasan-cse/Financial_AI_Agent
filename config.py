"""Runtime configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


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
    primary_llm_provider: str


def load_settings() -> Settings:
    """Load validated settings for the API process."""

    app_dir = Path(__file__).resolve().parent
    outputs_dir = app_dir / "outputs"
    db_path = os.getenv("SQLITE_PATH", str(app_dir / "data" / "financial_agent.db"))
    cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]

    return Settings(
        app_dir=app_dir,
        outputs_dir=outputs_dir,
        sqlite_path=Path(db_path),
        api_key=os.getenv("API_KEY"),
        cors_origins=cors_origins,
        agent_timeout=int(os.getenv("AGENT_TIMEOUT", "120")),
        rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        rate_limit_max=int(os.getenv("RATE_LIMIT_MAX", "10")),
        ip_rate_limit_max=int(os.getenv("IP_RATE_LIMIT_MAX", "30")),
        session_ttl=int(os.getenv("SESSION_TTL", "1800")),
        max_sessions=int(os.getenv("MAX_SESSIONS", "100")),
        primary_llm_provider=os.getenv("PRIMARY_LLM_PROVIDER", "groq").strip().lower(),
    )


settings = load_settings()

