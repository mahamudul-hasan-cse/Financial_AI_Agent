"""Health and analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.agent.financial_agent import fallback_status, get_agent_service, llm_status
from backend.app.state import _stats, sessions
from config import get_groq_api_key, get_openrouter_api_key
from schemas import (
    FallbackStatus,
    HealthResponse,
    LastLLMUsage,
    NLPStatusDetail,
    StatsResponse,
)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + readiness: ``status`` is ``degraded`` when no LLM is usable."""
    from nlp_pipeline import SPACY_AVAILABLE, VADER_AVAILABLE

    llm = llm_status()
    last_used = get_agent_service().last_used
    return HealthResponse(
        status="ok" if llm["configured"] else "degraded",
        llm_provider=llm["provider"],
        llm_model=llm["model"],
        llm_configured=llm["configured"],
        detail=llm["detail"],
        groq_configured=bool(get_groq_api_key()),
        openrouter_configured=bool(get_openrouter_api_key()),
        fallback=FallbackStatus(**fallback_status()),
        last_llm=LastLLMUsage(**last_used) if last_used else None,
        active_sessions=len(sessions),
        total_messages=_stats["total_messages"],
        nlp_status=NLPStatusDetail(spacy_ner=SPACY_AVAILABLE, vader_sentiment=VADER_AVAILABLE),
    )


@router.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Usage statistics for the current server process (reset on restart)."""
    times = _stats["response_times_ms"]
    avg_ms = round(sum(times) / len(times), 1) if times else 0.0
    return StatsResponse(
        total_messages=_stats["total_messages"],
        active_sessions=len(sessions),
        total_sessions_created=_stats["total_sessions_created"],
        intent_distribution=dict(_stats["intent_counts"]),
        avg_response_time_ms=avg_ms,
        classifier_usage=dict(_stats["classifier_usage"]),
    )
