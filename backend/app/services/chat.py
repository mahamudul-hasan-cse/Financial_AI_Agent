"""Chat orchestration: NLP preprocessing, prompt building, agent calls."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import cache
from backend.app.agent.financial_agent import (
    ERROR_PREFIX,
    GENERIC_ERROR_MESSAGE,
    MODEL_INFO_PREFIX,
    STATUS_PREFIX,
    AgentNotConfiguredError,
    LLMRateLimitError,
    get_agent_service,
)
from backend.app.agent.prompts import get_intent_instruction
from backend.app.state import _stats
from nlp_pipeline import NLPMetadata, run_nlp_pipeline
from schemas import NLPMetadataSchema

logger = logging.getLogger("api.chat")

__all__ = [
    "AgentNotConfiguredError",
    "ERROR_PREFIX",
    "GENERIC_ERROR_MESSAGE",
    "LLMRateLimitError",
    "MODEL_INFO_PREFIX",
    "STATUS_PREFIX",
    "agent",
    "append_disclaimer",
    "build_prompt",
    "clean_response",
    "get_news_sentiment_context",
    "process_nlp_and_track",
]

_SENTIMENT_INTENTS: frozenset[str] = frozenset({"stock_price", "recommendation", "news"})
_SENTIMENT_TTL_SECONDS = 600

# Section labels the model sometimes emits at the start of a line
# ("Quick Take: ...").  Only line-leading labels are stripped so that
# legitimate text and table headers containing these words survive.
_LABELS = (
    "Quick Take",
    "Key Data",
    "What It Means",
    "Risk",
    "Next Steps",
    "Summary",
    "Overview",
    "Analysis",
    "Verdict",
    "Bottom Line",
    "Insight",
    "Context",
    "Recommendation",
)
_LABEL_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?:(?:#+|>|[-*])[ \t]+)?)(?:\*\*)?(?:"
    + "|".join(map(re.escape, _LABELS))
    + r"):(?:\*\*)?[ \t]*",
    re.MULTILINE,
)


def agent():
    """Return the agent service (patched in tests)."""
    return get_agent_service()


def clean_response(text: str) -> str:
    """Tidy an agent reply before it is persisted and shown as the final text."""
    cleaned = _LABEL_RE.sub(lambda m: m.group("prefix"), text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    logger.debug("clean_response: length before=%d after=%d", len(text), len(cleaned))
    return cleaned


def append_disclaimer(content: str) -> str:
    """Append the educational disclaimer to a non-empty financial response."""
    if not content.strip():
        return content
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    disclaimer = f"\n\n*Data as of {ts}. For educational purposes only — not financial advice.*"
    if "For educational purposes only — not financial advice." in content:
        return content
    return content + disclaimer


# Prompt context budget: the history is re-sent on every model round of a
# turn, so keep only recent messages and clip long past answers.
_CONTEXT_MESSAGES = 6
_CONTEXT_MESSAGE_CHARS = 600
_DISCLAIMER_RE = re.compile(r"\s*\*Data as of [^*]*not financial advice\.\*\s*$")


def _build_context(history: list[dict[str, str]]) -> str:
    """Build a compact conversation context from the last few messages."""
    lines: list[str] = []
    for msg in history[-_CONTEXT_MESSAGES:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = _DISCLAIMER_RE.sub("", msg["content"])
        if len(content) > _CONTEXT_MESSAGE_CHARS:
            content = content[:_CONTEXT_MESSAGE_CHARS].rstrip() + " …"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_prompt(
    user_message: str,
    history: list[dict[str, str]],
    intent: str = "general",
    sentiment_context: str | None = None,
) -> str:
    """Construct the full prompt: intent directive, sentiment, history, message.

    *history* is this session's own history, so context never crosses sessions.
    """
    parts: list[str] = [f"[INTENT: {intent.upper()}]\n{get_intent_instruction(intent)}"]
    if sentiment_context:
        parts.append(f"[MARKET SENTIMENT]\n{sentiment_context}")
    context = _build_context(history)
    if context:
        parts.append(f"Previous conversation:\n{context}")
    parts.append(f"User: {user_message}")
    return "\n\n".join(parts)


def _first_ticker(nlp_meta: NLPMetadataSchema) -> str | None:
    for ent in nlp_meta.entities:
        if ent.ticker:
            return ent.ticker
    return None


def get_news_sentiment_context(nlp_meta: NLPMetadataSchema) -> str | None:
    """VADER aggregate over recent headlines for price/recommendation/news intents.

    Returns e.g. ``"Market Sentiment: Bullish (+0.42) based on 5 recent headlines"``
    or ``None``.  Results are cached per ticker for 10 minutes.
    """
    if nlp_meta.intent not in _SENTIMENT_INTENTS:
        return None

    from nlp_pipeline import VADER_AVAILABLE, analyze_sentiment

    if not VADER_AVAILABLE:
        return None
    ticker = _first_ticker(nlp_meta)
    if not ticker:
        return None

    cache_key = f"news_sentiment:{ticker.upper()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached or None

    result = ""
    try:
        from market_service import get_recent_headlines

        headlines = [h["title"] for h in get_recent_headlines(ticker, limit=5)]
        if headlines:
            scores = [analyze_sentiment(h)["compound"] for h in headlines]
            avg = sum(scores) / len(scores)
            label = "Bullish" if avg >= 0.1 else "Bearish" if avg <= -0.1 else "Neutral"
            result = (
                f"Market Sentiment: {label} ({avg:+.2f}) based on "
                f"{len(headlines)} recent headlines"
            )
            logger.info("VADER sentiment for %s: %s (avg=%.2f)", ticker, label, avg)
    except Exception:
        logger.warning("News sentiment fetch failed for %s", ticker, exc_info=True)
    cache.set(cache_key, result, ttl=_SENTIMENT_TTL_SECONDS)
    return result or None


def _default_nlp_metadata() -> NLPMetadataSchema:
    """Return safe default NLP metadata when the pipeline fails."""
    return NLPMetadataSchema.model_validate(
        {
            "entities": [],
            "intent": "general",
            "intent_confidence": 0.0,
            "intent_method": "fallback",
            "sentiment": {"label": "unavailable", "score": 0.0, "compound": 0.0},
            "spacy_available": False,
            "vader_available": False,
        }
    )


def process_nlp_and_track(message: str, session_id: str) -> NLPMetadataSchema:
    """Run the NLP pipeline on *message* and update analytics counters.

    Never raises: on pipeline failure, logs and returns safe defaults.
    """
    _stats["total_messages"] += 1
    try:
        nlp_meta: NLPMetadata = run_nlp_pipeline(message)
    except Exception:
        logger.exception("NLP pipeline failed for session %s — using defaults", session_id)
        _stats["intent_counts"]["general"] += 1
        _stats["classifier_usage"]["fallback"] += 1
        return _default_nlp_metadata()

    _stats["intent_counts"][nlp_meta["intent"]] += 1
    _stats["classifier_usage"][nlp_meta.get("intent_method", "keyword")] += 1
    logger.info(
        "NLP [%s] intent=%s (%.0f%% %s) entities=%d sentiment=%s",
        session_id,
        nlp_meta["intent"],
        nlp_meta.get("intent_confidence", 0) * 100,
        nlp_meta.get("intent_method", "keyword"),
        len(nlp_meta["entities"]),
        nlp_meta["sentiment"]["label"],
    )
    return NLPMetadataSchema.model_validate(dict(nlp_meta))


def persist_exchange(
    session_id: str,
    history: list[dict[str, str]],
    user_message: str,
    assistant_message: str,
    nlp_meta: NLPMetadataSchema,
    structured: dict[str, Any] | None,
) -> None:
    """Append a user/assistant pair to the in-memory history and SQLite."""
    import persistence

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_message})
    persistence.append_message(
        session_id, role="user", content=user_message, nlp_metadata=nlp_meta.model_dump()
    )
    persistence.append_message(
        session_id,
        role="assistant",
        content=assistant_message,
        structured_response=structured,
    )
