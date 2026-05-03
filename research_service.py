"""Investor-focused formatting helpers for structured answer cards."""

from __future__ import annotations

import re
from typing import Any

from market_service import extract_symbols_from_entities, get_market_context

_ARTIFACT_RE = re.compile(r"([a-zA-Z0-9_.-]+\.(?:png|xlsx|pdf))")


def _first_non_empty_paragraph(text: str) -> str:
    """Return the first meaningful paragraph from agent output."""

    for chunk in re.split(r"\n\s*\n", text):
        cleaned = chunk.strip()
        if cleaned and not cleaned.startswith("|"):
            return cleaned
    return text.strip()


def _extract_risk_sentences(text: str) -> list[str]:
    """Find a few risk-like sentences from the raw answer."""

    sentences = re.split(r"(?<=[.!?])\s+", text)
    keywords = ("risk", "volatil", "uncertain", "pressure", "weak", "bear", "caution", "headwind")
    picked = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            picked.append(sentence.strip())
        if len(picked) >= 3:
            break
    return picked


def _default_next_actions(intent: str, symbol: str | None) -> list[str]:
    """Choose next actions based on the detected intent."""

    symbol_text = symbol or "this company"
    if intent == "comparison":
        return [
            "Review the dedicated comparison table before making a decision.",
            f"Open the company snapshot for the strongest candidate after comparing {symbol_text}.",
        ]
    if intent == "news":
        return [
            f"Track whether the latest headlines change the trend for {symbol_text}.",
            "Set a breaking news alert if you want push-style follow-up.",
        ]
    return [
        f"Open the snapshot page for {symbol_text} to review price, fundamentals, and headlines together.",
        "Save this session so you can revisit the research later.",
        "Add an alert if you want the app to notify you about important changes.",
    ]


def build_structured_response(
    user_message: str,
    raw_content: str,
    nlp_meta: Any,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a consistent investor answer card from the raw agent output."""

    warnings = warnings or []
    symbols = extract_symbols_from_entities(nlp_meta.entities if hasattr(nlp_meta, "entities") else [])
    symbol = symbols[0] if symbols else None
    market = None
    company_name = None
    metrics = []
    news_signals = []
    sources = [{"label": "AI synthesis", "kind": "llm"}]
    explainability = [
        {"label": "Intent", "value": getattr(nlp_meta, "intent", "general")},
        {
            "label": "Confidence",
            "value": f"{round(getattr(nlp_meta, 'intent_confidence', 0.0) * 100)}%",
        },
        {
            "label": "Entities",
            "value": ", ".join(
                [
                    (entity.ticker or entity.text)
                    for entity in getattr(nlp_meta, "entities", [])
                ]
            ) or "None detected",
        },
        {"label": "Pipeline", "value": "NLP + market data + LLM answer synthesis"},
    ]

    if symbol:
        try:
            market = get_market_context(symbol)
        except Exception:
            warnings.append(f"Live market context was unavailable for {symbol}.")
            market = None

    if market:
        company_name = market["company_name"]
        metrics = [
            {"label": "Current Price", "value": f"${market['price']:.2f}" if market["price"] is not None else "n/a", "tone": "neutral"},
            {"label": "Today", "value": f"{market['change_pct']:+.2f}%" if market["change_pct"] is not None else "n/a", "tone": "positive" if (market["change_pct"] or 0) >= 0 else "negative"},
            {"label": "1M Trend", "value": f"{market['week_change_pct']:+.2f}%" if market["week_change_pct"] is not None else "n/a", "tone": "positive" if (market["week_change_pct"] or 0) >= 0 else "negative"},
        ]
        metrics.extend(market["fundamentals"][:3])
        news_signals = [
            {
                "title": item["title"],
                "detail": item.get("publisher") or "Recent market headline",
                "tone": "neutral",
            }
            for item in market["headlines"][:3]
        ]
        sources.extend(market["sources"])
        sources.extend(
            [
                {
                    "label": item["title"],
                    "url": item.get("url"),
                    "kind": "news",
                }
                for item in market["headlines"][:3]
            ]
        )

    risk_sentences = _extract_risk_sentences(raw_content)
    risks = [
        {"title": "Risk", "detail": sentence, "tone": "risk"}
        for sentence in risk_sentences
    ]
    if not risks:
        risks = [
            {
                "title": "Execution risk",
                "detail": "Treat the answer as research support rather than financial advice, especially when market conditions move quickly.",
                "tone": "risk",
            }
        ]

    artifacts = list(dict.fromkeys(_ARTIFACT_RE.findall(raw_content)))
    intent = getattr(nlp_meta, "intent", "general")
    title_prefix = company_name or symbol or "Market research"
    title = (
        f"{title_prefix} comparison brief"
        if intent == "comparison"
        else f"{title_prefix} research brief"
    )

    return {
        "title": title,
        "summary": _first_non_empty_paragraph(raw_content) or user_message,
        "status": "partial" if warnings else "ok",
        "symbol": symbol,
        "company_name": company_name,
        "intent": intent,
        "summary_markdown": raw_content,
        "metrics": metrics,
        "news_signals": news_signals,
        "risks": risks,
        "next_actions": _default_next_actions(intent, symbol),
        "sources": sources[:8],
        "explainability": explainability,
        "artifacts": artifacts,
        "warnings": warnings,
    }

