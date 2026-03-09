"""Ablation study for the NLP pipeline.

Systematically disables individual NLP components and measures the impact
on pipeline quality. This demonstrates each component's contribution and
validates that the full pipeline is greater than the sum of its parts.

Ablation conditions:
  1. Full pipeline (all components enabled)
  2. No NER (entity extraction disabled)
  3. No intent classifier (falls back to "general")
  4. No sentiment analysis (VADER disabled)
  5. No RAG (retrieval disabled)
  6. Keyword-only intent (ML classifier disabled, keyword fallback)

Metrics per condition:
  - Intent accuracy on the gold-standard test set
  - Entity recall on the NER test set
  - Sentiment coverage (% of queries with non-"unavailable" sentiment)
  - RAG retrieval quality (average top-1 similarity score)

Run directly: python ablation_study.py
"""

from __future__ import annotations

import sys
from typing import Callable


def _intent_accuracy(classify_fn: Callable[[str], str]) -> float:
    """Compute intent accuracy using a given classify function."""
    from evaluation import INTENT_TEST_SET
    correct = sum(1 for text, label in INTENT_TEST_SET if classify_fn(text) == label)
    return round(correct / len(INTENT_TEST_SET), 4) if INTENT_TEST_SET else 0.0


def _entity_recall(extract_fn: Callable[[str], list[dict]]) -> float:
    """Compute entity recall using a given extraction function."""
    from evaluation import NER_TEST_SET
    total_expected = 0
    total_found = 0
    for text, expected in NER_TEST_SET:
        predicted = extract_fn(text)
        expected_set = {(e["text"], e["label"]) for e in expected}
        predicted_set = {(e["text"], e["label"]) for e in predicted}
        total_expected += len(expected_set)
        total_found += len(expected_set & predicted_set)
    return round(total_found / total_expected, 4) if total_expected > 0 else 0.0


def _sentiment_coverage(analyze_fn: Callable[[str], dict]) -> float:
    """Compute fraction of queries that get a real sentiment label."""
    from evaluation import INTENT_TEST_SET
    texts = [t for t, _ in INTENT_TEST_SET]
    available = sum(1 for t in texts if analyze_fn(t)["label"] != "unavailable")
    return round(available / len(texts), 4) if texts else 0.0


def _rag_quality() -> float:
    """Average top-1 retrieval score on a set of finance queries."""
    try:
        from tools.rag_tools import _retriever
    except Exception:
        return 0.0

    test_queries = [
        "What is the P/E ratio?",
        "How does dividend yield work?",
        "Explain RSI indicator",
        "What is market capitalization?",
        "How do moving averages work?",
        "What is beta in stocks?",
        "Explain earnings per share",
        "What is a short squeeze?",
        "How do interest rates affect stocks?",
        "What are the S&P 500 sectors?",
    ]
    scores = []
    for q in test_queries:
        chunks = _retriever.retrieve(q, top_k=1)
        if chunks:
            scores.append(chunks[0].score)
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def run_ablation_study() -> list[dict]:
    """Run all ablation conditions and return results."""
    import nlp_pipeline
    from nlp_pipeline import (
        extract_entities,
        classify_intent as keyword_classify,
        analyze_sentiment,
    )

    results: list[dict] = []

    # 1. Full pipeline (ML intent + enhanced NER + sentiment + RAG)
    try:
        from intent_classifier import predict_intent
        full_classify = lambda t: predict_intent(t)[0]
        ml_available = True
    except Exception:
        full_classify = keyword_classify
        ml_available = False

    results.append({
        "condition": "Full pipeline",
        "intent_accuracy": _intent_accuracy(full_classify),
        "entity_recall": _entity_recall(extract_entities),
        "sentiment_coverage": _sentiment_coverage(analyze_sentiment),
        "rag_quality": _rag_quality(),
    })

    # 2. No NER — replace extract_entities with a no-op
    no_ner_fn = lambda text: []
    results.append({
        "condition": "No NER",
        "intent_accuracy": _intent_accuracy(full_classify),
        "entity_recall": _entity_recall(no_ner_fn),
        "sentiment_coverage": _sentiment_coverage(analyze_sentiment),
        "rag_quality": _rag_quality(),
    })

    # 3. No intent — always return "general"
    no_intent_fn = lambda text: "general"
    results.append({
        "condition": "No intent classifier",
        "intent_accuracy": _intent_accuracy(no_intent_fn),
        "entity_recall": _entity_recall(extract_entities),
        "sentiment_coverage": _sentiment_coverage(analyze_sentiment),
        "rag_quality": _rag_quality(),
    })

    # 4. No sentiment — return unavailable
    no_sentiment_fn = lambda text: {"label": "unavailable", "score": 0.0, "compound": 0.0}
    results.append({
        "condition": "No sentiment",
        "intent_accuracy": _intent_accuracy(full_classify),
        "entity_recall": _entity_recall(extract_entities),
        "sentiment_coverage": _sentiment_coverage(no_sentiment_fn),
        "rag_quality": _rag_quality(),
    })

    # 5. No RAG — score is 0
    results.append({
        "condition": "No RAG",
        "intent_accuracy": _intent_accuracy(full_classify),
        "entity_recall": _entity_recall(extract_entities),
        "sentiment_coverage": _sentiment_coverage(analyze_sentiment),
        "rag_quality": 0.0,
    })

    # 6. Keyword-only intent (no ML)
    if ml_available:
        results.append({
            "condition": "Keyword-only intent",
            "intent_accuracy": _intent_accuracy(keyword_classify),
            "entity_recall": _entity_recall(extract_entities),
            "sentiment_coverage": _sentiment_coverage(analyze_sentiment),
            "rag_quality": _rag_quality(),
        })

    return results


def print_ablation_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  Ablation Study — NLP Pipeline Component Impact Analysis")
    print(f"  Financial AI Agent — CSE495B")
    print(f"{'=' * 80}")

    headers = ["Condition", "Intent Acc", "Entity Rec", "Sent Cov", "RAG Qual"]
    col_widths = [24, 12, 12, 12, 12]

    # Header
    header_line = "  " + "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(f"\n{header_line}")
    print("  " + "-" * sum(col_widths))

    full = results[0] if results else None

    for r in results:
        # Calculate deltas from full pipeline
        intent_str = f"{r['intent_accuracy']:.1%}"
        entity_str = f"{r['entity_recall']:.1%}"
        sent_str = f"{r['sentiment_coverage']:.1%}"
        rag_str = f"{r['rag_quality']:.3f}"

        if full and r["condition"] != "Full pipeline":
            intent_delta = r["intent_accuracy"] - full["intent_accuracy"]
            entity_delta = r["entity_recall"] - full["entity_recall"]
            sent_delta = r["sentiment_coverage"] - full["sentiment_coverage"]
            rag_delta = r["rag_quality"] - full["rag_quality"]

            if intent_delta != 0:
                intent_str += f" ({intent_delta:+.1%})"
            if entity_delta != 0:
                entity_str += f" ({entity_delta:+.1%})"
            if sent_delta != 0:
                sent_str += f" ({sent_delta:+.1%})"
            if rag_delta != 0:
                rag_str += f" ({rag_delta:+.3f})"

        line = "  " + r["condition"].ljust(col_widths[0])
        line += intent_str.ljust(col_widths[1])
        line += entity_str.ljust(col_widths[2])
        line += sent_str.ljust(col_widths[3])
        line += rag_str.ljust(col_widths[4])
        print(line)

    print()

    # Key findings
    print("  Key Findings:")
    print("  " + "-" * 60)
    if full:
        for r in results[1:]:
            degradations = []
            if r["intent_accuracy"] < full["intent_accuracy"]:
                degradations.append(f"intent acc {(r['intent_accuracy'] - full['intent_accuracy']):+.1%}")
            if r["entity_recall"] < full["entity_recall"]:
                degradations.append(f"entity recall {(r['entity_recall'] - full['entity_recall']):+.1%}")
            if r["sentiment_coverage"] < full["sentiment_coverage"]:
                degradations.append(f"sentiment cov {(r['sentiment_coverage'] - full['sentiment_coverage']):+.1%}")
            if r["rag_quality"] < full["rag_quality"]:
                degradations.append(f"RAG qual {(r['rag_quality'] - full['rag_quality']):+.3f}")
            if degradations:
                print(f"  - {r['condition']}: {', '.join(degradations)}")
            else:
                print(f"  - {r['condition']}: no degradation (component independent)")
    print()


if __name__ == "__main__":
    results = run_ablation_study()
    print_ablation_table(results)
