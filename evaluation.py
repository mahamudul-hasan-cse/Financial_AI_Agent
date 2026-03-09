"""Quantitative evaluation module for the NLP pipeline.

Provides rigorous measurement of each NLP component:
  1. Intent classification — precision, recall, F1 per class + macro averages
     for both ML (TF-IDF + LogReg) and keyword-based classifiers
  2. NER — precision and recall on a small labeled entity test set
  3. Side-by-side comparison table of ML vs keyword classifier

Run directly: python evaluation.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Tuple

# ── Gold-standard test set for intent classification ─────────────────────────
# 80 labeled examples (10 per intent) — held out from training data.
INTENT_TEST_SET: list[Tuple[str, str]] = [
    # stock_price (10)
    ("What's AAPL stock price right now?", "stock_price"),
    ("How much is a share of Google today?", "stock_price"),
    ("Current price of Tesla", "stock_price"),
    ("Get me a stock quote for AMZN", "stock_price"),
    ("What did META close at?", "stock_price"),
    ("Share price of Nvidia please", "stock_price"),
    ("How is MSFT stock doing?", "stock_price"),
    ("Price of Netflix stock", "stock_price"),
    ("What's the current value of AMD shares?", "stock_price"),
    ("Check the live quote for Disney", "stock_price"),

    # chart (10)
    ("Plot the price of AAPL for the last 6 months", "chart"),
    ("Show a chart of Tesla this year", "chart"),
    ("Graph Microsoft stock over 5 years", "chart"),
    ("Visualize NVDA stock trend for me", "chart"),
    ("I want to see a price chart for Google", "chart"),
    ("Draw a line chart of Amazon stock", "chart"),
    ("Show me the history of META stock price", "chart"),
    ("Create a graph of Netflix performance", "chart"),
    ("Display a chart for AMD over 3 months", "chart"),
    ("Can you plot Disney stock movement?", "chart"),

    # comparison (10)
    ("Compare Apple and Microsoft stocks", "comparison"),
    ("Tesla vs Rivian which is better?", "comparison"),
    ("How does NVDA compare to AMD?", "comparison"),
    ("Google versus Meta stock performance", "comparison"),
    ("Which stock has better returns AAPL or AMZN?", "comparison"),
    ("Side by side Netflix and Disney", "comparison"),
    ("Compare the two tech giants Apple and Google", "comparison"),
    ("Intel versus AMD comparison", "comparison"),
    ("Difference between JPM and BAC stock?", "comparison"),
    ("Which is a better investment Uber or Lyft?", "comparison"),

    # news (10)
    ("What are the latest news for AAPL?", "news"),
    ("Recent headlines about Tesla", "news"),
    ("Any news on Nvidia today?", "news"),
    ("What happened with Microsoft stock?", "news"),
    ("Show me recent updates on Amazon", "news"),
    ("Latest announcements from Google", "news"),
    ("Any breaking news about the market?", "news"),
    ("Recent financial reports for Netflix", "news"),
    ("What's the latest on the Fed meeting?", "news"),
    ("News about Disney earnings", "news"),

    # fundamentals (10)
    ("What is Apple's PE ratio?", "fundamentals"),
    ("Show Tesla's earnings per share", "fundamentals"),
    ("Revenue of Nvidia last quarter", "fundamentals"),
    ("Microsoft's profit margin", "fundamentals"),
    ("What is the market cap of Amazon?", "fundamentals"),
    ("Dividend yield for Coca-Cola stock", "fundamentals"),
    ("Google's debt to equity ratio", "fundamentals"),
    ("Free cash flow for Meta", "fundamentals"),
    ("EBITDA of Netflix", "fundamentals"),
    ("Balance sheet analysis for JPMorgan", "fundamentals"),

    # recommendation (10)
    ("Should I buy Apple stock?", "recommendation"),
    ("What do analysts rate Tesla?", "recommendation"),
    ("Is NVDA a buy or sell?", "recommendation"),
    ("Analyst recommendations for Microsoft", "recommendation"),
    ("Should I hold Amazon stock?", "recommendation"),
    ("Is Google a good investment right now?", "recommendation"),
    ("Target price for Meta stock", "recommendation"),
    ("Buy sell hold rating for Netflix", "recommendation"),
    ("Is AMD stock undervalued?", "recommendation"),
    ("Wall Street outlook on Disney", "recommendation"),

    # excel (10)
    ("Create an Excel file for AAPL", "excel"),
    ("Download Tesla data as spreadsheet", "excel"),
    ("Export Nvidia financials to xlsx", "excel"),
    ("Generate a spreadsheet report for Microsoft", "excel"),
    ("I need an Excel download for Amazon", "excel"),
    ("Make a spreadsheet with Google data", "excel"),
    ("Export this comparison to Excel", "excel"),
    ("Download stock data as xlsx file", "excel"),
    ("Create a financial spreadsheet for Netflix", "excel"),
    ("Excel export for AMD fundamentals", "excel"),

    # general (10)
    ("Hello how are you?", "general"),
    ("What can you help me with?", "general"),
    ("Tell me about investing basics", "general"),
    ("How does the stock market work?", "general"),
    ("Good morning", "general"),
    ("Thanks for your help", "general"),
    ("What is a bear market?", "general"),
    ("Explain how to read financial statements", "general"),
    ("I'm new here what can I ask?", "general"),
    ("What time does the market close?", "general"),
]

# ── NER test set ─────────────────────────────────────────────────────────────
# Each entry: (text, expected_entities) where expected_entities is a list of
# dicts with at minimum {"text": ..., "label": ...}
NER_TEST_SET: list[Tuple[str, list[dict]]] = [
    # ── TICKER examples ──
    ("What is AAPL stock price?", [{"text": "AAPL", "label": "TICKER"}]),
    ("Compare MSFT vs GOOGL", [{"text": "MSFT", "label": "TICKER"}, {"text": "GOOGL", "label": "TICKER"}]),
    ("Show me NVDA performance", [{"text": "NVDA", "label": "TICKER"}]),
    ("Is TSLA a good buy?", [{"text": "TSLA", "label": "TICKER"}]),
    ("Get quote for AMZN", [{"text": "AMZN", "label": "TICKER"}]),
    ("Chart META over 1 year", [{"text": "META", "label": "TICKER"}]),

    # ── ORG examples ──
    ("Tesla earnings report", [{"text": "Tesla", "label": "ORG"}]),
    ("Microsoft announced layoffs", [{"text": "Microsoft", "label": "ORG"}]),
    ("Apple released new products", [{"text": "Apple", "label": "ORG"}]),
    ("Goldman Sachs quarterly results", [{"text": "Goldman Sachs", "label": "ORG"}]),
    ("What is Nvidia revenue?", [{"text": "Nvidia", "label": "ORG"}]),

    # ── FIN_METRIC examples ──
    ("Show me the P/E ratio for Apple", [{"text": "P/E ratio", "label": "FIN_METRIC"}]),
    ("What is the RSI for NVDA?", [
        {"text": "RSI", "label": "FIN_METRIC"},
        {"text": "NVDA", "label": "TICKER"},
    ]),
    ("EBITDA of Microsoft in FY2023", [
        {"text": "EBITDA", "label": "FIN_METRIC"},
        {"text": "FY2023", "label": "DATE_REF"},
    ]),
    ("What is the ROE for this company?", [{"text": "ROE", "label": "FIN_METRIC"}]),
    ("Show me the dividend yield", [{"text": "dividend yield", "label": "FIN_METRIC"}]),
    ("Current market cap of Tesla", [{"text": "market cap", "label": "FIN_METRIC"}]),
    ("Free cash flow for Amazon", [{"text": "free cash flow", "label": "FIN_METRIC"}]),

    # ── DATE_REF examples ──
    ("Revenue was $1.2B in Q3 2024", [
        {"text": "$1.2B", "label": "MONEY"},
        {"text": "Q3 2024", "label": "DATE_REF"},
    ]),
    ("Results for FY2023", [{"text": "FY2023", "label": "DATE_REF"}]),
    ("Performance in Q1 2025", [{"text": "Q1 2025", "label": "DATE_REF"}]),
    ("H1 2024 earnings call", [{"text": "H1 2024", "label": "DATE_REF"}]),
    ("Growth in Q4 2025 was strong", [{"text": "Q4 2025", "label": "DATE_REF"}]),

    # ── MONEY examples ──
    ("Stock worth $500M with EPS of $3.50", [
        {"text": "$500M", "label": "MONEY"},
        {"text": "EPS", "label": "FIN_METRIC"},
        {"text": "$3.50", "label": "MONEY"},
    ]),
    ("Revenue hit $10B last quarter", [{"text": "$10B", "label": "MONEY"}]),
    ("Share price dropped to $150", [{"text": "$150", "label": "MONEY"}]),
    ("Profit was $2.5M in FY2023", [
        {"text": "$2.5M", "label": "MONEY"},
        {"text": "FY2023", "label": "DATE_REF"},
    ]),
    ("Target price is $500", [{"text": "$500", "label": "MONEY"}]),

    # ── Multi-entity examples ──
    ("AAPL reported $3.50 EPS in Q4 2025", [
        {"text": "AAPL", "label": "TICKER"},
        {"text": "$3.50", "label": "MONEY"},
        {"text": "EPS", "label": "FIN_METRIC"},
        {"text": "Q4 2025", "label": "DATE_REF"},
    ]),
    ("Tesla market cap is $800B with P/E ratio of 60", [
        {"text": "Tesla", "label": "ORG"},
        {"text": "market cap", "label": "FIN_METRIC"},
        {"text": "$800B", "label": "MONEY"},
        {"text": "P/E ratio", "label": "FIN_METRIC"},
    ]),
    ("MSFT dividend yield in Q2 2025 was $0.75", [
        {"text": "MSFT", "label": "TICKER"},
        {"text": "dividend yield", "label": "FIN_METRIC"},
        {"text": "Q2 2025", "label": "DATE_REF"},
        {"text": "$0.75", "label": "MONEY"},
    ]),
]


def compute_classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> dict:
    """Compute per-class precision, recall, F1, and macro averages.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional ordered list of label names.

    Returns:
        Dict with "per_class" metrics and "macro" averages.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    per_class: dict[str, dict] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    n_classes = len(labels)
    macro_p = sum(m["precision"] for m in per_class.values()) / n_classes if n_classes else 0
    macro_r = sum(m["recall"] for m in per_class.values()) / n_classes if n_classes else 0
    macro_f1 = sum(m["f1"] for m in per_class.values()) / n_classes if n_classes else 0

    return {
        "per_class": per_class,
        "macro": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        },
        "accuracy": round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true), 4) if y_true else 0,
    }


def build_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    """Build a confusion matrix as a 2D list.

    Rows = true labels, Columns = predicted labels.
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1
    return matrix


def evaluate_intent_classifiers() -> dict:
    """Evaluate both ML and keyword classifiers on the gold-standard test set.

    Returns:
        Dict with "ml" and "keyword" evaluation results.
    """
    from nlp_pipeline import classify_intent as keyword_classify

    texts = [t for t, _ in INTENT_TEST_SET]
    y_true = [l for _, l in INTENT_TEST_SET]
    labels = sorted(set(y_true))

    results: dict = {}

    # Keyword classifier
    kw_preds = [keyword_classify(t) for t in texts]
    results["keyword"] = compute_classification_metrics(y_true, kw_preds, labels)

    # ML classifier
    try:
        from intent_classifier import predict_intent
        ml_preds = [predict_intent(t)[0] for t in texts]
        results["ml"] = compute_classification_metrics(y_true, ml_preds, labels)
        results["ml"]["confusion_matrix"] = build_confusion_matrix(y_true, ml_preds, labels)
        results["ml"]["labels"] = labels
    except Exception as e:
        results["ml"] = {"error": str(e)}

    results["keyword"]["confusion_matrix"] = build_confusion_matrix(y_true, kw_preds, labels)
    results["keyword"]["labels"] = labels

    return results


def evaluate_ner() -> dict:
    """Evaluate NER on the small labeled test set.

    Computes entity-level precision and recall by matching on (text, label) pairs.
    """
    from nlp_pipeline import extract_entities

    total_expected = 0
    total_predicted = 0
    total_correct = 0

    per_example: list[dict] = []

    for text, expected_entities in NER_TEST_SET:
        predicted = extract_entities(text)
        expected_set = {(e["text"], e["label"]) for e in expected_entities}
        predicted_set = {(e["text"], e["label"]) for e in predicted}

        correct = expected_set & predicted_set
        total_expected += len(expected_set)
        total_predicted += len(predicted_set)
        total_correct += len(correct)

        per_example.append({
            "text": text,
            "expected": list(expected_set),
            "predicted": list(predicted_set),
            "correct": list(correct),
            "missed": list(expected_set - predicted_set),
        })

    precision = total_correct / total_predicted if total_predicted > 0 else 0
    recall = total_correct / total_expected if total_expected > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_expected": total_expected,
        "total_predicted": total_predicted,
        "total_correct": total_correct,
        "per_example": per_example,
    }


def evaluate_ner_per_type() -> dict:
    """Evaluate NER with per-entity-type precision, recall, F1 breakdown.

    Returns:
        Dict mapping entity type → {"precision", "recall", "f1", "support"}.
    """
    from nlp_pipeline import extract_entities

    # Collect per-type TP, FP, FN
    type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for text, expected_entities in NER_TEST_SET:
        predicted = extract_entities(text)
        expected_set = {(e["text"], e["label"]) for e in expected_entities}
        predicted_set = {(e["text"], e["label"]) for e in predicted}

        correct = expected_set & predicted_set
        missed = expected_set - predicted_set
        spurious = predicted_set - expected_set

        for _, label in correct:
            type_stats[label]["tp"] += 1
        for _, label in missed:
            type_stats[label]["fn"] += 1
        for _, label in spurious:
            type_stats[label]["fp"] += 1

    results: dict[str, dict] = {}
    for etype, stats in sorted(type_stats.items()):
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[etype] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    return results


# ── RAG Evaluation ──────────────────────────────────────────────────────────
# Gold-standard (query, expected_keywords) pairs for retrieval evaluation.
# We check whether retrieved chunks contain expected keywords (content-based).
RAG_GOLD_SET: list[Tuple[str, list[str]]] = [
    ("What is the P/E ratio?", ["price", "earnings", "P/E"]),
    ("How does dividend yield work?", ["dividend", "yield"]),
    ("Explain RSI indicator", ["RSI", "relative strength", "overbought"]),
    ("What is market capitalization?", ["market cap", "shares", "outstanding"]),
    ("How do moving averages work?", ["moving average", "SMA", "EMA"]),
    ("What is beta in stocks?", ["beta", "volatility", "market"]),
    ("Explain earnings per share", ["earnings", "EPS", "share"]),
    ("What is a short squeeze?", ["short", "squeeze", "cover"]),
    ("How do interest rates affect stocks?", ["interest rate", "fed", "bond"]),
    ("What are the S&P 500 sectors?", ["sector", "S&P", "technology"]),
]


def evaluate_rag(top_k: int = 5) -> dict:
    """Evaluate RAG retrieval using MRR and Recall@K.

    Uses keyword-matching against retrieved chunk content since we don't have
    exact chunk IDs. A chunk is "relevant" if it contains any expected keyword.

    Args:
        top_k: Number of chunks to retrieve per query.

    Returns:
        Dict with "mrr", "recall_at_k", "k", and per-query details.
    """
    try:
        from tools.rag_tools import _retriever
    except Exception:
        return {"mrr": 0.0, "recall_at_k": 0.0, "k": top_k, "available": False}

    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    per_query: list[dict] = []

    for query, expected_keywords in RAG_GOLD_SET:
        chunks = _retriever.retrieve(query, top_k=top_k)
        chunk_texts = [c.text.lower() if hasattr(c, "text") else str(c).lower() for c in chunks]

        # Find rank of first relevant chunk
        first_relevant_rank = 0
        relevant_count = 0
        for i, ct in enumerate(chunk_texts):
            is_relevant = any(kw.lower() in ct for kw in expected_keywords)
            if is_relevant:
                relevant_count += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = i + 1

        rr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
        reciprocal_ranks.append(rr)
        # Recall: fraction of queries where at least one relevant chunk was found
        recalls.append(1.0 if relevant_count > 0 else 0.0)

        per_query.append({
            "query": query,
            "reciprocal_rank": round(rr, 4),
            "relevant_chunks_found": relevant_count,
        })

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    recall_at_k = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "mrr": round(mrr, 4),
        "recall_at_k": round(recall_at_k, 4),
        "k": top_k,
        "available": True,
        "per_query": per_query,
    }


def print_classification_report(name: str, metrics: dict) -> None:
    """Pretty-print a classification report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        return

    print(f"\n  {'Class':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>9}")
    print(f"  {'-' * 48}")

    for label, m in metrics["per_class"].items():
        print(f"  {label:<18} {m['precision']:>7.2%} {m['recall']:>7.2%} {m['f1']:>7.2%} {m['support']:>9}")

    macro = metrics["macro"]
    print(f"  {'-' * 48}")
    print(f"  {'macro avg':<18} {macro['precision']:>7.2%} {macro['recall']:>7.2%} {macro['f1']:>7.2%}")
    print(f"  {'accuracy':<18} {metrics['accuracy']:>7.2%}")

    # Confusion matrix
    if "confusion_matrix" in metrics:
        labels = metrics["labels"]
        matrix = metrics["confusion_matrix"]
        print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
        header = "  " + " " * 14 + "".join(f"{l[:6]:>7}" for l in labels)
        print(header)
        for i, label in enumerate(labels):
            row = "  " + f"{label[:13]:<14}" + "".join(f"{matrix[i][j]:>7}" for j in range(len(labels)))
            print(row)


def print_ner_report(ner_metrics: dict) -> None:
    """Pretty-print NER evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  NER Evaluation")
    print(f"{'=' * 60}")
    print(f"  Precision: {ner_metrics['precision']:.2%}")
    print(f"  Recall:    {ner_metrics['recall']:.2%}")
    print(f"  F1:        {ner_metrics['f1']:.2%}")
    print(f"  ({ner_metrics['total_correct']}/{ner_metrics['total_expected']} entities correctly identified)")

    print(f"\n  Per-example breakdown:")
    for ex in ner_metrics["per_example"]:
        status = "OK" if not ex["missed"] else "MISS"
        print(f"    [{status}] \"{ex['text'][:50]}\"")
        if ex["missed"]:
            for m in ex["missed"]:
                print(f"          missed: {m}")


def print_ner_per_type_report(per_type: dict) -> None:
    """Pretty-print per-entity-type NER metrics."""
    print(f"\n{'=' * 60}")
    print(f"  NER Per-Entity-Type Breakdown")
    print(f"{'=' * 60}")
    print(f"\n  {'Entity Type':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>9}")
    print(f"  {'-' * 48}")
    for etype, m in per_type.items():
        print(f"  {etype:<18} {m['precision']:>7.2%} {m['recall']:>7.2%} {m['f1']:>7.2%} {m['support']:>9}")


def print_rag_report(rag_metrics: dict) -> None:
    """Pretty-print RAG evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  RAG Retrieval Evaluation")
    print(f"{'=' * 60}")
    if not rag_metrics.get("available", False):
        print("  RAG retriever not available — skipped.")
        return
    print(f"  MRR (Mean Reciprocal Rank): {rag_metrics['mrr']:.2%}")
    print(f"  Recall@{rag_metrics['k']}:                 {rag_metrics['recall_at_k']:.2%}")
    print(f"\n  Per-query breakdown:")
    for q in rag_metrics.get("per_query", []):
        print(f"    RR={q['reciprocal_rank']:.2f}  chunks={q['relevant_chunks_found']}  \"{q['query'][:50]}\"")


def run_full_evaluation() -> dict:
    """Run all evaluations and return combined results."""
    intent_results = evaluate_intent_classifiers()
    ner_results = evaluate_ner()
    ner_per_type = evaluate_ner_per_type()
    rag_results = evaluate_rag()
    return {
        "intent": intent_results,
        "ner": ner_results,
        "ner_per_type": ner_per_type,
        "rag": rag_results,
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NLP Pipeline Evaluation Report")
    print("  Financial AI Agent — CSE495B")
    print("=" * 60)

    results = run_full_evaluation()

    print_classification_report("Keyword Intent Classifier", results["intent"]["keyword"])
    print_classification_report("ML Intent Classifier (TF-IDF + LogReg)", results["intent"].get("ml", {"error": "not available"}))
    print_ner_report(results["ner"])
    print_ner_per_type_report(results["ner_per_type"])
    print_rag_report(results["rag"])

    # Summary comparison
    print(f"\n{'=' * 60}")
    print(f"  Summary: ML vs Keyword Classifier")
    print(f"{'=' * 60}")

    kw = results["intent"]["keyword"]
    ml = results["intent"].get("ml", {})

    print(f"  {'Metric':<20} {'Keyword':>10} {'ML':>10} {'Delta':>10}")
    print(f"  {'-' * 50}")

    if "accuracy" in ml:
        for metric_name in ["accuracy"]:
            kw_val = kw[metric_name]
            ml_val = ml[metric_name]
            delta = ml_val - kw_val
            sign = "+" if delta >= 0 else ""
            print(f"  {metric_name:<20} {kw_val:>9.2%} {ml_val:>9.2%} {sign}{delta:>9.2%}")

        for metric_name in ["precision", "recall", "f1"]:
            kw_val = kw["macro"][metric_name]
            ml_val = ml["macro"][metric_name]
            delta = ml_val - kw_val
            sign = "+" if delta >= 0 else ""
            print(f"  {'macro ' + metric_name:<20} {kw_val:>9.2%} {ml_val:>9.2%} {sign}{delta:>9.2%}")
    else:
        print("  ML classifier not available")

    print()
