"""Systematic error analysis for the ML intent classifier.

Analyzes misclassifications from the gold-standard test set:
  1. Error rate summary
  2. Each misclassified example with true/predicted/confidence
  3. Most confused label pairs
  4. Confidence calibration: mean confidence for correct vs incorrect predictions

Run directly: python error_analysis.py
"""

from __future__ import annotations

from collections import Counter, defaultdict


def run_error_analysis() -> dict:
    """Run full error analysis on the ML classifier.

    Returns:
        Dict with "error_rate", "misclassifications", "confused_pairs",
        "calibration", and "total"/"correct"/"incorrect" counts.
    """
    from evaluation import INTENT_TEST_SET
    from intent_classifier import predict_intent

    total = len(INTENT_TEST_SET)
    correct_items: list[dict] = []
    incorrect_items: list[dict] = []
    confused_pairs: Counter = Counter()

    for text, true_label in INTENT_TEST_SET:
        pred_label, confidence = predict_intent(text)
        item = {
            "text": text,
            "true": true_label,
            "predicted": pred_label,
            "confidence": confidence,
        }
        if pred_label == true_label:
            correct_items.append(item)
        else:
            incorrect_items.append(item)
            confused_pairs[(true_label, pred_label)] += 1

    # Confidence calibration
    correct_confs = [item["confidence"] for item in correct_items]
    incorrect_confs = [item["confidence"] for item in incorrect_items]

    mean_correct_conf = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    mean_incorrect_conf = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0

    # Sort confused pairs by frequency
    sorted_pairs = confused_pairs.most_common()

    return {
        "total": total,
        "correct": len(correct_items),
        "incorrect": len(incorrect_items),
        "error_rate": round(len(incorrect_items) / total, 4) if total > 0 else 0.0,
        "misclassifications": incorrect_items,
        "confused_pairs": sorted_pairs,
        "calibration": {
            "mean_confidence_correct": round(mean_correct_conf, 4),
            "mean_confidence_incorrect": round(mean_incorrect_conf, 4),
            "confidence_gap": round(mean_correct_conf - mean_incorrect_conf, 4),
        },
    }


def print_error_analysis(analysis: dict) -> None:
    """Pretty-print the error analysis report."""
    print(f"\n{'=' * 65}")
    print(f"  Error Analysis — ML Intent Classifier")
    print(f"  Financial AI Agent — CSE495B")
    print(f"{'=' * 65}")

    # 1. Error rate summary
    print(f"\n  Error Rate Summary")
    print(f"  {'-' * 50}")
    print(f"  Total test examples:  {analysis['total']}")
    print(f"  Correct predictions:  {analysis['correct']}")
    print(f"  Incorrect predictions: {analysis['incorrect']}")
    print(f"  Error rate:           {analysis['error_rate']:.2%}")

    # 2. Misclassified examples
    if analysis["misclassifications"]:
        print(f"\n  Misclassified Examples")
        print(f"  {'-' * 50}")
        for item in analysis["misclassifications"]:
            print(f"    \"{item['text'][:55]}\"")
            print(f"      True: {item['true']:<18} Predicted: {item['predicted']:<18} Conf: {item['confidence']:.2%}")
    else:
        print(f"\n  No misclassifications — perfect accuracy!")

    # 3. Most confused label pairs
    if analysis["confused_pairs"]:
        print(f"\n  Most Confused Label Pairs")
        print(f"  {'-' * 50}")
        print(f"  {'True Label':<18} {'Predicted As':<18} {'Count':>6}")
        for (true, pred), count in analysis["confused_pairs"]:
            print(f"  {true:<18} {pred:<18} {count:>6}")

    # 4. Confidence calibration
    cal = analysis["calibration"]
    print(f"\n  Confidence Calibration")
    print(f"  {'-' * 50}")
    print(f"  Mean confidence (correct):   {cal['mean_confidence_correct']:.2%}")
    print(f"  Mean confidence (incorrect): {cal['mean_confidence_incorrect']:.2%}")
    print(f"  Confidence gap:              {cal['confidence_gap']:.2%}")

    if cal["confidence_gap"] > 0.1:
        print(f"  → Good: the model is notably more confident on correct predictions.")
    elif cal["confidence_gap"] > 0:
        print(f"  → Moderate: some calibration signal, but confidence gap is small.")
    else:
        print(f"  → Warning: model is not well-calibrated (equal or higher confidence on errors).")

    print()


if __name__ == "__main__":
    analysis = run_error_analysis()
    print_error_analysis(analysis)
