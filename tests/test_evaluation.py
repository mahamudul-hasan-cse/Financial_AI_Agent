"""Tests for the evaluation module (evaluation.py).

Validates that the evaluation framework runs without errors and produces
valid metrics for both intent classification and NER.
"""

import pytest


class TestEvaluationMetrics:
    """Tests for compute_classification_metrics."""

    def test_perfect_predictions(self):
        from evaluation import compute_classification_metrics
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "a", "b"]
        result = compute_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["macro"]["f1"] == 1.0

    def test_completely_wrong(self):
        from evaluation import compute_classification_metrics
        y_true = ["a", "a", "a"]
        y_pred = ["b", "b", "b"]
        result = compute_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == 0.0

    def test_returns_per_class(self):
        from evaluation import compute_classification_metrics
        y_true = ["a", "b", "c"]
        y_pred = ["a", "b", "c"]
        result = compute_classification_metrics(y_true, y_pred)
        assert "a" in result["per_class"]
        assert "b" in result["per_class"]
        assert "c" in result["per_class"]

    def test_metrics_in_range(self):
        from evaluation import compute_classification_metrics
        y_true = ["a", "b", "a", "b", "c"]
        y_pred = ["a", "a", "b", "b", "c"]
        result = compute_classification_metrics(y_true, y_pred)
        for label, m in result["per_class"].items():
            assert 0 <= m["precision"] <= 1
            assert 0 <= m["recall"] <= 1
            assert 0 <= m["f1"] <= 1


class TestConfusionMatrix:
    """Tests for build_confusion_matrix."""

    def test_correct_shape(self):
        from evaluation import build_confusion_matrix
        labels = ["a", "b", "c"]
        matrix = build_confusion_matrix(["a", "b", "c"], ["a", "b", "c"], labels)
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)

    def test_diagonal_for_perfect(self):
        from evaluation import build_confusion_matrix
        labels = ["a", "b"]
        matrix = build_confusion_matrix(["a", "b", "a"], ["a", "b", "a"], labels)
        assert matrix[0][0] == 2  # a correct
        assert matrix[1][1] == 1  # b correct
        assert matrix[0][1] == 0  # no confusion


class TestIntentTestSet:
    """Tests for the gold-standard intent test set."""

    def test_test_set_size(self):
        from evaluation import INTENT_TEST_SET
        assert len(INTENT_TEST_SET) >= 70

    def test_all_intents_represented(self):
        from evaluation import INTENT_TEST_SET
        intents = {label for _, label in INTENT_TEST_SET}
        expected = {"stock_price", "chart", "comparison", "news",
                    "fundamentals", "recommendation", "excel", "general"}
        assert intents == expected


class TestIntentEvaluation:
    """Integration tests for evaluate_intent_classifiers."""

    def test_runs_without_error(self):
        from evaluation import evaluate_intent_classifiers
        results = evaluate_intent_classifiers()
        assert "keyword" in results
        assert "ml" in results

    def test_keyword_accuracy_reasonable(self):
        from evaluation import evaluate_intent_classifiers
        results = evaluate_intent_classifiers()
        assert results["keyword"]["accuracy"] > 0.3  # keyword should get at least some right

    def test_ml_accuracy_higher_than_keyword(self):
        from evaluation import evaluate_intent_classifiers
        results = evaluate_intent_classifiers()
        if "error" not in results["ml"]:
            assert results["ml"]["accuracy"] >= results["keyword"]["accuracy"]


class TestNEREvaluation:
    """Integration tests for evaluate_ner."""

    def test_runs_without_error(self):
        from evaluation import evaluate_ner
        results = evaluate_ner()
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results

    def test_metrics_in_range(self):
        from evaluation import evaluate_ner
        results = evaluate_ner()
        assert 0 <= results["precision"] <= 1
        assert 0 <= results["recall"] <= 1
        assert 0 <= results["f1"] <= 1


class TestFullEvaluation:
    """Tests for run_full_evaluation."""

    def test_runs_without_error(self):
        from evaluation import run_full_evaluation
        results = run_full_evaluation()
        assert "intent" in results
        assert "ner" in results
