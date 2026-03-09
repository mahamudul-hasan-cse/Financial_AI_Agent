"""Tests for the ML intent classifier (intent_classifier.py).

Validates that the TF-IDF + Logistic Regression pipeline trains correctly,
produces valid predictions with confidence scores, and handles edge cases.
"""

import pytest


class TestMLClassifier:
    """Tests for the ML intent classification pipeline."""

    def test_predict_returns_tuple(self):
        from intent_classifier import predict_intent
        result = predict_intent("What is AAPL price?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_returns_valid_intent(self):
        from intent_classifier import predict_intent
        valid_intents = {
            "stock_price", "chart", "comparison", "news",
            "fundamentals", "recommendation", "excel", "general",
        }
        intent, confidence = predict_intent("Show me a chart of Tesla")
        assert intent in valid_intents

    def test_confidence_in_range(self):
        from intent_classifier import predict_intent
        _, confidence = predict_intent("What is the P/E ratio of Apple?")
        assert 0.0 <= confidence <= 1.0

    def test_stock_price_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("What is the current price of Microsoft?")
        assert intent == "stock_price"

    def test_chart_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("Plot a graph of NVDA stock")
        assert intent == "chart"

    def test_comparison_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("Compare Apple vs Google")
        assert intent == "comparison"

    def test_news_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("Latest news about Amazon")
        assert intent == "news"

    def test_fundamentals_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("What is Tesla earnings per share?")
        assert intent == "fundamentals"

    def test_recommendation_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("Should I buy NVDA stock?")
        assert intent == "recommendation"

    def test_excel_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("Create an Excel spreadsheet for AAPL")
        assert intent == "excel"

    def test_general_intent(self):
        from intent_classifier import predict_intent
        intent, _ = predict_intent("Hello how are you?")
        assert intent == "general"

    def test_get_intent_labels(self):
        from intent_classifier import get_intent_labels
        labels = get_intent_labels()
        assert isinstance(labels, list)
        assert len(labels) == 8
        assert "stock_price" in labels
        assert "general" in labels

    def test_training_data_not_empty(self):
        from intent_classifier import TRAINING_DATA
        assert len(TRAINING_DATA) >= 150  # at least 150 examples

    def test_all_intents_in_training_data(self):
        from intent_classifier import TRAINING_DATA
        intents = {label for _, label in TRAINING_DATA}
        expected = {"stock_price", "chart", "comparison", "news",
                    "fundamentals", "recommendation", "excel", "general"}
        assert intents == expected

    def test_confidence_higher_for_clear_queries(self):
        """ML classifier should be more confident on unambiguous queries."""
        from intent_classifier import predict_intent
        _, conf_clear = predict_intent("What is the current stock price of Apple?")
        _, conf_vague = predict_intent("hmm")
        # Clear query should have higher confidence than a vague one
        assert conf_clear > conf_vague

    def test_empty_input(self):
        from intent_classifier import predict_intent
        intent, confidence = predict_intent("")
        assert isinstance(intent, str)
        assert 0.0 <= confidence <= 1.0
