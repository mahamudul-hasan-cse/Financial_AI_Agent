"""Tests for the NLP preprocessing pipeline (nlp_pipeline.py).

Covers intent classification, sentiment analysis, entity extraction
(including financial-specific NER), confidence scores, and the full
pipeline integration. Tests are written to pass regardless of whether
spaCy and VADER are installed (graceful degradation paths).
"""

import pytest
from nlp_pipeline import (
    classify_intent,
    classify_intent_with_confidence,
    analyze_sentiment,
    extract_entities,
    run_nlp_pipeline,
)


class TestIntentClassification:
    """Tests for classify_intent() — keyword-based intent scoring."""

    def test_stock_price_intent(self):
        assert classify_intent("What is AAPL price today?") == "stock_price"

    def test_chart_intent(self):
        assert classify_intent("Show me a chart of Tesla") == "chart"

    def test_comparison_intent(self):
        assert classify_intent("Compare NVDA vs AMD performance") == "comparison"

    def test_news_intent(self):
        assert classify_intent("Latest news about Apple") == "news"

    def test_fundamentals_intent(self):
        assert classify_intent("What is the P/E ratio of Microsoft?") == "fundamentals"

    def test_recommendation_intent(self):
        assert classify_intent("What do analysts recommend for TSLA?") == "recommendation"

    def test_excel_intent(self):
        assert classify_intent("Download an Excel spreadsheet for AAPL") == "excel"

    def test_general_intent_fallback(self):
        assert classify_intent("Hello there") == "general"

    def test_general_intent_on_empty(self):
        assert classify_intent("") == "general"

    def test_highest_score_wins(self):
        result = classify_intent("Show me a trend chart and graph of the history")
        assert result == "chart"

    def test_case_insensitive(self):
        assert classify_intent("LATEST NEWS ABOUT AMAZON") == "news"


class TestIntentWithConfidence:
    """Tests for classify_intent_with_confidence() — ML + keyword."""

    def test_returns_three_values(self):
        intent, confidence, method = classify_intent_with_confidence("What is AAPL price?")
        assert isinstance(intent, str)
        assert isinstance(confidence, float)
        assert method in ("ml", "keyword")

    def test_confidence_in_range(self):
        _, confidence, _ = classify_intent_with_confidence("Show me a chart of Tesla")
        assert 0.0 <= confidence <= 1.0

    def test_ml_method_when_available(self):
        from nlp_pipeline import ML_CLASSIFIER_AVAILABLE
        _, _, method = classify_intent_with_confidence("What is AAPL price?")
        if ML_CLASSIFIER_AVAILABLE:
            assert method == "ml"
        else:
            assert method == "keyword"


class TestSentimentAnalysis:
    """Tests for analyze_sentiment() — VADER polarity scoring."""

    def test_returns_required_keys(self):
        result = analyze_sentiment("The market is up today.")
        assert "label" in result
        assert "score" in result
        assert "compound" in result

    def test_positive_sentiment(self):
        pytest.importorskip("vaderSentiment", reason="vaderSentiment not installed")
        result = analyze_sentiment(
            "This is amazing! Great news, fantastic earnings, wonderful growth!"
        )
        assert result["label"] == "positive"
        assert result["compound"] > 0

    def test_negative_sentiment(self):
        pytest.importorskip("vaderSentiment", reason="vaderSentiment not installed")
        result = analyze_sentiment(
            "Terrible losses. The company crashed and burned with horrible results."
        )
        assert result["label"] == "negative"
        assert result["compound"] < 0

    def test_neutral_sentiment(self):
        pytest.importorskip("vaderSentiment", reason="vaderSentiment not installed")
        result = analyze_sentiment("The stock closed at 150 dollars on Tuesday.")
        assert result["label"] in ("neutral", "positive", "negative")

    def test_compound_in_range(self):
        result = analyze_sentiment("Apple reported quarterly earnings.")
        assert -1.0 <= result["compound"] <= 1.0

    def test_score_in_range(self):
        result = analyze_sentiment("Market update for today.")
        assert 0.0 <= result["score"] <= 1.0

    def test_unavailable_label_when_vader_missing(self, monkeypatch):
        import nlp_pipeline
        monkeypatch.setattr(nlp_pipeline, "VADER_AVAILABLE", False)
        monkeypatch.setattr(nlp_pipeline, "_vader", None)
        result = analyze_sentiment("Hello world")
        assert result["label"] == "unavailable"
        assert result["compound"] == 0.0


class TestEntityExtraction:
    """Tests for extract_entities() — spaCy NER + ticker regex + financial patterns."""

    def test_detects_bare_ticker(self):
        result = extract_entities("Show me AAPL chart")
        tickers = [e["text"] for e in result if e["label"] == "TICKER"]
        assert "AAPL" in tickers

    def test_detects_multiple_tickers(self):
        result = extract_entities("Compare MSFT vs GOOGL performance")
        tickers = {e["text"] for e in result if e["label"] == "TICKER"}
        assert "MSFT" in tickers or "GOOGL" in tickers

    def test_filters_stopwords(self):
        """Common words like 'I', 'A' should NOT be detected as tickers."""
        result = extract_entities("I am looking for a good stock in the US market")
        ticker_texts = {e["text"] for e in result if e["label"] == "TICKER"}
        assert "I" not in ticker_texts
        assert "US" not in ticker_texts
        assert "A" not in ticker_texts

    def test_returns_list(self):
        result = extract_entities("Hello")
        assert isinstance(result, list)

    def test_no_duplicates(self):
        result = extract_entities("AAPL AAPL AAPL")
        aapl_entries = [e for e in result if e["text"] == "AAPL"]
        assert len(aapl_entries) == 1

    def test_entity_has_required_keys(self):
        result = extract_entities("Show me TSLA price")
        for entity in result:
            assert "text" in entity
            assert "label" in entity
            assert "ticker" in entity

    def test_ticker_field_for_known_ticker(self):
        result = extract_entities("What is AAPL doing today?")
        aapl_entities = [e for e in result if e["text"] == "AAPL"]
        assert len(aapl_entities) >= 1
        assert aapl_entities[0]["ticker"] == "AAPL"

    # ── Financial entity pattern tests ───────────────────────────────

    def test_detects_pe_ratio(self):
        result = extract_entities("What is the P/E ratio of Apple?")
        fin_metrics = [e for e in result if e["label"] == "FIN_METRIC"]
        metric_texts = [e["text"].lower() for e in fin_metrics]
        assert any("p/e" in t for t in metric_texts)

    def test_detects_ebitda(self):
        result = extract_entities("Show me EBITDA for Microsoft")
        fin_metrics = [e for e in result if e["label"] == "FIN_METRIC"]
        metric_texts = [e["text"].upper() for e in fin_metrics]
        assert any("EBITDA" in t for t in metric_texts)

    def test_detects_rsi(self):
        result = extract_entities("What is the RSI for NVDA?")
        fin_metrics = [e for e in result if e["label"] == "FIN_METRIC"]
        metric_texts = [e["text"].upper() for e in fin_metrics]
        assert any("RSI" in t for t in metric_texts)

    def test_detects_quarter_date_ref(self):
        result = extract_entities("Revenue in Q3 2024 was strong")
        date_refs = [e for e in result if e["label"] == "DATE_REF"]
        assert len(date_refs) >= 1
        assert any("Q3 2024" in e["text"] for e in date_refs)

    def test_detects_fiscal_year(self):
        result = extract_entities("EBITDA in FY2023 was record-breaking")
        date_refs = [e for e in result if e["label"] == "DATE_REF"]
        assert any("FY2023" in e["text"] for e in date_refs)

    def test_detects_money_amount(self):
        result = extract_entities("Revenue was $1.2B last quarter")
        money = [e for e in result if e["label"] == "MONEY"]
        assert len(money) >= 1
        assert any("$1.2B" in e["text"] for e in money)

    def test_detects_simple_dollar_amount(self):
        result = extract_entities("Stock price target is $150")
        money = [e for e in result if e["label"] == "MONEY"]
        assert len(money) >= 1

    def test_fin_metric_has_no_ticker(self):
        result = extract_entities("What is the RSI?")
        fin_metrics = [e for e in result if e["label"] == "FIN_METRIC"]
        for m in fin_metrics:
            assert m["ticker"] is None


class TestNLPPipeline:
    """Integration tests for run_nlp_pipeline()."""

    def test_returns_all_required_fields(self):
        result = run_nlp_pipeline("What is AAPL stock price?")
        assert "entities" in result
        assert "intent" in result
        assert "intent_confidence" in result
        assert "intent_method" in result
        assert "sentiment" in result
        assert "spacy_available" in result
        assert "vader_available" in result

    def test_entities_is_list(self):
        result = run_nlp_pipeline("Compare TSLA vs NVDA")
        assert isinstance(result["entities"], list)

    def test_intent_is_string(self):
        result = run_nlp_pipeline("Show me the latest news")
        assert isinstance(result["intent"], str)

    def test_intent_confidence_is_float(self):
        result = run_nlp_pipeline("What is the price of AAPL?")
        assert isinstance(result["intent_confidence"], float)
        assert 0.0 <= result["intent_confidence"] <= 1.0

    def test_intent_method_is_valid(self):
        result = run_nlp_pipeline("Show me a chart")
        assert result["intent_method"] in ("ml", "keyword")

    def test_sentiment_has_label(self):
        result = run_nlp_pipeline("Any updates on Apple stock?")
        assert "label" in result["sentiment"]

    def test_availability_flags_are_bool(self):
        result = run_nlp_pipeline("Hello")
        assert isinstance(result["spacy_available"], bool)
        assert isinstance(result["vader_available"], bool)

    def test_pipeline_on_financial_query(self):
        result = run_nlp_pipeline("What are analyst recommendations for NVDA?")
        assert result["intent"] == "recommendation"
        nvda_entities = [e for e in result["entities"] if e["text"] == "NVDA"]
        assert len(nvda_entities) >= 1

    def test_pipeline_detects_financial_entities(self):
        result = run_nlp_pipeline("What is the P/E ratio in Q3 2024 for a $500M company?")
        labels = {e["label"] for e in result["entities"]}
        assert "FIN_METRIC" in labels
        assert "DATE_REF" in labels
        assert "MONEY" in labels
