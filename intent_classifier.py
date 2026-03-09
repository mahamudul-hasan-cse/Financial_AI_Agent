"""ML-based intent classifier using TF-IDF + Logistic Regression.

Replaces the keyword-matching approach with a trained sklearn pipeline for
more robust intent classification. Training data is synthetic but covers
diverse phrasings for each of the 8 financial intents.

Architecture:
  - Text preprocessing: lowercase → remove punctuation (keep $) → lemmatize
  - TF-IDF vectorizer (unigrams + bigrams) transforms preprocessed text to sparse features
  - Logistic Regression (one-vs-rest) maps features to intent probabilities
  - predict_intent() returns both the predicted label and its confidence score

Supports cross-validation and hyperparameter tuning for rigorous evaluation.
Falls back to keyword classifier when sklearn is unavailable.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Text Preprocessing ──────────────────────────────────────────────────────
try:
    import spacy as _spacy_mod
    _lemmatizer_nlp = _spacy_mod.load("en_core_web_sm", disable=["ner", "parser"])
    _SPACY_LEMMA_AVAILABLE = True
except Exception:
    _SPACY_LEMMA_AVAILABLE = False
    _lemmatizer_nlp = None

# Punctuation to remove (keep $ since it's meaningful in finance)
_PUNCT_TO_REMOVE = string.punctuation.replace("$", "")
_PUNCT_TABLE = str.maketrans("", "", _PUNCT_TO_REMOVE)


def preprocess_text(text: str) -> str:
    """Preprocess text for the intent classifier.

    Pipeline: lowercase → remove punctuation (keep $) → lemmatize → strip.
    Falls back to just lowercase + punctuation removal if spaCy unavailable.

    Args:
        text: Raw user query.

    Returns:
        Cleaned and normalized text string.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation (keep $)
    text = text.translate(_PUNCT_TABLE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Lemmatize with spaCy if available
    if _SPACY_LEMMA_AVAILABLE and _lemmatizer_nlp is not None:
        doc = _lemmatizer_nlp(text)
        text = " ".join(token.lemma_ for token in doc)

    return text.strip()

# ── Training data ────────────────────────────────────────────────────────────
# ~25 examples per intent, 8 intents = ~200 labeled training samples.
TRAINING_DATA: list[Tuple[str, str]] = [
    # stock_price
    ("What is the current price of AAPL?", "stock_price"),
    ("How much is Tesla stock trading at today?", "stock_price"),
    ("Show me the price of Microsoft", "stock_price"),
    ("What is NVDA worth right now?", "stock_price"),
    ("Get me the current stock quote for Amazon", "stock_price"),
    ("How is GOOGL trading?", "stock_price"),
    ("Tell me the share price of Netflix", "stock_price"),
    ("What did Apple close at today?", "stock_price"),
    ("Current value of META stock", "stock_price"),
    ("What's the market price of AMD?", "stock_price"),
    ("How much does one share of Disney cost?", "stock_price"),
    ("Price check on TSLA", "stock_price"),
    ("What is Intel stock at right now?", "stock_price"),
    ("Get the live price of Bitcoin", "stock_price"),
    ("How much is a share of Nvidia today?", "stock_price"),
    ("Tell me the closing price of JPMorgan", "stock_price"),
    ("What is the stock value of Goldman Sachs?", "stock_price"),
    ("Real time quote for MSFT", "stock_price"),
    ("Current trading price of Uber", "stock_price"),
    ("How is Shopify stock doing price wise?", "stock_price"),
    ("What's the latest price on AMZN?", "stock_price"),
    ("How much is SPOT stock?", "stock_price"),
    ("Get me a quote for Oracle", "stock_price"),
    ("What is the stock price for Broadcom?", "stock_price"),
    ("Show the current market value of Adobe", "stock_price"),

    # chart
    ("Show me a chart of AAPL over the last year", "chart"),
    ("Plot Tesla stock price history", "chart"),
    ("Generate a graph of Microsoft performance", "chart"),
    ("Visualize NVDA stock trend", "chart"),
    ("Chart the price movement of Amazon", "chart"),
    ("Show me a line chart of Google stock", "chart"),
    ("Draw a graph showing Netflix over 6 months", "chart"),
    ("I want to see a price chart for META", "chart"),
    ("Display the stock history of AMD", "chart"),
    ("Create a visualization of Disney stock", "chart"),
    ("Show me the trend for TSLA this year", "chart"),
    ("Plot the movement of Intel shares", "chart"),
    ("Give me a chart comparing stock prices", "chart"),
    ("Visualize the performance of JPM over time", "chart"),
    ("Can you graph the price of gold?", "chart"),
    ("Draw a stock chart for Coinbase", "chart"),
    ("Show price history graph for Spotify", "chart"),
    ("I need a chart of Palantir stock", "chart"),
    ("Plot AAPL and MSFT together", "chart"),
    ("Show me a candlestick chart for Tesla", "chart"),
    ("Generate a performance chart for the S&P 500", "chart"),
    ("Line chart of Robinhood stock please", "chart"),
    ("Trend analysis chart for Uber", "chart"),
    ("Display price movements for QCOM", "chart"),
    ("Show a 1 year graph of Apple stock", "chart"),

    # comparison
    ("Compare AAPL vs MSFT performance", "comparison"),
    ("Which is better, Tesla or Rivian?", "comparison"),
    ("NVDA versus AMD which stock is stronger?", "comparison"),
    ("Compare Google and Meta stocks", "comparison"),
    ("What's the difference between Apple and Samsung stock?", "comparison"),
    ("Tesla vs Ford comparison", "comparison"),
    ("Compare the market cap of Amazon and Walmart", "comparison"),
    ("Which stock performed better, Netflix or Disney?", "comparison"),
    ("How does JPMorgan compare against Bank of America?", "comparison"),
    ("Side by side comparison of Intel and AMD", "comparison"),
    ("AAPL vs GOOGL which has better returns?", "comparison"),
    ("Compare both Tesla and Uber stocks", "comparison"),
    ("Put Microsoft against Apple", "comparison"),
    ("Which one should I pick, NVDA or AMD?", "comparison"),
    ("How do these two stocks compare: META vs SNAP?", "comparison"),
    ("Difference between Coinbase and Robinhood stock", "comparison"),
    ("Compare Salesforce and Oracle performance", "comparison"),
    ("Is Adobe better than Microsoft for investment?", "comparison"),
    ("Head to head Palantir vs Snowflake", "comparison"),
    ("Which tech stock is stronger between AAPL and AMZN?", "comparison"),
    ("Compare dividends of Coca-Cola versus Pepsi", "comparison"),
    ("Netflix vs Disney Plus stocks side by side", "comparison"),
    ("How does Spotify stack up against Apple Music stock?", "comparison"),
    ("Uber versus Lyft stock comparison", "comparison"),
    ("Compare growth rates of NVDA and AVGO", "comparison"),

    # news
    ("What's the latest news about Apple?", "news"),
    ("Any recent headlines about Tesla?", "news"),
    ("Show me news for NVDA", "news"),
    ("What happened with Microsoft today?", "news"),
    ("Latest updates on Amazon stock", "news"),
    ("Recent news about Google earnings", "news"),
    ("What did Meta announce today?", "news"),
    ("Any new releases from Netflix?", "news"),
    ("Show recent reports about AMD", "news"),
    ("What's in the news about Disney?", "news"),
    ("Breaking news for the stock market", "news"),
    ("Latest headlines on cryptocurrency", "news"),
    ("What announcements did JPMorgan make?", "news"),
    ("Recent financial news about Intel", "news"),
    ("Any update on the Fed decision?", "news"),
    ("Show me market news today", "news"),
    ("What's happening with bank stocks?", "news"),
    ("Latest report on Tesla earnings call", "news"),
    ("Any news about the IPO market?", "news"),
    ("Recent headlines about tech stocks", "news"),
    ("What's the buzz around Nvidia?", "news"),
    ("Show me the latest on Coinbase", "news"),
    ("Any breaking news about Apple today?", "news"),
    ("Recent developments at Goldman Sachs", "news"),
    ("What are the top financial stories today?", "news"),

    # fundamentals
    ("What is the P/E ratio of Apple?", "fundamentals"),
    ("Show me Tesla's earnings per share", "fundamentals"),
    ("What is NVDA's revenue growth?", "fundamentals"),
    ("Microsoft profit margins", "fundamentals"),
    ("Amazon's market capitalization", "fundamentals"),
    ("What is the dividend yield of Coca-Cola?", "fundamentals"),
    ("Show me fundamental analysis for META", "fundamentals"),
    ("What is Google's debt to equity ratio?", "fundamentals"),
    ("EPS for Netflix last quarter", "fundamentals"),
    ("Cash flow analysis for AMD", "fundamentals"),
    ("Disney's EBITDA numbers", "fundamentals"),
    ("What are Intel's fundamentals like?", "fundamentals"),
    ("Revenue and profit for JPMorgan", "fundamentals"),
    ("Show me the balance sheet for Apple", "fundamentals"),
    ("What is Tesla's free cash flow?", "fundamentals"),
    ("Operating margin of Microsoft", "fundamentals"),
    ("What's the book value of Goldman Sachs?", "fundamentals"),
    ("Return on equity for Amazon", "fundamentals"),
    ("Quarterly earnings of NVDA", "fundamentals"),
    ("What is the PE of the S&P 500?", "fundamentals"),
    ("Show me key financial metrics for Uber", "fundamentals"),
    ("Gross margin for Adobe", "fundamentals"),
    ("What's Salesforce revenue this year?", "fundamentals"),
    ("Net income for Broadcom", "fundamentals"),
    ("Earnings report analysis for Apple", "fundamentals"),

    # recommendation
    ("What do analysts recommend for AAPL?", "recommendation"),
    ("Should I buy Tesla stock?", "recommendation"),
    ("Is NVDA a good buy right now?", "recommendation"),
    ("Analyst ratings for Microsoft", "recommendation"),
    ("What's the target price for Amazon?", "recommendation"),
    ("Should I sell my Google shares?", "recommendation"),
    ("Is Meta a hold or a sell?", "recommendation"),
    ("Buy sell or hold Netflix?", "recommendation"),
    ("What do analysts say about AMD?", "recommendation"),
    ("Is Disney stock a good investment?", "recommendation"),
    ("Analyst outlook for Tesla", "recommendation"),
    ("Should I hold Intel stock?", "recommendation"),
    ("What's the consensus rating for JPM?", "recommendation"),
    ("Is it time to buy the dip on Apple?", "recommendation"),
    ("Upgrade or downgrade news for NVDA?", "recommendation"),
    ("What's the analyst price target for Uber?", "recommendation"),
    ("Is Coinbase a recommended buy?", "recommendation"),
    ("Buy recommendation for Palantir?", "recommendation"),
    ("Should I invest in Shopify?", "recommendation"),
    ("Is it a good time to sell bank stocks?", "recommendation"),
    ("Wall Street ratings for Broadcom", "recommendation"),
    ("Analyst consensus on Adobe stock", "recommendation"),
    ("Should I hold or sell my Microsoft shares?", "recommendation"),
    ("Is Spotify a strong buy?", "recommendation"),
    ("Investment outlook for Oracle", "recommendation"),

    # excel
    ("Download an Excel report for AAPL", "excel"),
    ("Create a spreadsheet with Tesla data", "excel"),
    ("Export NVDA financials to Excel", "excel"),
    ("Generate an xlsx file for Microsoft", "excel"),
    ("I need a spreadsheet comparing these stocks", "excel"),
    ("Download stock data as Excel", "excel"),
    ("Create an Excel report for Amazon fundamentals", "excel"),
    ("Export this to a spreadsheet", "excel"),
    ("Give me an xlsx download for Netflix", "excel"),
    ("Spreadsheet with AMD financial data", "excel"),
    ("Download comparison data as Excel", "excel"),
    ("Create Excel with Disney stock info", "excel"),
    ("Export earnings data to xlsx", "excel"),
    ("Generate a financial spreadsheet for Intel", "excel"),
    ("I want to download this as Excel", "excel"),
    ("Make me a spreadsheet with stock prices", "excel"),
    ("Excel report for JPMorgan please", "excel"),
    ("Export portfolio data to spreadsheet", "excel"),
    ("Create an Excel file with dividends data", "excel"),
    ("Download fundamental analysis as xlsx", "excel"),
    ("Spreadsheet export of the comparison", "excel"),
    ("Generate Excel for Goldman Sachs", "excel"),
    ("Give me a downloadable spreadsheet", "excel"),
    ("Export to Excel with charts data", "excel"),
    ("Create xlsx report for Uber", "excel"),

    # general
    ("Hello", "general"),
    ("Hi there", "general"),
    ("What can you do?", "general"),
    ("Help me with stocks", "general"),
    ("Tell me about yourself", "general"),
    ("How does this work?", "general"),
    ("Thank you", "general"),
    ("Goodbye", "general"),
    ("Who are you?", "general"),
    ("What features do you have?", "general"),
    ("Can you help me invest?", "general"),
    ("I'm new to investing", "general"),
    ("Explain the stock market to me", "general"),
    ("What is a mutual fund?", "general"),
    ("How do ETFs work?", "general"),
    ("Tell me about portfolio diversification", "general"),
    ("What's a good investment strategy?", "general"),
    ("How do I start trading?", "general"),
    ("What is market volatility?", "general"),
    ("Explain options trading", "general"),
    ("How do bonds work?", "general"),
    ("What is a bull market?", "general"),
    ("Thanks for the help", "general"),
    ("Good morning", "general"),
    ("What time does the market open?", "general"),
]

# ── ML Pipeline ──────────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

_model: "Pipeline | None" = None
_is_trained: bool = False


def _train_model() -> "Pipeline | None":
    """Train the TF-IDF + Logistic Regression pipeline on synthetic data."""
    global _model, _is_trained
    if not SKLEARN_AVAILABLE:
        return None
    if _is_trained and _model is not None:
        return _model

    texts = [preprocess_text(t) for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(texts, labels)
    _model = pipeline
    _is_trained = True
    logger.info("ML intent classifier trained on %d examples", len(texts))
    return _model


def predict_intent(text: str) -> Tuple[str, float]:
    """Predict intent using the ML classifier.

    Args:
        text: Raw user query.

    Returns:
        Tuple of (intent_label, confidence) where confidence is the
        probability of the predicted class (0.0–1.0).
    """
    model = _train_model()
    if model is None:
        return ("general", 0.0)

    processed = preprocess_text(text)
    prediction = model.predict([processed])[0]
    probabilities = model.predict_proba([processed])[0]
    confidence = float(max(probabilities))
    return (prediction, round(confidence, 4))


def get_intent_labels() -> list[str]:
    """Return the list of intent labels the classifier knows."""
    model = _train_model()
    if model is None:
        return []
    return list(model.classes_)


def cross_validate(k: int = 5) -> dict:
    """Run stratified k-fold cross-validation on the training data.

    Answers the question: "How well does this model generalize beyond the
    training set?" Uses StratifiedKFold to preserve class distribution.

    Args:
        k: Number of folds (default 5).

    Returns:
        Dict with "fold_accuracies" (list[float]), "mean" (float), "std" (float).

    Raises:
        RuntimeError: If sklearn is not available.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn is required for cross-validation")

    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    texts = [preprocess_text(t) for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies: list[float] = []

    for train_idx, val_idx in skf.split(texts, labels):
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=5000,
                stop_words="english", sublinear_tf=True,
            )),
            ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs")),
        ])
        pipe.fit(train_texts, train_labels)
        preds = pipe.predict(val_texts)
        acc = sum(1 for p, t in zip(preds, val_labels) if p == t) / len(val_labels)
        fold_accuracies.append(round(acc, 4))

    return {
        "fold_accuracies": fold_accuracies,
        "mean": round(float(np.mean(fold_accuracies)), 4),
        "std": round(float(np.std(fold_accuracies)), 4),
        "k": k,
    }


def tune_hyperparameters() -> dict:
    """Grid search over C and ngram_range with stratified 5-fold CV.

    Tests C ∈ [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0] ×
    ngram_range ∈ [(1,1), (1,2), (1,3)] = 24 combinations.

    Returns:
        Dict with "best_params", "best_score", and "all_results" for plotting.

    Raises:
        RuntimeError: If sklearn is not available.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn is required for hyperparameter tuning")

    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    texts = [preprocess_text(t) for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_results: list[dict] = []
    best_score = -1.0
    best_params: dict = {}

    for C in C_values:
        for ngram in ngram_ranges:
            fold_scores: list[float] = []
            for train_idx, val_idx in skf.split(texts, labels):
                train_texts = [texts[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_texts = [texts[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]

                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(
                        ngram_range=ngram, max_features=5000,
                        stop_words="english", sublinear_tf=True,
                    )),
                    ("clf", LogisticRegression(max_iter=1000, C=C, solver="lbfgs")),
                ])
                pipe.fit(train_texts, train_labels)
                preds = pipe.predict(val_texts)
                acc = sum(1 for p, t in zip(preds, val_labels) if p == t) / len(val_labels)
                fold_scores.append(acc)

            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores))
            result = {
                "C": C,
                "ngram_range": ngram,
                "mean_accuracy": round(mean_score, 4),
                "std_accuracy": round(std_score, 4),
            }
            all_results.append(result)

            if mean_score > best_score:
                best_score = mean_score
                best_params = {"C": C, "ngram_range": ngram}

    return {
        "best_params": best_params,
        "best_score": round(best_score, 4),
        "all_results": all_results,
    }


# Eagerly train on import so first call is fast
try:
    _train_model()
except Exception as e:
    logger.warning("Failed to train ML intent classifier: %s", e)
