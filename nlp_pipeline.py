"""NLP preprocessing pipeline for the Financial AI Agent.

Provides four NLP capabilities that run before the LLM agent on every query:

1. Named Entity Recognition (NER) — spaCy en_core_web_sm identifies organisations,
   products, and people; a lookup table maps known company names to ticker symbols.
   Bare TICKER patterns (ALL_CAPS 1-5 chars) are also detected via regex.
   Financial-specific patterns detect FIN_METRIC, DATE_REF, and MONEY entities.

2. Intent Classification — ML-based (TF-IDF + Logistic Regression) as primary,
   with keyword-pattern scoring as fallback. Returns both intent and confidence.

3. Sentiment Analysis — VADER (Valence Aware Dictionary and sEntiment Reasoner)
   produces a compound polarity score and a positive / negative / neutral label.
   VADER requires no model download and is tuned for short, informal text.

All three components degrade gracefully when optional libraries are unavailable.
"""

import re
from typing import TypedDict

# ── spaCy NER ────────────────────────────────────────────────────────────────
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    _nlp = None

# ── VADER Sentiment ──────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False
    _vader = None

# ── ML Intent Classifier ────────────────────────────────────────────────────
try:
    from intent_classifier import predict_intent as _ml_predict_intent
    ML_CLASSIFIER_AVAILABLE = True
except Exception:
    ML_CLASSIFIER_AVAILABLE = False
    _ml_predict_intent = None

# ── Company name → ticker lookup ─────────────────────────────────────────────
ORG_TO_TICKER: dict[str, str | None] = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "salesforce": "CRM",
    "intel": "INTC",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "paypal": "PYPL",
    "shopify": "SHOP",
    "disney": "DIS",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "coinbase": "COIN",
    "palantir": "PLTR",
    "openai": None,
    "berkshire": "BRK-B",
    "jpmorgan": "JPM",
    "j.p. morgan": "JPM",
    "goldman sachs": "GS",
    "bank of america": "BAC",
    "twitter": "X",
    "oracle": "ORCL",
    "ibm": "IBM",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "adobe": "ADBE",
    "snap": "SNAP",
    "spotify": "SPOT",
    "doordash": "DASH",
    "robinhood": "HOOD",
}

# ── Intent keyword patterns ──────────────────────────────────────────────────
_INTENT_PATTERNS: dict[str, list[str]] = {
    "stock_price": [
        r"\bprice\b", r"\bcurrent\b", r"\btoday\b", r"\bhow much\b",
        r"\bworth\b", r"\btrading at\b", r"\bquote\b", r"\bvalue\b",
        r"\bclose[sd]?\b",
    ],
    "chart": [
        r"\bchart\b", r"\bgraph\b", r"\bplot\b", r"\bvisuali[sz]e?\b",
        r"\bhistory\b", r"\btrend\b", r"\bmovement\b", r"\bsho[uw] me\b",
    ],
    "comparison": [
        r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b", r"\bbetter\b",
        r"\bdifference\b", r"\bwhich\b", r"\bboth\b", r"\bagainst\b",
    ],
    "news": [
        r"\bnews\b", r"\blatest\b", r"\brecent\b", r"\bheadlines\b",
        r"\bupdate\b", r"\bannounce\b", r"\brelease\b", r"\breport\b",
    ],
    "fundamentals": [
        r"\bpe ratio\b", r"\bp/e\b", r"\bearnings\b", r"\brevenue\b",
        r"\beps\b", r"\bdividend\b", r"\bmarket cap\b", r"\bfundamental\b",
        r"\bprofit\b", r"\bmargin\b", r"\bdebt\b", r"\bcash flow\b",
    ],
    "recommendation": [
        r"\brecommend\b", r"\banalyst\b", r"\bbuy\b", r"\bsell\b",
        r"\bhold\b", r"\brating\b", r"\boutlook\b", r"\btarget price\b",
        r"\bupgrade\b", r"\bdowngrade\b",
    ],
    "excel": [
        r"\bexcel\b", r"\bspreadsheet\b", r"\bdownload\b",
        r"\bexport\b", r"\bxlsx\b",
    ],
}

# All-caps words that are NOT stock tickers
_TICKER_STOPWORDS: set[str] = {
    "I", "A", "AI", "OK", "US", "IT", "IS", "BE", "DO", "GO",
    "THE", "AND", "FOR", "OR", "IN", "TO", "OF", "AT", "ME",
    "MY", "WE", "HE", "SHE", "AS", "BY", "AN", "ON", "UP",
    "NO", "SO", "IF", "BUT", "NOT", "CAN", "AM", "ARE",
    "PE", "CEO", "CFO", "IPO", "ETF", "GDP", "FED", "SEC",
    "NAV", "ROI", "ROE", "EPS", "YOY", "QOQ", "TTM",
}

# ── Financial entity regex patterns ──────────────────────────────────────────
# FIN_METRIC: common financial metrics and indicators
_FIN_METRIC_PATTERN = re.compile(
    r'\b('
    r'P/E\s*ratio|PE\s*ratio|P/E|'
    r'EPS|'
    r'RSI|'
    r'EBITDA|'
    r'ROI|ROE|ROA|'
    r'ROIC|'
    r'EV/EBITDA|'
    r'P/B\s*ratio|PB\s*ratio|'
    r'P/S\s*ratio|PS\s*ratio|'
    r'NAV|'
    r'DCF|'
    r'WACC|'
    r'beta|'
    r'market\s*cap(?:italization)?|'
    r'free\s*cash\s*flow|'
    r'dividend\s*yield|'
    r'debt[- ]to[- ]equity'
    r')\b',
    re.IGNORECASE
)

# DATE_REF: fiscal quarters and years
_DATE_REF_PATTERN = re.compile(
    r'\b('
    r'Q[1-4]\s*\d{4}|'           # Q1 2024
    r'FY\s*\d{4}|'               # FY2023
    r'H[12]\s*\d{4}|'            # H1 2024
    r'\d{4}\s*Q[1-4]|'           # 2024 Q1
    r'(?:first|second|third|fourth)\s+quarter\s+\d{4}'  # first quarter 2024
    r')\b',
    re.IGNORECASE
)

# MONEY: currency amounts
_MONEY_PATTERN = re.compile(
    r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[BMKTbmkt](?:illion|illion)?)?'  # $150, $1.2B, $500M, $3.50
)


class NLPMetadata(TypedDict):
    """Structured output from the NLP preprocessing pipeline."""
    entities: list[dict]       # [{text, label, ticker|None}]
    intent: str                # one of the INTENT_PATTERNS keys or "general"
    intent_confidence: float   # ML classifier confidence (0.0–1.0), 0 for keyword
    intent_method: str         # "ml" or "keyword"
    sentiment: dict            # {label, score, compound}
    spacy_available: bool
    vader_available: bool


# ── Public API ───────────────────────────────────────────────────────────────

def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text and map companies to ticker symbols.

    Uses spaCy NER for proper named entities (ORG, PERSON, PRODUCT), regex
    for bare uppercase ticker patterns, and financial-specific patterns for
    FIN_METRIC, DATE_REF, and MONEY entities.

    Args:
        text: Raw user input string.

    Returns:
        List of dicts: [{"text": str, "label": str, "ticker": str | None}]
    """
    entities: list[dict] = []
    seen: set[str] = set()

    # spaCy pass — proper named entity recognition
    if SPACY_AVAILABLE and _nlp is not None:
        doc = _nlp(text)
        for ent in doc.ents:
            if ent.label_ not in ("ORG", "PERSON", "GPE", "PRODUCT"):
                continue
            key = ent.text.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            ticker = ORG_TO_TICKER.get(key)
            entities.append({"text": ent.text, "label": ent.label_, "ticker": ticker})

    # Financial metric patterns (FIN_METRIC) — run before ticker regex
    # so that terms like RSI, EPS are classified as metrics, not tickers
    for match in _FIN_METRIC_PATTERN.finditer(text):
        metric_text = match.group(0).strip()
        key = metric_text.lower()
        if key not in seen:
            seen.add(key)
            entities.append({"text": metric_text, "label": "FIN_METRIC", "ticker": None})

    # Date/quarter reference patterns (DATE_REF)
    for match in _DATE_REF_PATTERN.finditer(text):
        date_text = match.group(0).strip()
        key = date_text.lower()
        if key not in seen:
            seen.add(key)
            entities.append({"text": date_text, "label": "DATE_REF", "ticker": None})

    # Currency amount patterns (MONEY)
    for match in _MONEY_PATTERN.finditer(text):
        money_text = match.group(0).strip()
        key = money_text.lower()
        if key not in seen:
            seen.add(key)
            entities.append({"text": money_text, "label": "MONEY", "ticker": None})

    # Regex pass — detect bare ticker symbols (ALL_CAPS, 1–5 chars)
    for match in re.finditer(r'\b([A-Z]{1,5})\b', text):
        t = match.group(1)
        if t in _TICKER_STOPWORDS or t.lower() in seen:
            continue
        seen.add(t.lower())
        entities.append({"text": t, "label": "TICKER", "ticker": t})

    return entities


def classify_intent(text: str) -> str:
    """Classify user query into a financial intent category (keyword-based).

    Each intent has a set of regex patterns. The intent whose patterns match
    the most times wins. Ties are broken by dict insertion order.

    Args:
        text: Raw user input string.

    Returns:
        Intent string: "stock_price" | "chart" | "comparison" | "news" |
        "fundamentals" | "recommendation" | "excel" | "general"
    """
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for intent, patterns in _INTENT_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower, re.IGNORECASE))
        if score > 0:
            scores[intent] = score

    return max(scores, key=lambda k: scores[k]) if scores else "general"


def classify_intent_with_confidence(text: str) -> tuple[str, float, str]:
    """Classify intent using ML classifier (primary) or keyword fallback.

    Args:
        text: Raw user input string.

    Returns:
        Tuple of (intent, confidence, method) where method is "ml" or "keyword".
    """
    if ML_CLASSIFIER_AVAILABLE and _ml_predict_intent is not None:
        try:
            intent, confidence = _ml_predict_intent(text)
            return (intent, confidence, "ml")
        except Exception:
            pass

    # Keyword fallback — confidence is 0 since keyword matching is rule-based
    return (classify_intent(text), 0.0, "keyword")


def analyze_sentiment(text: str) -> dict:
    """Score text sentiment using VADER.

    VADER assigns four scores (positive, negative, neutral, compound) without
    requiring a pre-trained neural model, making it fast and dependency-light.
    The compound score ranges from -1 (most negative) to +1 (most positive).

    Args:
        text: Text to score (user query or agent response).

    Returns:
        {"label": "positive"|"negative"|"neutral"|"unavailable",
         "score": float,   # confidence of the winning class
         "compound": float # VADER compound score (-1 to +1)}
    """
    if not VADER_AVAILABLE or _vader is None:
        return {"label": "unavailable", "score": 0.0, "compound": 0.0}

    raw = _vader.polarity_scores(text)
    compound = raw["compound"]

    if compound >= 0.1:
        label = "positive"
    elif compound <= -0.1:
        label = "negative"
    else:
        label = "neutral"

    confidence = round(max(raw["pos"], raw["neg"], raw["neu"]), 3)
    return {"label": label, "score": confidence, "compound": round(compound, 3)}


def run_nlp_pipeline(text: str) -> NLPMetadata:
    """Run the full NLP preprocessing pipeline on a user query.

    Executes entity extraction, intent classification (ML primary, keyword
    fallback), and sentiment analysis in sequence. Designed to complete in
    < 100 ms as a fast preprocessing step before the LLM agent is called.

    Args:
        text: Raw user input message.

    Returns:
        NLPMetadata with entities, intent, confidence, sentiment, and flags.
    """
    intent, confidence, method = classify_intent_with_confidence(text)

    return NLPMetadata(
        entities=extract_entities(text),
        intent=intent,
        intent_confidence=confidence,
        intent_method=method,
        sentiment=analyze_sentiment(text),
        spacy_available=SPACY_AVAILABLE,
        vader_available=VADER_AVAILABLE,
    )
