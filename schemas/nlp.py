"""NLP-domain Pydantic models and enums.

Contains the typed representations of NER entities, sentiment analysis
results, and the full NLP preprocessing metadata that is attached to
every chat turn.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────
class IntentMethod(str, Enum):
    """How the intent was determined."""

    ml = "ml"
    keyword = "keyword"


class SentimentLabel(str, Enum):
    """Coarse sentiment bucket."""

    positive = "positive"
    negative = "negative"
    neutral = "neutral"


# ── Sub-models ───────────────────────────────────────────────────────
class Entity(BaseModel):
    """A single named entity extracted by the NLP pipeline."""

    text: str = Field(..., description="Surface text of the entity.")
    label: str = Field(..., description="Entity type (ORG, PERSON, FIN_METRIC, …).")
    ticker: str | None = Field(
        default=None,
        description="Stock ticker symbol if the entity maps to one.",
    )

    model_config = {"frozen": True}


class SentimentResult(BaseModel):
    """VADER sentiment analysis output."""

    label: SentimentLabel = Field(..., description="Coarse sentiment label.")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    compound: float = Field(
        ..., ge=-1.0, le=1.0, description="VADER compound polarity score."
    )

    model_config = {"frozen": True}


class NLPMetadataSchema(BaseModel):
    """Full NLP preprocessing result attached to every chat turn.

    Can be constructed from the internal ``NLPMetadata`` TypedDict via
    ``NLPMetadataSchema.model_validate(nlp_meta_dict)``.
    """

    entities: list[Entity] = Field(default_factory=list)
    intent: str = Field(..., description="Classified intent key.")
    intent_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Classifier confidence."
    )
    intent_method: str = Field(
        default="keyword", description="'ml' or 'keyword'."
    )
    sentiment: SentimentResult
    spacy_available: bool = False
    vader_available: bool = False

    model_config = {"frozen": True}


class NLPStatusDetail(BaseModel):
    """NLP component availability flags (nested in ``HealthResponse``)."""

    spacy_ner: bool = False
    vader_sentiment: bool = False

    model_config = {"frozen": True}
