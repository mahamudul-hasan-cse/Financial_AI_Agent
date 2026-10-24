"""Shared loader for the spaCy model.

Both the NLP pipeline (NER) and the intent classifier (lemmatisation) use
the same spaCy model so only one copy is held in memory.  The model name
comes from the ``SPACY_MODEL`` env var (default ``en_core_web_sm``).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

SPACY_MODEL_NAME: str = settings.spacy_model


@lru_cache(maxsize=1)
def load_spacy_model() -> Any | None:
    """Load and cache the configured spaCy model, or return ``None`` if unavailable."""
    try:
        import spacy

        return spacy.load(SPACY_MODEL_NAME)
    except Exception:
        logger.warning(
            "spaCy model %r could not be loaded — install it with "
            "`python -m spacy download %s`. Falling back to regex-only NER.",
            SPACY_MODEL_NAME,
            SPACY_MODEL_NAME,
        )
        return None
