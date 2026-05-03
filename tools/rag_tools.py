"""Retrieval-Augmented Generation (RAG) tool for the Financial AI Agent.

Implements a lightweight RAG pipeline that enriches agent queries with relevant
financial knowledge retrieved from a small document store.

Pipeline:
  1. Query expansion: user query is expanded with financial synonyms for
     broader recall.
  2. Documents are chunked and encoded into dense vector embeddings using
     sentence-transformers (all-MiniLM-L6-v2 — ~90 MB, auto-downloaded once).
  3. At query time, the query is encoded with the same model.
  4. Cosine similarity ranks chunks; the top candidates are retrieved.
  5. Re-ranking: a lightweight keyword-overlap scorer re-ranks the initial
     candidates for improved precision.
  6. Source citations: each chunk carries an ID so the agent can cite sources.

Fallback: When sentence-transformers is unavailable, falls back to TF-IDF
(scikit-learn) for embedding-free retrieval using sparse bag-of-words vectors.

Academic relevance:
  - Demonstrates dense vs. sparse retrieval trade-offs.
  - Shows grounding / hallucination-reduction via retrieved context.
  - Illustrates the standard RAG pattern (encoder → index → retrieve → generate).
  - Re-ranking demonstrates a two-stage retrieval architecture.
  - Query expansion shows recall-improving techniques.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger("rag_tools")

from agno.tools import Toolkit

# ── Embedding backends ────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    DENSE_AVAILABLE = True
except ImportError:
    DENSE_AVAILABLE = False

# Lazy-loaded on first use — avoids 3-5s startup delay from loading 90MB model
_st_model: "SentenceTransformer | None" = None


def _get_st_model() -> "SentenceTransformer | None":
    """Return the sentence-transformers model, loading it on first call."""
    global _st_model
    if _st_model is None and DENSE_AVAILABLE:
        try:
            _st_model = SentenceTransformer("all-mpnet-base-v2")
        except Exception:
            pass
    return _st_model

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine
    SPARSE_AVAILABLE = True
except Exception:
    SPARSE_AVAILABLE = False

# ── Financial synonym map for query expansion ────────────────────────────────
_FINANCIAL_SYNONYMS: dict[str, list[str]] = {
    "pe": ["price to earnings", "P/E ratio", "valuation multiple"],
    "p/e": ["price to earnings", "PE ratio", "valuation"],
    "eps": ["earnings per share", "net income per share"],
    "revenue": ["sales", "top line", "income"],
    "profit": ["earnings", "net income", "bottom line"],
    "dividend": ["dividend yield", "payout", "distribution"],
    "market cap": ["market capitalization", "market value"],
    "rsi": ["relative strength index", "momentum indicator", "overbought oversold"],
    "moving average": ["50-day", "200-day", "trend indicator", "golden cross"],
    "beta": ["volatility", "market risk", "systematic risk"],
    "short": ["short interest", "short squeeze", "bearish"],
    "analyst": ["rating", "consensus", "price target", "recommendation"],
    "sector": ["GICS", "industry", "S&P 500 sectors"],
    "interest rate": ["fed", "federal reserve", "discount rate", "monetary policy"],
    "ebitda": ["earnings before interest tax depreciation amortization", "operating earnings"],
    "debt": ["leverage", "debt to equity", "liabilities"],
    "cash flow": ["free cash flow", "operating cash flow", "FCF"],
}


def expand_query(query: str) -> str:
    """Expand a query with financial synonyms for improved recall.

    Scans the query for known financial terms and appends related synonyms.
    This increases the chance of matching relevant knowledge chunks that
    use different terminology for the same concept.

    Args:
        query: Original user query.

    Returns:
        Expanded query string with appended synonyms.
    """
    query_lower = query.lower()
    expansions: list[str] = []

    for term, synonyms in _FINANCIAL_SYNONYMS.items():
        if term in query_lower:
            expansions.extend(synonyms[:2])  # limit to 2 synonyms per term

    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ── Built-in financial knowledge chunks ──────────────────────────────────────
# Concise reference snippets covering common concepts the agent is asked about.
# In production these would be loaded from a vector database.
# Each chunk has an ID for citation purposes.
_KNOWLEDGE_BASE: list[dict[str, str]] = [
    # Valuation
    {"id": "KB001", "text":
    "The Price-to-Earnings (P/E) ratio divides a stock's current share price by "
    "its earnings per share. A high P/E may signal growth expectations; a low P/E "
    "can indicate undervaluation or declining business prospects."},

    {"id": "KB002", "text":
    "Forward P/E uses projected future earnings rather than trailing (past 12 months) "
    "earnings. Analysts prefer forward P/E when evaluating growth stocks."},

    {"id": "KB003", "text":
    "Enterprise Value (EV) equals market capitalisation plus total debt minus cash. "
    "EV/EBITDA is widely used for comparing companies regardless of capital structure."},

    # Dividends
    {"id": "KB004", "text":
    "Dividend yield is the annual dividend per share divided by the current share price. "
    "High dividend yields can attract income investors but may also signal a falling stock price."},

    {"id": "KB005", "text":
    "A dividend payout ratio above 80% may be unsustainable; companies with strong "
    "free cash flow can safely sustain higher ratios."},

    # Market concepts
    {"id": "KB006", "text":
    "Beta measures a stock's volatility relative to the market. Beta > 1 means more "
    "volatile than the index; Beta < 1 means less volatile. Negative beta stocks tend "
    "to move opposite to the market (e.g. gold miners)."},

    {"id": "KB007", "text":
    "Market capitalisation is share price × shares outstanding. Classifications: "
    "large-cap (> $10B), mid-cap ($2B–$10B), small-cap ($300M–$2B), micro-cap (< $300M)."},

    # Technical analysis
    {"id": "KB008", "text":
    "The 50-day and 200-day moving averages are widely watched trend indicators. "
    "A 'golden cross' (50-day crossing above 200-day) is a bullish signal; a "
    "'death cross' (50-day crossing below 200-day) is bearish."},

    {"id": "KB009", "text":
    "RSI (Relative Strength Index) ranges from 0–100. Above 70 indicates overbought "
    "conditions; below 30 suggests oversold. RSI is a momentum oscillator."},

    # Earnings
    {"id": "KB010", "text":
    "Earnings Per Share (EPS) = net income ÷ diluted shares outstanding. "
    "Beat or miss relative to analyst consensus EPS estimates typically drives "
    "significant post-earnings stock moves."},

    {"id": "KB011", "text":
    "Revenue growth rate = (current quarter revenue − prior year quarter revenue) / "
    "prior year quarter revenue × 100. Consistent double-digit growth signals a strong business."},

    # Risk
    {"id": "KB012", "text":
    "Diversification reduces unsystematic (company-specific) risk but cannot eliminate "
    "systematic (market-wide) risk. Modern Portfolio Theory formalises this with the "
    "efficient frontier."},

    {"id": "KB013", "text":
    "Short interest as a percentage of float above 20% is considered high and can "
    "lead to a short squeeze if the stock rises sharply."},

    # Analyst ratings
    {"id": "KB014", "text":
    "Analyst consensus ratings range from Strong Buy → Buy → Hold → Underperform → Sell. "
    "Price targets represent analysts' 12-month expected share price. A large gap between "
    "the current price and the average target suggests significant upside or downside."},

    # Sectors
    {"id": "KB015", "text":
    "The S&P 500 is divided into 11 GICS sectors: Information Technology, Health Care, "
    "Financials, Consumer Discretionary, Communication Services, Industrials, Consumer "
    "Staples, Energy, Utilities, Real Estate, and Materials."},

    # Interest rates
    {"id": "KB016", "text":
    "Rising interest rates increase the discount rate applied to future cash flows, "
    "which reduces the present value of growth stocks more than value stocks. "
    "This is why tech stocks are sensitive to Fed policy."},
]

# Plain text list for backward compatibility
_KNOWLEDGE_TEXTS: list[str] = [chunk["text"] for chunk in _KNOWLEDGE_BASE]
_CHUNK_IDS: list[str] = [chunk["id"] for chunk in _KNOWLEDGE_BASE]


# Minimum similarity score to include a chunk in results (same for both backends)
MIN_SCORE: float = 0.05


class RetrievedChunk(NamedTuple):
    text: str
    score: float
    chunk_id: str


def _keyword_overlap_score(query: str, document: str) -> float:
    """Compute a keyword overlap re-ranking score between query and document.

    Uses normalized term overlap (Jaccard-like) on lowercased tokens.
    This is a lightweight alternative to cross-encoder re-ranking.

    Args:
        query: The search query.
        document: The document text.

    Returns:
        Float score in [0, 1] representing keyword overlap.
    """
    q_tokens = set(re.findall(r'\w+', query.lower()))
    d_tokens = set(re.findall(r'\w+', document.lower()))
    if not q_tokens or not d_tokens:
        return 0.0
    intersection = q_tokens & d_tokens
    return len(intersection) / len(q_tokens)


class RAGRetriever:
    """Vector store and retrieval engine for financial knowledge chunks.

    Uses sentence-transformers for dense retrieval when available,
    falling back to TF-IDF sparse retrieval otherwise. Includes
    query expansion, re-ranking, and source citation support.
    """

    # Directory for cached embeddings
    _CACHE_DIR = Path(__file__).parent.parent / "outputs" / ".embedding_cache"

    def __init__(self, documents: list[dict[str, str]] | list[str]) -> None:
        # Support both new dict format and legacy plain string list
        if documents and isinstance(documents[0], dict):
            self.documents = [d["text"] for d in documents]
            self.chunk_ids = [d["id"] for d in documents]
        else:
            self.documents = list(documents)
            self.chunk_ids = [f"chunk_{i}" for i in range(len(documents))]
        self._dense_index: np.ndarray | None = None
        self._tfidf: "TfidfVectorizer | None" = None
        self._tfidf_matrix = None
        self._build_index()

    def _content_hash(self) -> str:
        """Compute a hash of all document texts for cache invalidation."""
        combined = "\n".join(self.documents)
        return hashlib.md5(combined.encode()).hexdigest()[:12]

    def _build_index(self) -> None:
        """Encode all documents into the vector index.

        Tries to load pre-computed dense embeddings from disk first.
        If the cache is stale or missing, computes fresh embeddings
        and saves them to disk for next startup.
        """
        if DENSE_AVAILABLE:
            try:
                model = _get_st_model()
                if model is not None:
                    # Try loading cached embeddings from disk
                    cache_file = self._CACHE_DIR / f"kb_embeddings_{self._content_hash()}.npy"
                    if cache_file.exists():
                        self._dense_index = np.load(str(cache_file))
                        expected_dim = model.get_sentence_embedding_dimension()
                        if (self._dense_index.shape[0] == len(self.documents)
                                and self._dense_index.shape[1] == expected_dim):
                            logger.info(
                                "Loaded cached embeddings for %d chunks from %s",
                                len(self.documents), cache_file.name,
                            )
                            return
                        # Shape mismatch (document count or embedding dim) — recompute
                        self._dense_index = None

                    # Compute fresh embeddings
                    self._dense_index = model.encode(
                        self.documents, convert_to_numpy=True, show_progress_bar=False
                    )
                    # Save to disk for next startup
                    try:
                        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        np.save(str(cache_file), self._dense_index)
                        logger.info(
                            "Cached %d chunk embeddings to %s",
                            len(self.documents), cache_file.name,
                        )
                    except Exception:
                        pass  # non-critical — just skip caching
                    return
            except Exception:
                pass  # fall through to TF-IDF
        if SPARSE_AVAILABLE:
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._tfidf_matrix = self._tfidf.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3, use_reranking: bool = True, use_expansion: bool = True) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant document chunks for a query.

        Pipeline: query expansion → initial retrieval → re-ranking.

        Args:
            query: User's financial question.
            top_k: Number of chunks to return.
            use_reranking: Whether to apply keyword-overlap re-ranking.
            use_expansion: Whether to apply query expansion.

        Returns:
            List of RetrievedChunk(text, score, chunk_id) sorted by descending relevance.
        """
        # Step 1: Query expansion
        expanded_query = expand_query(query) if use_expansion else query

        # Step 2: Initial retrieval (fetch more candidates for re-ranking)
        fetch_k = min(top_k * 2, len(self.documents)) if use_reranking else top_k

        if DENSE_AVAILABLE and self._dense_index is not None:
            candidates = self._dense_retrieve(expanded_query, fetch_k)
        elif SPARSE_AVAILABLE and self._tfidf is not None:
            candidates = self._sparse_retrieve(expanded_query, fetch_k)
        else:
            return []

        # Step 3: Re-ranking with keyword overlap
        if use_reranking and len(candidates) > top_k:
            for i, chunk in enumerate(candidates):
                rerank_score = _keyword_overlap_score(query, chunk.text)
                # Combined score: 70% embedding similarity + 30% keyword overlap
                combined = 0.7 * chunk.score + 0.3 * rerank_score
                candidates[i] = RetrievedChunk(
                    text=chunk.text, score=round(combined, 4), chunk_id=chunk.chunk_id
                )
            candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates[:top_k]

    def _dense_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Cosine similarity over sentence-transformer embeddings."""
        model = _get_st_model()
        if model is None or self._dense_index is None:
            return []
        q_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        doc_norms = np.linalg.norm(self._dense_index, axis=1, keepdims=True)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0 or np.any(doc_norms == 0):
            return []
        scores = (self._dense_index / doc_norms) @ (q_vec / q_norm).T
        scores = scores.flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self.documents[i],
                score=round(float(scores[i]), 4),
                chunk_id=self.chunk_ids[i],
            )
            for i in top_indices
            if scores[i] > MIN_SCORE
        ]

    def _sparse_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Cosine similarity over TF-IDF sparse vectors (fallback)."""
        q_vec = self._tfidf.transform([query])
        scores = _sk_cosine(q_vec, self._tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            RetrievedChunk(
                text=self.documents[i],
                score=round(float(scores[i]), 4),
                chunk_id=self.chunk_ids[i],
            )
            for i in top_indices
            if scores[i] > MIN_SCORE
        ]


# Singleton retriever built from the built-in knowledge base
_retriever = RAGRetriever(_KNOWLEDGE_BASE)


class RAGTools(Toolkit):
    """agno Toolkit wrapping the RAG retriever for use by the agent."""

    def __init__(self, retriever: RAGRetriever | None = None):
        super().__init__(name="rag_tools")
        self._retriever = retriever or _retriever
        self.register(self.retrieve_financial_context)

    def retrieve_financial_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant financial knowledge chunks for a user query.

        Use this tool FIRST when the user asks about valuation metrics (P/E, EV/EBITDA),
        analyst ratings, technical indicators (RSI, moving averages), dividends, beta,
        market cap, or any foundational financial concept.

        Args:
            query: The user's question or topic to search for.
            top_k: Number of knowledge chunks to retrieve (default 3, max 5).

        Returns:
            Formatted string with the most relevant knowledge snippets, their
            similarity scores, and source citation IDs. Use these as grounding
            context in your response.
        """
        top_k = max(1, min(int(top_k), 5))
        chunks = self._retriever.retrieve(query, top_k=top_k)

        if not chunks:
            return "No relevant context found in the knowledge base for this query."

        backend = "dense (sentence-transformers)" if self._retriever._dense_index is not None else "sparse (TF-IDF)"
        lines = [f"Retrieved {len(chunks)} relevant chunks via {backend} (with query expansion + re-ranking):\n"]
        for i, chunk in enumerate(chunks, 1):
            lines.append(f"[{i}] [Source: {chunk.chunk_id}] (score={chunk.score:.3f})\n{chunk.text}\n")

        return "\n".join(lines)
