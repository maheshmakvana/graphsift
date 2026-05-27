"""Hybrid BM25 + vector search for code symbols.

Combines BM25 keyword matching with sparse TF-IDF cosine similarity
for improved code symbol retrieval. Uses existing TF-IDF embeddings
stored by _tool_embed_graph in node metadata (``_tfidf`` key).

Usage::

    from graphsift.hybrid_search import HybridSearcher

    searcher = HybridSearcher(alpha=0.3)
    results = searcher.search("auth login handler", nodes, top_k=10)
    for node, score in results:
        print(f"{node.qualified_name}: {score:.4f}")
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Any

from .models import GraphNode

logger = logging.getLogger(__name__)

# Regex for camelCase-aware tokenisation — matches the same pattern used by
# _tool_embed_graph in mcp_server.py so that query vectors align with stored
# TF-IDF term keys.
_TOKEN_RE = re.compile(
    r"[a-zA-Z][a-z]*|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+"
)


class HybridSearcher:
    """Hybrid search combining BM25 keyword matching with TF-IDF vector similarity.

    Each node receives a score of::

        score = alpha * bm25_score + (1 - alpha) * cosine(query_vec, node_tfidf)

    When a node has no ``_tfidf`` vector in its metadata the score falls back
    to the raw BM25 score alone.

    Args:
        alpha: Weight for BM25 signal (0..1).  Vector weight = 1 - alpha.
            Default 0.3 means BM25 contributes 30% and vector 70%.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._alpha = alpha

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        nodes: list[GraphNode],
        top_k: int = 20,
    ) -> list[tuple[GraphNode, float]]:
        """Search *nodes* by hybrid BM25 + sparse-vector score.

        Args:
            query: Free-text search query.
            nodes: Candidate :class:`GraphNode` instances to score.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(node, score)`` tuples sorted descending by score.
        """
        if not query or not nodes:
            return []

        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return []

        # Build a sparse query vector using the same camelCase-aware
        # tokenisation that was used when creating the stored _tfidf vectors.
        query_vec = self._build_query_vector(query)

        scored: list[tuple[GraphNode, float]] = []
        for node in nodes:
            bm25 = self._bm25_score(node, query_tokens)

            tfidf: dict[str, float] | None = node.metadata.get("_tfidf")
            if tfidf and query_vec:
                vec_score = self.sparse_cosine(query_vec, tfidf)
                score = self._alpha * bm25 + (1.0 - self._alpha) * vec_score
            else:
                # No vector available — fall back to BM25-only.
                score = bm25

            if score > 0.0:
                scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Query tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize_query(text: str) -> dict[str, int]:
        """Tokenise a free-text query for BM25 (word-level, lowercased)."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        freq: dict[str, int] = defaultdict(int)
        for t in tokens:
            freq[t] += 1
        return dict(freq)

    @staticmethod
    def _build_query_vector(text: str) -> dict[str, float]:
        """Build a sparse query vector (simple TF) for cosine similarity.

        Uses the same camelCase-aware splitting as ``_tool_embed_graph`` so
        that term keys align with stored ``_tfidf`` vectors.
        """
        tokens = _TOKEN_RE.findall(text)
        tokens = [t.lower() for t in tokens if len(t) > 1]
        if not tokens:
            return {}
        tf: dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        # Normalise by document length (matching the TF part of stored vectors)
        total = len(tokens)
        return {t: c / total for t, c in tf.items()}

    # ------------------------------------------------------------------
    # BM25 scoring for a single GraphNode
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_doc_terms(node: GraphNode) -> list[str]:
        """Extract bag-of-words terms from a node for BM25.

        Splits on word boundaries and underscore boundaries so that
        ``handle_request`` produces ``["handle", "request"]`` — matching
        the behaviour of ``RelevanceRanker._bm25_score`` in ``core.py``.
        """
        text = " ".join(
            filter(
                None,
                [
                    node.name,
                    node.qualified_name,
                    node.signature,
                    node.file_path,
                ],
            )
        )
        words = re.findall(r"\b\w+\b", text.lower())
        terms: list[str] = []
        for w in words:
            terms.extend(w.split("_"))
        return [t for t in terms if t]

    @classmethod
    def _bm25_score(cls, node: GraphNode, query_tokens: dict[str, int]) -> float:
        """Compute a BM25 score for *node* against *query_tokens*.

        Uses the same (simplified, no-corpus-IDF) formulation as
        ``RelevanceRanker._bm25_score`` in ``core.py``.
        """
        if not query_tokens:
            return 0.0

        doc_terms = cls._extract_doc_terms(node)
        doc_freq: dict[str, int] = defaultdict(int)
        for t in doc_terms:
            doc_freq[t] += 1

        k1, b = 1.5, 0.75
        avg_dl = 20.0
        dl = len(doc_terms)
        score = 0.0
        for term in query_tokens:
            if term not in doc_freq:
                continue
            tf = doc_freq[term]
            idf = math.log(1 + 1.0 / (0.5 + 0.5))  # simplified IDF
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm

        return min(1.0, score / max(len(query_tokens), 1))

    # ------------------------------------------------------------------
    # Sparse cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def sparse_cosine(a: dict[str, float], b: dict[str, float]) -> float:
        """Cosine similarity between two sparse TF-IDF dictionaries.

        Computes the dot product over the intersection of keys, normalised by
        the product of L2 norms.  Works with any ``dict[str, float]`` sparse
        vectors — no numpy dependency.

        Args:
            a: First sparse vector (e.g. query TF).
            b: Second sparse vector (e.g. document TF-IDF).

        Returns:
            Cosine similarity in ``[0, 1]``.
        """
        if not a or not b:
            return 0.0

        # Dot product over the smaller dict for efficiency.
        if len(a) > len(b):
            a, b = b, a

        dot = 0.0
        for key, val in a.items():
            if key in b:
                dot += val * b[key]

        if dot == 0.0:
            return 0.0

        # L2 norms
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """The BM25 weight in ``[0, 1]``."""
        return self._alpha

    def __repr__(self) -> str:
        return f"HybridSearcher(alpha={self._alpha})"
