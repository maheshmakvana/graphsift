"""Hybrid BM25 + vector search for code symbols.

Combines BM25 keyword matching with sparse TF-IDF cosine similarity
and optional dense vector embeddings via sentence-transformers for
improved code symbol retrieval.

Uses Reciprocal Rank Fusion (RRF) when multiple retrieval signals
are available, producing more robust rankings than weighted sum.

Usage::

    from graphsift.hybrid_search import HybridSearcher

    searcher = HybridSearcher(alpha=0.3)

    # Original BM25 + sparse TF-IDF
    results = searcher.search("auth login handler", nodes, top_k=10)

    # RRF fusion (BM25 + sparse TF-IDF + optional dense vectors)
    results = searcher.search_rrf("auth login handler", nodes, top_k=10)

    # With dense vector embeddings via sentence-transformers
    results = searcher.search("auth login handler", nodes, method="rrf-dense")
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

    # ------------------------------------------------------------------
    # RRF Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def rrf_fuse(
        rank_lists: list[list[tuple[GraphNode, float]]],
        k: int = 60,
        top_k: int = 20,
    ) -> list[tuple[GraphNode, float]]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion.

        RRF combines rankings from different retrieval methods (BM25,
        sparse vector, dense vector) without requiring score normalization,
        since it operates on rank positions rather than absolute scores.

        Args:
            rank_lists: List of ranked result lists, each ``[(node, score), ...]``.
            k: RRF constant (default 60, standard value).
            top_k: Maximum results to return.

        Returns:
            Fused list of ``(node, rrf_score)`` sorted descending.
        """
        if not rank_lists:
            return []

        rrf_scores: dict[str, float] = {}
        node_map: dict[str, GraphNode] = {}

        for ranked in rank_lists:
            for rank, (node, _) in enumerate(ranked, 1):
                rrf_scores[node.node_id] = rrf_scores.get(node.node_id, 0.0) + 1.0 / (k + rank)
                if node.node_id not in node_map:
                    node_map[node.node_id] = node

        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [(node_map[nid], score) for nid, score in fused[:top_k]]

    def search_rrf(
        self,
        query: str,
        nodes: list[GraphNode],
        top_k: int = 20,
        include_dense: bool = False,
    ) -> list[tuple[GraphNode, float]]:
        """Search using RRF fusion of BM25 + sparse TF-IDF (and optional dense).

        Args:
            query: Free-text search query.
            nodes: Candidate GraphNode instances.
            top_k: Maximum results to return.
            include_dense: If True, also run dense vector search via
                sentence-transformers (lazy-loaded). Requires
                ``sentence-transformers`` to be installed.

        Returns:
            List of ``(node, rrf_score)`` sorted descending.
        """
        if not query or not nodes:
            return []

        # BM25 ranking
        bm25_results = self._rank_bm25(query, nodes, top_k=top_k * 2)
        if not bm25_results:
            return []

        # Sparse TF-IDF ranking
        query_vec = self._build_query_vector(query)
        sparse_results: list[tuple[GraphNode, float]] = []
        for node in nodes:
            tfidf = node.metadata.get("_tfidf")
            if tfidf and query_vec:
                score = self.sparse_cosine(query_vec, tfidf)
                if score > 0:
                    sparse_results.append((node, score))
        sparse_results.sort(key=lambda x: x[1], reverse=True)

        rank_lists: list[list[tuple[GraphNode, float]]] = [bm25_results, sparse_results]

        # Dense vector search (optional)
        if include_dense:
            try:
                dense_results = self._search_dense(query, nodes, top_k=top_k * 2)
                if dense_results:
                    rank_lists.append(dense_results)
            except ImportError:
                pass  # sentence-transformers not installed — skip dense

        return self.rrf_fuse(rank_lists, top_k=top_k)

    # ------------------------------------------------------------------
    # Dense vector search via sentence-transformers
    # ------------------------------------------------------------------

    def _search_dense(
        self,
        query: str,
        nodes: list[GraphNode],
        top_k: int = 20,
    ) -> list[tuple[GraphNode, float]]:
        """Search using dense embeddings via sentence-transformers.

        Lazy-loads the model — ``sentence-transformers`` is an optional
        dependency that is only imported when this method is called.

        Generates embeddings on first call and caches the model.
        For accuracy, encodes both the query and a concatenation of
        ``qualified_name + signature`` for each node.

        Args:
            query: Free-text search query.
            nodes: Candidate GraphNode instances.
            top_k: Maximum results to return.

        Returns:
            List of ``(node, cosine_similarity)`` sorted descending.
        """
        import numpy as np  # noqa: PLC0415

        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        # Lazy-load model (cached in class attribute)
        if not hasattr(HybridSearcher, "_dense_model"):
            HybridSearcher._dense_model = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )

        model = HybridSearcher._dense_model  # type: ignore[attr-defined]

        # Build node texts for embedding
        node_texts = [
            f"{n.qualified_name} {n.signature[:200]} {n.name}"
            for n in nodes
        ]
        if not node_texts:
            return []

        # Encode query and nodes (batched for efficiency)
        query_emb = model.encode(query, normalize_embeddings=True)
        node_embs = model.encode(node_texts, normalize_embeddings=True)

        # Compute cosine similarities (dot product on normalized vecs)
        scores = np.dot(node_embs, query_emb)

        scored = [(nodes[i], float(scores[i])) for i in range(len(nodes)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Internal ranking helpers
    # ------------------------------------------------------------------

    def _rank_bm25(
        self,
        query: str,
        nodes: list[GraphNode],
        top_k: int = 20,
    ) -> list[tuple[GraphNode, float]]:
        """Rank nodes by BM25 score only."""
        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return []

        scored: list[tuple[GraphNode, float]] = []
        for node in nodes:
            score = self._bm25_score(node, query_tokens)
            if score > 0:
                scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Extended search method
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        nodes: list[GraphNode],
        top_k: int = 20,
        method: str = "hybrid",
    ) -> list[tuple[GraphNode, float]]:
        """Search *nodes* by hybrid BM25 + vector score.

        Args:
            query: Free-text search query.
            nodes: Candidate :class:`GraphNode` instances to score.
            top_k: Maximum number of results to return.
            method: One of ``"hybrid"`` (default, alpha-weighted BM25 + sparse),
                ``"rrf"`` (RRF fusion of BM25 + sparse), or
                ``"rrf-dense"`` (RRF with BM25 + sparse + dense vectors).

        Returns:
            List of ``(node, score)`` tuples sorted descending by score.
        """
        if method == "rrf":
            return self.search_rrf(query, nodes, top_k=top_k, include_dense=False)
        if method == "rrf-dense":
            return self.search_rrf(query, nodes, top_k=top_k, include_dense=True)
        # Default: original alpha-weighted hybrid
        return self._search_hybrid(query, nodes, top_k=top_k)

    def _search_hybrid(
        self,
        query: str,
        nodes: list[GraphNode],
        top_k: int = 20,
    ) -> list[tuple[GraphNode, float]]:
        """Original alpha-weighted hybrid search (BM25 + sparse TF-IDF)."""
        if not query or not nodes:
            return []

        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return []

        query_vec = self._build_query_vector(query)

        scored: list[tuple[GraphNode, float]] = []
        for node in nodes:
            bm25 = self._bm25_score(node, query_tokens)

            tfidf = node.metadata.get("_tfidf")
            if tfidf and query_vec:
                vec_score = self.sparse_cosine(query_vec, tfidf)
                score = self._alpha * bm25 + (1.0 - self._alpha) * vec_score
            else:
                score = bm25

            if score > 0.0:
                scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @property
    def alpha(self) -> float:
        """The BM25 weight in ``[0, 1]``."""
        return self._alpha

    def __repr__(self) -> str:
        return f"HybridSearcher(alpha={self._alpha})"
