"""Tests for graphsift hybrid search (BM25 + TF-IDF vector similarity)."""

from __future__ import annotations

import math

import pytest

from graphsift.hybrid_search import HybridSearcher
from graphsift.models import GraphNode, Language, NodeKind


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def searcher() -> HybridSearcher:
    return HybridSearcher(alpha=0.3)


@pytest.fixture
def nodes() -> list[GraphNode]:
    """A small set of GraphNode fixtures covering various scenarios."""
    return [
        GraphNode(
            node_id="auth.py::authenticate",
            file_path="src/auth.py",
            kind=NodeKind.FUNCTION,
            name="authenticate",
            qualified_name="auth.authenticate",
            signature="def authenticate(username: str, password: str) -> bool",
            line_start=10,
            line_end=25,
            language=Language.PYTHON,
            metadata={
                "_tfidf": {
                    "authenticate": 0.35,
                    "username": 0.25,
                    "password": 0.25,
                    "auth": 0.15,
                },
                "docstring": "Authenticate a user.",
            },
        ),
        GraphNode(
            node_id="auth.py::login",
            file_path="src/auth.py",
            kind=NodeKind.FUNCTION,
            name="login",
            qualified_name="auth.login",
            signature="def login(credentials: dict) -> str",
            line_start=30,
            line_end=45,
            language=Language.PYTHON,
            metadata={
                "_tfidf": {
                    "login": 0.40,
                    "credentials": 0.30,
                    "session": 0.20,
                    "auth": 0.10,
                },
            },
        ),
        GraphNode(
            node_id="auth.py::AuthHandler",
            file_path="src/auth.py",
            kind=NodeKind.CLASS,
            name="AuthHandler",
            qualified_name="auth.AuthHandler",
            signature="class AuthHandler:",
            line_start=1,
            line_end=8,
            language=Language.PYTHON,
            metadata={
                "_tfidf": {
                    "auth": 0.30,
                    "handler": 0.30,
                    "authenticate": 0.25,
                    "token": 0.15,
                },
            },
        ),
        # Node WITHOUT TF-IDF vector (for fallback testing)
        GraphNode(
            node_id="utils.py::format_date",
            file_path="src/utils.py",
            kind=NodeKind.FUNCTION,
            name="format_date",
            qualified_name="utils.format_date",
            signature="def format_date(timestamp: int) -> str",
            line_start=5,
            line_end=12,
            language=Language.PYTHON,
            metadata={},  # No _tfidf key
        ),
        # Node from a completely unrelated area
        GraphNode(
            node_id="payments.py::process_payment",
            file_path="src/payments.py",
            kind=NodeKind.FUNCTION,
            name="process_payment",
            qualified_name="payments.process_payment",
            signature="def process_payment(amount: float, currency: str) -> str",
            line_start=50,
            line_end=65,
            language=Language.PYTHON,
            metadata={
                "_tfidf": {
                    "process": 0.30,
                    "payment": 0.35,
                    "amount": 0.20,
                    "currency": 0.15,
                },
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Sparse cosine similarity
# ---------------------------------------------------------------------------


class TestSparseCosine:
    def test_identical_vectors(self):
        v = {"foo": 0.5, "bar": 0.3}
        assert HybridSearcher.sparse_cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = {"foo": 0.5, "bar": 0.3}
        b = {"baz": 0.7, "qux": 0.2}
        assert HybridSearcher.sparse_cosine(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = {"foo": 1.0, "bar": 0.0}
        b = {"foo": 1.0, "baz": 0.0}
        # Both have norm 1.0, dot product = 1.0*1.0 = 1.0
        assert HybridSearcher.sparse_cosine(a, b) == pytest.approx(1.0)

    def test_partial_overlap_different_magnitudes(self):
        a = {"foo": 1.0, "bar": 2.0}
        b = {"foo": 2.0, "baz": 1.0}
        # dot = 1*2 + 2*0 + 0*1 = 2
        # ||a|| = sqrt(1 + 4) = sqrt(5) ≈ 2.236
        # ||b|| = sqrt(4 + 1) = sqrt(5) ≈ 2.236
        # cos = 2 / 5 = 0.4
        assert HybridSearcher.sparse_cosine(a, b) == pytest.approx(0.4)

    def test_empty_dict(self):
        assert HybridSearcher.sparse_cosine({}, {"a": 1.0}) == pytest.approx(0.0)
        assert HybridSearcher.sparse_cosine({"a": 1.0}, {}) == pytest.approx(0.0)
        assert HybridSearcher.sparse_cosine({}, {}) == pytest.approx(0.0)

    def zero_norm_vector(self):
        """Vector with zero norm (all zero or empty) should return 0."""
        assert HybridSearcher.sparse_cosine({"a": 0.0}, {"a": 1.0}) == pytest.approx(0.0)

    def test_iterates_smaller_dict(self):
        """sparse_cosine should iterate over the smaller dict for efficiency."""
        big = {str(i): float(i) for i in range(100)}
        small = {"key": 1.0}
        # Should not raise and should compute correctly
        result = HybridSearcher.sparse_cosine(big, small)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Hybrid search — scoring
# ---------------------------------------------------------------------------


class TestHybridSearch:
    def test_search_ranks_relevant_highest(self, searcher, nodes):
        """Search for 'auth login' should rank auth/login/authenticate highest."""
        results = searcher.search("auth login", nodes, top_k=5)
        assert len(results) > 0

        names = [n.name for n, _ in results]
        # Auth-related nodes should appear before payment-related ones
        auth_idx = next(i for i, name in enumerate(names) if "auth" in name.lower())
        payment_idx = next(
            (i for i, name in enumerate(names) if "payment" in name.lower()),
            len(names),
        )
        assert auth_idx < payment_idx, (
            f"Auth-related nodes should rank higher than payment. Order: {names}"
        )

    def test_search_payment_query(self, searcher, nodes):
        """Search for 'process payment' should rank payments.process_payment first."""
        results = searcher.search("process payment", nodes, top_k=5)
        assert len(results) > 0
        top_name = results[0][0].name
        assert "payment" in top_name.lower(), (
            f"Expected payment-related node first, got {top_name}"
        )

    def test_search_returns_scores(self, searcher, nodes):
        """All returned items should have a score > 0."""
        results = searcher.search("authenticate", nodes, top_k=5)
        for node, score in results:
            assert score > 0.0, f"Node {node.name} has zero score"
            assert score <= 1.0, f"Node {node.name} has score > 1.0"

    def test_search_sorted_descending(self, searcher, nodes):
        """Results should be sorted by score descending."""
        results = searcher.search("auth", nodes, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), (
            f"Scores not sorted descending: {scores}"
        )

    def test_empty_query_returns_empty(self, searcher, nodes):
        assert searcher.search("", nodes) == []

    def test_empty_nodes_returns_empty(self, searcher):
        assert searcher.search("auth", []) == []

    def test_top_k_respected(self, searcher, nodes):
        results = searcher.search("auth", nodes, top_k=2)
        assert len(results) <= 2

    def test_alpha_zero_pure_vector(self, nodes):
        """alpha=0 should use only vector similarity."""
        s = HybridSearcher(alpha=0.0)
        results = s.search("authenticate password", nodes, top_k=5)
        assert len(results) > 0
        # authenticate node should rank high since it has strong tfidf match
        top_name = results[0][0].name
        assert top_name in ("authenticate", "AuthHandler")

    def test_alpha_one_pure_bm25(self, nodes):
        """alpha=1 should use only BM25."""
        s = HybridSearcher(alpha=1.0)
        results = s.search("format date", nodes, top_k=5)
        assert len(results) > 0
        # format_date should rank first via BM25 (no tfidf on that node)
        assert results[0][0].name == "format_date"


# ---------------------------------------------------------------------------
# Fallback behaviour (no TF-IDF vectors)
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_no_tfidf(self):
        """Nodes without _tfidf should still be findable via BM25."""
        s = HybridSearcher(alpha=0.3)
        nodes = [
            GraphNode(
                node_id="utils.py::greet",
                file_path="src/utils.py",
                kind=NodeKind.FUNCTION,
                name="greet",
                qualified_name="utils.greet",
                signature="def greet(name: str) -> str",
                line_start=1,
                line_end=3,
                language=Language.PYTHON,
                metadata={},
            ),
            GraphNode(
                node_id="core.py::run",
                file_path="src/core.py",
                kind=NodeKind.FUNCTION,
                name="run",
                qualified_name="core.run",
                signature="def run() -> None",
                line_start=10,
                line_end=15,
                language=Language.PYTHON,
                metadata={},
            ),
        ]
        results = s.search("greet", nodes, top_k=5)
        assert len(results) == 1
        assert results[0][0].name == "greet"

    def test_mixed_tfidf_availability(self, searcher, nodes):
        """Mixed nodes (some with tfidf, some without) should all be scorable."""
        results = searcher.search("auth format date", nodes, top_k=10)
        assert len(results) > 1
        # Both auth nodes and format_date should appear
        names = {n.name for n, _ in results}
        assert "format_date" in names, "format_date should appear via BM25 fallback"
        assert any("auth" in n.name.lower() for n, _ in results), (
            "Auth-related nodes should appear"
        )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError):
            HybridSearcher(alpha=-0.1)

    def test_invalid_alpha_above_one(self):
        with pytest.raises(ValueError):
            HybridSearcher(alpha=1.5)

    def test_valid_alphas(self):
        for alpha in [0.0, 0.3, 0.5, 1.0]:
            s = HybridSearcher(alpha=alpha)
            assert s.alpha == alpha

    def test_default_alpha(self):
        s = HybridSearcher()
        assert s.alpha == 0.3


# ---------------------------------------------------------------------------
# BM25 scoring internals
# ---------------------------------------------------------------------------


class TestBM25:
    """Verify BM25 scoring behaviour on individual GraphNodes."""

    def test_bm25_direct_match(self, searcher):
        node = GraphNode(
            node_id="api.py::handle_request",
            file_path="src/api.py",
            kind=NodeKind.FUNCTION,
            name="handle_request",
            qualified_name="api.handle_request",
            signature="",
            line_start=1,
            line_end=5,
            language=Language.PYTHON,
            metadata={},
        )
        query_tokens = searcher._tokenize_query("handle request")
        score = searcher._bm25_score(node, query_tokens)
        assert score > 0.0
        assert score <= 1.0

    def test_bm25_no_match(self, searcher):
        node = GraphNode(
            node_id="api.py::handle_request",
            file_path="src/api.py",
            kind=NodeKind.FUNCTION,
            name="handle_request",
            qualified_name="api.handle_request",
            signature="",
            line_start=1,
            line_end=5,
            language=Language.PYTHON,
            metadata={},
        )
        query_tokens = searcher._tokenize_query("xyzzy_nonexistent")
        score = searcher._bm25_score(node, query_tokens)
        assert score == 0.0

    def test_bm25_empty_query(self, searcher):
        node = GraphNode(
            node_id="api.py::handle_request",
            file_path="src/api.py",
            kind=NodeKind.FUNCTION,
            name="handle_request",
            qualified_name="api.handle_request",
            signature="",
            line_start=1,
            line_end=5,
            language=Language.PYTHON,
            metadata={},
        )
        assert searcher._bm25_score(node, {}) == 0.0


# ---------------------------------------------------------------------------
# Query vector building
# ---------------------------------------------------------------------------


class TestQueryVector:
    def test_build_query_vector_aligns_with_tfidf_terms(self):
        """Query vector terms should use camelCase splitting matching _tfidf keys."""
        vec = HybridSearcher._build_query_vector("AuthHandler")
        # Should split "AuthHandler" -> ["auth", "handler"]
        assert "auth" in vec
        assert "handler" in vec

    def test_query_vector_snake_case(self):
        vec = HybridSearcher._build_query_vector("format_date")
        # Should produce ["format", "date"]
        assert "format" in vec
        assert "date" in vec

    def test_query_vector_empty(self):
        assert HybridSearcher._build_query_vector("") == {}

    def test_query_vector_short_tokens_skipped(self):
        """Single-char tokens should be excluded."""
        vec = HybridSearcher._build_query_vector("a b c")
        assert vec == {}
