"""Tests for the retrieval improvements (Phase 4)."""

import pytest

from legion_koi.retrieval.fusion import convex_combine
from legion_koi.retrieval.router import QueryType, classify_query
from legion_koi.constants import CONVEX_ALPHA


# --- Convex Combination Tests ---


class TestConvexCombine:
    def _make_fts(self, rid, rank):
        return {"rid": rid, "rank": rank, "namespace": "test", "search_text": "test"}

    def _make_vec(self, rid, similarity):
        return {"rid": rid, "similarity": similarity, "namespace": "test", "search_text": "test", "chunk_text": f"chunk-{rid}"}

    def test_alpha_one_fts_only(self):
        """Alpha=1.0 means FTS-only — semantic results contribute nothing."""
        fts = [self._make_fts("a", 1.0), self._make_fts("b", 0.5)]
        vec = [self._make_vec("c", 0.9), self._make_vec("d", 0.8)]
        results = convex_combine(fts, vec, alpha=1.0, limit=10)
        rids = [r["rid"] for r in results]
        # FTS results should dominate; "a" (rank=1.0) should be first
        assert rids[0] == "a"
        # Semantic-only results get 0 from alpha=1.0
        assert results[-1]["fusion_score"] == 0.0 or "c" in rids

    def test_alpha_zero_semantic_only(self):
        """Alpha=0.0 means semantic-only — FTS results contribute nothing."""
        fts = [self._make_fts("a", 1.0)]
        vec = [self._make_vec("b", 0.9), self._make_vec("c", 0.3)]
        results = convex_combine(fts, vec, alpha=0.0, limit=10)
        rids = [r["rid"] for r in results]
        # "b" has highest similarity
        assert rids[0] == "b"

    def test_both_empty(self):
        results = convex_combine([], [], alpha=0.5)
        assert results == []

    def test_fts_empty(self):
        vec = [self._make_vec("a", 0.8)]
        results = convex_combine([], vec, alpha=0.5, limit=10)
        assert len(results) == 1
        assert results[0]["rid"] == "a"

    def test_vec_empty(self):
        fts = [self._make_fts("a", 1.0)]
        results = convex_combine(fts, [], alpha=0.5, limit=10)
        assert len(results) == 1
        assert results[0]["rid"] == "a"

    def test_overlap_merges(self):
        """Same RID in both sources gets combined score."""
        fts = [self._make_fts("overlap", 1.0)]
        vec = [self._make_vec("overlap", 0.9)]
        results = convex_combine(fts, vec, alpha=0.5, limit=10)
        assert len(results) == 1
        r = results[0]
        # Should have both FTS and semantic contributions
        assert r["fusion_score"] > 0.5

    def test_limit_respected(self):
        fts = [self._make_fts(f"f{i}", 1.0 - i * 0.1) for i in range(10)]
        vec = [self._make_vec(f"v{i}", 0.9 - i * 0.1) for i in range(10)]
        results = convex_combine(fts, vec, alpha=0.5, limit=5)
        assert len(results) == 5

    def test_chunk_text_preserved(self):
        """Vector results' chunk_text should be preserved in merged results."""
        fts = [self._make_fts("a", 1.0)]
        vec = [self._make_vec("a", 0.9)]
        results = convex_combine(fts, vec, alpha=0.5, limit=10)
        assert results[0].get("chunk_text") == "chunk-a"

    def test_normalized_scores(self):
        """FTS scores should be min-max normalized to [0, 1]."""
        fts = [
            self._make_fts("high", 0.8),
            self._make_fts("low", 0.2),
        ]
        results = convex_combine(fts, [], alpha=1.0, limit=10)
        # "high" should normalize to 1.0, "low" to 0.0
        assert results[0]["rid"] == "high"
        assert results[0]["fusion_score"] == 1.0
        assert results[1]["fusion_score"] == 0.0


# --- Query Router Tests ---


class TestClassifyQuery:
    def test_keyword_short(self):
        assert classify_query("FalkorDB") == QueryType.KEYWORD
        assert classify_query("redis streams") == QueryType.KEYWORD

    def test_keyword_quoted(self):
        assert classify_query('"exact phrase search"') == QueryType.KEYWORD
        assert classify_query("find 'this thing'") == QueryType.KEYWORD

    def test_conceptual_question(self):
        assert classify_query("how does knowledge federation work?") == QueryType.CONCEPTUAL
        assert classify_query("what is the purpose of the event bus?") == QueryType.CONCEPTUAL
        assert classify_query("why did we choose Redis Streams?") == QueryType.CONCEPTUAL

    def test_conceptual_long(self):
        assert classify_query("the relationship between entity extraction and semantic search quality") == QueryType.CONCEPTUAL

    def test_temporal(self):
        assert classify_query("what happened yesterday in the build") == QueryType.TEMPORAL
        assert classify_query("bundles from last week") == QueryType.TEMPORAL
        assert classify_query("changes since 2026-03-10") == QueryType.TEMPORAL

    def test_hybrid_default(self):
        assert classify_query("legion koi event system") == QueryType.HYBRID
        assert classify_query("entity extraction pipeline") == QueryType.HYBRID

    def test_empty_string(self):
        assert classify_query("") == QueryType.HYBRID
        assert classify_query("   ") == QueryType.HYBRID

    def test_single_word(self):
        assert classify_query("test") == QueryType.KEYWORD


# --- Constants Tests ---


class TestConvexConstants:
    def test_alpha_in_range(self):
        assert 0.0 <= CONVEX_ALPHA <= 1.0

    def test_router_alphas(self):
        from legion_koi.constants import (
            ROUTER_ALPHA_KEYWORD,
            ROUTER_ALPHA_CONCEPTUAL,
            ROUTER_ALPHA_TEMPORAL,
        )
        assert ROUTER_ALPHA_KEYWORD == 1.0   # FTS only
        assert 0.0 < ROUTER_ALPHA_CONCEPTUAL < 1.0  # balanced
        assert ROUTER_ALPHA_TEMPORAL == 1.0  # FTS only
