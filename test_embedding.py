"""
Test suite for the Hybrid Retrieval Upgrade:
  - BGE (BAAI/bge-small-en-v1.5) embedding service
  - Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
  - Hybrid ranking (TF-IDF 30% + BGE 70%)
  - Graceful fallback paths
  - API response field compatibility
"""

import unittest
import numpy as np


class TestEmbeddingService(unittest.TestCase):
    """Tests for BGE EmbeddingService singleton."""

    @classmethod
    def setUpClass(cls):
        from backend.embedding_service import embedding_service, cross_encoder_service
        cls.svc = embedding_service
        cls.ce_svc = cross_encoder_service

    # ─── 1. Model loading ────────────────────────────────────────────────────
    def test_bge_model_loaded(self):
        self.assertTrue(
            self.svc.available,
            "BGE model should load successfully"
        )
        print("[OK] BGE model loaded")

    def test_cross_encoder_model_loaded(self):
        self.assertTrue(
            self.ce_svc.available,
            "CrossEncoder model should load successfully"
        )
        print("[OK] CrossEncoder model loaded")

    # ─── 2. Query embedding dimension ────────────────────────────────────────
    def test_query_embedding_dimension(self):
        emb = self.svc.encode_query("transformers and reasoning")
        self.assertEqual(len(emb), 384, f"Expected 384, got {len(emb)}")
        print(f"[OK] Query embedding dimension = {len(emb)}")

    # ─── 3. Document embedding ───────────────────────────────────────────────
    def test_document_embedding(self):
        docs = [
            "Attention Is All You Need — transformers architecture",
            "Graph Neural Networks fail on out-of-distribution data",
        ]
        embeddings = self.svc.encode_documents(docs)
        self.assertEqual(embeddings.shape, (2, 384))
        print(f"[OK] Document embeddings shape = {embeddings.shape}")

    # ─── 4. Cosine similarity ────────────────────────────────────────────────
    def test_cosine_similarity_normalized(self):
        from backend.embedding_service import calculate_embedding_similarity
        q = self.svc.encode_query("machine learning reasoning")
        docs = self.svc.encode_documents([
            "machine learning and reasoning capabilities",
            "unrelated marine biology dolphin paper",
        ])
        scores = calculate_embedding_similarity(q, docs)
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[0], scores[1],
                           "Related paper should score higher than unrelated paper")
        print(f"[OK] Cosine similarity: relevant={scores[0]:.3f} vs unrelated={scores[1]:.3f}")

    # ─── 5. Normalize scores ─────────────────────────────────────────────────
    def test_normalize_scores(self):
        from backend.embedding_service import normalize_scores
        normed = normalize_scores([0.1, 0.5, 0.9])
        self.assertAlmostEqual(normed[0], 0.0, places=3)
        self.assertAlmostEqual(normed[-1], 1.0, places=3)
        print("[OK] normalize_scores min=0.0 max=1.0 correct")

    def test_normalize_scores_equal(self):
        from backend.embedding_service import normalize_scores
        normed = normalize_scores([0.5, 0.5, 0.5])
        self.assertTrue(all(v == 0.5 for v in normed))
        print("[OK] normalize_scores equal values → 0.5 fallback")

    # ─── 6. TF-IDF ranking ───────────────────────────────────────────────────
    def test_tfidf_ranking(self):
        from backend.embedding_service import _tfidf_score_papers
        papers = [
            {"id": "1", "title": "Transformers fail at reasoning", "text": "limitations of transformer reasoning", "raw_text": "limitations of transformer reasoning"},
            {"id": "2", "title": "Dolphin taxonomy survey", "text": "marine biology dolphins classification", "raw_text": "marine biology dolphins classification"},
        ]
        ranked = _tfidf_score_papers("transformers reasoning failure", papers)
        self.assertIn("tfidf_score", ranked[0])
        self.assertGreater(ranked[0]["tfidf_score"], ranked[-1]["tfidf_score"])
        print(f"[OK] TF-IDF ranking: top paper = '{ranked[0]['title'][:40]}'")

    # ─── 7. Hybrid ranking ───────────────────────────────────────────────────
    def test_hybrid_ranking(self):
        from backend.embedding_service import hybrid_rerank_results
        papers = [
            {"id": "1", "title": "Transformers fail at logical reasoning tasks", "text": "methodological flaw", "raw_text": "transformers fail logical reasoning benchmark contamination"},
            {"id": "2", "title": "Deep sea fish biology study", "text": "marine biology fish", "raw_text": "deep sea fish biology marine study unrelated"},
        ]
        ranked = hybrid_rerank_results("transformers logical reasoning", papers)
        self.assertIn("hybrid_score", ranked[0])
        self.assertIn("tfidf_score", ranked[0])
        self.assertIn("embedding_score", ranked[0])
        self.assertEqual(ranked[0]["id"], "1", "Relevant paper should rank first")
        print(f"[OK] Hybrid ranking: top paper = '{ranked[0]['title'][:40]}' (hybrid={ranked[0]['hybrid_score']})")

    # ─── 8. Cross-Encoder ranking ────────────────────────────────────────────
    def test_cross_encoder_ranking(self):
        from backend.embedding_service import rerank_with_cross_encoder
        papers = [
            {"id": "1", "title": "Transformers fail at reasoning", "text": "", "raw_text": "transformers fail at logical reasoning benchmark contamination"},
            {"id": "2", "title": "Deep sea marine biology", "text": "", "raw_text": "deep sea fish marine biology study"},
        ]
        ranked = rerank_with_cross_encoder("transformers and reasoning failures", papers)
        self.assertIn("cross_encoder_score", ranked[0])
        self.assertEqual(ranked[0]["id"], "1", "Relevant paper should rank first in cross-encoder")
        print(f"[OK] Cross-Encoder ranking: top paper = '{ranked[0]['title'][:40]}' (score={ranked[0]['cross_encoder_score']})")

    # ─── 9. Empty paper list ────────────────────────────────────────────────
    def test_empty_paper_list_hybrid(self):
        from backend.embedding_service import hybrid_rerank_results
        result = hybrid_rerank_results("any query", [])
        self.assertEqual(result, [])
        print("[OK] hybrid_rerank_results([]) returns empty list")

    def test_empty_paper_list_cross_encoder(self):
        from backend.embedding_service import rerank_with_cross_encoder
        result = rerank_with_cross_encoder("any query", [])
        self.assertEqual(result, [])
        print("[OK] rerank_with_cross_encoder([]) returns empty list")

    # ─── 10. Build embedding text ────────────────────────────────────────────
    def test_build_embedding_text(self):
        from backend.embedding_service import build_embedding_text
        paper = {"title": "Test Paper", "raw_text": "A" * 5000}
        text = build_embedding_text(paper)
        self.assertIn("Test Paper", text)
        self.assertLessEqual(len(text), 4020, "Text should be limited to ~4000 chars + title")
        print(f"[OK] build_embedding_text length = {len(text)} chars (<=4020)")

    # ─── 11. Embedding cache ─────────────────────────────────────────────────
    def test_embedding_cache_hit(self):
        from backend.embedding_service import (
            rerank_results_embedding, _embedding_cache, get_paper_embedding_cache_key
        )
        paper = {"id": "cache-test", "title": "Cache Test Paper", "text": "cache test content", "raw_text": "cache test content"}
        # First call: miss → computes and caches
        rerank_results_embedding("cache test", [paper])
        key = get_paper_embedding_cache_key(paper)
        self.assertIn(key, _embedding_cache, "Paper should be cached after first call")
        print("[OK] Embedding cache populated after first call")

    # ─── 12. API response field compatibility ────────────────────────────────
    def test_api_response_fields_present(self):
        """Verify hybrid ranking produces all expected API score fields."""
        from backend.embedding_service import hybrid_rerank_results, rerank_with_cross_encoder
        papers = [
            {"id": "api-test", "title": "Transformers generalization failure", "text": "flaw", "raw_text": "transformers fail generalization OOD"},
        ]
        ranked = hybrid_rerank_results("transformers generalization", papers)
        cross_ranked = rerank_with_cross_encoder("transformers generalization", ranked)
        paper = cross_ranked[0]
        for field in ["tfidf_score", "embedding_score", "hybrid_score", "cross_encoder_score"]:
            self.assertIn(field, paper, f"Missing field: {field}")
        print(f"[OK] API response fields present: tfidf={paper['tfidf_score']}, "
              f"emb={paper['embedding_score']}, hybrid={paper['hybrid_score']}, "
              f"ce={paper['cross_encoder_score']}")


class TestFallbackBehavior(unittest.TestCase):
    """Tests for graceful degradation when models are unavailable."""

    def test_hybrid_fallback_when_bge_unavailable(self):
        """If BGE is marked unavailable, hybrid should fall back to TF-IDF."""
        from backend.embedding_service import hybrid_rerank_results, embedding_service
        original = embedding_service._loaded
        try:
            embedding_service._loaded = False  # simulate failure
            papers = [
                {"id": "1", "title": "Transformers reasoning", "text": "transformers", "raw_text": "transformers reasoning test"},
            ]
            result = hybrid_rerank_results("transformers reasoning", papers)
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
            print("[OK] hybrid_rerank_results gracefully fell back to TF-IDF when BGE unavailable")
        finally:
            embedding_service._loaded = original

    def test_cross_encoder_fallback_when_unavailable(self):
        """If CrossEncoder is unavailable, should fall back to hybrid_score sorting."""
        from backend.embedding_service import rerank_with_cross_encoder, cross_encoder_service
        original = cross_encoder_service._loaded
        try:
            cross_encoder_service._loaded = False  # simulate failure
            papers = [
                {"id": "1", "title": "A", "text": "a", "raw_text": "a", "hybrid_score": 0.9},
                {"id": "2", "title": "B", "text": "b", "raw_text": "b", "hybrid_score": 0.3},
            ]
            result = rerank_with_cross_encoder("test query", papers)
            self.assertEqual(result[0]["id"], "1", "Should sort by hybrid_score fallback")
            print("[OK] rerank_with_cross_encoder gracefully fell back to hybrid_score when unavailable")
        finally:
            cross_encoder_service._loaded = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
