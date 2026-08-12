import hashlib
import math
import os
import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIMENSION = 384

# In-memory cache: sha256(title+text) → numpy array
_embedding_cache: Dict[str, np.ndarray] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Document text builder
# ─────────────────────────────────────────────────────────────────────────────
def build_embedding_text(paper: Dict[str, Any]) -> str:
    """Build embedding input text from paper title + abstract."""
    title = paper.get("title", "")
    text = paper.get("raw_text", paper.get("text", ""))
    return f"{title}\n{text[:4000]}".strip()


def get_paper_embedding_cache_key(paper: Dict[str, Any]) -> str:
    """Stable SHA256 cache key for a paper's embedding text."""
    text = build_embedding_text(paper)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# BGE Embedding Service
# Priority chain:
#   1. sentence_transformers  (local / paid tier  — full torch, GPU-ready)
#   2. fastembed              (Render free tier   — ONNX Runtime, ~200 MB RAM)
#   3. TF-IDF only            (last resort if both fail)
# ─────────────────────────────────────────────────────────────────────────────
class EmbeddingService:
    """
    Wraps BAAI/bge-small-en-v1.5.

    Backend auto-selected at startup:
    - ``sentence_transformers`` when torch is available (local dev / paid hosting)
    - ``fastembed`` (ONNX Runtime) otherwise — fits comfortably in Render’s
      free 512 MB tier with the full BGE-small model (~23 MB ONNX weights).

    encode_query() prepends the BGE retrieval instruction for both backends.
    """

    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        self._backend: str = "none"   # "sentence_transformers" | "fastembed" | "none"
        self._load()

    def _load(self) -> None:
        # ── Attempt 1: sentence_transformers (full torch install) ──────────────
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] Loading {BGE_MODEL_NAME} via sentence_transformers ...")
            self.model = SentenceTransformer(BGE_MODEL_NAME)
            self._backend = "sentence_transformers"
            self._loaded = True
            print(f"[Embedding] sentence_transformers ready (dim={EMBEDDING_DIMENSION})")
            return
        except ImportError:
            print("[Embedding] sentence_transformers not installed — trying fastembed …")
        except Exception as exc:
            print(f"[Embedding] sentence_transformers failed ({exc}) — trying fastembed …")

        # ── Attempt 2: fastembed (ONNX Runtime, Render free tier) ──────────────
        try:
            from fastembed import TextEmbedding
            print(f"[Embedding] Loading {BGE_MODEL_NAME} via fastembed (ONNX) ...")
            self.model = TextEmbedding(model_name=BGE_MODEL_NAME)
            self._backend = "fastembed"
            self._loaded = True
            print(f"[Embedding] fastembed ready (dim={EMBEDDING_DIMENSION})")
            return
        except Exception as exc:
            print(f"[Embedding] fastembed failed ({exc}) — falling back to TF-IDF only.")
            self.model = None
            self._backend = "none"
            self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded and self.model is not None

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not self.available:
            raise RuntimeError("No embedding backend loaded")

        if self._backend == "fastembed":
            # fastembed.embed() returns a generator of already-L2-normalised arrays
            single = isinstance(texts, str)
            input_list = [texts] if single else texts
            result = np.array(list(self.model.embed(input_list)), dtype=np.float32)
            if normalize_embeddings:
                # Vectors are already unit-norm from fastembed, but re-normalise
                # just in case a custom model or future version differs.
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                result = result / np.clip(norms, 1e-9, None)
            return result[0] if single else result

        # sentence_transformers backend
        return self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Prepend BGE v1.5 retrieval instruction for queries."""
        query_text = "Represent this sentence for searching relevant passages: " + query
        return self.encode(query_text)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode document texts (no instruction prefix)."""
        return self.encode(documents)

    def dimension(self) -> int:
        return EMBEDDING_DIMENSION



# ─────────────────────────────────────────────────────────────────────────────
# Cross-Encoder Service
# ─────────────────────────────────────────────────────────────────────────────
class CrossEncoderService:
    """
    Wraps cross-encoder/ms-marco-MiniLM-L-6-v2.
    Requires sentence_transformers (torch).  On Render free tier where only
    fastembed is available, this service stays unavailable and the caller
    falls back to hybrid BGE+TF-IDF ranking automatically.
    """

    def __init__(self) -> None:
        self.model = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"[CrossEncoder] Loading {CROSS_ENCODER_MODEL_NAME} ...")
            self.model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            self._loaded = True
            print("[CrossEncoder] Model loaded")
        except ImportError:
            print("[CrossEncoder] sentence_transformers not installed — CrossEncoder unavailable (hybrid BGE fallback active).")
            self.model = None
            self._loaded = False
        except Exception as exc:
            print(f"[CrossEncoder] WARNING: Failed to load {CROSS_ENCODER_MODEL_NAME}: {exc}")
            print("[CrossEncoder] Will use hybrid BGE+TF-IDF ranking as fallback.")
            self.model = None
            self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded and self.model is not None

    def predict(self, pairs: List[List[str]]) -> List[float]:
        if not self.available:
            raise RuntimeError("CrossEncoder not loaded")
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


# ─────────────────────────────────────────────────────────────────────────────
# Singletons (loaded once at module import / startup)
# ─────────────────────────────────────────────────────────────────────────────
embedding_service = EmbeddingService()
cross_encoder_service = CrossEncoderService()


# ─────────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ─────────────────────────────────────────────────────────────────────────────
def calculate_embedding_similarity(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> List[float]:
    """
    Cosine similarity via dot product (embeddings are L2-normalized).
    Returns a list of float scores, one per document.
    """
    q = np.asarray(query_embedding)
    D = np.asarray(document_embeddings)
    scores = D @ q
    return scores.tolist()


def normalize_scores(values: List[float]) -> List[float]:
    """Min-max normalization to [0, 1]."""
    if not values:
        return []
    values = [float(v) for v in values]
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [0.5] * len(values)
    return [(v - min_v) / (max_v - min_v) for v in values]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF retrieval (the actual TF-IDF logic, extracted from live_agent.py)
# ─────────────────────────────────────────────────────────────────────────────
def _tfidf_score_papers(
    query: str, papers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Pure-Python TF-IDF + cosine similarity.
    Attaches 'tfidf_score' to each paper; returns papers sorted descending.
    """
    if not papers:
        return []

    documents = [query] + [
        (p.get("title", "") + " " + p.get("text", p.get("raw_text", ""))[:300])
        for p in papers
    ]

    tokenized = [re.findall(r"\w+", doc.lower()) for doc in documents]
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    N = len(documents)

    def tfidf_vector(tokens):
        tf = Counter(tokens)
        return {
            term: count * math.log((N + 1) / (df[term] + 1))
            for term, count in tf.items()
        }

    def cosine_sim(v1, v2):
        dot = sum(v1.get(t, 0) * v2.get(t, 0) for t in set(v1) & set(v2))
        m1 = math.sqrt(sum(x ** 2 for x in v1.values()))
        m2 = math.sqrt(sum(x ** 2 for x in v2.values()))
        return dot / (m1 * m2) if m1 and m2 else 0.0

    vectors = [tfidf_vector(tokens) for tokens in tokenized]
    query_vec = vectors[0]
    doc_vecs = vectors[1:]

    for paper, dv in zip(papers, doc_vecs):
        raw = cosine_sim(query_vec, dv)
        paper["tfidf_score"] = round(float(raw), 4)

    return sorted(papers, key=lambda x: x.get("tfidf_score", 0), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# BGE Embedding retrieval
# ─────────────────────────────────────────────────────────────────────────────
def rerank_results_embedding(
    query: str,
    papers: List[Dict[str, Any]],
    query_embedding: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    BGE semantic retrieval.
    Attaches 'embedding_score' to each paper; returns sorted descending.
    Accepts a pre-computed query_embedding to avoid duplicate encode calls.
    """
    if not papers or not embedding_service.available:
        return papers

    try:
        if query_embedding is None:
            query_embedding = embedding_service.encode_query(query)

        cached_embeddings: Dict[int, np.ndarray] = {}
        doc_texts: List[str] = [build_embedding_text(p) for p in papers]

        for idx, paper in enumerate(papers):
            key = get_paper_embedding_cache_key(paper)
            if key in _embedding_cache:
                cached_embeddings[idx] = _embedding_cache[key]

        # Batch-encode only uncached documents
        uncached_indices = [i for i in range(len(papers)) if i not in cached_embeddings]
        if uncached_indices:
            uncached_texts = [doc_texts[i] for i in uncached_indices]
            new_vecs = embedding_service.encode_documents(uncached_texts)
            for local_i, global_i in enumerate(uncached_indices):
                vec = new_vecs[local_i]
                _embedding_cache[get_paper_embedding_cache_key(papers[global_i])] = vec
                cached_embeddings[global_i] = vec

        doc_matrix = np.stack([cached_embeddings[i] for i in range(len(papers))])
        scores = calculate_embedding_similarity(query_embedding, doc_matrix)

        for paper, score in zip(papers, scores):
            paper["embedding_score"] = round(float(score), 4)

        ranked = sorted(papers, key=lambda x: x.get("embedding_score", 0), reverse=True)
        return ranked[:top_k] if top_k else ranked

    except Exception as exc:
        print(f"[Embedding] Retrieval error: {exc} — falling back to input order")
        return papers[:top_k] if top_k else papers


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retrieval (TF-IDF 30% + BGE 70%)
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_rerank_results(
    query: str,
    papers: List[Dict[str, Any]],
    query_embedding: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Hybrid ranking: TF-IDF (30%) + BGE semantic similarity (70%).
    Falls back to TF-IDF-only if BGE is unavailable.
    """
    if not papers:
        return []

    # Step 1: TF-IDF scores
    tfidf_ranked = _tfidf_score_papers(query, list(papers))
    tfidf_norm = normalize_scores([p.get("tfidf_score", 0.0) for p in tfidf_ranked])
    for paper, score in zip(tfidf_ranked, tfidf_norm):
        paper["tfidf_normalized"] = score

    # Step 2: BGE embedding scores (with fallback)
    if not embedding_service.available:
        print("[Hybrid Retrieval] BGE unavailable — using TF-IDF only")
        for paper, score in zip(tfidf_ranked, tfidf_norm):
            paper["hybrid_score"] = round(score, 4)
        ranked = sorted(tfidf_ranked, key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return ranked[:top_k] if top_k else ranked

    try:
        embedding_ranked = rerank_results_embedding(
            query, tfidf_ranked, query_embedding=query_embedding, top_k=None
        )
        emb_norm = normalize_scores([p.get("embedding_score", 0.0) for p in embedding_ranked])
        for paper, score in zip(embedding_ranked, emb_norm):
            paper["embedding_normalized"] = score
            paper["hybrid_score"] = round(
                0.30 * paper.get("tfidf_normalized", 0.0) + 0.70 * score, 4
            )

        ranked = sorted(embedding_ranked, key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return ranked[:top_k] if top_k else ranked

    except Exception as exc:
        print(f"[Hybrid Retrieval] BGE error: {exc} — using TF-IDF only")
        ranked = sorted(tfidf_ranked, key=lambda x: x.get("tfidf_normalized", 0), reverse=True)
        return ranked[:top_k] if top_k else ranked


# ─────────────────────────────────────────────────────────────────────────────
# Real Cross-Encoder reranker
# ─────────────────────────────────────────────────────────────────────────────
def rerank_with_cross_encoder(
    query: str,
    papers: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Pairwise relevance reranker using cross-encoder/ms-marco-MiniLM-L-6-v2.
    Falls back to hybrid_score ranking if Cross-Encoder is unavailable.
    """
    if not papers:
        return []

    if not cross_encoder_service.available:
        print("[CrossEncoder] Unavailable — using hybrid score ranking")
        ranked = sorted(
            papers,
            key=lambda x: x.get("hybrid_score", x.get("tfidf_normalized", 0)),
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked

    try:
        pairs = [[query, build_embedding_text(paper)] for paper in papers]
        scores = cross_encoder_service.predict(pairs)

        for paper, score in zip(papers, scores):
            paper["cross_encoder_score"] = round(score, 4)

        ranked = sorted(papers, key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
        return ranked[:top_k] if top_k else ranked

    except Exception as exc:
        print(f"[CrossEncoder] Error: {exc} — using hybrid score ranking")
        ranked = sorted(
            papers,
            key=lambda x: x.get("hybrid_score", x.get("tfidf_normalized", 0)),
            reverse=True,
        )
        return ranked[:top_k] if top_k else ranked


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_embedding(text: str) -> List[float]:
    return embedding_service.encode_query(text).tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embedding_service.encode_documents(texts).tolist()
