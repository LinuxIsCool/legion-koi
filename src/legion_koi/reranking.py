"""Cross-encoder reranking — re-score (query, document) pairs for precision.

Two backends, same protocol (mirrors the Embedder pattern):
  - CrossEncoderReranker: sentence-transformers CrossEncoder (GPU, recommended)
  - OllamaReranker: Ollama /api/rerank endpoint (future, when Ollama adds support)

Usage:
    from legion_koi.reranking import create_reranker

    reranker = create_reranker("cross-encoder")  # or "ollama"
    ranked = reranker.rerank("my query", ["doc1", "doc2", "doc3"], top_k=2)
    # [(original_index, score), ...] sorted by score descending

For long documents, use rerank_chunked() which chunks each document and
scores each chunk, using the best chunk score per document:
    ranked = reranker.rerank_chunked("my query", long_docs, top_k=5)
"""

import os
from typing import Protocol

import httpx
import structlog

from .constants import (
    DEFAULT_OLLAMA_RERANKER_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_RERANKER_MODEL,
    RERANK_BATCH_SIZE,
    RERANK_DEFAULT_TOP_K,
    RERANK_MAX_CHUNKS_PER_DOC,
)

log = structlog.stdlib.get_logger()


class Reranker(Protocol):
    """Provider-agnostic reranking interface."""

    def rerank(
        self, query: str, documents: list[str], top_k: int = RERANK_DEFAULT_TOP_K
    ) -> list[tuple[int, float]]:
        """Rerank documents for a query.

        Returns [(original_index, score)] sorted by score descending.
        """
        ...

    def is_available(self) -> bool: ...
    def get_model(self) -> str: ...


class CrossEncoderReranker:
    """BAAI/bge-reranker-v2-m3 via sentence-transformers CrossEncoder. GPU-accelerated."""

    def __init__(self, model: str = DEFAULT_RERANKER_MODEL):
        self._model = model
        self._encoder = None  # lazy init

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self._model)
        return self._encoder

    def rerank(
        self, query: str, documents: list[str], top_k: int = RERANK_DEFAULT_TOP_K
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        encoder = self._get_encoder()
        all_scores = []
        for i in range(0, len(documents), RERANK_BATCH_SIZE):
            batch = documents[i : i + RERANK_BATCH_SIZE]
            pairs = [(query, doc) for doc in batch]
            scores = encoder.predict(pairs)
            all_scores.extend(scores.tolist())
        indexed = sorted(enumerate(all_scores), key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def is_available(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401

            return True
        except ImportError:
            return False

    def get_model(self) -> str:
        return self._model


class OllamaReranker:
    """Reranking via Ollama's /api/rerank endpoint.

    NOTE: Requires Ollama with rerank support (PR #7219). As of Ollama 0.17.7,
    this endpoint is not yet available. This backend exists for forward
    compatibility — use CrossEncoderReranker until Ollama merges rerank support.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = DEFAULT_OLLAMA_RERANKER_MODEL,
    ):
        self._base_url = (
            base_url or os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        ).rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=60.0)

    def rerank(
        self, query: str, documents: list[str], top_k: int = RERANK_DEFAULT_TOP_K
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        resp = self._client.post(
            f"{self._base_url}/api/rerank",
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_k": top_k,
            },
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [(r["index"], r["relevance_score"]) for r in results]

    def is_available(self) -> bool:
        try:
            resp = self._client.get(
                f"{self._base_url}/api/tags",
                timeout=5.0,
            )
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(self._model in m for m in models)
        except Exception:
            return False

    def get_model(self) -> str:
        return self._model


# -- Chunk-and-score --


# Max chunks to score per candidate document. Large documents can produce
# hundreds of chunks — scoring all of them is slow. We take the first few
# plus evenly spaced samples to cover the document without explosion.
def rerank_chunked(
    reranker: Reranker,
    query: str,
    documents: list[str],
    top_k: int = RERANK_DEFAULT_TOP_K,
    max_chunks_per_doc: int = RERANK_MAX_CHUNKS_PER_DOC,
) -> list[tuple[int, float]]:
    """Chunk long documents and rerank using best-chunk-per-document scoring.

    Instead of truncating documents, chunks each one and scores sampled chunks.
    Each document's score is its best (max) chunk score. This ensures the
    reranker sees relevant passages regardless of document length.

    For documents with many chunks, samples evenly across the document
    (first, last, and spaced in between) up to max_chunks_per_doc.
    """
    from .chunking import chunk_text

    # Build flat list of (doc_index, chunk_text) pairs
    all_chunks: list[tuple[int, str]] = []
    for doc_idx, doc in enumerate(documents):
        chunks = chunk_text(doc)
        if not chunks:
            all_chunks.append((doc_idx, doc))
        elif len(chunks) <= max_chunks_per_doc:
            for chunk in chunks:
                all_chunks.append((doc_idx, chunk))
        else:
            # Sample: first, last, and evenly spaced in between
            indices = [0, len(chunks) - 1]
            step = len(chunks) / (max_chunks_per_doc - 1)
            for i in range(1, max_chunks_per_doc - 1):
                idx = int(i * step)
                if idx not in indices:
                    indices.append(idx)
            for idx in sorted(set(indices))[:max_chunks_per_doc]:
                all_chunks.append((doc_idx, chunks[idx]))

    if not all_chunks:
        return []

    # Score all chunks
    chunk_texts = [c[1] for c in all_chunks]
    chunk_scores = reranker.rerank(query, chunk_texts, top_k=len(chunk_texts))

    # Map back to documents: best chunk score per document
    doc_best: dict[int, float] = {}
    for chunk_flat_idx, score in chunk_scores:
        doc_idx = all_chunks[chunk_flat_idx][0]
        if doc_idx not in doc_best or score > doc_best[doc_idx]:
            doc_best[doc_idx] = score

    # Sort documents by their best chunk score
    sorted_docs = sorted(doc_best.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs[:top_k]


def rerank_results(
    reranker: Reranker,
    query: str,
    results: list[dict],
    top_k: int = RERANK_DEFAULT_TOP_K,
    max_chunks_per_doc: int = RERANK_MAX_CHUNKS_PER_DOC,
) -> list[dict]:
    """Rerank search result dicts using cross-encoder scoring.

    For each result, builds a reranking text:
    - Results WITH chunk_text (from vector search): use chunk directly
    - Results WITHOUT chunk_text (FTS-only): use first CHUNK_CHARS of search_text
      (FTS already ranked these by keyword relevance, so the beginning often has
      the most relevant content for short docs, and for long docs we accept the
      approximation rather than trying to sample random chunks)

    Returns reranked result dicts (same shape as input, reordered).
    """
    from .constants import CHUNK_CHARS

    from .chunking import chunk_text as do_chunk

    texts: list[str] = []
    for r in results:
        ct = r.get("chunk_text")
        if ct and len(ct) > 10:
            texts.append(ct)
        else:
            # Prefer ts_headline snippets (query-focused passages) over blind truncation
            headline = r.get("headline")
            if headline and len(headline) > 20:
                texts.append(headline)
            else:
                # Chunk instead of truncate (hard rule: never truncate data)
                st = r.get("search_text", "") or ""
                chunks = do_chunk(st)
                # Use first chunk as reranking passage (best available without truncation)
                texts.append(chunks[0] if chunks else st)

    if not texts:
        return results[:top_k]

    scored = reranker.rerank(query, texts, top_k=top_k)
    return [results[idx] for idx, _score in scored]


# -- Factory --


def create_reranker(
    backend: str = "auto",
    model: str | None = None,
    base_url: str | None = None,
) -> Reranker:
    """Create a reranker.

    Backends:
      - 'cross-encoder': sentence-transformers CrossEncoder (GPU, recommended)
      - 'flag': alias for cross-encoder (backward compat)
      - 'ollama': Ollama /api/rerank (not yet available in Ollama 0.17.7)
      - 'auto' (default): tries cross-encoder first, falls back to Ollama
    """
    if backend == "ollama":
        return OllamaReranker(
            base_url=base_url,
            model=model or DEFAULT_OLLAMA_RERANKER_MODEL,
        )
    if backend in ("cross-encoder", "flag"):
        return CrossEncoderReranker(model=model or DEFAULT_RERANKER_MODEL)

    # Auto-detect: prefer CrossEncoder if importable
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401

        return CrossEncoderReranker(model=model or DEFAULT_RERANKER_MODEL)
    except ImportError:
        return OllamaReranker(
            base_url=base_url,
            model=model or DEFAULT_OLLAMA_RERANKER_MODEL,
        )
