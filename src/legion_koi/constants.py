"""Shared constants for legion-koi — named values replacing magic numbers.

Organized by domain. Each constant documents WHY it has its value.
"""

# --- Chunking ---
# Target: 400 tokens ≈ 1600 chars (English avg ~4 chars/token)
CHUNK_CHARS = 1600
CHUNK_OVERLAP_CHARS = 400  # 25% overlap ≈ 100 tokens
# Documents shorter than this produce a single chunk (no splitting overhead)
MIN_SPLIT_CHARS = 2000

# --- Embedding ---
# Safety net for embedder input — chunks are pre-sized by chunking module.
# Only fires if unchunked text is passed directly.
MAX_EMBED_CHARS = 2000
EMBED_BATCH_SIZE = 20  # Balances throughput vs API rate limits

# --- Reranking ---
# bge-reranker-v2-m3 supports 8192 tokens, but GPU memory limits batch size.
# 16 pairs × ~500 tokens each fits comfortably on a 12GB GPU.
RERANK_BATCH_SIZE = 16
RERANK_DEFAULT_TOP_K = 10
# How many hybrid candidates to fetch before reranking (more = better recall, slower)
RERANK_CANDIDATE_POOL = 50
# Max chunks to score per candidate document (controls latency for large docs)
RERANK_MAX_CHUNKS_PER_DOC = 5

# --- Search ---
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_THREAD_LIMIT = 50
DEFAULT_LIST_LIMIT = 50
# Reciprocal Rank Fusion smoothing constant (standard value from Cormack et al.)
RRF_K = 60
# Fetch multiplier for hybrid search — get 3x results from each source for RRF merging
RRF_FETCH_MULTIPLIER = 3

# --- Text limits ---
# PostgreSQL tsvector max is 1MB; cap search_text well below that
MAX_SEARCH_TEXT = 500_000
# Display preview lengths for MCP tool output
PREVIEW_SHORT = 150   # message content previews
PREVIEW_MEDIUM = 200  # summaries, search_text previews
PREVIEW_LONG = 500    # extended previews

# --- Default models ---
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_OLLAMA_RERANKER_MODEL = "bge-reranker-v2-m3"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"

# --- HNSW index tuning ---
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

# --- Retry / backoff ---
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [2, 5, 15]
