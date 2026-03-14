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
# DEPRECATED in Phase 4 — kept for backward compatibility with evaluation scripts
RRF_K = 60
# Fetch multiplier for hybrid search — get 3x results from each source for merging
RRF_FETCH_MULTIPLIER = 3
# Convex combination alpha (Phase 4) — FTS weight. Semantic weight = 1 - alpha.
# 0.85 = FTS-dominant, reflects FTS baseline 0.967 Recall@10 vs semantic 0.072.
# Tuned via scripts/tune_alpha.py against golden query set.
CONVEX_ALPHA = 0.85
# Query router alpha overrides per query type
ROUTER_ALPHA_KEYWORD = 1.0    # FTS only — exact terms don't benefit from semantics
ROUTER_ALPHA_CONCEPTUAL = 0.5 # Balanced — semantic captures meaning
ROUTER_ALPHA_TEMPORAL = 1.0   # FTS only with date filtering
# ROUTER_ALPHA_HYBRID uses CONVEX_ALPHA (the learned default)

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

# --- Hippo bridge ---
HIPPO_REDIS_HOST = "localhost"
HIPPO_REDIS_PORT = 6380  # FalkorDB container maps 6379->6380 to avoid Redis conflicts
HIPPO_GRAPH_NAME = "hippo"
# Bonus weight for entity-resolved RIDs in merged scoring (additive to RRF)
ENTITY_RRF_BONUS = 0.01

# --- HNSW index tuning ---
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

# --- Entity extraction ---
# Max chars per LLM extraction call — balances cost vs context
ENTITY_EXTRACT_MAX_CHARS = 3000
# Bundles per backfill iteration (limits memory, enables progress logging)
ENTITY_BACKFILL_BATCH_SIZE = 50
# Cap per chunk to prevent hallucination on rich documents
ENTITY_MAX_PER_CHUNK = 30
# Namespaces to skip — koi-net.node is internal protocol state
ENTITY_EXTRACTION_SKIP_NAMESPACES = frozenset({"koi-net.node"})
# Discard entities below this confidence — noise is recoverable, but clutter costs
ENTITY_DEFAULT_CONFIDENCE_FLOOR = 0.3

# --- Retry / backoff ---
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = [2, 5, 15]

# --- Event system (Phase 1) ---
# Redis connection for event bus (reuses FalkorDB container)
EVENT_REDIS_HOST = "localhost"
EVENT_REDIS_PORT = 6380  # FalkorDB container, same as HIPPO_REDIS_PORT
# Stream naming: koi:events:{event_type}
EVENT_STREAM_PREFIX = "koi:events:"
# Dead letter queue: koi:dlq:{stream_name}
EVENT_DLQ_PREFIX = "koi:dlq:"
# PostgreSQL NOTIFY channel name
EVENT_PG_CHANNEL = "koi_events"
# Consumer polling: how many messages per read
EVENT_CONSUMER_POLL_BATCH = 10
# Consumer polling: block timeout in milliseconds (2s balances latency vs CPU)
EVENT_CONSUMER_BLOCK_MS = 2000
# Max retries before sending to dead letter queue
EVENT_DLQ_MAX_RETRIES = 3

# --- Observability (Phase 2) ---
# Health dimension weights (must sum to 1.0)
HEALTH_WEIGHT_AVAILABILITY = 0.35
HEALTH_WEIGHT_PERFORMANCE = 0.25
HEALTH_WEIGHT_QUALITY = 0.25
HEALTH_WEIGHT_GROWTH = 0.15
# systemd user services to monitor
HEALTH_SERVICES = [
    "hippo-graph.service",
    "letta-stack.service",
    "legion-koi.service",
    "legion-messages.service",
]

# --- Message filtering ---
# Seconds before refreshing thread classification cache from messages.db
MESSAGE_THREAD_CACHE_TTL = 3600
# Participation ratio below this → LOW_SIGNAL instead of ACTIVE
# 0.002 = 1 post per 500 messages. Groups like Curve Finance (25/423K = 0.00006)
# and Commons Stack (6/14K = 0.0004) fall below. BCRG (641/5306 = 0.12) stays ACTIVE.
LOW_SIGNAL_PARTICIPATION_FLOOR = 0.002
# Consumer-level content quality gate — skip embedding/extraction for trivially short messages
# ("gm", "yes", emoji) that add no knowledge even in included threads
EMBED_MIN_CONTENT_CHARS = 20
EXTRACT_MIN_CONTENT_CHARS = 30

# --- Resilience / Circuit Breaker (Phase 3) ---
# Number of consecutive failures before opening the circuit
CIRCUIT_FAILURE_THRESHOLD = 3
# Seconds to wait before testing recovery (OPEN → HALF_OPEN)
CIRCUIT_RECOVERY_TIMEOUT_SECONDS = 30
