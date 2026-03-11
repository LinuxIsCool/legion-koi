"""PostgreSQL storage for KOI-net bundles with full-text search and vector embeddings."""

import json
from datetime import datetime, timezone

import structlog
import psycopg
from psycopg.rows import dict_row

log = structlog.stdlib.get_logger()

# Re-export for use by backfill script and handlers
__all__ = ["PostgresStorage", "_extract_search_text"]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS bundles (
    rid TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    reference TEXT NOT NULL,
    contents JSONB NOT NULL,
    search_text TEXT NOT NULL DEFAULT '',
    sha256_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', search_text)
    ) STORED
);
CREATE INDEX IF NOT EXISTS idx_bundles_namespace ON bundles(namespace);
CREATE INDEX IF NOT EXISTS idx_bundles_search ON bundles USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_bundles_contents ON bundles USING GIN(contents);
"""

EMBEDDING_CONFIGS_SQL = """
CREATE TABLE IF NOT EXISTS embedding_configs (
    config_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL DEFAULT 1000,
    chunk_overlap FLOAT NOT NULL DEFAULT 0.0,
    description TEXT,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _config_table_name(config_id: str) -> str:
    """Sanitized table name for an embedding config."""
    safe = config_id.replace("-", "_").replace(".", "_")
    return f"embeddings_{safe}"


def _config_table_sql(config_id: str, dimension: int) -> str:
    """Generate per-config embeddings table DDL."""
    table = _config_table_name(config_id)
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    rid TEXT NOT NULL REFERENCES bundles(rid) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_text TEXT,
    embedding vector({dimension}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (rid, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_{table}_hnsw
    ON {table} USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


# Legacy single-table support (kept for migration path)
def _embeddings_sql(dimension: int) -> str:
    """Generate legacy embeddings table DDL with parameterized vector dimension."""
    return f"""
CREATE TABLE IF NOT EXISTS embeddings (
    rid TEXT PRIMARY KEY REFERENCES bundles(rid) ON DELETE CASCADE,
    embedding vector({dimension}),
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


# PostgreSQL tsvector max is 1MB; cap search_text well below that
_MAX_SEARCH_TEXT = 500_000


def _extract_search_text(namespace: str, contents: dict) -> str:
    """Build search text from bundle contents based on namespace."""
    fm = contents.get("frontmatter", {})

    if namespace == "legion.claude-journal":
        title = fm.get("title", "")
        body = contents.get("body", "")
        return f"{title} {body}"

    if namespace == "legion.claude-venture":
        title = fm.get("title", "")
        description = fm.get("description", "")
        body = contents.get("body", "")
        tags = " ".join(fm.get("tags", []))
        return f"{title} {description} {body} {tags}"

    if namespace == "legion.claude-session":
        cwd = contents.get("cwd", "") or ""
        summary = contents.get("summary", "") or ""
        return f"{cwd} {summary}"

    if namespace == "legion.claude-recording":
        filename = contents.get("filename", "") or ""
        source = contents.get("source", "") or ""
        title = contents.get("title", "") or ""
        notes = contents.get("notes", "") or ""
        date_recorded = contents.get("date_recorded", "") or ""
        transcript = contents.get("transcript_text", "") or ""
        text = f"{filename} {source} {title} {notes} {date_recorded} {transcript}"
        return text[:_MAX_SEARCH_TEXT]

    if namespace == "legion.claude-message":
        text = contents.get("content", "") or ""
        return text[:_MAX_SEARCH_TEXT]

    # Fallback: JSON dump
    text = json.dumps(contents)
    return text[:_MAX_SEARCH_TEXT]


class PostgresStorage:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

    def initialize(self, embedding_dim: int | None = None):
        """Create tables and indexes. If embedding_dim is provided, create/migrate legacy embeddings table."""
        conn = self._get_conn()
        conn.execute(SCHEMA_SQL)

        # Always create the config registry
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.errors.InsufficientPrivilege:
            pass
        conn.execute(EMBEDDING_CONFIGS_SQL)

        if embedding_dim is not None:
            self._ensure_embeddings_table(conn, embedding_dim)

        log.info("postgres.initialized", embedding_dim=embedding_dim)

    def _ensure_embeddings_table(self, conn: psycopg.Connection, dimension: int):
        """Create or migrate embeddings table to match the target vector dimension."""
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.errors.InsufficientPrivilege:
            log.warning("postgres.pgvector_extension_skip", msg="Already exists or no perms")
            conn.rollback() if not conn.autocommit else None

        # Check if table exists and has correct dimension
        row = conn.execute(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'embeddings'::regclass AND attname = 'embedding'
            """,
        ).fetchone()

        if row is not None:
            # pgvector stores dimension directly in atttypmod
            current_dim = row["atttypmod"] if row["atttypmod"] > 0 else None
            if current_dim == dimension:
                log.info("postgres.embeddings_table_ok", dimension=dimension)
                return
            log.info("postgres.embeddings_dimension_mismatch",
                     current=current_dim, target=dimension, action="recreate")
            conn.execute("DROP TABLE IF EXISTS embeddings")

        conn.execute(_embeddings_sql(dimension))
        log.info("postgres.embeddings_table_created", dimension=dimension)

    def upsert_bundle(
        self,
        rid: str,
        namespace: str,
        reference: str,
        contents: dict,
        sha256_hash: str,
        created_at: datetime | None = None,
    ):
        """Insert or update a bundle."""
        now = datetime.now(timezone.utc)
        search_text = _extract_search_text(namespace, contents)
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO bundles (rid, namespace, reference, contents, search_text, sha256_hash, created_at, updated_at)
            VALUES (%(rid)s, %(namespace)s, %(reference)s, %(contents)s, %(search_text)s, %(sha256_hash)s, %(created_at)s, %(updated_at)s)
            ON CONFLICT (rid) DO UPDATE SET
                contents = EXCLUDED.contents,
                search_text = EXCLUDED.search_text,
                sha256_hash = EXCLUDED.sha256_hash,
                updated_at = EXCLUDED.updated_at
            """,
            {
                "rid": rid,
                "namespace": namespace,
                "reference": reference,
                "contents": psycopg.types.json.Jsonb(contents),
                "search_text": search_text,
                "sha256_hash": sha256_hash,
                "created_at": created_at or now,
                "updated_at": now,
            },
        )

    def upsert_bundles_batch(self, bundles: list[dict]):
        """Batch upsert multiple bundles in a single transaction.

        Each dict must have: rid, namespace, reference, contents, sha256_hash.
        Optional: created_at.
        """
        if not bundles:
            return
        now = datetime.now(timezone.utc)
        conn = self._get_conn()
        with conn.transaction():
            with conn.cursor() as cur:
                for b in bundles:
                    search_text = _extract_search_text(b["namespace"], b["contents"])
                    cur.execute(
                        """
                        INSERT INTO bundles (rid, namespace, reference, contents, search_text, sha256_hash, created_at, updated_at)
                        VALUES (%(rid)s, %(namespace)s, %(reference)s, %(contents)s, %(search_text)s, %(sha256_hash)s, %(created_at)s, %(updated_at)s)
                        ON CONFLICT (rid) DO UPDATE SET
                            contents = EXCLUDED.contents,
                            search_text = EXCLUDED.search_text,
                            sha256_hash = EXCLUDED.sha256_hash,
                            updated_at = EXCLUDED.updated_at
                        """,
                        {
                            "rid": b["rid"],
                            "namespace": b["namespace"],
                            "reference": b["reference"],
                            "contents": psycopg.types.json.Jsonb(b["contents"]),
                            "search_text": search_text,
                            "sha256_hash": b["sha256_hash"],
                            "created_at": b.get("created_at") or now,
                            "updated_at": now,
                        },
                    )
        log.info("postgres.batch_upsert", count=len(bundles))

    def get_bundle(self, rid: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM bundles WHERE rid = %s", (rid,)
        ).fetchone()
        return dict(row) if row else None

    def search_text(self, query: str, namespace: str | None = None, limit: int = 20) -> list[dict]:
        """Full-text search across bundles."""
        conn = self._get_conn()
        if namespace:
            rows = conn.execute(
                """
                SELECT rid, namespace, reference, contents, search_text,
                       ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS rank
                FROM bundles
                WHERE search_vector @@ websearch_to_tsquery('english', %s)
                  AND namespace = %s
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, namespace, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT rid, namespace, reference, contents, search_text,
                       ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS rank
                FROM bundles
                WHERE search_vector @@ websearch_to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_rids(self, namespace: str | None = None) -> list[dict]:
        conn = self._get_conn()
        if namespace:
            rows = conn.execute(
                "SELECT rid, namespace, reference, created_at, updated_at FROM bundles WHERE namespace = %s ORDER BY created_at DESC",
                (namespace,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT rid, namespace, reference, created_at, updated_at FROM bundles ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Counts per namespace."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT namespace, count(*) as count FROM bundles GROUP BY namespace ORDER BY namespace"
        ).fetchall()
        return {r["namespace"]: r["count"] for r in rows}

    def get_thread_messages(self, thread_id: str, limit: int = 50) -> list[dict]:
        """Get messages in a thread, ordered by platform timestamp."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT rid, namespace, reference, contents, created_at
            FROM bundles
            WHERE namespace = 'legion.claude-message'
              AND contents->>'thread_id' = %s
            ORDER BY contents->>'platform_ts' ASC
            LIMIT %s
            """,
            (thread_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Embedding methods --

    @staticmethod
    def _vec_literal(embedding: list[float]) -> str:
        """Format embedding as pgvector string literal."""
        return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

    def upsert_embedding(self, rid: str, embedding: list[float], model: str):
        """Insert or update a single embedding."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO embeddings (rid, embedding, model)
            VALUES (%s, %s::vector, %s)
            ON CONFLICT (rid) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                model = EXCLUDED.model,
                created_at = NOW()
            """,
            (rid, self._vec_literal(embedding), model),
        )

    def upsert_embeddings_batch(self, items: list[dict]):
        """Bulk upsert embeddings. Each dict: {rid, embedding, model}."""
        if not items:
            return
        conn = self._get_conn()
        with conn.transaction():
            with conn.cursor() as cur:
                for item in items:
                    cur.execute(
                        """
                        INSERT INTO embeddings (rid, embedding, model)
                        VALUES (%s, %s::vector, %s)
                        ON CONFLICT (rid) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            model = EXCLUDED.model,
                            created_at = NOW()
                        """,
                        (item["rid"], self._vec_literal(item["embedding"]), item["model"]),
                    )

    def search_semantic(
        self, query_embedding: list[float], namespace: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Pure vector similarity search using cosine distance."""
        conn = self._get_conn()
        vec = self._vec_literal(query_embedding)
        if namespace:
            rows = conn.execute(
                """
                SELECT b.rid, b.namespace, b.reference, b.contents, b.search_text,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM embeddings e
                JOIN bundles b ON b.rid = e.rid
                WHERE b.namespace = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, namespace, vec, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT b.rid, b.namespace, b.reference, b.contents, b.search_text,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM embeddings e
                JOIN bundles b ON b.rid = e.rid
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, vec, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_hybrid(
        self,
        query: str,
        query_embedding: list[float],
        namespace: str | None = None,
        limit: int = 20,
        k: int = 60,
    ) -> list[dict]:
        """Hybrid search: Reciprocal Rank Fusion of FTS + vector results.

        RRF score = 1/(k + fts_rank) + 1/(k + vec_rank) per result.
        k=60 is the standard RRF constant (balances head vs tail).
        """
        # Get both result sets (fetch more than limit for better fusion)
        fetch = limit * 3
        fts_results = self.search_text(query, namespace=namespace, limit=fetch)
        vec_results = self.search_semantic(query_embedding, namespace=namespace, limit=fetch)

        # Build RRF scores
        scores: dict[str, float] = {}
        bundle_map: dict[str, dict] = {}

        for rank, r in enumerate(fts_results, 1):
            rid = r["rid"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)
            bundle_map[rid] = r

        for rank, r in enumerate(vec_results, 1):
            rid = r["rid"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)
            if rid not in bundle_map:
                bundle_map[rid] = r

        # Sort by RRF score, return top results
        sorted_rids = sorted(scores, key=lambda rid: scores[rid], reverse=True)[:limit]
        results = []
        for rid in sorted_rids:
            r = bundle_map[rid]
            r["rrf_score"] = scores[rid]
            results.append(r)
        return results

    def get_embedding_stats(self) -> dict:
        """Embedding coverage per namespace, grouped by model."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT b.namespace,
                   count(b.rid) AS total,
                   count(e.rid) AS embedded,
                   e.model
            FROM bundles b
            LEFT JOIN embeddings e ON b.rid = e.rid
            GROUP BY b.namespace, e.model
            ORDER BY b.namespace
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def get_unembedded_rids(self, namespace: str | None = None, limit: int = 500) -> list[dict]:
        """Find bundles that don't have embeddings yet."""
        conn = self._get_conn()
        if namespace:
            rows = conn.execute(
                """
                SELECT b.rid, b.namespace, b.search_text
                FROM bundles b
                LEFT JOIN embeddings e ON b.rid = e.rid
                WHERE e.rid IS NULL AND b.namespace = %s
                ORDER BY b.created_at
                LIMIT %s
                """,
                (namespace, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT b.rid, b.namespace, b.search_text
                FROM bundles b
                LEFT JOIN embeddings e ON b.rid = e.rid
                WHERE e.rid IS NULL
                ORDER BY b.created_at
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- Multi-config embedding methods --

    def register_embedding_config(
        self,
        config_id: str,
        provider: str,
        model: str,
        dimensions: int,
        chunk_size: int = 1000,
        chunk_overlap: float = 0.0,
        description: str = "",
        is_default: bool = False,
    ) -> str:
        """Register an embedding configuration and create its table. Returns config_id."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO embedding_configs (config_id, provider, model, dimensions, chunk_size, chunk_overlap, description, is_default)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (config_id) DO UPDATE SET
                provider = EXCLUDED.provider, model = EXCLUDED.model,
                dimensions = EXCLUDED.dimensions, chunk_size = EXCLUDED.chunk_size,
                chunk_overlap = EXCLUDED.chunk_overlap, description = EXCLUDED.description,
                is_default = EXCLUDED.is_default
            """,
            (config_id, provider, model, dimensions, chunk_size, chunk_overlap, description, is_default),
        )
        # Create per-config table
        table = _config_table_name(config_id)
        try:
            conn.execute(_config_table_sql(config_id, dimensions))
            log.info("embedding_config.registered", config_id=config_id, table=table, dimensions=dimensions)
        except Exception:
            # Table might exist with wrong dimension — check and recreate
            try:
                row = conn.execute(
                    f"SELECT atttypmod FROM pg_attribute WHERE attrelid = '{table}'::regclass AND attname = 'embedding'"
                ).fetchone()
                if row and row["atttypmod"] != dimensions:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
                    conn.execute(_config_table_sql(config_id, dimensions))
                    log.info("embedding_config.recreated", config_id=config_id, dimensions=dimensions)
            except Exception:
                conn.execute(_config_table_sql(config_id, dimensions))
        return config_id

    def list_embedding_configs(self) -> list[dict]:
        """List all registered embedding configurations."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM embedding_configs ORDER BY created_at").fetchall()
        return [dict(r) for r in rows]

    def get_default_config(self) -> dict | None:
        """Get the default embedding config."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM embedding_configs WHERE is_default = TRUE LIMIT 1").fetchone()
        return dict(row) if row else None

    def upsert_config_embedding(self, config_id: str, rid: str, embedding: list[float], chunk_index: int = 0, chunk_text: str | None = None):
        """Insert or update an embedding in a config-specific table."""
        table = _config_table_name(config_id)
        conn = self._get_conn()
        conn.execute(
            f"""
            INSERT INTO {table} (rid, chunk_index, chunk_text, embedding)
            VALUES (%s, %s, %s, %s::vector)
            ON CONFLICT (rid, chunk_index) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                created_at = NOW()
            """,
            (rid, chunk_index, chunk_text, self._vec_literal(embedding)),
        )

    def upsert_config_embeddings_batch(self, config_id: str, items: list[dict]):
        """Bulk upsert embeddings into a config-specific table.

        Each dict: {rid, embedding, chunk_index (default 0), chunk_text (optional)}.
        """
        if not items:
            return
        table = _config_table_name(config_id)
        conn = self._get_conn()
        with conn.transaction():
            with conn.cursor() as cur:
                for item in items:
                    cur.execute(
                        f"""
                        INSERT INTO {table} (rid, chunk_index, chunk_text, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        ON CONFLICT (rid, chunk_index) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            embedding = EXCLUDED.embedding,
                            created_at = NOW()
                        """,
                        (
                            item["rid"],
                            item.get("chunk_index", 0),
                            item.get("chunk_text"),
                            self._vec_literal(item["embedding"]),
                        ),
                    )

    def search_config_semantic(
        self, config_id: str, query_embedding: list[float], namespace: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Semantic search against a specific embedding config."""
        table = _config_table_name(config_id)
        conn = self._get_conn()
        vec = self._vec_literal(query_embedding)
        if namespace:
            rows = conn.execute(
                f"""
                SELECT b.rid, b.namespace, b.reference, b.contents, b.search_text,
                       e.chunk_index, e.chunk_text,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM {table} e
                JOIN bundles b ON b.rid = e.rid
                WHERE b.namespace = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, namespace, vec, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT b.rid, b.namespace, b.reference, b.contents, b.search_text,
                       e.chunk_index, e.chunk_text,
                       1 - (e.embedding <=> %s::vector) AS similarity
                FROM {table} e
                JOIN bundles b ON b.rid = e.rid
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, vec, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_config_hybrid(
        self,
        config_id: str,
        query: str,
        query_embedding: list[float],
        namespace: str | None = None,
        limit: int = 20,
        k: int = 60,
    ) -> list[dict]:
        """Hybrid search against a specific embedding config using RRF."""
        fetch = limit * 3
        fts_results = self.search_text(query, namespace=namespace, limit=fetch)
        vec_results = self.search_config_semantic(config_id, query_embedding, namespace=namespace, limit=fetch)

        scores: dict[str, float] = {}
        bundle_map: dict[str, dict] = {}

        for rank, r in enumerate(fts_results, 1):
            rid = r["rid"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)
            bundle_map[rid] = r

        for rank, r in enumerate(vec_results, 1):
            rid = r["rid"]
            scores[rid] = scores.get(rid, 0) + 1.0 / (k + rank)
            if rid not in bundle_map:
                bundle_map[rid] = r

        sorted_rids = sorted(scores, key=lambda rid: scores[rid], reverse=True)[:limit]
        results = []
        for rid in sorted_rids:
            r = bundle_map[rid]
            r["rrf_score"] = scores[rid]
            results.append(r)
        return results

    def get_config_stats(self) -> list[dict]:
        """Embedding coverage per config per namespace."""
        conn = self._get_conn()
        configs = self.list_embedding_configs()
        results = []
        for cfg in configs:
            table = _config_table_name(cfg["config_id"])
            try:
                rows = conn.execute(
                    f"""
                    SELECT b.namespace, count(DISTINCT e.rid) AS embedded, count(b.rid) AS total
                    FROM bundles b
                    LEFT JOIN {table} e ON b.rid = e.rid
                    GROUP BY b.namespace
                    ORDER BY b.namespace
                    """
                ).fetchall()
                for r in rows:
                    results.append({
                        "config_id": cfg["config_id"],
                        "model": cfg["model"],
                        "dimensions": cfg["dimensions"],
                        "namespace": r["namespace"],
                        "embedded": r["embedded"],
                        "total": r["total"],
                    })
            except Exception:
                log.warning("embedding_config.stats_error", config_id=cfg["config_id"])
        return results

    def get_config_unembedded_rids(self, config_id: str, namespace: str | None = None, limit: int = 500) -> list[dict]:
        """Find bundles not yet embedded for a specific config."""
        table = _config_table_name(config_id)
        conn = self._get_conn()
        if namespace:
            rows = conn.execute(
                f"""
                SELECT b.rid, b.namespace, b.search_text
                FROM bundles b
                LEFT JOIN {table} e ON b.rid = e.rid
                WHERE e.rid IS NULL AND b.namespace = %s
                ORDER BY b.created_at
                LIMIT %s
                """,
                (namespace, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT b.rid, b.namespace, b.search_text
                FROM bundles b
                LEFT JOIN {table} e ON b.rid = e.rid
                WHERE e.rid IS NULL
                ORDER BY b.created_at
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            log.info("postgres.closed")
