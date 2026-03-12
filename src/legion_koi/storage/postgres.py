"""PostgreSQL storage for KOI-net bundles with full-text search and vector embeddings."""

import json
from importlib.resources import files as pkg_files
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

ENTITIES_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS entities (
    entity_id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    supertype TEXT NOT NULL DEFAULT '',
    name_normalized TEXT NOT NULL,
    description TEXT,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mention_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(name_normalized, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entities_name_norm ON entities(name_normalized);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_supertype ON entities(supertype);
CREATE INDEX IF NOT EXISTS idx_entities_trgm ON entities USING GIN(name_normalized gin_trgm_ops);

CREATE TABLE IF NOT EXISTS bundle_entities (
    rid TEXT NOT NULL REFERENCES bundles(rid) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    mention_count INTEGER NOT NULL DEFAULT 1,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (rid, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_bundle_entities_entity ON bundle_entities(entity_id);

CREATE TABLE IF NOT EXISTS relations (
    relation_id SERIAL PRIMARY KEY,
    source_entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    evidence TEXT DEFAULT '',
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mention_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(source_entity_id, target_entity_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id);

CREATE TABLE IF NOT EXISTS bundle_relations (
    rid TEXT NOT NULL REFERENCES bundles(rid) ON DELETE CASCADE,
    relation_id INTEGER NOT NULL REFERENCES relations(relation_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (rid, relation_id)
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


from ..constants import MAX_SEARCH_TEXT, RRF_K, RRF_FETCH_MULTIPLIER, DEFAULT_SEARCH_LIMIT, DEFAULT_THREAD_LIMIT


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

    if namespace == "legion.claude-logging":
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
        return text[:MAX_SEARCH_TEXT]

    if namespace == "legion.claude-message":
        text = contents.get("content", "") or ""
        return text[:MAX_SEARCH_TEXT]

    if namespace == "legion.claude-web.conversation":
        parts = []
        name = contents.get("name") or ""
        if name:
            parts.append(name)
        summary = contents.get("summary") or ""
        if summary:
            parts.append(summary)
        for msg in contents.get("chat_messages", []):
            text = msg.get("text") or ""
            if text:
                parts.append(text)
        return "\n".join(parts)[:MAX_SEARCH_TEXT]

    if namespace == "legion.claude-web.project":
        parts = []
        name = contents.get("name") or ""
        if name:
            parts.append(name)
        desc = contents.get("description") or ""
        if desc:
            parts.append(desc)
        prompt = contents.get("prompt_template") or ""
        if prompt:
            parts.append(prompt)
        for doc in contents.get("docs", []):
            filename = doc.get("filename") or ""
            if filename:
                parts.append(filename)
            content = doc.get("content") or ""
            if content:
                parts.append(content)
        return "\n".join(parts)[:MAX_SEARCH_TEXT]

    if namespace == "legion.claude-web.memory":
        text = contents.get("conversations_memory", "")
        for _uuid, mem_text in contents.get("project_memories", {}).items():
            if isinstance(mem_text, str):
                text += "\n" + mem_text
        return text[:MAX_SEARCH_TEXT]

    if namespace == "legion.claude-code":
        parts = []
        cwd = contents.get("cwd") or ""
        if cwd:
            parts.append(cwd)
        summary = contents.get("summary") or ""
        if summary:
            parts.append(summary)
        return "\n".join(parts)[:MAX_SEARCH_TEXT]

    if namespace == "legion.claude-github":
        parts = []
        name = contents.get("name") or ""
        if name:
            parts.append(name)
        desc = contents.get("description") or ""
        if desc:
            parts.append(desc)
        topics = contents.get("topics") or []
        if topics:
            parts.append(" ".join(topics))
        readme = contents.get("readme_content") or ""
        if readme:
            parts.append(readme)
        return "\n".join(parts)[:MAX_SEARCH_TEXT]

    # Fallback: JSON dump
    text = json.dumps(contents)
    return text[:MAX_SEARCH_TEXT]


class PostgresStorage:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

    def initialize(self):
        """Create tables and indexes."""
        conn = self._get_conn()
        conn.execute(SCHEMA_SQL)

        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg.errors.InsufficientPrivilege:
            pass
        conn.execute(EMBEDDING_CONFIGS_SQL)

        # Entity extraction tables (pg_trgm for fuzzy name search)
        try:
            conn.execute(ENTITIES_SCHEMA_SQL)
        except psycopg.errors.InsufficientPrivilege:
            log.warning("postgres.pg_trgm_unavailable")
        except Exception as e:
            log.warning("postgres.entities_schema_error", error=str(e))

        # Event system trigger (Phase 1) — fires PG NOTIFY on bundle changes
        try:
            from pathlib import Path
            trigger_sql_path = Path(__file__).parent.parent / "events" / "pg_trigger.sql"
            trigger_sql = trigger_sql_path.read_text()
            conn.execute(trigger_sql)
            log.info("postgres.trigger_installed", trigger="bundles_notify")
        except Exception as e:
            log.warning("postgres.trigger_error", error=str(e))

        log.info("postgres.initialized")

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

    def search_text(self, query: str, namespace: str | None = None, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        """Full-text search across bundles with ts_headline snippet extraction."""
        conn = self._get_conn()
        headline_opts = (
            "MaxFragments=3, MaxWords=60, MinWords=20, "
            "FragmentDelimiter= ... , StartSel=**, StopSel=**"
        )
        if namespace:
            rows = conn.execute(
                f"""
                SELECT rid, namespace, reference, contents, search_text,
                       ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS rank,
                       ts_headline('english', search_text, websearch_to_tsquery('english', %s),
                                   '{headline_opts}') AS headline
                FROM bundles
                WHERE search_vector @@ websearch_to_tsquery('english', %s)
                  AND namespace = %s
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, query, namespace, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"""
                SELECT rid, namespace, reference, contents, search_text,
                       ts_rank(search_vector, websearch_to_tsquery('english', %s)) AS rank,
                       ts_headline('english', search_text, websearch_to_tsquery('english', %s),
                                   '{headline_opts}') AS headline
                FROM bundles
                WHERE search_vector @@ websearch_to_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
                """,
                (query, query, query, limit),
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

    def get_thread_messages(self, thread_id: str, limit: int = DEFAULT_THREAD_LIMIT) -> list[dict]:
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

    def delete_config_embeddings(self, config_id: str, rid: str) -> None:
        """Delete all chunks for a RID from a config table."""
        table = _config_table_name(config_id)
        conn = self._get_conn()
        conn.execute(f"DELETE FROM {table} WHERE rid = %s", (rid,))

    def search_config_semantic(
        self, config_id: str, query_embedding: list[float], namespace: str | None = None, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict]:
        """Semantic search against a specific embedding config.

        Oversamples chunks and deduplicates by RID (keeping best similarity)
        to handle multi-chunk documents correctly.
        """
        table = _config_table_name(config_id)
        conn = self._get_conn()
        vec = self._vec_literal(query_embedding)
        # Oversample to ensure enough unique RIDs after dedup
        fetch_limit = limit * 5
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
                (vec, namespace, vec, fetch_limit),
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
                (vec, vec, fetch_limit),
            ).fetchall()
        # Deduplicate: keep best chunk per RID
        seen: dict[str, dict] = {}
        for r in rows:
            row = dict(r)
            rid = row["rid"]
            if rid not in seen or row["similarity"] > seen[rid]["similarity"]:
                seen[rid] = row
        results = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)[:limit]
        return results

    def search_config_hybrid(
        self,
        config_id: str,
        query: str,
        query_embedding: list[float],
        namespace: str | None = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
        k: int = RRF_K,
    ) -> list[dict]:
        """Hybrid search against a specific embedding config using RRF."""
        fetch = limit * RRF_FETCH_MULTIPLIER
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
            elif r.get("chunk_text") and not bundle_map[rid].get("chunk_text"):
                # Preserve chunk_text from vector results for reranking
                bundle_map[rid]["chunk_text"] = r["chunk_text"]

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

    # -- Entity extraction methods --

    def upsert_bundle_entities(self, rid: str, entities: list[dict]) -> int:
        """Store extracted entities for a bundle.

        Each entity dict: {name, entity_type, supertype, confidence, name_normalized}.
        Upserts into entities table, links via bundle_entities.
        Returns number of entities linked.
        """
        if not entities:
            return 0
        conn = self._get_conn()
        now = datetime.now(timezone.utc)
        linked = 0

        # Clear old extractions for this bundle
        conn.execute("DELETE FROM bundle_entities WHERE rid = %s", (rid,))

        with conn.transaction():
            with conn.cursor() as cur:
                for e in entities:
                    name_norm = e["name_normalized"]
                    # Upsert entity
                    cur.execute(
                        """
                        INSERT INTO entities (name, entity_type, supertype, name_normalized,
                                              first_seen, last_seen, mention_count)
                        VALUES (%(name)s, %(entity_type)s, %(supertype)s, %(name_normalized)s,
                                %(now)s, %(now)s, 1)
                        ON CONFLICT (name_normalized, entity_type) DO UPDATE SET
                            last_seen = %(now)s,
                            mention_count = entities.mention_count + 1,
                            name = CASE
                                WHEN length(%(name)s) > length(entities.name)
                                THEN %(name)s ELSE entities.name
                            END
                        RETURNING entity_id
                        """,
                        {
                            "name": e["name"],
                            "entity_type": e["entity_type"],
                            "supertype": e.get("supertype", ""),
                            "name_normalized": name_norm,
                            "now": now,
                        },
                    )
                    row = cur.fetchone()
                    entity_id = row["entity_id"]

                    # Link to bundle
                    cur.execute(
                        """
                        INSERT INTO bundle_entities (rid, entity_id, confidence, created_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (rid, entity_id) DO UPDATE SET
                            confidence = GREATEST(bundle_entities.confidence, EXCLUDED.confidence)
                        """,
                        (rid, entity_id, e.get("confidence", 1.0), now),
                    )
                    linked += 1

        return linked

    def get_bundle_entities(self, rid: str) -> list[dict]:
        """Get entities extracted from a specific bundle."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT e.entity_id, e.name, e.entity_type, e.supertype,
                   be.confidence, e.mention_count
            FROM bundle_entities be
            JOIN entities e ON e.entity_id = be.entity_id
            WHERE be.rid = %s
            ORDER BY be.confidence DESC
            """,
            (rid,),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_bundles_by_entity(
        self, name: str, entity_type: str | None = None, limit: int = 20
    ) -> list[dict]:
        """Find bundles mentioning an entity. Trigram matching on name."""
        conn = self._get_conn()
        name_norm = " ".join(name.lower().split())
        if entity_type:
            rows = conn.execute(
                """
                SELECT b.rid, b.namespace, b.reference, be.confidence,
                       e.name AS entity_name, e.entity_type,
                       similarity(e.name_normalized, %s) AS sim
                FROM bundle_entities be
                JOIN entities e ON e.entity_id = be.entity_id
                JOIN bundles b ON b.rid = be.rid
                WHERE e.name_normalized %% %s AND e.entity_type = %s
                ORDER BY sim DESC, be.confidence DESC
                LIMIT %s
                """,
                (name_norm, name_norm, entity_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT b.rid, b.namespace, b.reference, be.confidence,
                       e.name AS entity_name, e.entity_type,
                       similarity(e.name_normalized, %s) AS sim
                FROM bundle_entities be
                JOIN entities e ON e.entity_id = be.entity_id
                JOIN bundles b ON b.rid = be.rid
                WHERE e.name_normalized %% %s
                ORDER BY sim DESC, be.confidence DESC
                LIMIT %s
                """,
                (name_norm, name_norm, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_entity_cooccurrence(self, name: str, limit: int = 20) -> list[dict]:
        """Find entities that co-appear in bundles with the given entity."""
        conn = self._get_conn()
        name_norm = " ".join(name.lower().split())
        rows = conn.execute(
            """
            SELECT e2.name, e2.entity_type, e2.supertype,
                   count(DISTINCT be2.rid) AS shared_bundles,
                   e2.mention_count
            FROM entities e1
            JOIN bundle_entities be1 ON be1.entity_id = e1.entity_id
            JOIN bundle_entities be2 ON be2.rid = be1.rid AND be2.entity_id != e1.entity_id
            JOIN entities e2 ON e2.entity_id = be2.entity_id
            WHERE e1.name_normalized %% %s
            GROUP BY e2.entity_id, e2.name, e2.entity_type, e2.supertype, e2.mention_count
            ORDER BY shared_bundles DESC, e2.mention_count DESC
            LIMIT %s
            """,
            (name_norm, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_entities(
        self,
        query: str,
        entity_type: str | None = None,
        supertype: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Search entities by name with optional type/supertype filter."""
        conn = self._get_conn()
        query_norm = " ".join(query.lower().split())
        conditions = ["e.name_normalized %% %s"]
        params: list = [query_norm]

        if entity_type:
            conditions.append("e.entity_type = %s")
            params.append(entity_type)
        if supertype:
            conditions.append("e.supertype = %s")
            params.append(supertype)

        where = " AND ".join(conditions)
        params.extend([query_norm, query_norm, limit])

        rows = conn.execute(
            f"""
            SELECT e.entity_id, e.name, e.entity_type, e.supertype,
                   e.mention_count, e.first_seen, e.last_seen,
                   similarity(e.name_normalized, %s) AS sim
            FROM entities e
            WHERE {where}
            ORDER BY similarity(e.name_normalized, %s) DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_unextracted_rids(
        self, namespace: str | None = None, limit: int = 500
    ) -> list[dict]:
        """Find bundles with no entity extractions."""
        conn = self._get_conn()
        if namespace:
            rows = conn.execute(
                """
                SELECT b.rid, b.namespace, b.search_text
                FROM bundles b
                LEFT JOIN bundle_entities be ON b.rid = be.rid
                WHERE be.rid IS NULL AND b.namespace = %s
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
                LEFT JOIN bundle_entities be ON b.rid = be.rid
                WHERE be.rid IS NULL
                ORDER BY b.created_at
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_entity_stats(self) -> dict:
        """Entity extraction statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT count(*) AS cnt FROM entities").fetchone()
        by_type = conn.execute(
            "SELECT entity_type, count(*) AS cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC"
        ).fetchall()
        by_supertype = conn.execute(
            "SELECT supertype, count(*) AS cnt FROM entities GROUP BY supertype ORDER BY cnt DESC"
        ).fetchall()
        coverage = conn.execute(
            """
            SELECT
                (SELECT count(DISTINCT rid) FROM bundle_entities) AS extracted,
                (SELECT count(*) FROM bundles) AS total
            """
        ).fetchone()
        top_entities = conn.execute(
            """
            SELECT name, entity_type, supertype, mention_count
            FROM entities ORDER BY mention_count DESC LIMIT 20
            """
        ).fetchall()
        return {
            "total_entities": total["cnt"],
            "by_type": {r["entity_type"]: r["cnt"] for r in by_type},
            "by_supertype": {r["supertype"]: r["cnt"] for r in by_supertype},
            "extraction_coverage": {
                "bundles_with_entities": coverage["extracted"],
                "total_bundles": coverage["total"],
            },
            "top_entities": [dict(r) for r in top_entities],
        }

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            log.info("postgres.closed")
