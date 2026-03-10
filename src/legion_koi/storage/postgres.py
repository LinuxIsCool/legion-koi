"""PostgreSQL storage for KOI-net bundles with full-text search."""

import json
from datetime import datetime, timezone

import structlog
import psycopg
from psycopg.rows import dict_row

log = structlog.stdlib.get_logger()

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

# Deferred until pgvector is installed (Phase 2c)
EMBEDDINGS_SQL = """
CREATE TABLE IF NOT EXISTS embeddings (
    rid TEXT PRIMARY KEY REFERENCES bundles(rid) ON DELETE CASCADE,
    embedding vector(1536),
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


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

    # Fallback: JSON dump
    return json.dumps(contents)


class PostgresStorage:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn: psycopg.Connection | None = None

    def _get_conn(self) -> psycopg.Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

    def initialize(self):
        """Create tables and indexes if they don't exist."""
        conn = self._get_conn()
        conn.execute(SCHEMA_SQL)
        try:
            conn.execute(EMBEDDINGS_SQL)
        except psycopg.errors.UndefinedObject:
            log.info("postgres.pgvector_not_installed", msg="Embeddings table deferred")
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

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            log.info("postgres.closed")
