#!/usr/bin/env -S python3 -u
"""Bulk backfill embeddings for all bundles in PostgreSQL.

Resilient: retries with exponential backoff, per-item fallback on batch failure,
progress logging, reconnects on DB errors.

Usage:
    uv run python scripts/backfill_embeddings.py
    uv run python scripts/backfill_embeddings.py --provider ollama --model mxbai-embed-large
    uv run python scripts/backfill_embeddings.py --namespace legion.claude-journal
"""

import argparse
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.embeddings import create_embedder, _BATCH_SIZE


DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")

NAMESPACES_ORDER = [
    "koi-net.node",
    "legion.claude-venture",
    "legion.claude-logging",
    "legion.claude-journal",
    "legion.claude-recording",
    "legion.claude-message",
    "legion.claude-web.conversation",
    "legion.claude-web.project",
    "legion.claude-web.memory",
    "legion.claude-code",
    "legion.claude-github",
]

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 15]  # seconds between retries


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def _is_client_error(e: Exception) -> bool:
    """Check if this is a 4xx error (content problem, won't succeed on retry)."""
    return "400" in str(e) or "422" in str(e) or "413" in str(e)


def embed_batch_resilient(embedder, rids, texts, model, conn, stats):
    """Embed a batch with retry and per-item fallback.

    4xx errors skip retries (content problem, not transient).
    5xx/timeout errors retry with exponential backoff.
    """
    # Try batch first
    for attempt in range(MAX_RETRIES):
        try:
            vectors = embedder.embed_batch(texts, input_type="passage")
            _upsert_batch(conn, list(zip(rids, vectors)), model)
            stats["embedded"] += len(vectors)
            return
        except Exception as e:
            if _is_client_error(e):
                # Content problem — go straight to per-item fallback
                break
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"    Batch retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {type(e).__name__}")
                time.sleep(wait)
            else:
                print(f"    Batch failed after {MAX_RETRIES} retries, falling back to single-item")

    # Per-item fallback — no retries for 4xx, retries for transient errors
    for rid, text in zip(rids, texts):
        try:
            vec = embedder.embed(text, input_type="passage")
            _upsert_batch(conn, [(rid, vec)], model)
            stats["embedded"] += 1
        except Exception as e:
            if _is_client_error(e):
                stats["skipped"] += 1
                stats["failed_rids"].append(rid)
                continue
            # Transient error — retry once
            try:
                time.sleep(2)
                vec = embedder.embed(text, input_type="passage")
                _upsert_batch(conn, [(rid, vec)], model)
                stats["embedded"] += 1
            except Exception:
                stats["skipped"] += 1
                stats["failed_rids"].append(rid)


def _upsert_batch(conn, pairs, model):
    """Upsert (rid, vector) pairs."""
    with conn.transaction():
        with conn.cursor() as cur:
            for rid, vec in pairs:
                cur.execute(
                    """
                    INSERT INTO embeddings (rid, embedding, model)
                    VALUES (%s, %s::vector, %s)
                    ON CONFLICT (rid) DO UPDATE SET
                        embedding = EXCLUDED.embedding, model = EXCLUDED.model, created_at = NOW()
                    """,
                    (rid, vec_literal(vec), model),
                )


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings for legion-koi bundles")
    parser.add_argument("--provider", choices=["telus", "ollama"], help="Embedding provider")
    parser.add_argument("--model", help="Model override")
    parser.add_argument("--namespace", help="Only backfill this namespace")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE, help="Texts per API call")
    parser.add_argument("--fetch-size", type=int, default=500, help="DB fetch batch size")
    parser.add_argument("--dry-run", action="store_true", help="Count without embedding")
    args = parser.parse_args()

    embedder = create_embedder(provider=args.provider, model=args.model)
    model = embedder.get_model()
    dims = embedder.get_dimensions()
    print(f"Embedder: {model} ({dims} dims)")

    if not args.dry_run:
        avail = embedder.is_available()
        if not avail:
            print("ERROR: Embedder is not available. Check credentials/connectivity.")
            sys.exit(1)
        print("Embedder connectivity: OK")

    # Ensure table exists
    from legion_koi.storage.postgres import PostgresStorage
    storage = PostgresStorage(dsn=DSN)
    if not args.dry_run:
        storage.initialize(embedding_dim=dims)
        print(f"Embeddings table ready (vector({dims}))")
    storage.close()

    conn = get_conn()
    namespaces = [args.namespace] if args.namespace else NAMESPACES_ORDER

    total_stats = {"embedded": 0, "skipped": 0, "failed_rids": []}
    start_time = time.time()

    for ns in namespaces:
        row = conn.execute(
            """
            SELECT count(*) AS cnt FROM bundles b
            LEFT JOIN embeddings e ON b.rid = e.rid
            WHERE e.rid IS NULL AND b.namespace = %s
              AND btrim(b.search_text) != ''
              AND b.search_text NOT LIKE %s
              AND b.search_text NOT LIKE %s
              AND b.search_text NOT LIKE %s
              AND left(b.search_text, 100) !~ '[A-Za-z0-9+/=]{60,}'
            """,
            (ns, '--00000000%', '<!doctype%', '<!DOCTYPE%'),
        ).fetchone()
        unembedded = row["cnt"]

        if unembedded == 0:
            print(f"[{ns}] All embedded, skipping")
            continue

        print(f"[{ns}] {unembedded} unembedded bundles")

        if args.dry_run:
            continue

        ns_stats = {"embedded": 0, "skipped": 0, "failed_rids": []}

        while True:
            try:
                rows = conn.execute(
                    """
                    SELECT b.rid, b.search_text FROM bundles b
                    LEFT JOIN embeddings e ON b.rid = e.rid
                    WHERE e.rid IS NULL AND b.namespace = %s
                      AND btrim(b.search_text) != ''
                      AND b.search_text NOT LIKE %s
                      AND b.search_text NOT LIKE %s
                      AND b.search_text NOT LIKE %s
                      AND left(b.search_text, 100) !~ '[A-Za-z0-9+/=]{60,}'
                    ORDER BY b.created_at
                    LIMIT %s
                    """,
                    (ns, '--00000000%', '<!doctype%', '<!DOCTYPE%', args.fetch_size),
                ).fetchall()
            except Exception as e:
                print(f"  DB error fetching: {e}. Reconnecting...")
                time.sleep(2)
                try:
                    conn.close()
                except Exception:
                    pass
                conn = get_conn()
                continue

            if not rows:
                break

            for i in range(0, len(rows), args.batch_size):
                batch = rows[i:i + args.batch_size]
                valid = [(r["rid"], r["search_text"]) for r in batch
                         if r["search_text"] and r["search_text"].strip()]

                if not valid:
                    ns_stats["skipped"] += len(batch)
                    continue

                rids = [v[0] for v in valid]
                texts = [v[1] for v in valid]

                embed_batch_resilient(embedder, rids, texts, model, conn, ns_stats)

                elapsed = time.time() - start_time
                done = total_stats["embedded"] + ns_stats["embedded"]
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{ns}] {ns_stats['embedded']}/{unembedded} embedded ({rate:.1f}/s, model: {model})")

                time.sleep(0.1)

        total_stats["embedded"] += ns_stats["embedded"]
        total_stats["skipped"] += ns_stats["skipped"]
        total_stats["failed_rids"].extend(ns_stats["failed_rids"])
        print(f"  [{ns}] Done: {ns_stats['embedded']} embedded, {ns_stats['skipped']} skipped")

    elapsed = time.time() - start_time
    print(f"\nComplete: {total_stats['embedded']} embedded, {total_stats['skipped']} skipped in {elapsed:.1f}s")

    if total_stats["failed_rids"]:
        print(f"\nFailed RIDs ({len(total_stats['failed_rids'])}):")
        for rid in total_stats["failed_rids"][:50]:
            print(f"  {rid}")
        if len(total_stats["failed_rids"]) > 50:
            print(f"  ... and {len(total_stats['failed_rids']) - 50} more")

    conn.close()


if __name__ == "__main__":
    main()
