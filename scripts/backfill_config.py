#!/usr/bin/env -S python3 -u
"""Backfill embeddings for a specific embedding configuration.

Supports any registered config — different models, dimensions, chunk sizes.
Can run multiple instances in parallel for different configs.

Usage:
    # Register + backfill with Telus E5 (default)
    uv run python scripts/backfill_config.py --config telus-e5-1024

    # Register + backfill with Ollama mxbai
    uv run python scripts/backfill_config.py --config ollama-mxbai-1024 --provider ollama --model mxbai-embed-large

    # Register + backfill with Ollama nomic
    uv run python scripts/backfill_config.py --config ollama-nomic-768 --provider ollama --model nomic-embed-text

    # List all configs
    uv run python scripts/backfill_config.py --list

    # Dry run
    uv run python scripts/backfill_config.py --config telus-e5-1024 --dry-run
"""

import argparse
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.embeddings import create_embedder, _BATCH_SIZE
from legion_koi.storage.postgres import PostgresStorage, _config_table_name

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
RETRY_BACKOFF = [2, 5, 15]


def _is_client_error(e: Exception) -> bool:
    return "400" in str(e) or "422" in str(e) or "413" in str(e)


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def embed_batch_resilient(embedder, rids, texts, config_id, conn, stats):
    """Embed a batch with retry and per-item fallback."""
    table = _config_table_name(config_id)

    for attempt in range(MAX_RETRIES):
        try:
            vectors = embedder.embed_batch(texts, input_type="passage")
            _upsert_batch(conn, table, list(zip(rids, texts, vectors)))
            stats["embedded"] += len(vectors)
            return
        except Exception as e:
            if _is_client_error(e):
                break
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"    Batch retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {type(e).__name__}")
                time.sleep(wait)

    # Per-item fallback
    for rid, text in zip(rids, texts):
        try:
            vec = embedder.embed(text, input_type="passage")
            _upsert_batch(conn, table, [(rid, text, vec)])
            stats["embedded"] += 1
        except Exception as e:
            if _is_client_error(e):
                stats["skipped"] += 1
                continue
            try:
                time.sleep(2)
                vec = embedder.embed(text, input_type="passage")
                _upsert_batch(conn, table, [(rid, text, vec)])
                stats["embedded"] += 1
            except Exception:
                stats["skipped"] += 1


def _upsert_batch(conn, table, triples):
    """Upsert (rid, text, vector) triples into config table."""
    with conn.transaction():
        with conn.cursor() as cur:
            for rid, text, vec in triples:
                cur.execute(
                    f"""
                    INSERT INTO {table} (rid, chunk_index, chunk_text, embedding)
                    VALUES (%s, 0, %s, %s::vector)
                    ON CONFLICT (rid, chunk_index) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,
                    (rid, text[:200], vec_literal(vec)),
                )


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings for a specific config")
    parser.add_argument("--config", help="Config ID (e.g., telus-e5-1024, ollama-mxbai-1024)")
    parser.add_argument("--provider", choices=["telus", "ollama"], help="Embedding provider")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max chars per chunk")
    parser.add_argument("--description", default="", help="Config description")
    parser.add_argument("--default", action="store_true", help="Set as default config")
    parser.add_argument("--namespace", help="Only backfill this namespace")
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE, help="Texts per API call")
    parser.add_argument("--fetch-size", type=int, default=500, help="DB fetch batch size")
    parser.add_argument("--dry-run", action="store_true", help="Count without embedding")
    parser.add_argument("--list", action="store_true", help="List all configs and exit")
    args = parser.parse_args()

    storage = PostgresStorage(dsn=DSN)
    storage.initialize()

    if args.list:
        configs = storage.list_embedding_configs()
        if not configs:
            print("No embedding configs registered.")
        for cfg in configs:
            default = " [DEFAULT]" if cfg["is_default"] else ""
            table = _config_table_name(cfg["config_id"])
            print(f"  {cfg['config_id']}{default}: {cfg['provider']}/{cfg['model']} "
                  f"({cfg['dimensions']}d, chunk={cfg['chunk_size']})")
            # Count
            try:
                conn = get_conn()
                row = conn.execute(f"SELECT count(*) AS cnt FROM {table}").fetchone()
                print(f"    Table: {table}, Rows: {row['cnt']}")
                conn.close()
            except Exception:
                print(f"    Table: {table} (not created)")
        storage.close()
        return

    if not args.config:
        print("ERROR: --config is required (e.g., --config telus-e5-1024)")
        sys.exit(1)

    # Create embedder
    embedder = create_embedder(provider=args.provider, model=args.model)
    model = embedder.get_model()
    dims = embedder.get_dimensions()
    provider = args.provider or ("telus" if "telus" in args.config or "e5" in args.config else "ollama")
    print(f"Config: {args.config}")
    print(f"Embedder: {provider}/{model} ({dims} dims, chunk_size={args.chunk_size})")

    if not args.dry_run:
        avail = embedder.is_available()
        if not avail:
            print("ERROR: Embedder is not available.")
            sys.exit(1)
        print("Embedder connectivity: OK")

    # Register config
    storage.register_embedding_config(
        config_id=args.config,
        provider=provider,
        model=model,
        dimensions=dims,
        chunk_size=args.chunk_size,
        description=args.description or f"{provider}/{model} {dims}d chunk={args.chunk_size}",
        is_default=args.default,
    )
    print(f"Config registered: {args.config}")
    storage.close()

    # Backfill
    conn = get_conn()
    table = _config_table_name(args.config)
    namespaces = [args.namespace] if args.namespace else NAMESPACES_ORDER

    total_stats = {"embedded": 0, "skipped": 0}
    start_time = time.time()

    for ns in namespaces:
        row = conn.execute(
            f"""
            SELECT count(*) AS cnt FROM bundles b
            LEFT JOIN {table} e ON b.rid = e.rid
            WHERE e.rid IS NULL AND b.namespace = %s
              AND btrim(b.search_text) != ''
              AND b.search_text NOT LIKE %s
              AND b.search_text NOT LIKE %s
              AND b.search_text NOT LIKE %s
              AND left(b.search_text, 100) !~ '[A-Za-z0-9+/=]{{60,}}'
            """,
            (ns, '--00000000%', '<!doctype%', '<!DOCTYPE%'),
        ).fetchone()
        unembedded = row["cnt"]

        if unembedded == 0:
            print(f"[{ns}] All embedded for {args.config}, skipping")
            continue

        print(f"[{ns}] {unembedded} unembedded bundles")

        if args.dry_run:
            continue

        ns_stats = {"embedded": 0, "skipped": 0}

        while True:
            try:
                rows = conn.execute(
                    f"""
                    SELECT b.rid, b.search_text FROM bundles b
                    LEFT JOIN {table} e ON b.rid = e.rid
                    WHERE e.rid IS NULL AND b.namespace = %s
                      AND btrim(b.search_text) != ''
                      AND b.search_text NOT LIKE %s
                      AND b.search_text NOT LIKE %s
                      AND b.search_text NOT LIKE %s
                      AND left(b.search_text, 100) !~ '[A-Za-z0-9+/=]{{60,}}'
                    ORDER BY b.created_at
                    LIMIT %s
                    """,
                    (ns, '--00000000%', '<!doctype%', '<!DOCTYPE%', args.fetch_size),
                ).fetchall()
            except Exception as e:
                print(f"  DB error: {e}. Reconnecting...")
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

                embed_batch_resilient(embedder, rids, texts, args.config, conn, ns_stats)

                elapsed = time.time() - start_time
                done = total_stats["embedded"] + ns_stats["embedded"]
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{ns}] {ns_stats['embedded']}/{unembedded} embedded "
                      f"({rate:.1f}/s, config: {args.config})")

                time.sleep(0.1)

        total_stats["embedded"] += ns_stats["embedded"]
        total_stats["skipped"] += ns_stats["skipped"]
        print(f"  [{ns}] Done: {ns_stats['embedded']} embedded, {ns_stats['skipped']} skipped")

    elapsed = time.time() - start_time
    print(f"\nComplete [{args.config}]: {total_stats['embedded']} embedded, "
          f"{total_stats['skipped']} skipped in {elapsed:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
