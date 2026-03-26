#!/usr/bin/env -S python3 -u
"""Backfill embeddings for bundles that have search_text but no embeddings.

Usage:
    uv run python scripts/backfill_embeddings.py --namespace legion.claude-youtube
    uv run python scripts/backfill_embeddings.py --dry-run
    uv run python scripts/backfill_embeddings.py --batch-size 50
    uv run python scripts/backfill_embeddings.py                         # all namespaces
"""

import argparse
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.chunking import chunk_text
from legion_koi.contextual import extract_preamble, prepend_preamble
from legion_koi.embeddings import create_embedder
from legion_koi.storage.postgres import PostgresStorage, _extract_search_text, _config_table_name

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")

NAMESPACES_ORDER = [
    "legion.claude-youtube",
    "legion.claude-journal",
    "legion.claude-venture",
    "legion.claude-web.conversation",
    "legion.claude-recording",
    "legion.claude-github",
    "legion.claude-code",
    "legion.claude-message",
]

# Minimum search_text length to bother embedding
MIN_EMBED_CHARS = 50


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def count_unembedded(conn, ns, config_id):
    """Count bundles in namespace with no embeddings for this config."""
    table = _config_table_name(config_id)
    row = conn.execute(
        f"""
        SELECT count(*) AS cnt FROM bundles b
        LEFT JOIN {table} be ON b.rid = be.rid
        WHERE be.rid IS NULL
          AND b.namespace = %s
          AND btrim(coalesce(b.search_text, '')) != ''
          AND length(b.search_text) >= %s
        """,
        (ns, MIN_EMBED_CHARS),
    ).fetchone()
    return row["cnt"]


def fetch_unembedded(conn, ns, config_id, limit):
    """Fetch bundles needing embeddings."""
    table = _config_table_name(config_id)
    return conn.execute(
        f"""
        SELECT b.rid, b.namespace, b.contents, b.search_text FROM bundles b
        LEFT JOIN {table} be ON b.rid = be.rid
        WHERE be.rid IS NULL
          AND b.namespace = %s
          AND btrim(coalesce(b.search_text, '')) != ''
          AND length(b.search_text) >= %s
        ORDER BY b.created_at
        LIMIT %s
        """,
        (ns, MIN_EMBED_CHARS, limit),
    ).fetchall()


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings for bundles")
    parser.add_argument("--namespace", help="Only backfill this namespace")
    parser.add_argument("--batch-size", type=int, default=25, help="Bundles per batch")
    parser.add_argument("--dry-run", action="store_true", help="Count only")
    parser.add_argument("--config-id", help="Only process this embedding config")
    args = parser.parse_args()

    storage = PostgresStorage(dsn=DSN)
    storage.initialize()
    conn = get_conn()

    configs = storage.list_embedding_configs()
    if args.config_id:
        configs = [c for c in configs if c["config_id"] == args.config_id]

    if not configs:
        print("No embedding configs found. Register one first.")
        return

    print(f"Embedding configs: {[c['config_id'] for c in configs]}")

    namespaces = [args.namespace] if args.namespace else NAMESPACES_ORDER
    total_stats = {"rids": 0, "chunks": 0, "errors": 0}
    start_time = time.time()

    for cfg in configs:
        config_id = cfg["config_id"]
        provider = cfg["provider"]
        model = cfg["model"]
        is_contextual = config_id.endswith("-ctx")

        print(f"\n=== Config: {config_id} (provider={provider}, model={model}, contextual={is_contextual}) ===")

        try:
            embedder = create_embedder(provider=provider, model=model)
        except Exception as e:
            print(f"  Failed to create embedder: {e}")
            continue

        for ns in namespaces:
            total_count = count_unembedded(conn, ns, config_id)
            if total_count == 0:
                print(f"  [{ns}] All embedded, skipping")
                continue

            if args.dry_run:
                print(f"  [{ns}] {total_count} unembedded bundles")
                continue

            print(f"  [{ns}] {total_count} unembedded bundles")
            ns_stats = {"rids": 0, "chunks": 0, "errors": 0}

            while True:
                try:
                    rows = fetch_unembedded(conn, ns, config_id, args.batch_size)
                except Exception as e:
                    print(f"    DB error: {e}. Reconnecting...")
                    time.sleep(2)
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = get_conn()
                    continue

                if not rows:
                    break

                for r in rows:
                    rid = r["rid"]
                    contents = r["contents"] or {}

                    # Get text to embed: try _extract_search_text, fall back to stored search_text
                    text = _extract_search_text(ns, contents)
                    if not text or not text.strip():
                        text = r.get("search_text") or ""
                    if not text or len(text) < MIN_EMBED_CHARS:
                        ns_stats["rids"] += 1
                        continue

                    try:
                        chunks = chunk_text(text)
                        if not chunks:
                            ns_stats["rids"] += 1
                            continue

                        preamble = extract_preamble(ns, contents) if is_contextual else ""

                        # Delete existing embeddings for this config+rid (in case of partial)
                        storage.delete_config_embeddings(config_id, rid)

                        for i, chunk in enumerate(chunks):
                            embed_input = prepend_preamble(preamble, chunk) if preamble else chunk
                            vec = embedder.embed(embed_input, input_type="passage")
                            storage.upsert_config_embedding(
                                config_id=config_id,
                                rid=rid,
                                embedding=vec,
                                chunk_index=i,
                                chunk_text=chunk,
                            )
                            ns_stats["chunks"] += 1

                        ns_stats["rids"] += 1
                        elapsed = time.time() - start_time
                        rate = (total_stats["rids"] + ns_stats["rids"]) / elapsed if elapsed > 0 else 0
                        print(
                            f"    [{ns}] {ns_stats['rids']}/{total_count} RIDs, "
                            f"{ns_stats['chunks']} chunks ({rate:.1f} RIDs/s) "
                            f"last: {len(chunks)} chunks from {len(text)} chars"
                        )
                    except Exception as e:
                        ns_stats["errors"] += 1
                        print(f"    [{ns}] ERROR on {rid}: {e}")

            for k in total_stats:
                total_stats[k] += ns_stats[k]
            print(f"    [{ns}] Done: {ns_stats['rids']} RIDs, {ns_stats['chunks']} chunks, "
                  f"{ns_stats['errors']} errors")

    elapsed = time.time() - start_time
    print(f"\nComplete: {total_stats['rids']} RIDs, {total_stats['chunks']} chunks, "
          f"{total_stats['errors']} errors in {elapsed:.1f}s")
    conn.close()
    storage.close()


if __name__ == "__main__":
    main()
