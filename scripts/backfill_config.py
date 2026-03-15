#!/usr/bin/env -S python3 -u
"""Backfill embeddings for a specific embedding configuration.

Supports any registered config — different models, dimensions, chunk sizes.
Can run multiple instances in parallel for different configs.
Uses document chunking: large texts are split into overlapping passages.

Usage:
    # Register + backfill with Telus E5 (default)
    uv run python scripts/backfill_config.py --config telus-e5-1024

    # Register + backfill with Ollama mxbai
    uv run python scripts/backfill_config.py --config ollama-mxbai-1024 --provider ollama --model mxbai-embed-large

    # Register + backfill with Ollama nomic
    uv run python scripts/backfill_config.py --config ollama-nomic-768 --provider ollama --model nomic-embed-text

    # Re-chunk all existing embeddings (after changing chunk params)
    uv run python scripts/backfill_config.py --config ollama-nomic-768 --rechunk --provider ollama --model nomic-embed-text

    # List all configs
    uv run python scripts/backfill_config.py --list

    # Dry run (count chunks)
    uv run python scripts/backfill_config.py --config telus-e5-1024 --dry-run
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
from legion_koi.constants import EMBED_BATCH_SIZE, MAX_RETRIES, RETRY_BACKOFF_SECONDS, CHUNK_CHARS
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
    "legion.claude-pageindex",
    "legion.claude-contact",
    "legion.claude-research",
    "legion.claude-plan",
    "legion.claude-task",
    "legion.claude-backlog",
]

RETRY_BACKOFF = RETRY_BACKOFF_SECONDS

# Junk text filters (base64 blobs, HTML doctype, binary hashes)
JUNK_FILTERS = """
    AND btrim(b.search_text) != ''
    AND b.search_text NOT LIKE %s
    AND b.search_text NOT LIKE %s
    AND b.search_text NOT LIKE %s
    AND left(b.search_text, 100) !~ '[A-Za-z0-9+/=]{60,}'
"""
JUNK_PARAMS = ('--00000000%', '<!doctype%', '<!DOCTYPE%')


def _is_client_error(e: Exception) -> bool:
    return "400" in str(e) or "422" in str(e) or "413" in str(e)


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def upsert_chunks(conn, table, chunk_items):
    """Upsert [(rid, chunk_index, chunk_text, vector), ...] into config table."""
    with conn.transaction():
        with conn.cursor() as cur:
            for rid, chunk_index, chunk_str, vec in chunk_items:
                cur.execute(
                    f"""
                    INSERT INTO {table} (rid, chunk_index, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (rid, chunk_index) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,
                    (rid, chunk_index, chunk_str, vec_literal(vec)),
                )


def delete_stale_chunks(conn, table, rid, new_chunk_count):
    """Remove chunks beyond the new count (document may have shrunk)."""
    conn.execute(
        f"DELETE FROM {table} WHERE rid = %s AND chunk_index >= %s",
        (rid, new_chunk_count),
    )


def embed_and_store_chunks(embedder, conn, table, chunk_tuples, stats, contextual=False):
    """Embed a batch of (rid, chunk_index, chunk_text, preamble) tuples and store them.

    Uses batch embedding with retry and per-item fallback.
    When contextual=True, prepends preamble to each chunk for embedding only.
    """
    if contextual:
        texts = [prepend_preamble(t[3], t[2]) for t in chunk_tuples]
    else:
        texts = [t[2] for t in chunk_tuples]

    for attempt in range(MAX_RETRIES):
        try:
            vectors = embedder.embed_batch(texts, input_type="passage")
            items = [
                (ct[0], ct[1], ct[2], vec)
                for ct, vec in zip(chunk_tuples, vectors)
            ]
            upsert_chunks(conn, table, items)
            stats["chunks"] += len(vectors)
            return
        except Exception as e:
            if _is_client_error(e):
                break
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"    Batch retry {attempt + 1}/{MAX_RETRIES} after {wait}s: {type(e).__name__}")
                time.sleep(wait)

    # Per-item fallback
    for ct in chunk_tuples:
        rid, chunk_index, text = ct[0], ct[1], ct[2]
        embed_text = prepend_preamble(ct[3], text) if contextual else text
        try:
            vec = embedder.embed(embed_text, input_type="passage")
            upsert_chunks(conn, table, [(rid, chunk_index, text, vec)])
            stats["chunks"] += 1
        except Exception as e:
            if _is_client_error(e):
                stats["skipped"] += 1
                continue
            try:
                time.sleep(2)
                vec = embedder.embed(embed_text, input_type="passage")
                upsert_chunks(conn, table, [(rid, chunk_index, text, vec)])
                stats["chunks"] += 1
            except Exception:
                stats["skipped"] += 1


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def fetch_unembedded(conn, table, ns, fetch_size, contextual=False):
    """Fetch bundles that have no embeddings yet."""
    cols = "b.rid, b.search_text, b.namespace, b.contents" if contextual else "b.rid, b.search_text"
    return conn.execute(
        f"""
        SELECT {cols} FROM bundles b
        LEFT JOIN {table} e ON b.rid = e.rid
        WHERE e.rid IS NULL AND b.namespace = %s
          {JUNK_FILTERS}
        ORDER BY b.created_at
        LIMIT %s
        """,
        (ns, *JUNK_PARAMS, fetch_size),
    ).fetchall()


def fetch_all_for_rechunk(conn, ns, fetch_size, offset, contextual=False):
    """Fetch all bundles in a namespace for rechunking."""
    cols = "b.rid, b.search_text, b.namespace, b.contents" if contextual else "b.rid, b.search_text"
    return conn.execute(
        f"""
        SELECT {cols} FROM bundles b
        WHERE b.namespace = %s
          {JUNK_FILTERS}
        ORDER BY b.created_at
        LIMIT %s OFFSET %s
        """,
        (ns, *JUNK_PARAMS, fetch_size, offset),
    ).fetchall()


def count_unembedded(conn, table, ns):
    """Count bundles with no embeddings."""
    row = conn.execute(
        f"""
        SELECT count(*) AS cnt FROM bundles b
        LEFT JOIN {table} e ON b.rid = e.rid
        WHERE e.rid IS NULL AND b.namespace = %s
          {JUNK_FILTERS}
        """,
        (ns, *JUNK_PARAMS),
    ).fetchone()
    return row["cnt"]


def count_all_valid(conn, ns):
    """Count all valid bundles in a namespace."""
    row = conn.execute(
        f"""
        SELECT count(*) AS cnt FROM bundles b
        WHERE b.namespace = %s
          {JUNK_FILTERS}
        """,
        (ns, *JUNK_PARAMS),
    ).fetchone()
    return row["cnt"]


def process_rows(rows, embedder, conn, table, config_id, batch_size, stats, start_time, ns, total_count, contextual=False):
    """Chunk and embed a batch of rows."""
    # Build flat list of (rid, chunk_index, chunk_text, preamble)
    all_chunks = []
    rid_chunk_counts = {}
    for r in rows:
        text = r["search_text"]
        if not text or not text.strip():
            stats["skipped"] += 1
            continue
        chunks = chunk_text(text)
        rid_chunk_counts[r["rid"]] = len(chunks)
        preamble = ""
        if contextual:
            preamble = extract_preamble(r["namespace"], r["contents"])
        for i, c in enumerate(chunks):
            all_chunks.append((r["rid"], i, c, preamble))

    if not all_chunks:
        return

    # Embed in batches
    for batch_start in range(0, len(all_chunks), batch_size):
        batch_slice = all_chunks[batch_start:batch_start + batch_size]
        embed_and_store_chunks(embedder, conn, table, batch_slice, stats, contextual=contextual)

        elapsed = time.time() - start_time
        rate = stats["chunks"] / elapsed if elapsed > 0 else 0
        print(f"  [{ns}] {stats['rids']}/{total_count} RIDs, "
              f"{stats['chunks']} chunks ({rate:.1f} chunks/s)")

        time.sleep(0.05)

    # Delete stale chunks for rechunked RIDs
    for rid, count in rid_chunk_counts.items():
        delete_stale_chunks(conn, table, rid, count)

    stats["rids"] += len(rid_chunk_counts)


def main():
    parser = argparse.ArgumentParser(description="Backfill embeddings for a specific config")
    parser.add_argument("--config", help="Config ID (e.g., telus-e5-1024, ollama-mxbai-1024)")
    parser.add_argument("--provider", choices=["telus", "ollama"], help="Embedding provider")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_CHARS, help=f"Max chars per chunk (default: {CHUNK_CHARS})")
    parser.add_argument("--description", default="", help="Config description")
    parser.add_argument("--default", action="store_true", help="Set as default config")
    parser.add_argument("--namespace", help="Only backfill this namespace")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE, help="Chunks per API call")
    parser.add_argument("--fetch-size", type=int, default=100, help="DB fetch batch size (RIDs)")
    parser.add_argument("--dry-run", action="store_true", help="Count chunks without embedding")
    parser.add_argument("--rechunk", action="store_true",
                        help="Re-embed ALL RIDs (not just unembedded) — use after changing chunk params")
    parser.add_argument("--contextual", action="store_true",
                        help="Prepend document metadata preamble to chunks before embedding (creates '-ctx' configs)")
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
            try:
                conn = get_conn()
                row = conn.execute(f"SELECT count(*) AS cnt FROM {table}").fetchone()
                rid_row = conn.execute(f"SELECT count(DISTINCT rid) AS cnt FROM {table}").fetchone()
                print(f"    Table: {table}, Chunks: {row['cnt']}, RIDs: {rid_row['cnt']}")
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
    print(f"Embedder: {provider}/{model} ({dims} dims, chunk_chars={args.chunk_size})")
    ctx_label = " + contextual" if args.contextual else ""
    print(f"Mode: {'rechunk ALL' if args.rechunk else 'unembedded only'}{ctx_label}")

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

    total_stats = {"rids": 0, "chunks": 0, "skipped": 0}
    start_time = time.time()

    for ns in namespaces:
        if args.rechunk:
            total_count = count_all_valid(conn, ns)
        else:
            total_count = count_unembedded(conn, table, ns)

        if total_count == 0:
            print(f"[{ns}] {'All embedded' if not args.rechunk else 'No valid bundles'}, skipping")
            continue

        # Dry run: count estimated chunks
        if args.dry_run:
            sample = conn.execute(
                f"""
                SELECT b.search_text FROM bundles b
                WHERE b.namespace = %s
                  {JUNK_FILTERS}
                LIMIT 500
                """,
                (ns, *JUNK_PARAMS),
            ).fetchall()
            total_chunks = sum(len(chunk_text(r["search_text"])) for r in sample if r["search_text"])
            avg_chunks = total_chunks / len(sample) if sample else 1
            est_total = int(total_count * avg_chunks)
            print(f"[{ns}] {total_count} RIDs, ~{avg_chunks:.1f} chunks/RID, ~{est_total} total chunks")
            continue

        print(f"[{ns}] {total_count} {'RIDs to rechunk' if args.rechunk else 'unembedded RIDs'}")

        ns_stats = {"rids": 0, "chunks": 0, "skipped": 0}
        offset = 0
        seen_rids = set()

        while True:
            try:
                if args.rechunk:
                    rows = fetch_all_for_rechunk(conn, ns, args.fetch_size, offset, contextual=args.contextual)
                    offset += len(rows)
                else:
                    rows = fetch_unembedded(conn, table, ns, args.fetch_size, contextual=args.contextual)
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

            # Filter out previously-seen RIDs that failed to embed (prevents infinite loop)
            if not args.rechunk:
                new_rows = [r for r in rows if r["rid"] not in seen_rids]
                if not new_rows:
                    stuck_rids = [r["rid"] for r in rows]
                    print(f"  [{ns}] Skipping {len(stuck_rids)} stuck RIDs: {stuck_rids[:5]}")
                    break
                seen_rids.update(r["rid"] for r in new_rows)
                rows = new_rows

            process_rows(rows, embedder, conn, table, args.config,
                         args.batch_size, ns_stats, start_time, ns, total_count,
                         contextual=args.contextual)

        total_stats["rids"] += ns_stats["rids"]
        total_stats["chunks"] += ns_stats["chunks"]
        total_stats["skipped"] += ns_stats["skipped"]
        print(f"  [{ns}] Done: {ns_stats['rids']} RIDs, {ns_stats['chunks']} chunks, "
              f"{ns_stats['skipped']} skipped")

    elapsed = time.time() - start_time
    print(f"\nComplete [{args.config}]: {total_stats['rids']} RIDs, {total_stats['chunks']} chunks, "
          f"{total_stats['skipped']} skipped in {elapsed:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
