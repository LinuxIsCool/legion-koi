#!/usr/bin/env -S python3 -u
"""Backfill entity extraction across all or specific namespaces.

Usage:
    uv run python scripts/backfill_entities.py                              # all namespaces
    uv run python scripts/backfill_entities.py --namespace legion.claude-journal
    uv run python scripts/backfill_entities.py --dry-run                    # count only
    uv run python scripts/backfill_entities.py --batch-size 25              # smaller batches
    uv run python scripts/backfill_entities.py --stats                      # entity stats
    uv run python scripts/backfill_entities.py --config fast                # regex-only baseline
"""

import argparse
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.constants import ENTITY_BACKFILL_BATCH_SIZE, ENTITY_EXTRACTION_SKIP_NAMESPACES
from legion_koi.extraction import normalize_entity_name
from legion_koi.storage.postgres import PostgresStorage

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")

NAMESPACES_ORDER = [
    "legion.claude-journal",
    "legion.claude-venture",
    "legion.claude-web.conversation",
    "legion.claude-recording",
    "legion.claude-github",
    "legion.claude-youtube",
    "legion.claude-code",
    "legion.claude-message",
]

# Skip junk text
JUNK_FILTERS = """
    AND btrim(b.search_text) != ''
    AND b.search_text NOT LIKE %s
    AND b.search_text NOT LIKE %s
    AND b.search_text NOT LIKE %s
    AND left(b.search_text, 100) !~ '[A-Za-z0-9+/=]{60,}'
"""
JUNK_PARAMS = ('--00000000%', '<!doctype%', '<!DOCTYPE%')


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def count_unextracted(conn, ns):
    row = conn.execute(
        f"""
        SELECT count(*) AS cnt FROM bundles b
        LEFT JOIN bundle_entities be ON b.rid = be.rid
        WHERE be.rid IS NULL AND b.namespace = %s
          {JUNK_FILTERS}
        """,
        (ns, *JUNK_PARAMS),
    ).fetchone()
    return row["cnt"]


def fetch_unextracted(conn, ns, limit):
    return conn.execute(
        f"""
        SELECT b.rid, b.namespace, b.search_text FROM bundles b
        LEFT JOIN bundle_entities be ON b.rid = be.rid
        WHERE be.rid IS NULL AND b.namespace = %s
          {JUNK_FILTERS}
        ORDER BY b.created_at
        LIMIT %s
        """,
        (ns, *JUNK_PARAMS, limit),
    ).fetchall()


def process_batch(rows, pipeline, storage, stats, start_time, ns, total_count):
    """Extract entities from a batch of bundles."""
    for r in rows:
        rid = r["rid"]
        text = r["search_text"]
        if not text or not text.strip():
            stats["skipped"] += 1
            continue

        try:
            result = pipeline.run(rid, r["namespace"], text)
            if result.entities:
                entity_dicts = [
                    {
                        "name": e.name,
                        "entity_type": e.entity_type,
                        "supertype": e.supertype,
                        "confidence": e.confidence,
                        "name_normalized": normalize_entity_name(e.name),
                    }
                    for e in result.entities
                ]
                storage.upsert_bundle_entities(rid, entity_dicts)
                stats["entities"] += len(entity_dicts)
            stats["rids"] += 1

            elapsed = time.time() - start_time
            rate = stats["rids"] / elapsed if elapsed > 0 else 0
            print(
                f"  [{ns}] {stats['rids']}/{total_count} RIDs, "
                f"{stats['entities']} entities ({rate:.1f} RIDs/s) "
                f"last: {len(result.entities)} in {result.extraction_time_seconds:.2f}s"
            )
        except Exception as e:
            stats["errors"] += 1
            print(f"  [{ns}] ERROR on {rid}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Backfill entity extraction")
    parser.add_argument("--namespace", help="Only backfill this namespace")
    parser.add_argument("--batch-size", type=int, default=ENTITY_BACKFILL_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Count only")
    parser.add_argument("--stats", action="store_true", help="Show entity stats and exit")
    parser.add_argument("--config", default="default", help="Pipeline config name (default, fast)")
    args = parser.parse_args()

    storage = PostgresStorage(dsn=DSN)
    storage.initialize()

    if args.stats:
        try:
            entity_stats = storage.get_entity_stats()
            import json
            print(json.dumps(entity_stats, indent=2, default=str))
        except Exception as e:
            print(f"Error getting stats: {e}")
        storage.close()
        return

    # Initialize pipeline
    from legion_koi.extraction.pipeline import ExtractionPipeline
    pipeline = ExtractionPipeline(config_name=args.config)
    print(f"Pipeline: {args.config}")
    print(f"Backends: {[e.get_name() for e in pipeline._extractors]}")

    conn = get_conn()
    namespaces = [args.namespace] if args.namespace else NAMESPACES_ORDER
    namespaces = [ns for ns in namespaces if ns not in ENTITY_EXTRACTION_SKIP_NAMESPACES]

    total_stats = {"rids": 0, "entities": 0, "skipped": 0, "errors": 0}
    start_time = time.time()

    for ns in namespaces:
        total_count = count_unextracted(conn, ns)
        if total_count == 0:
            print(f"[{ns}] All extracted, skipping")
            continue

        if args.dry_run:
            print(f"[{ns}] {total_count} unextracted RIDs")
            continue

        print(f"[{ns}] {total_count} unextracted RIDs")
        ns_stats = {"rids": 0, "entities": 0, "skipped": 0, "errors": 0}

        while True:
            try:
                rows = fetch_unextracted(conn, ns, args.batch_size)
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

            process_batch(rows, pipeline, storage, ns_stats, start_time, ns, total_count)

        for k in total_stats:
            total_stats[k] += ns_stats[k]
        print(f"  [{ns}] Done: {ns_stats['rids']} RIDs, {ns_stats['entities']} entities, "
              f"{ns_stats['errors']} errors")

    elapsed = time.time() - start_time
    print(f"\nComplete: {total_stats['rids']} RIDs, {total_stats['entities']} entities, "
          f"{total_stats['errors']} errors in {elapsed:.1f}s")
    conn.close()
    storage.close()


if __name__ == "__main__":
    main()
