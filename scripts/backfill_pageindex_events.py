#!/usr/bin/env -S python3 -u
"""Backfill BUNDLE_CREATED events for pageindex bundles.

PageIndex bundles were inserted via direct SQL (KOI sync), bypassing the
PostgreSQL NOTIFY trigger. This script republishes BUNDLE_CREATED events
to the Redis stream so embed and extract consumers process them.

Usage:
    uv run python scripts/backfill_pageindex_events.py                # publish all
    uv run python scripts/backfill_pageindex_events.py --dry-run      # count only
    uv run python scripts/backfill_pageindex_events.py --batch-size 50
"""

import argparse
import os
import sys
import time

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.events.bus import EventBus
from legion_koi.events.schemas import KoiEvent, BUNDLE_CREATED

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")
NAMESPACE = "legion.claude-pageindex"
DEFAULT_BATCH_SIZE = 100


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def fetch_pageindex_bundles(conn, limit, offset):
    """Fetch pageindex bundles in batches."""
    return conn.execute(
        """
        SELECT rid, namespace, reference, created_at
        FROM bundles
        WHERE namespace = %s
        ORDER BY created_at
        LIMIT %s OFFSET %s
        """,
        (NAMESPACE, limit, offset),
    ).fetchall()


def count_pageindex_bundles(conn):
    row = conn.execute(
        "SELECT count(*) AS cnt FROM bundles WHERE namespace = %s",
        (NAMESPACE,),
    ).fetchone()
    return row["cnt"]


def main():
    parser = argparse.ArgumentParser(
        description="Backfill BUNDLE_CREATED events for pageindex bundles"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Bundles per batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument("--dry-run", action="store_true", help="Count only")
    args = parser.parse_args()

    conn = get_conn()
    total = count_pageindex_bundles(conn)
    print(f"PageIndex bundles: {total}")

    if args.dry_run or total == 0:
        conn.close()
        return

    bus = EventBus()
    if not bus.ping():
        print("ERROR: Redis (event bus) not reachable on port 6380")
        sys.exit(1)
    print("Event bus: connected")

    published = 0
    start_time = time.time()
    offset = 0

    while offset < total:
        rows = fetch_pageindex_bundles(conn, args.batch_size, offset)
        if not rows:
            break

        for row in rows:
            event = KoiEvent(
                type=BUNDLE_CREATED,
                subject=row["rid"],
                data={
                    "namespace": row["namespace"],
                    "reference": row.get("reference", ""),
                },
            )
            bus.publish(event)
            published += 1

        offset += len(rows)
        elapsed = time.time() - start_time
        rate = published / elapsed if elapsed > 0 else 0
        print(f"  Published {published}/{total} events ({rate:.0f}/s)")

        # Gentle pacing — don't flood the stream
        time.sleep(0.1)

    elapsed = time.time() - start_time
    print(f"\nComplete: {published} BUNDLE_CREATED events in {elapsed:.1f}s")
    print(f"Consumers will process embeddings + entity extraction asynchronously.")

    bus.close()
    conn.close()


if __name__ == "__main__":
    main()
