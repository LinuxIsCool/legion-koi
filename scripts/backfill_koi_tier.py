"""Backfill _koi_tier metadata on existing message bundles.

Reads thread_id and platform from each bundle's contents JSON,
classifies via MessageFilter, and updates the _koi_tier field.
Only touches bundles where _koi_tier is NULL.

Usage:
    cd ~/legion-koi && uv run python scripts/backfill_koi_tier.py [--dry-run]
"""

import sys
from pathlib import Path

import psycopg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legion_koi.sensors.message_filter import MessageFilter

BATCH_SIZE = 5000


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    dry_run = "--dry-run" in sys.argv

    config = load_config()
    sensors = config.get("sensors", {})
    postgres = config.get("postgres", {})
    dsn = postgres.get("dsn", "postgresql://localhost/personal_koi")

    msg_filter = MessageFilter(
        messages_db_path=Path(sensors.get(
            "message_db_path", "~/.claude/local/messages/messages.db"
        )).expanduser(),
        self_sender_ids=sensors.get("message_self_sender_ids", ["telegram:user:1441369482"]),
        thread_includes=sensors.get("message_thread_includes", []),
        thread_excludes=sensors.get("message_thread_excludes", []),
        enable=True,
    )

    conn = psycopg.connect(dsn, autocommit=False)

    # Count bundles needing backfill
    null_count = conn.execute("""
        SELECT COUNT(*) FROM bundles
        WHERE namespace = 'legion.claude-message'
          AND (contents->>'_koi_tier') IS NULL
    """).fetchone()[0]

    total_count = conn.execute("""
        SELECT COUNT(*) FROM bundles
        WHERE namespace = 'legion.claude-message'
    """).fetchone()[0]

    print(f"Message bundles: {total_count:,} total, {null_count:,} missing _koi_tier")

    if null_count == 0:
        print("Nothing to backfill.")
        conn.close()
        return

    # Fetch all bundles needing backfill
    print("Classifying bundles...")
    cursor = conn.execute("""
        SELECT rid,
               contents->>'thread_id' AS thread_id,
               contents->>'platform' AS platform
        FROM bundles
        WHERE namespace = 'legion.claude-message'
          AND (contents->>'_koi_tier') IS NULL
    """)

    # Build update list grouped by tier for batch updates
    tier_updates: dict[str, list[str]] = {}
    for row in cursor:
        rid, thread_id, platform = row
        tier = msg_filter.classify(thread_id or "", platform or "telegram")
        tier_updates.setdefault(tier.value, []).append(rid)

    # Report
    print("\nTier distribution:")
    for tier_value, rids in sorted(tier_updates.items(), key=lambda x: -len(x[1])):
        print(f"  {tier_value:<15} {len(rids):>10,}")
    print(f"  {'TOTAL':<15} {sum(len(r) for r in tier_updates.values()):>10,}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        conn.close()
        return

    # Apply updates in batches per tier
    updated = 0
    for tier_value, rids in tier_updates.items():
        for i in range(0, len(rids), BATCH_SIZE):
            batch = rids[i:i + BATCH_SIZE]
            placeholders = ",".join(["%s"] * len(batch))
            conn.execute(f"""
                UPDATE bundles
                SET contents = jsonb_set(contents::jsonb, '{{_koi_tier}}', %s::jsonb)
                WHERE rid IN ({placeholders})
            """, [f'"{tier_value}"'] + batch)
            conn.commit()
            updated += len(batch)
            print(f"  Updated {updated:,} / {null_count:,}")

    print(f"\nBackfill complete: {updated:,} bundles updated.")
    conn.close()


if __name__ == "__main__":
    main()
