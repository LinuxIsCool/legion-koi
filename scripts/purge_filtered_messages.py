"""Purge message bundles that no longer pass the relevance filter.

Reads the MessageFilter config from config.yaml, checks every
legion.claude-message bundle in PostgreSQL, and deletes those that
would be filtered out. Cascades to embeddings and bundle_entities.

Usage:
    cd ~/legion-koi && uv run python scripts/purge_filtered_messages.py [--dry-run] [--yes]
"""

import sys
from pathlib import Path

import psycopg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legion_koi.sensors.message_filter import MessageFilter


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    dry_run = "--dry-run" in sys.argv
    auto_confirm = "--yes" in sys.argv

    config = load_config()
    sensors = config.get("sensors", {})
    postgres = config.get("postgres", {})
    dsn = postgres.get("dsn", "postgresql://localhost/personal_koi")

    # Build filter from config
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

    # Count before
    before_count = conn.execute(
        "SELECT COUNT(*) FROM bundles WHERE namespace = 'legion.claude-message'"
    ).fetchone()[0]
    print(f"Message bundles before: {before_count:,}")

    # Fetch all message bundle thread_ids and platforms
    print("Scanning bundles for filter classification...")
    cursor = conn.execute("""
        SELECT rid,
               contents->>'thread_id' AS thread_id,
               contents->>'platform' AS platform
        FROM bundles
        WHERE namespace = 'legion.claude-message'
    """)

    to_delete = []
    to_keep = 0
    for row in cursor:
        rid, thread_id, platform = row
        if not msg_filter.should_ingest(thread_id or "", platform or "telegram"):
            to_delete.append(rid)
        else:
            to_keep += 1

    print(f"To delete: {len(to_delete):,}")
    print(f"To keep:   {to_keep:,}")

    if not to_delete:
        print("Nothing to purge.")
        conn.close()
        return

    if dry_run:
        print(f"[DRY RUN] Would delete {len(to_delete):,} bundles. No changes made.")
        conn.close()
        return

    # Confirm
    if not auto_confirm:
        answer = input(f"\nDelete {len(to_delete):,} bundles? This cascades to embeddings and entities. [y/N] ")
        if answer.lower() != "y":
            print("Aborted.")
            conn.close()
            return

    # Delete in batches to avoid huge transactions
    BATCH_SIZE = 5000
    deleted = 0
    for i in range(0, len(to_delete), BATCH_SIZE):
        batch = to_delete[i:i + BATCH_SIZE]
        placeholders = ",".join(["%s"] * len(batch))

        # Delete embeddings first (foreign key)
        conn.execute(
            f"DELETE FROM embeddings WHERE rid IN ({placeholders})", batch
        )
        # Delete entities
        conn.execute(
            f"DELETE FROM bundle_entities WHERE rid IN ({placeholders})", batch
        )
        # Delete bundles
        conn.execute(
            f"DELETE FROM bundles WHERE rid IN ({placeholders})", batch
        )
        conn.commit()
        deleted += len(batch)
        print(f"  Deleted {deleted:,} / {len(to_delete):,}")

    # Count after
    after_count = conn.execute(
        "SELECT COUNT(*) FROM bundles WHERE namespace = 'legion.claude-message'"
    ).fetchone()[0]
    print(f"\nMessage bundles after: {after_count:,}")
    print(f"Purged: {before_count - after_count:,}")

    conn.close()

    # Vacuum — requires autocommit, so use a fresh connection
    print("Running VACUUM ANALYZE...")
    vacuum_conn = psycopg.connect(dsn, autocommit=True)
    vacuum_conn.execute("VACUUM ANALYZE bundles")
    vacuum_conn.execute("VACUUM ANALYZE embeddings")
    vacuum_conn.execute("VACUUM ANALYZE bundle_entities")
    vacuum_conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
