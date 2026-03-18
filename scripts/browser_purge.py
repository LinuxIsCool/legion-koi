"""Purge browser history bundles matching a domain pattern.

Reads the KOI PostgreSQL DSN from config.yaml, scans all
legion.claude-browser-history bundles, and deletes those whose domain
matches the given fnmatch pattern. Cascades to embeddings and bundle_entities.

Optionally appends the pattern to the suppression list to prevent re-ingestion.

Usage:
    cd ~/legion-koi && uv run python scripts/browser_purge.py --domain "*.td.com" --dry-run
    cd ~/legion-koi && uv run python scripts/browser_purge.py --domain "*.td.com" --yes --add-to-suppression
"""

import fnmatch
import sys
from pathlib import Path

import psycopg
import yaml

NAMESPACE = "legion.claude-browser-history"
SUPPRESSION_FILE = Path("~/.config/claude-browser-history/suppressed_domains.txt").expanduser()
BATCH_SIZE = 5000


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    dry_run = "--dry-run" in sys.argv
    auto_confirm = "--yes" in sys.argv
    add_to_suppression = "--add-to-suppression" in sys.argv

    # Parse --domain argument
    domain_pattern = None
    for i, arg in enumerate(sys.argv):
        if arg == "--domain" and i + 1 < len(sys.argv):
            domain_pattern = sys.argv[i + 1].lower()
            break
    if not domain_pattern:
        print("Usage: browser_purge.py --domain PATTERN [--dry-run] [--yes] [--add-to-suppression]")
        print("  PATTERN: fnmatch glob, e.g. '*.td.com', 'accounts.google.com'")
        sys.exit(1)

    config = load_config()
    postgres = config.get("postgres", {})
    dsn = postgres.get("dsn", "postgresql://localhost/personal_koi")

    conn = psycopg.connect(dsn, autocommit=False)

    # Count before
    before_count = conn.execute(
        f"SELECT COUNT(*) FROM bundles WHERE namespace = %s", (NAMESPACE,)
    ).fetchone()[0]
    print(f"Browser history bundles: {before_count:,}")

    # Scan all bundles for domain match
    print(f"Scanning for domains matching: {domain_pattern}")
    cursor = conn.execute("""
        SELECT rid, contents->>'domain' AS domain
        FROM bundles
        WHERE namespace = %s
    """, (NAMESPACE,))

    to_delete = []
    to_keep = 0
    for row in cursor:
        rid, domain = row
        domain = (domain or "").lower()
        if fnmatch.fnmatch(domain, domain_pattern):
            to_delete.append(rid)
        else:
            to_keep += 1

    print(f"Matching:  {len(to_delete):,}")
    print(f"Keeping:   {to_keep:,}")

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

    # Delete in batches
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
        f"SELECT COUNT(*) FROM bundles WHERE namespace = %s", (NAMESPACE,)
    ).fetchone()[0]
    print(f"\nBrowser history bundles after: {after_count:,}")
    print(f"Purged: {before_count - after_count:,}")

    conn.close()

    # Vacuum — requires autocommit
    print("Running VACUUM ANALYZE...")
    vacuum_conn = psycopg.connect(dsn, autocommit=True)
    vacuum_conn.execute("VACUUM ANALYZE bundles")
    vacuum_conn.execute("VACUUM ANALYZE embeddings")
    vacuum_conn.execute("VACUUM ANALYZE bundle_entities")
    vacuum_conn.close()
    print("Done.")

    # Optionally add pattern to suppression list
    if add_to_suppression:
        if SUPPRESSION_FILE.exists():
            existing_lines = {
                line.strip().lower()
                for line in SUPPRESSION_FILE.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
            if domain_pattern not in existing_lines:
                with open(SUPPRESSION_FILE, "a") as f:
                    f.write(f"\n# Added by browser_purge.py\n{domain_pattern}\n")
                print(f"Added '{domain_pattern}' to {SUPPRESSION_FILE}")
            else:
                print(f"Pattern '{domain_pattern}' already in {SUPPRESSION_FILE}")
        else:
            print(f"Warning: {SUPPRESSION_FILE} not found, skipping --add-to-suppression")


if __name__ == "__main__":
    main()
