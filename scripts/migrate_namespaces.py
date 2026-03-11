#!/usr/bin/env -S python3 -u
"""Namespace consolidation migration.

Renames old namespaces to new consolidated scheme:
  legion.claude-conversation → legion.claude-web.conversation
  legion.claude-project      → legion.claude-web.project
  legion.claude-memory       → legion.claude-web.memory
  legion.claude-session      → legion.claude-logging
  legion.claude-transcript   → legion.claude-code

Also updates RID prefixes to match new namespaces.
Must update embedding tables too since they FK to bundles.rid.

Usage:
    uv run python scripts/migrate_namespaces.py --dry-run
    uv run python scripts/migrate_namespaces.py
"""

import argparse
import sys

import psycopg
from psycopg.rows import dict_row

DSN = "postgresql://shawn@localhost/personal_koi"

# (old_namespace, new_namespace, old_orn_prefix, new_orn_prefix)
RENAMES = [
    ("legion.claude-conversation", "legion.claude-web.conversation",
     "orn:legion.claude-conversation:", "orn:legion.claude-web.conversation:"),
    ("legion.claude-project", "legion.claude-web.project",
     "orn:legion.claude-project:", "orn:legion.claude-web.project:"),
    ("legion.claude-memory", "legion.claude-web.memory",
     "orn:legion.claude-memory:", "orn:legion.claude-web.memory:"),
    ("legion.claude-session", "legion.claude-logging",
     "orn:legion.claude-session:", "orn:legion.claude-logging:"),
    ("legion.claude-transcript", "legion.claude-code",
     "orn:legion.claude-transcript:", "orn:legion.claude-code:"),
]

# Embedding tables that FK to bundles.rid
EMBEDDING_TABLES = [
    ("embeddings", "embeddings_rid_fkey"),
    ("embeddings_ollama_mxbai_1024", "embeddings_ollama_mxbai_1024_rid_fkey"),
    ("embeddings_ollama_nomic_768", "embeddings_ollama_nomic_768_rid_fkey"),
    ("embeddings_telus_e5_1024", "embeddings_telus_e5_1024_rid_fkey"),
]


def main():
    parser = argparse.ArgumentParser(description="Migrate namespaces")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = psycopg.connect(DSN, row_factory=dict_row, autocommit=True)

    # Show before state
    print("=== Before ===")
    rows = conn.execute(
        "SELECT namespace, count(*) AS cnt FROM bundles GROUP BY namespace ORDER BY namespace"
    ).fetchall()
    for r in rows:
        print(f"  {r['namespace']}: {r['cnt']}")

    if args.dry_run:
        print("\n=== Dry Run — changes that would be made ===")
        for old_ns, new_ns, old_prefix, new_prefix in RENAMES:
            row = conn.execute(
                "SELECT count(*) AS cnt FROM bundles WHERE namespace = %s", (old_ns,)
            ).fetchone()
            if row["cnt"] > 0:
                print(f"  {old_ns} → {new_ns}: {row['cnt']} bundles")
                # Check how many RIDs need updating
                rid_row = conn.execute(
                    "SELECT count(*) AS cnt FROM bundles WHERE rid LIKE %s",
                    (old_prefix + "%",)
                ).fetchone()
                print(f"    RIDs to update: {rid_row['cnt']}")
                # Check embedding tables
                for table, _ in EMBEDDING_TABLES:
                    try:
                        emb_row = conn.execute(
                            f"SELECT count(*) AS cnt FROM {table} WHERE rid LIKE %s",
                            (old_prefix + "%",)
                        ).fetchone()
                        if emb_row["cnt"] > 0:
                            print(f"    {table}: {emb_row['cnt']} embeddings to update")
                    except Exception:
                        pass
        conn.close()
        return

    print("\n=== Migrating ===")
    try:
        with conn.transaction():
            # Step 1: Drop FK constraints
            print("Dropping FK constraints...")
            for table, constraint in EMBEDDING_TABLES:
                try:
                    conn.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint}")
                    print(f"  Dropped {constraint}")
                except Exception as e:
                    print(f"  Skip {constraint}: {e}")

            # Step 2: Update namespaces and RIDs
            for old_ns, new_ns, old_prefix, new_prefix in RENAMES:
                # Update namespace
                result = conn.execute(
                    "UPDATE bundles SET namespace = %s WHERE namespace = %s",
                    (new_ns, old_ns),
                )
                ns_count = result.rowcount
                print(f"  Namespace {old_ns} → {new_ns}: {ns_count} rows")

                # Update RIDs in bundles
                result = conn.execute(
                    "UPDATE bundles SET rid = replace(rid, %s, %s) WHERE rid LIKE %s",
                    (old_prefix, new_prefix, old_prefix + "%"),
                )
                rid_count = result.rowcount
                print(f"  RIDs in bundles: {rid_count}")

                # Update RIDs in embedding tables
                for table, _ in EMBEDDING_TABLES:
                    try:
                        result = conn.execute(
                            f"UPDATE {table} SET rid = replace(rid, %s, %s) WHERE rid LIKE %s",
                            (old_prefix, new_prefix, old_prefix + "%"),
                        )
                        if result.rowcount > 0:
                            print(f"  RIDs in {table}: {result.rowcount}")
                    except Exception:
                        pass

            # Step 3: Re-add FK constraints
            print("Re-adding FK constraints...")
            for table, constraint in EMBEDDING_TABLES:
                try:
                    conn.execute(
                        f"ALTER TABLE {table} ADD CONSTRAINT {constraint} "
                        f"FOREIGN KEY (rid) REFERENCES bundles(rid) ON DELETE CASCADE"
                    )
                    print(f"  Added {constraint}")
                except Exception as e:
                    print(f"  Error adding {constraint}: {e}")
                    raise

        print("\n=== Migration committed ===")

    except Exception as e:
        print(f"\nMigration FAILED (rolled back): {e}")
        conn.close()
        sys.exit(1)

    # Show after state
    print("\n=== After ===")
    rows = conn.execute(
        "SELECT namespace, count(*) AS cnt FROM bundles GROUP BY namespace ORDER BY namespace"
    ).fetchall()
    for r in rows:
        print(f"  {r['namespace']}: {r['cnt']}")

    # Verify no orphaned embeddings
    for table, _ in EMBEDDING_TABLES:
        try:
            row = conn.execute(
                f"SELECT count(*) AS cnt FROM {table} e LEFT JOIN bundles b ON e.rid = b.rid WHERE b.rid IS NULL"
            ).fetchone()
            if row["cnt"] > 0:
                print(f"  WARNING: {row['cnt']} orphaned embeddings in {table}")
            else:
                print(f"  {table}: no orphans")
        except Exception:
            pass

    conn.close()


if __name__ == "__main__":
    main()
