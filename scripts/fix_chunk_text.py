"""Batch-restore full chunk_text in embeddings table.

The original backfill stored chunk[:200] (truncated). This script re-chunks
each bundle's search_text and batch-UPDATEs the correct full chunk for each
(rid, chunk_index) pair.

Uses temp table + UPDATE FROM for fast batch updates.
"""

import sys

import psycopg

# Add src to path for chunking import
sys.path.insert(0, "src")
from legion_koi.chunking import chunk_text

DSN = "postgresql://localhost/personal_koi"
TABLE = "embeddings_ollama_nomic_768"
BATCH_SIZE = 500  # RIDs per transaction


def main():
    conn = psycopg.connect(DSN, autocommit=False)

    # Get all RIDs that need fixing (truncated chunk_text)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT DISTINCT e.rid
            FROM {TABLE} e
            WHERE length(e.chunk_text) <= 200
        """)
        rids_to_fix = [r[0] for r in cur.fetchall()]

    print(f"RIDs to fix: {len(rids_to_fix)}")
    total_updated = 0

    for batch_start in range(0, len(rids_to_fix), BATCH_SIZE):
        batch_rids = rids_to_fix[batch_start : batch_start + BATCH_SIZE]

        with conn.cursor() as cur:
            # Fetch search_text for this batch of RIDs
            cur.execute(
                "SELECT rid, search_text FROM bundles WHERE rid = ANY(%s)",
                (batch_rids,),
            )
            bundle_texts = {r[0]: r[1] for r in cur.fetchall()}

            # Build update tuples: (rid, chunk_index, full_chunk_text)
            updates = []
            for rid in batch_rids:
                search_text = bundle_texts.get(rid)
                if not search_text:
                    continue
                chunks = chunk_text(search_text)
                if not chunks:
                    continue
                for i, chunk in enumerate(chunks):
                    updates.append((rid, i, chunk))

            if updates:
                # Use a temp table + UPDATE FROM for speed
                cur.execute("""
                    CREATE TEMP TABLE _chunk_fix (
                        rid TEXT, chunk_index INTEGER, chunk_text TEXT
                    ) ON COMMIT DROP
                """)
                # Batch insert with executemany
                cur.executemany(
                    "INSERT INTO _chunk_fix (rid, chunk_index, chunk_text) VALUES (%s, %s, %s)",
                    updates,
                )
                cur.execute(f"""
                    UPDATE {TABLE} e
                    SET chunk_text = f.chunk_text
                    FROM _chunk_fix f
                    WHERE e.rid = f.rid AND e.chunk_index = f.chunk_index
                """)
                total_updated += cur.rowcount

            conn.commit()

        done = min(batch_start + BATCH_SIZE, len(rids_to_fix))
        print(f"  {done}/{len(rids_to_fix)} RIDs processed, {total_updated} rows updated")

    conn.close()
    print(f"Done. Total rows updated: {total_updated}")


if __name__ == "__main__":
    main()
