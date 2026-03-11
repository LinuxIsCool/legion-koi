#!/usr/bin/env -S uv run python
"""Build the hippo-KOI bridge: add koi_rids properties to hippo entities.

For each entity in FalkorDB, collects all RELATES edge sources,
converts each to a KOI RID, and stores as a JSON koi_rids property
on the entity node. Also verifies RIDs exist in KOI's PostgreSQL.

Usage:
    uv run python scripts/build_hippo_bridge.py
    uv run python scripts/build_hippo_bridge.py --dry-run
    uv run python scripts/build_hippo_bridge.py --verify
"""

import argparse
import json
import os
import sys

import redis

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from legion_koi.hippo_bridge import hippo_source_to_koi_rid, NAMESPACE_MAP
from legion_koi.constants import HIPPO_REDIS_HOST, HIPPO_REDIS_PORT, HIPPO_GRAPH_NAME


def parse_graph_result(result: list) -> list[dict]:
    if not result or len(result) < 2:
        return []
    header = result[0]
    rows = result[1] if len(result) > 1 else []
    if not header or not rows:
        return []
    return [dict(zip(header, row)) for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Build hippo-KOI bridge")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't modify graph")
    parser.add_argument("--verify", action="store_true", help="Verify RIDs exist in KOI PostgreSQL")
    args = parser.parse_args()

    r = redis.Redis(host=HIPPO_REDIS_HOST, port=HIPPO_REDIS_PORT, decode_responses=True)

    # Test connection
    try:
        r.ping()
    except redis.ConnectionError:
        print("Error: Cannot connect to FalkorDB. Is hippo-graph running?", file=sys.stderr)
        sys.exit(1)

    # 1. Get all entities
    result = r.execute_command("GRAPH.QUERY", HIPPO_GRAPH_NAME,
                               "MATCH (n:Entity) RETURN n.name")
    entities = [row["n.name"] for row in parse_graph_result(result)]
    print(f"Found {len(entities)} entities in hippo")

    # 2. For each entity, collect sources from all edges
    total_rids = 0
    entities_with_rids = 0
    source_prefix_counts: dict[str, int] = {}
    missing_rids: list[str] = []

    # Optionally connect to KOI for verification
    storage = None
    if args.verify:
        from legion_koi.storage.postgres import PostgresStorage
        dsn = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")
        storage = PostgresStorage(dsn=dsn)
        print("Connected to KOI PostgreSQL for verification")

    for i, entity in enumerate(entities):
        esc = entity.replace('"', '\\"')

        # Get all sources from edges touching this entity
        out_result = r.execute_command(
            "GRAPH.QUERY", HIPPO_GRAPH_NAME,
            f'MATCH (s:Entity {{name: "{esc}"}})-[r:RELATES]->() RETURN DISTINCT r.source AS source'
        )
        in_result = r.execute_command(
            "GRAPH.QUERY", HIPPO_GRAPH_NAME,
            f'MATCH ()-[r:RELATES]->(o:Entity {{name: "{esc}"}}) RETURN DISTINCT r.source AS source'
        )

        sources = set()
        for row in parse_graph_result(out_result) + parse_graph_result(in_result):
            src = row.get("source")
            if src:
                sources.add(src)

        # Convert to KOI RIDs
        rids = []
        for src in sources:
            prefix = src.partition(":")[0]
            source_prefix_counts[prefix] = source_prefix_counts.get(prefix, 0) + 1
            rid = hippo_source_to_koi_rid(src)
            if rid:
                rids.append(rid)

        if rids:
            entities_with_rids += 1
            total_rids += len(rids)

            # Verify RIDs exist in KOI
            if storage:
                for rid in rids:
                    bundle = storage.get_bundle(rid)
                    if not bundle:
                        missing_rids.append(rid)

            if not args.dry_run:
                # Store koi_rids as JSON property on entity
                rids_json = json.dumps(rids).replace('"', '\\"')
                try:
                    r.execute_command(
                        "GRAPH.QUERY", HIPPO_GRAPH_NAME,
                        f'MATCH (n:Entity {{name: "{esc}"}}) SET n.koi_rids = "{rids_json}"'
                    )
                except redis.ResponseError as e:
                    print(f"  Warning: Failed to set koi_rids on '{entity}': {e}", file=sys.stderr)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(entities)} entities...")

    # Summary
    print(f"\nResults:")
    print(f"  Entities total:     {len(entities)}")
    print(f"  Entities with RIDs: {entities_with_rids}")
    print(f"  Total RID links:    {total_rids}")
    print(f"  Source prefixes:    {json.dumps(source_prefix_counts, indent=2)}")

    if args.dry_run:
        print("\n  (dry run — no changes made)")
    else:
        print(f"\n  Stored koi_rids on {entities_with_rids} entities")

    if missing_rids:
        print(f"\n  Missing in KOI ({len(missing_rids)} RIDs):")
        for rid in missing_rids[:20]:
            print(f"    {rid}")
        if len(missing_rids) > 20:
            print(f"    ... and {len(missing_rids) - 20} more")


if __name__ == "__main__":
    main()
