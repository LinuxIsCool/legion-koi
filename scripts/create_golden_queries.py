#!/usr/bin/env -S python3 -u
"""Interactive tool to build and maintain the golden query set for retrieval evaluation.

Usage:
    # Discover results for a query (read-only exploration)
    uv run python scripts/create_golden_queries.py --discover "Block Science"

    # Add a new golden query interactively
    uv run python scripts/create_golden_queries.py --add

    # Review and update an existing query
    uv run python scripts/create_golden_queries.py --review --id factual-001

    # Validate all RIDs exist in DB
    uv run python scripts/create_golden_queries.py --validate

    # Print stats about the golden set
    uv run python scripts/create_golden_queries.py --stats
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.embeddings import create_embedder
from legion_koi.storage.postgres import PostgresStorage

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")

GOLDEN_PATH = Path("scripts/golden_queries.jsonl")
CATEGORIES = ["factual", "entity", "temporal", "cross-namespace", "exact-match", "semantic"]


# --- Discovery ---


def discover(storage: PostgresStorage, query: str, limit: int = 20):
    """Run FTS + semantic + hybrid, deduplicate, print numbered table."""
    print(f"\nDiscovering results for: \"{query}\"\n")

    all_results = {}  # rid -> {best_score, source, namespace, preview}

    # FTS
    try:
        fts_results = storage.search_text(query, limit=limit)
        for i, r in enumerate(fts_results):
            rid = r["rid"]
            if rid not in all_results or r.get("rank", 0) > all_results[rid]["score"]:
                preview = (r.get("search_text") or "")[:120].replace("\n", " ")
                all_results[rid] = {
                    "score": r.get("rank", 0),
                    "source": "fts",
                    "namespace": r.get("namespace", ""),
                    "preview": preview,
                    "position": i,
                }
    except Exception as e:
        print(f"  FTS failed: {e}")

    # Semantic + Hybrid with available configs
    configs = storage.list_embedding_configs()
    embedder = None
    embedding = None

    for cfg in configs:
        try:
            embedder = create_embedder(provider=cfg["provider"], model=cfg["model"])
            embedding = embedder.embed(query, input_type="query")
            break
        except Exception:
            continue

    if embedding is not None:
        # Semantic (first available config)
        if configs:
            try:
                sem_results = storage.search_config_semantic(
                    configs[0]["config_id"], embedding, limit=limit
                )
                for i, r in enumerate(sem_results):
                    rid = r["rid"]
                    score = r.get("similarity", 0)
                    if rid not in all_results or score > all_results[rid]["score"]:
                        preview = (r.get("search_text") or "")[:120].replace("\n", " ")
                        all_results[rid] = {
                            "score": score,
                            "source": f"semantic/{configs[0]['config_id']}",
                            "namespace": r.get("namespace", ""),
                            "preview": preview,
                            "position": i,
                        }
            except Exception as e:
                print(f"  Semantic search failed: {e}")

        # Hybrid (all configs)
        for cfg in configs:
            try:
                cfg_embedder = create_embedder(provider=cfg["provider"], model=cfg["model"])
                cfg_embedding = cfg_embedder.embed(query, input_type="query")
                hyb_results = storage.search_config_hybrid(
                    cfg["config_id"], query, cfg_embedding, limit=limit
                )
                for i, r in enumerate(hyb_results):
                    rid = r["rid"]
                    score = r.get("rrf_score", 0)
                    if rid not in all_results or score > all_results[rid]["score"]:
                        preview = (r.get("search_text") or "")[:120].replace("\n", " ")
                        all_results[rid] = {
                            "score": score,
                            "source": f"hybrid/{cfg['config_id']}",
                            "namespace": r.get("namespace", ""),
                            "preview": preview,
                            "position": i,
                        }
            except Exception:
                continue
    else:
        print("  WARN: no embedder available, skipping semantic/hybrid")

    if not all_results:
        print("  No results found.")
        return []

    # Sort by score descending
    sorted_rids = sorted(all_results.items(), key=lambda x: x[1]["score"], reverse=True)

    # Print table
    print(f" {'#':>3}  {'RID':<60} {'NS':<16} {'Score':>6}  {'Source':<20} Preview")
    print("─" * 140)
    for i, (rid, info) in enumerate(sorted_rids, 1):
        ns_short = info["namespace"].replace("legion.claude-", "")
        print(f" {i:>3}  {rid:<60} {ns_short:<16} {info['score']:>6.3f}  {info['source']:<20} {info['preview'][:40]}")

    print(f"\n  Total: {len(sorted_rids)} unique results\n")
    return sorted_rids


# --- Add Query ---


def add_query(storage: PostgresStorage):
    """Interactive loop to add a new golden query."""
    print("\n--- Add Golden Query ---\n")

    # Query text
    query = input("Query text: ").strip()
    if not query:
        print("Aborted.")
        return

    # Category
    print(f"Categories: {', '.join(CATEGORIES)}")
    category = input("Category: ").strip()
    if category not in CATEGORIES:
        print(f"Invalid category. Must be one of: {', '.join(CATEGORIES)}")
        return

    # Auto-generate ID
    existing = load_queries()
    cat_prefix = category.split("-")[0][:8]  # factual, entity, temporal, cross, exact, semantic
    cat_count = sum(1 for q in existing if q.get("category") == category)
    query_id = f"{cat_prefix}-{cat_count + 1:03d}"
    print(f"Auto-assigned ID: {query_id}")

    # Discover candidates
    print("\nRunning discovery...")
    candidates = discover(storage, query)

    if not candidates:
        print("No candidates found. Add RIDs manually or try a different query.")
        return

    # Select RIDs
    print("Select RIDs: enter numbers with grades, e.g. '1,3,7=2 5,12=1'")
    print("  =2 means highly relevant, =1 means partially relevant")
    selection = input("Selection: ").strip()
    if not selection:
        print("No selection. Saving without relevant RIDs is not allowed.")
        return

    relevant_rids = {}
    for part in selection.split():
        if "=" in part:
            nums_str, grade_str = part.split("=", 1)
            grade = int(grade_str)
        else:
            nums_str = part
            grade = 2  # default to highly relevant

        for num_str in nums_str.split(","):
            num_str = num_str.strip()
            if not num_str:
                continue
            idx = int(num_str) - 1  # 1-indexed
            if 0 <= idx < len(candidates):
                rid = candidates[idx][0]
                relevant_rids[rid] = grade

    if not relevant_rids:
        print("No valid RIDs selected. Aborted.")
        return

    # Expected namespaces (derived from selected RIDs)
    namespaces = list(set(
        candidates[i][1]["namespace"]
        for i in range(len(candidates))
        if candidates[i][0] in relevant_rids
    ))

    # Notes
    notes = input("Notes (optional): ").strip()

    # Build entry
    entry = {
        "id": query_id,
        "query": query,
        "category": category,
        "expected_namespaces": namespaces,
        "relevant_rids": relevant_rids,
        "notes": notes or "",
    }

    # Append
    with open(GOLDEN_PATH, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    print(f"\nAdded {query_id}: \"{query}\" with {len(relevant_rids)} relevant RIDs")
    print(f"  Grades: {dict(Counter(relevant_rids.values()))}")


# --- Review ---


def review_query(storage: PostgresStorage, query_id: str):
    """Re-run discovery for an existing query and show current judgments."""
    queries = load_queries()
    q = next((q for q in queries if q["id"] == query_id), None)
    if not q:
        print(f"Query '{query_id}' not found.")
        return

    print(f"\n--- Review: {q['id']} ---")
    print(f"Query: {q['query']}")
    print(f"Category: {q['category']}")
    print(f"Current relevant RIDs ({len(q['relevant_rids'])}):")
    for rid, grade in q["relevant_rids"].items():
        marker = "★★" if grade == 2 else "★"
        print(f"  {marker} {rid}")

    print("\nFresh discovery:")
    candidates = discover(storage, q["query"])

    # Mark which candidates are already judged
    if candidates:
        print("Current judgments in results:")
        for i, (rid, info) in enumerate(candidates, 1):
            if rid in q["relevant_rids"]:
                grade = q["relevant_rids"][rid]
                print(f"  #{i} {rid} → grade {grade}")

    print("\nTo update, edit scripts/golden_queries.jsonl directly.")


# --- Validate ---


def validate_queries(storage: PostgresStorage, prune: bool = False):
    """Check all RIDs in golden set exist in DB."""
    queries = load_queries()
    if not queries:
        print("No queries loaded.")
        return

    conn = psycopg.connect(DSN, row_factory=dict_row)
    missing_total = 0

    for q in queries:
        missing = []
        for rid in q["relevant_rids"]:
            row = conn.execute("SELECT 1 FROM bundles WHERE rid = %s", (rid,)).fetchone()
            if not row:
                missing.append(rid)

        if missing:
            print(f"  {q['id']}: {len(missing)} missing RIDs")
            for rid in missing:
                print(f"    ✗ {rid}")
            missing_total += len(missing)

    conn.close()

    if missing_total == 0:
        print(f"All RIDs valid across {len(queries)} queries.")
    else:
        print(f"\n{missing_total} missing RIDs total.")
        if prune:
            print("Pruning not yet implemented — edit golden_queries.jsonl directly.")


# --- Stats ---


def print_stats():
    """Print stats about the golden query set."""
    queries = load_queries()
    if not queries:
        print("No queries found.")
        return

    print(f"\n--- Golden Query Stats ---\n")
    print(f"Total queries: {len(queries)}")

    # By category
    cat_counts = Counter(q["category"] for q in queries)
    print(f"\nBy category:")
    for cat in CATEGORIES:
        count = cat_counts.get(cat, 0)
        print(f"  {cat:<20} {count}")

    # Unique RIDs
    all_rids = set()
    grade_counts = Counter()
    for q in queries:
        for rid, grade in q["relevant_rids"].items():
            all_rids.add(rid)
            grade_counts[grade] += 1

    print(f"\nUnique relevant RIDs: {len(all_rids)}")
    print(f"Grade distribution: {dict(sorted(grade_counts.items()))}")

    # Namespace coverage
    ns_counts = Counter()
    for q in queries:
        for ns in q.get("expected_namespaces", []):
            ns_counts[ns] += 1

    print(f"\nNamespace coverage:")
    for ns, count in ns_counts.most_common():
        print(f"  {ns:<40} {count} queries")


# --- Helpers ---


def load_queries() -> list[dict]:
    """Load all golden queries."""
    if not GOLDEN_PATH.exists():
        return []
    queries = []
    with open(GOLDEN_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Build and maintain golden query set")
    parser.add_argument("--discover", type=str, metavar="QUERY",
                        help="Run discovery search for a query")
    parser.add_argument("--add", action="store_true",
                        help="Add a new golden query interactively")
    parser.add_argument("--review", action="store_true",
                        help="Review an existing query")
    parser.add_argument("--id", type=str,
                        help="Query ID for --review")
    parser.add_argument("--validate", action="store_true",
                        help="Validate all RIDs exist in DB")
    parser.add_argument("--prune", action="store_true",
                        help="Remove stale RIDs (with --validate)")
    parser.add_argument("--stats", action="store_true",
                        help="Print golden set statistics")
    parser.add_argument("--limit", type=int, default=20,
                        help="Discovery result limit (default: 20)")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if not any([args.discover, args.add, args.review, args.validate]):
        parser.print_help()
        return

    # Connect for DB operations
    storage = PostgresStorage(DSN)

    if args.discover:
        discover(storage, args.discover, args.limit)
    elif args.add:
        add_query(storage)
    elif args.review:
        if not args.id:
            print("--review requires --id")
            return
        review_query(storage, args.id)
    elif args.validate:
        validate_queries(storage, args.prune)

    storage.close()


if __name__ == "__main__":
    main()
