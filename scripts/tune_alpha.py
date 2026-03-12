#!/usr/bin/env -S python3 -u
"""Grid search for optimal convex combination alpha.

Sweeps alpha from 0.0 to 1.0 in steps, evaluates each against the
golden query set, and reports the optimal value for Recall@10.

Usage:
    uv run python scripts/tune_alpha.py
    uv run python scripts/tune_alpha.py --step 0.02 --config telus-e5-1024
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")

from legion_koi.embeddings import create_embedder
from legion_koi.storage.postgres import PostgresStorage

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")
GOLDEN_QUERIES_PATH = Path("scripts/golden_queries.jsonl")

# Step size for alpha grid search (0.05 = 21 evaluations)
DEFAULT_STEP = 0.05
# Primary metric for optimization
PRIMARY_METRIC_K = 10


def load_golden_queries(path: Path) -> list[dict]:
    """Load golden queries from JSONL."""
    import json

    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            if q.get("relevant_rids"):
                queries.append(q)
    return queries


def recall_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    top_k = set(retrieved[:k])
    rel = set(relevant.keys())
    return len(top_k & rel) / len(rel) if rel else 0.0


def mrr(retrieved: list[str], relevant: dict[str, int]) -> float:
    for i, rid in enumerate(retrieved):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Tune convex alpha via grid search")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP, help="Alpha step size")
    parser.add_argument("--config", type=str, default=None, help="Embedding config ID")
    parser.add_argument("--k", type=int, default=PRIMARY_METRIC_K, help="K for Recall@K")
    args = parser.parse_args()

    if not GOLDEN_QUERIES_PATH.exists():
        print(f"ERROR: golden queries not found at {GOLDEN_QUERIES_PATH}")
        sys.exit(1)

    queries = load_golden_queries(GOLDEN_QUERIES_PATH)
    print(f"Loaded {len(queries)} golden queries")

    storage = PostgresStorage(DSN)

    # Resolve config
    if args.config:
        config_id = args.config
    else:
        default = storage.get_default_config()
        config_id = default["config_id"] if default else storage.list_embedding_configs()[0]["config_id"]

    config = next(c for c in storage.list_embedding_configs() if c["config_id"] == config_id)
    embedder = create_embedder(provider=config["provider"], model=config["model"])
    print(f"Config: {config_id} ({config['provider']}/{config['model']})")

    # Pre-embed all queries
    print("Embedding queries...")
    query_embeddings = {}
    for q in queries:
        query_embeddings[q["id"]] = embedder.embed(q["query"], input_type="query")

    # Pre-fetch FTS and semantic results for all queries (they don't change with alpha)
    from legion_koi.constants import RRF_FETCH_MULTIPLIER

    fetch_limit = 20 * RRF_FETCH_MULTIPLIER
    print("Fetching FTS and semantic results...")
    fts_cache = {}
    vec_cache = {}
    for q in queries:
        fts_cache[q["id"]] = storage.search_text(q["query"], limit=fetch_limit)
        vec_cache[q["id"]] = storage.search_config_semantic(
            config_id, query_embeddings[q["id"]], limit=fetch_limit
        )

    # Grid search
    from legion_koi.retrieval.fusion import convex_combine

    alphas = []
    step = args.step
    a = 0.0
    while a <= 1.0 + step / 2:
        alphas.append(round(a, 4))
        a += step

    print(f"\nGrid search: {len(alphas)} alpha values (step={step})\n")
    print(f"{'Alpha':>8}  {'Recall@' + str(args.k):>10}  {'MRR':>8}")
    print("-" * 32)

    best_alpha = 0.0
    best_recall = 0.0
    best_mrr = 0.0

    for alpha in alphas:
        recalls = []
        mrrs = []
        for q in queries:
            results = convex_combine(
                fts_cache[q["id"]],
                vec_cache[q["id"]],
                alpha=alpha,
                limit=20,
            )
            retrieved = [r["rid"] for r in results]
            recalls.append(recall_at_k(retrieved, q["relevant_rids"], args.k))
            mrrs.append(mrr(retrieved, q["relevant_rids"]))

        avg_recall = sum(recalls) / len(recalls)
        avg_mrr = sum(mrrs) / len(mrrs)

        marker = " <-- best" if avg_recall > best_recall else ""
        print(f"{alpha:>8.3f}  {avg_recall:>10.4f}  {avg_mrr:>8.4f}{marker}")

        if avg_recall > best_recall or (avg_recall == best_recall and avg_mrr > best_mrr):
            best_alpha = alpha
            best_recall = avg_recall
            best_mrr = avg_mrr

    print(f"\nOptimal alpha: {best_alpha}")
    print(f"  Recall@{args.k}: {best_recall:.4f}")
    print(f"  MRR: {best_mrr:.4f}")
    print(f"\nTo apply: set CONVEX_ALPHA = {best_alpha} in src/legion_koi/constants.py")

    storage.close()


if __name__ == "__main__":
    main()
