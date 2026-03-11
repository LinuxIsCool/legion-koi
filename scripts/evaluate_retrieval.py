#!/usr/bin/env -S python3 -u
"""Retrieval evaluation framework — measure search quality against golden queries.

Runs every (mode, config) combination against a golden query set with graded
relevance judgments. Computes standard IR metrics: Recall@K, MRR, NDCG@K, MAP@K.

Usage:
    # Run all modes and configs
    uv run python scripts/evaluate_retrieval.py

    # Specific modes and configs
    uv run python scripts/evaluate_retrieval.py --modes fts,hybrid --configs telus-e5-1024

    # Compare against baseline
    uv run python scripts/evaluate_retrieval.py --compare scripts/eval_results/baseline.json

    # With latency and per-query detail
    uv run python scripts/evaluate_retrieval.py --latency --verbose
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.embeddings import create_embedder
from legion_koi.storage.postgres import PostgresStorage

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")

GOLDEN_QUERIES_PATH = Path("scripts/golden_queries.jsonl")
EVAL_RESULTS_DIR = Path("scripts/eval_results")

ALL_MODES = ["fts", "semantic", "hybrid"]


# --- Metrics (pure Python) ---


def recall_at_k(retrieved_rids: list[str], relevant_rids: dict[str, int], k: int) -> float:
    top_k = set(retrieved_rids[:k])
    relevant = set(relevant_rids.keys())
    return len(top_k & relevant) / len(relevant) if relevant else 0.0


def mrr(retrieved_rids: list[str], relevant_rids: dict[str, int]) -> float:
    for i, rid in enumerate(retrieved_rids):
        if rid in relevant_rids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_rids: list[str], relevant_rids: dict[str, int], k: int) -> float:
    dcg = sum(
        relevant_rids.get(rid, 0) / math.log2(i + 2)
        for i, rid in enumerate(retrieved_rids[:k])
    )
    ideal_rels = sorted(relevant_rids.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(retrieved_rids: list[str], relevant_rids: dict[str, int], k: int) -> float:
    relevant = set(relevant_rids.keys())
    hits, sum_prec = 0, 0.0
    for i, rid in enumerate(retrieved_rids[:k]):
        if rid in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant) if relevant else 0.0


def compute_metrics(
    retrieved_rids: list[str], relevant_rids: dict[str, int], k_values: list[int]
) -> dict:
    """Compute all metrics for one query result."""
    m = {"mrr": mrr(retrieved_rids, relevant_rids)}
    for k in k_values:
        m[f"recall@{k}"] = recall_at_k(retrieved_rids, relevant_rids, k)
        m[f"ndcg@{k}"] = ndcg_at_k(retrieved_rids, relevant_rids, k)
        m[f"map@{k}"] = average_precision_at_k(retrieved_rids, relevant_rids, k)
    return m


# --- Golden Query Loading ---


def load_golden_queries(path: Path, category_filter: str | None = None) -> list[dict]:
    """Load golden queries from JSONL, optionally filtering by category."""
    queries = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN: skipping line {line_num}: {e}")
                continue
            if not q.get("relevant_rids"):
                print(f"  WARN: skipping {q.get('id', f'line {line_num}')}: no relevant RIDs")
                continue
            if category_filter and q.get("category") != category_filter:
                continue
            queries.append(q)
    return queries


# --- Embedding Cache ---


class EmbeddingCache:
    """Cache embeddings by (provider, model, query) to avoid redundant API calls."""

    def __init__(self):
        self._cache: dict[str, list[float]] = {}  # "provider:model:query" -> embedding
        self._embedders: dict[str, object] = {}  # "provider:model" -> Embedder

    def get_or_embed(self, provider: str, model: str, query: str) -> list[float] | None:
        cache_key = f"{provider}:{model}:{query}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedder_key = f"{provider}:{model}"
        if embedder_key not in self._embedders:
            try:
                self._embedders[embedder_key] = create_embedder(provider=provider, model=model)
            except Exception as e:
                print(f"  WARN: cannot create embedder {embedder_key}: {e}")
                self._embedders[embedder_key] = None

        embedder = self._embedders[embedder_key]
        if embedder is None:
            return None

        try:
            embedding = embedder.embed(query, input_type="query")
            self._cache[cache_key] = embedding
            return embedding
        except Exception as e:
            print(f"  WARN: embedding failed for {embedder_key}: {e}")
            return None


# --- Search Configuration Discovery ---


def discover_configs(storage: PostgresStorage) -> list[dict]:
    """Get all registered embedding configs from DB."""
    return storage.list_embedding_configs()


def build_eval_matrix(
    modes: list[str], config_ids: list[str], all_configs: list[dict]
) -> list[dict]:
    """Build the (mode, config) combinations to evaluate.

    Returns list of dicts with keys: mode, config_id, provider, model, label
    """
    combos = []

    config_map = {c["config_id"]: c for c in all_configs}

    for mode in modes:
        if mode == "fts":
            combos.append({
                "mode": "fts",
                "config_id": None,
                "provider": None,
                "model": None,
                "label": "fts",
            })
        elif mode in ("semantic", "hybrid"):
            # Legacy embeddings (single-table)
            if "legacy" in config_ids:
                combos.append({
                    "mode": mode,
                    "config_id": "legacy",
                    "provider": None,
                    "model": None,
                    "label": f"{mode}/legacy",
                })

            # Per-config embeddings
            for cid in config_ids:
                if cid == "legacy":
                    continue
                if cid not in config_map:
                    print(f"  WARN: config '{cid}' not found in DB, skipping")
                    continue
                cfg = config_map[cid]
                combos.append({
                    "mode": mode,
                    "config_id": cid,
                    "provider": cfg.get("provider"),
                    "model": cfg.get("model"),
                    "label": f"{mode}/{cid}",
                })

    return combos


# --- Search Runner ---


def run_search(
    storage: PostgresStorage,
    combo: dict,
    query: str,
    embedding: list[float] | None,
    limit: int,
) -> list[str]:
    """Execute one search and return retrieved RIDs in order."""
    mode = combo["mode"]
    config_id = combo["config_id"]

    try:
        if mode == "fts":
            results = storage.search_text(query, limit=limit)
        elif mode == "semantic":
            if embedding is None:
                return []
            if config_id == "legacy":
                results = storage.search_semantic(embedding, limit=limit)
            else:
                results = storage.search_config_semantic(config_id, embedding, limit=limit)
        elif mode == "hybrid":
            if embedding is None:
                return []
            if config_id == "legacy":
                results = storage.search_hybrid(query, embedding, limit=limit)
            else:
                results = storage.search_config_hybrid(config_id, query, embedding, limit=limit)
        else:
            return []
    except Exception as e:
        print(f"  WARN: search failed for {combo['label']}: {e}")
        return []

    return [r["rid"] for r in results]


# --- Evaluation Engine ---


def evaluate(
    storage: PostgresStorage,
    queries: list[dict],
    combos: list[dict],
    embed_cache: EmbeddingCache,
    k_values: list[int],
    limit: int,
    measure_latency: bool,
    verbose: bool,
) -> list[dict]:
    """Run full evaluation. Returns list of configuration results."""
    results = []

    for combo in combos:
        label = combo["label"]
        provider = combo["provider"]
        model = combo["model"]

        per_query = []
        latencies = []

        for q in queries:
            qid = q["id"]
            query_text = q["query"]
            relevant = q["relevant_rids"]

            # Get embedding if needed
            embedding = None
            if combo["mode"] in ("semantic", "hybrid"):
                if combo["config_id"] == "legacy":
                    # Legacy uses default embedder
                    embedding = embed_cache.get_or_embed(
                        provider or "telus", model or "default", query_text
                    )
                else:
                    embedding = embed_cache.get_or_embed(provider, model, query_text)

                if embedding is None and combo["mode"] == "semantic":
                    per_query.append({
                        "id": qid,
                        "skipped": True,
                        "reason": "embedding unavailable",
                    })
                    continue

            # Run search
            t0 = time.monotonic()
            retrieved = run_search(storage, combo, query_text, embedding, limit)
            elapsed_ms = (time.monotonic() - t0) * 1000

            if measure_latency:
                latencies.append(elapsed_ms)

            # Compute metrics
            metrics = compute_metrics(retrieved, relevant, k_values)

            entry = {"id": qid, **metrics, "retrieved_count": len(retrieved)}
            if verbose:
                entry["retrieved"] = retrieved[:20]
            per_query.append(entry)

            if verbose:
                found = set(retrieved) & set(relevant.keys())
                missed = set(relevant.keys()) - set(retrieved)
                print(f"  {label} | {qid}: MRR={metrics['mrr']:.3f} "
                      f"R@10={metrics.get('recall@10', 0):.3f} "
                      f"found={len(found)} missed={len(missed)}")

        # Aggregate metrics
        scored = [q for q in per_query if "mrr" in q]
        if not scored:
            print(f"  WARN: no scored queries for {label}")
            continue

        agg = {}
        metric_keys = [k for k in scored[0] if k not in ("id", "retrieved", "retrieved_count", "skipped", "reason")]
        for key in metric_keys:
            vals = [q[key] for q in scored]
            agg[key] = sum(vals) / len(vals)

        # Per-category aggregation
        per_category = {}
        for q_result in scored:
            q_data = next((q for q in queries if q["id"] == q_result["id"]), None)
            if not q_data:
                continue
            cat = q_data.get("category", "unknown")
            if cat not in per_category:
                per_category[cat] = []
            per_category[cat].append(q_result)

        cat_agg = {}
        for cat, cat_queries in per_category.items():
            cat_agg[cat] = {}
            for key in metric_keys:
                vals = [q[key] for q in cat_queries]
                cat_agg[cat][key] = sum(vals) / len(vals)

        # Per-namespace aggregation
        per_ns = {}
        for q_result in scored:
            q_data = next((q for q in queries if q["id"] == q_result["id"]), None)
            if not q_data:
                continue
            for rid in q_data["relevant_rids"]:
                # Extract namespace from RID: orn:namespace:reference
                parts = rid.split(":", 2)
                if len(parts) >= 2:
                    ns = parts[1]
                    if ns not in per_ns:
                        per_ns[ns] = []
                    per_ns[ns].append(q_result)

        ns_agg = {}
        for ns, ns_queries in per_ns.items():
            ns_agg[ns] = {}
            # Deduplicate by query id
            seen = set()
            unique = []
            for q in ns_queries:
                if q["id"] not in seen:
                    seen.add(q["id"])
                    unique.append(q)
            for key in metric_keys:
                vals = [q[key] for q in unique]
                ns_agg[ns][key] = sum(vals) / len(vals)

        # Latency stats
        latency_stats = None
        if measure_latency and latencies:
            latencies_sorted = sorted(latencies)
            latency_stats = {
                "mean": sum(latencies) / len(latencies),
                "p50": latencies_sorted[len(latencies_sorted) // 2],
                "p95": latencies_sorted[int(len(latencies_sorted) * 0.95)],
                "max": latencies_sorted[-1],
            }

        results.append({
            "mode": combo["mode"],
            "config": combo["config_id"],
            "label": label,
            "metrics": agg,
            "latency_ms": latency_stats,
            "per_namespace": ns_agg,
            "per_category": cat_agg,
            "per_query": per_query,
        })

    return results


# --- Output Formatting ---


def print_table(results: list[dict], k_values: list[int], compare_data: dict | None = None):
    """Print formatted evaluation table."""
    # Build comparison lookup: label -> metrics
    prev = {}
    if compare_data:
        for cfg in compare_data.get("configurations", []):
            prev[cfg.get("label", "")] = cfg.get("metrics", {})

    # Header columns
    metric_cols = []
    for k in k_values:
        metric_cols.append(f"Recall@{k}")
    metric_cols.append("MRR")
    for k in k_values:
        if k == k_values[0]:
            continue  # Skip first k for NDCG/MAP to keep table compact
        metric_cols.append(f"NDCG@{k}")
        metric_cols.append(f"MAP@{k}")

    # Map display names to metric keys
    col_keys = []
    for k in k_values:
        col_keys.append(f"recall@{k}")
    col_keys.append("mrr")
    for k in k_values:
        if k == k_values[0]:
            continue
        col_keys.append(f"ndcg@{k}")
        col_keys.append(f"map@{k}")

    label_width = max(24, max((len(r["label"]) for r in results), default=24))
    col_width = 10

    # Header
    header = f"{'Configuration':<{label_width}}"
    for col in metric_cols:
        header += f"  {col:>{col_width}}"
    if any(r.get("latency_ms") for r in results):
        header += f"  {'Lat(ms)':>{col_width}}"

    print(f"\n=== Retrieval Evaluation ({len(results)} configs) ===\n")
    print(header)
    print("─" * len(header))

    is_tty = sys.stdout.isatty()

    for r in results:
        metrics = r["metrics"]
        line = f"{r['label']:<{label_width}}"

        for key in col_keys:
            val = metrics.get(key, 0.0)
            cell = f"{val:.3f}"

            # Delta from comparison
            if r["label"] in prev and key in prev[r["label"]]:
                delta = val - prev[r["label"]][key]
                if abs(delta) >= 0.001:
                    sign = "+" if delta > 0 else ""
                    delta_str = f"{sign}{delta:.3f}"
                    if is_tty:
                        color = "\033[32m" if delta > 0 else "\033[31m"
                        reset = "\033[0m"
                        cell = f"{val:.3f}{color}({delta_str}){reset}"
                    else:
                        cell = f"{val:.3f}({delta_str})"

            line += f"  {cell:>{col_width}}"

        if r.get("latency_ms"):
            lat = r["latency_ms"]
            line += f"  {lat['mean']:>{col_width - 2}.0f}ms"

        print(line)

    print()


def print_category_breakdown(results: list[dict], k_values: list[int]):
    """Print per-category metrics for each config."""
    # Gather all categories
    all_cats = set()
    for r in results:
        all_cats.update(r.get("per_category", {}).keys())
    if not all_cats:
        return

    print("=== Per-Category Breakdown ===\n")
    k = k_values[1] if len(k_values) > 1 else k_values[0]

    for r in results:
        cat_data = r.get("per_category", {})
        if not cat_data:
            continue
        print(f"  {r['label']}:")
        for cat in sorted(cat_data.keys()):
            m = cat_data[cat]
            print(f"    {cat:<20} Recall@{k}={m.get(f'recall@{k}', 0):.3f}  "
                  f"MRR={m.get('mrr', 0):.3f}  NDCG@{k}={m.get(f'ndcg@{k}', 0):.3f}")
        print()


def build_json_report(
    results: list[dict], query_count: int, limit: int, k_values: list[int]
) -> dict:
    """Build the JSON report structure."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query_count": query_count,
        "limit": limit,
        "k_values": k_values,
        "configurations": results,
    }


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against golden queries")
    parser.add_argument("--queries", type=Path, default=GOLDEN_QUERIES_PATH,
                        help="Path to golden queries JSONL")
    parser.add_argument("--modes", type=str, default=None,
                        help="Comma-separated search modes: fts,semantic,hybrid (default: all)")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config IDs (default: all registered + legacy)")
    parser.add_argument("--k", type=str, default="5,10,20",
                        help="Comma-separated K values for Recall@K (default: 5,10,20)")
    parser.add_argument("--limit", type=int, default=20,
                        help="Search result limit (default: 20)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save JSON report to path (default: auto-timestamped)")
    parser.add_argument("--compare", type=Path, default=None,
                        help="Compare against previous JSON report")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter queries by category")
    parser.add_argument("--latency", action="store_true",
                        help="Measure per-query timing")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-query detail")
    args = parser.parse_args()

    # Parse K values
    k_values = [int(k.strip()) for k in args.k.split(",")]

    # Parse modes
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else ALL_MODES

    # Load golden queries
    if not args.queries.exists():
        print(f"ERROR: golden queries not found at {args.queries}")
        sys.exit(1)
    queries = load_golden_queries(args.queries, args.category)
    print(f"Loaded {len(queries)} golden queries from {args.queries}")
    if not queries:
        print("ERROR: no queries to evaluate")
        sys.exit(1)

    # Load comparison data
    compare_data = None
    if args.compare:
        if not args.compare.exists():
            print(f"WARN: comparison file not found: {args.compare}")
        else:
            with open(args.compare) as f:
                compare_data = json.load(f)
            print(f"Loaded comparison baseline: {args.compare} ({compare_data.get('query_count', '?')} queries)")

    # Connect to DB
    storage = PostgresStorage(DSN)
    print(f"Connected to {DSN}")

    # Discover configs
    all_configs = discover_configs(storage)
    config_ids = [c["config_id"] for c in all_configs]

    if args.configs:
        config_ids = [c.strip() for c in args.configs.split(",")]
    else:
        # Add legacy to the list if not already
        if "legacy" not in config_ids:
            config_ids.append("legacy")

    print(f"Modes: {modes}")
    print(f"Configs: {config_ids}")

    # Build eval matrix
    combos = build_eval_matrix(modes, config_ids, all_configs)
    print(f"Evaluation matrix: {len(combos)} configurations\n")

    # Run evaluation
    embed_cache = EmbeddingCache()
    results = evaluate(
        storage, queries, combos, embed_cache,
        k_values, args.limit, args.latency, args.verbose,
    )

    # Print results
    print_table(results, k_values, compare_data)

    if args.verbose:
        print_category_breakdown(results, k_values)

    # Save JSON report
    report = build_json_report(results, len(queries), args.limit, k_values)

    if args.output:
        output_path = args.output
    else:
        EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = EVAL_RESULTS_DIR / f"{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved: {output_path}")

    storage.close()


if __name__ == "__main__":
    main()
