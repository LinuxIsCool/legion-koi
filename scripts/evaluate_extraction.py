#!/usr/bin/env -S python3 -u
"""Evaluate entity extraction quality against gold standard.

Metrics:
- Entity F1 (exact match): name_normalized + type must match
- Entity F1 (partial match): name similarity > threshold + type match
- Per-type F1: breakdown by entity type
- Per-namespace F1: breakdown by content domain
- Coverage: % of gold entities found (recall)

Usage:
    uv run python scripts/evaluate_extraction.py
    uv run python scripts/evaluate_extraction.py --config fast    # evaluate regex-only
    uv run python scripts/evaluate_extraction.py --verbose        # show per-document details
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "src")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from legion_koi.extraction import normalize_entity_name

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_extractions.jsonl")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_results")


def fuzzy_match(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check if two names are similar enough to count as a match."""
    from rapidfuzz import fuzz
    return fuzz.ratio(normalize_entity_name(a), normalize_entity_name(b)) / 100.0 >= threshold


def compute_f1(predicted: list[dict], gold: list[dict], match_mode: str = "exact") -> dict:
    """Compute precision, recall, F1 for entity extraction.

    match_mode: 'exact' (name_normalized + type) or 'partial' (fuzzy name + type)
    """
    if not gold:
        return {"precision": 1.0 if not predicted else 0.0, "recall": 1.0, "f1": 1.0 if not predicted else 0.0}

    # Match predicted to gold
    matched_gold = set()
    matched_pred = set()

    for pi, p in enumerate(predicted):
        for gi, g in enumerate(gold):
            if gi in matched_gold:
                continue
            type_match = p.get("type", p.get("entity_type", "")) == g.get("type", g.get("entity_type", ""))
            if not type_match:
                continue

            if match_mode == "exact":
                name_match = normalize_entity_name(p["name"]) == normalize_entity_name(g["name"])
            else:
                name_match = fuzzy_match(p["name"], g["name"])

            if name_match:
                matched_gold.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": len(predicted) - tp, "fn": len(gold) - tp}


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity extraction")
    parser.add_argument("--config", default="default", help="Pipeline config to evaluate")
    parser.add_argument("--golden", default=GOLDEN_PATH, help="Path to golden extractions")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--match-mode", choices=["exact", "partial"], default="partial")
    args = parser.parse_args()

    if not os.path.exists(args.golden):
        print(f"Golden file not found: {args.golden}")
        print("Run: uv run python scripts/create_golden_extractions.py")
        sys.exit(1)

    # Load golden data
    golden_entries = []
    with open(args.golden) as f:
        for line in f:
            golden_entries.append(json.loads(line))
    print(f"Golden entries: {len(golden_entries)}")

    # Run extraction on each golden document
    from legion_koi.extraction.pipeline import ExtractionPipeline
    pipeline = ExtractionPipeline(config_name=args.config)
    print(f"Pipeline: {args.config}, backends: {[e.get_name() for e in pipeline._extractors]}")

    all_metrics = []
    type_metrics: dict[str, list] = defaultdict(list)
    ns_metrics: dict[str, list] = defaultdict(list)

    for entry in golden_entries:
        rid = entry["rid"]
        ns = entry["namespace"]
        text = entry.get("text_preview", "")
        gold = entry.get("entities", [])

        # Get full text from DB if preview is truncated
        try:
            import psycopg
            from psycopg.rows import dict_row
            dsn = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")
            conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
            row = conn.execute("SELECT search_text FROM bundles WHERE rid = %s", (rid,)).fetchone()
            if row:
                text = row["search_text"]
            conn.close()
        except Exception:
            pass

        if not text:
            continue

        result = pipeline.run(rid, ns, text)
        predicted = [
            {"name": e.name, "type": e.entity_type}
            for e in result.entities
        ]

        metrics = compute_f1(predicted, gold, match_mode=args.match_mode)
        metrics["rid"] = rid
        metrics["namespace"] = ns
        all_metrics.append(metrics)

        ns_metrics[ns].append(metrics)
        for g in gold:
            t = g.get("type", g.get("entity_type", ""))
            type_metrics[t].append(metrics)

        if args.verbose:
            print(f"\n  {rid}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
                  f"(pred={len(predicted)}, gold={len(gold)})")
            # Show false negatives
            for g in gold:
                matched = any(
                    fuzzy_match(p["name"], g["name"]) and p["type"] == g.get("type", g.get("entity_type", ""))
                    for p in predicted
                )
                if not matched:
                    print(f"    MISS: {g['name']} ({g.get('type', g.get('entity_type', ''))})")

    # Aggregate metrics
    if not all_metrics:
        print("No documents evaluated.")
        return

    avg_p = sum(m["precision"] for m in all_metrics) / len(all_metrics)
    avg_r = sum(m["recall"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)

    print(f"\n{'='*60}")
    print(f"Overall ({args.match_mode} match, {len(all_metrics)} docs):")
    print(f"  Precision: {avg_p:.4f}")
    print(f"  Recall:    {avg_r:.4f}")
    print(f"  F1:        {avg_f1:.4f}")

    print(f"\nPer namespace:")
    for ns, ms in sorted(ns_metrics.items()):
        ns_f1 = sum(m["f1"] for m in ms) / len(ms)
        ns_r = sum(m["recall"] for m in ms) / len(ms)
        print(f"  {ns}: F1={ns_f1:.4f} R={ns_r:.4f} ({len(ms)} docs)")

    print(f"\nPer type:")
    for t, ms in sorted(type_metrics.items()):
        t_f1 = sum(m["f1"] for m in ms) / len(ms)
        print(f"  {t}: F1={t_f1:.4f} ({len(ms)} docs)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"extraction_{args.config}_{args.match_mode}.json")
    results = {
        "config": args.config,
        "match_mode": args.match_mode,
        "documents": len(all_metrics),
        "overall": {"precision": avg_p, "recall": avg_r, "f1": avg_f1},
        "per_namespace": {
            ns: {
                "f1": sum(m["f1"] for m in ms) / len(ms),
                "recall": sum(m["recall"] for m in ms) / len(ms),
                "count": len(ms),
            }
            for ns, ms in ns_metrics.items()
        },
        "per_document": all_metrics,
    }
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {result_path}")


if __name__ == "__main__":
    main()
