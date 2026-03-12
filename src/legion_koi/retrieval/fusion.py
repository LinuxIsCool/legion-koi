"""Convex combination fusion — replaces Reciprocal Rank Fusion.

score = alpha * fts_norm + (1 - alpha) * semantic_norm

FTS scores (ts_rank) are min-max normalized across the result set.
Semantic scores (cosine similarity) are already in [0, 1].

Why convex over RRF: RRF discards score magnitude — a result ranked #1
by a wide margin gets the same boost as one ranked #1 by a hair.
Convex combination preserves score proportionality and allows tuning
the FTS/semantic balance via a single alpha parameter.
"""

from __future__ import annotations


def convex_combine(
    fts_results: list[dict],
    vec_results: list[dict],
    alpha: float,
    limit: int = 20,
) -> list[dict]:
    """Merge FTS and semantic results using convex combination.

    Args:
        fts_results: Dicts with 'rid' and 'rank' (ts_rank score).
        vec_results: Dicts with 'rid' and 'similarity' (cosine similarity, 0-1).
        alpha: Weight for FTS component (0-1). semantic weight = 1 - alpha.
        limit: Max results to return.

    Returns:
        Merged results sorted by combined score, each with 'fusion_score'.
    """
    if not fts_results and not vec_results:
        return []

    # Normalize FTS scores via min-max
    fts_scores: dict[str, float] = {}
    if fts_results:
        raw_scores = [r.get("rank", 0.0) or 0.0 for r in fts_results]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score

        for r in fts_results:
            raw = r.get("rank", 0.0) or 0.0
            # Normalize to [0, 1]; if all scores are equal, assign 1.0
            fts_scores[r["rid"]] = (raw - min_score) / score_range if score_range > 0 else 1.0

    # Semantic scores are already 0-1 (cosine similarity)
    vec_scores: dict[str, float] = {}
    for r in vec_results:
        vec_scores[r["rid"]] = r.get("similarity", 0.0) or 0.0

    # Merge all RIDs
    all_rids = set(fts_scores.keys()) | set(vec_scores.keys())

    # Build bundle map (prefer FTS version for headline, vec version for chunk_text)
    bundle_map: dict[str, dict] = {}
    for r in fts_results:
        bundle_map[r["rid"]] = r
    for r in vec_results:
        rid = r["rid"]
        if rid not in bundle_map:
            bundle_map[rid] = r
        else:
            # Merge: keep chunk_text from vector results for reranking
            if r.get("chunk_text") and not bundle_map[rid].get("chunk_text"):
                bundle_map[rid]["chunk_text"] = r["chunk_text"]
                bundle_map[rid]["chunk_index"] = r.get("chunk_index")

    # Compute convex combination
    combined: dict[str, float] = {}
    for rid in all_rids:
        fts_component = alpha * fts_scores.get(rid, 0.0)
        vec_component = (1 - alpha) * vec_scores.get(rid, 0.0)
        combined[rid] = fts_component + vec_component

    # Sort and return
    sorted_rids = sorted(combined, key=lambda r: combined[r], reverse=True)[:limit]
    results = []
    for rid in sorted_rids:
        r = bundle_map[rid]
        r["fusion_score"] = combined[rid]
        results.append(r)
    return results
