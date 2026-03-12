#!/usr/bin/env -S python3 -u
"""Create gold standard entity extraction dataset for evaluation.

Interactive tool: samples documents, runs LLM pre-annotation, presents
for human review (accept/reject/modify/add).

Usage:
    uv run python scripts/create_golden_extractions.py
    uv run python scripts/create_golden_extractions.py --count 10 --namespace legion.claude-journal
    uv run python scripts/create_golden_extractions.py --auto  # accept LLM annotations without review
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.storage.postgres import _extract_search_text

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_extractions.jsonl")

SAMPLE_NAMESPACES = [
    "legion.claude-journal",
    "legion.claude-recording",
    "legion.claude-message",
    "legion.claude-code",
    "legion.claude-web.conversation",
]

# How many per namespace
DEFAULT_PER_NS = 10


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def sample_bundles(conn, namespace: str, count: int) -> list[dict]:
    """Random sample of bundles from a namespace (with non-empty search_text)."""
    rows = conn.execute(
        """
        SELECT rid, namespace, contents, search_text
        FROM bundles
        WHERE namespace = %s AND btrim(search_text) != ''
        ORDER BY random()
        LIMIT %s
        """,
        (namespace, count),
    ).fetchall()
    return [dict(r) for r in rows]


def extract_with_pipeline(rid: str, namespace: str, text: str) -> list[dict]:
    """Run extraction pipeline on text, return entity dicts."""
    from legion_koi.extraction import run_extraction
    result = run_extraction(rid, namespace, text)
    return [
        {
            "name": e.name,
            "type": e.entity_type,
            "supertype": e.supertype,
            "confidence": e.confidence,
        }
        for e in result.entities
    ]


def review_interactive(rid: str, text_preview: str, entities: list[dict]) -> list[dict]:
    """Interactive review: show entities, let user accept/reject/modify."""
    print(f"\n{'='*80}")
    print(f"RID: {rid}")
    print(f"Text preview:\n{text_preview[:500]}")
    print(f"\n--- LLM-suggested entities ({len(entities)}) ---")
    for i, e in enumerate(entities, 1):
        print(f"  {i}. {e['name']} ({e['type']}/{e['supertype']}) conf={e['confidence']:.2f}")

    print("\nActions: [a]ccept all, [r]eview each, [s]kip, [q]uit")
    action = input("> ").strip().lower()

    if action == "q":
        return None  # Signal to quit
    if action == "s":
        return []  # Skip this doc
    if action == "a":
        return entities

    # Review each
    reviewed = []
    for i, e in enumerate(entities, 1):
        print(f"  {i}. {e['name']} ({e['type']}/{e['supertype']})")
        sub = input("    [a]ccept, [r]eject, [m]odify? ").strip().lower()
        if sub == "a" or sub == "":
            reviewed.append(e)
        elif sub == "m":
            name = input(f"    Name [{e['name']}]: ").strip() or e["name"]
            etype = input(f"    Type [{e['type']}]: ").strip() or e["type"]
            reviewed.append({**e, "name": name, "type": etype})
        # 'r' = reject, skip

    # Add missing entities
    while True:
        add = input("  Add entity? (name or Enter to finish): ").strip()
        if not add:
            break
        etype = input(f"    Type for '{add}': ").strip()
        reviewed.append({
            "name": add,
            "type": etype,
            "supertype": "",
            "confidence": 1.0,
        })

    return reviewed


def main():
    parser = argparse.ArgumentParser(description="Create gold standard extractions")
    parser.add_argument("--count", type=int, default=DEFAULT_PER_NS, help="Samples per namespace")
    parser.add_argument("--namespace", help="Only sample from this namespace")
    parser.add_argument("--auto", action="store_true", help="Accept LLM annotations without review")
    parser.add_argument("--append", action="store_true", help="Append to existing file")
    args = parser.parse_args()

    conn = get_conn()

    # Load existing entries to avoid duplicates
    existing_rids = set()
    if os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH) as f:
            for line in f:
                entry = json.loads(line)
                existing_rids.add(entry["rid"])
        print(f"Existing golden entries: {len(existing_rids)}")

    namespaces = [args.namespace] if args.namespace else SAMPLE_NAMESPACES
    entries = []

    for ns in namespaces:
        samples = sample_bundles(conn, ns, args.count * 2)  # oversample for skips
        random.shuffle(samples)
        ns_count = 0

        for bundle in samples:
            if ns_count >= args.count:
                break
            if bundle["rid"] in existing_rids:
                continue

            text = bundle["search_text"]
            if not text or len(text.strip()) < 50:
                continue

            print(f"\n[{ns}] Extracting: {bundle['rid']}")
            try:
                llm_entities = extract_with_pipeline(bundle["rid"], ns, text)
            except Exception as e:
                print(f"  Extraction error: {e}")
                continue

            if args.auto:
                reviewed = llm_entities
            else:
                reviewed = review_interactive(bundle["rid"], text, llm_entities)
                if reviewed is None:
                    break  # Quit signal

            entry = {
                "rid": bundle["rid"],
                "namespace": ns,
                "text_preview": text[:500],
                "entities": reviewed,
                "relations": [],
                "annotator": "shawn+claude" if not args.auto else "claude-auto",
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
            entries.append(entry)
            existing_rids.add(bundle["rid"])
            ns_count += 1
            print(f"  Saved: {len(reviewed)} entities")

    # Write
    mode = "a" if args.append or os.path.exists(GOLDEN_PATH) else "w"
    with open(GOLDEN_PATH, mode) as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    total = len(existing_rids)
    print(f"\nDone: {len(entries)} new entries, {total} total in {GOLDEN_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
