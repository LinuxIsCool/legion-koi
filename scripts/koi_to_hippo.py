#!/usr/bin/env -S python3 -u
# /// script
# requires-python = ">=3.11"
# dependencies = ["psycopg[binary]", "redis", "httpx", "pyyaml"]
# ///
"""Bridge KOI bundles to Hippo graph — extract triples from KOI search_text and insert into FalkorDB.

Closes the critical gap: KOI has bundles with search_text and extracted entities,
but Hippo only indexes from markdown files on disk. This script reads from KOI
PostgreSQL, chunks the text, extracts triples via TELUS Ollama (two-stage NER+OpenIE),
and inserts into FalkorDB with proper source IDs.

Usage:
    uv run python scripts/koi_to_hippo.py --namespace legion.claude-youtube
    uv run python scripts/koi_to_hippo.py --namespace legion.claude-youtube --dry-run
    uv run python scripts/koi_to_hippo.py --namespace legion.claude-youtube --batch-size 10
    uv run python scripts/koi_to_hippo.py --resume
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psycopg
import redis
from psycopg.rows import dict_row

# Add hippo scripts to path for shared utilities
HIPPO_SCRIPTS = Path.home() / ".claude" / "plugins" / "local" / "legion-plugins" / "plugins" / "claude-hippo" / "scripts"
sys.path.insert(0, str(HIPPO_SCRIPTS))

from hippo_common import load_config, load_secrets, get_redis, graph_query

# Add legion-koi src to path
sys.path.insert(0, "src")

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")

# State file for checkpoint/resume
STATE_DIR = Path.home() / ".claude" / "local" / "hippo" / "koi-bridge"

# Namespace -> hippo content_type mapping (for extraction hints)
NAMESPACE_CONTENT_TYPE = {
    "legion.claude-youtube": "youtube",
    "legion.claude-journal": "journal",
    "legion.claude-venture": "venture",
    "legion.claude-github": "code",
    "legion.claude-recording": "recording",
    "legion.claude-code": "code",
    "legion.claude-message": "message",
}

# Namespace -> hippo source prefix
NAMESPACE_SOURCE_PREFIX = {
    "legion.claude-youtube": "youtube",
    "legion.claude-journal": "journal",
    "legion.claude-venture": "venture",
    "legion.claude-github": "github",
    "legion.claude-recording": "recording",
    "legion.claude-code": "code",
    "legion.claude-message": "message",
}

# Type hints for extraction prompts
TYPE_HINTS = {
    "youtube": "YouTube video transcript from a developer channel. Focus on: tools demonstrated, projects built, coding patterns, AI techniques, key takeaways, people mentioned.",
    "journal": "Focus on: decisions made, tools mentioned, people, ventures, events, reflections.",
    "venture": "Focus on: goals, milestones, stakeholders, technologies, status, deadlines.",
    "code": "Focus on: libraries, frameworks, services, architectural patterns, design decisions.",
    "recording": "Focus on: speakers, decisions, action items, tools discussed, commitments.",
    "message": "Focus on: relationships between people, projects discussed, decisions, commitments.",
}


def get_koi_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def load_state(namespace: str) -> set:
    """Load set of already-processed RIDs for this namespace."""
    state_file = STATE_DIR / f"{namespace.replace('.', '_')}.json"
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text())
            return set(data.get("processed_rids", []))
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def save_state(namespace: str, processed_rids: set):
    """Save processed RIDs for checkpoint/resume."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_file = STATE_DIR / f"{namespace.replace('.', '_')}.json"
    state_file.write_text(json.dumps({
        "namespace": namespace,
        "processed_rids": list(processed_rids),
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }))


def fetch_bundles(conn, namespace: str, processed_rids: set, limit: int):
    """Fetch bundles that haven't been processed yet."""
    rows = conn.execute(
        """
        SELECT rid, namespace, reference, contents, search_text
        FROM bundles
        WHERE namespace = %s
          AND btrim(coalesce(search_text, '')) != ''
          AND length(search_text) >= 100
        ORDER BY created_at
        """,
        (namespace,),
    ).fetchall()
    # Filter out already-processed in Python (simpler than large NOT IN clause)
    return [r for r in rows if r["rid"] not in processed_rids][:limit]


def chunk_document(text: str, chunk_size: int = 2000, overlap: int = 400) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries."""
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + "\n\n" + para
        else:
            current = (current + "\n\n" + para) if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def extract_entities_ner(text: str, content_type: str, secrets: dict,
                        config: dict, client: httpx.Client) -> list[str]:
    """Stage 1: NER — extract named entities from text."""
    url = secrets.get("TELUS_OLLAMA_URL", "")
    key = secrets.get("TELUS_OLLAMA_KEY", "")
    model = config.get("extraction", {}).get("model", "gpt-oss:120b")

    if not url or not key:
        return []

    hint = TYPE_HINTS.get(content_type, "Extract all substantive entities.")

    prompt = f"""Extract all named entities from this text.
Include: people, organizations, tools, software, locations, projects, events, concepts.
{hint}
Normalize: proper nouns keep original case, common nouns lowercase.
Keep entities concise (1-4 words).
SKIP placeholder values like "none", "not yet", "unknown", "0", "true", "false".

Output ONLY a JSON array of strings:
["entity1", "entity2", ...]

Text:
{text}"""

    import re
    timeout = config.get("extraction", {}).get("api_timeout", 30)

    try:
        resp = client.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        if content.startswith("```"):
            content = re.sub(r"^```\w*\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        cleaned = re.sub(r",\s*([}\]])", r"\1", content.strip())
        decoder = json.JSONDecoder()
        entities, _ = decoder.raw_decode(cleaned)
        if isinstance(entities, list):
            return [e.strip() for e in entities if isinstance(e, str) and e.strip()]
        return []
    except Exception as e:
        print(f"    NER failed: {e}", file=sys.stderr)
        return []


def extract_triples(text: str, content_type: str, secrets: dict,
                    config: dict, named_entities: list[str],
                    client: httpx.Client) -> list[dict]:
    """Stage 2: OpenIE — extract triples conditioned on NER entities."""
    url = secrets.get("TELUS_OLLAMA_URL", "")
    key = secrets.get("TELUS_OLLAMA_KEY", "")
    model = config.get("extraction", {}).get("model", "gpt-oss:120b")

    if not url or not key:
        return []

    import re
    hint = TYPE_HINTS.get(content_type, "Extract all meaningful entities and relationships.")
    entity_list_json = json.dumps(named_entities)

    prompt = f"""Extract relationships between these entities from the text as JSON triples.

Named entities found in this text:
{entity_list_json}

Rules:
- Each triple MUST contain at least one of the named entities above as subject or object
- Relations should be verb phrases (e.g., "decided-to-use", "is-a", "relates-to")
- Keep entities concise (1-4 words)
- SKIP placeholder values like "none", "not yet", "unknown", "0", "true", "false", "null"
- {hint}
- You may discover additional entities not in the list — include them if substantive

Output ONLY valid JSON in this exact format, nothing else:
{{"triples": [["subject", "relation", "object"], ...]}}

Text:
{text}"""

    try:
        resp = client.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        raw = re.sub(r",\s*([}\]])", r"\1", raw.strip())
        import re as _re
        obj_match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if obj_match:
            raw = obj_match.group()

        decoder = json.JSONDecoder()
        try:
            data, _ = decoder.raw_decode(raw)
        except json.JSONDecodeError:
            match = _re.search(r'"triples"\s*:\s*(\[\s*\[.*?\]\s*\])', raw, _re.DOTALL)
            if match:
                arr, _ = decoder.raw_decode(match.group(1))
                data = {"triples": arr}
            else:
                raise

        triples = data.get("triples", [])
        return [{"subject": t[0], "relation": t[1], "object": t[2]} for t in triples if len(t) >= 3]
    except Exception as e:
        print(f"    Triple extraction failed: {e}", file=sys.stderr)
        return []


def insert_triple(r_conn: redis.Redis, db: str, subject: str, relation: str, obj: str,
                  source: str, timestamp: str):
    """Insert a single triple into FalkorDB."""
    cypher = """
    MERGE (s:Entity {name: $subject})
    ON CREATE SET s.created = $timestamp
    MERGE (o:Entity {name: $obj})
    ON CREATE SET o.created = $timestamp
    MERGE (s)-[r:RELATES {type: $relation}]->(o)
    ON CREATE SET r.weight = 1.0, r.source = $source, r.last_accessed = $timestamp
    ON MATCH SET r.weight = r.weight + 0.1, r.last_accessed = $timestamp
    """
    try:
        graph_query(r_conn, db, cypher, params={
            "subject": subject, "obj": obj, "relation": relation,
            "source": source, "timestamp": timestamp,
        })
    except redis.exceptions.ResponseError as e:
        print(f"    Warning: Triple insert failed ({subject}, {relation}, {obj}): {e}", file=sys.stderr)


def make_source_id(namespace: str, reference: str) -> str:
    """Create a hippo source ID from KOI namespace + reference.

    Maps e.g. 'legion.claude-youtube' + 'indydevdan/abc123' -> 'youtube:indydevdan/abc123'
    """
    prefix = NAMESPACE_SOURCE_PREFIX.get(namespace, namespace.split(".")[-1].replace("claude-", ""))
    return f"{prefix}:{reference}"


def process_bundle(bundle: dict, config: dict, secrets: dict,
                   r_conn: redis.Redis, db: str, llm_client: httpx.Client,
                   chunk_delay: float) -> tuple[int, int]:
    """Process a single KOI bundle: chunk, extract, insert into Hippo.

    Returns (triple_count, entity_count).
    """
    rid = bundle["rid"]
    namespace = bundle["namespace"]
    reference = bundle["reference"]
    text = bundle["search_text"] or ""

    if len(text) < 100:
        return (0, 0)

    content_type = NAMESPACE_CONTENT_TYPE.get(namespace, "external")
    source_id = make_source_id(namespace, reference)
    timestamp = datetime.now(timezone.utc).isoformat()

    chunk_size = config.get("extraction", {}).get("chunk_size", 2000)
    chunk_overlap = config.get("extraction", {}).get("chunk_overlap", 400)
    chunks = chunk_document(text, chunk_size, chunk_overlap)

    all_triples = []
    seen_triples = set()

    for chunk in chunks:
        # Two-stage: NER then conditioned OpenIE
        ner_entities = extract_entities_ner(chunk, content_type, secrets, config, llm_client)
        if not ner_entities:
            continue
        chunk_triples = extract_triples(chunk, content_type, secrets, config, ner_entities, llm_client)

        for t in chunk_triples:
            key = (t["subject"].lower().strip(),
                   t["relation"].lower().strip(),
                   t["object"].lower().strip())
            if key not in seen_triples:
                seen_triples.add(key)
                all_triples.append(t)

        if len(chunks) > 1:
            time.sleep(chunk_delay)

    if not all_triples:
        return (0, 0)

    # Collect unique entities
    entities = set()
    for t in all_triples:
        entities.add(t["subject"].lower().strip())
        entities.add(t["object"].lower().strip())

    # Insert triples
    for t in all_triples:
        insert_triple(
            r_conn, db,
            t["subject"].lower().strip(),
            t["relation"].lower().strip(),
            t["object"].lower().strip(),
            source_id, timestamp,
        )

    return (len(all_triples), len(entities))


def main():
    parser = argparse.ArgumentParser(description="Bridge KOI bundles to Hippo graph")
    parser.add_argument("--namespace", required=True, help="KOI namespace to process")
    parser.add_argument("--batch-size", type=int, default=25, help="Bundles per batch")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no extraction")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed RIDs")
    parser.add_argument("--reset", action="store_true", help="Clear state and reprocess all")
    parser.add_argument("--chunk-delay", type=float, default=0.3, help="Delay between chunks (rate limit)")
    args = parser.parse_args()

    # Load hippo config and secrets
    config = load_config()
    secrets = load_secrets()
    db = config["backend"]["database"]

    # Connect to FalkorDB
    r_conn = get_redis(config)
    try:
        r_conn.ping()
    except redis.ConnectionError:
        print("Error: Cannot connect to FalkorDB. Is hippo-graph running?", file=sys.stderr)
        sys.exit(1)

    # Connect to KOI PostgreSQL
    koi_conn = get_koi_conn()

    # Load state
    if args.reset:
        processed_rids = set()
        save_state(args.namespace, processed_rids)
        print(f"State reset for {args.namespace}")
    elif args.resume:
        processed_rids = load_state(args.namespace)
        print(f"Resuming: {len(processed_rids)} already processed")
    else:
        processed_rids = load_state(args.namespace)
        if processed_rids:
            print(f"Found {len(processed_rids)} previously processed. Use --reset to reprocess all.")

    # Fetch bundles
    bundles = fetch_bundles(koi_conn, args.namespace, processed_rids, args.batch_size)
    total_remaining = len(bundles)

    if total_remaining == 0:
        print(f"No unprocessed bundles in {args.namespace}")
        return

    if args.dry_run:
        print(f"[{args.namespace}] {total_remaining} bundles to process (dry run)")
        return

    print(f"[{args.namespace}] Processing {total_remaining} bundles")
    print(f"  FalkorDB: {db}, chunk_delay: {args.chunk_delay}s")

    stats = {"triples": 0, "entities": 0, "processed": 0, "errors": 0}
    start_time = time.time()
    llm_timeout = config.get("extraction", {}).get("api_timeout", 30)

    with httpx.Client(timeout=llm_timeout) as llm_client:
        for i, bundle in enumerate(bundles):
            rid = bundle["rid"]
            reference = bundle["reference"]
            try:
                triple_count, entity_count = process_bundle(
                    bundle, config, secrets, r_conn, db, llm_client, args.chunk_delay,
                )
                stats["triples"] += triple_count
                stats["entities"] += entity_count
                stats["processed"] += 1

                # Checkpoint after every bundle
                processed_rids.add(rid)
                save_state(args.namespace, processed_rids)

                elapsed = time.time() - start_time
                rate = stats["processed"] / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i+1}/{total_remaining}] {reference}: "
                    f"{triple_count} triples, {entity_count} entities "
                    f"({rate:.2f} bundles/s, total: {stats['triples']} triples)"
                )
            except Exception as e:
                stats["errors"] += 1
                processed_rids.add(rid)  # Mark as processed to skip on resume
                save_state(args.namespace, processed_rids)
                print(f"  [{i+1}/{total_remaining}] ERROR on {reference}: {e}")

    elapsed = time.time() - start_time
    print(f"\nComplete: {stats['processed']} bundles, {stats['triples']} triples, "
          f"{stats['entities']} entities, {stats['errors']} errors in {elapsed:.1f}s")
    print(f"State saved to {STATE_DIR}")
    print(f"\nNext steps:")
    print(f"  cd {HIPPO_SCRIPTS}")
    print(f"  uv run hippo_typify.py    # Type new entities")
    print(f"  uv run hippo_consolidate.py  # Merge duplicates")
    koi_conn.close()


if __name__ == "__main__":
    main()
