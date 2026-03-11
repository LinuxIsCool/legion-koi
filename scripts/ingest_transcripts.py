#!/usr/bin/env -S python3 -u
"""Ingest native Claude Code session transcripts into PostgreSQL.

Each .jsonl file = one session = one bundle in legion.claude-code.
Extracts user messages + assistant text blocks as search text.
Skips tool_use blocks, file-history-snapshot, and progress events.

Usage:
    uv run python scripts/ingest_transcripts.py
    uv run python scripts/ingest_transcripts.py --dry-run
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

sys.path.insert(0, "src")

DSN = "postgresql://shawn@localhost/personal_koi"
TRANSCRIPT_DIR = Path("~/.claude/projects/-home-shawn").expanduser()

# PostgreSQL tsvector max is 1MB; cap well below
_MAX_SEARCH_TEXT = 500_000


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def parse_transcript(path: Path) -> dict:
    """Parse a .jsonl transcript file into structured contents."""
    session_id = path.stem  # UUID filename
    lines = path.read_text().splitlines()

    metadata = {
        "session_id": session_id,
        "cwd": None,
        "version": None,
        "git_branch": None,
        "started_at": None,
        "ended_at": None,
    }

    user_messages = []
    assistant_texts = []
    tool_uses = []
    total_lines = len(lines)

    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type", "")
        timestamp = obj.get("timestamp")

        # Extract session metadata from first relevant line
        if metadata["cwd"] is None and obj.get("cwd"):
            metadata["cwd"] = obj["cwd"]
            metadata["version"] = obj.get("version")
            metadata["git_branch"] = obj.get("gitBranch")
        if timestamp:
            if metadata["started_at"] is None:
                metadata["started_at"] = timestamp
            metadata["ended_at"] = timestamp

        if msg_type == "user":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                # Skip system-generated messages (hooks, commands)
                if content.startswith("<local-command-caveat>"):
                    continue
                user_messages.append(content)
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text and not text.startswith("<local-command-caveat>"):
                            user_messages.append(text)

        elif msg_type == "assistant":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                assistant_texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            assistant_texts.append(text)
                    elif block.get("type") == "tool_use":
                        tool_uses.append({
                            "name": block.get("name", ""),
                            "id": block.get("id", ""),
                        })

    # Build a summary from the first few user messages
    summary_parts = user_messages[:5]
    summary = " | ".join(s[:200] for s in summary_parts)[:500]

    return {
        "metadata": metadata,
        "contents": {
            "session_id": session_id,
            "cwd": metadata["cwd"],
            "version": metadata["version"],
            "git_branch": metadata["git_branch"],
            "started_at": metadata["started_at"],
            "ended_at": metadata["ended_at"],
            "summary": summary,
            "user_message_count": len(user_messages),
            "assistant_text_count": len(assistant_texts),
            "tool_use_count": len(tool_uses),
            "total_lines": total_lines,
            "tools_used": list({t["name"] for t in tool_uses}),
        },
        "search_text": build_search_text(metadata, user_messages, assistant_texts),
    }


def build_search_text(metadata: dict, user_messages: list, assistant_texts: list) -> str:
    """Build search text from user messages + assistant text blocks."""
    parts = []

    cwd = metadata.get("cwd") or ""
    if cwd:
        parts.append(cwd)

    # User messages are the most valuable for search — they represent intent
    for msg in user_messages:
        parts.append(msg)

    # Assistant text blocks capture explanations and decisions
    for text in assistant_texts:
        parts.append(text)

    return "\n".join(parts)[:_MAX_SEARCH_TEXT]


def main():
    parser = argparse.ArgumentParser(description="Ingest Claude Code transcripts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jsonl_files = sorted(TRANSCRIPT_DIR.glob("*.jsonl"))
    print(f"Transcript files: {len(jsonl_files)}")
    print(f"Source: {TRANSCRIPT_DIR}")

    if args.dry_run:
        print("*** DRY RUN ***")

    conn = get_conn()

    # Check existing
    existing = set()
    rows = conn.execute(
        "SELECT rid FROM bundles WHERE namespace = 'legion.claude-code'"
    ).fetchall()
    existing = {r["rid"] for r in rows}
    print(f"Already ingested: {len(existing)}")

    stats = {"inserted": 0, "updated": 0, "skipped": 0, "empty": 0}
    start = time.time()

    for i, path in enumerate(jsonl_files):
        session_id = path.stem

        try:
            parsed = parse_transcript(path)
        except Exception as e:
            print(f"  Error parsing {path.name}: {e}")
            stats["skipped"] += 1
            continue

        contents = parsed["contents"]
        search_text = parsed["search_text"]

        if contents["user_message_count"] == 0:
            stats["empty"] += 1
            continue

        # RID: orn:legion.claude-code:{date}/{session_id_prefix}
        started = (contents.get("started_at") or "")[:10]
        reference = f"{started}/{session_id[:8]}"
        rid = f"orn:legion.claude-code:{reference}"
        content_hash = sha256(json.dumps(contents, sort_keys=True, default=str))

        if args.dry_run:
            stats["inserted"] += 1
            print(f"  [{i+1}] {rid}: {contents['user_message_count']} user msgs, "
                  f"{contents['assistant_text_count']} assistant texts, "
                  f"{contents['tool_use_count']} tool uses")
            continue

        started_at = contents.get("started_at")
        if started_at:
            try:
                started_at = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            except Exception:
                started_at = datetime.now(timezone.utc)
        else:
            started_at = datetime.now(timezone.utc)

        try:
            conn.execute(
                """
                INSERT INTO bundles (rid, namespace, reference, contents, search_text, sha256_hash, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (rid) DO UPDATE SET
                    contents = EXCLUDED.contents,
                    search_text = EXCLUDED.search_text,
                    sha256_hash = EXCLUDED.sha256_hash,
                    updated_at = NOW()
                """,
                (rid, "legion.claude-code", reference, Jsonb(contents),
                 search_text, content_hash, started_at),
            )
            if rid in existing:
                stats["updated"] += 1
            else:
                stats["inserted"] += 1
        except Exception as e:
            print(f"  Error on {rid}: {e}")
            stats["skipped"] += 1

    elapsed = time.time() - start

    # Final count
    row = conn.execute(
        "SELECT count(*) AS cnt FROM bundles WHERE namespace = 'legion.claude-code'"
    ).fetchone()
    total = conn.execute("SELECT count(*) AS cnt FROM bundles").fetchone()

    print(f"\n=== Summary ({elapsed:.1f}s) ===")
    print(f"  Inserted: {stats['inserted']}, Updated: {stats['updated']}, "
          f"Empty: {stats['empty']}, Errors: {stats['skipped']}")
    print(f"  legion.claude-code: {row['cnt']}")
    print(f"  Total bundles: {total['cnt']}")

    conn.close()


if __name__ == "__main__":
    main()
