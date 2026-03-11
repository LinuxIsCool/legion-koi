#!/usr/bin/env -S python3 -u
"""Ingest Claude Web export (conversations, projects, memories) into PostgreSQL.

Bulk direct-to-PostgreSQL ingestion — same pattern as Phase 2b/2c.
Each conversation → 1 bundle in legion.claude-conversation.
Each project → 1 bundle in legion.claude-project.
Memories → 1 bundle in legion.claude-memory.

Usage:
    uv run python scripts/ingest_claude_web.py
    uv run python scripts/ingest_claude_web.py --dry-run
    uv run python scripts/ingest_claude_web.py --only conversations
    uv run python scripts/ingest_claude_web.py --only projects
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

from legion_koi.storage.postgres import _extract_search_text

DSN = "postgresql://shawn@localhost/personal_koi"
DUMP_DIR = Path("~/Workspace/knowledge-archive/claude-web-dump").expanduser()

# PostgreSQL tsvector max is 1MB; cap well below
_MAX_SEARCH_TEXT = 500_000


def slugify(name: str) -> str:
    """Convert a conversation/project name to a URL-safe slug."""
    slug = name.lower().strip()
    slug = slug.replace(" ", "-").replace("_", "-")
    # Keep alphanumeric, hyphens, dots
    slug = "".join(c for c in slug if c.isalnum() or c in "-.")
    # Collapse multiple hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    return slug[:80] or "untitled"


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def extract_conversation_search_text(convo: dict) -> str:
    """Build search text from a Claude Web conversation."""
    parts = []
    name = convo.get("name") or ""
    if name:
        parts.append(name)
    summary = convo.get("summary") or ""
    if summary:
        parts.append(summary)
    for msg in convo.get("chat_messages", []):
        text = msg.get("text") or ""
        if text:
            parts.append(text)
    return "\n".join(parts)[:_MAX_SEARCH_TEXT]


def extract_project_search_text(project: dict) -> str:
    """Build search text from a Claude Web project."""
    parts = []
    name = project.get("name") or ""
    if name:
        parts.append(name)
    desc = project.get("description") or ""
    if desc:
        parts.append(desc)
    prompt = project.get("prompt_template") or ""
    if prompt:
        parts.append(prompt)
    for doc in project.get("docs", []):
        filename = doc.get("filename") or ""
        if filename:
            parts.append(filename)
        content = doc.get("content") or ""
        if content:
            parts.append(content)
    return "\n".join(parts)[:_MAX_SEARCH_TEXT]


def build_conversation_contents(convo: dict) -> dict:
    """Build bundle contents for a conversation."""
    messages = []
    for msg in convo.get("chat_messages", []):
        m = {
            "uuid": msg.get("uuid"),
            "sender": msg.get("sender"),
            "text": msg.get("text") or "",
            "created_at": msg.get("created_at"),
        }
        # Include attachments/files metadata if present
        attachments = msg.get("attachments") or []
        if attachments:
            m["attachments"] = attachments
        files = msg.get("files") or []
        if files:
            m["files"] = files
        messages.append(m)

    return {
        "uuid": convo.get("uuid"),
        "name": convo.get("name") or "Untitled",
        "summary": convo.get("summary"),
        "created_at": convo.get("created_at"),
        "updated_at": convo.get("updated_at"),
        "message_count": len(messages),
        "chat_messages": messages,
    }


def build_project_contents(project: dict) -> dict:
    """Build bundle contents for a project."""
    docs = []
    for doc in project.get("docs", []):
        docs.append({
            "uuid": doc.get("uuid"),
            "filename": doc.get("filename"),
            "content": doc.get("content") or "",
            "created_at": doc.get("created_at"),
        })

    return {
        "uuid": project.get("uuid"),
        "name": project.get("name") or "Untitled",
        "description": project.get("description") or "",
        "is_private": project.get("is_private", True),
        "prompt_template": project.get("prompt_template") or "",
        "created_at": project.get("created_at"),
        "updated_at": project.get("updated_at"),
        "doc_count": len(docs),
        "docs": docs,
    }


def ingest_conversations(conn, dry_run: bool = False):
    """Ingest conversations.json → legion.claude-conversation bundles."""
    path = DUMP_DIR / "conversations.json"
    if not path.exists():
        print(f"  File not found: {path}")
        return

    with open(path) as f:
        convos = json.load(f)

    print(f"  Conversations: {len(convos)}")

    # Check existing
    existing = set()
    rows = conn.execute(
        "SELECT rid FROM bundles WHERE namespace = 'legion.claude-conversation'"
    ).fetchall()
    existing = {r["rid"] for r in rows}
    print(f"  Already ingested: {len(existing)}")

    stats = {"inserted": 0, "updated": 0, "skipped": 0, "empty": 0}

    for i, convo in enumerate(convos):
        uuid = convo.get("uuid", "")
        name = convo.get("name") or "Untitled"
        messages = convo.get("chat_messages", [])

        if not messages:
            stats["empty"] += 1
            continue

        # RID: orn:legion.claude-conversation:{date}/{uuid-prefix}-{slug}
        created = convo.get("created_at", "")[:10]
        uuid_prefix = uuid[:8] if uuid else "0000"
        slug = slugify(name)
        reference = f"{created}/{uuid_prefix}-{slug}"
        rid = f"orn:legion.claude-conversation:{reference}"

        contents = build_conversation_contents(convo)
        search_text = extract_conversation_search_text(convo)
        content_hash = sha256(json.dumps(contents, sort_keys=True, default=str))

        if dry_run:
            stats["inserted"] += 1
            continue

        created_at = convo.get("created_at")
        if created_at:
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

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
                (rid, "legion.claude-conversation", reference, Jsonb(contents),
                 search_text, content_hash, created_at),
            )
            if rid in existing:
                stats["updated"] += 1
            else:
                stats["inserted"] += 1
        except Exception as e:
            print(f"  Error on {rid}: {e}")
            stats["skipped"] += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(convos)}] {stats['inserted']} inserted, {stats['updated']} updated")

    print(f"  Done: {stats['inserted']} inserted, {stats['updated']} updated, "
          f"{stats['empty']} empty, {stats['skipped']} errors")


def ingest_projects(conn, dry_run: bool = False):
    """Ingest projects.json → legion.claude-project bundles."""
    path = DUMP_DIR / "projects.json"
    if not path.exists():
        print(f"  File not found: {path}")
        return

    with open(path) as f:
        projects = json.load(f)

    print(f"  Projects: {len(projects)}")

    existing = set()
    rows = conn.execute(
        "SELECT rid FROM bundles WHERE namespace = 'legion.claude-project'"
    ).fetchall()
    existing = {r["rid"] for r in rows}
    print(f"  Already ingested: {len(existing)}")

    stats = {"inserted": 0, "updated": 0, "skipped": 0}

    for project in projects:
        name = project.get("name") or "Untitled"
        uuid = project.get("uuid", "")
        uuid_prefix = uuid[:8] if uuid else "0000"
        slug = slugify(name)
        reference = f"{uuid_prefix}-{slug}"
        rid = f"orn:legion.claude-project:{reference}"

        contents = build_project_contents(project)
        search_text = extract_project_search_text(project)
        content_hash = sha256(json.dumps(contents, sort_keys=True, default=str))

        if dry_run:
            stats["inserted"] += 1
            continue

        created_at = project.get("created_at")
        if created_at:
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

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
                (rid, "legion.claude-project", reference, Jsonb(contents),
                 search_text, content_hash, created_at),
            )
            if rid in existing:
                stats["updated"] += 1
            else:
                stats["inserted"] += 1
        except Exception as e:
            print(f"  Error on {rid}: {e}")
            stats["skipped"] += 1

    print(f"  Done: {stats['inserted']} inserted, {stats['updated']} updated, {stats['skipped']} errors")


def ingest_memories(conn, dry_run: bool = False):
    """Ingest memories.json → legion.claude-memory bundle."""
    path = DUMP_DIR / "memories.json"
    if not path.exists():
        print(f"  File not found: {path}")
        return

    with open(path) as f:
        memories = json.load(f)

    if not memories:
        print("  No memories found")
        return

    mem = memories[0] if isinstance(memories, list) else memories
    conversations_memory = mem.get("conversations_memory", "")
    project_memories = mem.get("project_memories", {})

    print(f"  Conversations memory: {len(conversations_memory)} chars")
    print(f"  Project memories: {len(project_memories)} entries")

    contents = {
        "conversations_memory": conversations_memory,
        "project_memories": project_memories,
        "account_uuid": mem.get("account_uuid"),
    }

    search_text = conversations_memory
    for _uuid, text in project_memories.items():
        if isinstance(text, str):
            search_text += "\n" + text
    search_text = search_text[:_MAX_SEARCH_TEXT]

    rid = "orn:legion.claude-memory:claude-web-export"
    reference = "claude-web-export"
    content_hash = sha256(json.dumps(contents, sort_keys=True, default=str))

    if dry_run:
        print("  [dry-run] Would insert 1 memory bundle")
        return

    conn.execute(
        """
        INSERT INTO bundles (rid, namespace, reference, contents, search_text, sha256_hash, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        ON CONFLICT (rid) DO UPDATE SET
            contents = EXCLUDED.contents,
            search_text = EXCLUDED.search_text,
            sha256_hash = EXCLUDED.sha256_hash,
            updated_at = NOW()
        """,
        (rid, "legion.claude-memory", reference, Jsonb(contents),
         search_text, content_hash),
    )
    print("  Done: 1 memory bundle upserted")


def main():
    parser = argparse.ArgumentParser(description="Ingest Claude Web export into legion-koi")
    parser.add_argument("--dry-run", action="store_true", help="Count without writing")
    parser.add_argument("--only", choices=["conversations", "projects", "memories"],
                        help="Only ingest this type")
    args = parser.parse_args()

    print(f"Source: {DUMP_DIR}")
    print(f"DSN: {DSN}")
    if args.dry_run:
        print("*** DRY RUN ***")

    conn = get_conn()
    start = time.time()

    if not args.only or args.only == "conversations":
        print("\n=== Conversations ===")
        ingest_conversations(conn, dry_run=args.dry_run)

    if not args.only or args.only == "projects":
        print("\n=== Projects ===")
        ingest_projects(conn, dry_run=args.dry_run)

    if not args.only or args.only == "memories":
        print("\n=== Memories ===")
        ingest_memories(conn, dry_run=args.dry_run)

    elapsed = time.time() - start

    # Final count
    row = conn.execute(
        """
        SELECT namespace, count(*) AS cnt FROM bundles
        WHERE namespace IN ('legion.claude-conversation', 'legion.claude-project', 'legion.claude-memory')
        GROUP BY namespace ORDER BY namespace
        """
    ).fetchall()
    print(f"\n=== Summary ({elapsed:.1f}s) ===")
    for r in row:
        print(f"  {r['namespace']}: {r['cnt']}")
    total = conn.execute("SELECT count(*) AS cnt FROM bundles").fetchone()
    print(f"  Total bundles: {total['cnt']}")

    conn.close()


if __name__ == "__main__":
    main()
