#!/usr/bin/env -S python3 -u
"""Ingest GitHub repos into PostgreSQL.

One bundle per repo in legion.claude-github.
Uses `gh` CLI (must be authenticated) via subprocess.

Usage:
    uv run python scripts/ingest_github.py --owner <github-user>
    uv run python scripts/ingest_github.py --owner <github-user> --dry-run
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

sys.path.insert(0, "src")

from legion_koi.storage.postgres import _extract_search_text

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")
NAMESPACE = "legion.claude-github"


def sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def gh_json(args: list[str]) -> dict | list | None:
    """Run a gh CLI command and parse JSON output."""
    try:
        result = subprocess.run(
            ["gh"] + args,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


REPO_FIELDS = (
    "name,description,url,primaryLanguage,repositoryTopics,isPrivate,"
    "createdAt,updatedAt,defaultBranchRef,stargazerCount,forkCount,issues"
)


def fetch_repos(owner: str) -> list[dict]:
    """Fetch all repos for an owner via gh repo list.

    gh handles pagination internally — --limit sets total count.
    GitHub API max is 1000 per listing; set to 4000 to be safe.
    """
    result = subprocess.run(
        ["gh", "repo", "list", owner, "--json", REPO_FIELDS,
         "--limit", "4000"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    return []


def fetch_readme(owner: str, repo_name: str) -> str:
    """Fetch README content for a repo. Handles rate limits with backoff."""
    for attempt in range(3):
        result = subprocess.run(
            ["gh", "api", f"repos/{owner}/{repo_name}/readme"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            if "rate limit" in result.stderr.lower() or "403" in result.stderr:
                wait = 5 * (attempt + 1)
                print(f"    Rate limited fetching README for {repo_name}, waiting {wait}s...")
                time.sleep(wait)
                continue
            return ""
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return ""
        if data and data.get("content"):
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except Exception:
                return ""
        return ""
    return ""


def build_contents(repo: dict, owner: str, readme: str) -> dict:
    """Build bundle contents from repo data."""
    # Handle both API formats (direct API vs gh repo list)
    name = repo.get("name") or ""
    description = repo.get("description") or ""
    url = repo.get("url") or repo.get("html_url") or ""

    # Languages
    language = repo.get("language") or ""
    primary_lang = repo.get("primaryLanguage")
    if primary_lang and isinstance(primary_lang, dict):
        language = primary_lang.get("name") or ""

    # Topics
    topics = repo.get("topics") or []
    repo_topics = repo.get("repositoryTopics")
    if repo_topics and isinstance(repo_topics, list):
        for t in repo_topics:
            if isinstance(t, dict):
                topic_name = t.get("name") or ""
                if not topic_name and "topic" in t:
                    topic_name = t["topic"].get("name", "")
                if topic_name:
                    topics.append(topic_name)

    # Private
    is_private = repo.get("private") or repo.get("isPrivate") or False

    # Dates
    created_at = repo.get("created_at") or repo.get("createdAt") or ""
    updated_at = repo.get("updated_at") or repo.get("updatedAt") or ""

    # Branch
    default_branch = repo.get("default_branch") or ""
    branch_ref = repo.get("defaultBranchRef")
    if branch_ref and isinstance(branch_ref, dict):
        default_branch = branch_ref.get("name") or ""

    # Counts
    stars = repo.get("stargazers_count") or repo.get("stargazerCount") or 0
    forks = repo.get("forks_count") or repo.get("forkCount") or 0
    open_issues = repo.get("open_issues_count") or 0
    issues = repo.get("issues")
    if issues and isinstance(issues, dict):
        open_issues = issues.get("totalCount", 0)

    return {
        "name": name,
        "description": description,
        "url": url,
        "language": language,
        "topics": topics,
        "is_private": is_private,
        "created_at": created_at,
        "updated_at": updated_at,
        "default_branch": default_branch,
        "stars": stars,
        "forks": forks,
        "open_issues_count": open_issues,
        "readme_content": readme,
        "owner": owner,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest GitHub repos into legion-koi")
    parser.add_argument("--owner", required=True, help="GitHub owner/org")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Owner: {args.owner}")
    print(f"Namespace: {NAMESPACE}")

    # Fetch repos
    print("Fetching repos...")
    repos = fetch_repos(args.owner)
    print(f"Found: {len(repos)} repos")

    if not repos:
        print("No repos found. Is `gh` authenticated?")
        return

    if args.dry_run:
        for repo in repos:
            name = repo.get("name") or ""
            desc = repo.get("description") or ""
            print(f"  {name}: {desc[:80]}")
        return

    conn = get_conn()

    # Check existing RIDs for insert vs update tracking
    existing = set()
    rows = conn.execute(
        "SELECT rid FROM bundles WHERE namespace = %s", (NAMESPACE,)
    ).fetchall()
    existing = {r["rid"] for r in rows}
    print(f"Already ingested: {len(existing)}")

    stats = {"inserted": 0, "updated": 0, "errors": 0}
    start = time.time()
    batch = []
    BATCH_SIZE = 50

    for i, repo in enumerate(repos):
        name = repo.get("name") or ""
        if not name:
            continue

        # Fetch README
        readme = fetch_readme(args.owner, name)

        contents = build_contents(repo, args.owner, readme)
        reference = f"{args.owner}/{name}"
        rid = f"orn:{NAMESPACE}:{reference}"
        content_hash = sha256(json.dumps(contents, sort_keys=True, default=str))
        search_text = _extract_search_text(NAMESPACE, contents)

        created_at = contents.get("created_at")
        if created_at:
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

        batch.append((rid, reference, contents, search_text, content_hash, created_at, rid in existing))

        if len(batch) >= BATCH_SIZE or i == len(repos) - 1:
            try:
                with conn.transaction():
                    for rid, ref, cont, stxt, chash, cat, is_update in batch:
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
                            (rid, NAMESPACE, ref, Jsonb(cont), stxt, chash, cat),
                        )
                        if is_update:
                            stats["updated"] += 1
                        else:
                            stats["inserted"] += 1
            except Exception as e:
                print(f"  Batch error: {e}")
                stats["errors"] += len(batch)
            batch = []

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(repos)}] {stats['inserted']} inserted, {stats['updated']} updated")

    elapsed = time.time() - start

    # Final count
    row = conn.execute(
        "SELECT count(*) AS cnt FROM bundles WHERE namespace = %s", (NAMESPACE,)
    ).fetchone()
    total = conn.execute("SELECT count(*) AS cnt FROM bundles").fetchone()

    print(f"\n=== Summary ({elapsed:.1f}s) ===")
    print(f"  Inserted: {stats['inserted']}, Updated: {stats['updated']}, Errors: {stats['errors']}")
    print(f"  {NAMESPACE}: {row['cnt']}")
    print(f"  Total bundles: {total['cnt']}")

    conn.close()


if __name__ == "__main__":
    main()
