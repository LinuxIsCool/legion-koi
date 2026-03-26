#!/usr/bin/env -S python3 -u
"""Detect cross-references between YouTube videos and GitHub repos.

Scans YouTube transcripts for mentions of GitHub repo names and URLs,
then stores cross-references as KOI bundle_entities (type: ProjectReference).

Usage:
    uv run python scripts/detect_cross_references.py --dry-run
    uv run python scripts/detect_cross_references.py
"""

import argparse
import os
import re
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, "src")

from legion_koi.storage.postgres import PostgresStorage

DSN = os.environ.get("LEGION_KOI_DSN", "postgresql://localhost/personal_koi")


def get_conn():
    return psycopg.connect(DSN, row_factory=dict_row, autocommit=True)


def fetch_github_repos(conn, owner_filter: str | None = None) -> list[dict]:
    """Get GitHub repo bundles for matching.

    If owner_filter is set, only match repos from that owner (e.g. 'disler').
    """
    query = """
        SELECT rid, reference, contents->>'name' as name,
               contents->>'full_name' as full_name,
               contents->>'description' as description
        FROM bundles
        WHERE namespace = 'legion.claude-github'
        ORDER BY reference
    """
    rows = conn.execute(query).fetchall()
    if owner_filter:
        rows = [r for r in rows if r["full_name"] and r["full_name"].startswith(f"{owner_filter}/")]
    return rows


def fetch_docked_repos(owner: str) -> list[dict]:
    """Get docked repos from filesystem (dock generated SKILL.md files)."""
    dock_dir = Path.home() / ".claude" / "local" / "dock" / "generated" / owner
    if not dock_dir.exists():
        return []

    repos = []
    for skill_file in dock_dir.glob("*/SKILL.md"):
        repo_name = skill_file.parent.name
        repos.append({
            "rid": f"dock:{owner}/{repo_name}",
            "reference": f"{owner}/{repo_name}",
            "name": repo_name,
            "full_name": f"{owner}/{repo_name}",
            "description": "",
        })
    return repos


def fetch_youtube_bundles(conn) -> list[dict]:
    """Get all YouTube bundles with search_text."""
    return conn.execute(
        """
        SELECT rid, reference, contents->>'title' as title, search_text
        FROM bundles
        WHERE namespace = 'legion.claude-youtube'
          AND length(search_text) > 50
        ORDER BY created_at
        """
    ).fetchall()


def build_matchers(repos: list[dict]) -> list[dict]:
    """Build regex matchers for each repo."""
    matchers = []
    for repo in repos:
        name = repo["name"] or ""
        full_name = repo["full_name"] or ""
        if not name:
            continue

        # Patterns to match:
        # 1. Exact repo name (word-bounded, case-insensitive)
        # 2. GitHub URL: github.com/owner/repo
        # 3. Hyphenated name as spoken words: "single file agents" for "single-file-agents"
        patterns = []

        # Exact name match (word-bounded)
        escaped = re.escape(name)
        patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))

        # GitHub URL
        if full_name:
            escaped_full = re.escape(full_name)
            patterns.append(re.compile(rf"github\.com/{escaped_full}", re.IGNORECASE))

        # Hyphen-to-space variant (e.g., "single-file-agents" -> "single file agents")
        if "-" in name:
            spaced = name.replace("-", r"[\s\-]")
            patterns.append(re.compile(rf"\b{spaced}\b", re.IGNORECASE))

        matchers.append({
            "repo_rid": repo["rid"],
            "repo_name": name,
            "full_name": full_name,
            "patterns": patterns,
        })

    return matchers


def detect_references(video: dict, matchers: list[dict]) -> list[dict]:
    """Find repo references in a video's search_text."""
    text = video["search_text"] or ""
    if not text:
        return []

    refs = []
    for matcher in matchers:
        for pattern in matcher["patterns"]:
            if pattern.search(text):
                refs.append({
                    "repo_name": matcher["repo_name"],
                    "repo_rid": matcher["repo_rid"],
                    "full_name": matcher["full_name"],
                })
                break  # One match per repo is enough

    return refs


def main():
    parser = argparse.ArgumentParser(description="Detect video-to-repo cross-references")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without storing")
    parser.add_argument("--owner", default="disler", help="GitHub owner to filter repos (default: disler)")
    parser.add_argument("--min-name-length", type=int, default=5, help="Skip repo names shorter than this")
    args = parser.parse_args()

    conn = get_conn()
    storage = PostgresStorage(dsn=DSN)
    storage.initialize()

    # Try KOI first, then fall back to docked repos
    repos = fetch_github_repos(conn, owner_filter=args.owner if args.owner else None)
    if not repos and args.owner:
        repos = fetch_docked_repos(args.owner)
    # Filter out very short/generic names
    repos = [r for r in repos if r["name"] and len(r["name"]) >= args.min_name_length]
    videos = fetch_youtube_bundles(conn)

    print(f"Repos: {len(repos)}, Videos: {len(videos)}")

    if not repos:
        print("No GitHub repos found. Dock some repos first.")
        return

    matchers = build_matchers(repos)
    print(f"Matchers built for {len(matchers)} repos")

    total_refs = 0
    video_with_refs = 0

    for video in videos:
        refs = detect_references(video, matchers)
        if not refs:
            continue

        video_with_refs += 1
        ref_names = [r["repo_name"] for r in refs]
        print(f"  {video['reference']}: {video['title']}")
        print(f"    -> {', '.join(ref_names)}")

        if not args.dry_run:
            # Add ProjectReference entities without replacing existing extractions.
            # Uses raw SQL to INSERT entities + link without deleting existing.
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)

            for ref in refs:
                name = ref["full_name"] or ref["repo_name"]
                name_normalized = ref["repo_name"].lower()

                # Upsert entity
                row = conn.execute(
                    """
                    INSERT INTO entities (name, entity_type, supertype, name_normalized,
                                          first_seen, last_seen, mention_count)
                    VALUES (%s, 'ProjectReference', 'artifact', %s, %s, %s, 1)
                    ON CONFLICT (name_normalized, entity_type) DO UPDATE SET
                        last_seen = %s, mention_count = entities.mention_count + 1
                    RETURNING entity_id
                    """,
                    (name, name_normalized, now, now, now),
                ).fetchone()
                entity_id = row["entity_id"]

                # Link to bundle (skip if already linked)
                conn.execute(
                    """
                    INSERT INTO bundle_entities (rid, entity_id, confidence)
                    VALUES (%s, %s, 0.9)
                    ON CONFLICT (rid, entity_id) DO NOTHING
                    """,
                    (video["rid"], entity_id),
                )
                total_refs += 1

        total_refs += len(refs) if args.dry_run else 0

    print(f"\nResults: {video_with_refs} videos with references, {total_refs} cross-references")
    if args.dry_run:
        print("(dry run — nothing stored)")

    conn.close()
    storage.close()


if __name__ == "__main__":
    main()
