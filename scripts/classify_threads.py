"""Classify Telegram threads into relevance tiers for message filtering.

Generates a YAML-ready report for Shawn to review. Output goes to stdout.
Shawn moves lurk groups between T3 (include) and T4 (skip), then the result
populates config.yaml overrides.

Usage:
    cd ~/legion-koi && uv run python scripts/classify_threads.py
"""

import sqlite3
import sys
from pathlib import Path

# Shawn's Telegram sender ID
SELF_SENDER_ID = "telegram:user:1441369482"

# Domain keywords that flag a lurk group as potentially relevant
DOMAIN_KEYWORDS = [
    "regen", "token", "engineering", "bcrg", "commons", "gaia", "refi",
    "defi", "koi", "cadcad", "praise", "tec", "inverter", "biofi",
    "mycofi", "indigenomics", "reforestation", "permaculture", "dao",
    "governance", "impact", "climate", "carbon", "biodiversity",
    "stewardship", "land", "water", "food", "seed", "soil",
    "regenerative", "web3", "funding", "grants", "open source",
    "decentralized", "cooperative", "mutual", "solidarity",
    "bioregion", "watershed", "ecology", "circular",
]

DB_PATH = Path("~/.claude/local/messages/messages.db").expanduser()


def main():
    if not DB_PATH.exists():
        print(f"ERROR: messages.db not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Get all Telegram threads with type and message counts
    rows = conn.execute("""
        SELECT
            t.id AS thread_id,
            t.title,
            t.thread_type,
            COUNT(m.id) AS msg_count,
            SUM(CASE WHEN m.sender_id = ? THEN 1 ELSE 0 END) AS shawn_posts
        FROM threads t
        LEFT JOIN messages m ON m.thread_id = t.id
        WHERE t.platform = 'telegram'
        GROUP BY t.id
        ORDER BY msg_count DESC
    """, (SELF_SENDER_ID,)).fetchall()

    conn.close()

    # Classify
    t1_dms = []      # DMs — always ingest
    t2_active = []    # Groups where Shawn posted
    t3_candidates = []  # Lurk groups matching domain keywords
    t4_candidates = []  # Lurk groups, no keyword match

    for row in rows:
        thread_id = row["thread_id"]
        title = row["title"] or "(untitled)"
        thread_type = row["thread_type"] or "unknown"
        msg_count = row["msg_count"] or 0
        shawn_posts = row["shawn_posts"] or 0

        if thread_type in ("dm", "private"):
            t1_dms.append({
                "thread_id": thread_id,
                "title": title,
                "msg_count": msg_count,
                "shawn_posts": shawn_posts,
            })
            continue

        if shawn_posts > 0:
            t2_active.append({
                "thread_id": thread_id,
                "title": title,
                "msg_count": msg_count,
                "shawn_posts": shawn_posts,
            })
            continue

        # Lurk group — check domain keywords
        title_lower = title.lower()
        matches = [kw for kw in DOMAIN_KEYWORDS if kw in title_lower]
        if matches:
            t3_candidates.append({
                "thread_id": thread_id,
                "title": title,
                "msg_count": msg_count,
                "keyword_match": ", ".join(matches),
            })
        else:
            t4_candidates.append({
                "thread_id": thread_id,
                "title": title,
                "msg_count": msg_count,
            })

    # Output
    total_msgs = sum(r["msg_count"] or 0 for r in rows)
    print(f"# Thread Classification Report")
    print(f"# Total Telegram threads: {len(rows)}")
    print(f"# Total Telegram messages: {total_msgs:,}")
    print()

    # T1: DMs
    t1_msgs = sum(d["msg_count"] for d in t1_dms)
    print(f"# === T1: DMs (always ingest) — {len(t1_dms)} threads, {t1_msgs:,} messages ===")
    print(f"# (No config needed — DMs are auto-included)")
    print()

    # T2: Active groups
    t2_msgs = sum(d["msg_count"] for d in t2_active)
    print(f"# === T2: Active groups (Shawn posted) — {len(t2_active)} threads, {t2_msgs:,} messages ===")
    print(f"# (No config needed — auto-included by participation)")
    for g in t2_active:
        print(f"#   {g['thread_id']}  # {g['title']} ({g['msg_count']:,} msgs, {g['shawn_posts']} posts)")
    print()

    # T3 candidates: lurk groups with keyword match
    t3_msgs = sum(d["msg_count"] for d in t3_candidates)
    print(f"# === T3 CANDIDATES: Lurk groups with domain keyword match — {len(t3_candidates)} threads, {t3_msgs:,} messages ===")
    print(f"# Review: move to message_thread_includes to ingest, or leave out to skip")
    print(f"message_thread_includes:")
    for g in t3_candidates:
        print(f'  - "{g["thread_id"]}"  # {g["title"]} ({g["msg_count"]:,} msgs) [keywords: {g["keyword_match"]}]')
    print()

    # T4 candidates: lurk groups, no keyword match
    t4_msgs = sum(d["msg_count"] for d in t4_candidates)
    print(f"# === T4 CANDIDATES: Lurk groups, no keyword match — {len(t4_candidates)} threads, {t4_msgs:,} messages ===")
    print(f"# These will be SKIPPED unless moved to message_thread_includes above")
    for g in t4_candidates:
        print(f'#   "{g["thread_id"]}"  # {g["title"]} ({g["msg_count"]:,} msgs)')
    print()

    # Suggested excludes: very high-volume groups even if Shawn posted
    HIGH_VOLUME_THRESHOLD = 10_000
    high_volume = [g for g in t2_active if g["msg_count"] >= HIGH_VOLUME_THRESHOLD and g["shawn_posts"] < 50]
    if high_volume:
        print(f"# === SUGGESTED EXCLUDES: High-volume groups with minimal participation ===")
        print(f"message_thread_excludes:")
        for g in high_volume:
            print(f'  - "{g["thread_id"]}"  # {g["title"]} ({g["msg_count"]:,} msgs, only {g["shawn_posts"]} posts)')
    print()

    # Summary
    print(f"# === SUMMARY ===")
    print(f"# T1 (DMs):              {len(t1_dms):>4} threads, {t1_msgs:>10,} messages")
    print(f"# T2 (Active groups):    {len(t2_active):>4} threads, {t2_msgs:>10,} messages")
    print(f"# T3 (Lurk + keywords):  {len(t3_candidates):>4} threads, {t3_msgs:>10,} messages")
    print(f"# T4 (Lurk, skip):       {len(t4_candidates):>4} threads, {t4_msgs:>10,} messages")
    print(f"#")
    print(f"# After filtering (T1+T2+T3, minus excludes): estimate depends on your review")


if __name__ == "__main__":
    main()
