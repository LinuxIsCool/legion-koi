"""Generate Telegram leave/mute recommendations based on thread classification.

Outputs a human-readable report of:
- Groups in the exclude list (should leave)
- RECON groups with high message counts where Shawn never posted (candidates to mute)
- Groups with 0 messages (dead, should leave)

Informational only — not connected to any config.

Usage:
    cd ~/legion-koi && uv run python scripts/telegram_hygiene.py
"""

import sqlite3
import sys
from pathlib import Path

import yaml

SELF_SENDER_ID = "telegram:user:1441369482"
DB_PATH = Path("~/.claude/local/messages/messages.db").expanduser()
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

# Groups with more messages than this threshold and zero Shawn posts
# are candidates for muting in the Telegram app
MUTE_CANDIDATE_THRESHOLD = 1000


def load_excludes() -> set[str]:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return set(config.get("sensors", {}).get("message_thread_excludes", []))


def main():
    if not DB_PATH.exists():
        print(f"ERROR: messages.db not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    excludes = load_excludes()

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            t.id AS thread_id,
            t.title,
            t.thread_type,
            COUNT(m.id) AS msg_count,
            COALESCE(SUM(
                CASE WHEN m.sender_id = ? THEN 1 ELSE 0 END
            ), 0) AS shawn_posts
        FROM threads t
        LEFT JOIN messages m ON m.thread_id = t.id
        WHERE t.platform = 'telegram'
          AND t.thread_type NOT IN ('dm', 'private')
        GROUP BY t.id
        ORDER BY msg_count DESC
    """, (SELF_SENDER_ID,)).fetchall()

    conn.close()

    # Categorize
    should_leave = []   # In exclude list
    dead_groups = []     # 0 messages, not excluded
    mute_candidates = [] # High volume, zero participation, not excluded

    for row in rows:
        tid = row["thread_id"]
        title = row["title"] or "(untitled)"
        msg_count = row["msg_count"]
        shawn_posts = row["shawn_posts"]

        if tid in excludes:
            should_leave.append((tid, title, msg_count))
        elif msg_count == 0:
            dead_groups.append((tid, title))
        elif shawn_posts == 0 and msg_count >= MUTE_CANDIDATE_THRESHOLD:
            mute_candidates.append((tid, title, msg_count))

    # Report
    print("=" * 70)
    print("TELEGRAM HYGIENE REPORT")
    print("=" * 70)

    print(f"\n## LEAVE ({len(should_leave)} groups in exclude list)")
    print("-" * 50)
    for tid, title, count in should_leave:
        print(f"  {title:<45} {count:>8,} msgs")
        print(f"    {tid}")

    print(f"\n## DEAD ({len(dead_groups)} groups with 0 messages)")
    print("-" * 50)
    for tid, title in dead_groups:
        print(f"  {title}")
        print(f"    {tid}")

    print(f"\n## MUTE CANDIDATES ({len(mute_candidates)} high-volume lurk groups)")
    print(f"   (>{MUTE_CANDIDATE_THRESHOLD:,} msgs, zero Shawn posts)")
    print("-" * 50)
    for tid, title, count in mute_candidates:
        print(f"  {title:<45} {count:>8,} msgs")
        print(f"    {tid}")

    total = len(should_leave) + len(dead_groups) + len(mute_candidates)
    print(f"\n{'=' * 70}")
    print(f"Total actionable: {total}")
    print(f"  Leave: {len(should_leave)} | Dead: {len(dead_groups)} | Mute: {len(mute_candidates)}")


if __name__ == "__main__":
    main()
