"""Reset the message sensor state file.

Sets last_seen_rowid to the current max rowid in messages.db,
dropping the 87MB hash-per-message state to ~30 bytes. The sensor
will resume polling from the current position without re-processing.

Usage:
    cd ~/legion-koi && uv run python scripts/reset_message_state.py
"""

import json
import sqlite3
import sys
from pathlib import Path


def main():
    db_path = Path("~/.claude/local/messages/messages.db").expanduser()
    state_path = Path("./state/message_state.json")

    if not db_path.exists():
        print(f"ERROR: messages.db not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    # Get current max rowid
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    max_rowid = conn.execute("SELECT MAX(rowid) FROM messages").fetchone()[0]
    conn.close()

    if max_rowid is None:
        print("ERROR: No messages in database", file=sys.stderr)
        sys.exit(1)

    # Show current state size
    if state_path.exists():
        old_size = state_path.stat().st_size
        old_state = json.loads(state_path.read_text())
        old_entries = len(old_state) - 1  # minus last_seen_rowid key
        print(f"Current state: {old_size:,} bytes, {old_entries:,} hash entries")
        print(f"Current last_seen_rowid: {old_state.get('last_seen_rowid', 'not set')}")
    else:
        print("No existing state file.")

    print(f"Max rowid in messages.db: {max_rowid}")

    # Confirm
    answer = input(f"\nReset state to just last_seen_rowid={max_rowid}? [y/N] ")
    if answer.lower() != "y":
        print("Aborted.")
        return

    new_state = {"last_seen_rowid": str(max_rowid)}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(new_state, indent=2))

    new_size = state_path.stat().st_size
    print(f"State reset: {new_size} bytes")
    print("Done.")


if __name__ == "__main__":
    main()
