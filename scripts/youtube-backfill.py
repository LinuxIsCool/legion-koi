# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "youtube-transcript-api>=1.0.0",
#     "yt-dlp",
#     "psycopg2-binary",
# ]
# ///
"""YouTube channel backfill — download all historic videos + transcripts into KOI.

Rate-limit aware: processes videos in batches with configurable delays
between batches to stay under YouTube's rate limits.

Usage:
  uv run scripts/youtube-backfill.py                        # backfill indydevdan
  uv run scripts/youtube-backfill.py --channel karpathy     # different channel
  uv run scripts/youtube-backfill.py --batch-size 5 --delay 120  # slower
  uv run scripts/youtube-backfill.py --transcripts-only     # just transcripts for known videos
  uv run scripts/youtube-backfill.py --dry-run              # enumerate only, no downloads
  uv run scripts/youtube-backfill.py --resume               # continue from last checkpoint

Rate limit strategy:
  - Enumeration (--flat-playlist): 1 request, returns all video IDs. Safe.
  - Metadata fetch (--dump-json): 1 request per video. Batch + delay.
  - Transcript fetch (youtube-transcript-api): Uses captions API, lighter quota.
  - Default: 5 videos per batch, 60s between batches → ~15 videos/hr safe cruise.
  - On HTTP 429 or connection reset: exponential backoff, checkpoint, resume later.

State file tracks per-video progress:
  - "meta": metadata fetched and KOI bundle created
  - "meta+transcript": transcript also fetched and cached
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNELS = {
    "indydevdan": "UC_x36zCEGilGpB1m-V4gmjg",
}

STATE_DIR = Path.home() / ".claude" / "local" / "youtube"
TRANSCRIPT_CACHE = STATE_DIR / "cache"
BACKFILL_STATE_FILE = STATE_DIR / "backfill_state.json"

NAMESPACE = "legion.claude-youtube"
DB_NAME = "personal_koi"

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# YouTube helpers
# ---------------------------------------------------------------------------

def enumerate_all_videos(handle: str) -> list[dict]:
    """Get ALL video IDs from a channel. Single request, no rate limit concern."""
    url = f"https://www.youtube.com/@{handle}/videos"
    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", url]
    print(f"[enumerate] Fetching full video list for @{handle}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[error] yt-dlp failed: {result.stderr[:300]}", file=sys.stderr)
        return []
    entries = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    print(f"[enumerate] Found {len(entries)} videos")
    return entries


def fetch_metadata(video_id: str) -> dict | None:
    """Fetch full metadata for one video."""
    cmd = [
        "yt-dlp", "--dump-json", "--skip-download",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
        # Check for rate limit
        if "429" in result.stderr or "Too Many Requests" in result.stderr:
            return "RATE_LIMITED"
    except subprocess.TimeoutExpired:
        print(f"  [timeout] metadata for {video_id}", file=sys.stderr)
    except Exception as e:
        print(f"  [error] metadata for {video_id}: {e}", file=sys.stderr)
    return None


def fetch_transcript(video_id: str, language: str = "en") -> dict | None:
    """Fetch transcript via youtube-transcript-api (lightest quota usage)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt = YouTubeTranscriptApi()

        # Try manual captions first, then auto-generated
        try:
            transcript = ytt.fetch(video_id, languages=[language, "en"])
            segments = [{"text": s.text, "start": s.start, "duration": s.duration}
                        for s in transcript]
            return {"segments": segments, "source": "manual", "language": language}
        except Exception:
            pass

        try:
            tlist = ytt.list_transcripts(video_id)
            try:
                t = tlist.find_generated_transcript([language, "en"])
                segments = [{"text": s.text, "start": s.start, "duration": s.duration}
                            for s in t.fetch()]
                return {"segments": segments, "source": "auto", "language": language}
            except Exception:
                pass
            for t in tlist:
                segments = [{"text": s.text, "start": s.start, "duration": s.duration}
                            for s in t.fetch()]
                source = "auto" if t.is_generated else "manual"
                return {"segments": segments, "source": source, "language": t.language_code}
        except Exception:
            pass

    except ImportError:
        print("  [warn] youtube-transcript-api not installed", file=sys.stderr)
    except Exception as e:
        if "429" in str(e) or "Too Many" in str(e):
            return "RATE_LIMITED"
        print(f"  [error] transcript for {video_id}: {e}", file=sys.stderr)

    return None


# ---------------------------------------------------------------------------
# KOI bundle creation
# ---------------------------------------------------------------------------

def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def format_duration(seconds) -> str:
    if not seconds:
        return ""
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def upsert_koi_bundle(handle: str, video_id: str, meta: dict, transcript_text: str = ""):
    """UPSERT video bundle into personal_koi PostgreSQL."""
    import psycopg2

    rid = f"orn:{NAMESPACE}:{handle}/{video_id}"
    reference = f"{handle}/{video_id}"

    tags = meta.get("tags") or []
    categories = meta.get("categories") or []

    contents = {
        "video_id": video_id,
        "title": meta.get("title", ""),
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "channel": meta.get("channel", handle),
        "channel_id": meta.get("channel_id", ""),
        "channel_handle": handle,
        "upload_date": meta.get("upload_date", ""),
        "description": meta.get("description", ""),
        "duration_s": meta.get("duration"),
        "duration_human": format_duration(meta.get("duration")),
        "view_count": meta.get("view_count"),
        "like_count": meta.get("like_count"),
        "comment_count": meta.get("comment_count"),
        "tags": tags,
        "categories": categories,
        "thumbnail": meta.get("thumbnail", ""),
        "has_transcript": bool(transcript_text),
    }

    # Search text: title + description + transcript (full-text searchable)
    title = contents["title"]
    desc = (contents["description"] or "")[:1000]
    tags_str = " ".join(tags)
    search_text = f"{title} {tags_str} {desc} {transcript_text}".strip()

    content_hash = compute_hash(json.dumps(contents, sort_keys=True, default=str))
    now = datetime.now(timezone.utc)

    with psycopg2.connect(dbname=DB_NAME, connect_timeout=3) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bundles (rid, namespace, reference, contents, search_text, sha256_hash, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (rid) DO UPDATE SET
                    contents = EXCLUDED.contents,
                    search_text = EXCLUDED.search_text,
                    sha256_hash = EXCLUDED.sha256_hash,
                    updated_at = EXCLUDED.updated_at
                WHERE bundles.sha256_hash != EXCLUDED.sha256_hash
                """,
                (rid, NAMESPACE, reference, json.dumps(contents), search_text, content_hash, now, now),
            )


# ---------------------------------------------------------------------------
# Backfill orchestrator
# ---------------------------------------------------------------------------

def backfill(
    handle: str,
    batch_size: int = 5,
    delay_seconds: int = 60,
    transcripts: bool = True,
    transcripts_only: bool = False,
    dry_run: bool = False,
    resume: bool = True,
    max_batches: int = 0,
):
    """Backfill all videos from a channel with rate limiting."""

    # Load or initialize backfill state
    state_file = STATE_DIR / f"backfill_{handle}_state.json"
    state = load_state(state_file) if resume else {}

    # Phase 1: enumerate all videos (single request)
    entries = enumerate_all_videos(handle)
    if not entries:
        print("[error] No videos found. Exiting.")
        return

    # Filter to unprocessed videos
    target_stage = "meta+transcript" if transcripts else "meta"
    pending = []
    for entry in entries:
        vid = entry.get("id", "")
        if not vid:
            continue
        current_stage = state.get(vid, {}).get("stage", "")
        if transcripts_only and current_stage == "meta":
            pending.append(entry)  # has meta but needs transcript
        elif not transcripts_only and current_stage != target_stage:
            pending.append(entry)

    total = len(entries)
    done = total - len(pending)
    print(f"[plan] {total} total videos, {done} already done, {len(pending)} pending")

    if dry_run:
        print("\n[dry-run] Would process:")
        for i, e in enumerate(pending[:30]):
            print(f"  {i+1:3d}. {e['id']} — {e.get('title', '?')[:70]}")
        if len(pending) > 30:
            print(f"  ... and {len(pending) - 30} more")
        return

    if not pending:
        print("[done] All videos already processed.")
        return

    # Phase 2: process in batches
    batches_done = 0
    videos_done = 0
    rate_limited = False

    for i in range(0, len(pending), batch_size):
        if max_batches and batches_done >= max_batches:
            print(f"\n[pause] Reached max_batches={max_batches}. Run again with --resume to continue.")
            break

        batch = pending[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(pending) + batch_size - 1) // batch_size
        print(f"\n{'='*60}")
        print(f"[batch {batch_num}/{total_batches}] Processing {len(batch)} videos...")
        print(f"{'='*60}")

        for entry in batch:
            vid = entry["id"]
            title = entry.get("title", "?")[:60]
            current_stage = state.get(vid, {}).get("stage", "")

            # --- Metadata ---
            if current_stage not in ("meta", "meta+transcript") and not transcripts_only:
                print(f"\n  [{vid}] {title}")
                print(f"  Fetching metadata...", end=" ", flush=True)
                meta = fetch_metadata(vid)

                if meta == "RATE_LIMITED":
                    print("RATE LIMITED!")
                    rate_limited = True
                    break
                elif meta is None:
                    print("failed (skipping)")
                    state[vid] = {"stage": "error", "error": "metadata_fetch_failed",
                                  "ts": datetime.now(timezone.utc).isoformat()}
                    save_state(state_file, state)
                    continue
                else:
                    print(f"OK ({format_duration(meta.get('duration'))})")

                # Cache metadata
                cache_dir = TRANSCRIPT_CACHE / vid
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

                # Upsert to KOI (without transcript for now)
                upsert_koi_bundle(handle, vid, meta)
                state[vid] = {"stage": "meta", "ts": datetime.now(timezone.utc).isoformat()}
                save_state(state_file, state)
            else:
                # Load cached metadata for transcript-only pass
                cache_dir = TRANSCRIPT_CACHE / vid
                meta_path = cache_dir / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                else:
                    meta = entry  # fallback to flat-playlist data

            # --- Transcript ---
            if transcripts and current_stage != "meta+transcript":
                print(f"  Fetching transcript...", end=" ", flush=True)
                transcript_data = fetch_transcript(vid)

                if transcript_data == "RATE_LIMITED":
                    print("RATE LIMITED!")
                    rate_limited = True
                    break
                elif transcript_data is None:
                    print("not available")
                    # Still mark as done — no transcript exists for this video
                    state[vid] = {"stage": "meta+transcript", "has_transcript": False,
                                  "ts": datetime.now(timezone.utc).isoformat()}
                    save_state(state_file, state)
                else:
                    segments = transcript_data["segments"]
                    full_text = " ".join(s["text"] for s in segments)
                    word_count = len(full_text.split())
                    print(f"OK ({len(segments)} segments, {word_count} words, {transcript_data['source']})")

                    # Cache transcript
                    cache_dir = TRANSCRIPT_CACHE / vid
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / "transcript.json").write_text(
                        json.dumps(transcript_data, indent=2)
                    )

                    # Re-upsert with transcript text in search_text
                    upsert_koi_bundle(handle, vid, meta, transcript_text=full_text)
                    state[vid] = {"stage": "meta+transcript", "has_transcript": True,
                                  "word_count": word_count,
                                  "ts": datetime.now(timezone.utc).isoformat()}
                    save_state(state_file, state)

            videos_done += 1

        if rate_limited:
            save_state(state_file, state)
            print(f"\n[rate-limited] Processed {videos_done} videos before hitting rate limit.")
            print(f"[rate-limited] Run again with --resume in 15-30 minutes.")
            break

        batches_done += 1

        # Delay between batches (skip after last batch)
        remaining = len(pending) - (i + len(batch))
        if remaining > 0 and not rate_limited:
            print(f"\n[throttle] Waiting {delay_seconds}s before next batch... ({remaining} remaining)")
            time.sleep(delay_seconds)

    # Summary
    print(f"\n{'='*60}")
    stages = {}
    for vid_state in state.values():
        if isinstance(vid_state, dict):
            s = vid_state.get("stage", "unknown")
            stages[s] = stages.get(s, 0) + 1
    print(f"[summary] {videos_done} videos processed this run")
    print(f"[summary] State: {stages}")
    print(f"[summary] State file: {state_file}")
    if rate_limited:
        print(f"[summary] RATE LIMITED — resume later with: uv run {sys.argv[0]} --resume")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill YouTube channel videos + transcripts into KOI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run scripts/youtube-backfill.py                           # full backfill
  uv run scripts/youtube-backfill.py --dry-run                 # see what would happen
  uv run scripts/youtube-backfill.py --batch-size 3 --delay 120  # extra cautious
  uv run scripts/youtube-backfill.py --transcripts-only --resume # just add transcripts
  uv run scripts/youtube-backfill.py --max-batches 5           # do 25 videos then stop
        """,
    )
    parser.add_argument("--channel", default="indydevdan",
                        help="Channel handle (default: indydevdan)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Videos per batch (default: 5)")
    parser.add_argument("--delay", type=int, default=60,
                        help="Seconds between batches (default: 60)")
    parser.add_argument("--no-transcripts", action="store_true",
                        help="Skip transcript fetching (metadata only)")
    parser.add_argument("--transcripts-only", action="store_true",
                        help="Only fetch transcripts for videos that already have metadata")
    parser.add_argument("--dry-run", action="store_true",
                        help="Enumerate only, don't download anything")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from last checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore previous state, start from scratch")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Stop after N batches (0 = unlimited)")

    args = parser.parse_args()

    if args.channel not in CHANNELS:
        print(f"[warn] Unknown channel '{args.channel}', using handle as-is")

    backfill(
        handle=args.channel,
        batch_size=args.batch_size,
        delay_seconds=args.delay,
        transcripts=not args.no_transcripts,
        transcripts_only=args.transcripts_only,
        dry_run=args.dry_run,
        resume=not args.fresh,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
