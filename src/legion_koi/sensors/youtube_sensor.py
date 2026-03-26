"""YouTube channel sensor — polls channels via yt-dlp for new videos.

Unlike DatabaseSensor (SQLite) or BaseSensor (filesystem), this sensor
polls an external service. It uses yt-dlp --flat-playlist to enumerate
recent videos, then fetches full metadata for any new ones.

Design decisions:
  - yt-dlp over RSS: YouTube RSS feeds return 404 as of Mar 2026.
  - --flat-playlist: Fast enumeration without downloading anything.
  - Full metadata fetch only for NEW videos (dedup by video_id in state).
  - Poll interval defaults to 604800s (1 week) — indydevdan uploads Mondays.
  - Channels configured as list of {handle, channel_id} dicts in config.
"""

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.youtube import LegionYoutube
from . import state as sensor_state

log = structlog.stdlib.get_logger()


@dataclass
class YouTubeChannel:
    """A YouTube channel to monitor."""
    handle: str       # e.g. "indydevdan"
    channel_id: str   # e.g. "UC_x36zCEGilGpB1m-V4gmjg"
    # Optional: limit how many recent videos to check per poll
    max_videos: int = 15


class YouTubeSensor:
    """Polls YouTube channels for new videos, creates KOI bundles."""

    def __init__(
        self,
        channels: list[YouTubeChannel],
        state_path: Path,
        kobj_push: callable,
        poll_interval: float = 604800.0,
    ):
        self.channels = channels
        self.state_path = Path(state_path)
        self.kobj_push = kobj_push
        self.poll_interval = poll_interval
        self.state = sensor_state.load(self.state_path)
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # yt-dlp helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enumerate_videos(channel: YouTubeChannel) -> list[dict]:
        """Get recent video IDs + titles from a channel via yt-dlp --flat-playlist."""
        url = f"https://www.youtube.com/@{channel.handle}/videos"
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            "--playlist-end", str(channel.max_videos),
            url,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                log.warning(
                    "youtube.enumerate_failed",
                    channel=channel.handle,
                    stderr=result.stderr[:500],
                )
                return []
            entries = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    entries.append(json.loads(line))
            return entries
        except subprocess.TimeoutExpired:
            log.warning("youtube.enumerate_timeout", channel=channel.handle)
            return []
        except Exception:
            log.exception("youtube.enumerate_error", channel=channel.handle)
            return []

    @staticmethod
    def _fetch_metadata(video_id: str) -> dict | None:
        """Fetch full metadata for a single video via yt-dlp --dump-json."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--skip-download",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            log.exception("youtube.metadata_error", video_id=video_id)
        return None

    # ------------------------------------------------------------------
    # Bundle creation
    # ------------------------------------------------------------------

    def _make_bundle(self, channel: YouTubeChannel, video_id: str, meta: dict) -> Bundle:
        """Create a KOI bundle from video metadata."""
        rid = LegionYoutube(channel=channel.handle, video_id=video_id)

        # Duration formatting
        duration_s = meta.get("duration")
        duration_human = ""
        if duration_s:
            m, s = divmod(int(duration_s), 60)
            h, m = divmod(m, 60)
            if h:
                duration_human = f"{h}h{m:02d}m{s:02d}s"
            else:
                duration_human = f"{m}m{s:02d}s"

        # Extract tags/categories
        tags = meta.get("tags") or []
        categories = meta.get("categories") or []

        contents = {
            "video_id": video_id,
            "title": meta.get("title", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": meta.get("channel", channel.handle),
            "channel_id": meta.get("channel_id", channel.channel_id),
            "channel_handle": channel.handle,
            "upload_date": meta.get("upload_date", ""),
            "description": meta.get("description", ""),
            "duration_s": duration_s,
            "duration_human": duration_human,
            "view_count": meta.get("view_count"),
            "like_count": meta.get("like_count"),
            "comment_count": meta.get("comment_count"),
            "tags": tags,
            "categories": categories,
            "thumbnail": meta.get("thumbnail", ""),
        }

        return Bundle.generate(rid=rid, contents=contents)

    # ------------------------------------------------------------------
    # Poll / scan
    # ------------------------------------------------------------------

    def _poll_channel(self, channel: YouTubeChannel) -> list[Bundle]:
        """Poll a single channel for new videos."""
        entries = self._enumerate_videos(channel)
        if not entries:
            return []

        bundles = []
        for entry in entries:
            video_id = entry.get("id", "")
            if not video_id:
                continue

            ref_key = f"{channel.handle}/{video_id}"

            # Skip if already seen (state stores video_id → hash)
            if ref_key in self.state:
                continue

            # New video — fetch full metadata
            log.info("youtube.new_video", channel=channel.handle, video_id=video_id,
                     title=entry.get("title", "")[:80])
            meta = self._fetch_metadata(video_id)
            if meta is None:
                # Fallback: use flat-playlist entry (less data but still useful)
                meta = entry

            bundle = self._make_bundle(channel, video_id, meta)
            bundles.append(bundle)

            # Store hash for dedup
            content_hash = sensor_state.compute_hash(
                json.dumps({"id": video_id, "title": meta.get("title", "")},
                           sort_keys=True)
            )
            self.state[ref_key] = content_hash

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles

    def poll(self) -> list[Bundle]:
        """Poll all configured channels."""
        all_bundles = []
        for channel in self.channels:
            bundles = self._poll_channel(channel)
            all_bundles.extend(bundles)
        return all_bundles

    def scan_all(self) -> list[Bundle]:
        """Initial scan — same as poll."""
        return self.poll()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Timer callback."""
        if not self._running:
            return
        with self._lock:
            try:
                bundles = self.poll()
                for bundle in bundles:
                    self.kobj_push(bundle=bundle)
                if bundles:
                    log.info("youtube.poll_complete", count=len(bundles))
            except Exception:
                log.exception("youtube.poll_error")
        if self._running:
            self._timer = threading.Timer(self.poll_interval, self._poll_loop)
            self._timer.daemon = True
            self._timer.start()

    def start(self):
        """Start the polling loop."""
        if not self.channels:
            log.warning("youtube.no_channels", msg="No channels configured")
            return
        self._running = True
        self._timer = threading.Timer(self.poll_interval, self._poll_loop)
        self._timer.daemon = True
        self._timer.start()
        handles = [c.handle for c in self.channels]
        log.info("youtube.started", channels=handles, poll_interval=self.poll_interval)

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            log.info("youtube.stopped")
