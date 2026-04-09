"""Voice sensor — polls voice events JSONL files for new entries."""

import json
import threading
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.voice_event import LegionVoiceEvent
from . import state as sensor_state

log = structlog.stdlib.get_logger()

EVENTS_DIR = Path("~/.claude/local/voice/events").expanduser()


class VoiceSensor:
    """Polls JSONL voice event logs for new entries.

    Each JSONL file covers one month (YYYY-MM.jsonl). The sensor tracks the
    last-processed line offset per file to avoid re-reading.
    """

    def __init__(
        self,
        state_path: Path,
        kobj_push: callable,
        poll_interval: float = 60.0,
    ):
        self.events_dir = EVENTS_DIR
        self.state_path = Path(state_path)
        self.kobj_push = kobj_push
        self.poll_interval = poll_interval
        self.state = sensor_state.load(self.state_path)
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.Lock()

    def poll(self) -> list[Bundle]:
        """Read new lines from all JSONL files since last poll."""
        if not self.events_dir.exists():
            return []

        bundles = []
        for jsonl_path in sorted(self.events_dir.glob("*.jsonl")):
            file_key = jsonl_path.name
            last_offset = int(self.state.get(f"offset:{file_key}", 0))

            try:
                file_size = jsonl_path.stat().st_size
                if file_size <= last_offset:
                    continue

                with open(jsonl_path, "r", encoding="utf-8") as f:
                    f.seek(last_offset)
                    line_num = int(self.state.get(f"line:{file_key}", 0))

                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        line_num += 1
                        bundle = self._event_to_bundle(event, file_key, line_num)
                        if bundle:
                            bundles.append(bundle)

                    new_offset = f.tell()
                    self.state[f"offset:{file_key}"] = str(new_offset)
                    self.state[f"line:{file_key}"] = str(line_num)

            except Exception:
                log.exception("voice_sensor.read_error", path=str(jsonl_path))

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles

    def _event_to_bundle(self, event: dict, file_key: str, line_num: int) -> Bundle | None:
        """Convert a single JSONL event to a KOI bundle."""
        ts = event.get("ts", "")
        event_name = event.get("event", "unknown")
        sid = event.get("sid", "")

        # RID: YYYY-MM-DD/HH-MM-SS-event_name (from timestamp)
        # e.g., 2026-04-01/00-00-18-SessionStart
        if ts:
            # Parse ISO timestamp: "2026-04-01T00:00:18.155155+00:00"
            date_part = ts[:10]  # 2026-04-01
            time_part = ts[11:19].replace(":", "-")  # 00-00-18
            ref = f"{date_part}/{time_part}-{event_name}"
        else:
            ref = f"{file_key}/{line_num:06d}-{event_name}"

        # Dedup: check if we've already seen this exact ref
        content_str = json.dumps(event, sort_keys=True)
        content_hash = sensor_state.compute_hash(content_str)
        if sensor_state.has_changed(ref, content_hash, self.state) is None:
            return None

        self.state[ref] = content_hash

        rid = LegionVoiceEvent(event_ref=ref)

        # Build search text for FTS
        parts = [
            event_name,
            event.get("theme", ""),
            event.get("sound", ""),
            event.get("focus_state", ""),
        ]
        tts_text = event.get("tts_text", "")
        if tts_text:
            parts.append(tts_text)
        tts_voice = event.get("tts_voice", "")
        if tts_voice:
            parts.append(tts_voice)
        search_text = " ".join(p for p in parts if p)

        contents = {
            "event": event_name,
            "timestamp": ts,
            "session_id": sid,
            "theme": event.get("theme", ""),
            "sound": event.get("sound", ""),
            "volume": event.get("volume"),
            "muted": event.get("muted", False),
            "focus_state": event.get("focus_state", ""),
            "tts_text": tts_text,
            "tts_voice": tts_voice,
            "elapsed_ms": event.get("ms"),
            "search_text": search_text,
        }

        return Bundle.generate(rid=rid, contents=contents)

    def scan_all(self) -> list[Bundle]:
        """Full scan of all JSONL files."""
        return self.poll()

    def _poll_loop(self):
        if not self._running:
            return
        with self._lock:
            try:
                bundles = self.poll()
                for bundle in bundles:
                    self.kobj_push(bundle=bundle)
                if bundles:
                    sensor_state.save(self.state_path, self.state)
                    log.info("poll.complete", sensor="VoiceSensor", count=len(bundles))
            except Exception:
                log.exception("poll.error", sensor="VoiceSensor")
        if self._running:
            self._timer = threading.Timer(self.poll_interval, self._poll_loop)
            self._timer.daemon = True
            self._timer.start()

    def start(self):
        if not self.events_dir.exists():
            log.warning("sensor.events_dir_missing", path=str(self.events_dir))
            return
        self._running = True
        self._timer = threading.Timer(self.poll_interval, self._poll_loop)
        self._timer.daemon = True
        self._timer.start()
        log.info("sensor.started", events_dir=str(self.events_dir), sensor="VoiceSensor")

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()
            log.info("sensor.stopped", sensor="VoiceSensor")
