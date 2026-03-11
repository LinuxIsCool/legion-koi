"""Recording sensor — polls recordings DB for media file metadata."""

import json
import re
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.recording import LegionRecording
from . import state as sensor_state
from .db_sensor import DatabaseSensor

log = structlog.stdlib.get_logger()

# Match both formats: [0.00s -> 5.00s] and [00:16]
_TIMESTAMP_RE = re.compile(r"^\[\d+[\.:]\d+(?:s?\s*->\s*\d+\.\d+s)?\]\s*")


def _stem_from_filename(filename: str) -> str:
    """Strip media extension to get a human-readable identifier."""
    return Path(filename).stem


def _load_transcript(path: str) -> str | None:
    """Load transcript file, stripping timestamp prefixes."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        lines = []
        for line in text.splitlines():
            cleaned = _TIMESTAMP_RE.sub("", line)
            if cleaned.strip():
                lines.append(cleaned.strip())
        return " ".join(lines) if lines else None
    except OSError:
        log.warning("recording.transcript_unreadable", path=path)
        return None


class RecordingSensor(DatabaseSensor):
    def poll(self) -> list[Bundle]:
        if not self.db_path.exists():
            return []

        last_rowid = int(self.state.get("last_seen_rowid", 0))
        bundles = []

        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT r.rowid AS rowid, r.id, r.filename, r.path, r.source,
                       r.media_type, r.duration_human, r.duration_seconds,
                       r.date_recorded, r.resolution, r.title, r.notes,
                       r.file_size_bytes,
                       t.path AS transcript_path,
                       t.word_count AS transcript_word_count
                FROM recordings r
                LEFT JOIN transcripts t ON t.recording_id = r.id
                WHERE r.rowid > ?
                ORDER BY r.rowid
                """,
                (last_rowid,),
            )
            for row in cursor:
                rowid = row["rowid"]
                source = row["source"] or "unknown"
                identifier = _stem_from_filename(row["filename"])

                contents = {
                    "filename": row["filename"],
                    "path": row["path"],
                    "source": source,
                    "media_type": row["media_type"],
                    "duration_human": row["duration_human"],
                    "duration_seconds": row["duration_seconds"],
                    "date_recorded": row["date_recorded"],
                    "resolution": row["resolution"],
                    "title": row["title"],
                    "notes": row["notes"],
                    "file_size_bytes": row["file_size_bytes"],
                    "has_transcript": row["transcript_path"] is not None,
                }
                if row["transcript_path"]:
                    contents["transcript_path"] = row["transcript_path"]
                    contents["transcript_word_count"] = row["transcript_word_count"]
                    transcript_text = _load_transcript(row["transcript_path"])
                    if transcript_text:
                        contents["transcript_text"] = transcript_text

                content_hash = sensor_state.compute_hash(
                    json.dumps(contents, sort_keys=True, default=str)
                )
                ref_key = f"{source}/{identifier}"
                change = sensor_state.has_changed(ref_key, content_hash, self.state)
                if change is None:
                    self.state["last_seen_rowid"] = str(rowid)
                    continue

                rid = LegionRecording(source=source, identifier=identifier)
                bundle = Bundle.generate(rid=rid, contents=contents)
                bundles.append(bundle)

                self.state[ref_key] = content_hash
                self.state["last_seen_rowid"] = str(rowid)

                log.info(
                    "recording.detected",
                    change=change,
                    rid=str(rid),
                    filename=row["filename"],
                )
        finally:
            conn.close()

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles
