"""Transcript sensor — polls transcripts DB for structured transcript metadata."""

import json

import structlog
from rid_lib.ext import Bundle

from ..rid_types.transcript import LegionTranscript
from . import state as sensor_state
from .db_sensor import DatabaseSensor

log = structlog.stdlib.get_logger()


def _make_identifier(row: dict) -> str:
    """Build a human-readable identifier from transcript metadata.

    Prefers title-based slug; falls back to UUID.
    """
    title = row.get("title") or ""
    date = row.get("date_recorded") or row.get("created_at") or ""
    if title:
        # Slugify: lowercase, replace non-alphanum with hyphens, strip
        slug = title.lower()
        slug = "".join(c if c.isalnum() or c in " -_" else "" for c in slug)
        slug = "-".join(slug.split())[:80]
        if date and len(date) >= 10:
            return f"{date[:10]}-{slug}"
        return slug
    # Fallback to UUID
    return row.get("uuid") or str(row.get("id", "unknown"))


class TranscriptSensor(DatabaseSensor):
    def poll(self) -> list[Bundle]:
        if not self.db_path.exists():
            return []

        last_rowid = int(self.state.get("last_seen_rowid", 0))
        bundles = []

        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT t.rowid AS rowid, t.id, t.uuid, t.title,
                       t.recording_id, t.recording_path,
                       t.backend, t.model, t.language, t.status,
                       t.duration_ms, t.utterance_count, t.speaker_count,
                       t.word_count, t.confidence, t.tags, t.consent_tier,
                       t.created_at, t.updated_at,
                       t.full_text
                FROM transcripts t
                WHERE t.rowid > ?
                ORDER BY t.rowid
                """,
                (last_rowid,),
            )
            for row in cursor:
                rowid = row["rowid"]
                row_dict = dict(row)

                identifier = _make_identifier(row_dict)

                # Build contents for the KOI bundle
                contents = {
                    "transcript_id": row["id"],
                    "uuid": row["uuid"],
                    "title": row["title"],
                    "recording_id": row["recording_id"],
                    "backend": row["backend"],
                    "model": row["model"],
                    "language": row["language"],
                    "status": row["status"],
                    "duration_ms": row["duration_ms"],
                    "utterance_count": row["utterance_count"],
                    "speaker_count": row["speaker_count"],
                    "word_count": row["word_count"],
                    "confidence": row["confidence"],
                    "tags": row["tags"],
                    "consent_tier": row["consent_tier"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

                # Include full_text for semantic search in KOI
                if row["full_text"]:
                    contents["full_text"] = row["full_text"]

                if row["recording_path"]:
                    contents["recording_path"] = row["recording_path"]

                content_hash = sensor_state.compute_hash(
                    json.dumps(contents, sort_keys=True, default=str)
                )
                ref_key = identifier
                change = sensor_state.has_changed(ref_key, content_hash, self.state)
                if change is None:
                    self.state["last_seen_rowid"] = str(rowid)
                    continue

                rid = LegionTranscript(identifier=identifier)
                bundle = Bundle.generate(rid=rid, contents=contents)
                bundles.append(bundle)

                self.state[ref_key] = content_hash
                self.state["last_seen_rowid"] = str(rowid)

                log.info(
                    "transcript.detected",
                    change=change,
                    rid=str(rid),
                    title=row["title"],
                )
        finally:
            conn.close()

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles
