"""Logging sensor — polls Claude Code logging DB for session metadata."""

import json

import structlog
from rid_lib.ext import Bundle

from ..rid_types.session import LegionSession
from . import state as sensor_state
from .db_sensor import DatabaseSensor

log = structlog.stdlib.get_logger()


class LoggingSensor(DatabaseSensor):
    def poll(self) -> list[Bundle]:
        if not self.db_path.exists():
            return []

        last_rowid = int(self.state.get("last_seen_rowid", 0))
        bundles = []

        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT rowid, * FROM sessions WHERE rowid > ? ORDER BY rowid",
                (last_rowid,),
            )
            for row in cursor:
                rowid = row["rowid"]
                session_id = row["id"]

                contents = {
                    "session_id": session_id,
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "cwd": row["cwd"],
                    "summary": row["summary"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "event_count": row["event_count"],
                    "total_tokens": row["total_tokens"],
                }

                # Check hash for dedup on restart
                content_hash = sensor_state.compute_hash(json.dumps(contents, sort_keys=True))
                change = sensor_state.has_changed(session_id, content_hash, self.state)
                if change is None:
                    self.state["last_seen_rowid"] = str(rowid)
                    continue

                rid = LegionSession(session_id=session_id)
                bundle = Bundle.generate(rid=rid, contents=contents)
                bundles.append(bundle)

                self.state[session_id] = content_hash
                self.state["last_seen_rowid"] = str(rowid)

                log.info(
                    "logging.detected",
                    change=change,
                    rid=str(rid),
                    cwd=row["cwd"] or "",
                )
        finally:
            conn.close()

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles
