"""Message sensor — polls messages DB for chat messages."""

import json

import structlog
from rid_lib.ext import Bundle

from ..rid_types.message import LegionMessage
from . import state as sensor_state
from .db_sensor import DatabaseSensor

log = structlog.stdlib.get_logger()

BATCH_SIZE = 1000


class MessageSensor(DatabaseSensor):
    def __init__(self, **kwargs):
        super().__init__(batch_size=BATCH_SIZE, **kwargs)

    def poll(self) -> list[Bundle]:
        if not self.db_path.exists():
            return []

        last_rowid = int(self.state.get("last_seen_rowid", 0))
        bundles = []

        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT rowid, * FROM messages WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, BATCH_SIZE),
            )
            for row in cursor:
                rowid = row["rowid"]
                message_id = row["id"]

                contents = {
                    "message_id": message_id,
                    "platform": row["platform"],
                    "thread_id": row["thread_id"],
                    "sender_id": row["sender_id"],
                    "content": row["content"],
                    "content_type": row["content_type"],
                    "platform_ts": row["platform_ts"],
                }

                content_hash = sensor_state.compute_hash(
                    json.dumps(contents, sort_keys=True, default=str)
                )
                change = sensor_state.has_changed(message_id, content_hash, self.state)
                if change is None:
                    self.state["last_seen_rowid"] = str(rowid)
                    continue

                rid = LegionMessage(message_id=message_id)
                bundle = Bundle.generate(rid=rid, contents=contents)
                bundles.append(bundle)

                self.state[message_id] = content_hash
                self.state["last_seen_rowid"] = str(rowid)

        finally:
            conn.close()

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles
