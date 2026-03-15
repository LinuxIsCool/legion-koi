"""Contact sensor — polls messages DB for ContactRank scores."""

from __future__ import annotations

import json

import structlog
from rid_lib.ext import Bundle

from ..rid_types.contact import LegionContact
from . import state as sensor_state
from .db_sensor import DatabaseSensor

log = structlog.stdlib.get_logger()

BATCH_SIZE = 500


class ContactSensor(DatabaseSensor):
    def poll(self) -> list[Bundle]:
        if not self.db_path.exists():
            return []

        last_computed = self.state.get("last_poll_computed_at", "")
        bundles = []

        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                SELECT cs.identity_id, cs.frequency, cs.recency,
                       cs.reciprocity, cs.channel_diversity, cs.dm_ratio,
                       cs.structural, cs.temporal_regularity,
                       cs.response_latency, cs.composite, cs.dunbar_layer,
                       cs.confidence, cs.computed_at,
                       i.display_name
                FROM contact_scores cs
                JOIN identities i ON i.id = cs.identity_id
                WHERE cs.computed_at > ?
                ORDER BY cs.computed_at
                LIMIT ?
                """,
                (last_computed, BATCH_SIZE),
            )
            for row in cursor:
                identity_id = row["identity_id"]

                contents = {
                    "identity_id": identity_id,
                    "display_name": row["display_name"],
                    "composite": row["composite"],
                    "dunbar_layer": row["dunbar_layer"],
                    "confidence": row["confidence"],
                    "frequency": row["frequency"],
                    "recency": row["recency"],
                    "reciprocity": row["reciprocity"],
                    "channel_diversity": row["channel_diversity"],
                    "dm_ratio": row["dm_ratio"],
                    "structural": row["structural"],
                    "temporal_regularity": row["temporal_regularity"],
                    "response_latency": row["response_latency"],
                    "computed_at": row["computed_at"],
                }

                content_hash = sensor_state.compute_hash(
                    json.dumps(contents, sort_keys=True, default=str)
                )
                change = sensor_state.has_changed(identity_id, content_hash, self.state)
                if change is None:
                    continue

                rid = LegionContact(identity_id=identity_id)
                bundle = Bundle.generate(rid=rid, contents=contents)
                bundles.append(bundle)

                self.state[identity_id] = content_hash
                # Track high-water mark for next poll
                self.state["last_poll_computed_at"] = row["computed_at"]

        finally:
            conn.close()

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles
