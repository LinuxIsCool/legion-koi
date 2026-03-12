"""CloudEvents-compatible event envelope for KOI event bus."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


# Core event type constants
BUNDLE_CREATED = "bundle.created"
BUNDLE_UPDATED = "bundle.updated"
ENTITY_EXTRACTED = "entity.extracted"
EMBEDDING_COMPUTED = "embedding.computed"
SERVICE_DEGRADED = "service.degraded"
SERVICE_DOWN = "service.down"

EVENT_SOURCE = "legion-koi"


@dataclass
class KoiEvent:
    """CloudEvents-compatible envelope for the KOI event bus.

    Follows CloudEvents spec v1.0 — type, source, subject, data, time, id.
    Serialized as flat JSON for Redis Streams (XADD accepts string key-value pairs).
    """

    type: str
    source: str = EVENT_SOURCE
    subject: str = ""
    data: dict = field(default_factory=dict)
    time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_stream_dict(self) -> dict[str, str]:
        """Serialize for Redis Streams XADD (all values must be strings)."""
        import json

        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "subject": self.subject,
            "data": json.dumps(self.data),
            "time": self.time,
        }

    @classmethod
    def from_stream_dict(cls, raw: dict[str, str]) -> KoiEvent:
        """Deserialize from Redis Streams XREADGROUP entry."""
        import json

        return cls(
            id=raw.get("id", str(uuid.uuid4())),
            type=raw.get("type", ""),
            source=raw.get("source", EVENT_SOURCE),
            subject=raw.get("subject", ""),
            data=json.loads(raw.get("data", "{}")),
            time=raw.get("time", ""),
        )

    def to_dict(self) -> dict:
        """Plain dict representation."""
        return asdict(self)
