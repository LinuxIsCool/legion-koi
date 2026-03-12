"""Redis Streams producer/consumer primitives for the KOI event bus.

Uses the FalkorDB Redis instance on port 6380 (already running).
Stream naming: koi:events:{event_type} (e.g. koi:events:bundle.created)
Dead letter queue: koi:dlq:{stream_name}
"""

from __future__ import annotations

import redis
import structlog

from ..constants import (
    EVENT_REDIS_HOST,
    EVENT_REDIS_PORT,
    EVENT_STREAM_PREFIX,
    EVENT_DLQ_PREFIX,
)
from .schemas import KoiEvent

log = structlog.stdlib.get_logger()


def stream_name(event_type: str) -> str:
    """Canonical stream name for an event type."""
    return f"{EVENT_STREAM_PREFIX}{event_type}"


def dlq_name(stream: str) -> str:
    """Dead letter queue stream for a given event stream."""
    return f"{EVENT_DLQ_PREFIX}{stream}"


class EventBus:
    """Thin wrapper around Redis Streams for producing and consuming events.

    One EventBus instance per process — shared across producer (PG listener)
    and consumers. Manages connection lifecycle and consumer group creation.
    """

    def __init__(
        self,
        host: str = EVENT_REDIS_HOST,
        port: int = EVENT_REDIS_PORT,
    ):
        self._redis = redis.Redis(host=host, port=port, decode_responses=True)

    def publish(self, event: KoiEvent) -> str:
        """Publish an event to its type-specific stream. Returns stream entry ID."""
        sname = stream_name(event.type)
        entry_id = self._redis.xadd(sname, event.to_stream_dict())
        log.debug("event.published", stream=sname, event_type=event.type, subject=event.subject)
        return entry_id

    def ensure_group(self, sname: str, group: str) -> None:
        """Create a consumer group if it doesn't exist. Idempotent."""
        try:
            self._redis.xgroup_create(sname, group, id="0", mkstream=True)
            log.info("event.group_created", stream=sname, group=group)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def read_group(
        self,
        sname: str,
        group: str,
        consumer_id: str,
        count: int = 10,
        block_ms: int = 2000,
    ) -> list[tuple[str, dict[str, str]]]:
        """Read new messages from a consumer group.

        Returns list of (entry_id, fields_dict) tuples.
        """
        results = self._redis.xreadgroup(
            group,
            consumer_id,
            {sname: ">"},
            count=count,
            block=block_ms,
        )
        if not results:
            return []
        # results is [(stream_name, [(entry_id, fields), ...])]
        return [(eid, fields) for eid, fields in results[0][1]]

    def ack(self, sname: str, group: str, entry_id: str) -> None:
        """Acknowledge successful processing of a message."""
        self._redis.xack(sname, group, entry_id)

    def claim_pending(
        self,
        sname: str,
        group: str,
        consumer_id: str,
        min_idle_ms: int = 5000,
        count: int = 10,
    ) -> list[tuple[str, dict[str, str]]]:
        """Claim idle pending messages for retry.

        Uses XAUTOCLAIM to take ownership of messages that have been pending
        (unacked) for at least min_idle_ms. This enables retry of failed messages.
        Returns list of (entry_id, fields_dict) tuples.
        """
        try:
            # XAUTOCLAIM returns (next_start_id, [(entry_id, fields), ...], [deleted_ids])
            result = self._redis.xautoclaim(
                sname, group, consumer_id, min_idle_time=min_idle_ms, start_id="0-0", count=count
            )
            if not result or not result[1]:
                return []
            return [(eid, fields) for eid, fields in result[1]]
        except Exception:
            return []

    def send_to_dlq(self, sname: str, event: KoiEvent, error: str) -> None:
        """Move a failed event to the dead letter queue."""
        dname = dlq_name(sname)
        fields = event.to_stream_dict()
        fields["error"] = error
        self._redis.xadd(dname, fields)
        log.warning("event.dlq", stream=sname, dlq=dname, event_id=event.id, error=error)

    def pending_count(self, sname: str, group: str) -> int:
        """Number of pending (unacked) messages in a consumer group."""
        try:
            info = self._redis.xpending(sname, group)
            return info.get("pending", 0) if isinstance(info, dict) else (info[0] if info else 0)
        except Exception:
            return 0

    def stream_length(self, sname: str) -> int:
        """Total entries in a stream."""
        try:
            return self._redis.xlen(sname)
        except Exception:
            return 0

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self._redis.ping()
        except Exception:
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        self._redis.close()
