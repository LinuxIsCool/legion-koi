"""Abstract event consumer framework with consumer groups and DLQ.

Each consumer subscribes to a Redis Stream via a consumer group,
processes events, and routes failures to a dead letter queue.

Retry semantics: on failure, the message is NOT acknowledged — Redis
will redeliver it on the next XREADGROUP call (via pending entries list).
After EVENT_DLQ_MAX_RETRIES failures for the same entry, the message is
moved to the dead letter queue and acknowledged.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import structlog

from ..constants import EVENT_CONSUMER_POLL_BATCH, EVENT_CONSUMER_BLOCK_MS, EVENT_DLQ_MAX_RETRIES
from .schemas import KoiEvent
from .bus import EventBus, stream_name

log = structlog.stdlib.get_logger()

# Max tracked pending entries before pruning stale ones
_MAX_RETRY_TRACKING = 1000


class EventConsumer(ABC):
    """Base class for event consumers.

    Subclasses define:
    - event_type: which event type to subscribe to
    - group: consumer group name (shared across instances for load balancing)
    - handle(): the actual processing logic
    """

    event_type: str
    group: str

    def __init__(self, bus: EventBus, consumer_id: str | None = None):
        self._bus = bus
        self._consumer_id = consumer_id or f"{self.group}-0"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Track retry counts by stream entry_id (not event.id)
        self._retry_counts: dict[str, int] = {}

    @abstractmethod
    def handle(self, event: KoiEvent) -> None:
        """Process a single event. Raise on failure to trigger retry/DLQ."""
        ...

    @property
    def stream(self) -> str:
        return stream_name(self.event_type)

    def start(self) -> None:
        """Start consuming in a background thread."""
        self._bus.ensure_group(self.stream, self.group)
        self._thread = threading.Thread(
            target=self._consume_loop,
            name=f"consumer-{self.group}",
            daemon=True,
        )
        self._thread.start()
        log.info("consumer.started", group=self.group, stream=self.stream)

    def stop(self) -> None:
        """Signal the consumer to stop."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("consumer.stopped", group=self.group)

    def _consume_loop(self) -> None:
        """Main loop: read new messages, then claim pending (unacked) for retry."""
        while not self._stop.is_set():
            try:
                # First: read NEW messages from the stream
                messages = self._bus.read_group(
                    self.stream,
                    self.group,
                    self._consumer_id,
                    count=EVENT_CONSUMER_POLL_BATCH,
                    block_ms=EVENT_CONSUMER_BLOCK_MS,
                )
                for entry_id, fields in messages:
                    if self._stop.is_set():
                        break
                    self._process_message(entry_id, fields)

                # Then: claim and retry pending messages (previously failed)
                self._process_pending()

            except Exception:
                if not self._stop.is_set():
                    log.warning("consumer.loop_error", group=self.group, exc_info=True)

    def _process_pending(self) -> None:
        """Reclaim and retry pending (unacked) messages from previous failures."""
        try:
            pending = self._bus.claim_pending(
                self.stream,
                self.group,
                self._consumer_id,
                min_idle_ms=5000,  # Only retry messages idle for 5+ seconds
                count=EVENT_CONSUMER_POLL_BATCH,
            )
            for entry_id, fields in pending:
                if self._stop.is_set():
                    break
                self._process_message(entry_id, fields)
        except Exception:
            pass  # Pending claim failures are non-critical

    def _process_message(self, entry_id: str, fields: dict[str, str]) -> None:
        """Process one message: handle → ack, or leave unacked for retry → DLQ."""
        event = KoiEvent.from_stream_dict(fields)
        try:
            self.handle(event)
            self._bus.ack(self.stream, self.group, entry_id)
            self._retry_counts.pop(entry_id, None)
            log.debug("consumer.processed", group=self.group, event_id=event.id, subject=event.subject)
        except Exception as exc:
            retries = self._retry_counts.get(entry_id, 0) + 1
            self._retry_counts[entry_id] = retries

            if retries >= EVENT_DLQ_MAX_RETRIES:
                # Max retries exhausted — move to dead letter queue
                self._bus.send_to_dlq(self.stream, event, str(exc))
                self._bus.ack(self.stream, self.group, entry_id)
                self._retry_counts.pop(entry_id, None)
                log.error(
                    "consumer.dlq",
                    group=self.group,
                    event_id=event.id,
                    retries=retries,
                    error=str(exc),
                )
            else:
                # Do NOT ack — Redis will redeliver via pending entries list
                log.warning(
                    "consumer.retry_pending",
                    group=self.group,
                    event_id=event.id,
                    attempt=retries,
                    error=str(exc),
                )

            # Prune stale retry tracking entries to prevent unbounded growth
            if len(self._retry_counts) > _MAX_RETRY_TRACKING:
                oldest_keys = sorted(self._retry_counts.keys())[:_MAX_RETRY_TRACKING // 2]
                for k in oldest_keys:
                    self._retry_counts.pop(k, None)
