"""Abstract event consumer framework with consumer groups and DLQ.

Each consumer subscribes to a Redis Stream via a consumer group,
processes events, and routes failures to a dead letter queue.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

import structlog

from ..constants import EVENT_CONSUMER_POLL_BATCH, EVENT_CONSUMER_BLOCK_MS, EVENT_DLQ_MAX_RETRIES
from .schemas import KoiEvent
from .bus import EventBus, stream_name

log = structlog.stdlib.get_logger()


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
        """Main loop: read from stream, handle events, ack or DLQ."""
        while not self._stop.is_set():
            try:
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
            except Exception:
                if not self._stop.is_set():
                    log.warning("consumer.loop_error", group=self.group, exc_info=True)

    def _process_message(self, entry_id: str, fields: dict[str, str]) -> None:
        """Process one message: handle → ack, or retry → DLQ."""
        event = KoiEvent.from_stream_dict(fields)
        try:
            self.handle(event)
            self._bus.ack(self.stream, self.group, entry_id)
            # Clear retry count on success
            self._retry_counts.pop(event.id, None)
            log.debug("consumer.processed", group=self.group, event_id=event.id, subject=event.subject)
        except Exception as exc:
            retries = self._retry_counts.get(event.id, 0) + 1
            self._retry_counts[event.id] = retries

            if retries >= EVENT_DLQ_MAX_RETRIES:
                self._bus.send_to_dlq(self.stream, event, str(exc))
                self._bus.ack(self.stream, self.group, entry_id)
                self._retry_counts.pop(event.id, None)
                log.error(
                    "consumer.dlq",
                    group=self.group,
                    event_id=event.id,
                    retries=retries,
                    error=str(exc)[:200],
                )
            else:
                log.warning(
                    "consumer.retry",
                    group=self.group,
                    event_id=event.id,
                    attempt=retries,
                    error=str(exc)[:200],
                )
                # Ack to avoid re-delivery of same message, but the retry count
                # means subsequent events from same source will be tracked
                self._bus.ack(self.stream, self.group, entry_id)
