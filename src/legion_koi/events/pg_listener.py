"""PG LISTEN → Redis Streams bridge.

Listens on PostgreSQL 'koi_events' channel for bundle change notifications,
wraps them in CloudEvents envelopes, and publishes to Redis Streams.

Runs as a background thread started from __main__.py.
"""

from __future__ import annotations

import json
import threading

import psycopg
import structlog

from ..constants import EVENT_PG_CHANNEL
from .schemas import KoiEvent, BUNDLE_CREATED, BUNDLE_UPDATED
from .bus import EventBus

log = structlog.stdlib.get_logger()


class PgListener:
    """Bridges PostgreSQL NOTIFY events to Redis Streams.

    Uses psycopg's synchronous LISTEN/NOTIFY in a dedicated thread.
    The thread polls with a timeout so it can check the stop flag.
    """

    def __init__(self, dsn: str, bus: EventBus):
        self._dsn = dsn
        self._bus = bus
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the listener thread."""
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="pg-listener",
            daemon=True,
        )
        self._thread.start()
        log.info("pg_listener.started", channel=EVENT_PG_CHANNEL)

    def stop(self) -> None:
        """Signal the listener to stop and wait for thread exit."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        log.info("pg_listener.stopped")

    def _listen_loop(self) -> None:
        """Main loop: LISTEN on PG, bridge notifications to Redis."""
        try:
            conn = psycopg.connect(self._dsn, autocommit=True)
            conn.execute(f"LISTEN {EVENT_PG_CHANNEL}")
            log.info("pg_listener.listening", channel=EVENT_PG_CHANNEL)

            while not self._stop.is_set():
                # poll with 1-second timeout so we can check the stop flag
                gen = conn.notifies(timeout=1.0)
                for notify in gen:
                    try:
                        self._handle_notify(notify)
                    except Exception:
                        log.warning("pg_listener.handle_error", exc_info=True)
                    # Check stop between notifications
                    if self._stop.is_set():
                        break
                    # Break out of generator to re-enter the while loop
                    # (notifies() blocks until timeout or notification)
                    break

        except Exception:
            if not self._stop.is_set():
                log.exception("pg_listener.fatal")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handle_notify(self, notify) -> None:
        """Parse a PG notification and publish as a KoiEvent."""
        payload = json.loads(notify.payload)
        op = payload.get("op", "INSERT")
        rid = payload["rid"]
        namespace = payload["namespace"]

        event_type = BUNDLE_CREATED if op == "INSERT" else BUNDLE_UPDATED
        event = KoiEvent(
            type=event_type,
            subject=rid,
            data={"rid": rid, "namespace": namespace},
        )
        self._bus.publish(event)
