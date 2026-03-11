"""Base class for sensors that poll SQLite databases."""

import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from . import state as sensor_state

log = structlog.stdlib.get_logger()


class DatabaseSensor(ABC):
    """Base class for sensors that poll SQLite databases for new/updated rows."""

    def __init__(
        self,
        db_path: Path,
        state_path: Path,
        kobj_push: callable,
        poll_interval: float = 30.0,
        batch_size: int = 0,
    ):
        self.db_path = Path(db_path).expanduser()
        self.state_path = Path(state_path)
        self.kobj_push = kobj_push
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.state = sensor_state.load(self.state_path)
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        """Open read-only connection to the SQLite database."""
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    @abstractmethod
    def poll(self) -> list[Bundle]:
        """Query DB for new/updated rows. Return Bundles."""
        ...

    def scan_all(self) -> list[Bundle]:
        """Full scan — calls poll() repeatedly if batch_size > 0."""
        if self.batch_size <= 0:
            return self.poll()

        all_bundles = []
        total = 0
        while True:
            batch = self.poll()
            if not batch:
                break
            all_bundles.extend(batch)
            total += len(batch)
            log.info(
                "scan.progress",
                sensor=self.__class__.__name__,
                batch=len(batch),
                total=total,
            )
        return all_bundles

    def _poll_loop(self):
        """Timer callback: poll, push bundles, save state, reschedule."""
        if not self._running:
            return
        with self._lock:
            try:
                bundles = self.poll()
                for bundle in bundles:
                    self.kobj_push(bundle=bundle)
                if bundles:
                    sensor_state.save(self.state_path, self.state)
                    log.info(
                        "poll.complete",
                        sensor=self.__class__.__name__,
                        count=len(bundles),
                    )
            except Exception:
                log.exception("poll.error", sensor=self.__class__.__name__)
        if self._running:
            self._timer = threading.Timer(self.poll_interval, self._poll_loop)
            self._timer.daemon = True
            self._timer.start()

    def start(self):
        """Start the polling loop."""
        if not self.db_path.exists():
            log.warning(
                "sensor.db_missing",
                path=str(self.db_path),
                sensor=self.__class__.__name__,
            )
            return
        self._running = True
        self._timer = threading.Timer(self.poll_interval, self._poll_loop)
        self._timer.daemon = True
        self._timer.start()
        log.info(
            "sensor.started",
            db_path=str(self.db_path),
            poll_interval=self.poll_interval,
            sensor=self.__class__.__name__,
        )

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            log.info("sensor.stopped", sensor=self.__class__.__name__)
