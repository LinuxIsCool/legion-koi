"""Base sensor with watchdog filesystem monitoring."""

import threading
from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from rid_lib.ext import Bundle
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import state as sensor_state

log = structlog.stdlib.get_logger()


class _FileHandler(FileSystemEventHandler):
    def __init__(self, sensor: "BaseSensor"):
        self.sensor = sensor

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.sensor._on_file_event(Path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.sensor._on_file_event(Path(event.src_path))


class BaseSensor(ABC):
    def __init__(
        self,
        watch_dir: Path,
        state_path: Path,
        kobj_push: callable,
    ):
        self.watch_dir = Path(watch_dir).expanduser()
        self.state_path = Path(state_path)
        self.kobj_push = kobj_push
        self.state = sensor_state.load(self.state_path)
        self._observer: Observer | None = None
        self._lock = threading.Lock()

    @abstractmethod
    def process_file(self, path: Path) -> Bundle | None:
        """Process a single file and return a Bundle, or None to skip."""
        ...

    @abstractmethod
    def should_process(self, path: Path) -> bool:
        """Return True if this file should be processed by this sensor."""
        ...

    def _on_file_event(self, path: Path) -> None:
        if not self.should_process(path):
            return
        with self._lock:
            try:
                bundle = self.process_file(path)
                if bundle is not None:
                    self.kobj_push(bundle=bundle)
                    self.state[bundle.rid.reference] = bundle.manifest.sha256_hash
                    sensor_state.save(self.state_path, self.state)
                    log.info("sensor.processed", rid=str(bundle.rid), sensor=self.__class__.__name__)
            except Exception:
                log.exception("sensor.error", path=str(path), sensor=self.__class__.__name__)

    def scan_all(self) -> list[Bundle]:
        """Full scan of the watch directory. Returns all new/updated bundles."""
        bundles = []
        if not self.watch_dir.exists():
            log.warning("sensor.watch_dir_missing", path=str(self.watch_dir))
            return bundles

        for path in sorted(self.watch_dir.rglob("*")):
            if path.is_file() and self.should_process(path):
                try:
                    bundle = self.process_file(path)
                    if bundle is not None:
                        bundles.append(bundle)
                        self.state[bundle.rid.reference] = bundle.manifest.sha256_hash
                except Exception:
                    log.exception("sensor.scan_error", path=str(path))

        sensor_state.save(self.state_path, self.state)
        return bundles

    def start(self) -> None:
        if not self.watch_dir.exists():
            log.warning("sensor.watch_dir_missing", path=str(self.watch_dir))
            return
        self._observer = Observer()
        self._observer.schedule(_FileHandler(self), str(self.watch_dir), recursive=True)
        self._observer.daemon = True
        self._observer.start()
        log.info("sensor.started", watch_dir=str(self.watch_dir), sensor=self.__class__.__name__)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            log.info("sensor.stopped", sensor=self.__class__.__name__)
