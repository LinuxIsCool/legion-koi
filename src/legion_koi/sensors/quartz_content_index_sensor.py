"""Quartz contentIndex.json sensor — Phase 3 D1.

Watches ~/.claude/local/quartz/public/static/contentIndex.json. Each scan reads
the file (one big JSON dict, slug → entry), computes per-entry sha256, and
emits Bundle objects for new/changed entries into the legion.claude-quartz
namespace.

This is a one-file-many-bundles sensor — distinct from the per-file pattern
that JournalSensor + co. follow. We override scan_all() and _on_file_event()
to translate a single file mutation into N bundle emissions.

Standalone path (CURRENT — Phase 3 V0.2)
----------------------------------------
build-rhythm.sh fires this module directly after each successful atomic public
swap. We connect to PostgresStorage and upsert bundles ourselves. This is the
ONLY execution surface today.

  python3 -m legion_koi.sensors.quartz_content_index_sensor

Important: standalone path does NOT call storage.initialize() — assumes the
live legion-koi node has already created the schema + triggers. Calling
initialize() on a live DB races against NOTIFY listeners (re-attaches the
bundles_notify trigger).

Long-running path (FUTURE — Phase 4+, NOT YET WIRED)
----------------------------------------------------
The scan_all()-returns-list[Bundle] shape and watchdog start() method exist
ready for legion-koi __main__.py to instantiate this sensor like the other
17. Today __main__.py does NOT include QuartzContentIndexSensor in
all_sensors. Wiring requires:
  1. Import + instantiate alongside JournalSensor / VentureSensor / etc.
  2. Append to all_sensors at __main__.py:292.
  3. Decide watch_dir semantics — public/static/ is re-created on every
     atomic swap, so the watchdog Observer needs a re-arm hook on the swap
     event. Easier: keep build-rhythm as the trigger and skip watchdog.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from rid_lib.ext import Bundle

from ..rid_types.quartz import LegionQuartzPage
from . import state as sensor_state
from .base import BaseSensor

log = structlog.stdlib.get_logger()

# Default vault location — overridable via env var if quartz vault relocates.
DEFAULT_CONTENT_INDEX = Path.home() / ".claude/local/quartz/public/static/contentIndex.json"
DEFAULT_STATE_PATH = Path.home() / ".claude/local/quartz/sensor-state.json"

# Per-bundle content cap. contentIndex stores the full rendered text per page;
# at 70MB across 5,500 pages we average ~13KB/page but worst-case (a large
# research synthesis) can hit 200KB+. 50KB cap keeps bundle rows reasonable
# while preserving most semantic content for downstream embed/extract.
MAX_CONTENT_BYTES = 50_000

# State persistence cadence in scan loops. Save dedup state every N entries
# so a process kill mid-loop (OOM, SIGTERM) does not lose all dedup
# bookkeeping (review §4.1 fix). 500 chosen because state file ~760KB,
# atomic write ~10ms — negligible at every-500.
STATE_SAVE_EVERY = 500


@dataclass
class ScanResult:
    total_entries: int
    new_count: int
    update_count: int
    skipped_unchanged: int
    error_count: int

    def summary(self) -> str:
        return (
            f"total={self.total_entries} new={self.new_count} "
            f"updated={self.update_count} unchanged={self.skipped_unchanged} "
            f"errors={self.error_count}"
        )


def _build_bundle(slug: str, entry: dict, base_url: str) -> tuple[Bundle, str] | None:
    """Build (Bundle, sha256) for a single contentIndex entry. None on malformed."""
    if not isinstance(entry, dict):
        return None
    content_text = (entry.get("content") or "")[:MAX_CONTENT_BYTES]
    fm = entry.get("frontmatter") or {}
    contents = {
        "title": entry.get("title", ""),
        "tags": entry.get("tags") or [],
        "links": entry.get("links") or [],
        "content": content_text,
        "content_truncated": len(entry.get("content") or "") > MAX_CONTENT_BYTES,
        "frontmatter": fm,
        "permalink": f"{base_url}/{slug}",
        "source_plugin": fm.get("source_plugin"),
        "legion_visibility": fm.get("legion_visibility", "private"),
        "schema_version": fm.get("schema_version"),
    }
    sha = sensor_state.compute_hash(json.dumps(contents, sort_keys=True, default=str))
    rid = LegionQuartzPage(slug=slug)
    bundle = Bundle.generate(rid=rid, contents=contents)
    return bundle, sha


def _read_index(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        log.warning("quartz.contentindex.missing", path=str(path))
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        log.error("quartz.contentindex.malformed_json", err=str(e))
        return None
    if not isinstance(data, dict):
        log.error("quartz.contentindex.malformed_root", type=type(data).__name__)
        return None
    return data


class QuartzContentIndexSensor(BaseSensor):
    """One-file-many-bundles sensor on Quartz contentIndex.json."""

    def __init__(
        self,
        content_index_path: Path,
        state_path: Path,
        kobj_push,
        base_url: str = "http://legion:4321",
    ):
        # BaseSensor expects watch_dir; we pass the parent directory of the file.
        super().__init__(
            watch_dir=content_index_path.parent,
            state_path=state_path,
            kobj_push=kobj_push,
        )
        self.content_index_path = content_index_path
        self.base_url = base_url

    # ---- BaseSensor abstract contract (file-mode entry points) ----

    def should_process(self, path: Path) -> bool:
        # Only the contentIndex.json file matters for this sensor.
        try:
            return path.resolve() == self.content_index_path.resolve()
        except FileNotFoundError:
            return path == self.content_index_path

    def process_file(self, path: Path) -> Bundle | None:
        # Per-file path is a no-op; we use scan_all() for everything.
        # Returning None ensures BaseSensor._on_file_event short-circuits cleanly.
        return None

    # ---- Override watchdog hook so a contentIndex.json mutation triggers re-scan ----

    def _on_file_event(self, path: Path) -> None:
        if not self.should_process(path):
            return
        with self._lock:
            try:
                self._scan_and_emit(push=True)
            except Exception:
                log.exception("quartz.scan.failed")

    # ---- Override scan_all to use the multi-entry pattern ----

    def scan_all(self) -> list[Bundle]:
        bundles: list[Bundle] = []
        result = self._scan_and_emit(push=False, collect=bundles)
        log.info("quartz.scan.complete", **result.__dict__)
        return bundles

    # ---- Core scan logic ----

    def _scan_and_emit(
        self,
        *,
        push: bool,
        collect: list[Bundle] | None = None,
    ) -> ScanResult:
        """Read contentIndex, dedup, push or collect bundles. Returns ScanResult."""
        result = ScanResult(0, 0, 0, 0, 0)
        index = _read_index(self.content_index_path)
        if index is None:
            return result

        result.total_entries = len(index)
        # Build set of slugs we saw this scan, for forward orphan detection.
        seen: set[str] = set()
        # Periodic state save (review §4.1) — survive partial scans.
        emitted_since_save = 0
        for slug, entry in index.items():
            try:
                built = _build_bundle(slug, entry, self.base_url)
                if built is None:
                    result.error_count += 1
                    if result.error_count <= 10:  # cap log spam (review §4.2)
                        log.info(
                            "quartz.entry.skip_non_dict",
                            slug=slug,
                            entry_type=type(entry).__name__,
                        )
                    continue
                bundle, sha = built
                seen.add(slug)
                change = sensor_state.has_changed(slug, sha, self.state)
                if change is None:
                    result.skipped_unchanged += 1
                    continue
                if change == "NEW":
                    result.new_count += 1
                else:
                    result.update_count += 1
                if push:
                    self.kobj_push(bundle=bundle)
                if collect is not None:
                    collect.append(bundle)
                self.state[slug] = sha
                emitted_since_save += 1
                if emitted_since_save >= STATE_SAVE_EVERY:
                    try:
                        sensor_state.save(self.state_path, self.state)
                    except Exception:
                        log.exception(
                            "quartz.state.partial_save_failed",
                            path=str(self.state_path),
                        )
                    emitted_since_save = 0
            except Exception as e:
                result.error_count += 1
                log.error("quartz.entry.fail", slug=slug, err=str(e))

        # Persist state after scan (final flush)
        try:
            sensor_state.save(self.state_path, self.state)
        except Exception:
            log.exception("quartz.state.save_failed", path=str(self.state_path))

        return result


# ---- Standalone CLI (build-rhythm hook entrypoint) ----

def _standalone_scan(
    content_index_path: Path = DEFAULT_CONTENT_INDEX,
    state_path: Path = DEFAULT_STATE_PATH,
    base_url: str = "http://legion:4321",
) -> int:
    """Standalone scan: connect to PostgresStorage, upsert changed bundles, exit.

    Returns process exit code (0 = success, 1 = no contentIndex, 2 = DB failure).
    """
    from ..config import LegionKoiConfig
    from ..storage.postgres import PostgresStorage

    cfg = LegionKoiConfig()
    try:
        storage = PostgresStorage(dsn=cfg.postgres.dsn)
        # Review §3.2: do NOT call storage.initialize() — assumes the live
        # legion-koi node has already created tables + triggers. Initialize
        # would re-run DDL that races against the live process's NOTIFY
        # listeners (re-attaching bundles_notify trigger drops in-flight
        # events). Eagerly probe the connection instead.
        storage._get_conn()
    except Exception as e:
        log.error("quartz.standalone.db_unavailable", err=str(e))
        return 2

    state = sensor_state.load(state_path)
    index = _read_index(content_index_path)
    if index is None:
        return 1

    new_count = update_count = unchanged = err = 0
    emitted_since_save = 0
    for slug, entry in index.items():
        try:
            built = _build_bundle(slug, entry, base_url)
            if built is None:
                err += 1
                if err <= 10:
                    log.info(
                        "quartz.standalone.skip_non_dict",
                        slug=slug,
                        entry_type=type(entry).__name__,
                    )
                continue
            bundle, sha = built
            change = sensor_state.has_changed(slug, sha, state)
            if change is None:
                unchanged += 1
                continue
            if change == "NEW":
                new_count += 1
            else:
                update_count += 1
            storage.upsert_bundle(
                rid=str(bundle.rid),
                namespace=bundle.rid.namespace,
                reference=bundle.rid.reference,
                contents=bundle.contents,
                sha256_hash=sha,
            )
            state[slug] = sha
            emitted_since_save += 1
            if emitted_since_save >= STATE_SAVE_EVERY:
                # Periodic state save (review §4.1) — survive SIGTERM mid-loop.
                try:
                    sensor_state.save(state_path, state)
                except Exception:
                    log.exception(
                        "quartz.standalone.partial_save_failed",
                        path=str(state_path),
                    )
                emitted_since_save = 0
        except Exception as e:
            err += 1
            log.exception("quartz.standalone.entry_fail", slug=slug, err=str(e))

    sensor_state.save(state_path, state)
    log.info(
        "quartz.standalone.complete",
        total=len(index),
        new=new_count,
        updated=update_count,
        unchanged=unchanged,
        errors=err,
    )
    print(
        f"quartz-sensor: total={len(index)} new={new_count} "
        f"updated={update_count} unchanged={unchanged} errors={err}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_standalone_scan())
