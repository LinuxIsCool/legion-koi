"""Persona sensor — watches persona data directories for new entries.

Slug-parameterized: watches ~/.claude/local/personas/data/{slug}/ for JSONL
and markdown files that contain persona observations, beliefs, facts, etc.

Reusable for any persona (darren, carolanne, etc.) by instantiating with
different slugs.
"""

import json
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.persona import LegionPersona
from . import state as sensor_state
from .base import BaseSensor

log = structlog.stdlib.get_logger()

# Supported file types and their item_type mapping
ITEM_TYPE_MAP = {
    "messages.jsonl": "observation",
    "beliefs.jsonl": "belief",
    "facts.jsonl": "fact",
    "relationships.jsonl": "relationship",
    "style-samples.jsonl": "style-sample",
    "decisions.jsonl": "decision",
}

# For markdown files in the memory directory
MEMORY_FILE_MAP = {
    "curated_facts.md": "fact",
    "daily_log.md": "observation",
}


class PersonaSensor(BaseSensor):
    """Sensor for persona data files.

    Args:
        slug: Persona slug (e.g. 'darren')
        watch_dir: Base persona data directory
        state_path: Path to sensor state JSON
        kobj_push: Bundle push callback
    """

    def __init__(
        self,
        slug: str,
        watch_dir: Path,
        state_path: Path,
        kobj_push: callable,
    ):
        self.slug = slug
        super().__init__(watch_dir=watch_dir, state_path=state_path, kobj_push=kobj_push)

    def should_process(self, path: Path) -> bool:
        return path.suffix in (".jsonl", ".md", ".json")

    def process_file(self, path: Path) -> Bundle | None:
        """Process a persona data file into a Bundle.

        JSONL files: each line becomes a separate bundle (handled by scan_all override).
        Markdown files: the whole file becomes one bundle.
        JSON files: the whole file becomes one bundle.
        """
        if path.suffix == ".md":
            return self._process_markdown(path)
        elif path.suffix == ".json":
            return self._process_json(path)
        # JSONL handled specially in scan_all
        return None

    def _process_markdown(self, path: Path) -> Bundle | None:
        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        item_type = MEMORY_FILE_MAP.get(path.name, "observation")
        identifier = path.stem

        reference = f"{self.slug}:{item_type}:{identifier}"
        change = sensor_state.has_changed(reference, content_hash, self.state)
        if change is None:
            return None

        rid = LegionPersona(slug=self.slug, item_type=item_type, identifier=identifier)
        contents = {
            "persona_slug": self.slug,
            "source_type": "markdown",
            "item_type": item_type,
            "content": text,
            "file_path": str(path),
        }

        log.info("persona.detected", change=change, rid=str(rid), slug=self.slug)
        return Bundle.generate(rid=rid, contents=contents)

    def _process_json(self, path: Path) -> Bundle | None:
        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        identifier = path.stem
        reference = f"{self.slug}:fact:{identifier}"
        change = sensor_state.has_changed(reference, content_hash, self.state)
        if change is None:
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("persona.invalid_json", path=str(path))
            return None

        rid = LegionPersona(slug=self.slug, item_type="fact", identifier=identifier)
        contents = {
            "persona_slug": self.slug,
            "source_type": "json",
            "item_type": "fact",
            "content": data,
            "file_path": str(path),
        }

        log.info("persona.detected", change="new", rid=str(rid), slug=self.slug)
        return Bundle.generate(rid=rid, contents=contents)

    def scan_all(self) -> list[Bundle]:
        """Full scan — handles JSONL files specially (each line = bundle)."""
        bundles = []
        if not self.watch_dir.exists():
            log.warning("sensor.watch_dir_missing", path=str(self.watch_dir))
            return bundles

        for path in sorted(self.watch_dir.rglob("*")):
            if not path.is_file():
                continue
            if not self.should_process(path):
                continue

            try:
                if path.suffix == ".jsonl":
                    bundles.extend(self._scan_jsonl(path))
                else:
                    bundle = self.process_file(path)
                    if bundle is not None:
                        bundles.append(bundle)
                        self.state[bundle.rid.reference] = sensor_state.compute_hash(
                            path.read_text(encoding="utf-8")
                        )
            except Exception:
                log.exception("sensor.scan_error", path=str(path))

        sensor_state.save(self.state_path, self.state)
        return bundles

    def _scan_jsonl(self, path: Path) -> list[Bundle]:
        """Process a JSONL file — each line becomes a bundle."""
        bundles = []
        item_type = ITEM_TYPE_MAP.get(path.name, "observation")

        # Use file-level hash to detect changes (not per-line)
        text = path.read_text(encoding="utf-8")
        file_hash = sensor_state.compute_hash(text)
        file_key = f"__file__{path.name}"

        change = sensor_state.has_changed(file_key, file_hash, self.state)
        if change is None:
            return bundles

        line_count = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Use record ID or line number as identifier
            record_id = record.get("id", str(line_count))
            identifier = f"{path.stem}-{record_id}"

            rid = LegionPersona(slug=self.slug, item_type=item_type, identifier=identifier)
            contents = {
                "persona_slug": self.slug,
                "source_type": "jsonl",
                "item_type": item_type,
                "content": record,
                "file_path": str(path),
            }
            bundles.append(Bundle.generate(rid=rid, contents=contents))
            line_count += 1

        self.state[file_key] = file_hash
        log.info("persona.jsonl_scanned", path=str(path), count=line_count, slug=self.slug)
        return bundles
