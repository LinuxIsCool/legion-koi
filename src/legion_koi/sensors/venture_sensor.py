"""Venture sensor — watches legion-brain ventures directory for markdown files."""

from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.venture import LegionVenture
from . import state as sensor_state
from .base import BaseSensor
from .parsing import parse_frontmatter

log = structlog.stdlib.get_logger()


class VentureSensor(BaseSensor):
    def should_process(self, path: Path) -> bool:
        return path.suffix == ".md"

    def process_file(self, path: Path) -> Bundle | None:
        # Stage from parent directory name (active, paused, dormant, etc.)
        stage = path.parent.name
        # ID from filename stem
        venture_id = path.stem

        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        reference = f"{stage}/{venture_id}"
        change = sensor_state.has_changed(reference, content_hash, self.state)
        if change is None:
            return None

        frontmatter, body = parse_frontmatter(text)

        rid = LegionVenture(stage=stage, id=venture_id)
        contents = {
            "frontmatter": frontmatter,
            "body": body,
            "file_path": str(path),
        }

        log.info(
            "venture.detected",
            change=change,
            rid=str(rid),
            title=frontmatter.get("title", venture_id),
        )
        return Bundle.generate(rid=rid, contents=contents)
