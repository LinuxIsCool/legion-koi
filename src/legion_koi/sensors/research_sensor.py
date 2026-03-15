"""Research sensor — watches legion-brain research directory for markdown studies."""

from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.research import LegionResearch
from . import state as sensor_state
from .base import BaseSensor
from .parsing import parse_frontmatter

log = structlog.stdlib.get_logger()


class ResearchSensor(BaseSensor):
    def should_process(self, path: Path) -> bool:
        return path.suffix == ".md"

    def process_file(self, path: Path) -> Bundle | None:
        # Compute relative path from watch_dir as slug
        try:
            rel = path.relative_to(self.watch_dir)
        except ValueError:
            return None
        slug = str(rel.with_suffix(""))  # "hippo/retrieval-patterns" or "a2a-identity"

        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        change = sensor_state.has_changed(slug, content_hash, self.state)
        if change is None:
            return None

        frontmatter, body = parse_frontmatter(text)

        rid = LegionResearch(slug=slug)
        contents = {
            "frontmatter": frontmatter,
            "body": body,
            "file_path": str(path),
        }

        log.info(
            "research.detected",
            change=change,
            rid=str(rid),
            title=frontmatter.get("title", slug),
        )
        return Bundle.generate(rid=rid, contents=contents)
