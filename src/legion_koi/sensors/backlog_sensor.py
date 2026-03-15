"""Backlog sensor — watches backlog directory for task markdown files."""

from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.task import LegionTask
from . import state as sensor_state
from .base import BaseSensor
from .parsing import parse_frontmatter

log = structlog.stdlib.get_logger()


class BacklogSensor(BaseSensor):
    def should_process(self, path: Path) -> bool:
        return path.name.startswith("task-") and path.suffix == ".md"

    def process_file(self, path: Path) -> Bundle | None:
        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        frontmatter, body = parse_frontmatter(text)

        # Task ID from frontmatter 'id' field, falling back to filename prefix
        task_id = str(frontmatter.get("id", ""))
        if not task_id:
            # Filename like "task-009 - rhythms-agent-overhaul.md"
            task_id = path.stem.split(" - ")[0]

        change = sensor_state.has_changed(task_id, content_hash, self.state)
        if change is None:
            return None

        rid = LegionTask(task_id=task_id)
        contents = {
            "frontmatter": frontmatter,
            "body": body,
            "file_path": str(path),
        }

        log.info(
            "task.detected",
            change=change,
            rid=str(rid),
            title=frontmatter.get("title", task_id),
        )
        return Bundle.generate(rid=rid, contents=contents)
