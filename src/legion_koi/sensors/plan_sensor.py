"""Plan sensor -- watches ~/.claude/plans/ for markdown plan files."""

from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.plan import LegionPlan
from . import state as sensor_state
from .base import BaseSensor
from .plan_parsing import classify_plan, extract_h1, extract_bold_field

log = structlog.stdlib.get_logger()


class PlanSensor(BaseSensor):
    def should_process(self, path: Path) -> bool:
        return path.suffix == ".md"

    def process_file(self, path: Path) -> Bundle | None:
        slug = path.stem
        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        change = sensor_state.has_changed(slug, content_hash, self.state)
        if change is None:
            return None

        plan_type, parent_slug = classify_plan(slug)
        title = extract_h1(text) or slug
        goal = extract_bold_field(text, "Goal")

        rid = LegionPlan(slug=slug)
        contents = {
            "title": title,
            "goal": goal,
            "plan_type": plan_type,
            "parent_slug": parent_slug,
            "body": text,
            "file_path": str(path),
        }

        log.info(
            "plan.detected",
            change=change,
            rid=str(rid),
            title=title,
            plan_type=plan_type,
        )
        return Bundle.generate(rid=rid, contents=contents)
