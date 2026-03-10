"""Journal sensor — watches legion-brain journal directory for markdown entries."""

import re
from datetime import date, datetime
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.journal import LegionJournal
from . import state as sensor_state
from .base import BaseSensor

log = structlog.stdlib.get_logger()

# Matches paths like .../YYYY/MM/DD/HHMM-slug.md or .../YYYY/MM/DD/HH-MM-slug.md
DATE_DIR_PATTERN = re.compile(r".*/(\d{4})/(\d{2})/(\d{2})/(.+)\.md$")


def _make_serializable(obj):
    """Recursively convert ruamel types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body. Returns (frontmatter_dict, body)."""
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    from io import StringIO
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        fm = yaml.load(StringIO(parts[1])) or {}
        fm = _make_serializable(fm)
    except Exception:
        fm = {}
    body = parts[2].strip()
    return fm, body


def extract_rid_parts(path: Path) -> tuple[str, str] | None:
    """Extract (date, slug) from a journal file path.

    Expected: .../YYYY/MM/DD/<time-slug>.md
    Returns: ('YYYY-MM-DD', '<time-slug>')
    """
    match = DATE_DIR_PATTERN.search(str(path))
    if not match:
        return None
    year, month, day, filename = match.groups()
    date = f"{year}-{month}-{day}"
    return date, filename


class JournalSensor(BaseSensor):
    def should_process(self, path: Path) -> bool:
        if path.suffix != ".md":
            return False
        # Must match date directory structure
        return extract_rid_parts(path) is not None

    def process_file(self, path: Path) -> Bundle | None:
        parts = extract_rid_parts(path)
        if parts is None:
            return None

        date, slug = parts
        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        reference = f"{date}/{slug}"
        change = sensor_state.has_changed(reference, content_hash, self.state)
        if change is None:
            return None

        frontmatter, body = parse_frontmatter(text)

        # Only process atomic entries (skip daily/monthly/yearly summaries)
        entry_type = frontmatter.get("type", "")
        if entry_type and entry_type != "atomic":
            return None

        rid = LegionJournal(date=date, slug=slug)
        contents = {
            "frontmatter": frontmatter,
            "body": body,
            "file_path": str(path),
        }

        log.info(
            "journal.detected",
            change=change,
            rid=str(rid),
            title=frontmatter.get("title", ""),
        )
        return Bundle.generate(rid=rid, contents=contents)
