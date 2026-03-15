"""Dock sensor — watches generated skill directory for SKILL.md changes."""

import json
import re
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.dock import LegionDock
from . import state as sensor_state
from .base import BaseSensor
from .parsing import parse_frontmatter

log = structlog.stdlib.get_logger()

# Matches paths like .../generated/{owner}/{repo}/SKILL.md
SKILL_PATH_PATTERN = re.compile(r".*/generated/([^/]+)/([^/]+)/SKILL\.md$")

QUALITY_HISTORY_PATH = Path.home() / ".claude/local/dock/feedback/quality-history.jsonl"


def extract_rid_parts(path: Path) -> tuple[str, str] | None:
    """Extract (owner, repo) from a generated SKILL.md path.

    Expected: .../generated/{owner}/{repo}/SKILL.md
    Returns: ('owner', 'repo')
    """
    match = SKILL_PATH_PATTERN.search(str(path))
    if not match:
        return None
    return match.group(1), match.group(2)


def read_version_file(repo_dir: Path) -> dict | None:
    """Read .version file and return parsed fields."""
    version_path = repo_dir / ".version"
    if not version_path.exists():
        return None

    result = {}
    try:
        for line in version_path.read_text().splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
    except OSError:
        return None
    return result


def read_latest_quality(owner: str, repo: str) -> dict | None:
    """Read the latest quality entry for this repo from quality-history.jsonl."""
    if not QUALITY_HISTORY_PATH.exists():
        return None

    repo_key = f"{owner}/{repo}"
    latest = None
    try:
        for line in QUALITY_HISTORY_PATH.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("repo") == repo_key or entry.get("name") == f"{owner}--{repo}":
                latest = entry
    except (OSError, json.JSONDecodeError):
        pass
    return latest


class DockSensor(BaseSensor):
    def should_process(self, path: Path) -> bool:
        if path.name != "SKILL.md":
            return False
        return extract_rid_parts(path) is not None

    def process_file(self, path: Path) -> Bundle | None:
        parts = extract_rid_parts(path)
        if parts is None:
            return None

        owner, repo = parts
        text = path.read_text(encoding="utf-8")
        content_hash = sensor_state.compute_hash(text)

        reference = f"{owner}/{repo}"
        change = sensor_state.has_changed(reference, content_hash, self.state)
        if change is None:
            return None

        frontmatter, _body = parse_frontmatter(text)
        repo_dir = path.parent

        version = read_version_file(repo_dir)
        analyzed = (repo_dir / ".analyzed").exists()
        quality = read_latest_quality(owner, repo)

        rid = LegionDock(owner=owner, repo=repo)
        contents = {
            "frontmatter": frontmatter,
            "version": version,
            "analyzed": analyzed,
            "quality": quality,
            "file_path": str(path),
        }

        log.info(
            "dock.detected",
            change=change,
            rid=str(rid),
            name=frontmatter.get("name", f"{owner}/{repo}"),
        )
        return Bundle.generate(rid=rid, contents=contents)
