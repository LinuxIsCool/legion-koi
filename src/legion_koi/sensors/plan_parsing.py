"""Plan metadata extraction -- no YAML frontmatter, parse from markdown content."""

import re

DATED_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-")
# Real agent IDs are 17+ hex chars; require 8+ to avoid false positives like "build-agent-abc"
SUBAGENT_PATTERN = re.compile(r"^(.+)-agent-a[0-9a-f]{8,}$")
# H1 is always in the first ~50 lines; split(n) yields n+1 parts so use n-1
H1_SCAN_LIMIT = 50


def classify_plan(stem: str) -> tuple[str, str]:
    """Classify plan type and extract parent slug from filename stem.

    Returns (plan_type, parent_slug).
    plan_type: "dated" | "subagent" | "auto"
    parent_slug: non-empty only for subagent plans
    """
    m = SUBAGENT_PATTERN.match(stem)
    if m:
        return "subagent", m.group(1)
    if DATED_PATTERN.match(stem):
        return "dated", ""
    return "auto", ""


def extract_h1(text: str) -> str:
    """Extract first H1 title from markdown. Returns '' if none found."""
    for line in text.split("\n", H1_SCAN_LIMIT - 1):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return ""


def extract_bold_field(text: str, field: str) -> str:
    """Extract value from **Field:** pattern. Returns '' if not found."""
    pattern = re.compile(rf"\*\*{re.escape(field)}:\*\*\s*(.+)", re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""
