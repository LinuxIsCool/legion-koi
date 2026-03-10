"""Shared parsing utilities for sensors."""

from datetime import date, datetime


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
