"""Simple JSON state persistence — tracks what we've already seen by SHA256 hash."""

import hashlib
import json
from pathlib import Path


def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def load(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def has_changed(reference: str, current_hash: str, state: dict[str, str]) -> str | None:
    """Returns 'NEW', 'UPDATE', or None if unchanged."""
    if reference not in state:
        return "NEW"
    if state[reference] != current_hash:
        return "UPDATE"
    return None
