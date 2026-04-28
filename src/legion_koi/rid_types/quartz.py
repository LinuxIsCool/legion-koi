"""LegionQuartzPage RID — identity for Quartz-rendered pages.

Phase 3 D2 of claude-quartz roadmap. Each Quartz page has a stable RID
derived from its slug. Bundle contents change as pages update; RID does not.

Reference is the canonical Quartz slug (URL-style, no leading slash, no .html).
Examples:
  - "dreams/README"
  - "journal/2026-04-27/12-21-phase-1-v0-ship-claude-quartz"
  - "prompts/skill-anatomy"
"""
from rid_lib.core import ORN


class LegionQuartzPage(ORN):
    namespace = "legion.claude-quartz"

    def __init__(self, slug: str):
        if not slug or not isinstance(slug, str):
            raise ValueError(f"LegionQuartzPage slug must be non-empty str, got {slug!r}")
        # Normalize: strip leading/trailing slashes, drop .html if present.
        slug = slug.strip().lstrip("/").rstrip("/")
        if slug.endswith(".html"):
            slug = slug[:-5]
        self.slug = slug

    @property
    def reference(self) -> str:
        return self.slug

    @classmethod
    def from_reference(cls, reference: str):
        if not reference:
            raise ValueError("LegionQuartzPage reference must be non-empty")
        return cls(slug=reference)
