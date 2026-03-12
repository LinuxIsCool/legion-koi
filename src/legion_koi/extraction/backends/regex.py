"""Rule-based entity extraction — free, no API calls.

Extracts structured entities that have reliable patterns:
dates, URLs, file paths, code references, version numbers.
Augments LLM extraction with zero cost.
"""

import re

from ..models import Entity
from ..registry import Registry

# Reuse the NER registry from llm.py
from .llm import ner_registry


# --- Patterns ---

# ISO dates: 2026-03-11, 2026/03/11
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{2}[-/]\d{2})\b"
)

# HTTP/HTTPS URLs
_URL_RE = re.compile(
    r"https?://[^\s<>\"')\]]+",
)

# Unix file paths (at least 2 segments)
_PATH_RE = re.compile(
    r"(?:^|[\s(])((?:/[\w._-]+){2,}(?:/[\w._-]*)?)(?:\s|[),;:]|$)",
    re.MULTILINE,
)

# Python/JS module references: module.function(), Class.method
_CODE_REF_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"  # CamelCase class names
    r"|"
    r"\b(\w+(?:\.\w+)+)\(\)"  # dotted.function()
)

# Semantic version numbers: v1.2.3, 1.2.3-beta
_VERSION_RE = re.compile(
    r"\bv?(\d+\.\d+\.\d+(?:[-+][\w.]+)?)\b"
)

# ORN references: orn:namespace:reference
_ORN_RE = re.compile(
    r"\b(orn:[\w.-]+:[\w./_-]+)\b"
)


@ner_registry.register("regex")
class RegexEntityExtractor:
    """Rule-based extraction for structured entities with reliable patterns."""

    def extract_entities(self, text: str, entity_types: list[str], namespace: str = "") -> list[Entity]:
        if not text:
            return []

        entities = []

        # Dates → temporal supertype
        for m in _DATE_RE.finditer(text):
            entities.append(Entity(
                name=m.group(1),
                entity_type="Date",
                supertype="temporal",
                confidence=1.0,
            ))

        # URLs → place supertype
        for m in _URL_RE.finditer(text):
            url = m.group(0).rstrip(".,;:)")
            entities.append(Entity(
                name=url,
                entity_type="URL",
                supertype="place",
                confidence=1.0,
            ))

        # File paths → place supertype
        for m in _PATH_RE.finditer(text):
            path = m.group(1).rstrip(".,;:!?")
            if path and len(path) > 3:
                entities.append(Entity(
                    name=path,
                    entity_type="Path",
                    supertype="place",
                    confidence=0.9,
                ))

        # ORN references → artifact supertype
        for m in _ORN_RE.finditer(text):
            entities.append(Entity(
                name=m.group(1),
                entity_type="RID",
                supertype="artifact",
                confidence=1.0,
            ))

        # Version numbers → artifact supertype
        for m in _VERSION_RE.finditer(text):
            entities.append(Entity(
                name=m.group(1),
                entity_type="Version",
                supertype="artifact",
                confidence=0.9,
            ))

        # Deduplicate by (name, type)
        seen = set()
        deduped = []
        for e in entities:
            key = (e.name, e.entity_type)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        return deduped

    def get_name(self) -> str:
        return "regex"

    def is_available(self) -> bool:
        return True
