"""Protocol interfaces for extraction backends — structural subtyping.

Mirrors the Embedder/Reranker Protocol pattern from embeddings.py and reranking.py.
Backends implement these methods without inheriting from a base class.
"""

from typing import Protocol, runtime_checkable

from .models import Entity


@runtime_checkable
class EntityExtractor(Protocol):
    """Provider-agnostic entity extraction interface."""

    def extract_entities(self, text: str, entity_types: list[str], namespace: str = "") -> list[Entity]:
        """Extract entities from text, filtering to the given types."""
        ...

    def get_name(self) -> str:
        """Backend identifier (e.g. 'llm', 'regex')."""
        ...

    def is_available(self) -> bool:
        """Check if this backend can run (API keys, models, etc.)."""
        ...


@runtime_checkable
class EntityResolver(Protocol):
    """Deduplicates and merges entities across chunks/documents."""

    def resolve(self, entities: list[Entity]) -> list[Entity]:
        """Merge duplicates, normalize names, pick highest confidence."""
        ...

    def get_name(self) -> str:
        ...
