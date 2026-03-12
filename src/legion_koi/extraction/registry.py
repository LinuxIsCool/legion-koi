"""Generic registry for pluggable backends — decorator-based factory.

Generalizes the create_embedder() pattern into a reusable registry.

Usage:
    ner_registry = Registry[EntityExtractor]("ner")

    @ner_registry.register("llm")
    class LLMEntityExtractor:
        ...

    extractor = ner_registry.create("llm", model="gpt-oss:120b")
"""

from typing import TypeVar, Generic

T = TypeVar("T")


class Registry(Generic[T]):
    """Named registry with decorator-based registration and factory creation."""

    def __init__(self, name: str):
        self.name = name
        self._backends: dict[str, type] = {}

    def register(self, name: str):
        """Decorator to register a backend class under a name."""
        def decorator(cls):
            self._backends[name] = cls
            return cls
        return decorator

    def create(self, name: str, **kwargs) -> T:
        """Instantiate a registered backend by name."""
        if name not in self._backends:
            available = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unknown {self.name} backend: {name!r}. Available: {available}"
            )
        return self._backends[name](**kwargs)

    def list_backends(self) -> list[str]:
        return sorted(self._backends)

    def has(self, name: str) -> bool:
        return name in self._backends
