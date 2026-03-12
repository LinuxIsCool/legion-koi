"""Composable ontology — entity type definitions loaded from YAML config.

Two-tier type system: 7 universal supertypes with namespace-specific subtypes.
Adding a new entity type = edit ontology.yaml, no Python changes.
"""

from pathlib import Path

import structlog
import yaml

log = structlog.stdlib.get_logger()

_CONFIGS_DIR = Path(__file__).parent / "configs"


class OntologyRegistry:
    """Loads entity type definitions from YAML and maps namespaces to types."""

    def __init__(self, config_path: Path | None = None):
        self._path = config_path or (_CONFIGS_DIR / "ontology.yaml")
        self._supertypes: dict[str, str] = {}
        self._domains: dict[str, dict] = {}
        # type_name -> supertype mapping (flattened across all domains)
        self._type_to_supertype: dict[str, str] = {}
        # namespace -> list of type names
        self._namespace_types: dict[str, list[str]] = {}
        # namespace -> domain config (for prompt hints)
        self._namespace_domain: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            log.warning("ontology.not_found", path=str(self._path))
            return
        data = yaml.safe_load(self._path.read_text())
        self._supertypes = data.get("supertypes", {})
        self._domains = data.get("domains", {})

        for domain_name, domain_cfg in self._domains.items():
            namespaces = domain_cfg.get("namespaces", [])
            types = domain_cfg.get("types", {})
            type_names = list(types.keys())

            for type_name, type_def in types.items():
                supertype = type_def.get("supertype", "")
                self._type_to_supertype[type_name] = supertype

            for ns in namespaces:
                self._namespace_types[ns] = type_names
                self._namespace_domain[ns] = domain_cfg

    def get_types_for_namespace(self, namespace: str) -> list[str]:
        """Entity type names appropriate for this namespace."""
        return self._namespace_types.get(namespace, list(self._type_to_supertype.keys()))

    def get_supertype(self, entity_type: str) -> str:
        """Map any entity type to its supertype."""
        return self._type_to_supertype.get(entity_type, "")

    def get_type_descriptions(self, namespace: str) -> dict[str, str]:
        """Get {type_name: description} for a namespace's types."""
        domain_cfg = self._namespace_domain.get(namespace)
        if not domain_cfg:
            # Fallback: return all types
            result = {}
            for domain in self._domains.values():
                for type_name, type_def in domain.get("types", {}).items():
                    if type_name not in result:
                        result[type_name] = type_def.get("description", "")
            return result
        return {
            name: tdef.get("description", "")
            for name, tdef in domain_cfg.get("types", {}).items()
        }

    def get_prompt_hint(self, namespace: str) -> str:
        """Domain-specific extraction guidance for the LLM prompt."""
        domain_cfg = self._namespace_domain.get(namespace, {})
        return domain_cfg.get("prompt_hint", "")

    def get_all_supertypes(self) -> dict[str, str]:
        """Full supertype catalog: {name: description}."""
        return dict(self._supertypes)

    def get_all_types(self) -> dict[str, str]:
        """Full type catalog: {type_name: supertype}."""
        return dict(self._type_to_supertype)


# Module-level singleton
_default_ontology: OntologyRegistry | None = None


def get_ontology() -> OntologyRegistry:
    """Get or create the default ontology registry (singleton)."""
    global _default_ontology
    if _default_ontology is None:
        _default_ontology = OntologyRegistry()
    return _default_ontology
