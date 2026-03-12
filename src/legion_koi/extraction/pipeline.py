"""Extraction pipeline orchestrator — chains backends per config.

Loads pipeline config from YAML, instantiates backends from registries,
chunks text, extracts entities per chunk, merges, and optionally resolves.
"""

import time
import unicodedata
from pathlib import Path

import structlog
import yaml

from .models import Entity, ExtractionResult
from .ontology import OntologyRegistry, get_ontology
from .protocols import EntityExtractor
from .backends.llm import ner_registry
from ..chunking import chunk_text
from ..constants import ENTITY_EXTRACT_MAX_CHARS

log = structlog.stdlib.get_logger()

_CONFIGS_DIR = Path(__file__).parent / "configs"


def normalize_entity_name(name: str) -> str:
    """Normalize entity name for deduplication.

    Lowercase, strip whitespace, normalize unicode, collapse internal spaces.
    Single canonical implementation — import from extraction package.
    """
    name = unicodedata.normalize("NFKC", name)
    name = " ".join(name.lower().split())
    return name


def _merge_entities(entities: list[Entity]) -> list[Entity]:
    """Merge entities across chunks — dedup by normalized name + type, keep highest confidence."""
    merged: dict[tuple[str, str], Entity] = {}
    for e in entities:
        key = (normalize_entity_name(e.name), e.entity_type)
        if key not in merged or e.confidence > merged[key].confidence:
            merged[key] = e
    return list(merged.values())


def load_pipeline_config(config_name: str = "default") -> dict:
    """Load a pipeline config YAML by name."""
    path = _CONFIGS_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    return yaml.safe_load(path.read_text())


class ExtractionPipeline:
    """Orchestrates entity extraction: chunk → extract → merge → resolve."""

    def __init__(
        self,
        config_name: str = "default",
        ontology: OntologyRegistry | None = None,
    ):
        self._config = load_pipeline_config(config_name)
        self._config_name = config_name
        self._ontology = ontology or get_ontology()
        self._extractors = []
        self._resolver = None
        self._init_stages()

    def _init_stages(self) -> None:
        """Instantiate backends from config stages."""
        stages = self._config.get("stages", {})

        # NER backends (can have multiple: llm + regex)
        for stage_name in ("ner", "regex"):
            stage = stages.get(stage_name)
            if not stage:
                continue
            backend_name = stage.get("backend", stage_name)
            params = dict(stage.get("params", {}))
            # Inject ontology for LLM backend
            if backend_name == "llm":
                params["ontology"] = self._ontology
            try:
                extractor = ner_registry.create(backend_name, **params)
                if not isinstance(extractor, EntityExtractor):
                    log.error("pipeline.protocol_violation", backend=backend_name)
                    continue
                if extractor.is_available():
                    self._extractors.append(extractor)
                else:
                    log.warning("pipeline.backend_unavailable", backend=backend_name)
            except Exception as e:
                log.warning("pipeline.backend_init_failed", backend=backend_name, error=str(e))

        # Entity resolution
        resolve_stage = stages.get("resolve")
        if resolve_stage:
            # Fuzzy resolution is handled inline via _merge_entities + name normalization
            # Future: pluggable resolver backends via a separate registry
            pass

    def run(self, rid: str, namespace: str, text: str) -> ExtractionResult:
        """Run the full extraction pipeline on a single document."""
        start = time.monotonic()

        if not text or not text.strip():
            return ExtractionResult(rid=rid, namespace=namespace, pipeline_config=self._config_name)

        entity_types = self._ontology.get_types_for_namespace(namespace)

        # Chunk if text is long
        if len(text) > ENTITY_EXTRACT_MAX_CHARS:
            chunks = chunk_text(text, chunk_chars=ENTITY_EXTRACT_MAX_CHARS, overlap_chars=200)
        else:
            chunks = [text]

        # Extract from each chunk with each backend
        all_entities: list[Entity] = []
        for chunk_idx, chunk in enumerate(chunks):
            for extractor in self._extractors:
                try:
                    extracted = extractor.extract_entities(chunk, entity_types, namespace=namespace)
                    for e in extracted:
                        e.source_chunk = chunk_idx
                        # Ensure supertype is set
                        if not e.supertype:
                            e.supertype = self._ontology.get_supertype(e.entity_type)
                    all_entities.extend(extracted)
                except Exception as e:
                    log.warning(
                        "pipeline.extraction_error",
                        backend=extractor.get_name(),
                        rid=rid,
                        chunk=chunk_idx,
                        error=str(e),
                    )

        # Merge across chunks
        merged = _merge_entities(all_entities)

        elapsed = time.monotonic() - start
        return ExtractionResult(
            rid=rid,
            namespace=namespace,
            entities=merged,
            pipeline_config=self._config_name,
            extraction_time_seconds=round(elapsed, 3),
        )


# Module-level convenience
_default_pipeline: ExtractionPipeline | None = None


def get_pipeline(config_name: str = "default") -> ExtractionPipeline:
    """Get or create the default pipeline (singleton per config)."""
    global _default_pipeline
    if _default_pipeline is None or _default_pipeline._config_name != config_name:
        _default_pipeline = ExtractionPipeline(config_name=config_name)
    return _default_pipeline


def run_extraction(rid: str, namespace: str, text: str, config_name: str = "default") -> ExtractionResult:
    """Run extraction on a single document. Convenience wrapper."""
    pipeline = get_pipeline(config_name)
    return pipeline.run(rid, namespace, text)
