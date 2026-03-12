"""Modular entity extraction pipeline for legion-koi.

Public API:
    from legion_koi.extraction import run_extraction, load_pipeline_config
    from legion_koi.extraction import normalize_entity_name
    from legion_koi.extraction.models import Entity, ExtractionResult
    from legion_koi.extraction.ontology import get_ontology
"""

from .pipeline import run_extraction, load_pipeline_config, get_pipeline, normalize_entity_name
from .models import Entity, ExtractionResult
from .ontology import get_ontology, OntologyRegistry

__all__ = [
    "run_extraction",
    "load_pipeline_config",
    "get_pipeline",
    "normalize_entity_name",
    "Entity",
    "ExtractionResult",
    "get_ontology",
    "OntologyRegistry",
]
