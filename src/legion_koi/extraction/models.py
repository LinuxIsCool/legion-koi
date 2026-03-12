"""Pydantic models for entity extraction results."""

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A named entity extracted from text."""

    name: str
    entity_type: str  # e.g. "Person", "Tool" — from ontology.yaml
    supertype: str = ""  # e.g. "agent", "artifact" — set by ontology lookup
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source_chunk: int | None = None
    metadata: dict = Field(default_factory=dict)


class Relation(BaseModel):
    """A typed relationship between two entities.

    Not yet populated by any backend. Schema exists in PostgreSQL
    for Phase 8 relation extraction. Do not export until implemented.
    """

    source: str  # entity name
    target: str  # entity name
    relation_type: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence: str = ""  # supporting text span


class ExtractionResult(BaseModel):
    """Output of an extraction pipeline run on a single bundle."""

    rid: str
    namespace: str
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    pipeline_config: str = ""
    extraction_time_seconds: float = 0.0
