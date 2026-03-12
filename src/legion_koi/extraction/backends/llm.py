"""LLM-based entity extraction via Instructor + TELUS Ollama.

Uses Instructor for Pydantic-validated structured output from any
OpenAI-compatible API. Falls back to raw JSON parsing if structured
output fails (Ollama can be flaky with tool/function calling).
"""

import json
import os
import re
import time
from pathlib import Path

import structlog
from pydantic import BaseModel, Field

from ..models import Entity
from ..registry import Registry
from ..ontology import OntologyRegistry, get_ontology
from ...constants import (
    ENTITY_EXTRACT_MAX_CHARS,
    ENTITY_MAX_PER_CHUNK,
    ENTITY_DEFAULT_CONFIDENCE_FLOOR,
)

log = structlog.stdlib.get_logger()

# Registry instance — importable by pipeline
ner_registry: Registry = Registry("ner")


def _load_telus_env() -> dict[str, str]:
    """Load TELUS API secrets from dotenv file (same pattern as embeddings.py)."""
    env_path = Path("~/.claude/local/secrets/telus-api.env").expanduser()
    if not env_path.exists():
        return {}
    result = {}
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def _repair_json(raw: str) -> str:
    """Best-effort JSON repair — strip code fences, fix trailing commas."""
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    # Fix trailing commas before ] or }
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return raw


class _EntityItem(BaseModel):
    """Schema for a single entity in the LLM response."""
    name: str
    type: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class _ExtractionResponse(BaseModel):
    """Schema for the full LLM extraction response."""
    entities: list[_EntityItem] = Field(default_factory=list)


def _build_prompt(text: str, type_descriptions: dict[str, str], prompt_hint: str) -> str:
    """Build the extraction prompt with type definitions and domain hints."""
    type_lines = "\n".join(
        f"  - {name}: {desc}" for name, desc in type_descriptions.items()
    )
    hint_section = f"\nDomain guidance: {prompt_hint}\n" if prompt_hint else ""

    return f"""Extract named entities from the following text. Return a JSON object with an "entities" array.

Entity types to extract:
{type_lines}
{hint_section}
Rules:
- Extract specific, named entities only — not generic concepts
- Use the exact entity type names listed above
- Confidence: 1.0 = explicitly named, 0.7 = strongly implied, 0.5 = inferred
- Prefer recall over precision — include borderline entities with lower confidence
- Deduplicate: use the most complete form of a name (e.g. "FalkorDB" not "Falkor")
- Maximum {ENTITY_MAX_PER_CHUNK} entities

Text:
---
{text[:ENTITY_EXTRACT_MAX_CHARS]}
---

Return ONLY valid JSON: {{"entities": [{{"name": "...", "type": "...", "confidence": 0.9}}, ...]}}"""


@ner_registry.register("llm")
class LLMEntityExtractor:
    """Instructor-based entity extraction using TELUS Ollama or any OpenAI-compatible API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        ontology: OntologyRegistry | None = None,
    ):
        env = _load_telus_env()
        self._base_url = (
            base_url
            or os.environ.get("TELUS_OLLAMA_URL")
            or env.get("TELUS_OLLAMA_URL", "")
        )
        self._api_key = (
            api_key
            or os.environ.get("TELUS_OLLAMA_KEY")
            or env.get("TELUS_OLLAMA_KEY", "")
        )
        self._model = model or os.environ.get("TELUS_LLM_MODEL") or env.get("TELUS_LLM_MODEL", "gpt-oss:120b")
        self._ontology = ontology or get_ontology()
        self._client = None

    def _get_client(self):
        """Lazy-init the Instructor client."""
        if self._client is None:
            try:
                import instructor
                from openai import OpenAI

                # Ollama endpoints need /v1 for OpenAI compatibility
                base_url = self._base_url.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url}/v1"

                openai_client = OpenAI(
                    base_url=base_url,
                    api_key=self._api_key or "ollama",
                )
                self._client = instructor.from_openai(
                    openai_client,
                    mode=instructor.Mode.JSON,
                )
            except ImportError:
                log.warning("llm_extractor.instructor_not_installed")
                raise
        return self._client

    def extract_entities(self, text: str, entity_types: list[str], namespace: str = "") -> list[Entity]:
        """Extract entities using LLM with Instructor structured output."""
        if not text or not text.strip():
            return []

        type_descriptions = self._ontology.get_type_descriptions(namespace)
        # Filter to requested types
        filtered_descs = {k: v for k, v in type_descriptions.items() if k in entity_types}
        if not filtered_descs:
            log.warning("llm_extractor.namespace_unmapped", namespace=namespace,
                        requested_types=entity_types)
            filtered_descs = type_descriptions

        prompt_hint = self._ontology.get_prompt_hint(namespace)
        prompt = _build_prompt(text, filtered_descs, prompt_hint)

        start = time.monotonic()
        entities = self._call_llm(prompt)
        elapsed = time.monotonic() - start
        log.debug("llm_extractor.extracted", count=len(entities), elapsed_s=f"{elapsed:.2f}")

        # Set supertypes and filter by confidence
        result = []
        for e in entities:
            if e.confidence < ENTITY_DEFAULT_CONFIDENCE_FLOOR:
                continue
            e.supertype = self._ontology.get_supertype(e.entity_type)
            result.append(e)

        return result[:ENTITY_MAX_PER_CHUNK]

    def _call_llm(self, prompt: str) -> list[Entity]:
        """Call LLM with Instructor, falling back to raw JSON on failure."""
        # Try Instructor structured output first
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self._model,
                response_model=_ExtractionResponse,
                messages=[{"role": "user", "content": prompt}],
                max_retries=2,
            )
            return [
                Entity(
                    name=item.name,
                    entity_type=item.type,
                    confidence=item.confidence,
                )
                for item in response.entities
            ]
        except Exception as e:
            log.info("llm_extractor.instructor_fallback", error=str(e))

        # Fallback: raw OpenAI call + JSON parsing
        try:
            from openai import OpenAI

            base_url = self._base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
            raw_client = OpenAI(base_url=base_url, api_key=self._api_key or "ollama")
            response = raw_client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            return self._parse_raw_json(raw)
        except Exception as e:
            log.warning("llm_extractor.failed", error=str(e))
            return []

    def _parse_raw_json(self, raw: str) -> list[Entity]:
        """Parse raw JSON response with repair logic."""
        raw = _repair_json(raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try raw_decode for partial JSON
            decoder = json.JSONDecoder()
            try:
                data, _ = decoder.raw_decode(raw)
            except json.JSONDecodeError:
                log.warning("llm_extractor.json_parse_failed", raw_preview=raw[:200])
                return []

        # Handle both {"entities": [...]} and bare [...]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("entities", [])
        else:
            return []

        entities = []
        for item in items:
            if isinstance(item, dict) and "name" in item and "type" in item:
                entities.append(Entity(
                    name=item["name"],
                    entity_type=item["type"],
                    confidence=float(item.get("confidence", 0.8)),
                ))
        return entities

    def get_name(self) -> str:
        return "llm"

    def is_available(self) -> bool:
        return bool(self._base_url and self._api_key)
