"""Unit tests for the modular entity extraction pipeline.

Covers: normalization, protocol compliance, JSON repair, regex extraction,
merge logic, ontology, and pipeline smoke test (regex-only).
"""

import pytest

from legion_koi.extraction import normalize_entity_name
from legion_koi.extraction.models import Entity, ExtractionResult


# --- 1. Normalization ---

class TestNormalizeEntityName:
    def test_lowercase(self):
        assert normalize_entity_name("FalkorDB") == "falkordb"

    def test_strip_and_collapse_spaces(self):
        assert normalize_entity_name("  Hello   World  ") == "hello world"

    def test_nfkc_normalization(self):
        # NFKC normalizes compatibility characters but preserves base unicode
        assert normalize_entity_name("café") == "café"

    def test_empty_string(self):
        assert normalize_entity_name("") == ""

    def test_already_normalized(self):
        assert normalize_entity_name("simple") == "simple"

    def test_tabs_and_newlines(self):
        assert normalize_entity_name("hello\t\nworld") == "hello world"


# --- 2. Protocol compliance ---

class TestProtocolCompliance:
    def test_llm_extractor_implements_protocol(self):
        from legion_koi.extraction.backends.llm import LLMEntityExtractor
        from legion_koi.extraction.protocols import EntityExtractor

        extractor = LLMEntityExtractor.__new__(LLMEntityExtractor)
        assert isinstance(extractor, EntityExtractor)

    def test_regex_extractor_implements_protocol(self):
        from legion_koi.extraction.backends.regex import RegexEntityExtractor
        from legion_koi.extraction.protocols import EntityExtractor

        extractor = RegexEntityExtractor()
        assert isinstance(extractor, EntityExtractor)

    def test_protocol_has_namespace_param(self):
        """Verify the Protocol signature includes namespace."""
        from legion_koi.extraction.protocols import EntityExtractor
        import inspect

        sig = inspect.signature(EntityExtractor.extract_entities)
        assert "namespace" in sig.parameters
        assert sig.parameters["namespace"].default == ""


# --- 3. JSON repair ---

class TestRepairJson:
    def test_strips_code_fences(self):
        from legion_koi.extraction.backends.llm import _repair_json

        raw = '```json\n{"entities": []}\n```'
        assert _repair_json(raw) == '{"entities": []}'

    def test_fixes_trailing_commas(self):
        from legion_koi.extraction.backends.llm import _repair_json

        raw = '{"entities": [{"name": "x", "type": "y"},]}'
        result = _repair_json(raw)
        assert ",]" not in result
        assert result == '{"entities": [{"name": "x", "type": "y"}]}'

    def test_strips_fences_and_fixes_commas(self):
        from legion_koi.extraction.backends.llm import _repair_json

        raw = '```\n{"items": [1, 2, 3,]}\n```'
        result = _repair_json(raw)
        assert result == '{"items": [1, 2, 3]}'

    def test_passthrough_clean_json(self):
        from legion_koi.extraction.backends.llm import _repair_json

        raw = '{"entities": [{"name": "test", "type": "Tool"}]}'
        assert _repair_json(raw) == raw


# --- 4. Regex extraction ---

class TestRegexExtraction:
    @pytest.fixture
    def extractor(self):
        from legion_koi.extraction.backends.regex import RegexEntityExtractor
        return RegexEntityExtractor()

    def test_extracts_dates(self, extractor):
        entities = extractor.extract_entities("Meeting on 2026-03-11 about planning", [])
        dates = [e for e in entities if e.entity_type == "Date"]
        assert len(dates) == 1
        assert dates[0].name == "2026-03-11"

    def test_extracts_urls(self, extractor):
        entities = extractor.extract_entities("See https://example.com/docs for info", [])
        urls = [e for e in entities if e.entity_type == "URL"]
        assert len(urls) == 1
        assert "example.com" in urls[0].name

    def test_extracts_paths(self, extractor):
        entities = extractor.extract_entities("Edit /home/shawn/.claude/config.json today", [])
        paths = [e for e in entities if e.entity_type == "Path"]
        assert len(paths) >= 1
        assert any("/home/shawn" in p.name for p in paths)

    def test_extracts_versions(self, extractor):
        entities = extractor.extract_entities("Upgraded to FalkorDB v3.2.1 today", [])
        versions = [e for e in entities if e.entity_type == "Version"]
        assert len(versions) == 1
        assert versions[0].name == "3.2.1"

    def test_extracts_orns(self, extractor):
        entities = extractor.extract_entities("Bundle orn:legion.journal:2026-03-11 processed", [])
        rids = [e for e in entities if e.entity_type == "RID"]
        assert len(rids) == 1
        assert rids[0].name == "orn:legion.journal:2026-03-11"

    def test_deduplicates(self, extractor):
        entities = extractor.extract_entities(
            "Date 2026-03-11 and again 2026-03-11 repeated", []
        )
        dates = [e for e in entities if e.entity_type == "Date"]
        assert len(dates) == 1

    def test_empty_text(self, extractor):
        assert extractor.extract_entities("", []) == []

    def test_namespace_param_accepted(self, extractor):
        """Regex extractor accepts namespace param (protocol compliance)."""
        entities = extractor.extract_entities("2026-03-11", [], namespace="legion.test")
        assert len(entities) >= 1


# --- 5. Merge logic ---

class TestMergeEntities:
    def test_deduplicates_by_normalized_name(self):
        from legion_koi.extraction.pipeline import _merge_entities

        entities = [
            Entity(name="FalkorDB", entity_type="Tool", confidence=0.9),
            Entity(name="falkordb", entity_type="Tool", confidence=0.8),
        ]
        merged = _merge_entities(entities)
        assert len(merged) == 1

    def test_keeps_highest_confidence(self):
        from legion_koi.extraction.pipeline import _merge_entities

        entities = [
            Entity(name="FalkorDB", entity_type="Tool", confidence=0.7),
            Entity(name="falkordb", entity_type="Tool", confidence=0.95),
        ]
        merged = _merge_entities(entities)
        assert len(merged) == 1
        assert merged[0].confidence == 0.95

    def test_different_types_not_merged(self):
        from legion_koi.extraction.pipeline import _merge_entities

        entities = [
            Entity(name="Python", entity_type="Tool", confidence=0.9),
            Entity(name="Python", entity_type="Language", confidence=0.9),
        ]
        merged = _merge_entities(entities)
        assert len(merged) == 2

    def test_empty_list(self):
        from legion_koi.extraction.pipeline import _merge_entities
        assert _merge_entities([]) == []


# --- 6. Ontology ---

class TestOntology:
    def test_ontology_loads(self):
        from legion_koi.extraction.ontology import OntologyRegistry
        ontology = OntologyRegistry()
        all_types = ontology.get_all_types()
        assert len(all_types) > 0

    def test_get_types_for_namespace(self):
        from legion_koi.extraction.ontology import OntologyRegistry
        ontology = OntologyRegistry()
        types = ontology.get_types_for_namespace("legion.claude-journal")
        assert isinstance(types, list)
        assert len(types) > 0

    def test_get_supertype(self):
        from legion_koi.extraction.ontology import OntologyRegistry
        ontology = OntologyRegistry()
        all_types = ontology.get_all_types()
        if all_types:
            first_type = next(iter(all_types))
            supertype = ontology.get_supertype(first_type)
            assert isinstance(supertype, str)

    def test_unknown_namespace_returns_defaults(self):
        from legion_koi.extraction.ontology import OntologyRegistry
        ontology = OntologyRegistry()
        all_types = ontology.get_all_types()
        fallback_types = ontology.get_types_for_namespace("nonexistent.namespace")
        # Unknown namespace should return all types as fallback
        assert len(fallback_types) == len(all_types)


# --- 7. Pipeline smoke test (regex-only, no LLM) ---

class TestPipelineFast:
    def test_fast_config_extracts(self):
        from legion_koi.extraction.pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(config_name="fast")
        result = pipeline.run(
            "test-rid",
            "legion.claude-journal",
            "Meeting on 2026-03-11 about FalkorDB v3.2.1 at /home/shawn/.claude/config",
        )
        assert isinstance(result, ExtractionResult)
        assert any(e.entity_type == "Date" for e in result.entities)
        assert any(e.entity_type == "Version" for e in result.entities)

    def test_empty_text_returns_empty_result(self):
        from legion_koi.extraction.pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(config_name="fast")
        result = pipeline.run("test-rid", "test.ns", "")
        assert result.entities == []

    def test_result_has_metadata(self):
        from legion_koi.extraction.pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(config_name="fast")
        result = pipeline.run("test-rid", "test.ns", "Some text on 2026-01-01")
        assert result.rid == "test-rid"
        assert result.namespace == "test.ns"
        assert result.pipeline_config == "fast"
        assert result.extraction_time_seconds >= 0


# --- 8. Export boundary ---

class TestExports:
    def test_normalize_entity_name_exported(self):
        from legion_koi.extraction import normalize_entity_name
        assert callable(normalize_entity_name)

    def test_relation_not_in_public_exports(self):
        from legion_koi.extraction import __all__
        assert "Relation" not in __all__

    def test_relation_still_importable_internally(self):
        from legion_koi.extraction.models import Relation
        assert Relation is not None
