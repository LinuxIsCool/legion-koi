"""Tests for research sensor, RID type, and search/preamble integration."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from legion_koi.rid_types.research import LegionResearch
from legion_koi.sensors.research_sensor import ResearchSensor
from legion_koi.storage.postgres import _extract_search_text
from legion_koi.contextual import extract_preamble


class TestLegionResearchRID:
    def test_roundtrip(self):
        rid = LegionResearch(slug="a2a-identity")
        assert rid.reference == "a2a-identity"
        restored = LegionResearch.from_reference("a2a-identity")
        assert restored.slug == "a2a-identity"

    def test_namespace(self):
        rid = LegionResearch(slug="test")
        assert rid.namespace == "legion.claude-research"

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            LegionResearch.from_reference("")

    def test_subdirectory_slug(self):
        rid = LegionResearch(slug="hippo/retrieval-patterns")
        assert rid.reference == "hippo/retrieval-patterns"
        restored = LegionResearch.from_reference("hippo/retrieval-patterns")
        assert restored.slug == "hippo/retrieval-patterns"

    def test_str_format(self):
        rid = LegionResearch(slug="a2a-identity")
        assert str(rid) == "orn:legion.claude-research:a2a-identity"


class TestResearchSensor:
    def test_should_process_md(self):
        sensor = ResearchSensor(
            watch_dir=Path("/tmp"),
            state_path=Path("/tmp/state.json"),
            kobj_push=MagicMock(),
        )
        assert sensor.should_process(Path("test.md")) is True
        assert sensor.should_process(Path("test.txt")) is False
        assert sensor.should_process(Path("test.py")) is False

    def test_process_file(self, tmp_path):
        research_file = tmp_path / "a2a-identity.md"
        research_file.write_text(
            "---\ntitle: A2A Identity Research\ntags:\n  - a2a\n  - identity\n"
            "status: complete\ncreated: 2026-03-10\n---\n\nBody of the study."
        )

        sensor = ResearchSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(research_file)

        assert bundle is not None
        assert bundle.rid.reference == "a2a-identity"
        assert bundle.contents["frontmatter"]["title"] == "A2A Identity Research"
        assert bundle.contents["frontmatter"]["tags"] == ["a2a", "identity"]
        assert bundle.contents["body"] == "Body of the study."

    def test_subdirectory_file(self, tmp_path):
        subdir = tmp_path / "hippo"
        subdir.mkdir()
        research_file = subdir / "retrieval-patterns.md"
        research_file.write_text(
            "---\ntitle: Retrieval Patterns\n---\n\nStudy content."
        )

        sensor = ResearchSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(research_file)

        assert bundle is not None
        assert bundle.rid.reference == "hippo/retrieval-patterns"

    def test_dedup_cross_restart(self, tmp_path):
        """State persists across sensor instances — same file is not re-ingested."""
        research_file = tmp_path / "test-study.md"
        research_file.write_text("---\ntitle: Test\n---\n\nContent")
        state_path = tmp_path / "state.json"

        # First sensor instance — processes the file
        sensor1 = ResearchSensor(
            watch_dir=tmp_path,
            state_path=state_path,
            kobj_push=MagicMock(),
        )
        first_scan = sensor1.scan_all()
        assert len(first_scan) == 1

        # Second sensor instance (simulates restart) — file unchanged
        sensor2 = ResearchSensor(
            watch_dir=tmp_path,
            state_path=state_path,
            kobj_push=MagicMock(),
        )
        second_scan = sensor2.scan_all()
        assert len(second_scan) == 0

    def test_no_frontmatter(self, tmp_path):
        """Files without frontmatter still get processed with empty frontmatter."""
        research_file = tmp_path / "async-wake-mechanism.md"
        research_file.write_text("# Async Wake Mechanism\n\nJust body text.")

        sensor = ResearchSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(research_file)

        assert bundle is not None
        assert bundle.rid.reference == "async-wake-mechanism"
        assert bundle.contents["frontmatter"] == {}
        assert "Just body text." in bundle.contents["body"]


class TestResearchSearchText:
    def test_search_text(self):
        contents = {
            "frontmatter": {
                "title": "T-SNE Acceleration",
                "tags": ["ml", "visualization"],
                "prompted_by": "journal entry on embeddings",
            },
            "body": "Full body of the research study.",
        }
        result = _extract_search_text("legion.claude-research", contents)
        assert "T-SNE Acceleration" in result
        assert "ml" in result
        assert "visualization" in result
        assert "journal entry on embeddings" in result
        assert "Full body of the research study." in result

    def test_search_text_missing_fields(self):
        contents = {"frontmatter": {}, "body": "Just a body"}
        result = _extract_search_text("legion.claude-research", contents)
        assert "Just a body" in result


class TestResearchPreamble:
    def test_preamble_full(self):
        contents = {
            "frontmatter": {
                "title": "T-SNE Research",
                "created": "2026-03-12",
                "status": "complete",
            }
        }
        result = extract_preamble("legion.claude-research", contents)
        assert result == "Research: T-SNE Research. Date: 2026-03-12. Status: complete."

    def test_preamble_title_only(self):
        contents = {"frontmatter": {"title": "Quick Study"}}
        result = extract_preamble("legion.claude-research", contents)
        assert result == "Research: Quick Study."

    def test_preamble_no_title(self):
        contents = {"frontmatter": {}}
        result = extract_preamble("legion.claude-research", contents)
        assert result == ""
