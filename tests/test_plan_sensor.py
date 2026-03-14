"""Tests for plan sensor, RID type, and parsing helpers."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from legion_koi.rid_types.plan import LegionPlan
from legion_koi.sensors.plan_parsing import (
    classify_plan,
    extract_h1,
    extract_bold_field,
)


class TestLegionPlanRID:
    def test_roundtrip(self):
        rid = LegionPlan(slug="fuzzy-pondering-yeti")
        assert rid.reference == "fuzzy-pondering-yeti"
        restored = LegionPlan.from_reference("fuzzy-pondering-yeti")
        assert restored.slug == "fuzzy-pondering-yeti"

    def test_namespace(self):
        rid = LegionPlan(slug="test")
        assert rid.namespace == "legion.claude-plan"

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            LegionPlan.from_reference("")


class TestClassifyPlan:
    def test_dated(self):
        assert classify_plan("2026-03-09-claude-journal-plan") == ("dated", "")

    def test_auto(self):
        assert classify_plan("fuzzy-pondering-yeti") == ("auto", "")

    def test_subagent(self):
        plan_type, parent = classify_plan(
            "snazzy-tumbling-reddy-agent-a401d870a72941948"
        )
        assert plan_type == "subagent"
        assert parent == "snazzy-tumbling-reddy"

    def test_subagent_preserves_full_parent(self):
        plan_type, parent = classify_plan(
            "precious-exploring-catmull-agent-ac012fbd03f0b1d80"
        )
        assert plan_type == "subagent"
        assert parent == "precious-exploring-catmull"

    def test_dated_takes_precedence_in_absence_of_subagent_suffix(self):
        # Dated pattern never collides with subagent — subagent needs -agent-a[hex]{8+}
        plan_type, parent = classify_plan("2026-03-09-some-plan")
        assert plan_type == "dated"
        assert parent == ""

    def test_short_hex_not_treated_as_subagent(self):
        # Only 3 hex chars — below the 8-char minimum, should be "auto" not "subagent"
        plan_type, _ = classify_plan("build-agent-abc")
        assert plan_type == "auto"


class TestExtractH1:
    def test_simple(self):
        assert extract_h1("# My Plan Title\n\nBody text") == "My Plan Title"

    def test_missing(self):
        assert extract_h1("No heading here\nJust body") == ""

    def test_skips_h2(self):
        assert extract_h1("## Not This\n# This One") == "This One"

    def test_strips_whitespace(self):
        assert extract_h1("#   Spaced Title  \n") == "Spaced Title"


class TestExtractBoldField:
    def test_goal(self):
        text = "Some preamble\n**Goal:** Build a knowledge sensor\n\nMore text"
        assert extract_bold_field(text, "Goal") == "Build a knowledge sensor"

    def test_missing(self):
        assert extract_bold_field("No bold fields here", "Goal") == ""

    def test_case_insensitive(self):
        text = "**goal:** lowercase works too"
        assert extract_bold_field(text, "Goal") == "lowercase works too"


from legion_koi.sensors.plan_sensor import PlanSensor


class TestPlanSensor:
    def test_should_process_md(self):
        sensor = PlanSensor(
            watch_dir=Path("/tmp"),
            state_path=Path("/tmp/state.json"),
            kobj_push=MagicMock(),
        )
        assert sensor.should_process(Path("test.md")) is True
        assert sensor.should_process(Path("test.txt")) is False
        assert sensor.should_process(Path("test.py")) is False

    def test_process_file(self, tmp_path):
        plan_file = tmp_path / "fuzzy-pondering-yeti.md"
        plan_file.write_text("# My Test Plan\n\n**Goal:** Test the sensor\n\nBody here.")

        sensor = PlanSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(plan_file)

        assert bundle is not None
        assert bundle.rid.reference == "fuzzy-pondering-yeti"
        assert bundle.contents["title"] == "My Test Plan"
        assert bundle.contents["goal"] == "Test the sensor"
        assert bundle.contents["plan_type"] == "auto"
        assert bundle.contents["parent_slug"] == ""

    def test_subagent_plan(self, tmp_path):
        plan_file = tmp_path / "snazzy-tumbling-reddy-agent-a401d870a72941948.md"
        plan_file.write_text("# Subagent Work\n\nDetails")

        sensor = PlanSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(plan_file)

        assert bundle.contents["plan_type"] == "subagent"
        assert bundle.contents["parent_slug"] == "snazzy-tumbling-reddy"

    def test_dedup_cross_restart(self, tmp_path):
        """State persists across sensor instances — same file is not re-ingested."""
        plan_file = tmp_path / "test-plan.md"
        plan_file.write_text("# Plan\n\nContent")
        state_path = tmp_path / "state.json"

        # First sensor instance — processes the file
        sensor1 = PlanSensor(
            watch_dir=tmp_path,
            state_path=state_path,
            kobj_push=MagicMock(),
        )
        first_scan = sensor1.scan_all()
        assert len(first_scan) == 1

        # Second sensor instance (simulates restart) — file unchanged
        sensor2 = PlanSensor(
            watch_dir=tmp_path,
            state_path=state_path,
            kobj_push=MagicMock(),
        )
        second_scan = sensor2.scan_all()
        assert len(second_scan) == 0


from legion_koi.storage.postgres import _extract_search_text
from legion_koi.contextual import extract_preamble


class TestPlanSearchIntegration:
    def test_search_text(self):
        contents = {"title": "My Plan", "goal": "Build a thing", "body": "Full body text"}
        result = _extract_search_text("legion.claude-plan", contents)
        assert "My Plan" in result
        assert "Build a thing" in result
        assert "Full body text" in result

    def test_preamble(self):
        contents = {"title": "My Plan", "plan_type": "auto"}
        result = extract_preamble("legion.claude-plan", contents)
        assert result == "Plan: My Plan. Type: auto."

    def test_preamble_no_title(self):
        contents = {"title": "", "plan_type": "auto"}
        result = extract_preamble("legion.claude-plan", contents)
        assert result == ""
