"""Tests for backlog sensor, RID type, and search/preamble integration."""

import pytest
from unittest.mock import MagicMock
from pathlib import Path

from legion_koi.rid_types.task import LegionTask
from legion_koi.sensors.backlog_sensor import BacklogSensor
from legion_koi.storage.postgres import _extract_search_text
from legion_koi.contextual import extract_preamble


class TestLegionTaskRID:
    def test_roundtrip(self):
        rid = LegionTask(task_id="task-009")
        assert rid.reference == "task-009"
        restored = LegionTask.from_reference("task-009")
        assert restored.task_id == "task-009"

    def test_namespace(self):
        rid = LegionTask(task_id="test")
        assert rid.namespace == "legion.claude-task"

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            LegionTask.from_reference("")

    def test_str_format(self):
        rid = LegionTask(task_id="task-009")
        assert str(rid) == "orn:legion.claude-task:task-009"


class TestBacklogSensor:
    def test_should_process(self):
        sensor = BacklogSensor(
            watch_dir=Path("/tmp"),
            state_path=Path("/tmp/state.json"),
            kobj_push=MagicMock(),
        )
        assert sensor.should_process(Path("task-009 - some-task.md")) is True
        assert sensor.should_process(Path("task-001.md")) is True
        assert sensor.should_process(Path("notes.md")) is False
        assert sensor.should_process(Path("task-009.txt")) is False
        assert sensor.should_process(Path("task-009.py")) is False

    def test_process_file_with_frontmatter(self, tmp_path):
        task_file = tmp_path / "task-009 - rhythms-agent-overhaul.md"
        task_file.write_text(
            "---\n"
            "id: task-009\n"
            "title: Rhythms Agent Overhaul\n"
            "status: active\n"
            "priority: high\n"
            "milestone: M3 — Knowledge Graph\n"
            "---\n\n"
            "Overhaul the rhythms agent to use new orchestration."
        )

        sensor = BacklogSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(task_file)

        assert bundle is not None
        assert bundle.rid.reference == "task-009"
        assert bundle.contents["frontmatter"]["title"] == "Rhythms Agent Overhaul"
        assert bundle.contents["frontmatter"]["status"] == "active"
        assert bundle.contents["frontmatter"]["milestone"] == "M3 — Knowledge Graph"
        assert "Overhaul the rhythms agent" in bundle.contents["body"]

    def test_process_file_id_from_filename(self, tmp_path):
        """When frontmatter has no 'id' field, task_id comes from filename."""
        task_file = tmp_path / "task-042 - mysterious-task.md"
        task_file.write_text("---\ntitle: Mystery\n---\n\nBody text.")

        sensor = BacklogSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(task_file)

        assert bundle is not None
        assert bundle.rid.reference == "task-042"

    def test_dedup_cross_restart(self, tmp_path):
        """State persists across sensor instances — same file is not re-ingested."""
        task_file = tmp_path / "task-001 - test.md"
        task_file.write_text("---\nid: task-001\ntitle: Test\n---\n\nContent")
        state_path = tmp_path / "state.json"

        # First sensor instance — processes the file
        sensor1 = BacklogSensor(
            watch_dir=tmp_path,
            state_path=state_path,
            kobj_push=MagicMock(),
        )
        first_scan = sensor1.scan_all()
        assert len(first_scan) == 1

        # Second sensor instance (simulates restart) — file unchanged
        sensor2 = BacklogSensor(
            watch_dir=tmp_path,
            state_path=state_path,
            kobj_push=MagicMock(),
        )
        second_scan = sensor2.scan_all()
        assert len(second_scan) == 0

    def test_no_frontmatter(self, tmp_path):
        """Files without frontmatter still process with empty frontmatter."""
        task_file = tmp_path / "task-099 - bare.md"
        task_file.write_text("# Just a heading\n\nNo frontmatter here.")

        sensor = BacklogSensor(
            watch_dir=tmp_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundle = sensor.process_file(task_file)

        assert bundle is not None
        assert bundle.rid.reference == "task-099"
        assert bundle.contents["frontmatter"] == {}


class TestTaskSearchText:
    def test_search_text(self):
        contents = {
            "frontmatter": {
                "title": "Rhythms Agent Overhaul",
                "status": "active",
                "milestone": "M3 — Knowledge Graph",
            },
            "body": "Full body of the task description.",
        }
        result = _extract_search_text("legion.claude-task", contents)
        assert "Rhythms Agent Overhaul" in result
        assert "active" in result
        assert "M3" in result
        assert "Full body of the task description." in result

    def test_search_text_missing_fields(self):
        contents = {"frontmatter": {}, "body": "Just a body"}
        result = _extract_search_text("legion.claude-task", contents)
        assert "Just a body" in result


class TestTaskPreamble:
    def test_preamble_full(self):
        contents = {
            "frontmatter": {
                "title": "Rhythms Agent Overhaul",
                "status": "active",
            }
        }
        result = extract_preamble("legion.claude-task", contents)
        assert result == "Task: Rhythms Agent Overhaul. Status: active."

    def test_preamble_title_only(self):
        contents = {"frontmatter": {"title": "Quick Task"}}
        result = extract_preamble("legion.claude-task", contents)
        assert result == "Task: Quick Task."

    def test_preamble_no_title(self):
        contents = {"frontmatter": {}}
        result = extract_preamble("legion.claude-task", contents)
        assert result == ""
