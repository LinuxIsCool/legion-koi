"""Verify legion.claude-task bundles carry a version: int Lamport clock
on frontmatter with safe default of 1.

Per mutation-paradigm-doctrine.md §6 + Task 9 byproduct: closes the
concurrent multi-agent write gap. The Lamport clock lives on the
frontmatter dict (where the rest of task fields live), defaulting to 1
when missing so legacy bundles read forward without migration."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koi_net.protocol.event import EventType
from koi_net.protocol.knowledge_object import KnowledgeObject
from legion_koi.handlers import TaskBundleHandler
from legion_koi.rid_types.task import LegionTask


def _make_handler() -> TaskBundleHandler:
    """Build a TaskBundleHandler with mocked log + pipeline.

    The pipeline mock's register_handler is called via __post_init__ but
    has no side effects for unit-test purposes."""
    return TaskBundleHandler(log=MagicMock(), pipeline=MagicMock())


def _make_task_kobj(frontmatter: dict) -> KnowledgeObject:
    """Build a minimal KnowledgeObject for handler validation."""
    rid = LegionTask(task_id="task-test-1")
    kobj = KnowledgeObject.from_rid(rid)
    kobj.event_type = EventType.NEW
    kobj.contents = {"frontmatter": frontmatter}
    return kobj


def test_task_bundle_normalizes_explicit_version():
    """Explicit version field passes through unchanged."""
    handler = _make_handler()
    kobj = _make_task_kobj({"title": "test", "status": "open", "version": 3})
    result = handler.handle(kobj)
    assert result is not None
    assert result.contents["frontmatter"]["version"] == 3


def test_task_bundle_defaults_version_to_1_when_missing():
    """Missing version on legacy bundle reads forward as version=1 — no migration."""
    handler = _make_handler()
    kobj = _make_task_kobj({"title": "test", "status": "open"})
    result = handler.handle(kobj)
    assert result is not None
    assert result.contents["frontmatter"]["version"] == 1


def test_task_bundle_rejects_non_integer_version():
    """Non-int version values raise ValueError."""
    handler = _make_handler()
    kobj = _make_task_kobj({"title": "test", "status": "open", "version": "three"})
    with pytest.raises(ValueError) as exc_info:
        handler.handle(kobj)
    assert "version" in str(exc_info.value)


def test_task_bundle_rejects_boolean_version():
    """Boolean is technically an int subclass but semantically wrong for a
    Lamport clock — reject explicitly."""
    handler = _make_handler()
    kobj = _make_task_kobj({"title": "test", "status": "open", "version": True})
    with pytest.raises(ValueError) as exc_info:
        handler.handle(kobj)
    assert "version" in str(exc_info.value)


def test_task_bundle_handle_returns_kobj():
    """Handler returns the kobj so the pipeline can continue."""
    handler = _make_handler()
    kobj = _make_task_kobj({"title": "test", "status": "open"})
    result = handler.handle(kobj)
    assert result is kobj


def test_task_bundle_warns_on_missing_title_with_version_still_set():
    """Title-missing warning still fires; version field still gets set on
    frontmatter. The two normalizations are independent."""
    handler = _make_handler()
    kobj = _make_task_kobj({"status": "open"})  # no title
    result = handler.handle(kobj)
    # No raise — missing title is a warning, not an error
    assert result is not None
    assert result.contents["frontmatter"]["version"] == 1
