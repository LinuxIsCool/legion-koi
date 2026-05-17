"""Three doctrine metadata fields added per mutation-paradigm-doctrine.md §6
horizon roadmap: visibility / archive_target / governance_chain. Safe
defaults so existing bundles read forward without migration.

These fields live on top-level kobj.contents (NOT nested in frontmatter)
because they are doctrine-level metadata spanning all bundle types."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from koi_net.protocol.event import EventType
from koi_net.protocol.knowledge_object import KnowledgeObject

from legion_koi.handlers import (
    JournalBundleHandler,
    TaskBundleHandler,
    _VISIBILITY_ENUM,
    _ARCHIVE_TARGET_ENUM,
    _GOVERNANCE_CHAIN_ENUM,
    _normalize_doctrine_metadata,
)
from legion_koi.rid_types.journal import LegionJournal
from legion_koi.rid_types.task import LegionTask


def _make_handler(handler_cls):
    """Build a handler with mocked log + pipeline."""
    return handler_cls(log=MagicMock(), pipeline=MagicMock())


def _make_journal_kobj(contents: dict) -> KnowledgeObject:
    rid = LegionJournal(date="2026-05-15", slug="metadata-test")
    kobj = KnowledgeObject.from_rid(rid)
    kobj.event_type = EventType.NEW
    kobj.contents = contents
    return kobj


def _make_task_kobj(contents: dict) -> KnowledgeObject:
    rid = LegionTask(task_id="task-metadata-test")
    kobj = KnowledgeObject.from_rid(rid)
    kobj.event_type = EventType.NEW
    kobj.contents = contents
    return kobj


def test_doctrine_enum_constants():
    """Enum constants exposed at module level for cross-handler reuse + tests."""
    assert _VISIBILITY_ENUM == frozenset({"private", "federated", "public"})
    assert _ARCHIVE_TARGET_ENUM == frozenset({"none", "arweave", "filecoin", "storj"})
    assert _GOVERNANCE_CHAIN_ENUM == frozenset({"none", "avalanche", "regen", "substrate"})


def test_normalize_sets_defaults_on_empty_contents():
    """Legacy bundle with no doctrine metadata gets all three default values."""
    contents = {"frontmatter": {"title": "x"}}
    _normalize_doctrine_metadata(contents)
    assert contents["visibility"] == "private"
    assert contents["archive_target"] == "none"
    assert contents["governance_chain"] == "none"
    # Existing frontmatter untouched
    assert contents["frontmatter"]["title"] == "x"


def test_normalize_preserves_explicit_values():
    """Explicitly-set doctrine fields pass through unchanged."""
    contents = {
        "frontmatter": {"title": "x"},
        "visibility": "federated",
        "archive_target": "arweave",
        "governance_chain": "avalanche",
    }
    _normalize_doctrine_metadata(contents)
    assert contents["visibility"] == "federated"
    assert contents["archive_target"] == "arweave"
    assert contents["governance_chain"] == "avalanche"


def test_normalize_rejects_invalid_visibility():
    contents = {"visibility": "secret"}
    with pytest.raises(ValueError) as exc:
        _normalize_doctrine_metadata(contents)
    assert "visibility" in str(exc.value)
    assert "secret" in str(exc.value)


def test_normalize_rejects_invalid_archive_target():
    contents = {"archive_target": "dropbox"}
    with pytest.raises(ValueError) as exc:
        _normalize_doctrine_metadata(contents)
    assert "archive_target" in str(exc.value)
    assert "dropbox" in str(exc.value)


def test_normalize_rejects_invalid_governance_chain():
    contents = {"governance_chain": "ethereum-mainnet"}
    with pytest.raises(ValueError) as exc:
        _normalize_doctrine_metadata(contents)
    assert "governance_chain" in str(exc.value)


def test_journal_handler_normalizes_doctrine_metadata():
    """JournalBundleHandler.handle() applies the doctrine metadata normalization."""
    handler = _make_handler(JournalBundleHandler)
    kobj = _make_journal_kobj({
        "frontmatter": {"title": "test", "created": "2026-05-15"}
    })
    result = handler.handle(kobj)
    assert result is not None
    assert result.contents["visibility"] == "private"
    assert result.contents["archive_target"] == "none"
    assert result.contents["governance_chain"] == "none"


def test_task_handler_normalizes_doctrine_metadata_alongside_version():
    """TaskBundleHandler retains its version Lamport clock (Task 9) AND adds
    the three doctrine metadata fields (Task 10)."""
    handler = _make_handler(TaskBundleHandler)
    kobj = _make_task_kobj({
        "frontmatter": {"title": "test", "status": "open"}
    })
    result = handler.handle(kobj)
    assert result is not None
    # Task 9 fields still work
    assert result.contents["frontmatter"]["version"] == 1
    # Task 10 fields added
    assert result.contents["visibility"] == "private"
    assert result.contents["archive_target"] == "none"
    assert result.contents["governance_chain"] == "none"


def test_invalid_doctrine_metadata_raises_on_journal_handler():
    """A bad enum value on any bundle is rejected at handler ingest time."""
    handler = _make_handler(JournalBundleHandler)
    kobj = _make_journal_kobj({
        "frontmatter": {"title": "x"},
        "visibility": "internal-only",  # not a valid enum
    })
    with pytest.raises(ValueError):
        handler.handle(kobj)
