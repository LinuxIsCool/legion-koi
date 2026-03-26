"""Tests for changelog sensor, RID type, and parsing logic."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from legion_koi.rid_types.changelog import LegionChangelog
from legion_koi.sensors.changelog_sensor import (
    ChangelogRepo,
    ChangelogSensor,
    KNOWN_HOOK_TYPES,
    extract_hook_types_from_docs,
    parse_changelog,
)


# ---------------------------------------------------------------------------
# Sample CHANGELOG.md content for testing
# ---------------------------------------------------------------------------

SAMPLE_CHANGELOG = """\
# Changelog

## 2.1.83

- Added `managed-settings.d/` drop-in directory alongside `managed-settings.json`
- Added `CwdChanged` and `FileChanged` hook events for reactive environment management
- Added `sandbox.failIfUnavailable` setting to exit with an error when sandbox is enabled
- Fixed mouse tracking escape sequences leaking to shell prompt after exit
- Fixed Claude Code hanging on exit on macOS
- Fixed screen flashing blank after being idle for a few seconds
- Improved Bedrock SDK cold-start latency by overlapping profile fetch

## 2.1.81

- Added `--bare` flag for scripted `-p` calls
- Added `--channels` permission relay
- Fixed multiple concurrent Claude Code sessions requiring repeated re-authentication
- Fixed voice mode silently swallowing retry failures

## 2.1.80

- Fixed a regression in MCP server startup
- Improved startup performance
"""


# ---------------------------------------------------------------------------
# RID Type Tests
# ---------------------------------------------------------------------------


class TestLegionChangelogRID:
    def test_roundtrip(self):
        rid = LegionChangelog(repo="anthropics/claude-code", version="2.1.83")
        assert rid.reference == "anthropics/claude-code/2.1.83"
        restored = LegionChangelog.from_reference("anthropics/claude-code/2.1.83")
        assert restored.repo == "anthropics/claude-code"
        assert restored.version == "2.1.83"

    def test_namespace(self):
        rid = LegionChangelog(repo="anthropics/claude-code", version="1.0.0")
        assert rid.namespace == "legion.claude-changelog"

    def test_str_format(self):
        rid = LegionChangelog(repo="anthropics/claude-code", version="2.1.83")
        assert str(rid) == "orn:legion.claude-changelog:anthropics/claude-code/2.1.83"

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            LegionChangelog.from_reference("")

    def test_no_slash_raises(self):
        with pytest.raises(ValueError):
            LegionChangelog.from_reference("noversion")

    def test_single_slash_works(self):
        """owner/version (no repo) — edge case, still valid reference."""
        rid = LegionChangelog.from_reference("owner/1.0.0")
        assert rid.repo == "owner"
        assert rid.version == "1.0.0"


# ---------------------------------------------------------------------------
# Changelog Parsing Tests
# ---------------------------------------------------------------------------


class TestParseChangelog:
    def test_basic_parse(self):
        versions = parse_changelog(SAMPLE_CHANGELOG)
        assert len(versions) == 3

    def test_version_order(self):
        versions = parse_changelog(SAMPLE_CHANGELOG)
        assert versions[0].version == "2.1.83"
        assert versions[1].version == "2.1.81"
        assert versions[2].version == "2.1.80"

    def test_bullet_count(self):
        versions = parse_changelog(SAMPLE_CHANGELOG)
        assert versions[0].bullet_count == 7  # 3 Added + 3 Fixed + 1 Improved

    def test_features_extracted(self):
        versions = parse_changelog(SAMPLE_CHANGELOG)
        v83 = versions[0]
        assert len(v83.features) == 3
        assert any("managed-settings.d" in f for f in v83.features)
        assert any("CwdChanged" in f for f in v83.features)

    def test_fixes_extracted(self):
        versions = parse_changelog(SAMPLE_CHANGELOG)
        v83 = versions[0]
        assert len(v83.fixes) == 3
        assert any("mouse tracking" in f for f in v83.fixes)

    def test_hook_types_mentioned(self):
        versions = parse_changelog(SAMPLE_CHANGELOG)
        v83 = versions[0]
        assert "CwdChanged" in v83.hook_types_mentioned
        assert "FileChanged" in v83.hook_types_mentioned

    def test_no_false_positive_hook_types(self):
        """Random backtick words that aren't hook types shouldn't match."""
        versions = parse_changelog(SAMPLE_CHANGELOG)
        v83 = versions[0]
        # "Bedrock" is backtick-free in our sample, but ensure no spurious matches
        for hook in v83.hook_types_mentioned:
            assert hook in KNOWN_HOOK_TYPES

    def test_empty_changelog(self):
        versions = parse_changelog("")
        assert versions == []

    def test_no_versions(self):
        versions = parse_changelog("# Changelog\n\nSome intro text.")
        assert versions == []

    def test_body_preserved(self):
        versions = parse_changelog(SAMPLE_CHANGELOG)
        assert "managed-settings.d" in versions[0].body


# ---------------------------------------------------------------------------
# Hook Type Extraction Tests
# ---------------------------------------------------------------------------


class TestExtractHookTypes:
    def test_from_sample_docs(self, tmp_path):
        docs = tmp_path / "hooks.md"
        docs.write_text("""\
| Event | Description |
|-------|-------------|
| `SessionStart` | Session begins |
| `PreToolUse` | Before tool call |
| `BrandNewHookType` | Something new |
""")
        found = extract_hook_types_from_docs(docs)
        assert "SessionStart" in found
        assert "PreToolUse" in found
        assert "BrandNewHookType" in found

    def test_missing_file(self, tmp_path):
        found = extract_hook_types_from_docs(tmp_path / "nonexistent.md")
        assert found == set()

    def test_known_hook_types_count(self):
        """Sanity check: we have 25 known hook types as of v2.1.84."""
        assert len(KNOWN_HOOK_TYPES) == 25


# ---------------------------------------------------------------------------
# Sensor Deduplication Tests
# ---------------------------------------------------------------------------


class TestChangelogSensorDedup:
    def test_second_scan_returns_empty(self, tmp_path):
        """Poll twice with same changelog — second should return 0 bundles."""
        # Set up a fake docked repo
        repo_dir = tmp_path / "anthropics" / "claude-code"
        repo_dir.mkdir(parents=True)
        (repo_dir / "CHANGELOG.md").write_text(SAMPLE_CHANGELOG)

        state_path = tmp_path / "state.json"
        bundles_collected = []

        sensor = ChangelogSensor(
            repos=[ChangelogRepo(owner="anthropics", repo="claude-code")],
            dock_repos_base=tmp_path,
            state_path=state_path,
            kobj_push=lambda bundle: bundles_collected.append(bundle),
            auto_update=False,  # Don't try git pull on fake dir
        )

        # First scan
        first = sensor.scan_all()
        assert len(first) == 3  # 3 versions

        # Second scan — all versions already seen
        second = sensor.poll()
        assert len(second) == 0

    def test_new_version_detected(self, tmp_path):
        """Add a new version after first scan — should detect it."""
        repo_dir = tmp_path / "anthropics" / "claude-code"
        repo_dir.mkdir(parents=True)
        (repo_dir / "CHANGELOG.md").write_text(SAMPLE_CHANGELOG)

        state_path = tmp_path / "state.json"

        sensor = ChangelogSensor(
            repos=[ChangelogRepo(owner="anthropics", repo="claude-code")],
            dock_repos_base=tmp_path,
            state_path=state_path,
            kobj_push=MagicMock(),
            auto_update=False,
        )

        sensor.scan_all()

        # Simulate a new release
        new_changelog = "## 2.1.84\n\n- Added something new\n\n" + SAMPLE_CHANGELOG.split("# Changelog\n\n", 1)[1]
        new_changelog = "# Changelog\n\n" + new_changelog
        (repo_dir / "CHANGELOG.md").write_text(new_changelog)

        new_bundles = sensor.poll()
        assert len(new_bundles) == 1
        # Verify it's the new version
        assert "2.1.84" in new_bundles[0].contents["version"]


# ---------------------------------------------------------------------------
# Integration test (requires real docked repo)
# ---------------------------------------------------------------------------


class TestChangelogSensorIntegration:
    REAL_CHANGELOG = Path.home() / ".claude/local/dock/repos/anthropics/claude-code/CHANGELOG.md"

    @pytest.mark.skipif(
        not REAL_CHANGELOG.exists(),
        reason="anthropics/claude-code not docked locally",
    )
    def test_scan_real_repo(self, tmp_path):
        """Scan the real docked anthropics/claude-code repo."""
        state_path = tmp_path / "state.json"

        sensor = ChangelogSensor(
            repos=[ChangelogRepo(owner="anthropics", repo="claude-code")],
            dock_repos_base=Path.home() / ".claude/local/dock/repos",
            state_path=state_path,
            kobj_push=MagicMock(),
            auto_update=False,
        )

        bundles = sensor.scan_all()
        # Should find many versions (CHANGELOG.md has 2521 lines)
        assert len(bundles) > 20

        # Latest should be v2.1.83 or newer
        versions = [b.contents["version"] for b in bundles]
        assert "2.1.83" in versions
