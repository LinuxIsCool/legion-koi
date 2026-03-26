"""Changelog sensor — polls docked repos for new release versions.

Reads CHANGELOG.md from docked GitHub repos, parses version entries,
and creates KOI bundles for each new release detected. Optionally
auto-updates repos via git pull before parsing.

Design decisions:
  - CHANGELOG.md over GitHub API: already on disk after git pull, zero
    rate limits, zero tokens, zero network dependency beyond git fetch.
  - Poll interval 6 hours: Claude Code releases ~weekly. 6h detection
    latency is fine. ericbuess/claude-code-docs updates every 3 hours.
  - Auto-pull inside sensor: sensor owns its data source end-to-end.
    No separate rhythm or cron needed.
  - YouTubeSensor pattern: threading.Timer self-rescheduling loop.
"""

import json
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from ..rid_types.changelog import LegionChangelog
from . import state as sensor_state

log = structlog.stdlib.get_logger()

# Regex to split CHANGELOG.md on version headers like "## 2.1.83"
VERSION_HEADER_RE = re.compile(r"^## (\S+)", re.MULTILINE)

# Known hook event type names (from official docs, 2026-03-26)
KNOWN_HOOK_TYPES = frozenset({
    "SessionStart", "SessionEnd",
    "UserPromptSubmit", "Notification", "Elicitation", "ElicitationResult",
    "PreToolUse", "PermissionRequest", "PostToolUse", "PostToolUseFailure",
    "SubagentStart", "SubagentStop", "TeammateIdle", "TaskCompleted",
    "Stop", "StopFailure",
    "InstructionsLoaded", "ConfigChange", "CwdChanged", "FileChanged",
    "WorktreeCreate", "WorktreeRemove",
    "PreCompact", "PostCompact",
})

# Pattern for backtick-quoted PascalCase identifiers in release notes
HOOK_MENTION_RE = re.compile(r"`([A-Z][a-zA-Z]+)`")

GIT_PULL_TIMEOUT_SECONDS = 30


@dataclass
class ChangelogRepo:
    """A docked repo to monitor for changelog updates."""
    owner: str                          # e.g. "anthropics"
    repo: str                           # e.g. "claude-code"
    changelog_file: str = "CHANGELOG.md"
    hooks_file: str | None = None       # e.g. "docs/hooks-guide.md"


@dataclass
class ParsedVersion:
    """A single version entry parsed from a CHANGELOG.md."""
    version: str
    body: str
    features: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    bullet_count: int = 0
    hook_types_mentioned: list[str] = field(default_factory=list)


def parse_changelog(text: str) -> list[ParsedVersion]:
    """Parse CHANGELOG.md into a list of version entries.

    Splits on '## <version>' headers. Returns newest-first order
    (same as file order).
    """
    parts = VERSION_HEADER_RE.split(text)
    # parts alternates: [preamble, version1, body1, version2, body2, ...]
    versions = []
    for i in range(1, len(parts) - 1, 2):
        version = parts[i].strip()
        body = parts[i + 1].strip()

        features = []
        fixes = []
        bullet_count = 0

        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                bullet_count += 1
                if stripped.startswith("- Added "):
                    features.append(stripped[2:])  # strip "- "
                elif stripped.startswith("- Fixed "):
                    fixes.append(stripped[2:])

        # Extract hook type mentions
        hook_mentions = []
        for match in HOOK_MENTION_RE.finditer(body):
            name = match.group(1)
            if name in KNOWN_HOOK_TYPES:
                hook_mentions.append(name)
        # Deduplicate while preserving order
        seen = set()
        hook_mentions = [h for h in hook_mentions if not (h in seen or seen.add(h))]

        versions.append(ParsedVersion(
            version=version,
            body=body,
            features=features,
            fixes=fixes,
            bullet_count=bullet_count,
            hook_types_mentioned=hook_mentions,
        ))

    return versions


def extract_hook_types_from_docs(hooks_file: Path) -> set[str]:
    """Parse a hooks documentation file for event type names.

    Looks for backtick-quoted PascalCase words that match the known
    hook types pattern (but may include new ones not yet in our set).
    """
    if not hooks_file.exists():
        return set()

    text = hooks_file.read_text(encoding="utf-8")
    found = set()
    for match in HOOK_MENTION_RE.finditer(text):
        name = match.group(1)
        # Heuristic: hook types are PascalCase, at least 3 chars
        if len(name) >= 3 and name[0].isupper():
            found.add(name)
    return found


class ChangelogSensor:
    """Polls docked repos for new changelog versions, creates KOI bundles."""

    def __init__(
        self,
        repos: list[ChangelogRepo],
        dock_repos_base: Path,
        state_path: Path,
        kobj_push: callable,
        poll_interval: float = 21600.0,  # 6 hours
        auto_update: bool = True,
    ):
        self.repos = repos
        self.dock_repos_base = Path(dock_repos_base)
        self.state_path = Path(state_path)
        self.kobj_push = kobj_push
        self.poll_interval = poll_interval
        self.auto_update = auto_update
        self.state = sensor_state.load(self.state_path)
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Git operations
    # ------------------------------------------------------------------

    @staticmethod
    def _update_repo(repo_path: Path) -> bool:
        """Pull latest changes from remote. Returns True if successful."""
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "pull", "--ff-only", "--update-shallow"],
                capture_output=True,
                text=True,
                timeout=GIT_PULL_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                log.debug("changelog.git_pull_ok", repo=str(repo_path))
                return True
            log.warning(
                "changelog.git_pull_failed",
                repo=str(repo_path),
                stderr=result.stderr[:300],
            )
        except subprocess.TimeoutExpired:
            log.warning("changelog.git_pull_timeout", repo=str(repo_path))
        except Exception:
            log.exception("changelog.git_pull_error", repo=str(repo_path))
        return False

    # ------------------------------------------------------------------
    # Bundle creation
    # ------------------------------------------------------------------

    def _make_bundle(
        self, repo: ChangelogRepo, version: ParsedVersion, is_latest: bool
    ) -> Bundle:
        """Create a KOI bundle from a parsed version entry."""
        repo_ref = f"{repo.owner}/{repo.repo}"
        rid = LegionChangelog(repo=repo_ref, version=version.version)

        contents = {
            "version": version.version,
            "repo": repo_ref,
            "body": version.body,
            "bullet_count": version.bullet_count,
            "features": version.features,
            "fixes": version.fixes,
            "hook_types_mentioned": version.hook_types_mentioned,
            "is_latest": is_latest,
        }

        return Bundle.generate(rid=rid, contents=contents)

    # ------------------------------------------------------------------
    # Hook type diffing
    # ------------------------------------------------------------------

    def _check_hook_types(self, repo: ChangelogRepo, repo_path: Path) -> None:
        """Check for new hook types in the docs file, if configured."""
        if not repo.hooks_file:
            return

        hooks_path = repo_path / repo.hooks_file
        found_types = extract_hook_types_from_docs(hooks_path)
        if not found_types:
            return

        state_key = f"hook_types/{repo.owner}/{repo.repo}"
        known_sorted = sorted(found_types)
        current_hash = sensor_state.compute_hash(json.dumps(known_sorted))

        prev_hash = self.state.get(state_key)
        if prev_hash == current_hash:
            return  # No change

        # Diff against KNOWN_HOOK_TYPES constant
        new_types = found_types - KNOWN_HOOK_TYPES
        if new_types:
            log.warning(
                "changelog.new_hook_types_detected",
                new_types=sorted(new_types),
                total=len(found_types),
                source=str(hooks_path),
            )
        else:
            log.info(
                "changelog.hook_types_updated",
                total=len(found_types),
                source=str(hooks_path),
            )

        self.state[state_key] = current_hash

    # ------------------------------------------------------------------
    # Poll / scan
    # ------------------------------------------------------------------

    def _poll_repo(self, repo: ChangelogRepo) -> list[Bundle]:
        """Poll a single repo for new changelog versions."""
        repo_path = self.dock_repos_base / repo.owner / repo.repo

        if not repo_path.exists():
            log.warning("changelog.repo_not_found", path=str(repo_path))
            return []

        # Auto-update
        if self.auto_update:
            self._update_repo(repo_path)

        # Read changelog
        changelog_path = repo_path / repo.changelog_file
        if not changelog_path.exists():
            log.warning("changelog.file_not_found", path=str(changelog_path))
            return []

        text = changelog_path.read_text(encoding="utf-8")
        versions = parse_changelog(text)
        if not versions:
            return []

        # Check hook types (side effect: logs warnings if new types found)
        self._check_hook_types(repo, repo_path)

        # Create bundles for new/changed versions
        bundles = []
        repo_ref = f"{repo.owner}/{repo.repo}"

        for i, version in enumerate(versions):
            ref_key = f"{repo_ref}/{version.version}"
            content_hash = sensor_state.compute_hash(version.body)
            change = sensor_state.has_changed(ref_key, content_hash, self.state)

            if change is None:
                continue  # Already seen, unchanged

            is_latest = (i == 0)
            bundle = self._make_bundle(repo, version, is_latest)
            bundles.append(bundle)

            log.info(
                "changelog.version_detected",
                change=change,
                repo=repo_ref,
                version=version.version,
                bullets=version.bullet_count,
                features=len(version.features),
                fixes=len(version.fixes),
                hook_types=version.hook_types_mentioned or None,
            )

            self.state[ref_key] = content_hash

        if bundles:
            sensor_state.save(self.state_path, self.state)

        return bundles

    def poll(self) -> list[Bundle]:
        """Poll all configured repos."""
        all_bundles = []
        for repo in self.repos:
            bundles = self._poll_repo(repo)
            all_bundles.extend(bundles)
        return all_bundles

    def scan_all(self) -> list[Bundle]:
        """Initial scan — same as poll."""
        return self.poll()

    # ------------------------------------------------------------------
    # Lifecycle (threading.Timer pattern from YouTubeSensor)
    # ------------------------------------------------------------------

    def _poll_loop(self):
        """Timer callback."""
        if not self._running:
            return
        with self._lock:
            try:
                bundles = self.poll()
                for bundle in bundles:
                    self.kobj_push(bundle=bundle)
                if bundles:
                    log.info("changelog.poll_complete", count=len(bundles))
            except Exception:
                log.exception("changelog.poll_error")
        if self._running:
            self._timer = threading.Timer(self.poll_interval, self._poll_loop)
            self._timer.daemon = True
            self._timer.start()

    def start(self):
        """Start the polling loop."""
        if not self.repos:
            log.warning("changelog.no_repos", msg="No repos configured")
            return
        self._running = True
        self._timer = threading.Timer(self.poll_interval, self._poll_loop)
        self._timer.daemon = True
        self._timer.start()
        repo_names = [f"{r.owner}/{r.repo}" for r in self.repos]
        log.info(
            "changelog.started",
            repos=repo_names,
            poll_interval=self.poll_interval,
            auto_update=self.auto_update,
        )

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            log.info("changelog.stopped")
