"""Browser history sensor — polls Firefox places.sqlite for history and bookmarks."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import structlog
from rid_lib.ext import Bundle

from ..rid_types.browser_history import LegionBrowserHistory
from . import state as sensor_state
from .db_sensor import DatabaseSensor
from .firefox_profiles import FirefoxProfile
from .privacy_config import PrivacyConfig
from .url_sanitizer import sanitize_url_ext, url_hash, is_suppressed

log = structlog.stdlib.get_logger()

# Firefox visit type constants (moz_historyvisits.visit_type)
VISIT_LINK = 1
VISIT_TYPED = 2
VISIT_BOOKMARK = 3
VISIT_EMBED = 4
VISIT_REDIRECT_PERMANENT = 5
VISIT_REDIRECT_TEMPORARY = 6
VISIT_DOWNLOAD = 7
VISIT_FRAMED_LINK = 8

INCLUDED_VISIT_TYPES = (VISIT_LINK, VISIT_TYPED, VISIT_BOOKMARK, VISIT_DOWNLOAD)

VISIT_TYPE_NAMES = {
    VISIT_LINK: "link",
    VISIT_TYPED: "typed",
    VISIT_BOOKMARK: "bookmark",
    VISIT_DOWNLOAD: "download",
}

# Firefox root bookmark GUIDs (stop walking parent chain at these)
ROOT_BOOKMARK_GUIDS = frozenset({
    "root________",
    "menu________",
    "toolbar_____",
    "unfiled_____",
    "mobile______",
    "tags________",
})

# Firefox timestamps are microseconds since Unix epoch
FIREFOX_EPOCH_DIVISOR = 1_000_000

# Maximum recent visits to store per URL
MAX_RECENT_VISITS = 100


def _firefox_ts_to_iso(ts_microseconds: int | None) -> str | None:
    """Convert Firefox microsecond timestamp to ISO 8601 string."""
    if not ts_microseconds:
        return None
    try:
        dt = datetime.fromtimestamp(ts_microseconds / FIREFOX_EPOCH_DIVISOR, tz=timezone.utc)
        return dt.isoformat()
    except (OSError, ValueError, OverflowError):
        return None


class BrowserHistorySensor(DatabaseSensor):
    """Polls Firefox places.sqlite databases for history and bookmark bundles."""

    def __init__(
        self,
        profiles: list[FirefoxProfile],
        state_path: Path,
        kobj_push: callable,
        poll_interval: float = 300.0,
        batch_size: int = 500,
        suppression_path: Path | None = None,
        param_policy_path: Path | None = None,
    ):
        # Pass first profile's places_path for base class db_path
        super().__init__(
            db_path=profiles[0].places_path if profiles else Path("/dev/null"),
            state_path=state_path,
            kobj_push=kobj_push,
            poll_interval=poll_interval,
            batch_size=batch_size,
        )
        self.profiles = profiles
        # Track temp dirs per-profile for safe-connect cleanup (thread-safe)
        self._tmp_dirs: dict[str, Path] = {}
        # Privacy config — hot-reloads on mtime change each poll cycle
        self._privacy = PrivacyConfig(
            suppression_path=suppression_path or Path("~/.config/claude-browser-history/suppressed_domains.txt").expanduser(),
            param_policy_path=param_policy_path or Path("~/.config/claude-browser-history/param_policy.yaml").expanduser(),
        )

    def _safe_connect(self, places_path: Path) -> tuple[sqlite3.Connection, Path | None]:
        """Connect to places.sqlite, handling Firefox locks.

        Returns (connection, tmp_dir_or_None). Caller must clean up tmp_dir.

        Strategy:
        1. Try read-only URI mode first
        2. On SQLITE_BUSY/locked, copy DB + WAL + SHM to temp dir, open immutable
        """
        # Try read-only first
        try:
            uri = f"file:{places_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            # Test the connection
            conn.execute("SELECT 1 FROM moz_places LIMIT 1")
            return conn, None
        except sqlite3.OperationalError:
            pass  # Fall through to copy strategy

        # Copy to temp dir and open as immutable
        tmp_dir = Path(tempfile.mkdtemp(prefix="koi-firefox-"))
        try:
            db_name = places_path.name
            shutil.copy2(places_path, tmp_dir / db_name)

            # Copy WAL and SHM if they exist (needed for consistent reads)
            for suffix in ("-wal", "-shm"):
                wal_path = places_path.parent / f"{db_name}{suffix}"
                if wal_path.exists():
                    shutil.copy2(wal_path, tmp_dir / f"{db_name}{suffix}")

            tmp_db = tmp_dir / db_name
            uri = f"file:{tmp_db}?immutable=1"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            return conn, tmp_dir
        except Exception:
            # Clean up temp dir on failure
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    @staticmethod
    def _close_safe(conn: sqlite3.Connection, tmp_dir: Path | None) -> None:
        """Close connection and clean up temp dir if present."""
        try:
            conn.close()
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def start(self):
        """Start polling — check that at least one profile exists."""
        if not self.profiles:
            log.warning("browser_history.no_profiles")
            return
        # Override base db_path check — we manage our own connections per profile
        self._running = True
        self._timer = threading.Timer(self.poll_interval, self._poll_loop)
        self._timer.daemon = True
        self._timer.start()
        log.info(
            "sensor.started",
            sensor=self.__class__.__name__,
            profiles=[p.slug for p in self.profiles],
            poll_interval=self.poll_interval,
        )

    def poll(self) -> list[Bundle]:
        """Poll all profiles for new history and bookmarks."""
        bundles = []
        for profile in self.profiles:
            if not profile.places_path.exists():
                continue
            try:
                bundles.extend(self._poll_profile(profile))
            except Exception:
                log.exception("browser_history.profile_error", slug=profile.slug)
        if bundles:
            sensor_state.save(self.state_path, self.state)
        return bundles

    def _poll_profile(self, profile: FirefoxProfile) -> list[Bundle]:
        """Poll a single Firefox profile for history and bookmarks."""
        conn, tmp_dir = self._safe_connect(profile.places_path)
        try:
            bundles = self._poll_history(conn, profile)
            bundles.extend(self._poll_bookmarks(conn, profile))
            return bundles
        finally:
            self._close_safe(conn, tmp_dir)

    def _poll_history(self, conn: sqlite3.Connection, profile: FirefoxProfile) -> list[Bundle]:
        """Poll for new history visits since last high-water mark."""
        state_key = f"{profile.slug}/last_visit_id"
        last_visit_id = int(self.state.get(state_key, 0))
        bundles = []

        placeholders = ",".join("?" for _ in INCLUDED_VISIT_TYPES)
        cursor = conn.execute(
            f"""
            SELECT v.id AS visit_id, v.visit_date, v.visit_type, v.from_visit,
                   p.id AS place_id, p.url, p.title, p.visit_count, p.frecency,
                   p.description
            FROM moz_historyvisits v
            JOIN moz_places p ON v.place_id = p.id
            WHERE v.id > ?
              AND v.visit_type IN ({placeholders})
            ORDER BY v.id
            LIMIT ?
            """,
            (last_visit_id, *INCLUDED_VISIT_TYPES, self.batch_size),
        )

        # Group visits by place_id to aggregate per-URL
        place_visits: dict[int, dict] = {}
        max_visit_id = last_visit_id

        for row in cursor:
            visit_id = row["visit_id"]
            place_id = row["place_id"]
            max_visit_id = max(max_visit_id, visit_id)

            if place_id not in place_visits:
                place_visits[place_id] = {
                    "url": row["url"],
                    "title": row["title"],
                    "visit_count": row["visit_count"],
                    "frecency": row["frecency"],
                    "description": row["description"],
                    "place_id": place_id,
                    "visits": [],
                }
            place_visits[place_id]["visits"].append({
                "visit_id": visit_id,
                "visit_date": row["visit_date"],
                "visit_type": row["visit_type"],
                "from_visit": row["from_visit"],
            })

        # Build bundles per URL
        for place_id, data in place_visits.items():
            url = data["url"] or ""
            domain = self._extract_domain(url)

            # Suppression check (uses configurable list)
            if is_suppressed(url, domain, self._privacy.suppression):
                continue

            sanitize_result = sanitize_url_ext(url, domain, self._privacy.params)
            url_clean = sanitize_result.url
            entry_hash = url_hash(url)

            # Aggregate visit types
            visit_type_counts: dict[str, int] = {}
            recent_visits = []
            first_visit_ts = None
            last_visit_ts = None

            for v in data["visits"]:
                vtype = v["visit_type"]
                vname = VISIT_TYPE_NAMES.get(vtype, f"type_{vtype}")
                visit_type_counts[vname] = visit_type_counts.get(vname, 0) + 1

                visit_iso = _firefox_ts_to_iso(v["visit_date"])
                if visit_iso:
                    recent_visits.append({"time": visit_iso, "type": vname})
                    if first_visit_ts is None or v["visit_date"] < first_visit_ts:
                        first_visit_ts = v["visit_date"]
                    if last_visit_ts is None or v["visit_date"] > last_visit_ts:
                        last_visit_ts = v["visit_date"]

            # Cap recent visits
            recent_visits = sorted(recent_visits, key=lambda x: x["time"], reverse=True)[:MAX_RECENT_VISITS]

            # Engagement data (optional — moz_places_metadata may not exist)
            engagement = self._get_engagement(conn, place_id)

            contents = {
                "type": "history",
                "url": url,
                "url_clean": url_clean,
                "domain": domain,
                "title": data["title"] or "",
                "visit_count": data["visit_count"] or 0,
                "first_visit": _firefox_ts_to_iso(first_visit_ts),
                "last_visit": _firefox_ts_to_iso(last_visit_ts),
                "frecency": data["frecency"] or 0,
                "visit_types": visit_type_counts,
                "recent_visits": recent_visits,
                "source_profile": profile.slug,
                "source_machine": profile.machine_name,
                "firefox_place_id": place_id,
                "description": data["description"] or "",
                "privacy": {
                    "url_sanitized": True,
                    "params_stripped": sanitize_result.params_stripped,
                    "domain_checked": True,
                    "policy_applied": sanitize_result.policy_applied,
                },
            }
            if engagement:
                contents["engagement"] = engagement

            # Dedup: hash of key fields detects updates
            content_hash = sensor_state.compute_hash(
                json.dumps({
                    "url_clean": url_clean,
                    "visit_count": data["visit_count"],
                    "title": data["title"],
                    "frecency": data["frecency"],
                }, sort_keys=True, default=str)
            )

            dedup_key = f"{profile.slug}/h-{entry_hash}"
            change = sensor_state.has_changed(dedup_key, content_hash, self.state)
            if change is None:
                continue

            rid = LegionBrowserHistory(profile=profile.slug, entry_id=f"h-{entry_hash}")
            bundle = Bundle.generate(rid=rid, contents=contents)
            bundles.append(bundle)
            self.state[dedup_key] = content_hash

        # Update high-water mark
        if max_visit_id > last_visit_id:
            self.state[state_key] = str(max_visit_id)

        return bundles

    def _poll_bookmarks(self, conn: sqlite3.Connection, profile: FirefoxProfile) -> list[Bundle]:
        """Poll for new or modified bookmarks since last high-water mark."""
        state_key = f"{profile.slug}/last_bookmark_modified"
        last_modified = int(self.state.get(state_key, 0))
        bundles = []

        cursor = conn.execute(
            """
            SELECT b.id, b.type, b.fk, b.parent, b.position, b.title,
                   b.dateAdded, b.lastModified, b.guid,
                   p.url, p.title AS page_title
            FROM moz_bookmarks b
            LEFT JOIN moz_places p ON b.fk = p.id
            WHERE b.type = 1 AND b.lastModified > ?
            ORDER BY b.lastModified
            LIMIT ?
            """,
            (last_modified, self.batch_size),
        )

        max_modified = last_modified

        for row in cursor:
            url = row["url"] or ""
            domain = self._extract_domain(url)
            guid = row["guid"] or ""

            # Suppression check (uses configurable list)
            if url and is_suppressed(url, domain, self._privacy.suppression):
                continue

            if url:
                sanitize_result = sanitize_url_ext(url, domain, self._privacy.params)
                url_clean = sanitize_result.url
            else:
                sanitize_result = sanitize_url_ext("", domain, self._privacy.params)
                url_clean = ""
            max_modified = max(max_modified, row["lastModified"] or 0)

            # Build folder path
            folder_path = self._get_folder_path(conn, row["parent"])

            # Get tags
            tags = self._get_tags(conn, row["fk"]) if row["fk"] else []

            contents = {
                "type": "bookmark",
                "url": url,
                "url_clean": url_clean,
                "domain": domain,
                "title": row["title"] or "",
                "page_title": row["page_title"] or "",
                "folder_path": folder_path,
                "tags": tags,
                "date_added": _firefox_ts_to_iso(row["dateAdded"]),
                "last_modified": _firefox_ts_to_iso(row["lastModified"]),
                "guid": guid,
                "source_profile": profile.slug,
                "source_machine": profile.machine_name,
                "privacy": {
                    "url_sanitized": bool(url),
                    "params_stripped": sanitize_result.params_stripped,
                    "domain_checked": True,
                    "policy_applied": sanitize_result.policy_applied,
                },
            }

            content_hash = sensor_state.compute_hash(
                json.dumps(contents, sort_keys=True, default=str)
            )

            dedup_key = f"{profile.slug}/b-{guid}"
            change = sensor_state.has_changed(dedup_key, content_hash, self.state)
            if change is None:
                continue

            rid = LegionBrowserHistory(profile=profile.slug, entry_id=f"b-{guid}")
            bundle = Bundle.generate(rid=rid, contents=contents)
            bundles.append(bundle)
            self.state[dedup_key] = content_hash

        # Update high-water mark
        if max_modified > last_modified:
            self.state[state_key] = str(max_modified)

        return bundles

    def _get_engagement(self, conn: sqlite3.Connection, place_id: int) -> dict | None:
        """Get engagement metadata for a place, if moz_places_metadata exists."""
        try:
            row = conn.execute(
                """
                SELECT total_view_time, key_presses, scrolling_distance, typing_time
                FROM moz_places_metadata
                WHERE place_id = ?
                LIMIT 1
                """,
                (place_id,),
            ).fetchone()
            if row:
                return {
                    "total_view_time_ms": row["total_view_time"] or 0,
                    "key_presses": row["key_presses"] or 0,
                    "scrolling_distance": row["scrolling_distance"] or 0,
                    "typing_time_ms": row["typing_time"] or 0,
                }
        except sqlite3.OperationalError:
            # Table doesn't exist on older Firefox versions
            pass
        return None

    def _get_folder_path(self, conn: sqlite3.Connection, parent_id: int | None) -> str:
        """Walk bookmark parent chain to build folder path."""
        parts = []
        current_id = parent_id
        seen = set()  # Guard against cycles

        while current_id and current_id not in seen:
            seen.add(current_id)
            row = conn.execute(
                "SELECT id, parent, title, guid FROM moz_bookmarks WHERE id = ?",
                (current_id,),
            ).fetchone()
            if not row:
                break
            if row["guid"] in ROOT_BOOKMARK_GUIDS:
                break
            if row["title"]:
                parts.append(row["title"])
            current_id = row["parent"]

        parts.reverse()
        return "/" + "/".join(parts) if parts else "/"

    def _get_tags(self, conn: sqlite3.Connection, place_id: int) -> list[str]:
        """Get tags for a bookmarked URL via the tag-folder JOIN pattern."""
        try:
            cursor = conn.execute(
                """
                SELECT t.title AS tag_name
                FROM moz_bookmarks b
                JOIN moz_bookmarks t ON b.parent = t.id
                JOIN moz_bookmarks tags_root ON t.parent = tags_root.id
                WHERE b.fk = ? AND tags_root.guid = 'tags________'
                """,
                (place_id,),
            )
            return [row["tag_name"] for row in cursor if row["tag_name"]]
        except sqlite3.OperationalError:
            return []

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""
