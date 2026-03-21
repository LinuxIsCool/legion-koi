"""Backfill archived Firefox profiles into KOI PostgreSQL.

Reads archived places.sqlite files (crash-recovery snapshots, manual backups,
fleet collections), merges overlapping data across snapshots, applies the Phase 2
privacy pipeline, and inserts bundles directly into PostgreSQL — identical format
to what the live sensor produces.

Usage:
    cd ~/legion-koi && uv run python scripts/browser_backfill.py \
      --manifest ~/.config/claude-browser-history/backfill_manifest.yaml --dry-run

    cd ~/legion-koi && uv run python scripts/browser_backfill.py \
      --manifest ~/.config/claude-browser-history/backfill_manifest.yaml --yes

    cd ~/legion-koi && uv run python scripts/browser_backfill.py \
      --manifest ~/.config/claude-browser-history/backfill_manifest.yaml \
      --profile legion-default-release --yes
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import psycopg
import yaml

# Add project source to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legion_koi.sensors.url_sanitizer import sanitize_url_ext, url_hash, is_suppressed
from legion_koi.sensors.privacy_config import PrivacyConfig
from legion_koi.sensors import state as sensor_state
from legion_koi.storage.postgres import PostgresStorage

# ---------------------------------------------------------------------------
# Constants (mirrored from browser_history_sensor.py)
# ---------------------------------------------------------------------------
VISIT_LINK = 1
VISIT_TYPED = 2
VISIT_BOOKMARK = 3
VISIT_DOWNLOAD = 7

INCLUDED_VISIT_TYPES = (VISIT_LINK, VISIT_TYPED, VISIT_BOOKMARK, VISIT_DOWNLOAD)

VISIT_TYPE_NAMES = {
    VISIT_LINK: "link",
    VISIT_TYPED: "typed",
    VISIT_BOOKMARK: "bookmark",
    VISIT_DOWNLOAD: "download",
}

ROOT_BOOKMARK_GUIDS = frozenset({
    "root________",
    "menu________",
    "toolbar_____",
    "unfiled_____",
    "mobile______",
    "tags________",
})

FIREFOX_EPOCH_DIVISOR = 1_000_000
MAX_RECENT_VISITS = 100
NAMESPACE = "legion.claude-browser-history"
BATCH_SIZE = 500
BACKFILL_STATE_PATH = Path("state/browser_backfill_state.json")


# ---------------------------------------------------------------------------
# Helper functions (adapted from browser_history_sensor.py as module-level)
# ---------------------------------------------------------------------------

def _safe_connect(places_path: Path) -> tuple[sqlite3.Connection, Path | None]:
    """Connect to places.sqlite, handling Firefox locks.

    Returns (connection, tmp_dir_or_None). Caller must clean up tmp_dir.
    """
    # Try read-only first
    try:
        uri = f"file:{places_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("SELECT 1 FROM moz_places LIMIT 1")
        return conn, None
    except sqlite3.OperationalError:
        pass

    # Copy to temp dir and open as immutable
    tmp_dir = Path(tempfile.mkdtemp(prefix="koi-backfill-"))
    try:
        db_name = places_path.name
        shutil.copy2(places_path, tmp_dir / db_name)
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
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _close_safe(conn: sqlite3.Connection, tmp_dir: Path | None) -> None:
    """Close connection and clean up temp dir if present."""
    try:
        conn.close()
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _firefox_ts_to_iso(ts_microseconds: int | None) -> str | None:
    """Convert Firefox microsecond timestamp to ISO 8601 string."""
    if not ts_microseconds:
        return None
    try:
        dt = datetime.fromtimestamp(ts_microseconds / FIREFOX_EPOCH_DIVISOR, tz=timezone.utc)
        return dt.isoformat()
    except (OSError, ValueError, OverflowError):
        return None


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def _get_engagement(conn: sqlite3.Connection, place_id: int) -> dict | None:
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
        pass
    return None


def _get_folder_path(conn: sqlite3.Connection, parent_id: int | None) -> str:
    """Walk bookmark parent chain to build folder path."""
    parts = []
    current_id = parent_id
    seen = set()

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


def _get_tags(conn: sqlite3.Connection, place_id: int) -> list[str]:
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


def _snapshot_sort_key(snapshot: dict) -> str:
    """Sort key for snapshots — newest first by label/path timestamp."""
    label = snapshot.get("label", "")
    # Extract date from crash-YYYYMMDD or similar patterns
    match = re.search(r"(\d{8})", label)
    if match:
        return match.group(1)
    # backup-current sorts last (oldest data equivalent)
    if "current" in label or "backup" in label:
        return "99999999"
    return label


# ---------------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------------

def merge_snapshots(snapshot_paths: list[tuple[Path, str]]) -> tuple[dict, dict]:
    """Merge multiple Firefox places.sqlite snapshots.

    Args:
        snapshot_paths: List of (path, label) tuples, sorted newest-first.

    Returns:
        (merged_places, merged_bookmarks) dicts keyed by URL and GUID respectively.
    """
    merged_places: dict[str, dict] = {}       # keyed by URL
    merged_bookmarks: dict[str, dict] = {}     # keyed by bookmark GUID

    for places_path, label in snapshot_paths:
        if not places_path.exists():
            print(f"  ⚠ Skipping missing: {places_path}")
            continue

        print(f"  Reading: {label} ({places_path.name})")
        conn, tmp_dir = _safe_connect(places_path)
        try:
            _merge_places_from_snapshot(conn, label, merged_places)
            _merge_bookmarks_from_snapshot(conn, label, merged_bookmarks)
        finally:
            _close_safe(conn, tmp_dir)

    return merged_places, merged_bookmarks


def _merge_places_from_snapshot(
    conn: sqlite3.Connection, label: str, merged: dict[str, dict]
) -> None:
    """Read all visited places from a snapshot and merge into the accumulator."""
    cursor = conn.execute(
        "SELECT id, url, title, visit_count, frecency, description "
        "FROM moz_places WHERE visit_count > 0"
    )

    placeholders = ",".join("?" for _ in INCLUDED_VISIT_TYPES)
    place_count = 0
    visit_count = 0

    for row in cursor:
        place_id = row["id"]
        url = row["url"] or ""
        if not url:
            continue

        place_count += 1

        # Collect visits for this place
        visits_cursor = conn.execute(
            f"SELECT visit_date, visit_type FROM moz_historyvisits "
            f"WHERE place_id = ? AND visit_type IN ({placeholders})",
            (place_id, *INCLUDED_VISIT_TYPES),
        )
        visit_set = {(v["visit_date"], v["visit_type"]) for v in visits_cursor}
        visit_count += len(visit_set)

        # Get engagement from this snapshot
        engagement = _get_engagement(conn, place_id)

        if url not in merged:
            merged[url] = {
                "title": row["title"] or "",
                "visit_count": row["visit_count"] or 0,
                "frecency": row["frecency"] or 0,
                "description": row["description"] or "",
                "visits": visit_set,
                "engagement": engagement,
                "source_label": label,
            }
        else:
            existing = merged[url]
            # Keep highest visit_count and frecency
            if (row["visit_count"] or 0) > existing["visit_count"]:
                existing["visit_count"] = row["visit_count"] or 0
            if (row["frecency"] or 0) > existing["frecency"]:
                existing["frecency"] = row["frecency"] or 0
            # Keep title from newest snapshot (first seen wins since we sort newest-first)
            # — title is already set from the first snapshot
            # Union all visits
            existing["visits"] |= visit_set
            # Keep engagement from first snapshot that has it
            if existing["engagement"] is None and engagement is not None:
                existing["engagement"] = engagement

    print(f"    {place_count:,} places, {visit_count:,} visits")


def _merge_bookmarks_from_snapshot(
    conn: sqlite3.Connection, label: str, merged: dict[str, dict]
) -> None:
    """Read all bookmarks from a snapshot and merge into the accumulator."""
    cursor = conn.execute(
        """
        SELECT b.id, b.type, b.fk, b.parent, b.position, b.title,
               b.dateAdded, b.lastModified, b.guid,
               p.url, p.title AS page_title
        FROM moz_bookmarks b
        LEFT JOIN moz_places p ON b.fk = p.id
        WHERE b.type = 1
        """
    )

    bm_count = 0
    for row in cursor:
        guid = row["guid"] or ""
        if not guid:
            continue

        bm_count += 1

        if guid not in merged:
            # First (newest) snapshot wins for bookmark data
            folder_path = _get_folder_path(conn, row["parent"])
            tags = _get_tags(conn, row["fk"]) if row["fk"] else []

            merged[guid] = {
                "url": row["url"] or "",
                "title": row["title"] or "",
                "page_title": row["page_title"] or "",
                "date_added": row["dateAdded"],
                "last_modified": row["lastModified"],
                "fk": row["fk"],
                "folder_path": folder_path,
                "tags": tags,
                "source_label": label,
            }

    if bm_count:
        print(f"    {bm_count:,} bookmarks")


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------

def build_bundles(
    merged_places: dict[str, dict],
    merged_bookmarks: dict[str, dict],
    profile_slug: str,
    machine_name: str,
    privacy_config: PrivacyConfig,
) -> tuple[list[dict], int]:
    """Build KOI bundle dicts from merged data.

    Returns (bundle_list, suppressed_count).
    """
    bundles = []
    suppressed = 0

    # --- History bundles ---
    for url, data in merged_places.items():
        domain = _extract_domain(url)

        if is_suppressed(url, domain, privacy_config.suppression):
            suppressed += 1
            continue

        sanitize_result = sanitize_url_ext(url, domain, privacy_config.params)
        url_clean = sanitize_result.url
        entry_hash = url_hash(url)

        # Aggregate visit types from the merged visit set
        visit_type_counts: dict[str, int] = {}
        recent_visits = []
        first_visit_ts: int | None = None
        last_visit_ts: int | None = None

        for visit_date, visit_type in data["visits"]:
            vname = VISIT_TYPE_NAMES.get(visit_type, f"type_{visit_type}")
            visit_type_counts[vname] = visit_type_counts.get(vname, 0) + 1

            visit_iso = _firefox_ts_to_iso(visit_date)
            if visit_iso:
                recent_visits.append({"time": visit_iso, "type": vname})
                if first_visit_ts is None or visit_date < first_visit_ts:
                    first_visit_ts = visit_date
                if last_visit_ts is None or visit_date > last_visit_ts:
                    last_visit_ts = visit_date

        # Cap and sort recent visits
        recent_visits = sorted(
            recent_visits, key=lambda x: x["time"], reverse=True
        )[:MAX_RECENT_VISITS]

        contents = {
            "type": "history",
            "url": url,
            "url_clean": url_clean,
            "domain": domain,
            "title": data["title"],
            "visit_count": data["visit_count"],
            "first_visit": _firefox_ts_to_iso(first_visit_ts),
            "last_visit": _firefox_ts_to_iso(last_visit_ts),
            "frecency": data["frecency"],
            "visit_types": visit_type_counts,
            "recent_visits": recent_visits,
            "source_profile": profile_slug,
            "source_machine": machine_name,
            "firefox_place_id": 0,  # Not meaningful across merged snapshots
            "description": data["description"],
            "privacy": {
                "url_sanitized": True,
                "params_stripped": sanitize_result.params_stripped,
                "domain_checked": True,
                "policy_applied": sanitize_result.policy_applied,
            },
        }
        if data.get("engagement"):
            contents["engagement"] = data["engagement"]

        content_hash = sensor_state.compute_hash(
            json.dumps({
                "url_clean": url_clean,
                "visit_count": data["visit_count"],
                "title": data["title"],
                "frecency": data["frecency"],
            }, sort_keys=True, default=str)
        )

        rid = f"orn:{NAMESPACE}:{profile_slug}/h-{entry_hash}"
        reference = f"{profile_slug}/h-{entry_hash}"

        bundles.append({
            "rid": rid,
            "namespace": NAMESPACE,
            "reference": reference,
            "contents": contents,
            "sha256_hash": content_hash,
        })

    # --- Bookmark bundles ---
    for guid, data in merged_bookmarks.items():
        url = data["url"]
        domain = _extract_domain(url)

        if url and is_suppressed(url, domain, privacy_config.suppression):
            suppressed += 1
            continue

        if url:
            sanitize_result = sanitize_url_ext(url, domain, privacy_config.params)
            url_clean = sanitize_result.url
        else:
            sanitize_result = sanitize_url_ext("", domain, privacy_config.params)
            url_clean = ""

        contents = {
            "type": "bookmark",
            "url": url,
            "url_clean": url_clean,
            "domain": domain,
            "title": data["title"],
            "page_title": data["page_title"],
            "folder_path": data["folder_path"],
            "tags": data["tags"],
            "date_added": _firefox_ts_to_iso(data["date_added"]),
            "last_modified": _firefox_ts_to_iso(data["last_modified"]),
            "guid": guid,
            "source_profile": profile_slug,
            "source_machine": machine_name,
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

        rid = f"orn:{NAMESPACE}:{profile_slug}/b-{guid}"
        reference = f"{profile_slug}/b-{guid}"

        bundles.append({
            "rid": rid,
            "namespace": NAMESPACE,
            "reference": reference,
            "contents": contents,
            "sha256_hash": content_hash,
        })

    return bundles, suppressed


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load legion-koi config.yaml for PostgreSQL DSN."""
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_backfill_state(state_path: Path) -> dict:
    """Load backfill checkpoint state."""
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"completed_profiles": {}}


def save_backfill_state(state_path: Path, state: dict) -> None:
    """Save backfill checkpoint state."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill archived Firefox profiles into KOI PostgreSQL."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="Path to backfill_manifest.yaml",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making changes",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip confirmation prompts",
    )
    parser.add_argument(
        "--profile", type=str, default=None,
        help="Only process this profile slug (e.g. legion-default-release)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process even if profile was already completed",
    )
    args = parser.parse_args()

    # Load manifest
    manifest_path = args.manifest.expanduser()
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}")
        sys.exit(1)
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    profiles = manifest.get("profiles", [])
    if not profiles:
        print("No profiles defined in manifest.")
        sys.exit(0)

    # Load KOI config for DSN
    config = load_config()
    postgres_config = config.get("postgres", {})
    dsn = postgres_config.get("dsn", "postgresql://localhost/personal_koi")

    # Load privacy config
    privacy_config = PrivacyConfig(
        suppression_path=Path("~/.config/claude-browser-history/suppressed_domains.txt").expanduser(),
        param_policy_path=Path("~/.config/claude-browser-history/param_policy.yaml").expanduser(),
    )

    # Load backfill checkpoint
    state_path = BACKFILL_STATE_PATH
    backfill_state = load_backfill_state(state_path)

    total_ingested = 0
    total_suppressed = 0

    for profile_def in profiles:
        machine_name = profile_def["machine_name"]
        profile_name = profile_def["profile_name"]
        slug = f"{machine_name}-{profile_name}"

        # Filter by --profile if specified
        if args.profile and args.profile != slug:
            continue

        # Skip if already completed (unless --force)
        if slug in backfill_state["completed_profiles"] and not args.force:
            prev = backfill_state["completed_profiles"][slug]
            print(f"\nSkipping {slug} (already completed on {prev.get('completed_at', '?')})")
            print(f"  Bundles ingested: {prev.get('bundles_ingested', '?')}")
            print(f"  Use --force to re-process.")
            continue

        # Resolve snapshot paths
        snapshots = profile_def.get("snapshots", [])
        resolved: list[tuple[Path, str]] = []
        for snap in snapshots:
            snap_path = Path(snap["path"]).expanduser()
            if snap_path.exists():
                resolved.append((snap_path, snap["label"]))
            else:
                print(f"  ⚠ Missing snapshot: {snap_path}")

        if not resolved:
            print(f"\nSkipping {slug}: no valid snapshots found")
            continue

        # Sort snapshots newest-first. Treat "backup-current" as oldest
        # (label contains no date, assign "00000000" so it falls last).
        resolved.sort(
            key=lambda x: "00000000" if "current" in x[1] or "backup" in x[1]
            else _snapshot_sort_key({"label": x[1]}),
            reverse=True,
        )

        print(f"\n{'='*60}")
        print(f"Profile: {slug} ({len(resolved)} snapshots)")
        print(f"{'='*60}")

        # Merge all snapshots
        merged_places, merged_bookmarks = merge_snapshots(resolved)
        print(f"\nMerge results:")
        print(f"  Unique URLs:      {len(merged_places):,}")
        print(f"  Unique bookmarks: {len(merged_bookmarks):,}")

        # Build bundles
        bundles, suppressed = build_bundles(
            merged_places, merged_bookmarks,
            profile_slug=slug,
            machine_name=machine_name,
            privacy_config=privacy_config,
        )
        total_suppressed += suppressed

        print(f"\nBundle construction:")
        print(f"  Suppressed:       {suppressed:,}")
        print(f"  Total bundles:    {len(bundles):,}")

        if args.dry_run:
            print(f"\n[DRY RUN] No changes made.")
            continue

        if not bundles:
            print("No bundles to ingest.")
            continue

        # Confirm
        if not args.yes:
            answer = input(f"\nIngest {len(bundles):,} bundles into PostgreSQL? [y/N] ")
            if answer.lower() != "y":
                print("Aborted.")
                continue

        # Insert in batches
        print(f"\nInserting into PostgreSQL...")
        storage = PostgresStorage(dsn)
        for i in range(0, len(bundles), BATCH_SIZE):
            batch = bundles[i:i + BATCH_SIZE]
            storage.upsert_bundles_batch(batch)
            inserted = min(i + BATCH_SIZE, len(bundles))
            print(f"  {inserted:,} / {len(bundles):,}")
        storage.close()

        total_ingested += len(bundles)

        # Update checkpoint
        backfill_state["completed_profiles"][slug] = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "snapshots_processed": len(resolved),
            "unique_urls": len(merged_places),
            "unique_bookmarks": len(merged_bookmarks),
            "bundles_ingested": len(bundles),
            "suppressed": suppressed,
        }
        save_backfill_state(state_path, backfill_state)
        print(f"  Checkpoint saved.")

    # Vacuum if we inserted anything
    if total_ingested > 0:
        print(f"\nRunning VACUUM ANALYZE...")
        vacuum_conn = psycopg.connect(dsn, autocommit=True)
        vacuum_conn.execute("VACUUM ANALYZE bundles")
        vacuum_conn.close()
        print("Done.")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Backfill complete")
    processed = [
        p for p in manifest.get("profiles", [])
        if not args.profile or f"{p['machine_name']}-{p['profile_name']}" == args.profile
    ]
    print(f"  Profiles processed: {len(processed)}")
    print(f"  Total bundles ingested: {total_ingested:,}")
    print(f"  Total suppressed: {total_suppressed:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
