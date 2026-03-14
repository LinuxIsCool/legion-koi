"""Thread-level relevance filter for message ingestion.

Classification logic (evaluated in order):
1. Explicit exclude list → EXCLUDED (blocked)
2. Non-Telegram platforms (Signal, Email, Slack) → ALWAYS (T1)
3. DMs (thread_type = dm or private) → ALWAYS (T1)
4. Explicit include list → INCLUDED (T3)
5. Self posted AND ratio ≥ LOW_SIGNAL_PARTICIPATION_FLOOR → ACTIVE (T2)
6. Self posted AND ratio < floor → LOW_SIGNAL (T2b)
7. Default → RECON (T4 — ingested at lowest priority)
"""

import sqlite3
import time
from enum import Enum
from pathlib import Path

import structlog

from ..constants import LOW_SIGNAL_PARTICIPATION_FLOOR, MESSAGE_THREAD_CACHE_TTL

log = structlog.stdlib.get_logger()


class MessageTier(Enum):
    ALWAYS = "always"           # T1: DMs, non-Telegram platforms
    ACTIVE = "active"           # T2: Shawn posted, ratio ≥ floor
    LOW_SIGNAL = "low_signal"   # T2b: Shawn posted, ratio < floor
    INCLUDED = "included"       # T3: Lurk group explicitly included
    RECON = "recon"             # T4: Unknown group — ingested at lowest priority
    EXCLUDED = "excluded"       # Explicit exclude override — NOT ingested


class MessageFilter:
    """Thread-level relevance filter for message ingestion.

    Opens a read-only SQLite connection to messages.db and caches thread
    classifications. Cache refreshes every MESSAGE_THREAD_CACHE_TTL seconds.
    """

    def __init__(
        self,
        messages_db_path: Path,
        self_sender_ids: list[str],
        thread_includes: list[str],
        thread_excludes: list[str],
        enable: bool = True,
    ):
        self._db_path = Path(messages_db_path).expanduser()
        self._self_sender_ids = set(self_sender_ids)
        self._thread_includes = set(thread_includes)
        self._thread_excludes = set(thread_excludes)
        self._enabled = enable

        # Cache: thread_id → {"thread_type": str, "shawn_posts": int, "msg_count": int}
        self._thread_cache: dict[str, dict] = {}
        self._cache_built_at: float = 0.0

    def should_ingest(self, thread_id: str, platform: str) -> bool:
        """Returns True if the message should become a KOI bundle."""
        if not self._enabled:
            return True
        tier = self.classify(thread_id, platform)
        return tier != MessageTier.EXCLUDED

    def classify(self, thread_id: str, platform: str) -> MessageTier:
        """Return the tier for a given thread."""
        if not self._enabled:
            return MessageTier.ALWAYS

        # 1. Explicit excludes — check first, overrides everything
        if thread_id in self._thread_excludes:
            return MessageTier.EXCLUDED

        # 2. Non-Telegram → always ingest
        if platform != "telegram":
            return MessageTier.ALWAYS

        # 3. DMs — check thread type from cache
        self._ensure_cache()
        info = self._thread_cache.get(thread_id)
        if info and info["thread_type"] in ("dm", "private"):
            return MessageTier.ALWAYS

        # 4. Explicit includes
        if thread_id in self._thread_includes:
            return MessageTier.INCLUDED

        # 5/6. Shawn posted — split ACTIVE vs LOW_SIGNAL by participation ratio
        if info and info["shawn_posts"] > 0:
            msg_count = info["msg_count"]
            if msg_count > 0:
                ratio = info["shawn_posts"] / msg_count
                if ratio >= LOW_SIGNAL_PARTICIPATION_FLOOR:
                    return MessageTier.ACTIVE
                return MessageTier.LOW_SIGNAL
            # Edge case: shawn_posts > 0 but msg_count = 0 shouldn't happen,
            # but if it does, treat as ACTIVE (participated but no count data)
            return MessageTier.ACTIVE

        # 7. Default: RECON — ingest at lowest priority
        return MessageTier.RECON

    def _ensure_cache(self) -> None:
        """Rebuild thread cache if stale or empty."""
        now = time.monotonic()
        if self._thread_cache and (now - self._cache_built_at) < MESSAGE_THREAD_CACHE_TTL:
            return
        self._build_thread_cache()
        self._cache_built_at = now

    def _build_thread_cache(self) -> None:
        """Query messages DB for thread types, self post counts, and total message counts."""
        if not self._db_path.exists():
            log.warning("message_filter.db_missing", path=str(self._db_path))
            return

        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # Build sender placeholders for the IN clause
            sender_placeholders = ",".join("?" for _ in self._self_sender_ids)
            sender_ids = list(self._self_sender_ids)

            rows = conn.execute(f"""
                SELECT
                    t.id AS thread_id,
                    t.thread_type,
                    COUNT(m.id) AS msg_count,
                    COALESCE(SUM(
                        CASE WHEN m.sender_id IN ({sender_placeholders}) THEN 1 ELSE 0 END
                    ), 0) AS shawn_posts
                FROM threads t
                LEFT JOIN messages m ON m.thread_id = t.id
                GROUP BY t.id
            """, sender_ids).fetchall()

            cache = {}
            for row in rows:
                cache[row["thread_id"]] = {
                    "thread_type": row["thread_type"] or "unknown",
                    "shawn_posts": row["shawn_posts"],
                    "msg_count": row["msg_count"],
                }
            self._thread_cache = cache
            log.info(
                "message_filter.cache_built",
                threads=len(cache),
                includes=len(self._thread_includes),
                excludes=len(self._thread_excludes),
            )
        finally:
            conn.close()

    def stats(self) -> dict:
        """Return classification stats for observability."""
        self._ensure_cache()
        counts = {tier: 0 for tier in MessageTier}
        for thread_id, info in self._thread_cache.items():
            # Determine platform — threads table doesn't store platform directly
            # in cache, so we infer: if it's in the cache, it came from the
            # threads table join. We classify as telegram since we only cache
            # telegram threads in practice, but use "telegram" as default.
            tier = self.classify(thread_id, "telegram")
            counts[tier] += 1
        return {tier.value: count for tier, count in counts.items()}
