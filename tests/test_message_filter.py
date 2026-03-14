"""Tests for MessageFilter — thread-level relevance filtering."""

import sqlite3

import pytest

from legion_koi.sensors.message_filter import MessageFilter, MessageTier


@pytest.fixture
def messages_db(tmp_path):
    """Create a minimal messages.db with threads and messages for testing."""
    db_path = tmp_path / "messages.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            title TEXT,
            thread_type TEXT,
            participants TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            platform TEXT NOT NULL,
            thread_id TEXT,
            sender_id TEXT,
            content TEXT,
            content_type TEXT DEFAULT 'text',
            reply_to TEXT,
            metadata TEXT DEFAULT '{}',
            platform_ts TEXT NOT NULL,
            synced_at TEXT NOT NULL
        )
    """)

    # Seed threads
    threads = [
        # DMs
        ("telegram:chat:100", "telegram", "Alice DM", "dm", "2024-01-01", "2024-01-01"),
        ("telegram:chat:101", "telegram", "Bob DM", "private", "2024-01-01", "2024-01-01"),
        # Groups where Shawn posted — high participation
        ("telegram:chat:200", "telegram", "Regen Network", "group", "2024-01-01", "2024-01-01"),
        # Lurk groups (no Shawn posts)
        ("telegram:chat:300", "telegram", "Curve Finance", "group", "2024-01-01", "2024-01-01"),
        ("telegram:chat:301", "telegram", "DeepFunding", "group", "2024-01-01", "2024-01-01"),
        # Signal thread
        ("signal:chat:400", "signal", "Signal Group", "group", "2024-01-01", "2024-01-01"),
        # Low signal group — Shawn posted but ratio < 0.002
        ("telegram:chat:500", "telegram", "Commons Stack", "group", "2024-01-01", "2024-01-01"),
    ]
    conn.executemany(
        "INSERT INTO threads (id, platform, title, thread_type, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        threads,
    )

    # Seed messages — Shawn posted in chat:200 (high ratio) and chat:500 (low ratio)
    messages = [
        # chat:200 — 2 messages, 1 from Shawn → ratio 0.5 (well above 0.002)
        ("msg1", "telegram", "telegram:chat:200", "telegram:user:1441369482", "hello", "text", "2024-01-01", "2024-01-01"),
        ("msg2", "telegram", "telegram:chat:200", "telegram:user:999", "hi", "text", "2024-01-01", "2024-01-01"),
        # chat:300 — 1 message, none from Shawn (lurk)
        ("msg3", "telegram", "telegram:chat:300", "telegram:user:999", "trade here", "text", "2024-01-01", "2024-01-01"),
        # chat:301 — 1 message, none from Shawn (lurk)
        ("msg4", "telegram", "telegram:chat:301", "telegram:user:999", "deep funding", "text", "2024-01-01", "2024-01-01"),
        # chat:100 — DM
        ("msg5", "telegram", "telegram:chat:100", "telegram:user:1441369482", "dm msg", "text", "2024-01-01", "2024-01-01"),
        # signal
        ("msg6", "signal", "signal:chat:400", "signal:user:abc", "signal msg", "text", "2024-01-01", "2024-01-01"),
    ]

    # chat:500 — 5000 messages from others, 2 from Shawn → ratio 2/5002 = 0.0004 (below 0.002)
    for i in range(5000):
        messages.append(
            (f"msg500_{i}", "telegram", "telegram:chat:500", "telegram:user:888", f"noise {i}", "text", "2024-01-01", "2024-01-01"),
        )
    messages.append(
        ("msg500_shawn1", "telegram", "telegram:chat:500", "telegram:user:1441369482", "rare post 1", "text", "2024-01-01", "2024-01-01"),
    )
    messages.append(
        ("msg500_shawn2", "telegram", "telegram:chat:500", "telegram:user:1441369482", "rare post 2", "text", "2024-01-01", "2024-01-01"),
    )

    conn.executemany(
        "INSERT INTO messages (id, platform, thread_id, sender_id, content, content_type, platform_ts, synced_at) VALUES (?,?,?,?,?,?,?,?)",
        messages,
    )
    conn.commit()
    conn.close()
    return db_path


SELF_IDS = ["telegram:user:1441369482"]


class TestMessageTierClassification:
    def test_dm_is_always(self, messages_db):
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.classify("telegram:chat:100", "telegram") == MessageTier.ALWAYS

    def test_private_is_always(self, messages_db):
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.classify("telegram:chat:101", "telegram") == MessageTier.ALWAYS

    def test_active_high_participation(self, messages_db):
        """Group where Shawn posted with ratio ≥ 0.002 → ACTIVE."""
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.classify("telegram:chat:200", "telegram") == MessageTier.ACTIVE

    def test_low_signal_group(self, messages_db):
        """Group where Shawn posted but ratio < 0.002 → LOW_SIGNAL."""
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.classify("telegram:chat:500", "telegram") == MessageTier.LOW_SIGNAL

    def test_lurk_group_recon(self, messages_db):
        """Lurk group not in includes → RECON (default-open)."""
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.classify("telegram:chat:300", "telegram") == MessageTier.RECON

    def test_lurk_group_included(self, messages_db):
        """Lurk group explicitly included → INCLUDED."""
        f = MessageFilter(messages_db, SELF_IDS, ["telegram:chat:301"], [], enable=True)
        assert f.classify("telegram:chat:301", "telegram") == MessageTier.INCLUDED

    def test_explicit_exclude(self, messages_db):
        """Explicit exclude overrides everything, even active groups."""
        f = MessageFilter(messages_db, SELF_IDS, [], ["telegram:chat:200"], enable=True)
        assert f.classify("telegram:chat:200", "telegram") == MessageTier.EXCLUDED

    def test_non_telegram_is_always(self, messages_db):
        """Signal, email, slack → ALWAYS regardless of participation."""
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.classify("signal:chat:400", "signal") == MessageTier.ALWAYS

    def test_unknown_thread_defaults_to_recon(self, messages_db):
        """Thread not in the DB at all → RECON (default-open)."""
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.classify("telegram:chat:999", "telegram") == MessageTier.RECON


class TestShouldIngest:
    def test_dm_ingested(self, messages_db):
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.should_ingest("telegram:chat:100", "telegram") is True

    def test_active_group_ingested(self, messages_db):
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.should_ingest("telegram:chat:200", "telegram") is True

    def test_recon_is_ingested(self, messages_db):
        """RECON groups are ingested (default-open behavior)."""
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.should_ingest("telegram:chat:300", "telegram") is True

    def test_low_signal_is_ingested(self, messages_db):
        """LOW_SIGNAL groups are ingested (just tagged for deranking)."""
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=True)
        assert f.should_ingest("telegram:chat:500", "telegram") is True

    def test_excluded_group_skipped(self, messages_db):
        """Only EXCLUDED prevents ingestion."""
        f = MessageFilter(messages_db, SELF_IDS, [], ["telegram:chat:200"], enable=True)
        assert f.should_ingest("telegram:chat:200", "telegram") is False

    def test_included_lurk_ingested(self, messages_db):
        f = MessageFilter(messages_db, SELF_IDS, ["telegram:chat:301"], [], enable=True)
        assert f.should_ingest("telegram:chat:301", "telegram") is True


class TestFilterDisabled:
    def test_disabled_always_ingests(self, messages_db):
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=False)
        assert f.should_ingest("telegram:chat:300", "telegram") is True

    def test_disabled_classifies_as_always(self, messages_db):
        f = MessageFilter(messages_db, SELF_IDS, [], [], enable=False)
        assert f.classify("telegram:chat:300", "telegram") == MessageTier.ALWAYS


class TestExcludeOverridesInclude:
    def test_exclude_wins_over_include(self, messages_db):
        """If a thread is in both includes and excludes, exclude wins."""
        f = MessageFilter(
            messages_db, SELF_IDS,
            thread_includes=["telegram:chat:301"],
            thread_excludes=["telegram:chat:301"],
            enable=True,
        )
        assert f.classify("telegram:chat:301", "telegram") == MessageTier.EXCLUDED
