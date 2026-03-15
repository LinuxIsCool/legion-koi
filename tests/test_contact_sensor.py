"""Tests for contact sensor, RID type, and search/preamble integration."""

import sqlite3

import pytest
from unittest.mock import MagicMock

from legion_koi.rid_types.contact import LegionContact
from legion_koi.sensors.contact_sensor import ContactSensor
from legion_koi.storage.postgres import _extract_search_text
from legion_koi.contextual import extract_preamble


class TestLegionContactRID:
    def test_roundtrip(self):
        rid = LegionContact(identity_id="id-42")
        assert rid.reference == "id-42"
        restored = LegionContact.from_reference("id-42")
        assert restored.identity_id == "id-42"

    def test_namespace(self):
        rid = LegionContact(identity_id="test")
        assert rid.namespace == "legion.claude-contact"

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError):
            LegionContact.from_reference("")

    def test_str_format(self):
        rid = LegionContact(identity_id="id-42")
        assert str(rid) == "orn:legion.claude-contact:id-42"


def _create_contact_db(db_path):
    """Create a minimal SQLite DB with contact_scores and identities tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE identities (
            id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE contact_scores (
            identity_id TEXT PRIMARY KEY REFERENCES identities(id),
            frequency REAL DEFAULT 0,
            recency REAL DEFAULT 0,
            reciprocity REAL DEFAULT 0,
            channel_diversity REAL DEFAULT 0,
            dm_ratio REAL DEFAULT 0,
            structural REAL DEFAULT 0,
            temporal_regularity REAL DEFAULT 0,
            response_latency REAL DEFAULT 0,
            composite REAL DEFAULT 0,
            dunbar_layer TEXT DEFAULT '',
            confidence REAL DEFAULT 0,
            computed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _insert_contact(conn, identity_id, display_name, composite=0.75,
                     dunbar_layer="active-network", computed_at="2026-03-15T12:00:00"):
    conn.execute(
        "INSERT OR REPLACE INTO identities (id, display_name) VALUES (?, ?)",
        (identity_id, display_name),
    )
    conn.execute(
        """INSERT OR REPLACE INTO contact_scores
           (identity_id, frequency, recency, reciprocity,
            channel_diversity, dm_ratio, structural,
            temporal_regularity, response_latency,
            composite, dunbar_layer, confidence, computed_at)
           VALUES (?, 0.5, 0.8, 0.6, 0.3, 0.4, 0.2, 0.7, 0.9, ?, ?, 0.85, ?)""",
        (identity_id, composite, dunbar_layer, computed_at),
    )
    conn.commit()


class TestContactSensor:
    def test_poll_produces_bundles(self, tmp_path):
        db_path = tmp_path / "messages.db"
        conn = _create_contact_db(db_path)
        _insert_contact(conn, "id-1", "Alice")
        _insert_contact(conn, "id-2", "Bob", computed_at="2026-03-15T13:00:00")
        conn.close()

        sensor = ContactSensor(
            db_path=db_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundles = sensor.poll()

        assert len(bundles) == 2
        rids = {b.rid.reference for b in bundles}
        assert rids == {"id-1", "id-2"}

        # Check contents structure
        alice = next(b for b in bundles if b.rid.reference == "id-1")
        assert alice.contents["display_name"] == "Alice"
        assert alice.contents["composite"] == 0.75
        assert alice.contents["dunbar_layer"] == "active-network"

    def test_dedup(self, tmp_path):
        """Second poll with same data produces no bundles."""
        db_path = tmp_path / "messages.db"
        conn = _create_contact_db(db_path)
        _insert_contact(conn, "id-1", "Alice")
        conn.close()

        sensor = ContactSensor(
            db_path=db_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        first = sensor.poll()
        assert len(first) == 1

        second = sensor.poll()
        assert len(second) == 0

    def test_composite_score_present(self, tmp_path):
        db_path = tmp_path / "messages.db"
        conn = _create_contact_db(db_path)
        _insert_contact(conn, "id-1", "Carol", composite=0.92)
        conn.close()

        sensor = ContactSensor(
            db_path=db_path,
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundles = sensor.poll()
        assert bundles[0].contents["composite"] == 0.92

    def test_missing_db_returns_empty(self, tmp_path):
        sensor = ContactSensor(
            db_path=tmp_path / "nonexistent.db",
            state_path=tmp_path / "state.json",
            kobj_push=MagicMock(),
        )
        bundles = sensor.poll()
        assert bundles == []


class TestContactSearchText:
    def test_search_text(self):
        contents = {
            "display_name": "Gregory Makosz",
            "dunbar_layer": "sympathy-group",
            "composite": 0.91,
        }
        result = _extract_search_text("legion.claude-contact", contents)
        assert "Gregory Makosz" in result
        assert "dunbar:sympathy-group" in result
        assert "score:0.91" in result

    def test_search_text_minimal(self):
        contents = {"display_name": "Unknown", "dunbar_layer": "", "composite": None}
        result = _extract_search_text("legion.claude-contact", contents)
        assert "Unknown" in result


class TestContactPreamble:
    def test_preamble_full(self):
        contents = {
            "display_name": "Gregory Makosz",
            "dunbar_layer": "sympathy-group",
        }
        result = extract_preamble("legion.claude-contact", contents)
        assert result == "Contact: Gregory Makosz. Layer: sympathy-group."

    def test_preamble_no_layer(self):
        contents = {"display_name": "Alice", "dunbar_layer": ""}
        result = extract_preamble("legion.claude-contact", contents)
        assert result == "Contact: Alice."

    def test_preamble_no_name(self):
        contents = {"display_name": "", "dunbar_layer": "active-network"}
        result = extract_preamble("legion.claude-contact", contents)
        assert result == ""
