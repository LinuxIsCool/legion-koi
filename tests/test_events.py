"""Tests for the event nervous system (Phase 1)."""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from legion_koi.events.schemas import (
    KoiEvent,
    BUNDLE_CREATED,
    BUNDLE_UPDATED,
    EMBEDDING_COMPUTED,
    ENTITY_EXTRACTED,
    EVENT_SOURCE,
)
from legion_koi.events.bus import EventBus, stream_name, dlq_name
from legion_koi.events.consumer import EventConsumer
from legion_koi.constants import (
    EVENT_STREAM_PREFIX,
    EVENT_DLQ_PREFIX,
    EVENT_DLQ_MAX_RETRIES,
)


# --- Schema tests ---


class TestKoiEvent:
    def test_defaults(self):
        e = KoiEvent(type=BUNDLE_CREATED)
        assert e.type == BUNDLE_CREATED
        assert e.source == EVENT_SOURCE
        assert e.subject == ""
        assert e.data == {}
        assert e.id  # non-empty UUID
        assert e.time  # non-empty ISO timestamp

    def test_to_stream_dict_all_strings(self):
        e = KoiEvent(
            type=BUNDLE_CREATED,
            subject="orn:legion.claude-journal:2026-03-11",
            data={"rid": "orn:legion.claude-journal:2026-03-11", "namespace": "legion.claude-journal"},
        )
        d = e.to_stream_dict()
        for k, v in d.items():
            assert isinstance(v, str), f"Key {k} has non-string value: {type(v)}"

    def test_roundtrip(self):
        original = KoiEvent(
            type=BUNDLE_UPDATED,
            subject="orn:test:123",
            data={"rid": "orn:test:123", "namespace": "test"},
        )
        stream_dict = original.to_stream_dict()
        restored = KoiEvent.from_stream_dict(stream_dict)
        assert restored.type == original.type
        assert restored.subject == original.subject
        assert restored.data == original.data
        assert restored.id == original.id
        assert restored.time == original.time

    def test_from_stream_dict_missing_fields(self):
        """Gracefully handles missing fields from Redis."""
        e = KoiEvent.from_stream_dict({})
        assert e.type == ""
        assert e.source == EVENT_SOURCE
        assert e.data == {}

    def test_event_types_are_strings(self):
        assert isinstance(BUNDLE_CREATED, str)
        assert isinstance(BUNDLE_UPDATED, str)
        assert isinstance(EMBEDDING_COMPUTED, str)
        assert isinstance(ENTITY_EXTRACTED, str)

    def test_to_dict(self):
        e = KoiEvent(type=BUNDLE_CREATED, subject="orn:x:y")
        d = e.to_dict()
        assert d["type"] == BUNDLE_CREATED
        assert d["subject"] == "orn:x:y"
        assert isinstance(d["data"], dict)


# --- Stream naming tests ---


class TestStreamNaming:
    def test_stream_name(self):
        assert stream_name(BUNDLE_CREATED) == f"{EVENT_STREAM_PREFIX}{BUNDLE_CREATED}"

    def test_dlq_name(self):
        sname = stream_name(BUNDLE_CREATED)
        assert dlq_name(sname) == f"{EVENT_DLQ_PREFIX}{sname}"


# --- Consumer framework tests ---


class _TestConsumer(EventConsumer):
    """Concrete consumer for testing."""

    event_type = BUNDLE_CREATED
    group = "test-group"

    def __init__(self, bus, handler_fn=None):
        super().__init__(bus, consumer_id="test-0")
        self._handler_fn = handler_fn or (lambda e: None)

    def handle(self, event: KoiEvent) -> None:
        self._handler_fn(event)


class TestConsumer:
    def test_stream_property(self):
        bus = MagicMock()
        c = _TestConsumer(bus)
        assert c.stream == stream_name(BUNDLE_CREATED)

    def test_process_message_success(self):
        bus = MagicMock()
        handled = []
        c = _TestConsumer(bus, handler_fn=lambda e: handled.append(e))

        event = KoiEvent(type=BUNDLE_CREATED, subject="orn:test:1")
        c._process_message("1-0", event.to_stream_dict())

        assert len(handled) == 1
        assert handled[0].subject == "orn:test:1"
        bus.ack.assert_called_once()

    def test_process_message_dlq_after_max_retries(self):
        bus = MagicMock()
        fail_count = 0

        def always_fail(e):
            nonlocal fail_count
            fail_count += 1
            raise RuntimeError("boom")

        c = _TestConsumer(bus, handler_fn=always_fail)
        event = KoiEvent(type=BUNDLE_CREATED, subject="orn:test:fail")
        stream_dict = event.to_stream_dict()

        # Process EVENT_DLQ_MAX_RETRIES times
        for _ in range(EVENT_DLQ_MAX_RETRIES):
            c._process_message("1-0", stream_dict)

        assert fail_count == EVENT_DLQ_MAX_RETRIES
        # Should have been sent to DLQ on the last attempt
        bus.send_to_dlq.assert_called_once()

    def test_retry_count_resets_on_success(self):
        bus = MagicMock()
        call_count = 0

        def fail_then_succeed(e):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first call fails")

        c = _TestConsumer(bus, handler_fn=fail_then_succeed)
        event = KoiEvent(type=BUNDLE_CREATED, subject="orn:test:retry")
        stream_dict = event.to_stream_dict()

        c._process_message("1-0", stream_dict)  # fails, retry count = 1
        c._process_message("1-0", stream_dict)  # succeeds, retry count cleared

        assert event.id not in c._retry_counts
        bus.send_to_dlq.assert_not_called()


# --- PG trigger SQL syntax test ---


class TestPgTriggerSql:
    def test_trigger_sql_is_valid_syntax(self):
        """Verify the trigger SQL file is parseable (basic checks)."""
        from pathlib import Path

        sql_path = Path(__file__).parent.parent / "src" / "legion_koi" / "events" / "pg_trigger.sql"
        sql = sql_path.read_text()
        assert "CREATE OR REPLACE FUNCTION notify_bundle_change()" in sql
        assert "CREATE TRIGGER bundles_notify" in sql
        assert "pg_notify" in sql
        assert "koi_events" in sql
        assert "AFTER INSERT OR UPDATE ON bundles" in sql


# --- EventBus tests (with mock Redis) ---


class TestEventBus:
    def test_publish_calls_xadd(self):
        with patch("legion_koi.events.bus.redis.Redis") as MockRedis:
            mock_redis = MockRedis.return_value
            mock_redis.xadd.return_value = "1-0"

            bus = EventBus()
            event = KoiEvent(type=BUNDLE_CREATED, subject="orn:test:pub")
            result = bus.publish(event)

            assert result == "1-0"
            mock_redis.xadd.assert_called_once()
            call_args = mock_redis.xadd.call_args
            assert call_args[0][0] == stream_name(BUNDLE_CREATED)

    def test_ensure_group_ignores_busygroup(self):
        """BUSYGROUP error means group already exists — should not raise."""
        import redis as redis_lib

        with patch("legion_koi.events.bus.redis.Redis") as MockRedis:
            mock_redis = MockRedis.return_value
            mock_redis.xgroup_create.side_effect = redis_lib.exceptions.ResponseError(
                "BUSYGROUP Consumer Group name already exists"
            )

            bus = EventBus()
            # Should not raise
            bus.ensure_group("koi:events:bundle.created", "test-group")

    def test_send_to_dlq(self):
        with patch("legion_koi.events.bus.redis.Redis") as MockRedis:
            mock_redis = MockRedis.return_value

            bus = EventBus()
            event = KoiEvent(type=BUNDLE_CREATED, subject="orn:test:dlq")
            sname = stream_name(BUNDLE_CREATED)
            bus.send_to_dlq(sname, event, "test error")

            mock_redis.xadd.assert_called_once()
            call_args = mock_redis.xadd.call_args
            assert call_args[0][0] == dlq_name(sname)
            assert "error" in call_args[0][1]
