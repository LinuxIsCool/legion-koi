"""Tests for the circuit breaker pattern (Phase 3: Supervisor Tree)."""

import time
from unittest.mock import MagicMock

import pytest

from legion_koi.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)
from legion_koi.constants import CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_RECOVERY_TIMEOUT_SECONDS


# --- State transitions ---

def test_starts_closed():
    cb = CircuitBreaker(name="test")
    assert cb.state == CircuitState.CLOSED


def test_stays_closed_on_success():
    cb = CircuitBreaker(name="test")
    result = cb.call(lambda: 42)
    assert result == 42
    assert cb.state == CircuitState.CLOSED


def test_opens_after_threshold_failures():
    cb = CircuitBreaker(name="test", failure_threshold=3)
    for _ in range(3):
        with pytest.raises(ValueError):
            cb.call(_failing_fn)
    assert cb.state == CircuitState.OPEN


def test_open_raises_circuit_open_error():
    cb = CircuitBreaker(name="test", failure_threshold=1)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError) as exc_info:
        cb.call(lambda: 42)
    assert exc_info.value.name == "test"


def test_half_open_after_recovery_timeout():
    cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout_seconds=0.5)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert cb.state == CircuitState.OPEN
    # Before timeout elapses, should still be OPEN
    assert cb.state == CircuitState.OPEN
    # After timeout elapses, transitions to HALF_OPEN
    time.sleep(0.6)
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_closes_on_success():
    cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout_seconds=0.5)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    time.sleep(0.6)
    assert cb.state == CircuitState.HALF_OPEN
    result = cb.call(lambda: "recovered")
    assert result == "recovered"
    assert cb.state == CircuitState.CLOSED


def test_half_open_reopens_on_failure():
    cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout_seconds=0.5)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    time.sleep(0.6)
    assert cb.state == CircuitState.HALF_OPEN
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert cb.state == CircuitState.OPEN


# --- Failure counting ---

def test_success_resets_failure_count():
    """Successes reset the consecutive failure counter."""
    cb = CircuitBreaker(name="test", failure_threshold=3)
    # Two failures, then a success
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(_failing_fn)
    cb.call(lambda: "ok")
    # Two more failures should NOT open (count was reset)
    for _ in range(2):
        with pytest.raises(ValueError):
            cb.call(_failing_fn)
    assert cb.state == CircuitState.CLOSED


def test_exact_threshold_opens():
    """Circuit opens at exactly the threshold, not before."""
    cb = CircuitBreaker(name="test", failure_threshold=3)
    for i in range(2):
        with pytest.raises(ValueError):
            cb.call(_failing_fn)
        assert cb.state == CircuitState.CLOSED, f"Should still be closed after {i+1} failures"
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert cb.state == CircuitState.OPEN


# --- Manual reset ---

def test_manual_reset():
    cb = CircuitBreaker(name="test", failure_threshold=1)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert cb.state == CircuitState.OPEN
    cb.reset()
    assert cb.state == CircuitState.CLOSED
    result = cb.call(lambda: "after_reset")
    assert result == "after_reset"


# --- Event emission ---

def test_emits_service_degraded_on_open():
    bus = MagicMock()
    cb = CircuitBreaker(name="test", failure_threshold=1, event_bus=bus)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert bus.publish.called
    event = bus.publish.call_args[0][0]
    assert event.type == "service.degraded"
    assert event.data["circuit"] == "test"


def test_emits_service_down_on_reopen():
    bus = MagicMock()
    cb = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout_seconds=0.5, event_bus=bus)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    time.sleep(0.6)
    # Now in HALF_OPEN, fail again → reopen → service.down
    bus.reset_mock()
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert bus.publish.called
    event = bus.publish.call_args[0][0]
    assert event.type == "service.down"


def test_no_event_without_bus():
    """Circuit breaker works fine without an event bus."""
    cb = CircuitBreaker(name="test", failure_threshold=1)
    with pytest.raises(ValueError):
        cb.call(_failing_fn)
    assert cb.state == CircuitState.OPEN


# --- CircuitOpenError attributes ---

def test_circuit_open_error_attributes():
    err = CircuitOpenError("myservice", until=time.monotonic() + 30)
    assert err.name == "myservice"
    assert "myservice" in str(err)


# --- Constants ---

def test_default_constants():
    assert CIRCUIT_FAILURE_THRESHOLD == 3
    assert CIRCUIT_RECOVERY_TIMEOUT_SECONDS == 30


# --- Helper ---

def _failing_fn():
    raise ValueError("simulated failure")
