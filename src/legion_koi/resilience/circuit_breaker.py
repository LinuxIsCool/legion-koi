"""Circuit breaker pattern for external service calls.

Three states:
- CLOSED (normal): calls pass through, failures tracked
- OPEN (failing): calls short-circuit with CircuitOpenError
- HALF_OPEN (testing): one probe call allowed to test recovery

State transitions:
- CLOSED → OPEN: failure_threshold failures within window
- OPEN → HALF_OPEN: recovery_timeout_seconds elapsed
- HALF_OPEN → CLOSED: probe succeeds
- HALF_OPEN → OPEN: probe fails

Applied to: TELUS API (embeddings), Ollama (extraction), FalkorDB (hippo bridge).
"""

from __future__ import annotations

import time
import threading
from enum import Enum
from typing import Any, Callable

import structlog

from ..constants import (
    CIRCUIT_FAILURE_THRESHOLD,
    CIRCUIT_RECOVERY_TIMEOUT_SECONDS,
)
from ..events.schemas import KoiEvent, SERVICE_DEGRADED, SERVICE_DOWN

log = structlog.stdlib.get_logger()


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and calls are blocked."""

    def __init__(self, name: str, until: float):
        remaining = max(0, until - time.monotonic())
        super().__init__(f"Circuit '{name}' is open, recovery in {remaining:.0f}s")
        self.name = name
        self.until = until


class CircuitBreaker:
    """Three-state circuit breaker for external service calls.

    Thread-safe: uses a lock for state transitions.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
        recovery_timeout_seconds: float = CIRCUIT_RECOVERY_TIMEOUT_SECONDS,
        event_bus=None,
    ):
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._event_bus = event_bus

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    log.info("circuit.half_open", name=self.name)
            return self._state

    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute fn through the circuit breaker.

        Raises CircuitOpenError if the circuit is open.
        On success in HALF_OPEN state, transitions to CLOSED.
        On failure, tracks failures and potentially opens the circuit.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(self.name, self._opened_at + self._recovery_timeout)

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                log.info("circuit.closed", name=self.name, msg="recovered")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success (consecutive failures model)
                self._failure_count = 0

    def _on_failure(self) -> None:
        """Record a failed call and potentially open the circuit."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — reopen
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                log.warning("circuit.reopened", name=self.name)
                self._emit_event(SERVICE_DOWN)

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    log.warning(
                        "circuit.opened",
                        name=self.name,
                        failures=self._failure_count,
                        threshold=self._failure_threshold,
                    )
                    self._emit_event(SERVICE_DEGRADED)

    def _emit_event(self, event_type: str) -> None:
        """Emit a service health event if event bus is available."""
        if self._event_bus:
            try:
                event = KoiEvent(
                    type=event_type,
                    subject=self.name,
                    data={
                        "circuit": self.name,
                        "state": self._state.value,
                        "failure_count": self._failure_count,
                    },
                )
                self._event_bus.publish(event)
            except Exception:
                pass  # Event emission is best-effort

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            log.info("circuit.reset", name=self.name)
