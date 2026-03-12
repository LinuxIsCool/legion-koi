"""Health dimension checkers — availability, performance, quality, growth.

Each checker returns (score: float 0-1, detail: str).
"""

from __future__ import annotations

import subprocess

import structlog

from ..constants import (
    HEALTH_SERVICES,
    EVENT_STREAM_PREFIX,
)
from ..events.schemas import BUNDLE_CREATED
from ..events.bus import stream_name

log = structlog.stdlib.get_logger()


# --- Availability ---


def check_availability(
    storage=None,
    event_bus=None,
) -> tuple[float, str]:
    """Check service availability. Score = fraction of healthy checks."""
    checks = {}
    total = 0
    healthy = 0

    # systemd services
    for service in HEALTH_SERVICES:
        total += 1
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", service],
                capture_output=True,
                text=True,
                timeout=5,
            )
            active = result.stdout.strip() == "active"
            checks[service] = "active" if active else result.stdout.strip()
            if active:
                healthy += 1
        except Exception:
            checks[service] = "check_failed"

    # PostgreSQL
    total += 1
    if storage:
        try:
            conn = storage._get_conn()
            conn.execute("SELECT 1")
            checks["postgresql"] = "ok"
            healthy += 1
        except Exception:
            checks["postgresql"] = "unreachable"
    else:
        checks["postgresql"] = "not_configured"

    # Redis/FalkorDB (via event bus)
    total += 1
    if event_bus and event_bus.ping():
        checks["redis"] = "ok"
        healthy += 1
    else:
        checks["redis"] = "unreachable"

    # Event consumer lag (if event bus available)
    if event_bus:
        try:
            sname = stream_name(BUNDLE_CREATED)
            lag = event_bus.pending_count(sname, "embed")
            checks["embed_lag"] = lag
            lag2 = event_bus.pending_count(sname, "extract")
            checks["extract_lag"] = lag2
        except Exception:
            pass

    score = healthy / total if total > 0 else 0.0
    detail = f"{healthy}/{total} checks pass"
    for k, v in checks.items():
        detail += f", {k}={v}"
    return score, detail


# --- Performance ---


def check_performance(
    event_bus=None,
) -> tuple[float, str]:
    """Check throughput and latency indicators.

    Without a time-series store, we use stream length as a proxy for
    throughput. Full latency tracking requires Phase 5+ instrumentation.
    """
    parts = []

    if event_bus:
        try:
            embed_stream = stream_name("embedding.computed")
            embed_count = event_bus.stream_length(embed_stream)
            parts.append(f"embed_events={embed_count}")

            extract_stream = stream_name("entity.extracted")
            extract_count = event_bus.stream_length(extract_stream)
            parts.append(f"extract_events={extract_count}")
        except Exception:
            pass

    # Without full instrumentation, assume baseline performance
    # Phase 5+ will add latency tracking via event timestamps
    score = 0.8 if parts else 0.5
    detail = ", ".join(parts) if parts else "no event data (baseline estimate)"
    return score, detail


# --- Quality ---


def check_quality(
    storage=None,
) -> tuple[float, str]:
    """Check entity extraction confidence and embedding coverage."""
    if not storage:
        return 0.0, "no storage"

    parts = []
    scores = []

    try:
        entity_stats = storage.get_entity_stats()

        # Entity type coverage: how many ontology types are represented
        type_count = len(entity_stats.get("by_type", {}))
        # Rough benchmark: 10+ types = good coverage
        type_score = min(type_count / 10, 1.0)
        scores.append(type_score)
        parts.append(f"entity_types={type_count}")

        # Extraction coverage
        cov = entity_stats.get("extraction_coverage", {})
        extracted = cov.get("bundles_with_entities", 0)
        total_bundles = cov.get("total_bundles", 1)
        extract_ratio = extracted / total_bundles if total_bundles > 0 else 0
        scores.append(extract_ratio)
        parts.append(f"extracted={extracted}/{total_bundles} ({extract_ratio:.1%})")
    except Exception as e:
        parts.append(f"entity_error={e}")

    try:
        config_stats = storage.get_config_stats()
        if config_stats:
            total_embedded = sum(s["embedded"] for s in config_stats)
            total_bundles = sum(s["total"] for s in config_stats) // max(1, len(set(s["config_id"] for s in config_stats)))
            embed_ratio = total_embedded / total_bundles if total_bundles > 0 else 0
            scores.append(embed_ratio)
            parts.append(f"embed_coverage={total_embedded}/{total_bundles} ({embed_ratio:.1%})")
    except Exception as e:
        parts.append(f"embed_error={e}")

    score = sum(scores) / len(scores) if scores else 0.0
    detail = ", ".join(parts) if parts else "no data"
    return score, detail


# --- Growth ---


def check_growth(
    storage=None,
) -> tuple[float, str]:
    """Check recent ingestion activity and coverage trends."""
    if not storage:
        return 0.0, "no storage"

    parts = []

    try:
        conn = storage._get_conn()

        # Bundles in last 7 days
        row = conn.execute(
            "SELECT count(*) AS cnt FROM bundles WHERE created_at > NOW() - INTERVAL '7 days'"
        ).fetchone()
        recent_bundles = row["cnt"] if row else 0
        daily_rate = recent_bundles / 7
        parts.append(f"{daily_rate:.0f} bundles/day (7d)")

        # Entities in last 7 days
        row = conn.execute(
            "SELECT count(*) AS cnt FROM entities WHERE first_seen > NOW() - INTERVAL '7 days'"
        ).fetchone()
        recent_entities = row["cnt"] if row else 0
        parts.append(f"{recent_entities} new entities (7d)")

        # Score: >10 bundles/day = healthy growth
        # Below 1/day = concerning (sensors may be down)
        if daily_rate >= 10:
            score = 1.0
        elif daily_rate >= 1:
            score = 0.5 + (daily_rate / 20)
        else:
            score = max(0.1, daily_rate)

    except Exception as e:
        score = 0.0
        parts.append(f"error={e}")

    detail = ", ".join(parts) if parts else "no data"
    return score, detail
