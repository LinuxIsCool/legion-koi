"""Health model — four-dimension scoring with weighted composite.

Dimensions:
- Availability (0.35): are all services and infrastructure running?
- Performance (0.25): is throughput/latency within acceptable range?
- Quality (0.25): entity extraction confidence, embedding coverage
- Growth (0.15): bundles/day, coverage trends
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..constants import (
    HEALTH_WEIGHT_AVAILABILITY,
    HEALTH_WEIGHT_PERFORMANCE,
    HEALTH_WEIGHT_QUALITY,
    HEALTH_WEIGHT_GROWTH,
)


@dataclass
class HealthScore:
    """Four-dimension health assessment with composite score."""

    availability: float = 0.0
    performance: float = 0.0
    quality: float = 0.0
    growth: float = 0.0
    details: dict = field(default_factory=dict)

    @property
    def composite(self) -> float:
        """Weighted composite score (0-100)."""
        raw = (
            self.availability * HEALTH_WEIGHT_AVAILABILITY
            + self.performance * HEALTH_WEIGHT_PERFORMANCE
            + self.quality * HEALTH_WEIGHT_QUALITY
            + self.growth * HEALTH_WEIGHT_GROWTH
        )
        return round(raw * 100, 1)

    def summary(self) -> str:
        """Human-readable health summary."""
        lines = [
            f"Health: {self.composite}/100",
            f"  Availability: {self.availability:.2f} — {self.details.get('availability', '')}",
            f"  Performance:  {self.performance:.2f} — {self.details.get('performance', '')}",
            f"  Quality:      {self.quality:.2f} — {self.details.get('quality', '')}",
            f"  Growth:       {self.growth:.2f} — {self.details.get('growth', '')}",
        ]
        return "\n".join(lines)


def compute_health(
    storage=None,
    event_bus=None,
) -> HealthScore:
    """Compute health score from all available checkers."""
    from .checkers import (
        check_availability,
        check_performance,
        check_quality,
        check_growth,
    )

    avail, avail_detail = check_availability(storage=storage, event_bus=event_bus)
    perf, perf_detail = check_performance(event_bus=event_bus)
    qual, qual_detail = check_quality(storage=storage)
    grow, grow_detail = check_growth(storage=storage)

    return HealthScore(
        availability=avail,
        performance=perf,
        quality=qual,
        growth=grow,
        details={
            "availability": avail_detail,
            "performance": perf_detail,
            "quality": qual_detail,
            "growth": grow_detail,
        },
    )
