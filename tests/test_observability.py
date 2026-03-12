"""Tests for the observability foundation (Phase 2)."""

from unittest.mock import MagicMock, patch

from legion_koi.observability.health import HealthScore, compute_health
from legion_koi.constants import (
    HEALTH_WEIGHT_AVAILABILITY,
    HEALTH_WEIGHT_PERFORMANCE,
    HEALTH_WEIGHT_QUALITY,
    HEALTH_WEIGHT_GROWTH,
)


class TestHealthScore:
    def test_composite_perfect(self):
        h = HealthScore(availability=1.0, performance=1.0, quality=1.0, growth=1.0)
        assert h.composite == 100.0

    def test_composite_zero(self):
        h = HealthScore(availability=0.0, performance=0.0, quality=0.0, growth=0.0)
        assert h.composite == 0.0

    def test_composite_weighted(self):
        h = HealthScore(availability=1.0, performance=0.0, quality=0.0, growth=0.0)
        expected = HEALTH_WEIGHT_AVAILABILITY * 100
        assert h.composite == round(expected, 1)

    def test_composite_mixed(self):
        h = HealthScore(availability=0.8, performance=0.6, quality=0.7, growth=0.5)
        expected = (
            0.8 * HEALTH_WEIGHT_AVAILABILITY
            + 0.6 * HEALTH_WEIGHT_PERFORMANCE
            + 0.7 * HEALTH_WEIGHT_QUALITY
            + 0.5 * HEALTH_WEIGHT_GROWTH
        ) * 100
        assert h.composite == round(expected, 1)

    def test_weights_sum_to_one(self):
        total = (
            HEALTH_WEIGHT_AVAILABILITY
            + HEALTH_WEIGHT_PERFORMANCE
            + HEALTH_WEIGHT_QUALITY
            + HEALTH_WEIGHT_GROWTH
        )
        assert abs(total - 1.0) < 1e-9

    def test_summary_format(self):
        h = HealthScore(
            availability=0.9,
            performance=0.7,
            quality=0.8,
            growth=0.6,
            details={
                "availability": "4/4 services",
                "performance": "baseline",
                "quality": "ok",
                "growth": "10/day",
            },
        )
        s = h.summary()
        assert "Health:" in s
        assert "Availability:" in s
        assert "Performance:" in s
        assert "Quality:" in s
        assert "Growth:" in s
        assert "4/4 services" in s

    def test_details_default_empty(self):
        h = HealthScore()
        assert h.details == {}


class TestCheckAvailability:
    @patch("legion_koi.observability.checkers.subprocess.run")
    def test_all_services_active(self, mock_run):
        """When all systemd services are active, availability should be high."""
        mock_run.return_value = MagicMock(stdout="active\n")

        mock_storage = MagicMock()
        mock_storage._get_conn.return_value.execute.return_value = None

        mock_bus = MagicMock()
        mock_bus.ping.return_value = True
        mock_bus.pending_count.return_value = 0

        from legion_koi.observability.checkers import check_availability

        score, detail = check_availability(storage=mock_storage, event_bus=mock_bus)
        # 4 services + PG + Redis = 6 checks, all pass
        assert score == 1.0
        assert "6/6" in detail

    def test_no_storage_no_bus(self):
        """With no storage or bus, only systemd checks run."""
        from legion_koi.observability.checkers import check_availability

        with patch("legion_koi.observability.checkers.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="inactive\n")
            score, detail = check_availability(storage=None, event_bus=None)
            # 4 services (all inactive) + PG (not configured) + Redis (unreachable) = 0/6
            assert score == 0.0


class TestCheckQuality:
    def test_no_storage(self):
        from legion_koi.observability.checkers import check_quality

        score, detail = check_quality(storage=None)
        assert score == 0.0
        assert "no storage" in detail

    def test_with_mock_storage(self):
        from legion_koi.observability.checkers import check_quality

        mock_storage = MagicMock()
        mock_storage.get_entity_stats.return_value = {
            "total_entities": 500,
            "by_type": {"Person": 100, "Tool": 80, "Concept": 70, "Organization": 50,
                        "Event": 40, "Location": 30, "Date": 20, "URL": 15, "Path": 10, "Version": 5},
            "extraction_coverage": {"bundles_with_entities": 800, "total_bundles": 1000},
        }
        mock_storage.get_config_stats.return_value = [
            {"config_id": "test", "embedded": 700, "total": 1000},
        ]

        score, detail = check_quality(storage=mock_storage)
        assert 0.0 < score <= 1.0
        assert "entity_types=10" in detail


class TestCheckGrowth:
    def test_no_storage(self):
        from legion_koi.observability.checkers import check_growth

        score, detail = check_growth(storage=None)
        assert score == 0.0

    def test_healthy_growth(self):
        from legion_koi.observability.checkers import check_growth

        mock_storage = MagicMock()
        mock_conn = MagicMock()
        mock_storage._get_conn.return_value = mock_conn

        # 100 bundles in 7 days = ~14/day
        mock_conn.execute.return_value.fetchone.side_effect = [
            {"cnt": 100},  # bundles
            {"cnt": 50},   # entities
        ]

        score, detail = check_growth(storage=mock_storage)
        assert score == 1.0
        assert "bundles/day" in detail


class TestComputeHealth:
    @patch("legion_koi.observability.checkers.check_availability", return_value=(0.9, "ok"))
    @patch("legion_koi.observability.checkers.check_performance", return_value=(0.8, "ok"))
    @patch("legion_koi.observability.checkers.check_quality", return_value=(0.7, "ok"))
    @patch("legion_koi.observability.checkers.check_growth", return_value=(0.6, "ok"))
    def test_compute_health_combines_checkers(self, *mocks):
        health = compute_health()
        assert health.availability == 0.9
        assert health.performance == 0.8
        assert health.quality == 0.7
        assert health.growth == 0.6
        assert health.composite > 0
