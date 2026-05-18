"""GET /koi-net/health endpoint for legion-koi.

Provides liveness + state snapshot. Mirrors personal-koi's health shape.

State source:
- ``app.state.health`` carries stub fields (node_rid, sensor_count, build_sha).
- ``app.state.postgres_dsn`` (optional) is queried live for bundle_count
  and last_bundle_at. Falls back to stub values on any error.

The handler stays in the FastAPI event loop; the synchronous psycopg call
is offloaded via ``asyncio.to_thread`` so the loop is not blocked.
"""

import asyncio

from fastapi import APIRouter, Request


def _fetch_bundle_stats_sync(dsn: str):
    """Open a fresh sync connection, return (count, last_iso) or None on error."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM bundles")
            count_row = cur.fetchone()
            count = count_row[0] if count_row else 0

            cur.execute("SELECT MAX(created_at) FROM bundles")
            last_row = cur.fetchone()
            last_dt = last_row[0] if last_row else None

    last_iso = last_dt.isoformat() if last_dt else None
    return count, last_iso


def make_health_router() -> APIRouter:
    router = APIRouter(prefix="/koi-net")

    @router.get("/health")
    async def health(request: Request):
        state = request.app.state.health
        bundle_count = state.bundle_count
        last = state.last_bundle_at

        dsn = getattr(request.app.state, "postgres_dsn", None)
        if dsn:
            try:
                bundle_count, last = await asyncio.to_thread(
                    _fetch_bundle_stats_sync, dsn
                )
            except Exception:
                # fall back to stub values; never fail the health endpoint
                pass

        return {
            "node_rid": state.node_rid,
            "sensor_count": state.sensor_count,
            "bundle_count": bundle_count,
            "last_bundle_at": last,
            "build_sha": state.build_sha,
        }

    return router
