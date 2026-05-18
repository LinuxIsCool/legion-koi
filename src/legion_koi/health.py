"""GET /koi-net/health endpoint for legion-koi.

Provides liveness + state snapshot. Mirrors personal-koi's health shape.
"""

from fastapi import APIRouter, Request


def make_health_router() -> APIRouter:
    router = APIRouter(prefix="/koi-net")

    @router.get("/health")
    async def health(request: Request):
        state = request.app.state.health
        return {
            "node_rid": state.node_rid,
            "sensor_count": state.sensor_count,
            "bundle_count": state.bundle_count,
            "last_bundle_at": state.last_bundle_at,
            "build_sha": state.build_sha,
        }

    return router
