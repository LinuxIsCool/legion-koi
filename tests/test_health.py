"""legion-koi /koi-net/health endpoint."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from legion_koi.health import make_health_router


@pytest.fixture
def client():
    app = FastAPI()
    class Stub:
        node_rid = "orn:koi-net.node:legion-koi+abc"
        sensor_count = 17
        bundle_count = 805_000
        last_bundle_at = "2026-05-17T13:00:00Z"
        build_sha = "deadbeef"
    app.state.health = Stub()
    app.include_router(make_health_router())
    return TestClient(app)


def test_health_returns_node_rid(client):
    resp = client.get("/koi-net/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_rid"] == "orn:koi-net.node:legion-koi+abc"


def test_health_returns_sensor_and_bundle_counts(client):
    resp = client.get("/koi-net/health")
    body = resp.json()
    assert body["sensor_count"] == 17
    assert body["bundle_count"] == 805_000


def test_health_returns_build_sha(client):
    resp = client.get("/koi-net/health")
    assert resp.json()["build_sha"] == "deadbeef"


def test_health_returns_last_bundle_at(client):
    resp = client.get("/koi-net/health")
    assert resp.json()["last_bundle_at"] == "2026-05-17T13:00:00Z"
