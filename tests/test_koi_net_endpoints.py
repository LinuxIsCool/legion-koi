"""Tests for the 4 new /koi-net/ read endpoints — Phase 2.5 unblocking
claude-koi MCP tools.

Uses FastAPI's TestClient against a real FastAPI app plus an in-memory
StorageStub that mimics the PostgresStorage interface. No DB required.
"""
from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from legion_koi.endpoints import make_endpoints_router


# ──────────────────────────────────────────────────────────────────────────
# Storage stub
# ──────────────────────────────────────────────────────────────────────────


class StorageStub:
    """Mimics enough of PostgresStorage for the endpoints tests.

    Implements only the *public* methods that endpoints.py touches: get_stats,
    get_bundle, search_text, get_bundle_entities, get_entity_neighbors, and
    get_namespace_temporal_neighbors. No raw-SQL escape hatch needed.
    """

    def __init__(
        self,
        bundles: dict[str, dict],
        stats: dict[str, int],
        entities_by_rid: dict[str, list[dict]] | None = None,
        neighbors_by_entity: list[dict] | None = None,
        namespace_neighbors: list[dict] | None = None,
    ):
        self._bundles = bundles
        self._stats = stats
        self._entities = entities_by_rid or {}
        self._neighbors_by_entity = neighbors_by_entity or []
        self._namespace_neighbors = namespace_neighbors or []

    def get_stats(self) -> dict:
        return dict(self._stats)

    def get_bundle(self, rid: str) -> dict | None:
        return self._bundles.get(rid)

    def search_text(self, query: str, namespace=None, limit: int = 20) -> list[dict]:
        results = []
        for rid, b in self._bundles.items():
            if namespace and b.get("namespace") != namespace:
                continue
            text = (b.get("search_text") or "") + " " + str(b.get("contents", ""))
            if query.lower() in text.lower():
                results.append(b)
            if len(results) >= limit:
                break
        return results

    def get_bundle_entities(self, rid: str) -> list[dict]:
        return list(self._entities.get(rid, []))

    def get_entity_neighbors(
        self,
        entity_ids: list[int],
        exclude_rid: str,
        limit: int = 20,
    ) -> list[dict]:
        if not entity_ids:
            return []
        return [
            r for r in self._neighbors_by_entity if r.get("rid") != exclude_rid
        ][:limit]

    def get_namespace_temporal_neighbors(
        self,
        namespace: str,
        window_start,
        window_end,
        exclude_rid: str,
        limit: int = 20,
    ) -> list[dict]:
        return [
            r
            for r in self._namespace_neighbors
            if r.get("namespace") == namespace and r.get("rid") != exclude_rid
        ][:limit]


# ──────────────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────────────


def _bundle_row(rid: str, namespace: str, ts: datetime, contents: dict | None = None) -> dict:
    return {
        "rid": rid,
        "namespace": namespace,
        "reference": rid.split(":", 2)[-1] if ":" in rid else rid,
        "contents": contents or {"body": f"contents-of-{rid}"},
        "search_text": f"searchable text for {rid} about claude and bundles",
        "sha256_hash": "deadbeef",
        "created_at": ts,
        "updated_at": ts,
    }


@pytest.fixture
def now():
    return datetime(2026, 5, 21, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def known_bundle_rid():
    return "orn:legion.claude-journal:2026-05-21-test"


@pytest.fixture
def app(now, known_bundle_rid):
    journal_rid = known_bundle_rid
    msg_rid = "orn:legion.claude-message:msg-001"
    neighbor_rid = "orn:legion.claude-journal:2026-05-21-neighbor"

    bundles = {
        journal_rid: _bundle_row(journal_rid, "legion.claude-journal", now),
        msg_rid: _bundle_row(msg_rid, "legion.claude-message", now),
        neighbor_rid: _bundle_row(neighbor_rid, "legion.claude-journal", now + timedelta(minutes=30)),
    }
    stats = {
        "legion.claude-journal": 2,
        "legion.claude-message": 1,
    }
    # Seed has one extracted entity; the neighbor shares it.
    entities_by_rid = {
        journal_rid: [{"entity_id": 42, "name": "claude", "entity_type": "system"}],
    }
    neighbors_by_entity = [
        _bundle_row(neighbor_rid, "legion.claude-journal", now + timedelta(minutes=30)),
    ]
    namespace_neighbors = [
        _bundle_row(neighbor_rid, "legion.claude-journal", now + timedelta(minutes=30)),
    ]

    storage = StorageStub(
        bundles=bundles,
        stats=stats,
        entities_by_rid=entities_by_rid,
        neighbors_by_entity=neighbors_by_entity,
        namespace_neighbors=namespace_neighbors,
    )

    fastapi_app = FastAPI()
    fastapi_app.state.storage = storage
    fastapi_app.include_router(make_endpoints_router())
    return fastapi_app


@pytest.fixture
def http_client(app):
    return TestClient(app)


# ──────────────────────────────────────────────────────────────────────────
# /koi-net/namespaces
# ──────────────────────────────────────────────────────────────────────────


def test_namespaces_returns_list(http_client):
    resp = http_client.get("/koi-net/namespaces")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)
    assert "namespaces" in body
    assert isinstance(body["namespaces"], list)
    assert body["namespaces"], "expected at least one namespace in stub"
    first = body["namespaces"][0]
    assert "namespace" in first
    assert "count" in first
    assert isinstance(first["count"], int)


def test_namespaces_includes_claude_journal(http_client):
    resp = http_client.get("/koi-net/namespaces")
    namespaces = [n["namespace"] for n in resp.json()["namespaces"]]
    assert "legion.claude-journal" in namespaces


def test_namespaces_ordered_by_count_desc(http_client):
    resp = http_client.get("/koi-net/namespaces")
    counts = [n["count"] for n in resp.json()["namespaces"]]
    assert counts == sorted(counts, reverse=True)


def test_namespaces_returns_503_when_storage_missing():
    """If app.state.storage is unset, return 503 rather than crashing."""
    app = FastAPI()
    app.include_router(make_endpoints_router())
    resp = TestClient(app).get("/koi-net/namespaces")
    assert resp.status_code == 503


# ──────────────────────────────────────────────────────────────────────────
# /koi-net/bundles/{rid}
# ──────────────────────────────────────────────────────────────────────────


def test_bundles_get_returns_404_for_unknown_rid(http_client):
    encoded = urllib.parse.quote("orn:nonexistent:abc", safe="")
    resp = http_client.get(f"/koi-net/bundles/{encoded}")
    assert resp.status_code == 404


def test_bundles_get_returns_bundle_for_known_rid(http_client, known_bundle_rid):
    encoded = urllib.parse.quote(known_bundle_rid, safe="")
    resp = http_client.get(f"/koi-net/bundles/{encoded}")
    assert resp.status_code == 200
    body = resp.json()
    assert "manifest" in body
    assert "contents" in body
    assert body["manifest"]["rid"] == known_bundle_rid
    assert body["manifest"]["namespace"] == "legion.claude-journal"


# ──────────────────────────────────────────────────────────────────────────
# /koi-net/search
# ──────────────────────────────────────────────────────────────────────────


def test_search_with_query_returns_results(http_client):
    resp = http_client.get("/koi-net/search", params={"q": "claude", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert isinstance(body["results"], list)
    assert len(body["results"]) <= 5
    assert body["results"], "expected at least one match for 'claude' in stub"


def test_search_with_namespace_filter(http_client):
    resp = http_client.get(
        "/koi-net/search",
        params={"q": "claude", "namespace": "legion.claude-journal", "limit": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"]
    for result in body["results"]:
        rid = result.get("rid") or result.get("manifest", {}).get("rid", "")
        assert "legion.claude-journal" in rid


def test_search_empty_query_returns_400(http_client):
    resp = http_client.get("/koi-net/search", params={"q": ""})
    assert resp.status_code == 400


def test_search_whitespace_query_returns_400(http_client):
    resp = http_client.get("/koi-net/search", params={"q": "   "})
    assert resp.status_code == 400


# ──────────────────────────────────────────────────────────────────────────
# /koi-net/bundles/{rid}/neighborhood
# ──────────────────────────────────────────────────────────────────────────


def test_neighborhood_returns_related_bundles(http_client, known_bundle_rid):
    encoded = urllib.parse.quote(known_bundle_rid, safe="")
    resp = http_client.get(f"/koi-net/bundles/{encoded}/neighborhood?depth=1")
    assert resp.status_code == 200
    body = resp.json()
    assert "bundles" in body
    assert isinstance(body["bundles"], list)
    assert "depth" in body
    assert "approximation" in body
    # Must include the seed.
    rids = [b["manifest"]["rid"] for b in body["bundles"]]
    assert known_bundle_rid in rids


def test_neighborhood_depth_zero_returns_only_self(http_client, known_bundle_rid):
    encoded = urllib.parse.quote(known_bundle_rid, safe="")
    resp = http_client.get(f"/koi-net/bundles/{encoded}/neighborhood?depth=0")
    assert resp.status_code == 200
    body = resp.json()
    rids = [b["manifest"]["rid"] for b in body["bundles"]]
    assert known_bundle_rid in rids
    assert len(rids) == 1
    assert body["approximation"] == "seed-only"


def test_neighborhood_invalid_rid_returns_404(http_client):
    encoded = urllib.parse.quote("orn:nonexistent:abc", safe="")
    resp = http_client.get(f"/koi-net/bundles/{encoded}/neighborhood")
    assert resp.status_code == 404


def test_neighborhood_uses_entity_cooccurrence_when_available(
    http_client, known_bundle_rid
):
    encoded = urllib.parse.quote(known_bundle_rid, safe="")
    resp = http_client.get(f"/koi-net/bundles/{encoded}/neighborhood?depth=1")
    body = resp.json()
    # Stub seeds entities + entity-neighbor row → approximation should be the
    # entity-cooccurrence variant.
    assert body["approximation"] == "entity-cooccurrence"
    assert len(body["bundles"]) >= 2  # seed + ≥1 neighbor
