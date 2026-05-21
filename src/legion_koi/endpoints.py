"""GET /koi-net/{namespaces,bundles,search,neighborhood} endpoints for legion-koi.

Phase 2.5 — unblock the 4 dormant claude-koi MCP tools (commit a4fc0b1 of
claude-koi Phase 2):

  GET /koi-net/namespaces                            → per-namespace bundle counts
  GET /koi-net/bundles/{rid}                         → fetch one bundle by RID
  GET /koi-net/search?q=...&namespace=...&limit=20   → FTS search
  GET /koi-net/bundles/{rid}/neighborhood?depth=N    → approximate graph neighborhood

Response shapes match what claude_koi.koi_client.KoiClient expects (see plugin
`claude-koi/claude_koi/koi_client.py`):

  /namespaces       → {"namespaces": [{"namespace": str, "count": int}, ...]}
  /bundles/{rid}    → {"manifest": {...}, "contents": {...}}  OR  404
  /search           → {"results": [{"rid", "manifest", "contents", ...}, ...]}
  /neighborhood     → {"bundles": [{...}], "depth": int, "approximation": str}

Read-only. No schema migration. All DB work is offloaded via asyncio.to_thread.

Neighborhood approximation:
  Phase 2.5 uses entity-co-occurrence when available (bundles sharing ≥1
  extracted entity from `bundle_entities`) AND/OR namespace+temporal proximity
  as a conservative fallback. Phase 5 ships sheaf-theoretic graph traversal.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request


# ──────────────────────────────────────────────────────────────────────────
# Bundle shaping
# ──────────────────────────────────────────────────────────────────────────


def _bundle_to_payload(row: dict) -> dict:
    """Shape a `bundles` table row into the {rid, manifest, contents} envelope
    callers expect. Mirrors the koi-net bundle layout used by personal-koi.

    The top-level ``rid`` is included alongside ``manifest.rid`` so consumers
    can read it without digging into the manifest. All endpoints
    (``/bundles/{rid}``, ``/search``, ``/neighborhood``) return this shape.
    """
    created = row.get("created_at")
    updated = row.get("updated_at")
    if isinstance(created, datetime):
        created = created.isoformat()
    if isinstance(updated, datetime):
        updated = updated.isoformat()
    rid = row.get("rid")
    manifest = {
        "rid": rid,
        "namespace": row.get("namespace"),
        "reference": row.get("reference"),
        "sha256_hash": row.get("sha256_hash"),
        "created_at": created,
        "updated_at": updated,
    }
    contents = row.get("contents") or {}
    return {"rid": rid, "manifest": manifest, "contents": contents}


# ──────────────────────────────────────────────────────────────────────────
# Sync DB workers (each runs under asyncio.to_thread)
# ──────────────────────────────────────────────────────────────────────────


def _fetch_namespaces_sync(storage) -> list[dict]:
    """Returns [{namespace, count}, ...] ordered by count DESC."""
    stats = storage.get_stats()  # {namespace: count}
    rows = [{"namespace": ns, "count": int(cnt)} for ns, cnt in stats.items()]
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


def _fetch_bundle_sync(storage, rid: str) -> dict | None:
    row = storage.get_bundle(rid)
    if not row:
        return None
    return _bundle_to_payload(row)


def _search_sync(storage, query: str, namespace: str | None, limit: int) -> list[dict]:
    rows = storage.search_text(query=query, namespace=namespace, limit=limit)
    return [_bundle_to_payload(r) for r in rows]


def _neighborhood_sync(storage, rid: str, depth: int) -> dict | None:
    """Approximate 1-hop neighborhood.

    Strategy:
      depth == 0          → just the seed.
      depth >= 1 and seed has extracted entities → bundles sharing ≥1 entity.
      Fallback (no entities) → same namespace + created_at within ±1h, cap 20.
    Returns None if the seed RID does not exist.
    """
    seed = storage.get_bundle(rid)
    if not seed:
        return None
    seed_payload = _bundle_to_payload(seed)

    if depth <= 0:
        return {
            "bundles": [seed_payload],
            "depth": 0,
            "approximation": "seed-only",
        }

    bundles: list[dict] = [seed_payload]
    seen_rids: set[str] = {rid}
    approximation = "namespace+temporal"

    # Try entity-co-occurrence first if extraction has run on the seed.
    try:
        ents = storage.get_bundle_entities(rid)
    except Exception:
        ents = []

    if ents:
        # Walk entities → bundles via storage API. Cap to 20 neighbors.
        approximation = "entity-cooccurrence"
        entity_ids = [int(e["entity_id"]) for e in ents]
        rows = storage.get_entity_neighbors(
            entity_ids=entity_ids,
            exclude_rid=rid,
            limit=20,
        )
        for r in rows:
            r_rid = r.get("rid")
            if not r_rid or r_rid in seen_rids:
                continue
            seen_rids.add(r_rid)
            bundles.append(_bundle_to_payload(r))

    if len(bundles) == 1:
        # Fallback: namespace + ±1h temporal proximity.
        approximation = "namespace+temporal"
        ns = seed.get("namespace")
        created = seed.get("created_at")
        if ns and isinstance(created, datetime):
            window_start = created - timedelta(hours=1)
            window_end = created + timedelta(hours=1)
            rows = storage.get_namespace_temporal_neighbors(
                namespace=ns,
                window_start=window_start,
                window_end=window_end,
                exclude_rid=rid,
                limit=20,
            )
            for r in rows:
                r_rid = r.get("rid")
                if not r_rid or r_rid in seen_rids:
                    continue
                seen_rids.add(r_rid)
                bundles.append(_bundle_to_payload(r))

    return {
        "bundles": bundles,
        "depth": depth,
        "approximation": approximation,
    }


# ──────────────────────────────────────────────────────────────────────────
# Router factory
# ──────────────────────────────────────────────────────────────────────────


def make_endpoints_router() -> APIRouter:
    """Builds the read-only endpoints router.

    The handler resolves storage from ``request.app.state.storage`` at call time
    (matches how health.py resolves ``postgres_dsn``). If storage is not set
    on app.state, returns 503.
    """
    router = APIRouter(prefix="/koi-net")

    def _get_storage(request: Request):
        storage = getattr(request.app.state, "storage", None)
        if storage is None:
            raise HTTPException(status_code=503, detail="storage_not_initialized")
        return storage

    @router.get("/namespaces")
    async def namespaces(request: Request):
        storage = _get_storage(request)
        rows = await asyncio.to_thread(_fetch_namespaces_sync, storage)
        return {"namespaces": rows}

    @router.get("/bundles/{rid:path}/neighborhood")
    async def neighborhood(
        request: Request,
        rid: str,
        depth: int = Query(default=1, ge=0, le=4),
    ):
        storage = _get_storage(request)
        # Strip a trailing slash that FastAPI's path converter may keep.
        rid = rid.rstrip("/")
        result = await asyncio.to_thread(_neighborhood_sync, storage, rid, depth)
        if result is None:
            raise HTTPException(status_code=404, detail="bundle_not_found")
        return result

    @router.get("/bundles/{rid:path}")
    async def bundle(request: Request, rid: str):
        storage = _get_storage(request)
        rid = rid.rstrip("/")
        result = await asyncio.to_thread(_fetch_bundle_sync, storage, rid)
        if result is None:
            raise HTTPException(status_code=404, detail="bundle_not_found")
        return result

    @router.get("/search")
    async def search(
        request: Request,
        q: str = Query(default=""),
        namespace: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=200),
    ):
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="missing_or_empty_query")
        storage = _get_storage(request)
        rows = await asyncio.to_thread(_search_sync, storage, q, namespace, limit)
        return {"results": rows, "query": q, "namespace": namespace, "limit": limit}

    return router


__all__ = ["make_endpoints_router"]
