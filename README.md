# legion-koi

Sovereign KOI-net node for Legion's knowledge federation. Built on [BlockScience's koi-net/rid-lib](https://github.com/BlockScience/koi-net) protocol.

## What it does

legion-koi watches Legion's data sources (filesystem, SQLite, PostgreSQL) and federates them as RID-identified knowledge bundles. Each data source gets a sensor. All bundles flow through PostgreSQL with full-text search (tsvector) and vector embeddings (pgvector) for semantic retrieval. An MCP server exposes 8 search/browse tools to Claude Code sessions.

**Current state**: 69,676 bundles across 11 namespaces (8 sensors + node identity). Keyword search, semantic search, and hybrid search (RRF fusion) all operational.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        legion-koi (FullNode)                         │
│                                                                     │
│  ┌────────────────┐   ┌──────────┐   ┌──────────────────────────┐  │
│  │ Sensors         │──▶│ KobjQueue │──▶│ Knowledge Pipeline       │  │
│  │                 │   │          │   │                          │  │
│  │ • journal  (fs) │   └──────────┘   │ RID → Manifest →        │  │
│  │ • venture  (fs) │                  │ Bundle → Network →      │  │
│  │ • recording (db)│                  │ Final (PostgreSQL +     │  │
│  │ • logging  (db) │                  │        Embedding)       │  │
│  │ • message  (db) │                  └──────────┬──────────────┘  │
│  └────────────────┘                              │                 │
│                                                  │                 │
│  ┌────────────────┐                     ┌────────▼──────────┐     │
│  │ Bulk Scripts    │────────────────────▶│   PostgreSQL        │     │
│  │ • claude-web    │                    │   personal_koi     │     │
│  │ • claude-code   │                    │                    │     │
│  │ • github        │                    │ bundles (69K rows)  │     │
│  └────────────────┘                    │ embeddings (vec1024)│     │
│                                         │ tsvector + HNSW     │     │
│  ┌──────────────────────┐              └────────┬───────────┘     │
│  │ KOI-net Protocol API  │                       │                 │
│  │ :8100/koi-net/        │              ┌─────────▼──────────┐     │
│  │ • rids/fetch          │              │ MCP Server (stdio)  │     │
│  │ • bundles/fetch       │              │ 8 tools             │     │
│  │ • manifests/fetch     │              └────────────────────┘     │
│  │ • events/broadcast    │                                         │
│  │ • events/poll         │                                         │
│  └──────────────────────┘                                         │
└─────────────────────────────────────────────────────────────────────┘
         ▲                                          ▲
         │ (future: federation)                     │
    ┌────┴────┐                         ┌───────────┴──────────────┐
    │ Other   │                         │ Data Sources              │
    │ KOI     │                         │ • ~/legion-brain/journal/ │
    │ nodes   │                         │ • ~/legion-brain/ventures │
    │         │                         │ • ~/.claude/local/ (DBs)  │
    └─────────┘                         │ • Claude Web export       │
                                        │ • Claude Code transcripts │
                                        │ • GitHub repos (gh CLI)   │
                                        └──────────────────────────┘
```

## Data

| Namespace | Bundles | Source | Ingestion |
|-----------|---------|--------|-----------|
| `legion.claude-message` | 62,995 | Telegram via SQLite | Sensor (DB poll) |
| `legion.claude-recording` | 4,720 | Otter.ai transcripts via SQLite | Sensor (DB poll) |
| `legion.claude-web.conversation` | 1,534 | Claude Web export (`conversations.json`) | Bulk script |
| `legion.claude-github` | 234 | GitHub repos via `gh` CLI | Bulk script |
| `legion.claude-web.project` | 65 | Claude Web export (`projects.json`) | Bulk script |
| `legion.claude-code` | 63 | Claude Code session transcripts (`.jsonl`) | Bulk script |
| `legion.claude-journal` | 52 | Markdown files (frontmatter + body) | Sensor (fs watch) |
| `legion.claude-logging` | 6 | Claude Code logging plugin SQLite | Sensor (DB poll) |
| `legion.claude-venture` | 5 | Markdown files (frontmatter + body) | Sensor (fs watch) |
| `legion.claude-web.memory` | 1 | Claude Web export (`memories.json`) | Bulk script |
| `koi-net.node` | 1 | Node identity | Internal |

## RID Types

| Type | Namespace | Reference Format | Example |
|------|-----------|-----------------|---------|
| Journal | `legion.claude-journal` | `YYYY-MM-DD/slug` | `orn:legion.claude-journal:2026-03-10/1225-the-plugin-koi-insight` |
| Venture | `legion.claude-venture` | `stage/id` | `orn:legion.claude-venture:active/oral-history-ontology` |
| Recording | `legion.claude-recording` | `source/identifier` | `orn:legion.claude-recording:otter/2026-03-10-standup` |
| Logging | `legion.claude-logging` | `session-id` | `orn:legion.claude-logging:abc123-def456` |
| Message | `legion.claude-message` | `platform:type:ids` | `orn:legion.claude-message:telegram:msg:-1003756725881:2408` |
| Conversation | `legion.claude-web.conversation` | `date/uuid-slug` | `orn:legion.claude-web.conversation:2025-10-14/daa326b1-regen-meeting` |
| Project | `legion.claude-web.project` | `uuid-slug` | `orn:legion.claude-web.project:a1b2c3d4-my-project` |
| Memory | `legion.claude-web.memory` | `export-id` | `orn:legion.claude-web.memory:claude-web-export` |
| Code | `legion.claude-code` | `date/session-prefix` | `orn:legion.claude-code:2026-03-10/fb92b996` |
| GitHub | `legion.claude-github` | `owner/repo` | `orn:legion.claude-github:LinuxIsCool/legion-koi` |

## MCP Tools

| Tool | Type | Purpose |
|------|------|---------|
| `search_bundles` | FTS | Keyword search (websearch syntax) |
| `semantic_search` | Vector | Find by meaning (cosine similarity) |
| `hybrid_search` | FTS+Vector | Combined via Reciprocal Rank Fusion (recommended) |
| `get_bundle` | Lookup | Fetch full bundle by RID |
| `read_transcript` | Lookup | Full transcript text with metadata header |
| `get_thread` | Lookup | Message thread by ID, chronological |
| `list_bundles` | Browse | List RIDs, filterable by namespace |
| `koi_stats` | Stats | Bundle counts + embedding coverage per namespace |

## Search

Three search modes:

- **Keyword** (`search_bundles`): PostgreSQL `websearch_to_tsquery` — supports AND, OR, quotes, exclusion. Fast, exact. Misses semantic matches.
- **Semantic** (`semantic_search`): pgvector cosine similarity via HNSW index. Finds meaning even when words don't overlap. Uses TELUS AI E5 embeddings (1024-dim, nvidia/nv-embedqa-e5-v5).
- **Hybrid** (`hybrid_search`): Reciprocal Rank Fusion merges keyword and semantic results. Best overall quality. Falls back to keyword-only if embedding API unavailable.

Embeddings use a provider abstraction (Telus or Ollama) with auto-detection from environment.

## Setup

```bash
cd ~/legion-koi
uv pip install -e .
echo 'PRIV_KEY_PASSWORD=your-password-here' > .env
```

Embedding credentials: `~/.claude/local/secrets/telus-api.env` (auto-loaded).

## Run

```bash
# Directly
uv run python -m legion_koi

# As systemd service
systemctl --user start legion-koi
systemctl --user status legion-koi
journalctl --user -u legion-koi -f

# MCP server (launched by Claude Code plugin)
uv run python -m legion_koi.mcp_server
```

## Backfill

```bash
# Bulk embed all bundles (one-time)
uv run python scripts/backfill_embeddings.py

# With explicit provider
uv run python scripts/backfill_embeddings.py --provider ollama --model mxbai-embed-large

# Multi-config backfill
uv run python scripts/backfill_config.py --config telus-e5-1024

# Dry run (counts only)
uv run python scripts/backfill_embeddings.py --dry-run
```

## Ingestion Scripts

```bash
# Claude Web export (conversations, projects, memories)
uv run python scripts/ingest_claude_web.py

# Claude Code session transcripts
uv run python scripts/ingest_transcripts.py

# GitHub repos
uv run python scripts/ingest_github.py

# Namespace migration (one-time, already run)
uv run python scripts/migrate_namespaces.py
```

## Development Phases

### Phase 1 — Foundation (complete)
KOI-net node, journal sensor, 5-endpoint federation protocol, filesystem watching, `.rid_cache/` bundle store.

### Phase 2a — Sensor Expansion (complete)
Venture, recording, logging sensors. 4 custom ORN types.

### Phase 2b — PostgreSQL + FTS (complete)
`personal_koi` database, tsvector full-text search, GIN indexes, bundle upsert pipeline.

### Phase 2c — Messages + MCP (complete)
Message sensor (62K Telegram), 6 MCP tools, direct bulk ingestion scripts for recordings and messages.

### Phase 3 — Embeddings + Semantic Search (complete)
Provider-abstracted embeddings (Telus/Ollama), pgvector with HNSW index, semantic search, hybrid search via RRF, inline embedding in handler, bulk backfill script. Multi-config embedding support.

### Phase 4 — Claude Web Ingestion (complete)
1,534 conversations + 65 projects + 1 memory bundle from Claude Web export. Namespace: `legion.claude-web.*` (3 sub-namespaces).

### Phase 5 — Complete Sensor Surface (complete)
- **5a**: Claude Code transcript ingestion (63 sessions). Namespace: `legion.claude-code`.
- **5b**: GitHub repo ingestion (234 repos via `gh` CLI). Namespace: `legion.claude-github`.
- Namespace consolidation: renamed 5 namespaces to align with sensor boundaries (claude-web.*, claude-logging, claude-code).

### Phase 6 — Multi-Chunk Embeddings + Reranking (planned)
Recursive chunking (512 tokens, 10-20% overlap), `embedding_chunks` table, parent-child retrieval, cross-encoder reranking. Re-backfill all bundles.

### Phase 7 — Entity Extraction + Cross-Bundle Linking (planned)
LLM-based entity extraction (people, orgs, topics), entity resolution/clustering, `entities` and `bundle_entities` tables, cross-bundle edges via shared entities, hippo integration. Reflection checkpoint: entity graph assessment and retrieval evaluation.

### Phase 8 — Interfaces + Federation (planned)
Force-directed graph visualization (Cosmograph/3d-force-graph), timeline dashboard, KOI-net federation with external nodes, cross-machine sync (Tailscale mesh), actuator nodes (summaries, notifications, context injection). Full retrospective.

## Dependencies

- [koi-net](https://github.com/BlockScience/koi-net) ~1.2 — KOI-net protocol
- [rid-lib](https://github.com/BlockScience/rid-lib) >=3.2.7 — Resource Identifiers
- FastAPI + uvicorn — protocol server (via koi-net)
- psycopg 3.2+ — PostgreSQL driver
- pgvector 0.8.2 — vector similarity search
- httpx — embedding API client
- watchdog — filesystem monitoring
- structlog — structured logging
- mcp — Model Context Protocol server
