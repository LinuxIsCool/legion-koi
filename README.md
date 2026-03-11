# legion-koi

Sovereign KOI-net node for Legion's knowledge federation. Built on [BlockScience's koi-net/rid-lib](https://github.com/BlockScience/koi-net) protocol.

## What it does

legion-koi watches Legion's data sources (filesystem, SQLite, PostgreSQL) and federates them as RID-identified knowledge bundles. Each data source gets a sensor. All bundles flow through PostgreSQL with full-text search (tsvector) and vector embeddings (pgvector) for semantic retrieval. An MCP server exposes 8 search/browse tools to Claude Code sessions.

**Current state**: 67,751 bundles across 6 namespaces, 296 transcripts (3.7M words), 62K+ Telegram messages. Keyword search, semantic search, and hybrid search (RRF fusion) all operational.

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
│  │ • session  (db) │                  │        Embedding)       │  │
│  │ • message  (db) │                  └──────────┬──────────────┘  │
│  └────────────────┘                              │                 │
│                                         ┌────────▼──────────┐     │
│  ┌──────────────────────┐              │   PostgreSQL        │     │
│  │ KOI-net Protocol API  │              │   personal_koi     │     │
│  │ :8100/koi-net/        │              │                    │     │
│  │ • rids/fetch          │              │ bundles (67K rows)  │     │
│  │ • bundles/fetch       │              │ embeddings (vec1024)│     │
│  │ • manifests/fetch     │              │ tsvector + HNSW     │     │
│  │ • events/broadcast    │              └────────┬───────────┘     │
│  │ • events/poll         │                       │                 │
│  └──────────────────────┘              ┌─────────▼──────────┐     │
│                                        │ MCP Server (stdio)  │     │
│                                        │ 8 tools             │     │
│                                        └────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
         ▲                                          ▲
         │ (future: federation)                     │
    ┌────┴────┐                         ┌───────────┴──────────────┐
    │ Other   │                         │ Data Sources              │
    │ KOI     │                         │ • ~/legion-brain/journal/ │
    │ nodes   │                         │ • ~/legion-brain/ventures │
    │         │                         │ • ~/.claude/local/ (DBs)  │
    └─────────┘                         └──────────────────────────┘
```

## Data

| Namespace | Bundles | Source | Sensor Type |
|-----------|---------|--------|-------------|
| `legion.claude-message` | 62,972 | Telegram via SQLite | DB poll |
| `legion.claude-recording` | 4,720 | Otter.ai transcripts via SQLite | DB poll |
| `legion.claude-journal` | 47 | Markdown files (frontmatter + body) | Filesystem watch |
| `legion.claude-session` | 6 | Claude Code logging JSONL | DB poll |
| `legion.claude-venture` | 5 | Markdown files (frontmatter + body) | Filesystem watch |
| `koi-net.node` | 1 | Node identity | Internal |

## RID Types

| Type | Namespace | Reference Format | Example |
|------|-----------|-----------------|---------|
| Journal | `legion.claude-journal` | `YYYY-MM-DD/slug` | `orn:legion.claude-journal:2026-03-10/1225-the-plugin-koi-insight` |
| Venture | `legion.claude-venture` | `stage/id` | `orn:legion.claude-venture:active/oral-history-ontology` |
| Recording | `legion.claude-recording` | `source/identifier` | `orn:legion.claude-recording:otter/2026-03-10-standup` |
| Session | `legion.claude-session` | `session-id` | `orn:legion.claude-session:abc123-def456` |
| Message | `legion.claude-message` | `platform:type:ids` | `orn:legion.claude-message:telegram:msg:-1003756725881:2408` |

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

# Dry run (counts only)
uv run python scripts/backfill_embeddings.py --dry-run
```

## Development Phases

### Phase 1 — Foundation (complete)
KOI-net node, journal sensor, 5-endpoint federation protocol, filesystem watching, `.rid_cache/` bundle store.

### Phase 2a — Sensor Expansion (complete)
Venture, recording, session sensors. 4 custom ORN types.

### Phase 2b — PostgreSQL + FTS (complete)
`personal_koi` database, tsvector full-text search, GIN indexes, bundle upsert pipeline.

### Phase 2c — Messages + MCP (complete)
Message sensor (62K Telegram), 6 MCP tools, direct bulk ingestion scripts for recordings and messages.

### Phase 3 — Embeddings + Semantic Search (complete)
Provider-abstracted embeddings (Telus/Ollama), pgvector with HNSW index, semantic search, hybrid search via RRF, inline embedding in handler, bulk backfill script.

### Phase 4 — Claude Web Sensor (planned)
Ingest 1,567 Claude Web conversations + 65 projects + memories. New namespaces: `legion.claude-conversation`, `legion.claude-project`. Bulk ingestion then live export pipeline.

### Phase 5 — Complete Sensor Surface (planned)
Native Claude Code transcript sensor (76 JSONL files), GitHub sensor (repos, issues, PRs, commits), backlog sensor, fix logging sensor. Differentiate `legion.claude-transcript` (full conversation history) from `legion.claude-session` (event metadata). Reflection checkpoint: full sensor census and search quality evaluation.

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
