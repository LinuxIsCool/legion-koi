# legion-koi

Sovereign KOI-net node for Legion's knowledge federation. Built on [BlockScience's koi-net/rid-lib](https://github.com/BlockScience/koi-net) protocol.

## What it does

legion-koi watches Legion's data directories for changes and federates them as RID-identified knowledge bundles over the KOI-net protocol. Each Claude plugin data directory becomes a sensor source.

**Phase 1** (current): Journal sensor watches `~/legion-brain/local/journal/` and serves entries via the 5-endpoint KOI-net protocol.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     legion-koi (FullNode)                    │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐  │
│  │ Sensors   │───▶│ KobjQueue │───▶│ Knowledge Pipeline   │  │
│  │           │    │          │    │                      │  │
│  │ • journal │    └──────────┘    │ RID → Manifest →     │  │
│  │ • venture │                    │ Bundle → Network →   │  │
│  │ • record  │                    │ Final                │  │
│  │ • session │                    └──────────┬───────────┘  │
│  └──────────┘                               │              │
│                                    ┌────────▼─────────┐    │
│  ┌──────────────────────┐         │   .rid_cache/     │    │
│  │ KOI-net Protocol API  │         │   (Bundle store)  │    │
│  │ :8100/koi-net/        │◀────────┤                   │    │
│  │ • rids/fetch          │         └───────────────────┘    │
│  │ • bundles/fetch       │                                  │
│  │ • manifests/fetch     │                                  │
│  │ • events/broadcast    │                                  │
│  │ • events/poll         │                                  │
│  └──────────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
         ▲                                        ▲
         │ (Phase 3: federation)                  │
    ┌────┴────┐                          ┌────────┴───────┐
    │ Other   │                          │ ~/legion-brain/ │
    │ KOI     │                          │ /local/journal/ │
    │ nodes   │                          │ /local/ventures │
    └─────────┘                          └────────────────┘
```

## RID Types

| Type | Namespace | Reference Format | Example |
|------|-----------|-----------------|---------|
| Journal | `legion.claude-journal` | `YYYY-MM-DD/slug` | `orn:legion.claude-journal:2026-03-10/1225-the-plugin-koi-insight` |
| Venture | `legion.claude-venture` | `stage/id` | `orn:legion.claude-venture:active/oral-history-ontology` |
| Recording | `legion.claude-recording` | `source/identifier` | `orn:legion.claude-recording:otter/2026-03-10-standup` |
| Session | `legion.claude-session` | `session-id` | `orn:legion.claude-session:abc123-def456` |

## Setup

```bash
cd ~/legion-koi
uv pip install -e .
echo 'PRIV_KEY_PASSWORD=your-password-here' > .env
```

## Run

```bash
# Directly
uv run python -m legion_koi

# As systemd service
systemctl --user start legion-koi
systemctl --user status legion-koi
journalctl --user -u legion-koi -f
```

## Test

```bash
# With node running:
uv run python test_node.py
```

## How it works

### Sensors

Sensors watch filesystem directories using [watchdog](https://github.com/gorakhargosh/watchdog). When a file is created, modified, or moved (atomic writes), the sensor:

1. Parses the file (YAML frontmatter + markdown body for journals)
2. Computes a SHA256 hash of the content
3. Checks state — is this NEW or an UPDATE?
4. Creates an RID (Resource Identifier) and a Bundle (RID + manifest + contents)
5. Pushes the Bundle to the KobjQueue

On startup, `scan_all()` walks the entire directory tree to seed the cache with existing entries.

### Knowledge Pipeline

The KobjQueue feeds into koi-net's 5-stage Knowledge Pipeline:

1. **RID handlers** — filter/validate the RID itself
2. **Manifest handlers** — `BasicManifestHandler` compares SHA256 hashes for deduplication (unchanged content → STOP_CHAIN)
3. **Bundle handlers** — `JournalBundleHandler` validates frontmatter, sets NEW/UPDATE event type
4. **Network handlers** — `SuppressNetworkHandler` (Phase 1: no federation yet)
5. **Final handlers** — `LoggingFinalHandler` logs all processed objects

After the Bundle stage, the pipeline writes to `.rid_cache/` (content-addressable JSON files).

### Protocol Endpoints

All requests require signed envelopes (ECDSA). The node generates a private key on first run (`priv_key.pem`).

| Endpoint | Purpose |
|----------|---------|
| `POST /koi-net/rids/fetch` | List cached RIDs, optionally filtered by type |
| `POST /koi-net/bundles/fetch` | Fetch full bundles by RID |
| `POST /koi-net/manifests/fetch` | Fetch manifests (RID + hash + timestamp) |
| `POST /koi-net/events/broadcast` | Receive events from other nodes |
| `POST /koi-net/events/poll` | Partial nodes pull events |

## Roadmap

### Phase 2: More sensors + storage
- Venture sensor (`~/legion-brain/local/ventures/`)
- Recording sensor (`~/.claude/local/recordings/`)
- Session sensor (logging JSONL files)
- PostgreSQL + pgvector for embedding storage
- MCP server for Claude sessions to search knowledge

### Phase 3: Federation + processing
- Connect to external KOI coordinator nodes
- Processor node for entity extraction (OHO ontology)
- Embedding generation via TELUS AI APIs

### Phase 4: Actuator nodes
- Nodes that produce outputs (notifications, summaries, dashboards)
- Cross-machine knowledge sync via Tailscale mesh

## Dependencies

- [koi-net](https://github.com/BlockScience/koi-net) ~1.2 — KOI-net protocol library
- [rid-lib](https://github.com/BlockScience/rid-lib) >=3.2.7 — Resource Identifier primitives
- FastAPI + uvicorn — protocol server (provided by koi-net)
- watchdog — filesystem monitoring
- structlog — structured logging
