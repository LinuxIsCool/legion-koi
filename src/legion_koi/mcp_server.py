"""MCP server exposing legion-koi knowledge graph to Claude Code."""

import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .storage.postgres import PostgresStorage
from .embeddings import create_embedder
from .constants import (
    RERANK_CANDIDATE_POOL, PREVIEW_SHORT, PREVIEW_MEDIUM,
    HIPPO_REDIS_HOST, HIPPO_REDIS_PORT, HIPPO_GRAPH_NAME, ENTITY_RRF_BONUS,
)
from .reranking import create_reranker, rerank_results

app = Server("legion-koi")

_storage: PostgresStorage | None = None
_config_embedders: dict[str, object] = {}
_hippo_circuit = None  # Lazy-initialized circuit breaker for FalkorDB


def _get_hippo_circuit():
    """Lazy-initialize the FalkorDB circuit breaker."""
    global _hippo_circuit
    if _hippo_circuit is None:
        from .resilience.circuit_breaker import CircuitBreaker
        _hippo_circuit = CircuitBreaker(name="hippo-falkordb")
    return _hippo_circuit


def _resolve_config(storage: PostgresStorage, config_id: str | None) -> str:
    """Resolve config_id — use explicit, or fall back to default config."""
    if config_id:
        return config_id
    default = storage.get_default_config()
    if default:
        return default["config_id"]
    configs = storage.list_embedding_configs()
    if configs:
        return configs[0]["config_id"]
    raise ValueError("No embedding configs registered")


def _embed_for_config(storage: PostgresStorage, config_id: str, query: str) -> list[float]:
    """Embed a query using the right embedder for a config. Caches embedders."""
    if config_id not in _config_embedders:
        configs = storage.list_embedding_configs()
        cfg = next((c for c in configs if c["config_id"] == config_id), None)
        if not cfg:
            raise ValueError(f"Unknown embedding config: {config_id}")
        _config_embedders[config_id] = create_embedder(
            provider=cfg["provider"], model=cfg["model"]
        )

    embedder = _config_embedders[config_id]
    return embedder.embed(query, input_type="query")


def _get_storage() -> PostgresStorage:
    global _storage
    if _storage is None:
        dsn = os.environ.get(
            "LEGION_KOI_DSN", "postgresql://localhost/personal_koi"
        )
        _storage = PostgresStorage(dsn=dsn)
    return _storage


def _format_results(results: list[dict]) -> str:
    """Format search results with namespace-specific summaries."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        ns = r["namespace"]
        contents = r.get("contents", {})
        rank = r.get("rank") or r.get("similarity") or r.get("rrf_score") or 0

        if ns == "legion.claude-recording":
            title = contents.get("title") or contents.get("filename", "")
            duration = contents.get("duration_human", "")
            date = contents.get("date_recorded", "")
            has_tx = " [transcript]" if contents.get("has_transcript") else ""
            summary = f"{title} ({duration}, {date}){has_tx}"
        elif ns == "legion.claude-message":
            sender = contents.get("sender_id", "")
            ts = (contents.get("platform_ts") or "")[:10]
            content_preview = (contents.get("content") or "")[:PREVIEW_SHORT]
            summary = f"[{sender}] {ts}: {content_preview}"
        elif ns == "legion.claude-journal":
            fm = contents.get("frontmatter", {})
            summary = fm.get("title", "") or fm.get("description", "")
        elif ns == "legion.claude-venture":
            fm = contents.get("frontmatter", {})
            summary = fm.get("title", "")
        elif ns == "legion.claude-logging":
            summary = contents.get("summary") or contents.get("cwd", "")
        elif ns == "legion.claude-web.conversation":
            summary = contents.get("name") or contents.get("summary") or ""
        elif ns == "legion.claude-web.project":
            summary = contents.get("name") or contents.get("description") or ""
        elif ns == "legion.claude-web.memory":
            summary = (contents.get("conversations_memory") or "")[:PREVIEW_MEDIUM]
        elif ns == "legion.claude-code":
            summary = contents.get("summary") or contents.get("cwd") or ""
        elif ns == "legion.claude-github":
            name = contents.get("name") or ""
            desc = contents.get("description") or ""
            summary = f"{name}: {desc}" if desc else name
        else:
            summary = (r.get("search_text") or "")[:PREVIEW_MEDIUM]

        if len(summary) > PREVIEW_MEDIUM:
            summary = summary[:PREVIEW_MEDIUM] + "..."

        score_label = "rrf" if "rrf_score" in r else "sim" if "similarity" in r else "rank"
        headline = r.get("headline")
        headline_line = f"\n   >> {headline}" if headline else ""
        lines.append(
            f"{i}. [{ns}] {r['rid']}\n"
            f"   {score_label}: {rank:.4f}\n"
            f"   {summary}{headline_line}"
        )
    return "\n\n".join(lines)


def _format_rid_list(rids: list[dict]) -> str:
    """Format RID list as a table-like output."""
    if not rids:
        return "No bundles found."

    lines = [f"{'Namespace':<30} {'Reference':<50} {'Updated'}"]
    lines.append("-" * 100)
    for r in rids:
        ns = r.get("namespace", "")
        ref = r.get("reference", "")
        updated = str(r.get("updated_at", ""))[:19]
        lines.append(f"{ns:<30} {ref:<50} {updated}")
    return "\n".join(lines)


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_bundles",
            description="Full-text search across all knowledge bundles (journals, ventures, recordings, messages, conversations, transcripts, GitHub repos).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (supports PostgreSQL websearch syntax: AND, OR, quotes for phrases, - for exclusion)",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace filter: legion.claude-journal, legion.claude-venture, legion.claude-recording, legion.claude-message, legion.claude-logging, legion.claude-web.conversation, legion.claude-web.project, legion.claude-web.memory, legion.claude-code, legion.claude-github",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_bundle",
            description="Get a specific bundle by its full RID (e.g. orn:legion.claude-journal:2026-03-10/1225-some-slug).",
            inputSchema={
                "type": "object",
                "properties": {
                    "rid": {
                        "type": "string",
                        "description": "Full RID string",
                    },
                },
                "required": ["rid"],
            },
        ),
        Tool(
            name="read_transcript",
            description="Read the full transcript text for a recording bundle. Returns metadata header + full transcript content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "rid": {
                        "type": "string",
                        "description": "Full RID of a recording (e.g. orn:legion.claude-recording:otter/2026-03-10-standup)",
                    },
                },
                "required": ["rid"],
            },
        ),
        Tool(
            name="get_thread",
            description="Get all messages in a conversation thread, ordered chronologically.",
            inputSchema={
                "type": "object",
                "properties": {
                    "thread_id": {
                        "type": "string",
                        "description": "Thread ID (e.g. telegram:chat:8576720966)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return (default 50)",
                        "default": 50,
                    },
                },
                "required": ["thread_id"],
            },
        ),
        Tool(
            name="list_bundles",
            description="List bundle RIDs, optionally filtered by namespace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace filter",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 50)",
                        "default": 50,
                    },
                },
            },
        ),
        Tool(
            name="semantic_search",
            description="Search by meaning using vector embeddings. Good for finding related content even when words don't overlap. Use for 'find things related to X' and cross-namespace discovery.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query — meaning matters more than exact words",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace filter",
                    },
                    "config": {
                        "type": "string",
                        "description": "Embedding config ID (e.g. telus-e5-1024, ollama-mxbai-1024). Omit to use default config.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="hybrid_search",
            description="Combined keyword + semantic search (recommended default). Uses Reciprocal Rank Fusion to merge full-text and vector results. Best overall retrieval quality. Enable rerank for higher precision.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — works with keywords, phrases, and natural language",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Optional namespace filter",
                    },
                    "config": {
                        "type": "string",
                        "description": "Embedding config ID (e.g. telus-e5-1024, ollama-mxbai-1024). Omit to use default config.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)",
                        "default": 20,
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Enable cross-encoder reranking for higher precision (default false)",
                        "default": False,
                    },
                    "rerank_backend": {
                        "type": "string",
                        "description": "Reranker backend: 'auto' (default), 'flag', or 'ollama'",
                        "default": "auto",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="entity_search",
            description="Search using hippo's entity graph + KOI document retrieval. Finds entities matching the query in the knowledge graph, traverses relationships via Personalized PageRank, resolves to KOI bundles, and merges with hybrid search. Best for queries about known people, projects, concepts, or tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — entity names and concepts work best",
                    },
                    "config": {
                        "type": "string",
                        "description": "Embedding config ID for hybrid search component. Omit to use default.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)",
                        "default": 20,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="find_entity_bundles",
            description="Find all bundles mentioning a specific entity. Uses trigram fuzzy matching on entity names.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to search for (e.g. 'FalkorDB', 'Shawn', 'KOI-net')",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "Optional entity type filter (e.g. 'Person', 'Tool', 'Concept')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 20)",
                        "default": 20,
                    },
                },
                "required": ["entity"],
            },
        ),
        Tool(
            name="entity_cooccurrence",
            description="Find entities that frequently co-appear with the given entity across bundles. Reveals connections like 'who works on what' and 'which tools are used together'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to find co-occurrences for",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max co-occurring entities (default 20)",
                        "default": 20,
                    },
                },
                "required": ["entity"],
            },
        ),
        Tool(
            name="entity_graph",
            description="Show an entity's neighborhood: co-occurring entities, types, bundle count. Combines KOI PostgreSQL entities with hippo FalkorDB graph when available.",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to explore",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max neighbors (default 20)",
                        "default": 20,
                    },
                },
                "required": ["entity"],
            },
        ),
        Tool(
            name="koi_stats",
            description="Get bundle counts per namespace, embedding coverage, and entity extraction statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="system_health",
            description="Four-dimension health check: availability (services, DB, Redis), performance (throughput), quality (extraction/embedding coverage), growth (ingestion rate). Returns composite 0-100 score.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    storage = _get_storage()

    if name == "search_bundles":
        results = storage.search_text(
            query=arguments["query"],
            namespace=arguments.get("namespace"),
            limit=arguments.get("limit", 20),
        )
        return [TextContent(type="text", text=_format_results(results))]

    elif name == "get_bundle":
        bundle = storage.get_bundle(arguments["rid"])
        if bundle:
            text = json.dumps(bundle, default=str, indent=2)
            return [TextContent(type="text", text=text)]
        return [TextContent(type="text", text="Bundle not found.")]

    elif name == "read_transcript":
        bundle = storage.get_bundle(arguments["rid"])
        if not bundle:
            return [TextContent(type="text", text="Bundle not found.")]
        contents = bundle.get("contents", {})
        transcript = contents.get("transcript_text")
        if not transcript:
            return [TextContent(type="text", text=f"No transcript available for this recording. has_transcript={contents.get('has_transcript', False)}")]
        header = f"# {contents.get('title') or contents.get('filename', 'Unknown')}\n"
        header += f"Source: {contents.get('source', '')} | Date: {contents.get('date_recorded', '')} | Duration: {contents.get('duration_human', '')}\n"
        header += f"Words: {contents.get('transcript_word_count', 'N/A')}\n\n"
        return [TextContent(type="text", text=header + transcript)]

    elif name == "get_thread":
        messages = storage.get_thread_messages(
            thread_id=arguments["thread_id"],
            limit=arguments.get("limit", 50),
        )
        if not messages:
            return [TextContent(type="text", text="No messages found in this thread.")]
        lines = []
        for m in messages:
            c = m.get("contents", {})
            ts = (c.get("platform_ts") or "")[:19]
            sender = c.get("sender_id", "")
            text = c.get("content", "")
            lines.append(f"[{ts}] {sender}:\n{text}")
        return [TextContent(type="text", text="\n\n".join(lines))]

    elif name == "list_bundles":
        rids = storage.list_rids(namespace=arguments.get("namespace"))
        limit = arguments.get("limit", 50)
        return [TextContent(type="text", text=_format_rid_list(rids[:limit]))]

    elif name == "semantic_search":
        try:
            config_id = _resolve_config(storage, arguments.get("config"))
            query_vec = _embed_for_config(storage, config_id, arguments["query"])
            results = storage.search_config_semantic(
                config_id=config_id,
                query_embedding=query_vec,
                namespace=arguments.get("namespace"),
                limit=arguments.get("limit", 20),
            )
            header = f"[config: {config_id}]\n\n"
            return [TextContent(type="text", text=header + _format_results(results))]
        except Exception as e:
            return [TextContent(type="text", text=f"Semantic search error: {e}")]

    elif name == "hybrid_search":
        try:
            from .retrieval.router import classify_query, QueryType
            from .constants import (
                CONVEX_ALPHA, ROUTER_ALPHA_KEYWORD,
                ROUTER_ALPHA_CONCEPTUAL, ROUTER_ALPHA_TEMPORAL,
            )

            config_id = _resolve_config(storage, arguments.get("config"))
            limit = arguments.get("limit", 20)
            rerank_enabled = arguments.get("rerank", False)
            query_text = arguments["query"]

            # Query routing — select alpha based on query type
            qtype = classify_query(query_text)
            alpha_map = {
                QueryType.KEYWORD: ROUTER_ALPHA_KEYWORD,
                QueryType.CONCEPTUAL: ROUTER_ALPHA_CONCEPTUAL,
                QueryType.TEMPORAL: ROUTER_ALPHA_TEMPORAL,
                QueryType.HYBRID: CONVEX_ALPHA,
            }
            alpha = alpha_map.get(qtype, CONVEX_ALPHA)

            # KEYWORD queries skip embedding entirely
            if qtype == QueryType.KEYWORD:
                results = storage.search_text(
                    query=query_text,
                    namespace=arguments.get("namespace"),
                    limit=limit,
                )
                header = f"[route: {qtype.value}, FTS-only]\n\n"
                return [TextContent(type="text", text=header + _format_results(results))]

            query_vec = _embed_for_config(storage, config_id, query_text)

            # Fetch more candidates when reranking (reranker reorders, needs headroom)
            fetch_limit = RERANK_CANDIDATE_POOL if rerank_enabled else limit
            results = storage.search_config_hybrid(
                config_id=config_id,
                query=query_text,
                query_embedding=query_vec,
                namespace=arguments.get("namespace"),
                limit=fetch_limit,
                alpha=alpha,
            )

            header = f"[config: {config_id}] [route: {qtype.value}, α={alpha}]"
            if rerank_enabled and results:
                backend = arguments.get("rerank_backend", "auto")
                reranker = create_reranker(backend=backend)
                results = rerank_results(reranker, query_text, results, top_k=limit)
                header += f" [reranked: {reranker.get_model()}]"

            header += "\n\n"
            return [TextContent(type="text", text=header + _format_results(results))]
        except Exception as e:
            # Fall back to FTS-only if embedding fails
            results = storage.search_text(
                query=arguments["query"],
                namespace=arguments.get("namespace"),
                limit=arguments.get("limit", 20),
            )
            header = f"(Embedding unavailable: {e} — falling back to keyword search)\n\n"
            return [TextContent(type="text", text=header + _format_results(results))]

    elif name == "entity_search":
        try:
            from .hippo_bridge import HippoBridge
            from .resilience.circuit_breaker import CircuitOpenError

            query_text = arguments["query"]
            limit = arguments.get("limit", 20)

            # 1. Hippo entity graph -> RIDs via PPR (circuit-protected)
            circuit = _get_hippo_circuit()
            entity_rids = []
            bridge = None
            try:
                bridge = HippoBridge(
                    redis_host=HIPPO_REDIS_HOST,
                    redis_port=HIPPO_REDIS_PORT,
                    graph_name=HIPPO_GRAPH_NAME,
                )
                entity_rids = circuit.call(bridge.entity_search, query_text, top_k=limit * 2)
            except CircuitOpenError:
                pass  # Fall back to hybrid-only (no entity boost)
            except Exception:
                pass  # FalkorDB unavailable — degrade gracefully

            # 2. KOI hybrid search -> results
            config_id = _resolve_config(storage, arguments.get("config"))
            query_vec = _embed_for_config(storage, config_id, query_text)
            hybrid_results = storage.search_config_hybrid(
                config_id=config_id,
                query=query_text,
                query_embedding=query_vec,
                limit=limit,
            )

            # 3. Merge: union + entity bonus scoring
            bundle_map: dict[str, dict] = {}
            scores: dict[str, float] = {}

            for rank, r in enumerate(hybrid_results, 1):
                rid = r["rid"]
                bundle_map[rid] = r
                scores[rid] = r.get("rrf_score", 1.0 / (60 + rank))

            # Entity-resolved RIDs get a bonus; fetch bundles if not already present
            for rank, rid in enumerate(entity_rids, 1):
                entity_score = ENTITY_RRF_BONUS / rank
                scores[rid] = scores.get(rid, 0) + entity_score
                if rid not in bundle_map:
                    bundle = storage.get_bundle(rid)
                    if bundle:
                        bundle_map[rid] = bundle

            sorted_rids = sorted(scores, key=lambda r: scores[r], reverse=True)[:limit]
            results = []
            for rid in sorted_rids:
                if rid in bundle_map:
                    r = bundle_map[rid]
                    r["rrf_score"] = scores[rid]
                    results.append(r)

            # Enrich with entity triples (skip if bridge unavailable)
            if bridge is not None:
                try:
                    results = circuit.call(bridge.enrich_results, results)
                except (CircuitOpenError, Exception):
                    pass  # Results still usable without enrichment

            # Format with entity annotations
            header = f"[config: {config_id}] [entity graph: {len(entity_rids)} RIDs from hippo]\n\n"
            lines = []
            for i, r in enumerate(results, 1):
                ns = r.get("namespace", "")
                contents = r.get("contents", {})
                rank_score = r.get("rrf_score", 0)

                # Brief summary
                if ns == "legion.claude-message":
                    sender = contents.get("sender_id", "")
                    ts = (contents.get("platform_ts") or "")[:10]
                    content_preview = (contents.get("content") or "")[:PREVIEW_SHORT]
                    summary = f"[{sender}] {ts}: {content_preview}"
                elif ns == "legion.claude-journal":
                    fm = contents.get("frontmatter", {})
                    summary = fm.get("title", "") or fm.get("description", "")
                elif ns == "legion.claude-recording":
                    summary = contents.get("title") or contents.get("filename", "")
                else:
                    summary = (r.get("search_text") or "")[:PREVIEW_MEDIUM]

                if len(summary) > PREVIEW_MEDIUM:
                    summary = summary[:PREVIEW_MEDIUM] + "..."

                entity_lines = ""
                if r.get("entities"):
                    entity_lines = "\n   " + "\n   ".join(r["entities"][:3])

                headline = r.get("headline")
                headline_line = f"\n   >> {headline}" if headline else ""

                lines.append(
                    f"{i}. [{ns}] {r.get('rid', '?')}\n"
                    f"   score: {rank_score:.4f}\n"
                    f"   {summary}{headline_line}{entity_lines}"
                )

            return [TextContent(type="text", text=header + "\n\n".join(lines) if lines else "No results found.")]
        except ImportError:
            return [TextContent(type="text", text="Entity search unavailable: redis package not installed. Run: uv pip install redis")]
        except Exception as e:
            return [TextContent(type="text", text=f"Entity search error: {e}")]

    elif name == "find_entity_bundles":
        try:
            results = storage.find_bundles_by_entity(
                name=arguments["entity"],
                entity_type=arguments.get("entity_type"),
                limit=arguments.get("limit", 20),
            )
            if not results:
                return [TextContent(type="text", text=f"No bundles found mentioning entity: {arguments['entity']}")]
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(
                    f"{i}. [{r['namespace']}] {r['rid']}\n"
                    f"   entity: {r['entity_name']} ({r['entity_type']}) sim={r['sim']:.3f} conf={r['confidence']:.2f}"
                )
            return [TextContent(type="text", text=f"Bundles mentioning '{arguments['entity']}':\n\n" + "\n\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Entity search error: {e}")]

    elif name == "entity_cooccurrence":
        try:
            results = storage.find_entity_cooccurrence(
                name=arguments["entity"],
                limit=arguments.get("limit", 20),
            )
            if not results:
                return [TextContent(type="text", text=f"No co-occurring entities found for: {arguments['entity']}")]
            lines = [f"Entities co-occurring with '{arguments['entity']}':\n"]
            for r in results:
                lines.append(
                    f"  {r['name']} ({r['entity_type']}/{r['supertype']}) "
                    f"— {r['shared_bundles']} shared bundles, {r['mention_count']} total mentions"
                )
            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Co-occurrence error: {e}")]

    elif name == "entity_graph":
        try:
            entity_name = arguments["entity"]
            limit = arguments.get("limit", 20)

            # KOI entities
            matches = storage.search_entities(entity_name, limit=1)
            if not matches:
                return [TextContent(type="text", text=f"Entity not found: {entity_name}")]

            entity = matches[0]
            cooccurrences = storage.find_entity_cooccurrence(entity_name, limit=limit)
            bundles = storage.find_bundles_by_entity(entity_name, limit=limit)

            lines = [
                f"# Entity: {entity['name']} ({entity['entity_type']}/{entity['supertype']})",
                f"Mentions: {entity['mention_count']} | First: {str(entity['first_seen'])[:10]} | Last: {str(entity['last_seen'])[:10]}",
                f"\n## Bundles ({len(bundles)})",
            ]
            for b in bundles[:10]:
                lines.append(f"  - [{b['namespace']}] {b['rid']} (conf={b['confidence']:.2f})")

            lines.append(f"\n## Co-occurring entities ({len(cooccurrences)})")
            for c in cooccurrences[:15]:
                lines.append(
                    f"  - {c['name']} ({c['entity_type']}) — {c['shared_bundles']} shared"
                )

            # Try hippo bridge for additional graph data (circuit-protected)
            try:
                from .hippo_bridge import HippoBridge
                from .resilience.circuit_breaker import CircuitOpenError
                hippo_cb = _get_hippo_circuit()
                hippo_bridge = HippoBridge(
                    redis_host=HIPPO_REDIS_HOST,
                    redis_port=HIPPO_REDIS_PORT,
                    graph_name=HIPPO_GRAPH_NAME,
                )
                # Use enrich_results to get triples for the entity's bundles
                enrichable = [{"rid": b["rid"]} for b in bundles[:5]]
                enriched = hippo_cb.call(hippo_bridge.enrich_results, enrichable)
                all_triples = []
                for r in enriched:
                    all_triples.extend(r.get("entities", []))
                if all_triples:
                    unique_triples = list(dict.fromkeys(all_triples))
                    lines.append(f"\n## Hippo graph triples ({len(unique_triples)})")
                    for t in unique_triples[:10]:
                        lines.append(f"  - {t}")
            except Exception:
                pass

            return [TextContent(type="text", text="\n".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"Entity graph error: {e}")]

    elif name == "koi_stats":
        stats = storage.get_stats()
        config_stats = storage.get_config_stats()
        configs = storage.list_embedding_configs()
        output = {
            "bundles": stats,
            "embedding_configs": configs,
            "config_embeddings": config_stats,
        }
        # Add entity stats
        try:
            entity_stats = storage.get_entity_stats()
            output["entities"] = entity_stats
        except Exception:
            output["entities"] = "not available (tables may not exist)"
        return [TextContent(type="text", text=json.dumps(output, indent=2, default=str))]

    elif name == "system_health":
        try:
            from .observability.health import compute_health
            from .events.bus import EventBus

            event_bus = None
            try:
                event_bus = EventBus()
                if not event_bus.ping():
                    event_bus = None
            except Exception:
                pass

            health = compute_health(storage=storage, event_bus=event_bus)
            text = health.summary()

            if event_bus:
                event_bus.close()
            return [TextContent(type="text", text=text)]
        except Exception as e:
            return [TextContent(type="text", text=f"Health check error: {e}")]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
