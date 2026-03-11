"""MCP server exposing legion-koi knowledge graph to Claude Code."""

import asyncio
import json
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .storage.postgres import PostgresStorage
from .embeddings import embed_query, get_embedder, create_embedder

app = Server("legion-koi")

_storage: PostgresStorage | None = None
_config_embedders: dict[str, object] = {}


def _embed_for_config(storage: PostgresStorage, config_id: str | None, query: str) -> list[float]:
    """Embed a query using the right embedder for a config. Caches embedders."""
    if not config_id:
        return embed_query(query)

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
            "LEGION_KOI_DSN", "postgresql://shawn@localhost/personal_koi"
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
            content_preview = (contents.get("content") or "")[:150]
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
            summary = (contents.get("conversations_memory") or "")[:200]
        elif ns == "legion.claude-code":
            summary = contents.get("summary") or contents.get("cwd") or ""
        elif ns == "legion.claude-github":
            name = contents.get("name") or ""
            desc = contents.get("description") or ""
            summary = f"{name}: {desc}" if desc else name
        else:
            summary = (r.get("search_text") or "")[:200]

        if len(summary) > 200:
            summary = summary[:200] + "..."

        score_label = "rrf" if "rrf_score" in r else "sim" if "similarity" in r else "rank"
        lines.append(
            f"{i}. [{ns}] {r['rid']}\n"
            f"   {score_label}: {rank:.4f}\n"
            f"   {summary}"
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
                        "description": "Embedding config ID (e.g. telus-e5-1024, ollama-mxbai-1024). Omit to use legacy table or default config.",
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
            description="Combined keyword + semantic search (recommended default). Uses Reciprocal Rank Fusion to merge full-text and vector results. Best overall retrieval quality.",
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
                        "description": "Embedding config ID (e.g. telus-e5-1024, ollama-mxbai-1024). Omit to use legacy table or default config.",
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
            name="koi_stats",
            description="Get bundle counts per namespace and embedding coverage in the knowledge graph.",
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
        config_id = arguments.get("config")
        try:
            query_vec = _embed_for_config(storage, config_id, arguments["query"])
            if config_id:
                results = storage.search_config_semantic(
                    config_id=config_id,
                    query_embedding=query_vec,
                    namespace=arguments.get("namespace"),
                    limit=arguments.get("limit", 20),
                )
            else:
                results = storage.search_semantic(
                    query_embedding=query_vec,
                    namespace=arguments.get("namespace"),
                    limit=arguments.get("limit", 20),
                )
            header = f"[config: {config_id or 'legacy'}]\n\n" if config_id else ""
            return [TextContent(type="text", text=header + _format_results(results))]
        except Exception as e:
            return [TextContent(type="text", text=f"Semantic search error: {e}")]

    elif name == "hybrid_search":
        config_id = arguments.get("config")
        try:
            query_vec = _embed_for_config(storage, config_id, arguments["query"])
            if config_id:
                results = storage.search_config_hybrid(
                    config_id=config_id,
                    query=arguments["query"],
                    query_embedding=query_vec,
                    namespace=arguments.get("namespace"),
                    limit=arguments.get("limit", 20),
                )
            else:
                results = storage.search_hybrid(
                    query=arguments["query"],
                    query_embedding=query_vec,
                    namespace=arguments.get("namespace"),
                    limit=arguments.get("limit", 20),
                )
            header = f"[config: {config_id or 'legacy'}]\n\n" if config_id else ""
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

    elif name == "koi_stats":
        stats = storage.get_stats()
        embedding_stats = storage.get_embedding_stats()
        config_stats = storage.get_config_stats()
        configs = storage.list_embedding_configs()
        try:
            embedder = get_embedder()
            model_info = {"model": embedder.get_model(), "dimensions": embedder.get_dimensions()}
        except Exception:
            model_info = {"model": "unavailable"}
        output = {
            "bundles": stats,
            "legacy_embeddings": embedding_stats,
            "embedding_configs": configs,
            "config_embeddings": config_stats,
            "active_embedder": model_info,
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2, default=str))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
