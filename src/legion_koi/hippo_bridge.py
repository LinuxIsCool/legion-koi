"""Bridge between hippo's entity graph (FalkorDB) and KOI's bundle storage (PostgreSQL).

Enables associative retrieval: query entities in hippo -> resolve to source bundles in KOI.
Hippo stores triples with source fields like "journal:2026/03/09/17-11-slug".
KOI stores RIDs like "orn:legion.claude-journal:2026/03/09/17-11-slug".
The mapping is deterministic.
"""

import json
import re
import structlog
from collections import defaultdict

log = structlog.get_logger()

# Hippo source prefix -> KOI namespace
NAMESPACE_MAP = {
    "journal": "legion.claude-journal",
    "venture": "legion.claude-venture",
    "inventory": "legion.claude-inventory",
    "ground": "legion.claude-ground",
    "task": "legion.claude-backlog",
    "youtube": "legion.claude-youtube",
}

# PPR defaults
PPR_DAMPING = 0.85
PPR_ITERATIONS = 30
PPR_EPSILON = 1e-6


def hippo_source_to_koi_rid(source: str) -> str:
    """Convert a hippo source ID to a KOI RID.

    Examples:
        journal:2026/03/09/17-11-slug -> orn:legion.claude-journal:2026/03/09/17-11-slug
        venture:kwaxala -> orn:legion.claude-venture:kwaxala
        inventory:legion -> orn:legion.claude-inventory:legion
        ground:gk01 -> orn:legion.claude-ground:gk01
    """
    prefix, _, ref = source.partition(":")
    if not ref:
        return ""
    ns = NAMESPACE_MAP.get(prefix)
    if ns:
        return f"orn:{ns}:{ref}"
    return ""


class HippoBridge:
    """Connects hippo's entity graph to KOI's bundle storage."""

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6380,
                 graph_name: str = "hippo"):
        import redis as redis_lib
        self._redis = redis_lib.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self._graph = graph_name

    def _query(self, cypher: str) -> list[dict]:
        """Execute a FalkorDB graph query and parse results."""
        try:
            result = self._redis.execute_command("GRAPH.QUERY", self._graph, cypher)
        except Exception as e:
            log.warning("hippo_bridge.query_error", error=str(e), cypher=cypher[:100])
            return []
        if not result or len(result) < 2:
            return []
        header = result[0]
        rows = result[1] if len(result) > 1 else []
        if not header or not rows:
            return []
        return [dict(zip(header, row)) for row in rows]

    def find_entities(self, query: str) -> list[str]:
        """Find entity names matching query terms (exact + fuzzy CONTAINS)."""
        # Extract meaningful words from query
        stopwords = {
            "what", "how", "why", "when", "where", "who", "is", "are", "was",
            "were", "do", "does", "did", "the", "a", "an", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "about", "i", "my", "me",
            "know", "tell", "show", "find", "get", "related", "all",
        }
        words = re.findall(r'\b[a-zA-Z][\w-]*\b', query.lower())
        terms = [w for w in words if w not in stopwords and len(w) > 1]

        entities = set()

        # Exact match
        for term in terms:
            esc = term.replace('"', '\\"')
            rows = self._query(f'MATCH (n:Entity) WHERE toLower(n.name) = "{esc}" RETURN n.name')
            for r in rows:
                entities.add(r["n.name"])

        # CONTAINS match for terms not yet found
        for term in terms:
            if any(term in e.lower() for e in entities):
                continue
            esc = term.replace('"', '\\"')
            rows = self._query(
                f'MATCH (n:Entity) WHERE toLower(n.name) CONTAINS "{esc}" RETURN n.name LIMIT 10'
            )
            for r in rows:
                entities.add(r["n.name"])

        return list(entities)

    def get_entity_sources(self, entity_name: str) -> list[str]:
        """Get all source IDs linked to an entity via RELATES edges."""
        esc = entity_name.replace('"', '\\"')
        # Sources from outgoing edges
        rows_out = self._query(
            f'MATCH (s:Entity {{name: "{esc}"}})-[r:RELATES]->() RETURN DISTINCT r.source AS source'
        )
        # Sources from incoming edges
        rows_in = self._query(
            f'MATCH ()-[r:RELATES]->(o:Entity {{name: "{esc}"}}) RETURN DISTINCT r.source AS source'
        )
        sources = set()
        for r in rows_out + rows_in:
            src = r.get("source")
            if src:
                sources.add(src)
        return list(sources)

    def get_entity_rids(self, entity_name: str) -> list[str]:
        """Get KOI RIDs linked to an entity."""
        sources = self.get_entity_sources(entity_name)
        rids = []
        for s in sources:
            rid = hippo_source_to_koi_rid(s)
            if rid:
                rids.append(rid)
        return rids

    def entity_search(self, query: str, top_k: int = 20) -> list[str]:
        """Full pipeline: find query entities -> PPR -> resolve to KOI RIDs.

        Returns a list of KOI RIDs ranked by PPR score, representing bundles
        that are associatively connected to the query through the entity graph.
        """
        # 1. Find seed entities
        seed_entities = self.find_entities(query)
        if not seed_entities:
            log.info("hippo_bridge.no_seeds", query=query)
            return []

        # 2. Pull graph and run PPR from seeds
        adj_out, adj_in, all_nodes = self._pull_graph()
        if not all_nodes:
            return []

        scores = self._run_ppr(adj_out, adj_in, all_nodes, seed_entities)

        # 3. Rank entities by PPR score, collect their source RIDs
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Collect RIDs from top PPR entities (not just seeds)
        rid_scores: dict[str, float] = {}
        for entity_name, ppr_score in ranked[:50]:
            for rid in self.get_entity_rids(entity_name):
                if rid not in rid_scores or ppr_score > rid_scores[rid]:
                    rid_scores[rid] = ppr_score

        # Sort by PPR score, return top_k
        sorted_rids = sorted(rid_scores, key=lambda r: rid_scores[r], reverse=True)
        log.info("hippo_bridge.entity_search",
                 query=query, seeds=len(seed_entities),
                 ppr_entities=len(ranked), rids=len(sorted_rids))
        return sorted_rids[:top_k]

    def enrich_results(self, results: list[dict]) -> list[dict]:
        """Add entity metadata to KOI search results (lightweight annotation)."""
        for r in results:
            rid = r.get("rid", "")
            # Reverse-map RID to hippo source
            source = self._rid_to_hippo_source(rid)
            if not source:
                continue
            esc = source.replace('"', '\\"')
            rows = self._query(
                f'MATCH (s)-[r:RELATES {{source: "{esc}"}}]->(o) '
                f'RETURN s.name AS subj, r.type AS rel, o.name AS obj LIMIT 5'
            )
            if rows:
                r["entities"] = [
                    f"{row['subj']} —{row['rel']}→ {row['obj']}" for row in rows
                ]
        return results

    def _pull_graph(self) -> tuple[dict, dict, set]:
        """Pull the edge list for PPR computation."""
        rows = self._query(
            "MATCH (s:Entity)-[r:RELATES]->(o:Entity) "
            "RETURN s.name AS src, o.name AS dst, r.weight AS weight"
        )
        adj_out: dict[str, list[tuple[str, float]]] = defaultdict(list)
        adj_in: dict[str, list[tuple[str, float]]] = defaultdict(list)
        all_nodes: set[str] = set()

        for row in rows:
            src, dst = row["src"], row["dst"]
            w = float(row["weight"]) if row["weight"] else 1.0
            all_nodes.add(src)
            all_nodes.add(dst)
            adj_out[src].append((dst, w))
            adj_in[dst].append((src, w))

        return adj_out, adj_in, all_nodes

    def _run_ppr(self, adj_out: dict, adj_in: dict, all_nodes: set,
                 seeds: list[str]) -> dict[str, float]:
        """Personalized PageRank from seed entities."""
        seed_set = set(seeds) & all_nodes
        if not seed_set:
            return {}

        p = {n: (1.0 / len(seed_set) if n in seed_set else 0.0) for n in all_nodes}
        scores = dict(p)

        # Precompute out-weight sums
        out_w = {}
        for n in all_nodes:
            total = sum(w for _, w in adj_out.get(n, []))
            out_w[n] = total if total > 0 else 1.0

        for _ in range(PPR_ITERATIONS):
            new_scores = {}
            max_delta = 0.0
            for node in all_nodes:
                incoming = sum(
                    scores.get(src, 0.0) * (w / out_w[src])
                    for src, w in adj_in.get(node, [])
                )
                new_scores[node] = (1 - PPR_DAMPING) * p[node] + PPR_DAMPING * incoming
                max_delta = max(max_delta, abs(new_scores[node] - scores.get(node, 0.0)))
            scores = new_scores
            if max_delta < PPR_EPSILON:
                break

        return scores

    @staticmethod
    def _rid_to_hippo_source(rid: str) -> str:
        """Reverse-map a KOI RID to a hippo source ID."""
        # orn:legion.claude-journal:2026/03/09/17-11-slug -> journal:2026/03/09/17-11-slug
        if not rid.startswith("orn:"):
            return ""
        rest = rid[4:]  # strip "orn:"
        ns, _, ref = rest.partition(":")
        if not ref:
            return ""
        # Reverse lookup
        for prefix, namespace in NAMESPACE_MAP.items():
            if ns == namespace:
                return f"{prefix}:{ref}"
        return ""
