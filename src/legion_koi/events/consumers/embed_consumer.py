"""Embedding consumer — subscribes to bundle.created, embeds search text.

Extracted from handlers.py:_embed_bundle(). Now runs asynchronously via
Redis Streams consumer group, with retry and DLQ support.
"""

from __future__ import annotations

import structlog

from ...constants import EMBED_MIN_CONTENT_CHARS
from ...storage.postgres import _extract_search_text, PostgresStorage
from ...resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from ..schemas import KoiEvent, BUNDLE_CREATED, EMBEDDING_COMPUTED
from ..consumer import EventConsumer
from ..bus import EventBus

log = structlog.stdlib.get_logger()


class EmbedConsumer(EventConsumer):
    """Embeds bundle search text into all active embedding configs.

    Subscribes to bundle.created events.
    On success, publishes embedding.computed event.
    Wraps embedding API calls in a circuit breaker per provider.
    """

    event_type = BUNDLE_CREATED
    group = "embed"

    def __init__(self, bus: EventBus, storage: PostgresStorage, consumer_id: str | None = None):
        super().__init__(bus, consumer_id)
        self._storage = storage
        self._circuits: dict[str, CircuitBreaker] = {}

    def handle(self, event: KoiEvent) -> None:
        rid = event.data["rid"]
        namespace = event.data["namespace"]

        bundle = self._storage.get_bundle(rid)
        if not bundle:
            log.debug("embed_consumer.bundle_not_found", rid=rid)
            return

        contents = bundle["contents"]
        search_text = _extract_search_text(namespace, contents)
        if not search_text or not search_text.strip():
            return

        # Quality gate: skip embedding for trivially short messages
        if namespace == "legion.claude-message" and len(search_text) < EMBED_MIN_CONTENT_CHARS:
            return

        from ...chunking import chunk_text
        from ...contextual import extract_preamble, prepend_preamble
        from ...embeddings import create_embedder

        chunks = chunk_text(search_text)
        if not chunks:
            return

        preamble = extract_preamble(namespace, contents)
        configs = self._storage.list_embedding_configs()
        total_chunks = 0

        for cfg in configs:
            provider = cfg["provider"]
            # Per-provider circuit breaker — if TELUS API is down, skip all TELUS configs
            if provider not in self._circuits:
                self._circuits[provider] = CircuitBreaker(
                    name=f"embed-{provider}", event_bus=self._bus
                )
            circuit = self._circuits[provider]

            try:
                cfg_embedder = create_embedder(
                    provider=provider, model=cfg["model"]
                )
                is_contextual = cfg["config_id"].endswith("-ctx")
                self._storage.delete_config_embeddings(cfg["config_id"], rid)
                for i, chunk in enumerate(chunks):
                    embed_input = prepend_preamble(preamble, chunk) if is_contextual else chunk
                    vec = circuit.call(cfg_embedder.embed, embed_input, input_type="passage")
                    self._storage.upsert_config_embedding(
                        config_id=cfg["config_id"],
                        rid=rid,
                        embedding=vec,
                        chunk_index=i,
                        chunk_text=chunk,
                    )
                    total_chunks += 1
            except CircuitOpenError:
                log.debug("embed_consumer.circuit_open", rid=rid, config=cfg["config_id"], provider=provider)
            except Exception:
                log.debug("embed_consumer.config_skip", rid=rid, config=cfg["config_id"], exc_info=True)

        if total_chunks > 0:
            self._bus.publish(KoiEvent(
                type=EMBEDDING_COMPUTED,
                subject=rid,
                data={"rid": rid, "config_count": len(configs), "chunk_count": total_chunks},
            ))
            log.debug("embed_consumer.done", rid=rid, chunks=total_chunks)
