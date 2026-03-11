"""Custom knowledge handlers for Legion KOI-net node."""

from dataclasses import dataclass
from logging import Logger

import structlog

from koi_net.components.interfaces import KnowledgeHandler, HandlerType
from koi_net.protocol.knowledge_object import KnowledgeObject
from koi_net.protocol.event import EventType

from .rid_types import LegionJournal, LegionVenture, LegionRecording, LegionMessage
from .storage.postgres import _extract_search_text

slog = structlog.stdlib.get_logger()

# Module-level storage reference — set by __main__ if PostgreSQL is available
_postgres_storage = None

REQUIRED_JOURNAL_FIELDS = {"title", "created"}


@dataclass
class JournalBundleHandler(KnowledgeHandler):
    """Validates journal bundle contents have required frontmatter fields."""

    handler_type = HandlerType.Bundle
    rid_types = (LegionJournal,)
    event_types = (EventType.NEW, EventType.UPDATE)

    def handle(self, kobj: KnowledgeObject) -> KnowledgeObject | None:
        frontmatter = kobj.contents.get("frontmatter", {})
        missing = REQUIRED_JOURNAL_FIELDS - set(frontmatter.keys())
        if missing:
            slog.warning(
                "journal.validation_warning",
                rid=str(kobj.rid),
                missing_fields=list(missing),
            )
        kobj.normalized_event_type = kobj.event_type or EventType.NEW
        return kobj


@dataclass
class RecordingBundleHandler(KnowledgeHandler):
    """Validates recording bundle contents have source and filename."""

    handler_type = HandlerType.Bundle
    rid_types = (LegionRecording,)
    event_types = (EventType.NEW, EventType.UPDATE)

    def handle(self, kobj: KnowledgeObject) -> KnowledgeObject | None:
        source = kobj.contents.get("source")
        filename = kobj.contents.get("filename")
        if not source or not filename:
            slog.warning(
                "recording.validation_warning",
                rid=str(kobj.rid),
                missing=["source" if not source else "", "filename" if not filename else ""],
            )
        kobj.normalized_event_type = kobj.event_type or EventType.NEW
        return kobj


@dataclass
class MessageBundleHandler(KnowledgeHandler):
    """Validates message bundle contents have content."""

    handler_type = HandlerType.Bundle
    rid_types = (LegionMessage,)
    event_types = (EventType.NEW, EventType.UPDATE)

    def handle(self, kobj: KnowledgeObject) -> KnowledgeObject | None:
        kobj.normalized_event_type = kobj.event_type or EventType.NEW
        return kobj


@dataclass
class SuppressNetworkHandler(KnowledgeHandler):
    """Phase 1: suppress all network broadcast (no external nodes yet)."""

    handler_type = HandlerType.Network

    def handle(self, kobj: KnowledgeObject) -> KnowledgeObject | None:
        kobj.network_targets = set()
        return kobj


REQUIRED_VENTURE_FIELDS = {"title"}


@dataclass
class VentureBundleHandler(KnowledgeHandler):
    """Validates venture bundle contents have required frontmatter fields."""

    handler_type = HandlerType.Bundle
    rid_types = (LegionVenture,)
    event_types = (EventType.NEW, EventType.UPDATE)

    def handle(self, kobj: KnowledgeObject) -> KnowledgeObject | None:
        frontmatter = kobj.contents.get("frontmatter", {})
        missing = REQUIRED_VENTURE_FIELDS - set(frontmatter.keys())
        if missing:
            slog.warning(
                "venture.validation_warning",
                rid=str(kobj.rid),
                missing_fields=list(missing),
            )
        kobj.normalized_event_type = kobj.event_type or EventType.NEW
        return kobj


def _embed_bundle(rid: str, namespace: str, contents: dict) -> None:
    """Embed a bundle's search text into legacy table + all active configs. Best-effort."""
    if _postgres_storage is None:
        return
    try:
        from .embeddings import embed_texts, get_embedder, create_embedder

        search_text = _extract_search_text(namespace, contents)
        if not search_text or not search_text.strip():
            return

        # Legacy table — uses the default embedder
        vectors = embed_texts([search_text], input_type="passage")
        embedder = get_embedder()
        _postgres_storage.upsert_embedding(
            rid=rid,
            embedding=vectors[0],
            model=embedder.get_model(),
        )

        # Config-based tables — embed into each registered config
        try:
            configs = _postgres_storage.list_embedding_configs()
            for cfg in configs:
                try:
                    cfg_embedder = create_embedder(
                        provider=cfg["provider"], model=cfg["model"]
                    )
                    vec = cfg_embedder.embed(search_text, input_type="passage")
                    _postgres_storage.upsert_config_embedding(
                        config_id=cfg["config_id"],
                        rid=rid,
                        embedding=vec,
                        chunk_text=search_text[:200],
                    )
                except Exception:
                    slog.debug("embedding.config_inline_skip", rid=rid, config=cfg["config_id"])
        except Exception:
            pass  # Config system not available — legacy-only is fine
    except Exception:
        slog.warning("embedding.inline_error", rid=rid, exc_info=True)


@dataclass
class PostgresStorageHandler(KnowledgeHandler):
    """Persist processed bundles to PostgreSQL for search and retrieval."""

    handler_type = HandlerType.Final

    def handle(self, kobj: KnowledgeObject) -> None:
        if _postgres_storage is None:
            return
        try:
            _postgres_storage.upsert_bundle(
                rid=str(kobj.rid),
                namespace=kobj.rid.namespace,
                reference=kobj.rid.reference,
                contents=kobj.contents,
                sha256_hash=kobj.bundle.manifest.sha256_hash if kobj.bundle else "",
            )
        except Exception:
            slog.exception("postgres.upsert_error", rid=str(kobj.rid))
            return

        # Inline embedding — best-effort, never blocks bundle storage
        _embed_bundle(str(kobj.rid), kobj.rid.namespace, kobj.contents)


@dataclass
class LoggingFinalHandler(KnowledgeHandler):
    """Log all processed objects with structlog."""

    handler_type = HandlerType.Final

    def handle(self, kobj: KnowledgeObject) -> None:
        slog.info(
            "pipeline.processed",
            rid=str(kobj.rid),
            event_type=str(kobj.normalized_event_type),
            rid_type=str(type(kobj.rid)),
        )
