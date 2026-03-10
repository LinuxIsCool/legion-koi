"""Custom knowledge handlers for Legion KOI-net node."""

from dataclasses import dataclass
from logging import Logger

import structlog

from koi_net.components.interfaces import KnowledgeHandler, HandlerType
from koi_net.protocol.knowledge_object import KnowledgeObject
from koi_net.protocol.event import EventType

from .rid_types import LegionJournal, LegionVenture

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
