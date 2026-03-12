"""Entity extraction consumer — subscribes to bundle.created, extracts entities.

Extracted from handlers.py:_extract_bundle_entities(). Now runs asynchronously
via Redis Streams consumer group, with retry and DLQ support.
"""

from __future__ import annotations

import structlog

from ...constants import ENTITY_EXTRACTION_SKIP_NAMESPACES
from ...storage.postgres import _extract_search_text
from ..schemas import KoiEvent, BUNDLE_CREATED, ENTITY_EXTRACTED
from ..consumer import EventConsumer
from ..bus import EventBus

log = structlog.stdlib.get_logger()


class ExtractConsumer(EventConsumer):
    """Extracts entities from bundle search text and stores them.

    Subscribes to bundle.created events.
    On success, publishes entity.extracted event.
    """

    event_type = BUNDLE_CREATED
    group = "extract"

    def __init__(self, bus: EventBus, storage, consumer_id: str | None = None):
        super().__init__(bus, consumer_id)
        self._storage = storage

    def handle(self, event: KoiEvent) -> None:
        rid = event.data["rid"]
        namespace = event.data["namespace"]

        if namespace in ENTITY_EXTRACTION_SKIP_NAMESPACES:
            return

        bundle = self._storage.get_bundle(rid)
        if not bundle:
            log.debug("extract_consumer.bundle_not_found", rid=rid)
            return

        contents = bundle["contents"]
        search_text = _extract_search_text(namespace, contents)
        if not search_text or not search_text.strip():
            return

        from ...extraction import run_extraction, normalize_entity_name

        result = run_extraction(rid, namespace, search_text)
        if not result.entities:
            return

        entity_dicts = []
        for e in result.entities:
            name_normalized = normalize_entity_name(e.name)
            entity_dicts.append({
                "name": e.name,
                "entity_type": e.entity_type,
                "supertype": e.supertype,
                "confidence": e.confidence,
                "name_normalized": name_normalized,
            })

        self._storage.upsert_bundle_entities(rid, entity_dicts)

        self._bus.publish(KoiEvent(
            type=ENTITY_EXTRACTED,
            subject=rid,
            data={"rid": rid, "entity_count": len(entity_dicts)},
        ))
        log.debug("extract_consumer.done", rid=rid, entities=len(entity_dicts))
