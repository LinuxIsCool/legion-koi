"""Legion KOI-net FullNode definition.

Legion is a FullNode (serves API) but ALSO needs to poll peers
(like a PartialNode). We add the NodePoller component so the
build graph wires it up alongside the server.
"""

import structlog
from dataclasses import dataclass

from koi_net.core import FullNode
from koi_net.components.poller import NodePoller

from .config import LegionKoiConfig

log = structlog.stdlib.get_logger()


@dataclass
class ResilientPoller(NodePoller):
    """NodePoller that skips malformed events instead of crashing.

    The upstream NodePoller propagates any exception from event processing
    up to ThreadedComponent._run(), which triggers a service-wide shutdown.
    One malformed RID from a peer should not kill the entire node.
    """

    def poll(self):
        """Polls neighbor nodes, skipping individual event errors."""
        for node_rid, events in self.resolver.poll_neighbors().items():
            for event in events:
                try:
                    self.kobj_queue.push(event=event, source=node_rid)
                except (TypeError, ValueError, KeyError) as exc:
                    log.warning(
                        "poller.event_skipped",
                        error=str(exc),
                        source=str(node_rid),
                        event_type=getattr(event, "type", "unknown"),
                    )
from .handlers import (
    BrowserHistoryBundleHandler,
    JournalBundleHandler,
    VentureBundleHandler,
    RecordingBundleHandler,
    MessageBundleHandler,
    PlanBundleHandler,
    ResearchBundleHandler,
    PostgresStorageHandler,
    LoggingFinalHandler,
)


class LegionKoiNode(FullNode):
    config_schema = LegionKoiConfig

    # Poller — FullNode doesn't include this by default (only PartialNode does).
    # We need it to poll Darren's node for vault-file sync and federation.
    # ResilientPoller wraps NodePoller with try/except per event to prevent
    # one malformed RID from crashing the entire service.
    poller: ResilientPoller = ResilientPoller

    # Custom handlers (added as class attributes — assembler wires them)
    browser_history_bundle_handler: BrowserHistoryBundleHandler = BrowserHistoryBundleHandler
    journal_bundle_handler: JournalBundleHandler = JournalBundleHandler
    venture_bundle_handler: VentureBundleHandler = VentureBundleHandler
    recording_bundle_handler: RecordingBundleHandler = RecordingBundleHandler
    message_bundle_handler: MessageBundleHandler = MessageBundleHandler
    plan_bundle_handler: PlanBundleHandler = PlanBundleHandler
    research_bundle_handler: ResearchBundleHandler = ResearchBundleHandler
    postgres_storage_handler: PostgresStorageHandler = PostgresStorageHandler
    logging_final_handler: LoggingFinalHandler = LoggingFinalHandler
