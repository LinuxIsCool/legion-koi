"""Legion KOI-net FullNode definition.

Legion is a FullNode (serves API) but ALSO needs to poll peers
(like a PartialNode). We add the NodePoller component so the
build graph wires it up alongside the server.
"""

from koi_net.core import FullNode
from koi_net.components.poller import NodePoller

from .config import LegionKoiConfig
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
    poller: NodePoller = NodePoller

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
