"""Legion KOI-net FullNode definition."""

from koi_net.core import FullNode

from .config import LegionKoiConfig
from .handlers import JournalBundleHandler, SuppressNetworkHandler, LoggingFinalHandler


class LegionKoiNode(FullNode):
    config_schema = LegionKoiConfig

    # Custom handlers (added as class attributes — assembler wires them)
    journal_bundle_handler: JournalBundleHandler = JournalBundleHandler
    suppress_network_handler: SuppressNetworkHandler = SuppressNetworkHandler
    logging_final_handler: LoggingFinalHandler = LoggingFinalHandler
