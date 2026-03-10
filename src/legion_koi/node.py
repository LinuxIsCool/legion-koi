"""Legion KOI-net FullNode definition."""

from koi_net.core import FullNode

from .config import LegionKoiConfig
from .handlers import (
    JournalBundleHandler,
    VentureBundleHandler,
    SuppressNetworkHandler,
    PostgresStorageHandler,
    LoggingFinalHandler,
)


class LegionKoiNode(FullNode):
    config_schema = LegionKoiConfig

    # Custom handlers (added as class attributes — assembler wires them)
    journal_bundle_handler: JournalBundleHandler = JournalBundleHandler
    venture_bundle_handler: VentureBundleHandler = VentureBundleHandler
    suppress_network_handler: SuppressNetworkHandler = SuppressNetworkHandler
    postgres_storage_handler: PostgresStorageHandler = PostgresStorageHandler
    logging_final_handler: LoggingFinalHandler = LoggingFinalHandler
