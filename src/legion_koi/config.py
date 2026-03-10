"""Legion KOI-net node configuration."""

from pathlib import Path

from pydantic import BaseModel, Field
from koi_net.config.full_node import FullNodeConfig, FullNodeProfile
from koi_net.config.koi_net_config import KoiNetConfig
from koi_net.config.server_config import ServerConfig
from koi_net.protocol.node import NodeProvides

from .rid_types import LegionJournal, LegionVenture, LegionRecording, LegionSession


class SensorConfig(BaseModel):
    journal_watch_dir: str = "~/legion-brain/local/journal/"
    journal_state_path: str = "./state/journal_state.json"


class LegionKoiConfig(FullNodeConfig):
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    server: ServerConfig = ServerConfig(port=8100)
    koi_net: KoiNetConfig = KoiNetConfig(
        node_name="legion-koi",
        node_profile=FullNodeProfile(
            provides=NodeProvides(
                event=[LegionJournal, LegionVenture, LegionRecording, LegionSession],
                state=[LegionJournal],
            ),
        ),
        cache_directory_path=Path(".rid_cache"),
    )
