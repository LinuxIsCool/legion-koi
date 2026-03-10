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
    venture_watch_dir: str = "~/legion-brain/local/ventures/"
    venture_state_path: str = "./state/venture_state.json"
    logging_db_path: str = "~/.claude/local/logging/-home-shawn/db/logging.db"
    logging_state_path: str = "./state/logging_state.json"
    logging_poll_interval: float = 60.0


class PostgresConfig(BaseModel):
    dsn: str = "postgresql://shawn@localhost/personal_koi"


class LegionKoiConfig(FullNodeConfig):
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    server: ServerConfig = ServerConfig(port=8100)
    koi_net: KoiNetConfig = KoiNetConfig(
        node_name="legion-koi",
        node_profile=FullNodeProfile(
            provides=NodeProvides(
                event=[LegionJournal, LegionVenture, LegionRecording, LegionSession],
                state=[LegionJournal, LegionVenture, LegionSession],
            ),
        ),
        cache_directory_path=Path(".rid_cache"),
    )
