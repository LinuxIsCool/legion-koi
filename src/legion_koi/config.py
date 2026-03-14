"""Legion KOI-net node configuration."""

import os
from pathlib import Path

from pydantic import BaseModel, Field
from koi_net.config.full_node import FullNodeConfig, FullNodeProfile
from koi_net.config.koi_net_config import KoiNetConfig
from koi_net.config.server_config import ServerConfig
from koi_net.protocol.node import NodeProvides

from .rid_types import LegionJournal, LegionVenture, LegionRecording, LegionSession, LegionMessage, LegionPlan


class SensorConfig(BaseModel):
    journal_watch_dir: str = "~/legion-brain/local/journal/"
    journal_state_path: str = "./state/journal_state.json"
    venture_watch_dir: str = "~/legion-brain/local/ventures/"
    venture_state_path: str = "./state/venture_state.json"
    logging_db_path: str = "~/.claude/local/logging/db/logging.db"
    logging_state_path: str = "./state/logging_state.json"
    logging_poll_interval: float = 60.0
    recording_db_path: str = "~/.claude/local/recordings/recordings.db"
    recording_state_path: str = "./state/recording_state.json"
    recording_poll_interval: float = 120.0
    message_db_path: str = "~/.claude/local/messages/messages.db"
    message_state_path: str = "./state/message_state.json"
    message_poll_interval: float = 60.0
    # Message filtering — thread-level relevance gating
    message_self_sender_ids: list[str] = ["telegram:user:1441369482"]
    message_thread_includes: list[str] = []   # Lurk group thread_ids to include (T3)
    message_thread_excludes: list[str] = []   # Thread_ids to always exclude
    message_enable_filtering: bool = True      # Master switch
    plan_watch_dir: str = "~/.claude/plans/"
    plan_state_path: str = "./state/plan_state.json"


class PostgresConfig(BaseModel):
    dsn: str = Field(default_factory=lambda: os.environ.get(
        "LEGION_KOI_DSN", "postgresql://localhost/personal_koi"
    ))


class LegionKoiConfig(FullNodeConfig):
    sensors: SensorConfig = Field(default_factory=SensorConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    server: ServerConfig = ServerConfig(port=8100)
    koi_net: KoiNetConfig = KoiNetConfig(
        node_name="legion-koi",
        node_profile=FullNodeProfile(
            provides=NodeProvides(
                event=[LegionJournal, LegionVenture, LegionRecording, LegionSession, LegionMessage, LegionPlan],
                state=[LegionJournal, LegionVenture, LegionRecording, LegionSession, LegionMessage, LegionPlan],
            ),
        ),
        cache_directory_path=Path(".rid_cache"),
    )
