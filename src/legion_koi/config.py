"""Legion KOI-net node configuration."""

import os
from pathlib import Path

from pydantic import BaseModel, Field
from koi_net.config.full_node import FullNodeConfig, FullNodeProfile
from koi_net.config.koi_net_config import KoiNetConfig
from koi_net.config.server_config import ServerConfig
from koi_net.protocol.node import NodeProvides

from .rid_types import LegionBrowserHistory, LegionContact, LegionJournal, LegionPersona, LegionTask, LegionTranscript, LegionVenture, LegionRecording, LegionSession, LegionMessage, LegionPlan, LegionResearch


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
    transcript_db_path: str = "~/.claude/local/transcripts/transcripts.db"
    transcript_state_path: str = "./state/transcript_state.json"
    transcript_poll_interval: float = 120.0
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
    research_watch_dir: str = "~/legion-brain/local/research/"
    research_state_path: str = "./state/research_state.json"
    contact_db_path: str = "~/.claude/local/messages/messages.db"
    contact_state_path: str = "./state/contact_state.json"
    contact_poll_interval: float = 300.0
    backlog_watch_dir: str = "~/.claude/local/backlog/"
    backlog_state_path: str = "./state/backlog_state.json"
    browser_history_firefox_dir: str = "~/.config/mozilla/firefox"
    browser_history_machine_name: str = "legion"
    browser_history_state_path: str = "./state/browser_history_state.json"
    browser_history_poll_interval: float = 300.0
    browser_history_batch_size: int = 500
    browser_history_enabled: bool = True
    browser_history_suppression_path: str = "~/.config/claude-browser-history/suppressed_domains.txt"
    browser_history_param_policy_path: str = "~/.config/claude-browser-history/param_policy.yaml"
    # Persona sensors — slug-parameterized, one per persona
    persona_slugs: list[str] = ["darren"]
    persona_data_base_dir: str = "~/.claude/local/personas/data/"
    persona_state_dir: str = "./state/"


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
                event=[LegionBrowserHistory, LegionContact, LegionJournal, LegionPersona, LegionTask, LegionTranscript, LegionVenture, LegionRecording, LegionSession, LegionMessage, LegionPlan, LegionResearch],
                state=[LegionBrowserHistory, LegionContact, LegionJournal, LegionPersona, LegionTask, LegionTranscript, LegionVenture, LegionRecording, LegionSession, LegionMessage, LegionPlan, LegionResearch],
            ),
        ),
        cache_directory_path=Path(".rid_cache"),
    )
