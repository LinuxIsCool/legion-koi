"""Legion KOI-net node entry point."""

import json
from pathlib import Path

import structlog

from . import handlers
from .node import LegionKoiNode
from .sensors.journal_sensor import JournalSensor
from .sensors.venture_sensor import VentureSensor
from .sensors.logging_sensor import LoggingSensor
from .sensors.recording_sensor import RecordingSensor
from .sensors.transcript_sensor import TranscriptSensor
from .sensors.message_sensor import MessageSensor
from .sensors.message_filter import MessageFilter
from .sensors.plan_sensor import PlanSensor
from .sensors.research_sensor import ResearchSensor
from .sensors.contact_sensor import ContactSensor
from .sensors.backlog_sensor import BacklogSensor
from .sensors.browser_history_sensor import BrowserHistorySensor
from .sensors.persona_sensor import PersonaSensor
from .sensors.firefox_profiles import discover_profiles
from .storage.postgres import PostgresStorage
from .events.bus import EventBus
from .events.pg_listener import PgListener
from .events.consumers.embed_consumer import EmbedConsumer
from .events.consumers.extract_consumer import ExtractConsumer
log = structlog.stdlib.get_logger()


def _ensure_identity(node) -> None:
    """Ensure node_rid is set from the loaded private key.

    koi-net's SecureManager sets node_rid during key *creation* but not
    when loading an existing PEM with a fresh config.yaml. This fills the gap.
    """
    if node.identity.rid is not None:
        return

    pub_key = node.secure_manager.priv_key.public_key()
    node.config.koi_net.node_rid = pub_key.to_node_rid(
        name=node.config.koi_net.node_name
    )
    if not node.config.koi_net.node_profile.public_key:
        node.config.koi_net.node_profile.public_key = pub_key.to_der()
    node.config.save_to_yaml()
    log.info("identity.derived_from_existing_key", rid=str(node.identity.rid))


def _backfill_postgres(storage: PostgresStorage, cache_dir: Path) -> None:
    """Backfill PostgreSQL from rid_cache for bundles cached before storage existed."""
    existing = storage.get_stats()
    if sum(existing.values()) > 0:
        log.info("postgres.backfill_skipped", msg="Already has data", stats=existing)
        return

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return

    count = 0
    for f in cache_path.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            manifest = data.get("manifest", {})
            contents = data.get("contents", {})
            rid_str = manifest.get("rid", "")
            # Extract namespace from RID string (orn:namespace:reference)
            parts = rid_str.split(":", 2)
            if len(parts) < 3:
                continue
            namespace = parts[1]
            reference = parts[2]
            sha256_hash = manifest.get("sha256_hash", "")
            storage.upsert_bundle(
                rid=rid_str,
                namespace=namespace,
                reference=reference,
                contents=contents if isinstance(contents, dict) else {"raw": contents},
                sha256_hash=sha256_hash,
            )
            count += 1
        except Exception:
            log.exception("postgres.backfill_error", file=str(f))

    log.info("postgres.backfill_complete", count=count)


def main():
    node = LegionKoiNode()
    _ensure_identity(node)
    cfg = node.config.sensors

    # PostgreSQL (optional — skip gracefully if unavailable)
    storage = None
    try:
        storage = PostgresStorage(dsn=node.config.postgres.dsn)
        storage.initialize()
        handlers._postgres_storage = storage
        log.info("postgres.connected")
    except Exception:
        log.warning("postgres.unavailable", msg="Running without persistent storage")

    # Filesystem sensors
    journal_sensor = JournalSensor(
        watch_dir=Path(cfg.journal_watch_dir).expanduser(),
        state_path=Path(cfg.journal_state_path),
        kobj_push=node.kobj_queue.push,
    )
    venture_sensor = VentureSensor(
        watch_dir=Path(cfg.venture_watch_dir).expanduser(),
        state_path=Path(cfg.venture_state_path),
        kobj_push=node.kobj_queue.push,
    )
    plan_sensor = PlanSensor(
        watch_dir=Path(cfg.plan_watch_dir).expanduser(),
        state_path=Path(cfg.plan_state_path),
        kobj_push=node.kobj_queue.push,
    )
    research_sensor = ResearchSensor(
        watch_dir=Path(cfg.research_watch_dir).expanduser(),
        state_path=Path(cfg.research_state_path),
        kobj_push=node.kobj_queue.push,
    )

    # Database sensors
    logging_sensor = LoggingSensor(
        db_path=Path(cfg.logging_db_path).expanduser(),
        state_path=Path(cfg.logging_state_path),
        kobj_push=node.kobj_queue.push,
        poll_interval=cfg.logging_poll_interval,
    )
    recording_sensor = RecordingSensor(
        db_path=Path(cfg.recording_db_path).expanduser(),
        state_path=Path(cfg.recording_state_path),
        kobj_push=node.kobj_queue.push,
        poll_interval=cfg.recording_poll_interval,
    )
    transcript_sensor = TranscriptSensor(
        db_path=Path(cfg.transcript_db_path).expanduser(),
        state_path=Path(cfg.transcript_state_path),
        kobj_push=node.kobj_queue.push,
        poll_interval=cfg.transcript_poll_interval,
    )
    message_filter = MessageFilter(
        messages_db_path=Path(cfg.message_db_path).expanduser(),
        self_sender_ids=cfg.message_self_sender_ids,
        thread_includes=cfg.message_thread_includes,
        thread_excludes=cfg.message_thread_excludes,
        enable=cfg.message_enable_filtering,
    )
    message_sensor = MessageSensor(
        message_filter=message_filter,
        db_path=Path(cfg.message_db_path).expanduser(),
        state_path=Path(cfg.message_state_path),
        kobj_push=node.kobj_queue.push,
        poll_interval=cfg.message_poll_interval,
    )
    contact_sensor = ContactSensor(
        db_path=Path(cfg.contact_db_path).expanduser(),
        state_path=Path(cfg.contact_state_path),
        kobj_push=node.kobj_queue.push,
        poll_interval=cfg.contact_poll_interval,
    )
    backlog_sensor = BacklogSensor(
        watch_dir=Path(cfg.backlog_watch_dir).expanduser(),
        state_path=Path(cfg.backlog_state_path),
        kobj_push=node.kobj_queue.push,
    )

    # Persona sensors — one per configured slug
    persona_sensors = []
    for slug in cfg.persona_slugs:
        watch_dir = Path(cfg.persona_data_base_dir).expanduser() / slug
        state_path = Path(cfg.persona_state_dir) / f"persona_{slug}_state.json"
        ps = PersonaSensor(
            slug=slug,
            watch_dir=watch_dir,
            state_path=state_path,
            kobj_push=node.kobj_queue.push,
        )
        persona_sensors.append(ps)
        log.info("persona_sensor.configured", slug=slug, watch_dir=str(watch_dir))

    # Browser history sensor
    browser_history_sensor = None
    if cfg.browser_history_enabled:
        profiles = discover_profiles(
            firefox_dir=Path(cfg.browser_history_firefox_dir).expanduser(),
            machine_name=cfg.browser_history_machine_name,
        )
        if profiles:
            browser_history_sensor = BrowserHistorySensor(
                profiles=profiles,
                state_path=Path(cfg.browser_history_state_path),
                kobj_push=node.kobj_queue.push,
                poll_interval=cfg.browser_history_poll_interval,
                batch_size=cfg.browser_history_batch_size,
                suppression_path=Path(cfg.browser_history_suppression_path).expanduser(),
                param_policy_path=Path(cfg.browser_history_param_policy_path).expanduser(),
            )
            log.info("browser_history.configured", profiles=[p.slug for p in profiles])
        else:
            log.warning("browser_history.no_profiles", firefox_dir=cfg.browser_history_firefox_dir)

    # Backfill PostgreSQL from rid_cache (bundles cached before PostgreSQL was added)
    if storage:
        _backfill_postgres(storage, node.config.koi_net.cache_directory_path)

    # Event system (Phase 1) — PG NOTIFY → Redis Streams → consumers
    event_bus = None
    pg_listener = None
    embed_consumer = None
    extract_consumer = None
    if storage:
        try:
            event_bus = EventBus()
            if event_bus.ping():
                pg_listener = PgListener(dsn=node.config.postgres.dsn, bus=event_bus)
                pg_listener.start()

                embed_consumer = EmbedConsumer(bus=event_bus, storage=storage)
                embed_consumer.start()

                extract_consumer = ExtractConsumer(bus=event_bus, storage=storage)
                extract_consumer.start()

                log.info("events.started", consumers=["embed", "extract"])
            else:
                log.warning("events.redis_unavailable", msg="Event system disabled — Redis not reachable")
                event_bus = None
        except Exception:
            log.warning("events.startup_error", exc_info=True, msg="Event system disabled")
            event_bus = None

    # Initial scans
    all_sensors = [journal_sensor, venture_sensor, plan_sensor, research_sensor, backlog_sensor, logging_sensor, recording_sensor, transcript_sensor, message_sensor, contact_sensor] + persona_sensors
    if browser_history_sensor:
        all_sensors.append(browser_history_sensor)
    for sensor in all_sensors:
        bundles = sensor.scan_all()
        for bundle in bundles:
            node.kobj_queue.push(bundle=bundle)
        log.info("scan.complete", sensor=sensor.__class__.__name__, count=len(bundles))

    # Start live monitoring
    for sensor in all_sensors:
        sensor.start()

    try:
        node.run()
    finally:
        # Shut down event consumers first (they depend on storage)
        if extract_consumer:
            extract_consumer.stop()
        if embed_consumer:
            embed_consumer.stop()
        if pg_listener:
            pg_listener.stop()
        if event_bus:
            event_bus.close()

        for sensor in all_sensors:
            sensor.stop()
        if storage:
            storage.close()


if __name__ == "__main__":
    main()
