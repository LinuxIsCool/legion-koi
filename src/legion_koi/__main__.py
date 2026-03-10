"""Legion KOI-net node entry point."""

import time
from pathlib import Path

import structlog

from .node import LegionKoiNode
from .sensors.journal_sensor import JournalSensor

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


def main():
    node = LegionKoiNode()

    # Ensure identity is set before node.start() triggers ProfileMonitor
    _ensure_identity(node)

    # Set up journal sensor
    sensor_config = node.config.sensors
    journal_sensor = JournalSensor(
        watch_dir=Path(sensor_config.journal_watch_dir).expanduser(),
        state_path=Path(sensor_config.journal_state_path),
        kobj_push=node.kobj_queue.push,
    )

    # Initial scan — ingest all existing journal entries
    log.info("scan.starting", sensor="journal")
    bundles = journal_sensor.scan_all()
    for bundle in bundles:
        node.kobj_queue.push(bundle=bundle)
    log.info("scan.complete", sensor="journal", count=len(bundles))

    # Start filesystem watcher
    journal_sensor.start()

    try:
        node.run()
    finally:
        journal_sensor.stop()


if __name__ == "__main__":
    main()
