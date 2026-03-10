"""Legion KOI-net node entry point."""

from pathlib import Path

import structlog
from rid_lib.ext import Bundle

from .node import LegionKoiNode
from .sensors.journal_sensor import JournalSensor

log = structlog.stdlib.get_logger()


def main():
    node = LegionKoiNode()

    # Write own identity to cache
    identity_bundle = Bundle.generate(
        rid=node.identity.rid,
        contents=node.identity.profile.model_dump(),
    )
    node.cache.write(identity_bundle)

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
