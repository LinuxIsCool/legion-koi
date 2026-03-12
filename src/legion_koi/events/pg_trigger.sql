-- PostgreSQL trigger: emit NOTIFY on bundle INSERT/UPDATE
-- Applied by storage/postgres.py:initialize()
-- Listener (pg_listener.py) bridges these to Redis Streams

CREATE OR REPLACE FUNCTION notify_bundle_change() RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify('koi_events', json_build_object(
        'op', TG_OP,
        'rid', NEW.rid,
        'namespace', NEW.namespace
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop and recreate to ensure function signature is current
DROP TRIGGER IF EXISTS bundles_notify ON bundles;
CREATE TRIGGER bundles_notify
    AFTER INSERT OR UPDATE ON bundles
    FOR EACH ROW EXECUTE FUNCTION notify_bundle_change();
