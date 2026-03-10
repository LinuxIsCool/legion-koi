"""Quick test script for the Legion KOI-net node.

Usage: Start node in one terminal, run this in another:
    uv run python test_node.py
"""

from legion_koi.rid_types import LegionJournal
from koi_net.protocol.api.models import FetchRids, FetchBundles
from koi_net.protocol.envelope import UnsignedEnvelope
from koi_net.protocol.secure import PrivateKey
from koi_net.config.env_config import EnvConfig
from rid_lib.types import KoiNetNode
from pathlib import Path
import httpx
import json

BASE_URL = "http://127.0.0.1:8100/koi-net"

# Load the node's private key from PEM
env = EnvConfig()
priv_key = PrivateKey.from_pem(Path("priv_key.pem").read_text(), env.priv_key_password)

# Read config to get our node RID
from ruamel.yaml import YAML
yaml = YAML()
with open("config.yaml") as f:
    config_data = yaml.load(f)

node_rid_str = config_data["koi_net"]["node_rid"]
# node_rid is like "orn:koi-net.node:legion-koi+hash" — parse as RID
our_rid = KoiNetNode.from_reference(node_rid_str.split(":", 2)[-1])

def make_request(endpoint: str, payload):
    """Sign and send a request."""
    envelope = UnsignedEnvelope(
        payload=payload,
        source_node=our_rid,
        target_node=our_rid,
    ).sign_with(priv_key)

    resp = httpx.post(
        f"{BASE_URL}/{endpoint}",
        content=envelope.model_dump_json(),
        headers={"Content-Type": "application/json"},
    )
    return resp


print("=== Fetch RIDs (all types) ===")
resp = make_request("rids/fetch", FetchRids())
data = resp.json()
if "payload" in data:
    rids = data["payload"]["rids"]
    print(f"Found {len(rids)} RIDs")
    for rid in rids[:5]:
        print(f"  {rid}")
    if len(rids) > 5:
        print(f"  ... and {len(rids) - 5} more")
else:
    print(f"Response: {json.dumps(data, indent=2)[:500]}")

print()
print("=== Fetch RIDs (journal only) ===")
resp = make_request("rids/fetch", FetchRids(rid_types=[LegionJournal]))
data = resp.json()
if "payload" in data:
    rids = data["payload"]["rids"]
    print(f"Found {len(rids)} journal RIDs")
    for rid in rids[:5]:
        print(f"  {rid}")
    if len(rids) > 5:
        print(f"  ... and {len(rids) - 5} more")
else:
    print(f"Response: {json.dumps(data, indent=2)[:500]}")

print()
print("=== Fetch Bundle (KOI insight entry) ===")
test_rid = "orn:legion.journal:2026-03-10/1225-the-plugin-koi-insight"
resp = make_request("bundles/fetch", FetchBundles(rids=[test_rid]))
data = resp.json()
if "payload" in data:
    bundles = data["payload"]["bundles"]
    if bundles:
        b = bundles[0]
        print(f"RID: {b['manifest']['rid']}")
        print(f"Title: {b['contents']['frontmatter']['title'][:80]}")
        print(f"Body: {b['contents']['body'][:200]}...")
        print(f"Hash: {b['manifest']['sha256_hash'][:16]}...")
    else:
        print(f"Not found. not_found: {data['payload'].get('not_found', [])}")
else:
    print(f"Response: {json.dumps(data, indent=2)[:500]}")
