#!/usr/bin/env bash
# Collect Firefox places.sqlite from a fleet machine via Tailscale SSH
# Usage: ./collect_firefox_profiles.sh <machine_name> <tailscale_ip>
#
# Copies all places.sqlite files found on the remote machine to
# ~/staging/firefox-backfill/<machine_name>/ and prints a manifest snippet.

set -euo pipefail
MACHINE=${1:?Usage: collect_firefox_profiles.sh <machine_name> <tailscale_ip>}
IP=${2:?Usage: collect_firefox_profiles.sh <machine_name> <tailscale_ip>}
STAGING="$HOME/staging/firefox-backfill/$MACHINE"

mkdir -p "$STAGING"

# Common Firefox profile locations
for dir in ".mozilla/firefox" ".config/mozilla/firefox" "snap/firefox/common/.mozilla/firefox"; do
  echo "Checking $dir on $MACHINE..."
  if ! ssh "$IP" "test -d ~/$dir" 2>/dev/null; then
    echo "  Not found — skipping."
    continue
  fi

  # Use process substitution to avoid subshell; preserves exit codes
  while IFS= read -r remote_path; do
    profile_dir=$(dirname "$remote_path" | xargs basename)
    local_dir="$STAGING/$profile_dir"
    mkdir -p "$local_dir"
    echo "  Copying $remote_path → $local_dir/"
    if scp "$IP:$remote_path" "$local_dir/places.sqlite"; then
      echo "    OK"
    else
      echo "    FAILED — skipping this profile"
    fi
  done < <(ssh "$IP" "find ~/$dir -name places.sqlite -type f" 2>/dev/null)
done

echo ""
echo "# Manifest snippet for $MACHINE:"
echo "  - machine_name: $MACHINE"
echo "    profile_name: default-release  # adjust if needed"
echo "    snapshots:"
for db in "$STAGING"/*/places.sqlite; do
  [ -f "$db" ] || continue
  profile=$(basename "$(dirname "$db")")
  echo "      - path: $db"
  echo "        label: $profile"
done
