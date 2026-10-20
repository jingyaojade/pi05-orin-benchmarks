#!/usr/bin/env bash
# Restore the recorded pre-deployment pi05 service state; leave sources/models intact.
set -euo pipefail
root=${DUAL_ROOT:-/home/nvidia/pi05-orin-dual}
state="$root/runtime/original-service-state.json"
test -f "$state"
sudo -n systemctl stop openpi-pi05-flashrt-dual qwen35-vlm || true
sudo -n systemctl disable openpi-pi05-flashrt-dual qwen35-vlm || true
enabled=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["enabled"])' "$state")
active=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["active"])' "$state")
if [[ "$enabled" == enabled ]]; then sudo -n systemctl enable openpi-pi05; fi
if [[ "$active" == active || "$active" == activating ]]; then
  sudo -n systemctl reset-failed openpi-pi05
  sudo -n systemctl start openpi-pi05
fi
