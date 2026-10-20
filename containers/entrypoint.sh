#!/usr/bin/env bash
set -euo pipefail
# Preserve any initialization hooks supplied by the selected NVIDIA base image.
if [[ -x /opt/nvidia/nvidia_entrypoint.sh ]]; then
  exec /opt/nvidia/nvidia_entrypoint.sh python /workspace/benchmarks/containers/runtime.py "$@"
fi
exec python /workspace/benchmarks/containers/runtime.py "$@"
