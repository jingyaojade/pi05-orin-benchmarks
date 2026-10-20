#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=6 JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
exec /home/nvidia/pi05-orin-flashrt/.venv/bin/python "$repo/dual_stack/vla_server.py" \
  --flashrt-root /home/nvidia/pi05-orin-flashrt/vendor/FlashRT \
  --checkpoint /home/nvidia/pi05-orin/models/pi05_libero \
  --fixture "$repo/reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz" \
  --host 127.0.0.1 --port "${VLA_PORT:-8081}" --websocket-port "${VLA_WEBSOCKET_PORT:-8000}"
