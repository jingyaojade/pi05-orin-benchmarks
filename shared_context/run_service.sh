#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=6 NO_PROXY=127.0.0.1,localhost PYTHONUNBUFFERED=1
exec /home/nvidia/pi05-orin-flashrt/.venv/bin/python "$repo/shared_context/service.py" \
  --context "${SHARED_VLM_CONTEXT:-8192}" --port "${VLA_PORT:-8081}" \
  --vlm-port "${VLM_PORT:-8082}" --websocket-port "${VLA_WEBSOCKET_PORT:-8000}"
