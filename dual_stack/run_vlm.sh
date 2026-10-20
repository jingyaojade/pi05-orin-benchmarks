#!/usr/bin/env bash
set -euo pipefail
root=${DUAL_ROOT:-/home/nvidia/pi05-orin-dual}
exec "$root/build/bin/llama-server" \
  -m "$root/models/Qwen3.5-9B/Qwen3.5-9B-Q5_K_M.gguf" \
  --mmproj "$root/models/Qwen3.5-9B/mmproj-F16.gguf" \
  --host 127.0.0.1 --port "${VLM_PORT:-8082}" \
  -ngl 99 -c 8192 -np 1 -t 4 -tb 4 -b 512 -ub 128 \
  --flash-attn on --jinja --chat-template-kwargs '{"enable_thinking":false}' \
  --no-cache-prompt --metrics --no-webui
