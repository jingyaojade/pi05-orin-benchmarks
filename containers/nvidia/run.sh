#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
mode=${1:-doctor}
if [[ $# -gt 0 ]]; then shift; fi
image=${IMAGE_TAG:-pi05-qwen:nvidia-candidate}
vla_gpu=${VLA_GPU:-0}; vlm_gpu=${VLM_GPU:-0}
if [[ ! $vla_gpu =~ ^[0-9]+$ || ! $vlm_gpu =~ ^[0-9]+$ ]]; then
  echo 'VLA_GPU and VLM_GPU must be nonnegative CUDA ordinals.' >&2
  exit 1
fi
args=(--rm --gpus all --shm-size 2g -e "VLA_GPU=$vla_gpu" -e "VLM_GPU=$vlm_gpu")
if [[ $mode != doctor ]]; then
  : "${ASSETS_DIR:?Set ASSETS_DIR to the exported pinned model assets}"
  test -d "$ASSETS_DIR"
  results=${RESULTS_DIR:-$repo/container-results}
  mkdir -p "$results"
  args+=(--mount "type=bind,src=$(cd "$ASSETS_DIR" && pwd),dst=/assets,readonly")
  args+=(--mount "type=bind,src=$(cd "$results" && pwd),dst=/results")
fi
if [[ $mode == serve ]]; then
  args+=(-p 127.0.0.1:8081:8081 -p 127.0.0.1:8082:8082 -p 127.0.0.1:8000:8000)
fi
# Private benchmark ports; only our own children are managed.
if [[ $mode == doctor || $mode == verify-assets ]]; then
  exec docker run "${args[@]}" "$image" "$mode" "$@"
else
  output=/results/$(date -u +%Y%m%dT%H%M%SZ)-$mode-gpu$vla_gpu-$vlm_gpu
  exec docker run "${args[@]}" "$image" "$mode" --output "$output" "$@"
fi
