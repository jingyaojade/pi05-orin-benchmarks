#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root=${SHARED_ROOT:-/home/nvidia/pi05-orin-shared-context}
llama=${DUAL_ROOT:-/home/nvidia/pi05-orin-dual}
mkdir -p "$root/build" "$root/logs" "$root/runtime"
test "$(cat "$llama/vendor/llama.cpp/PINNED_COMMIT")" = 3bb63cbcbfb62056c84fe9e521b0f3d756d29c32
g++ -std=c++17 -O2 -fPIC -shared "$repo/shared_context/bridge.cpp" \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
  -L"$llama/build/bin" -lllama-server-impl -lcudart -lcuda -lcupti -pthread \
  -Wl,-rpath,"$llama/build/bin" -Wl,-rpath,/usr/local/cuda/lib64 \
  -o "$root/build/libshared_context.so"
sha256sum "$repo/shared_context/bridge.cpp" "$root/build/libshared_context.so" \
  > "$root/logs/bridge-sha256.txt"
