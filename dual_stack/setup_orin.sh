#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root=${DUAL_ROOT:-/home/nvidia/pi05-orin-dual}
mkdir -p "$root/vendor" "$root/models/Qwen3.5-9B" "$root/logs"
export PATH=/home/nvidia/pi05-orin-flashrt/.venv/bin:/usr/local/cuda/bin:$PATH
pin=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["llama_cpp_commit"])' "$repo/dual_stack/sources.json")
if [[ ! -d "$root/vendor/llama.cpp" ]]; then
  curl --fail --location --retry 5 --connect-timeout 20 \
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$pin" -o "$root/vendor/llama-$pin.tar.gz"
  mkdir "$root/vendor/llama.cpp"
  tar -xzf "$root/vendor/llama-$pin.tar.gz" --strip-components=1 -C "$root/vendor/llama.cpp"
  printf '%s\n' "$pin" > "$root/vendor/llama.cpp/PINNED_COMMIT"
fi
test "$(cat "$root/vendor/llama.cpp/PINNED_COMMIT")" = "$pin"
cmake -S "$root/vendor/llama.cpp" -B "$root/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CUDA_FA_QUANTS=f16-f16 -DGGML_CUDA_NCCL=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
  2>&1 | tee "$root/logs/configure.log"
cmake --build "$root/build" --target llama-server llama-mtmd-cli -j${BUILD_JOBS:-8} \
  2>&1 | tee "$root/logs/build.log"
"$root/build/bin/llama-server" --version > "$root/logs/llama-version.txt" 2>&1
echo BUILD_COMPLETE
