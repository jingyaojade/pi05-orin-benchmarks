#!/usr/bin/env bash
set -euo pipefail
root=${FLASHRT_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin-flashrt}
baseline=${OPENPI_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin}
bench=${BENCHMARK_ROOT:-/home/nvidia/pi05-orin-benchmarks}
export PATH="$root/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export UV_LINK_MODE=copy
export UV_SKIP_WHEEL_FILENAME_CHECK=1
mkdir -p "$root/vendor" "$root/logs"
if [[ ! -x "$root/.venv/bin/python" ]]; then
  /home/nvidia/.local/bin/uv venv --python /usr/bin/python3.10 "$root/.venv"
fi
/home/nvidia/.local/bin/uv pip install --python "$root/.venv/bin/python" \
  "$baseline/downloads/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl" \
  'numpy==1.26.4' 'ml-dtypes==0.4.1' 'cmake>=3.24,<4' ninja pybind11 pyyaml safetensors sentencepiece Pillow
source_root="$root/vendor/FlashRT"
if [[ ! -d "$source_root/.git" ]]; then
  git init "$source_root"
  git -C "$source_root" remote add origin https://github.com/flashrt-project/FlashRT.git
  git -C "$source_root" fetch --depth 1 origin 6cb2f80caec2f561a1331c554a6d3bbf95d2ed55
  git -C "$source_root" checkout --detach FETCH_HEAD
fi
test "$(git -C "$source_root" rev-parse HEAD)" = 6cb2f80caec2f561a1331c554a6d3bbf95d2ed55
if [[ ! -d "$source_root/third_party/cutlass/.git" ]]; then
  git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git "$source_root/third_party/cutlass"
fi
if PYTHONPATH="$bench/flashrt" python -c 'from adapter import verify_patched_source; from pathlib import Path; import sys; verify_patched_source(Path(sys.argv[1]))' "$source_root" 2>/dev/null; then
  echo 'Verified existing pinned FlashRT patch'
else
  python "$bench/flashrt/prepare_patch.py" --flashrt-root "$source_root" --apply
fi
/home/nvidia/.local/bin/uv pip install --python "$root/.venv/bin/python" --no-deps -e "$source_root"
python -c 'import torch; print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
cmake -S "$source_root" -B "$source_root/build_orin_sm87" -G Ninja \
  -DGPU_ARCH=87 -DFA2_ARCH_NATIVE_ONLY=ON \
  '-DFA2_HDIMS=96;128;256' -DFA2_DTYPES=bf16 \
  -DFLASHRT_SLIM_BUILD=ON \
  -DFLASHRT_ENABLE_MOTUS=OFF -DFLASHRT_ENABLE_LINGBOT=OFF \
  -DFLASHRT_ENABLE_AUDIO_CODEBOOK=OFF -DFLASHRT_BUILD_QWEN3_VL=OFF \
  -DFLASHRT_ENABLE_NATIVE_CPP=OFF \
  -DPython3_EXECUTABLE="$root/.venv/bin/python"
cmake --build "$source_root/build_orin_sm87" --target flash_rt_kernels flash_rt_fa2 -j"${FLASHRT_BUILD_JOBS:-2}"
/home/nvidia/.local/bin/uv pip freeze --python "$root/.venv/bin/python" > "$root/requirements-built.txt"
git -C "$source_root/third_party/cutlass" rev-parse HEAD > "$root/cutlass-commit.txt"
sha256sum "$source_root"/flash_rt/*.so > "$root/extension-sha256.txt"
date -Iseconds
echo FLASHRT_BUILD_COMPLETE
