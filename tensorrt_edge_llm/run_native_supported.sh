#!/usr/bin/env bash
# PREPARED, NOT EXECUTED on the current Orin. Requires the supported newer SDK.
# Reuses the exact existing checkpoint; never downloads alternative robot weights.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${EDGELLM_WORK_ROOT:-"$HOME/pi05-orin-edge-llm"}
source_root=${EDGELLM_SOURCE_ROOT:-"$root/vendor/TensorRT-Edge-LLM"}
build_root=${EDGELLM_BUILD_ROOT:-"$root/build"}
checkpoint=${CHECKPOINT:-/home/nvidia/pi05-orin/models/pi05_libero}
fixture=${FIXTURE:?Set FIXTURE to the original baseline fixed_inputs.npz}
reference=${REFERENCE_ACTIONS:-"$(dirname -- "$fixture")/reference_actions.npy"}
tokenizer=${TOKENIZER:-/home/nvidia/pi05-orin/cache/big_vision/paligemma_tokenizer.model}
norm_stats=${NORM_STATS:-"$checkpoint/assets/physical-intelligence/libero/norm_stats.json"}
python=${EDGELLM_EXPORT_PYTHON:?Set EDGELLM_EXPORT_PYTHON to an isolated, upstream-compatible export Python}
input_python=${INPUT_PYTHON:-/home/nvidia/pi05-orin/.venv/bin/python}
onnx_root=${ONNX_DIR:-"$root/onnx/pi05_libero"}
engine_root=${ENGINE_DIR:-"$root/engines/pi05_libero"}
ref=95515c2f87fba8982db5a519f9022277667b3cc9
mkdir -p "$root/logs" "$root/inputs" "$root/vendor"

# Actual current system fails these guards. Do not bypass by replacing host libs.
/usr/bin/python3 - <<'PY'
import platform
import tensorrt as trt
version = tuple(int(x) for x in trt.__version__.split('.')[:2])
if platform.machine() != 'aarch64' or tuple(int(x) for x in platform.libc_ver()[1].split('.')[:2]) < (2,39):
    raise SystemExit('This prepared workflow targets the official aarch64 glibc>=2.39 platform.')
if version < (10,15):
    raise SystemExit('Native pi05 requires TensorRT>=10.15 (Attention/RotaryEmbedding).')
PY
test -f "$checkpoint/model.safetensors"
test -f "$checkpoint/config.json"
test -f "$norm_stats"
if [[ ! -d "$source_root/.git" ]]; then
  git clone --no-checkout https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$source_root"
fi
git -C "$source_root" fetch origin "$ref"
git -C "$source_root" checkout --detach "$ref"
test "$(git -C "$source_root" rev-parse HEAD)" = "$ref"
git -C "$source_root" submodule update --init --recursive

# Install/export dependencies only in the explicitly supplied isolated Python.
# Upstream v0.11.0 pyproject pins torch 2.13.0 / transformers 5.14.1. The old
# NVIDIA Torch2.3 baseline environment must never be passed here.
"$python" - <<'PY'
import torch,transformers
assert torch.__version__.split('+')[0] == '2.13.0', torch.__version__
assert transformers.__version__ == '5.14.1', transformers.__version__
PY
cmake -S "$source_root" -B "$build_root" \
  -DCMAKE_BUILD_TYPE=Release -DCUDA_CTK_VERSION=13.2 \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE="$source_root/cmake/aarch64_linux_toolchain.cmake" \
  -DEMBEDDED_TARGET=jetson-orin -DBUILD_EXPERIMENTAL_MODELS=ON \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_UNIT_TESTS=OFF 2>&1 | tee "$root/logs/configure.log"
cmake --build "$build_root" --target pi05_policy_build pi05_policy_inference -j2 \
  2>&1 | tee "$root/logs/build.log"
export EDGELLM_PLUGIN_PATH="$build_root/libNvInfer_edgellm_plugin.so"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"
"$python" -m tensorrt_edgellm.scripts.export "$checkpoint" "$onnx_root" \
  --dtype float16 --pi05-policy-config pi05_libero 2>&1 | tee "$root/logs/export.log"
mkdir -p "$onnx_root/assets"
cp -- "$norm_stats" "$onnx_root/assets/norm_stats.json"
sha256sum "$checkpoint/model.safetensors" "$checkpoint/config.json" "$fixture" "$norm_stats" \
  > "$root/logs/input_identity.sha256"
"$build_root/experimental_models/pi05/examples/pi05_policy_build" \
  --onnxDir "$onnx_root" --engineDir "$engine_root" 2>&1 | tee "$root/logs/engine_build.log"
"$input_python" "$script_dir/prepare_inputs.py" --fixture "$fixture" --tokenizer "$tokenizer" --output "$root/inputs"
tokens=$(cat "$root/inputs/token_ids.csv")
"$build_root/experimental_models/pi05/examples/pi05_policy_inference" \
  --engineDir "$engine_root" --pixelValues "$root/inputs/pixel_values.bin" \
  --tokens "$tokens" --noise "$root/inputs/x0.bin" --steps 10 --cudaGraph \
  --warmup 10 --iters 30 --output "$root/inputs/native_actions.safetensors" \
  2>&1 | tee "$root/logs/canonical_inference.log"
"$build_root/experimental_models/pi05/examples/pi05_policy_inference" \
  --engineDir "$engine_root" --inputFile "$root/inputs/observation.json" \
  --noise "$root/inputs/x0.bin" --steps 10 --cudaGraph \
  --warmup 10 --iters 30 --output "$root/inputs/native_robot_actions.json" \
  2>&1 | tee "$root/logs/policy_inference.log"
"$input_python" "$script_dir/compare_actions.py" --actual "$root/inputs/native_actions.safetensors" \
  --reference "$reference" --norm-stats "$norm_stats" --output "$root/inputs/accuracy_canonical.json"
"$input_python" "$script_dir/compare_actions.py" --actual "$root/inputs/native_robot_actions.json" \
  --reference "$reference" --norm-stats "$norm_stats" --output "$root/inputs/accuracy_policy.json"
echo 'Inference ran. Accuracy, timing boundary and memory validation remain required before publishing performance.'
