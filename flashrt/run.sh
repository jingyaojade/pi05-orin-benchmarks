#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=${FLASHRT_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin-flashrt}
baseline=${OPENPI_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin}
precision=${1:-bf16}
case "$precision" in bf16|int8) ;; *) echo 'precision must be bf16 or int8' >&2; exit 2;; esac
output=${2:-reports/flashrt-${precision}-2026-10-16}
fixture=${BASELINE_FIXTURE:-reports/baseline-2026-10-16/fixed_inputs.npz}
export PATH="$root/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export FLASHRT_PALIGEMMA_TOKENIZER="$baseline/cache/big_vision/paligemma_tokenizer.model"
mkdir -p logs
was_active=$(systemctl is-active openpi-pi05 || true)
restore_service() {
  if [[ "$was_active" == active ]]; then sudo -n systemctl start openpi-pi05; fi
}
trap restore_service EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sudo -n systemctl stop openpi-pi05
{
  date -Iseconds
  cat /proc/loadavg
  free -m
  ps -eo user,pid,pcpu,pmem,etime,comm --sort=-pcpu | head -25 || true
  sudo -n jetson_clocks --show
  sudo -n nvpmodel -q
} > "logs/flashrt-${precision}-environment-before.txt" 2>&1
python flashrt/benchmark.py \
  --flashrt-root "$root/vendor/FlashRT" \
  --checkpoint "$baseline/models/pi05_libero" \
  --fixture "$fixture" --precision "$precision" \
  --warmup 10 --iterations 30 --output "$output"
cp "logs/flashrt-${precision}-environment-before.txt" "$output/environment-before.txt"
cp "$root/requirements-built.txt" "$output/requirements-built.txt"
cp "$root/cutlass-commit.txt" "$output/cutlass-commit.txt"
cp "$root/extension-sha256.txt" "$output/extension-sha256.txt"
echo FLASHRT_BENCHMARK_COMPLETE
