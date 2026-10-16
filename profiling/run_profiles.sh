#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
deployment=${OPENPI_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin}
flashrt=${FLASHRT_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin-flashrt}
group=${1:-reports/engine-comparison-2026-10-16}
fixture="$group/baseline-latency/fixed_inputs.npz"
export PATH="$flashrt/.venv/bin:/usr/local/cuda/bin:$PATH"
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="$deployment/cache"
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
mkdir -p logs
was_active=$(systemctl is-active openpi-pi05 || true)
restore() {
  if [[ "$was_active" == active || "$was_active" == activating ]]; then
    if ! sudo -n systemctl start openpi-pi05; then
      # Repeated intentional benchmark stop/start cycles can exhaust this
      # service's configured three-starts-per-ten-minutes quota.
      if [[ "$(systemctl show openpi-pi05 -p Result --value)" == start-limit-hit ]]; then
        sudo -n systemctl reset-failed openpi-pi05
        sudo -n systemctl start openpi-pi05
      else
        return 1
      fi
    fi
  fi
}
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sudo -n systemctl stop openpi-pi05
echo "MEMORY_ROUND_START $(date -Iseconds)"
"$deployment/.venv/bin/python" profiling/memory.py \
  --backend baseline --deployment-root "$deployment" --fixture "$fixture" \
  --warmup 5 --iterations 3 --output "$group/baseline-memory"
for precision in bf16 int8; do
  "$flashrt/.venv/bin/python" profiling/memory.py \
    --backend flashrt --deployment-root "$deployment" \
    --flashrt-root "$flashrt/vendor/FlashRT" --checkpoint "$deployment/models/pi05_libero" \
    --fixture "$fixture" --precision "$precision" \
    --warmup 5 --iterations 3 --output "$group/flashrt-$precision-memory"
  "$flashrt/.venv/bin/python" profiling/pipeline_latency.py \
    --flashrt-root "$flashrt/vendor/FlashRT" --checkpoint "$deployment/models/pi05_libero" \
    --fixture "$fixture" --precision "$precision" \
    --warmup 10 --iterations 30 --output "$group/flashrt-$precision-pipeline-latency"
done
echo "PROFILE_ROUND_COMPLETE $(date -Iseconds)"
