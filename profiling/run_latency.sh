#!/usr/bin/env bash
# One service stop/start for the complete unchanged latency round.
set -euo pipefail
cd "$(dirname "$0")/.."
deployment=${OPENPI_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin}
flashrt=${FLASHRT_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin-flashrt}
group=${1:-reports/new-engine-comparison}
export PATH="$flashrt/.venv/bin:/usr/local/cuda/bin:$PATH"
export JAX_PLATFORMS=cpu OPENPI_DATA_HOME="$deployment/cache"
export OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
was_active=$(systemctl is-active openpi-pi05 || true)
restore() {
  if [[ "$was_active" == active || "$was_active" == activating ]]; then
    if ! sudo -n systemctl start openpi-pi05; then
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
"$deployment/.venv/bin/python" baseline/benchmark.py \
  --deployment-root "$deployment" --warmup 10 --iterations 30 \
  --output "$group/baseline-latency"
for precision in bf16 int8; do
  "$flashrt/.venv/bin/python" flashrt/benchmark.py \
    --flashrt-root "$flashrt/vendor/FlashRT" \
    --checkpoint "$deployment/models/pi05_libero" \
    --fixture "$group/baseline-latency/fixed_inputs.npz" \
    --precision "$precision" --warmup 10 --iterations 30 \
    --output "$group/flashrt-$precision-latency"
done
echo "LATENCY_ROUND_COMPLETE $(date -Iseconds)"
