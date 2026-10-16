#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
deployment_root="${OPENPI_DEPLOYMENT_ROOT:-/home/nvidia/pi05-orin}"
output="${1:-reports/baseline-2026-10-16}"
was_active=0
if systemctl is-active --quiet openpi-pi05; then was_active=1; fi
restore_service() {
  if [ "$was_active" -eq 1 ]; then
    sudo -n systemctl start openpi-pi05
  fi
}
trap restore_service EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
sudo -n systemctl stop openpi-pi05
export JAX_PLATFORMS=cpu
export OPENPI_DATA_HOME="$deployment_root/cache"
export OMP_NUM_THREADS=6
export PYTHONUNBUFFERED=1
"$deployment_root/.venv/bin/python" baseline/benchmark.py \
  --deployment-root "$deployment_root" \
  --iterations 30 --warmup 10 --output "$output"
echo BASELINE_BENCHMARK_COMPLETE
