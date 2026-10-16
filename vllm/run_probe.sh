#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python=${OPENPI_PYTHON:-/home/nvidia/pi05-orin/.venv/bin/python}
output=${1:-results/$(date +%Y-%m-%dT%H%M%S)-orin}
"$python" probe.py --output "$output"
