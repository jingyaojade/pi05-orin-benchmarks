#!/usr/bin/env bash
# Preparation for a NEW, compatible machine/stack. Not executed on this Orin.
# This is the upstream standard pi05_base recipe, not the LIBERO benchmark.
set -euo pipefail
: "${VLLM_OMNI_ROOT:?Set the checked-out vLLM-Omni source directory}"
: "${MODEL_DIR:?Set a local standard LeRobot pi05_base checkpoint directory}"
python - "$MODEL_DIR" <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys
from packaging.version import Version
if sys.version_info < (3, 12):
    raise SystemExit('The upstream recipe was validated with Python 3.12+; use a new compatible environment.')
for name in ('vllm', 'vllm-omni'):
    print(name, md.version(name))
if Version(Version(md.version('torch')).base_version) < Version('2.13.0'):
    raise SystemExit('Use the official matching vLLM / Torch binary stack, not the baseline Orin environment.')
config = json.loads((Path(sys.argv[1])/'config.json').read_text())
if config.get('chunk_size') != 50 or config.get('max_action_dim', 32) != 32:
    raise SystemExit('This script is only the upstream standard pi05_base 50x32 recipe. LIBERO needs a verified adapter.')
print('STANDARD_RECIPE_ONLY: this does not benchmark the LIBERO 10x7 policy.')
PY
cd "$VLLM_OMNI_ROOT"
test -f vllm_omni/deploy/pi05.yaml
exec vllm serve "$MODEL_DIR" --omni --host 127.0.0.1 --port "${PORT:-8001}" \
  --deploy-config vllm_omni/deploy/pi05.yaml --enforce-eager --disable-log-stats
