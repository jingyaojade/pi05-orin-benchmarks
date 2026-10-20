#!/usr/bin/env bash
# Explicit candidate activation after a completed evaluation; original units retained.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root=${SHARED_ROOT:-/home/nvidia/pi05-orin-shared-context}
result=${1:?Pass the completed shared-context result directory}
python3 - "$repo" "$root" "$result" <<'PY'
import hashlib,json,sys
from pathlib import Path
repo,root,result=map(Path,sys.argv[1:])
p=json.loads((result/'provenance.json').read_text())
assert p.get('completed') is True
assert json.loads((result/'evidence/context-validation.json').read_text())['passed']
assert hashlib.sha256((root/'build/libshared_context.so').read_bytes()).hexdigest()==p['bridge_sha256']
for name in ['service.py','bridge.cpp']:
 assert hashlib.sha256((repo/'shared_context'/name).read_bytes()).hexdigest()==p['script_sha256']['shared_context/'+name]
PY
mkdir -p "$root/runtime"
if [[ ! -f "$root/runtime/previous-services.json" ]]; then
  python3 - "$repo" "$root/runtime/previous-services.json" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'shared_context'))
from support import states
Path(sys.argv[2]).write_text(json.dumps(states(),indent=2)+'\n')
PY
fi
unit=/etc/systemd/system/openpi-vla-vlm-shared.service
if [[ -e "$unit" ]] && ! grep -q 'Managed by pi05-orin-benchmarks/shared_context' "$unit"; then
  echo 'Refusing to replace an unrelated unit' >&2; exit 1
fi
cat > "$root/runtime/openpi-vla-vlm-shared.service" <<EOF
# Managed by pi05-orin-benchmarks/shared_context
[Unit]
Description=pi05 and Qwen3.5 in one Python PID and CUDA context
After=network.target
StartLimitIntervalSec=600
StartLimitBurst=5
[Service]
Type=simple
User=nvidia
WorkingDirectory=$repo
ExecStart=/usr/bin/bash $repo/shared_context/run_service.sh
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=10
TimeoutStopSec=35
[Install]
WantedBy=multi-user.target
EOF
sudo -n install -m644 "$root/runtime/openpi-vla-vlm-shared.service" "$unit"
trap 'bash "$repo/shared_context/rollback.sh"' ERR
sudo -n systemctl daemon-reload
sudo -n systemctl stop openpi-pi05-flashrt-dual qwen35-vlm openpi-pi05
sudo -n systemctl disable openpi-pi05-flashrt-dual qwen35-vlm openpi-pi05
sudo -n systemctl enable --now openpi-vla-vlm-shared
python3 - <<'PY'
import json,time,urllib.request,subprocess
deadline=time.monotonic()+900
while time.monotonic()<deadline:
 if subprocess.run(['systemctl','is-failed','--quiet','openpi-vla-vlm-shared']).returncode==0:raise RuntimeError('Shared unit failed')
 try:
  for port in [8081,8082]:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:json.load(r)
  with urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=2) as r:assert r.status==200
  break
 except Exception:time.sleep(1)
else:raise TimeoutError('Shared readiness')
print('SHARED_SERVICE_HEALTHY')
PY
