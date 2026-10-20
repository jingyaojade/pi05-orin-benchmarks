#!/usr/bin/env bash
# Switch from the preserved original baseline service to the tested dual services.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root=${DUAL_ROOT:-/home/nvidia/pi05-orin-dual}
result=${1:?Pass a successfully completed run_suite.py result directory}
python3 - "$result/provenance.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p.get('completed') is True and not p.get('errors'), 'Only install after a completed GPU test'
PY
mkdir -p "$root/runtime"
if [[ ! -f "$root/runtime/original-service-state.json" ]]; then
  python3 - "$root/runtime/original-service-state.json" <<'PY'
import subprocess,json,sys
def get(kind):return subprocess.run(['systemctl',kind,'openpi-pi05'],capture_output=True,text=True).stdout.strip()
json.dump({'enabled':get('is-enabled'),'active':get('is-active')},open(sys.argv[1],'w'),indent=2)
PY
fi
for unit in openpi-pi05-flashrt-dual qwen35-vlm; do
  path="/etc/systemd/system/$unit.service"
  if [[ -e "$path" ]] && ! grep -q 'Managed by pi05-orin-benchmarks/dual_stack' "$path"; then
    echo "Refusing to replace an unrelated unit: $path" >&2; exit 1
  fi
done
for pair in 'openpi-pi05-flashrt-dual run_vla.sh' 'qwen35-vlm run_vlm.sh'; do
  read -r unit script <<< "$pair"
  cat > "$root/runtime/$unit.service" <<EOF
# Managed by pi05-orin-benchmarks/dual_stack
[Unit]
Description=Orin dual stack $unit
After=network.target
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
User=nvidia
WorkingDirectory=$repo
Environment=DUAL_ROOT=$root
Environment=OMP_NUM_THREADS=6
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/bash $repo/dual_stack/$script
Restart=on-failure
RestartSec=10
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
  sudo -n install -m 644 "$root/runtime/$unit.service" "/etc/systemd/system/$unit.service"
done
trap 'bash "$repo/dual_stack/restore_original.sh"' ERR
sudo -n systemctl daemon-reload
sudo -n systemctl stop openpi-pi05
sudo -n systemctl disable openpi-pi05
sudo -n systemctl enable --now openpi-pi05-flashrt-dual qwen35-vlm
python3 - <<'PY'
import json,time,urllib.request,subprocess
pending={8081:'openpi-pi05-flashrt-dual',8082:'qwen35-vlm'}
deadline=time.monotonic()+900
while pending and time.monotonic()<deadline:
 for port,unit in list(pending.items()):
  if subprocess.run(['systemctl','is-failed','--quiet',unit]).returncode==0:
   raise RuntimeError(unit+' failed; examine journal')
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:json.load(r)
   pending.pop(port)
  except Exception:pass
 time.sleep(1)
if pending:raise TimeoutError(str(pending))
print('DUAL_SERVICES_HEALTHY')
PY
