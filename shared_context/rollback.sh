#!/usr/bin/env bash
set -euo pipefail
root=${SHARED_ROOT:-/home/nvidia/pi05-orin-shared-context}
sudo -n systemctl stop openpi-vla-vlm-shared
sudo -n systemctl disable openpi-vla-vlm-shared
python3 - "$root/runtime/previous-services.json" <<'PY'
import json,subprocess,sys
p=json.load(open(sys.argv[1]))
for unit,state in p.items():
 if state['is-enabled']=='enabled':subprocess.run(['sudo','-n','systemctl','enable',unit],check=True)
 elif state['is-enabled']=='disabled':subprocess.run(['sudo','-n','systemctl','disable',unit],check=True)
 if state['is-active'] in ['active','activating']:
  r=subprocess.run(['sudo','-n','systemctl','start',unit])
  if r.returncode:
   reason=subprocess.check_output(['systemctl','show',unit,'-p','Result','--value'],text=True).strip()
   if reason=='start-limit-hit':
    subprocess.run(['sudo','-n','systemctl','reset-failed',unit],check=True)
    subprocess.run(['sudo','-n','systemctl','start',unit],check=True)
   else:raise RuntimeError('Restore failed: '+unit)
PY
