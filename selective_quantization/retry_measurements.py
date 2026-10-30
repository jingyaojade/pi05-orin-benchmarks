"""After the main suite, retry incomplete telemetry in new, immutable directories."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent-pid',type=int,required=True)
    p.add_argument('--extra-configs',type=Path);a=p.parse_args()
    root=a.root;results=root/'results';code=Path(__file__).resolve().parent
    while True:
        try:os.kill(a.parent_pid,0)
        except ProcessLookupError:break
        # A completed controller can briefly survive as a zombie.
        stat=Path(f'/proc/{a.parent_pid}/stat')
        try:
            if stat.read_text().split()[2]=='Z':break
        except FileNotFoundError:break
        time.sleep(5)
    main_provenance=json.loads((results/'provenance.json').read_text())
    if not main_provenance.get('completed'):
        print('Main suite did not complete; retries not started.',flush=True);return
    pending=[]
    for entry in main_provenance['runs']:
        name=entry['name'];folder=results/name
        cfg=root/'configs'/(name+'.json')
        if not cfg.is_file():continue
        result=folder/'result.json'
        if result.is_file():
            data=json.loads(result.read_text())
            if data.get('sustained',{}).get('telemetry',{}).get('sample_count',0)>=2:continue
        pending.append(name)
    provenance={'started_unix':time.time(),'pending':pending,'runs':[]}
    tasks=[(name+'_retry',name,json.loads((root/'configs'/(name+'.json')).read_text()))
           for name in pending]
    if a.extra_configs:
        for config in json.loads(a.extra_configs.read_text()):
            tasks.append((config['name'],None,config))
    provenance['extra_configurations']=[c for _,original,c in tasks if original is None]
    units=('openpi-pi05','openpi-pi05-flashrt-dual','qwen35-vlm','pi05-qwen-shared')
    state={u:subprocess.run(['systemctl','is-active',u],capture_output=True,text=True).stdout.strip() for u in units}
    provenance['service_state_before']=state
    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    env=dict(os.environ,OMP_NUM_THREADS='6',PYTHONUNBUFFERED='1')
    try:
        active=[u for u,s in state.items() if s in ('active','activating')]
        if active:subprocess.run(['sudo','-n','systemctl','stop',*active],check=True)
        for name,original,config in tasks:
            config=dict(config);config['name']=name
            path=root/'configs'/(name+'.json');write(path,config)
            valid=config.get('valid_num_views',2)
            teacher=results/('bf16_3valid_968' if valid==3 else 'bf16_2valid_968')/'actions.npy'
            command=['/home/nvidia/pi05-orin-flashrt/.venv/bin/python',str(code/'worker.py'),
                '--config',str(path),'--inputs',str(root/'inputs/cases.npz'),
                '--references',str(results/'references'),'--teacher',str(teacher),'--output',str(results/name)]
            print(json.dumps({'event':'retry_start','original':original,'name':name}),flush=True)
            with (results/(name+'.log')).open('w') as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=code.parent)
            provenance['runs'].append({'name':name,'original':original,'returncode':process.returncode})
            write(results/'retry_provenance.json',provenance)
            print(json.dumps({'event':'retry_finish','name':name,'returncode':process.returncode}),flush=True)
        provenance['completed']=True
    finally:
        provenance['restore']={}
        for unit,state_value in state.items():
            if state_value in ('active','activating'):
                r=subprocess.run(['sudo','-n','systemctl','start',unit],capture_output=True,text=True)
                provenance['restore'][unit]={'returncode':r.returncode,'stderr':r.stderr}
        provenance['service_state_after']={u:subprocess.run(['systemctl','is-active',u],
            capture_output=True,text=True).stdout.strip() for u in units}
        provenance['finished_unix']=time.time();write(results/'retry_provenance.json',provenance)


if __name__=='__main__':main()
