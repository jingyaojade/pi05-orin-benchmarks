#!/usr/bin/env python3
"""Deploy isolated local VLA/VLM processes and measure a reproducible scenario matrix."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from clients import request_json, vla_request, vlm_request
from telemetry import Monitor
from environment import snapshot


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/home/nvidia/pi05-orin-dual'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase-seconds', type=float, default=60)
    p.add_argument('--concurrent-seconds', type=float, default=180)
    p.add_argument('--idle-seconds', type=float, default=30)
    p.add_argument('--vlm-min-requests', type=int, default=8)
    p.add_argument('--vla-port', type=int, default=8081)
    p.add_argument('--vlm-port', type=int, default=8082)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output must be empty; preserve all earlier measurements')
    out.mkdir(parents=True,exist_ok=True)
    (args.root/'runtime').mkdir(exist_ok=True)
    from PIL import Image
    import numpy as np
    fixture=repo/'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz'
    with np.load(fixture,allow_pickle=False) as f:
        observation={k:f[k] for k in f.files}
    def encoded_image(value):
        stream=io.BytesIO();Image.fromarray(value).save(stream,format='PNG')
        return base64.b64encode(stream.getvalue()).decode()
    vla_body={'image_base64':encoded_image(observation['image']),
              'wrist_image_base64':encoded_image(observation['wrist_image']),
              'noise':observation['noise'].tolist(),'state':observation['state'].tolist(),
              'prompt':str(observation['prompt'].item())}
    vlm_images=[base64.b64encode(f.read_bytes()).decode() for f in sorted((repo/'dual_stack/fixtures').glob('scene-*.png'))]
    if len(vlm_images)!=2:
        raise RuntimeError('Run make_fixtures.py before benchmarking')
    prompt='从左到右描述桌面上三个物体的颜色和类别，只输出一句简短中文，不要解释。'
    vla_url=f'http://127.0.0.1:{args.vla_port}'
    vlm_url=f'http://127.0.0.1:{args.vlm_port}'
    for port in [args.vla_port,args.vlm_port]:
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))==0:
                raise RuntimeError(f'Port {port} is already in use; will not replace its owner')
    env=dict(os.environ,OMP_NUM_THREADS='6',JAX_PLATFORMS='cpu',PYTHONUNBUFFERED='1',NO_PROXY='127.0.0.1,localhost')
    vla_command=['/home/nvidia/pi05-orin-flashrt/.venv/bin/python',str(repo/'dual_stack/vla_server.py'),
                 '--flashrt-root','/home/nvidia/pi05-orin-flashrt/vendor/FlashRT',
                 '--checkpoint','/home/nvidia/pi05-orin/models/pi05_libero','--fixture',str(fixture),
                 '--port',str(args.vla_port)]
    vlm_command=[str(args.root/'build/bin/llama-server'),'-m',str(args.root/'models/Qwen3.5-9B/Qwen3.5-9B-Q5_K_M.gguf'),
                 '--mmproj',str(args.root/'models/Qwen3.5-9B/mmproj-F16.gguf'),
                 '--host','127.0.0.1','--port',str(args.vlm_port),'-ngl','99','-c','8192','-np','1',
                 '-t','4','-tb','4','-b','512','-ub','128','--flash-attn','on',
                 '--jinja','--chat-template-kwargs','{"enable_thinking":false}',
                 '--no-cache-prompt','--metrics','--no-webui']
    monitor=Monitor(out);services={};logs={};phases=[];samples=[];errors=[];reference=None
    was_active=subprocess.run(['systemctl','is-active','openpi-pi05'],capture_output=True,text=True).stdout.strip()
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),'config':vars(args)|{'root':str(args.root),'output':str(out)},
              'platform':platform.platform(),'vla_command':vla_command,'vlm_command':vlm_command,
              'source':json.loads((repo/'dual_stack/sources.json').read_text()),'vlm_prompt':prompt,
              'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'script_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (repo/'dual_stack').glob('*.py')},
              'original_service_state':was_active,'timing_clock':'time.monotonic_ns on the Orin client and services',
              'power_definition':'sum instantaneous VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0; exclude VDDQ already inside VIN_SYS_5V0; not wall-plug AC',
              'limitations':['Synthetic fixed workloads, no robot connected, not robot task success.',
                             '1 Hz observer runs throughout; overhead was not independently calibrated.',
                             'Sampled host/NvMap peaks are lower bounds; shared UMA domains overlap.',
                             'No power/clock/fan settings changed; dynamic clocks and background OS activity remain.',
                             'API timing is loopback HTTP, not external network latency.',
                             'VLA component timing uses explicit synchronization; it is not the untouched previous policy benchmark.',
                             'VLM output length may vary; prompt/decode timing and actual token counts are preserved.']}
    write(out/'provenance.json',metadata)
    write(out/'environment-before.json',snapshot())
    def set_phase(name):
        monitor.set(name,{k:v.pid for k,v in services.items() if v.poll() is None})
        print(json.dumps({'event':'phase','name':name}),flush=True)
    def mark(name,operation):
        set_phase(name);start=time.monotonic_ns()
        result=operation()
        phases.append({'name':name,'start_ns':start,'end_ns':time.monotonic_ns()})
        write(out/'phases.json',phases)
        return result
    def start(name,command,url):
        logs[name]=(out/(name+'-server.log')).open('a')
        process=subprocess.Popen(command,cwd=repo,env=env,stdout=logs[name],stderr=subprocess.STDOUT,start_new_session=True)
        services[name]=process;monitor.set(monitor.phase,{k:v.pid for k,v in services.items()})
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError(f'{name} server exited {process.returncode}; inspect log')
            try:
                value=request_json(url+'/health',timeout=3)
                if name=='vla':
                    metadata.setdefault('vla_instances',[]).append(request_json(url+'/metadata'))
                write(args.root/'runtime/pids.json',{k:v.pid for k,v in services.items()})
                write(out/'provenance.json',metadata)
                return value
            except Exception:
                time.sleep(1)
        raise TimeoutError(name+' startup exceeded 900 seconds')
    def stop(name):
        process=services.pop(name,None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        if name in logs:logs.pop(name).close()
    journal=(out/'requests.jsonl').open('w')
    journal_lock=threading.Lock()
    def one(kind,phase,index,warmup=False):
        nonlocal reference
        result=vla_request(vla_url,vla_body) if kind=='vla' else vlm_request(vlm_url,vlm_images[index%2],prompt)
        row={'kind':kind,'phase':phase,'index':index,'warmup':warmup,**result}
        if kind=='vla':
            action=np.asarray(result['response']['actions'])
            if reference is None:reference=action.copy()
            delta=action-reference
            row['validation']={'finite':bool(np.isfinite(action).all()),'shape':list(action.shape),
                               'instance_exact_match':bool(np.array_equal(action,reference)),
                               'max_abs_vs_instance':float(np.abs(delta).max())}
            if action.shape!=(10,7) or not np.isfinite(action).all() or not np.array_equal(action,reference):
                raise RuntimeError('VLA outputs changed within current process')
        with journal_lock:
            samples.append(row);journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
        return row
    def vla_load(phase,duration):
        deadline=time.monotonic()+duration;i=0
        while time.monotonic()<deadline or i<30:
            one('vla',phase,i);i+=1
    def vlm_load(phase,duration):
        deadline=time.monotonic()+duration;i=0
        while time.monotonic()<deadline or i<args.vlm_min_requests:
            one('vlm',phase,i);i+=1
    def concurrent(phase,duration):
        finished=threading.Event();problems=[]
        def vla_worker():
            i=0
            try:
                while not finished.is_set():one('vla',phase,i);i+=1
            except Exception as e:problems.append(repr(e))
        thread=threading.Thread(target=vla_worker);thread.start()
        try:vlm_load(phase,duration)
        finally:finished.set();thread.join(timeout=180)
        if thread.is_alive() or problems:raise RuntimeError(str(problems) or 'VLA worker did not exit')
    completed=False
    def interrupted(signum,frame):
        raise InterruptedError(f'Benchmark interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        monitor.start()
        subprocess.run(['sudo','-n','systemctl','stop','openpi-pi05'],check=True)
        mark('idle_system',lambda:time.sleep(args.idle_seconds))
        mark('load_vla',lambda:start('vla',vla_command,vla_url))
        mark('warm_vla',lambda:[one('vla','warm_vla',i,True) for i in range(5)])
        mark('vla_standalone',lambda:vla_load('vla_standalone',args.phase_seconds))
        metadata['vla_standalone_stats']=request_json(vla_url+'/stats');stop('vla')
        reference=None
        mark('idle_between_models',lambda:time.sleep(args.idle_seconds))
        mark('load_vlm',lambda:start('vlm',vlm_command,vlm_url))
        mark('warm_vlm',lambda:[one('vlm','warm_vlm',i,True) for i in range(2)])
        mark('vlm_standalone',lambda:vlm_load('vlm_standalone',args.phase_seconds))
        mark('load_vla_with_vlm',lambda:start('vla',vla_command,vla_url))
        mark('warm_vla_dual',lambda:[one('vla','warm_vla_dual',i,True) for i in range(5)])
        mark('dual_resident_idle',lambda:time.sleep(args.idle_seconds))
        mark('vla_dual_resident',lambda:vla_load('vla_dual_resident',args.phase_seconds))
        mark('vlm_dual_resident',lambda:vlm_load('vlm_dual_resident',args.phase_seconds))
        mark('concurrent_saturated',lambda:concurrent('concurrent_saturated',args.concurrent_seconds))
        metadata['vla_final_stats']=request_json(vla_url+'/stats')
        import urllib.request
        write(out/'vlm-final-metrics.json',{'raw':urllib.request.urlopen(vlm_url+'/metrics').read().decode()})
        completed=True
    except Exception as error:
        errors.append({'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        metadata.update(completed=completed,errors=errors,monitor_error=monitor.error,
                        finished_utc=datetime.now(timezone.utc).isoformat())
        for name in list(services):stop(name)
        if was_active in ('active','activating'):
            restore=subprocess.run(['sudo','-n','systemctl','start','openpi-pi05'])
            if restore.returncode:
                state=subprocess.check_output(['systemctl','show','openpi-pi05','-p','Result','--value'],text=True).strip()
                if state=='start-limit-hit':
                    subprocess.run(['sudo','-n','systemctl','reset-failed','openpi-pi05'],check=True)
                    subprocess.run(['sudo','-n','systemctl','start','openpi-pi05'],check=True)
                else:metadata['service_restore_failed']=True
        monitor.stop();journal.close()
        metadata['monitor_error']=monitor.error
        write(out/'environment-after.json',snapshot())
        write(out/'provenance.json',metadata)
        write(out/'phases.json',phases)
        print(json.dumps({'event':'suite_complete','completed':completed,'output':str(out)}),flush=True)


if __name__ == '__main__':
    main()
