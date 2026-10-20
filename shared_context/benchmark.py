#!/usr/bin/env python3
"""Matched fresh two-process vs shared-context tests with service restoration."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'dual_stack'))
from clients import request_json,vla_request,vlm_request
from environment import snapshot
from telemetry import Monitor
from support import preserve_services,stop_process,wait_ready,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase-seconds',type=float,default=60)
    p.add_argument('--concurrent-seconds',type=float,default=180)
    p.add_argument('--idle-seconds',type=float,default=30)
    p.add_argument('--periodic-seconds',type=float,default=60)
    p.add_argument('--variants',nargs='+',choices=['two_process','shared_stream','shared_device'],default=['two_process','shared_stream'])
    p.add_argument('--memory-probe',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import numpy as np
    from PIL import Image
    fixture=REPO/'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz'
    with np.load(fixture,allow_pickle=False) as f:obs={k:f[k] for k in f.files}
    def encoded(value):
        stream=io.BytesIO();Image.fromarray(value).save(stream,format='PNG');return base64.b64encode(stream.getvalue()).decode()
    body={'image_base64':encoded(obs['image']),'wrist_image_base64':encoded(obs['wrist_image']),
          'noise':obs['noise'].tolist(),'state':obs['state'].tolist(),'prompt':str(obs['prompt'].item())}
    images=[base64.b64encode(f.read_bytes()).decode() for f in sorted((REPO/'dual_stack/fixtures').glob('scene-*.png'))]
    prompt='从左到右描述桌面上三个物体的颜色和类别，只输出一句简短中文，不要解释。'
    dual=Path('/home/nvidia/pi05-orin-dual');runtime=Path('/home/nvidia/pi05-orin-shared-context')
    vla_url='http://127.0.0.1:8081';vlm_url='http://127.0.0.1:8082'
    env=dict(os.environ,OMP_NUM_THREADS='6',NO_PROXY='127.0.0.1,localhost',PYTHONUNBUFFERED='1')
    vla_command=[sys.executable,str(REPO/'dual_stack/vla_server.py'),'--flashrt-root','/home/nvidia/pi05-orin-flashrt/vendor/FlashRT',
                 '--checkpoint','/home/nvidia/pi05-orin/models/pi05_libero','--fixture',str(fixture),'--port','8081']
    vlm_command=[str(dual/'build/bin/llama-server'),'-m',str(dual/'models/Qwen3.5-9B/Qwen3.5-9B-Q5_K_M.gguf'),
                 '--mmproj',str(dual/'models/Qwen3.5-9B/mmproj-F16.gguf'),'--host','127.0.0.1','--port','8082',
                 '-ngl','99','-c','8192','-np','1','-t','4','-tb','4','-b','512','-ub','128','--flash-attn','on',
                 '--jinja','--chat-template-kwargs','{"enable_thinking":false}','--no-cache-prompt','--metrics','--no-webui']
    overall={'created_utc':datetime.now(timezone.utc).isoformat(),'completed':False,'variants':args.variants,
             'config':vars(args)|{'output':str(args.output)},'source':json.loads((REPO/'dual_stack/sources.json').read_text()),
             'script_sha256':{str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest()
                              for folder in ['shared_context','dual_stack'] for f in (REPO/folder).glob('*') if f.suffix in ['.py','.cpp','.sh']},
             'bridge_sha256':hashlib.sha256((runtime/'build/libshared_context.so').read_bytes()).hexdigest(),
             'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
             'limitations':['Fixed order, dynamic clocks and background OS remain; no thermal steady-state claim.',
                            '1 Hz observers add uncalibrated overhead; sampled memory peaks are lower bounds.',
                            'Two-process per-PID memory and shared-PID memory have overlapping UMA accounting domains.',
                            'Synthetic images and fixed VLA noise; no physical robot or task-success evaluation.',
                            'Shared service pins prompt during co-residency to avoid unsafe concurrent graph recapture.',
                            'CUPTI identity proof is a separate preflight; disabled during timed runs.']}
    write(args.output/'provenance.json',overall);write(args.output/'environment-before.json',snapshot())

    def run_variant(variant,out,context=8192,probe_only=False):
        out.mkdir();monitor=Monitor(out);services={};logs=[];phases=[];reference=None;rows=[]
        metadata={'variant':variant,'context_capacity':context,'completed':False,'created_utc':datetime.now(timezone.utc).isoformat()}
        journal=(out/'requests.jsonl').open('w',encoding='utf-8');jlock=threading.Lock()
        def set_phase(name):
            monitor.set(name,{k:v.pid for k,v in services.items() if v.poll() is None})
            print(json.dumps({'event':'phase','variant':variant,'name':name}),flush=True)
        def mark(name,operation):
            set_phase(name);begin=time.monotonic_ns();value=operation()
            phases.append({'name':name,'start_ns':begin,'end_ns':time.monotonic_ns()});write(out/'phases.json',phases);return value
        def start(name,command,url):
            stream=(out/(name+'-service.log')).open('w');logs.append(stream)
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            services[name]=process;monitor.set(monitor.phase,{k:v.pid for k,v in services.items()})
            wait_ready(process,url,request_json)
            metadata.setdefault('commands',{})[name]=command
        def load():
            if variant=='two_process':
                start('vla',vla_command,vla_url);start('vlm',vlm_command,vlm_url)
            else:
                command=[sys.executable,str(REPO/'shared_context/service.py'),'--context',str(context),
                         '--sync','device' if variant=='shared_device' else 'stream']
                start('shared',command,vla_url)
            metadata['vla_metadata']=request_json(vla_url+'/metadata')
            metadata['pids']={k:v.pid for k,v in services.items()}
            metadata['listeners']=subprocess.check_output(['ss','-lntp','( sport = :8081 or sport = :8082 )'],text=True)
            write(out/'provenance.json',metadata)
        def one(kind,phase,index,warmup=False,scheduled_ns=None):
            nonlocal reference
            result=vla_request(vla_url,body) if kind=='vla' else vlm_request(vlm_url,images[index%2],prompt)
            row={'kind':kind,'phase':phase,'index':index,'warmup':warmup,**result}
            if scheduled_ns is not None:row.update(scheduled_ns=scheduled_ns,start_lateness_ms=(result['start_ns']-scheduled_ns)/1e6)
            if kind=='vla':
                actions=np.asarray(result['response']['actions'])
                if reference is None:reference=actions.copy()
                row['validation']={'finite':bool(np.isfinite(actions).all()),'shape':list(actions.shape),
                                   'instance_exact_match':bool(np.array_equal(actions,reference)),
                                   'max_abs_vs_instance':float(np.abs(actions-reference).max())}
                if actions.shape!=(10,7) or not np.isfinite(actions).all() or not np.array_equal(actions,reference):
                    raise RuntimeError('VLA output changed within process')
            with jlock:rows.append(row);journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
            return row
        def load_requests(kind,phase,duration):
            deadline=time.monotonic()+duration;i=0
            while time.monotonic()<deadline or i<(30 if kind=='vla' else 8):one(kind,phase,i);i+=1
        def concurrent(phase,duration):
            done=threading.Event();failures=[]
            def vla_worker():
                try:
                    i=0
                    while not done.is_set():one('vla',phase,i);i+=1
                except Exception as error:failures.append(repr(error))
            thread=threading.Thread(target=vla_worker);thread.start()
            try:load_requests('vlm',phase,duration)
            finally:done.set();thread.join(timeout=900)
            if failures or thread.is_alive():raise RuntimeError(str(failures) or 'VLA worker did not stop')
        def periodic(phase,duration):
            origin=time.monotonic_ns();finish=origin+int(duration*1e9);failures=[]
            def worker(kind,period):
                try:
                    slot=0;i=0
                    while True:
                        target=origin+int(slot*period*1e9)
                        if target>=finish:break
                        wait=(target-time.monotonic_ns())/1e9
                        if wait>0:time.sleep(wait)
                        one(kind,phase,i,scheduled_ns=target);i+=1;slot+=1
                        # Drop missed triggers rather than accumulate an unbounded queue.
                        slot=max(slot,int((time.monotonic_ns()-origin)/(period*1e9))+1)
                except Exception as error:failures.append(repr(error))
            threads=[threading.Thread(target=worker,args=('vla',.5)),threading.Thread(target=worker,args=('vlm',5.0))]
            for thread in threads:thread.start()
            for thread in threads:thread.join(timeout=900)
            if failures or any(t.is_alive() for t in threads):raise RuntimeError(str(failures) or 'Periodic worker hung')
        try:
            monitor.start()
            mark('idle_system',lambda:time.sleep(args.idle_seconds))
            mark('load_models',load)
            mark('warm_models',lambda:([one('vla','warm_models',i,True) for i in range(5)],
                                      [one('vlm','warm_models',i,True) for i in range(2)]))
            mark('dual_resident_idle',lambda:time.sleep(args.idle_seconds))
            if probe_only:
                mark('memory_capacity_probe',lambda:[one('vlm','memory_capacity_probe',i) for i in range(8)])
            else:
                mark('vla_dual_resident',lambda:load_requests('vla','vla_dual_resident',args.phase_seconds))
                mark('vlm_dual_resident',lambda:load_requests('vlm','vlm_dual_resident',args.phase_seconds))
                mark('concurrent_saturated',lambda:concurrent('concurrent_saturated',args.concurrent_seconds))
                mark('periodic_2hz_0p2hz',lambda:periodic('periodic_2hz_0p2hz',args.periodic_seconds))
            metadata['vla_final_stats']=request_json(vla_url+'/stats')
            metadata['completed']=True
        except Exception as error:
            metadata['error']={'type':type(error).__name__,'message':str(error)};raise
        finally:
            for process in services.values():stop_process(process)
            monitor.stop();journal.close()
            for stream in logs:stream.close()
            metadata.update(monitor_error=monitor.error,finished_utc=datetime.now(timezone.utc).isoformat())
            write(out/'provenance.json',metadata);write(out/'phases.json',phases)
            print(json.dumps({'event':'variant_complete','variant':variant,'completed':metadata['completed']}),flush=True)
        if monitor.error:raise RuntimeError('Monitor failed: '+monitor.error)

    try:
        with preserve_services(args.output):
            for variant in args.variants:run_variant(variant,args.output/variant)
            if args.memory_probe:run_variant('shared_stream',args.output/'shared_context_16k',context=16384,probe_only=True)
        overall['completed']=True
    except Exception as error:overall['error']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        overall['finished_utc']=datetime.now(timezone.utc).isoformat()
        write(args.output/'environment-after.json',snapshot());write(args.output/'provenance.json',overall)
        print(json.dumps({'event':'experiment_complete','completed':overall['completed']}),flush=True)


if __name__=='__main__':main()
