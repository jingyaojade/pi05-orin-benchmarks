"""Matched workloads for NVIDIA/Thor PyTorch+llama candidates; no systemd interaction."""
import base64
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import signal
import threading
import time

from runtime import ROOT, FIXTURE, Models, commands
from telemetry_container import Monitor, kib_fields
from clients import request_json, vla_request, vlm_request
from report import stats
from device_config import bindings


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def run(args,environment,manifest):
    import numpy as np
    from PIL import Image
    out=args.output
    if out.exists():raise FileExistsError('Use a new results directory: '+str(out))
    out.mkdir(parents=True)
    with np.load(FIXTURE,allow_pickle=False) as f:obs={k:f[k] for k in f.files}
    def image(value):
        stream=io.BytesIO();Image.fromarray(value).save(stream,format='PNG')
        return base64.b64encode(stream.getvalue()).decode()
    body={'image_base64':image(obs['image']),'wrist_image_base64':image(obs['wrist_image']),
          'state':obs['state'].tolist(),'noise':obs['noise'].tolist(),'prompt':str(obs['prompt'].item())}
    images=[base64.b64encode((ROOT/'dual_stack/fixtures'/('scene-'+str(i)+'.png')).read_bytes()).decode() for i in [0,1]]
    prompt='从左到右描述桌面上三个物体的颜色和类别，只输出一句简短中文，不要解释。'
    models=Models(args.assets,out); monitor=Monitor(out,models); phases=[]; rows=[]; reference=None
    lock=threading.Lock(); journal=(out/'requests.jsonl').open('w',encoding='utf-8')
    metadata={'completed':False,'created_utc':datetime.now(timezone.utc).isoformat(),
              'environment':environment,'assets':manifest,'commands':commands(args.assets),
              'gpu_bindings':bindings(),
              'backend':'OpenPI PyTorch BF16 + llama.cpp CUDA; not FlashRT',
              'limitations':['This execution does not validate FlashRT on this GPU.',
                'Loopback API; PNG/base64 prepared in advance; synthetic fixed observations/noise.',
                '1Hz best-effort telemetry; sampled peaks are lower bounds.',
                'NVML device memory, process RSS and host memory have distinct/overlapping scopes; do not sum.',
                'Reported device power is not asserted to equal Orin three-rail or AC wall power.',
                'No power/clock/fan changes, host service changes or physical robot execution.']}
    def stop(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def phase(name,operation):
        monitor.phase=name;start=time.monotonic_ns();print(json.dumps({'phase':name}),flush=True)
        operation();phases.append({'name':name,'start_ns':start,'end_ns':time.monotonic_ns()})
        write(out/'phases.json',phases)
    def one(kind,name,index,warmup=False):
        nonlocal reference
        row={'kind':kind,'phase':name,'warmup':warmup,'index':index}
        if kind=='vla':
            row.update(vla_request('http://127.0.0.1:8081',body))
            value=np.asarray(row['response']['actions'])
            if reference is None:reference=value.copy()
            stable=np.array_equal(value,reference)
            row['validation']={'shape':list(value.shape),'finite':bool(np.isfinite(value).all()),'instance_exact_match':stable}
            if value.shape!=(10,7) or not np.isfinite(value).all() or not stable:
                raise RuntimeError('VLA fixed-input validation failed')
        else:
            row.update(vlm_request('http://127.0.0.1:8082',images[index%2],prompt))
            if row['timings'].get('cache_n',0)!=0 or row['finish_reason']=='length':
                raise RuntimeError('VLM cache/length mismatch')
        with lock:rows.append(row);journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');journal.flush()
    def sustained(kind,name,duration):
        deadline=time.monotonic()+duration;i=0
        while time.monotonic()<deadline or i<(30 if kind=='vla' else 8):one(kind,name,i);i+=1
    def concurrent():
        done=threading.Event();errors=[]
        def worker():
            try:
                i=0
                while not done.is_set():one('vla','concurrent_saturated',i);i+=1
            except Exception as error:errors.append(repr(error))
        thread=threading.Thread(target=worker);thread.start()
        try:sustained('vlm','concurrent_saturated',args.concurrent_seconds)
        finally:done.set();thread.join(timeout=900)
        if errors or thread.is_alive():raise RuntimeError(str(errors) or 'VLA did not stop')
    def periodic():
        origin=time.monotonic_ns();finish=origin+60_000_000_000;errors=[]
        def worker(kind,period):
            try:
                slot=0;i=0
                while True:
                    target=origin+int(slot*period*1e9)
                    if target>=finish:break
                    delay=(target-time.monotonic_ns())/1e9
                    if delay>0:time.sleep(delay)
                    one(kind,'periodic_2hz_0p2hz',i);i+=1;slot+=1
                    slot=max(slot,int((time.monotonic_ns()-origin)/(period*1e9))+1)
            except Exception as error:errors.append(repr(error))
        threads=[threading.Thread(target=worker,args=('vla',.5)),threading.Thread(target=worker,args=('vlm',5.0))]
        for t in threads:t.start()
        for t in threads:t.join(timeout=900)
        if errors or any(t.is_alive() for t in threads):raise RuntimeError(str(errors) or 'Periodic worker did not stop')
    try:
        monitor.start();phase('idle_system',lambda:time.sleep(args.idle_seconds))
        phase('load_models',models.start)
        metadata['vla_metadata']=request_json('http://127.0.0.1:8081/metadata')
        metadata['pids']={k:v.pid for k,v in models.processes.items()}
        phase('warm_models',lambda:([one('vla','warm_models',i,True) for i in range(5)],
                                   [one('vlm','warm_models',i,True) for i in range(2)]))
        phase('dual_resident_idle',lambda:time.sleep(args.idle_seconds))
        phase('vla_dual_resident',lambda:sustained('vla','vla_dual_resident',args.phase_seconds))
        phase('vlm_dual_resident',lambda:sustained('vlm','vlm_dual_resident',args.phase_seconds))
        phase('concurrent_saturated',concurrent)
        phase('periodic_2hz_0p2hz',periodic)
        metadata['vla_final_stats']=request_json('http://127.0.0.1:8081/stats')
        metadata['completed']=True
    except BaseException as error:
        metadata['error']={'type':type(error).__name__,'message':str(error)};raise
    finally:
        models.close();monitor.stop();journal.close()
        metadata.update(finished_utc=datetime.now(timezone.utc).isoformat(),monitor_error=monitor.error)
        if monitor.error:metadata['completed']=False
        write(out/'provenance.json',metadata)
    if monitor.error:raise RuntimeError(monitor.error)
    summarize(out,rows,phases,metadata)


def summarize(out,rows,phases,metadata):
    result={'schema_version':2,'metadata':metadata,'phases':{}}
    text=['# π0.5 + Qwen GPU双进程实测报告','',metadata['backend'],'',
          '| 场景 | VLA n / 完成Hz | VLA模型 mean / p95 ms | VLA API mean ms | VLM n / 完成Hz | VLM首字 mean ms | VLM完整 mean ms |',
          '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return 'N/A' if value is None else f'{value:.2f}'
    for phase in phases:
        duration=(phase['end_ns']-phase['start_ns'])/1e9;entries={}
        for kind in ['vla','vlm']:
            chosen=[r for r in rows if r['phase']==phase['name'] and r['kind']==kind and not r['warmup']]
            data={'n':len(chosen),'completed_hz':len(chosen)/duration,'api_wall_ms':stats([r['api_wall_ms'] for r in chosen])}
            if kind=='vla':
                for name in ['cpu_preprocess_ms','input_prepare_upload_ms','model_wall_ms','model_stream_ms','download_postprocess_ms']:
                    data[name]=stats([r['response']['timings'][name] for r in chosen])
            else:
                data['ttft_ms']=stats([r['ttft_ms'] for r in chosen])
                for name in ['prompt_ms','predicted_ms','prompt_n','predicted_n','cache_n']:
                    data[name]=stats([r['timings'].get(name) for r in chosen])
            entries[kind]=data
        result['phases'][phase['name']]={'duration_s':duration,**entries}
        v,l=entries['vla'],entries['vlm']
        if v['n'] or l['n']:
            text.append(f"| {phase['name']} | {v['n']} / {v['completed_hz']:.3f} | {fmt(v['model_wall_ms']['mean'])} / {fmt(v['model_wall_ms']['p95'])} | {fmt(v['api_wall_ms']['mean'])} | {l['n']} / {l['completed_hz']:.3f} | {fmt(l['ttft_ms']['mean'])} | {fmt(l['api_wall_ms']['mean'])} |")
    telemetry=[json.loads(line) for line in (out/'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    for phase in phases:
        samples=[r for r in telemetry if phase['start_ns']<=r['monotonic_ns']<=phase['end_ns']]
        data=result['phases'][phase['name']]
        def device_id(device):return device.get('uuid') or 'index:'+str(device.get('index'))
        identities=sorted({device_id(d) for r in samples for d in r.get('devices',[])})
        data['device_metrics']={identity:{key:stats([d.get(key) for r in samples for d in r.get('devices',[])
                                                   if device_id(d)==identity])
                                for key in ['memory.used','temperature.gpu','power.draw','utilization.gpu']}
                                for identity in identities}
        data['telemetry_samples']=len(samples)
        data['host_mem_available_bytes']=stats([kib_fields(r['host_meminfo_raw']).get('MemAvailable') for r in samples])
        data['process_memory_bytes']={}
        for kind in ['vla','vlm']:
            data['process_memory_bytes'][kind]={key:stats([
                kib_fields(r.get('processes',{}).get(kind,{}).get('smaps_rollup_raw','')).get(key)
                for r in samples]) for key in ['Rss','Pss']}
    text+=['','| 场景 | GPU UUID / 设备ID | GPU设备内存 max MiB | GPU温度 max °C | 设备功耗 mean / max W |',
           '|---|---|---:|---:|---:|']
    def gib(value):return fmt(value/2**30) if value is not None else 'N/A'
    for phase in phases:
        r=result['phases'][phase['name']]
        for identity,d in r['device_metrics'].items():
            text.append(f"| {phase['name']} | {identity} | {fmt(d['memory.used']['max'])} | {fmt(d['temperature.gpu']['max'])} | {fmt(d['power.draw']['mean'])} / {fmt(d['power.draw']['max'])} |")
        if not r['device_metrics']:text.append(f"| {phase['name']} | N/A | N/A | N/A | N/A |")
    text+=['','| 场景 | 主机可用内存 min GiB | VLA / VLM RSS max GiB |','|---|---:|---:|']
    for phase in phases:
        r=result['phases'][phase['name']];m=r['process_memory_bytes']
        text.append(f"| {phase['name']} | {gib(r['host_mem_available_bytes']['min'])} | {gib(m['vla']['Rss']['max'])} / {gib(m['vlm']['Rss']['max'])} |")
    text+=['','NVML温度/设备功耗/内存统计见summary.json及原始telemetry.jsonl。',
           'memory.used单位MiB，temperature.gpu为°C，power.draw为设备报告W，utilization.gpu为%。',
           '缺失读数为N/A；没有套用Orin三轨功耗求和，也没有假定Thor支持NvMap。',
           '主机内存包含其他负载，RSS包含共享映射；这些记账范围不能与设备内存求和。',
           'GPU指标按UUID分别记录，是设备级读数，可能包含其他任务；没有把第二张卡的指标归到第一张。',
           '这是PyTorch BF16与llama.cpp的结果，不能与Orin FlashRT之比作为纯硬件加速比。',
           'Torch单模型分层拆分另运行baseline命令；此组合报告不伪造内部vision/prefix/action独立耗时。','']
    write(out/'summary.json',result);(out/'report_zh.md').write_text('\n'.join(text),encoding='utf-8')
