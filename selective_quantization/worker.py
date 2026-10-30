"""One fresh process per measured configuration; inputs and outputs are explicit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from adapter import LiberoExperiment
from metrics import accuracy, stats, MetricsMonitor


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def load_cases(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as f: data={k:f[k] for k in f.files}
    return [{k:data[k][i] for k in ('image','wrist_image','third_image','state','noise','prompt')}
            for i in range(len(data['names']))],data['names'].tolist()


def build_adapter(config,args,fixture):
    values={k:v for k,v in config.items() if k in
        ('num_steps','num_views','valid_num_views','max_prompt_len','encoder_layers',
         'decoder_layers','modules','encoder_modules','decoder_modules')}
    obj=LiberoExperiment(args.flashrt_root,args.checkpoint,**values)
    obj.prepare(fixture)
    return obj


def model_sample(obj,noise,pair):
    torch=obj.torch
    obj.stage_model_inputs(noise)
    torch.cuda.synchronize()
    start=time.perf_counter_ns()
    pair[0].record(obj.model_stream)
    obj.replay_model_graph()
    pair[1].record(obj.model_stream)
    torch.cuda.synchronize()
    return (time.perf_counter_ns()-start)/1e6,float(pair[0].elapsed_time(pair[1]))


def run(args):
    import numpy as np
    import torch
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text())
    cases,names=load_cases(args.inputs)
    config.setdefault('num_steps',10);config.setdefault('num_views',3)
    config.setdefault('valid_num_views',2);config.setdefault('max_prompt_len',200)
    result={'config':config,'case_names':names,'completed':False,
        'input_sha256':hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        'timing_case':names[1], 'load_started_unix':time.time()}
    obj=build_adapter(config,args,cases[1])
    torch.cuda.synchronize()
    result['load_finished_unix']=time.time()
    result['metadata']=obj.metadata()
    actions=[]
    for case in cases:
        obj.prepare(case)
        actions.append(obj.infer(case['noise']))
    actions=np.stack(actions)
    np.save(out/'actions.npy',actions)
    if args.references:
        reference=args.references/f'openpi_{config["valid_num_views"]}valid_10steps.npy'
        result['accuracy_openpi']=accuracy(actions,np.load(reference,allow_pickle=False))
    if args.teacher and args.teacher.is_file():
        result['accuracy_bf16']=accuracy(actions,np.load(args.teacher,allow_pickle=False))
    obj.prepare(cases[1]); obj.stage_model_inputs(cases[1]['noise'],include_images=True)
    torch.cuda.synchronize()
    first=obj.infer(cases[1]['noise'])
    pair=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    warm=[];wall=[];cuda=[]
    for i in range(args.warmup+args.iterations):
        w,c=model_sample(obj,cases[1]['noise'],pair)
        if i<args.warmup:warm.append(w)
        else:wall.append(w);cuda.append(c)
    repeated=obj.download_model_actions()
    result['repeat_vs_first']=accuracy(repeated,first)
    if not result['repeat_vs_first']['exact_match']:
        raise RuntimeError('Fixed-input full graph changed during latency pass')
    result['latency']={'model_wall':stats(wall),'model_cuda_stream':stats(cuda),
                       'warmup_wall_ms':warm,'samples_wall_ms':wall,'samples_cuda_ms':cuda,
                       'boundary':'GPU-ready input -> original full CUDA graph -> GPU synchronized; staging/download excluded'}
    write_json(out/'latency_partial.json',result)
    print(json.dumps({'event':'latency_complete','name':config['name'],
                     'latency':result['latency']['model_wall'],
                     'cosine':result.get('accuracy_openpi',{}).get('cosine')}),flush=True)

    # Deliberately separate OS telemetry from the primary latency pass.
    monitor=MetricsMonitor(out/'telemetry',interval=.5,nvmap_interval=1)
    monitor.start();monitor.set_phase('sustained_model')
    torch.cuda.reset_peak_memory_stats()
    monitor.snapshot_torch(torch,'before_sustained')
    count=0;start=time.monotonic()
    try:
        while time.monotonic()-start < args.load_seconds or count < 3:
            model_sample(obj,cases[1]['noise'],pair)
            count+=1
        torch.cuda.synchronize()
        monitor.snapshot_torch(torch,'after_sustained')
    finally:
        monitor.stop()
    result['sustained']={'seconds':time.monotonic()-start,'iterations':count,
                         'telemetry':monitor.summary('sustained_model')}
    write_json(out/'telemetry_partial.json',result)
    if args.load_seconds >= 2 and result['sustained']['telemetry']['sample_count'] < 2:
        raise RuntimeError('Insufficient telemetry samples; this variant cannot support a memory/temperature report')
    # Extra stage graphs are created only AFTER primary latency and memory.
    graphs=obj.stage_graphs()
    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    stage_values={}
    for i in range(3+args.stage_iterations):
        stage_actions,timings=obj.run_stages(graphs,cases[1]['noise'],events)
        if i>=3:
            for key,value in timings.items():stage_values.setdefault(key,[]).append(value)
    result['stage_chain_vs_full']=accuracy(stage_actions,first)
    if not result['stage_chain_vs_full']['exact_match']:
        raise RuntimeError('Diagnostic stage chain is not equal to the candidate full graph')
    result['stages']={k:stats(v) for k,v in stage_values.items()}
    model_sample(obj,cases[1]['noise'],pair)
    result['restored_full_vs_first']=accuracy(obj.download_model_actions(),first)
    if not result['restored_full_vs_first']['exact_match']:
        raise RuntimeError('Original full graph changed after diagnostic capture')
    result['metadata']=obj.metadata()
    result['completed']=True;result['finished_unix']=time.time()
    write_json(out/'result.json',result)
    print(json.dumps({'event':'variant_complete','name':config['name']}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--references',type=Path)
    p.add_argument('--teacher',type=Path)
    p.add_argument('--flashrt-root',type=Path,default=Path('/home/nvidia/pi05-orin-flashrt/vendor/FlashRT'))
    p.add_argument('--checkpoint',type=Path,default=Path('/home/nvidia/pi05-orin/models/pi05_libero'))
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iterations',type=int,default=50)
    p.add_argument('--stage-iterations',type=int,default=10)
    p.add_argument('--load-seconds',type=float,default=30)
    a=p.parse_args()
    try:run(a)
    except Exception as e:
        a.output.mkdir(parents=True,exist_ok=True)
        write_json(a.output/'error.json',{'error':repr(e),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
