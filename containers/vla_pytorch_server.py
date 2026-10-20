#!/usr/bin/env python3
"""Portable HTTP wrapper around the existing OpenPI baseline, no FlashRT substitution."""
import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--fixture', required=True, type=Path)
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8081)
    p.add_argument('--websocket-port', type=int, default=0)
    args=p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(6)
    spec=importlib.util.spec_from_file_location('portable_baseline', ROOT/'baseline/benchmark.py')
    baseline=importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    policy,_,_,_=baseline.load_policy(argparse.Namespace(deployment_root=Path('/opt/pi05-deployment'),
                                                       checkpoint=args.checkpoint))
    with np.load(args.fixture, allow_pickle=False) as f:fixture={k:f[k] for k in f.files}
    lock=threading.Lock(); current_prompt=str(fixture['prompt'].item())
    events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]

    def infer(obs, noise):
        t0=time.perf_counter_ns(); cpu=baseline.prepare_cpu(policy, obs); t1=time.perf_counter_ns()
        inputs, observed, kwargs=baseline.transfer_and_observe(policy, cpu, noise)
        torch.cuda.synchronize(); t2=time.perf_counter_ns()
        events[0].record()
        actions=policy._sample_actions(policy._pytorch_device, observed, **kwargs)
        events[1].record(); torch.cuda.synchronize(); t3=time.perf_counter_ns()
        result=baseline.finish_cpu(policy, inputs, actions)
        torch.cuda.synchronize(); t4=time.perf_counter_ns()
        value=np.asarray(result['actions'])
        if value.shape!=(10,7) or not np.isfinite(value).all():raise RuntimeError('Invalid model actions')
        return value, {'cpu_preprocess_ms':(t1-t0)/1e6, 'input_prepare_upload_ms':(t2-t1)/1e6,
                       'model_wall_ms':(t3-t2)/1e6, 'model_stream_ms':events[0].elapsed_time(events[1]),
                       'download_postprocess_ms':(t4-t3)/1e6, 'policy_wall_ms':(t4-t0)/1e6}

    obs={'observation/image':fixture['image'], 'observation/wrist_image':fixture['wrist_image'],
         'observation/state':fixture['state'], 'prompt':current_prompt}
    with torch.inference_mode():
        for _ in range(10):reference,_=infer(obs,fixture['noise'])
    expected=np.load(args.fixture.with_name('reference_actions.npy'), allow_pickle=False)
    diff=reference.astype(np.float64)-expected.astype(np.float64)
    metadata={'model':'pi05_libero','backend':'OpenPI PyTorch BF16','pid':os.getpid(),
              'action_shape':[10,7],'denoise_steps':10,'torch':torch.__version__, 'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(0),'fixture_sha256':hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              'reference_actions':reference.tolist(), 'baseline_accuracy':{'rmse':float(np.sqrt(np.mean(diff**2))),
              'max_abs':float(np.max(np.abs(diff)))},
              'timing':'CPU transforms / synchronized upload / model / synchronized download; no inner-layer instrumentation'}

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def send(self,value,status=200):
            data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':self.send({'status':'ready','pid':os.getpid()})
            elif self.path=='/metadata':self.send(metadata)
            elif self.path=='/stats':
                with lock:self.send({'torch_live_bytes':torch.cuda.memory_allocated(),
                    'torch_reserved_bytes':torch.cuda.memory_reserved(),
                    'torch_allocator_lifetime_peak_bytes':torch.cuda.max_memory_allocated()})
            else:self.send({'error':'unknown endpoint'},404)
        def do_POST(self):
            nonlocal current_prompt
            if self.path!='/infer':self.send({'error':'unknown endpoint'},404); return
            try:
                begin=time.perf_counter_ns(); length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8*1024*1024:raise ValueError('Invalid body length')
                body=json.loads(self.rfile.read(length))
                def image(name):
                    im=np.asarray(Image.open(io.BytesIO(base64.b64decode(body[name],validate=True))).convert('RGB'))
                    if im.shape!=(224,224,3):raise ValueError('224x224 RGB required')
                    return im
                obs={'observation/image':image('image_base64'),'observation/wrist_image':image('wrist_image_base64'),
                     'observation/state':np.asarray(body.get('state',fixture['state']),dtype=np.float32),
                     'prompt':str(body.get('prompt',str(fixture['prompt'].item())))}
                noise=np.asarray(body['noise'],dtype=np.float32) if 'noise' in body else np.random.default_rng().standard_normal((10,32),dtype=np.float32)
                if noise.shape!=(10,32) or not np.isfinite(noise).all():raise ValueError('Invalid noise')
                if obs['observation/state'].shape!=(8,) or not np.isfinite(obs['observation/state']).all():raise ValueError('Invalid state')
                decoded=time.perf_counter_ns()
                with lock,torch.inference_mode():
                    acquired=time.perf_counter_ns(); changed=obs['prompt']!=current_prompt
                    actions,timings=infer(obs,noise); current_prompt=obs['prompt']
                    timings.update(http_decode_ms=(decoded-begin)/1e6,queue_wait_ms=(acquired-decoded)/1e6)
                self.send({'actions':actions.tolist(),'timings':timings,'prompt_reconfigured':changed})
            except (KeyError,ValueError,TypeError) as error:self.send({'error':str(error)},400)
            except Exception as error:self.send({'error':str(error)},500)

    server=ThreadingHTTPServer((args.host,args.port),Handler); server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    if args.websocket_port:
        sys.path.insert(0,str(ROOT/'dual_stack')); from websocket_bridge import serve
        threading.Thread(target=serve,args=(args.host,args.websocket_port,f'http://127.0.0.1:{args.port}'),daemon=True).start()
    stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set()); signal.signal(signal.SIGINT,lambda *_:stop.set())
    print(json.dumps({'ready':True,**metadata}),flush=True)
    try:stop.wait()
    finally:server.shutdown(); server.server_close()


if __name__=='__main__':main()
