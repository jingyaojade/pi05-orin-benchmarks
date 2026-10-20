#!/usr/bin/env python3
"""One Python PID: FlashRT plus the pinned llama-server implementation as a library."""
from __future__ import annotations
import argparse
import base64
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'dual_stack'))
from clients import request_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/nvidia/pi05-orin-shared-context'))
    p.add_argument('--dual-root',type=Path,default=Path('/home/nvidia/pi05-orin-dual'))
    p.add_argument('--models',choices=['vla','vlm','both'],default='both')
    p.add_argument('--port',type=int,default=8081)
    p.add_argument('--vlm-port',type=int,default=8082)
    p.add_argument('--context',type=int,default=8192)
    p.add_argument('--websocket-port',type=int,default=0)
    p.add_argument('--sync',choices=['stream','device'],default='stream')
    args=p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(6)
    torch.cuda.set_device(0)
    probe=torch.empty(1,device='cuda');torch.cuda.current_stream().synchronize();del probe
    # CDLL releases the GIL around native calls, including the blocking C++ loop.
    bridge=ctypes.CDLL(str(args.root/'build/libshared_context.so'),mode=ctypes.RTLD_LOCAL)
    bridge.sc_llama_run.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_char_p)]
    bridge.sc_llama_run.restype=ctypes.c_int
    bridge.sc_current_context.restype=ctypes.c_size_t
    bridge.sc_audit_stop.restype=ctypes.c_char_p
    bridge.sc_set_role.argtypes=[ctypes.c_int]
    bridge.sc_set_context.argtypes=[ctypes.c_size_t]
    primary=bridge.sc_current_context()
    if not primary:raise RuntimeError('Torch did not establish a CUDA context')
    fixture_path=REPO/'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz'
    metadata={'pid':os.getpid(),'model':'pi05_libero','topology':'single Python PID, embedded native llama implementation',
              'cuda_context':primary,'torch_version':torch.__version__,'cuda_build':torch.version.cuda,
              'models':args.models,'sync':args.sync,'context_capacity':args.context,'action_shape':[10,7],
              'fixture_sha256':hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
              'prompt_reconfiguration':'requires maintenance/restart while native VLM is resident'}
    adapter=None
    lock=threading.Lock()
    if args.models in ('vla','both'):
        sys.path.insert(0,str(REPO/'flashrt'))
        from adapter import LiberoFlashRT
        from benchmark import accuracy
        with np.load(fixture_path,allow_pickle=False) as archive:fixture={k:archive[k] for k in archive.files}
        with torch.inference_mode():
            adapter=LiberoFlashRT(Path('/home/nvidia/pi05-orin-flashrt/vendor/FlashRT'),
                                  Path('/home/nvidia/pi05-orin/models/pi05_libero'),'bf16')
            adapter.prepare(fixture)
            reference=adapter.infer(fixture['noise']).copy()
            for _ in range(10):
                if not np.array_equal(adapter.infer(fixture['noise']),reference):raise RuntimeError('Unstable VLA warmup')
        current_prompt=str(fixture['prompt'].item())
        metadata.update(reference_actions=reference.tolist(),tokenizer=adapter.tokenizer_provenance,
                        baseline_accuracy=accuracy(reference,np.load(fixture_path.with_name('reference_actions.npy')),np))
        events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]

    # The Qwen side is the SAME server implementation/flags/API, loaded in-process.
    native_thread=None;native_result={};stopping=threading.Event()
    if args.models in ('vlm','both'):
        command=['embedded-llama','-m',str(args.dual_root/'models/Qwen3.5-9B/Qwen3.5-9B-Q5_K_M.gguf'),
                 '--mmproj',str(args.dual_root/'models/Qwen3.5-9B/mmproj-F16.gguf'),
                 '--host','127.0.0.1','--port',str(args.vlm_port),'-ngl','99','-c',str(args.context),'-np','1',
                 '-t','4','-tb','4','-b','512','-ub','128','--flash-attn','on',
                 '--jinja','--chat-template-kwargs','{"enable_thinking":false}',
                 '--no-cache-prompt','--metrics','--no-webui']
        argv=(ctypes.c_char_p*(len(command)+1))(*[x.encode() for x in command],None)
        def native():
            if bridge.sc_set_context(primary):raise RuntimeError('Cannot bind llama to Torch context')
            native_result['code']=bridge.sc_llama_run(len(command),argv)
        native_thread=threading.Thread(target=native,name='embedded-llama',daemon=True);native_thread.start()
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if not native_thread.is_alive():raise RuntimeError('Embedded llama exited: '+str(native_result))
            try:
                request_json(f'http://127.0.0.1:{args.vlm_port}/health',timeout=2);break
            except Exception:time.sleep(.2)
        else:raise TimeoutError('Embedded llama startup')
        time.sleep(.2) # native readiness precedes its signal-handler registration
        metadata['vlm_command']=command

    def wait_own_stream():
        if args.sync=='device':torch.cuda.synchronize()
        else:adapter.model_stream.synchronize()

    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*unused):pass
        def send(self,value,status=200):
            payload=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path=='/health':self.send({'status':'ready','pid':os.getpid(),'models':args.models})
            elif self.path=='/metadata':self.send(metadata)
            elif self.path=='/stats':
                with lock:self.send({'torch_live_bytes':torch.cuda.memory_allocated(),
                    'torch_reserved_bytes':torch.cuda.memory_reserved(),
                    'torch_allocator_lifetime_peak_bytes':torch.cuda.max_memory_allocated()})
            elif self.path=='/audit/start':
                code=bridge.sc_audit_start();self.send({'code':code},200 if code==0 else 500)
            elif self.path=='/audit/stop':self.send(json.loads(bridge.sc_audit_stop()))
            elif self.path=='/profile/start':self.send({'code':bridge.sc_profiler_start()})
            elif self.path=='/profile/stop':self.send({'code':bridge.sc_profiler_stop()})
            else:self.send({'error':'unknown endpoint'},404)
        def do_POST(self):
            if self.path!='/infer' or adapter is None:self.send({'error':'VLA unavailable'},404);return
            try:
                begin=time.perf_counter_ns();length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=8*1024*1024:raise ValueError('Invalid body size')
                request=json.loads(self.rfile.read(length))
                def image(key):
                    value=np.asarray(Image.open(io.BytesIO(base64.b64decode(request[key],validate=True))).convert('RGB'))
                    if value.shape!=(224,224,3):raise ValueError('RGB 224x224 required')
                    return value
                obs={'image':image('image_base64'),'wrist_image':image('wrist_image_base64'),
                     'noise':np.asarray(request['noise'],dtype=np.float32) if 'noise' in request
                             else np.random.default_rng().standard_normal((10,32),dtype=np.float32),
                     'state':np.asarray(request.get('state',fixture['state']),dtype=np.float32),
                     'prompt':str(request.get('prompt',current_prompt))}
                if obs['noise'].shape!=(10,32) or not np.isfinite(obs['noise']).all():raise ValueError('Invalid noise')
                if obs['state'].shape!=(8,) or not np.isfinite(obs['state']).all():raise ValueError('Invalid state')
                if obs['prompt']!=current_prompt:
                    self.send({'error':'Prompt changes require maintenance/restart; CUDA recapture is not performed while Qwen may be active.'},409);return
                decoded=time.perf_counter_ns()
                with lock,torch.inference_mode():
                    acquired=time.perf_counter_ns()
                    if bridge.sc_set_context(primary):raise RuntimeError('Cannot bind VLA context')
                    torch.cuda.set_device(0);bridge.sc_set_role(1)
                    if bridge.sc_current_context()!=primary:raise RuntimeError('VLA context changed')
                    adapter.observation={'images':[obs['image'],obs['wrist_image'],np.zeros_like(obs['image'])]}
                    wait_own_stream();t0=time.perf_counter_ns()
                    adapter.stage_model_inputs(obs['noise'],include_images=True);wait_own_stream();t1=time.perf_counter_ns()
                    events[0].record(adapter.model_stream);adapter.replay_model_graph();events[1].record(adapter.model_stream)
                    wait_own_stream();t2=time.perf_counter_ns()
                    with torch.cuda.stream(adapter.model_stream):actions=adapter.download_model_actions()
                    wait_own_stream();t3=time.perf_counter_ns()
                    result={'actions':actions.tolist(),'prompt_reconfigured':False,'timings':{
                        'http_decode_ms':(decoded-begin)/1e6,'queue_wait_ms':(acquired-decoded)/1e6,
                        'setup_ms':(t0-acquired)/1e6,'input_prepare_upload_ms':(t1-t0)/1e6,
                        'model_wall_ms':(t2-t1)/1e6,'model_stream_ms':events[0].elapsed_time(events[1]),
                        'download_postprocess_ms':(t3-t2)/1e6,'policy_wall_ms':(t3-t0)/1e6,'handler_wall_ms':(t3-begin)/1e6}}
                self.send(result)
            except (ValueError,TypeError,KeyError) as error:self.send({'error':str(error)},400)
            except Exception as error:self.send({'error':str(error)},500);raise

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    if args.websocket_port:
        from websocket_bridge import serve
        threading.Thread(target=serve,args=('127.0.0.1',args.websocket_port,f'http://127.0.0.1:{args.port}'),daemon=True).start()
    def stop(signum,frame):stopping.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(json.dumps({'event':'shared_ready',**metadata}),flush=True)
    try:
        while not stopping.wait(.5):
            if native_thread is not None and not native_thread.is_alive():raise RuntimeError('Native loop exited: '+str(native_result))
    finally:
        server.shutdown();server.server_close()
        if native_thread is not None:bridge.sc_llama_stop();native_thread.join(timeout=20)


if __name__=='__main__':main()
