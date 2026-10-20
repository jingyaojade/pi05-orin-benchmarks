#!/usr/bin/env python3
"""Loopback FlashRT pi05-LIBERO JSON service, with explicit timing boundaries."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flashrt-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8081)
    p.add_argument('--websocket-port', type=int, default=0, help='Optional original OpenPI protocol; zero disables')
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flashrt'))
    import numpy as np
    import torch
    from PIL import Image
    from adapter import LiberoFlashRT
    from benchmark import accuracy
    torch.set_num_threads(6)
    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {key: archive[key] for key in archive.files}
    with torch.inference_mode():
        adapter = LiberoFlashRT(args.flashrt_root, args.checkpoint, 'bf16')
        adapter.prepare(fixture)
        reference = adapter.infer(fixture['noise']).copy()
        for _ in range(10):
            if not np.array_equal(adapter.infer(fixture['noise']), reference):
                raise RuntimeError('VLA warmup is not stable within this instance')
    torch.cuda.synchronize()
    lock = threading.Lock()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    current_prompt = str(fixture['prompt'].item())
    metadata = {'model': 'pi05_libero', 'backend': 'FlashRT BF16', 'pid': os.getpid(),
                'torch_version': torch.__version__, 'cuda_build': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(),
                'action_shape': [10, 7], 'action_horizon': 10, 'action_dim': 7, 'state_dim': 8, 'denoise_steps': 10,
                'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                'reference_actions': reference.tolist(), 'tokenizer': adapter.tokenizer_provenance,
                'baseline_accuracy': accuracy(reference, np.load(args.fixture.with_name('reference_actions.npy')), np),
                'timing': 'synchronized input staging / original production graph / download and action decoding'}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *unused):
            pass

        def send(self, value, status=200):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == '/health':
                self.send({'status': 'ready', 'pid': os.getpid()})
            elif self.path == '/metadata':
                self.send(metadata)
            elif self.path == '/stats':
                with lock:
                    self.send({'torch_live_bytes': torch.cuda.memory_allocated(),
                               'torch_reserved_bytes': torch.cuda.memory_reserved(),
                               'torch_allocator_lifetime_peak_bytes': torch.cuda.max_memory_allocated()})
            else:
                self.send({'error': 'unknown endpoint'}, 404)

        def do_POST(self):
            nonlocal current_prompt
            if self.path != '/infer':
                self.send({'error': 'unknown endpoint'}, 404)
                return
            try:
                begin = time.perf_counter_ns()
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8 * 1024 * 1024:
                    raise ValueError('JSON body size must be 1..8 MiB')
                request = json.loads(self.rfile.read(length))
                def image(key):
                    value = np.asarray(Image.open(io.BytesIO(base64.b64decode(request[key], validate=True))).convert('RGB'))
                    if value.shape != (224, 224, 3):
                        raise ValueError(f'{key} must decode to RGB 224x224')
                    return value
                obs = {'image': image('image_base64'), 'wrist_image': image('wrist_image_base64'),
                       'noise': np.asarray(request['noise'], dtype=np.float32) if 'noise' in request
                                else np.random.default_rng().standard_normal((10,32), dtype=np.float32),
                       'state': np.asarray(request.get('state', fixture['state']), dtype=np.float32),
                       'prompt': np.asarray(request.get('prompt', current_prompt))}
                if obs['noise'].shape != (10, 32) or not np.isfinite(obs['noise']).all():
                    raise ValueError('noise must be finite FP32 [10,32]')
                if obs['state'].shape != (8,) or not np.isfinite(obs['state']).all():
                    raise ValueError('state must be finite FP32 [8]')
                decoded = time.perf_counter_ns()
                with lock, torch.inference_mode():
                    acquired = time.perf_counter_ns()
                    prompt = str(obs['prompt'].item())
                    reconfigured = prompt != current_prompt
                    if reconfigured:
                        # Refresh frontend calibration/capture for a changed instruction.
                        # This setup cost is included in request time and flagged explicitly.
                        adapter.prepare(obs)
                        current_prompt = prompt
                    else:
                        adapter.observation = {'images': [obs['image'], obs['wrist_image'], np.zeros_like(obs['image'])]}
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    adapter.stage_model_inputs(obs['noise'], include_images=True)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter_ns()
                    events[0].record(adapter.model_stream)
                    adapter.replay_model_graph()
                    events[1].record(adapter.model_stream)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter_ns()
                    actions = adapter.download_model_actions()
                    torch.cuda.synchronize()
                    t3 = time.perf_counter_ns()
                    result = {'actions': actions.tolist(), 'prompt_reconfigured': reconfigured, 'timings': {
                        'http_decode_ms': (decoded-begin)/1e6, 'queue_wait_ms': (acquired-decoded)/1e6,
                        'setup_ms': (t0-acquired)/1e6, 'input_prepare_upload_ms': (t1-t0)/1e6,
                        'model_wall_ms': (t2-t1)/1e6, 'model_stream_ms': events[0].elapsed_time(events[1]),
                        'download_postprocess_ms': (t3-t2)/1e6, 'policy_wall_ms': (t3-t0)/1e6,
                        'handler_wall_ms': (t3-begin)/1e6}}
                self.send(result)
            except (ValueError, KeyError, TypeError) as error:
                self.send({'error': str(error)}, 400)
            except Exception as error:
                self.send({'error': str(error)}, 500)
                raise

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    if args.websocket_port:
        from websocket_bridge import serve
        thread = threading.Thread(target=serve, args=(args.host,args.websocket_port,f'http://{args.host}:{args.port}'),daemon=True)
        thread.start()
    print(json.dumps({'event': 'vla_ready', **metadata}), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
