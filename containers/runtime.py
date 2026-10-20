#!/usr/bin/env python3
"""Container entry point. Own child processes only; never manage host services."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz'
sys.path.insert(0, str(ROOT/'dual_stack'))
from clients import request_json
from assets import digest, verify


def prepare_assets(root):
    manifest = verify(root)
    cache = Path(os.environ.get('OPENPI_DATA_HOME', '/tmp/openpi-cache'))
    target = cache/'big_vision/paligemma_tokenizer.model'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/'tokenizer/paligemma_tokenizer.model', target)
    assert digest(FIXTURE) == json.loads((ROOT/'containers/pins.json').read_text())['fixture_sha256']
    return manifest


def doctor():
    import torch
    data = {'utc': datetime.now(timezone.utc).isoformat(), 'machine': platform.machine(),
            'python': sys.version, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'benchmark_commit': os.environ.get('BENCHMARK_COMMIT'),
            'pins': json.loads((ROOT/'containers/pins.json').read_text()),
            'gpu_validation': False}
    if platform.machine() != 'aarch64' or not torch.cuda.is_available():
        raise RuntimeError('This candidate requires aarch64 Thor with working NVIDIA GPU runtime')
    capability = torch.cuda.get_device_capability(0)
    if capability != (11, 0):
        raise RuntimeError('Thor candidate expects SM110, got '+str(capability))
    a = torch.randn((128, 128), device='cuda', dtype=torch.bfloat16)
    b = a @ a.T
    torch.cuda.synchronize()
    if not torch.isfinite(b).all().item():
        raise RuntimeError('BF16 CUDA smoke failed')
    data.update(gpu_validation=True, gpu=torch.cuda.get_device_name(0), capability=list(capability),
                memory_total_bytes=torch.cuda.get_device_properties(0).total_memory)
    return data


def commands(assets):
    return {
        'vla': [sys.executable, str(ROOT/'containers/vla_pytorch_server.py'), '--checkpoint',
                str(assets/'pi05_libero'), '--fixture', str(FIXTURE), '--host', '0.0.0.0',
                '--port', '8081', '--websocket-port', '8000'],
        'vlm': ['/opt/llama-build/bin/llama-server', '-m',
                str(assets/'Qwen3.5-9B/Qwen3.5-9B-Q5_K_M.gguf'), '--mmproj',
                str(assets/'Qwen3.5-9B/mmproj-F16.gguf'), '--host', '0.0.0.0', '--port', '8082',
                '-ngl', '99', '-c', '8192', '-np', '1', '-t', '4', '-tb', '4', '-b', '512',
                '-ub', '128', '--flash-attn', 'on', '--jinja', '--chat-template-kwargs',
                '{"enable_thinking":false}', '--no-cache-prompt', '--metrics', '--no-webui']}


class Models:
    def __init__(self, assets, logs):
        self.assets=assets; self.logs=Path(logs); self.processes={}; self.streams=[]
        self.logs.mkdir(parents=True, exist_ok=True)

    def start(self):
        for kind, command in commands(self.assets).items():
            stream=(self.logs/(kind+'-service.log')).open('w', encoding='utf-8'); self.streams.append(stream)
            proc=subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True)
            self.processes[kind]=proc
            deadline=time.monotonic()+900
            while time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise RuntimeError(kind+' failed; inspect '+str(self.logs))
                try:
                    request_json('http://127.0.0.1:'+('8081' if kind=='vla' else '8082')+'/health', timeout=2)
                    break
                except Exception:
                    time.sleep(.5)
            else:
                raise TimeoutError(kind+' readiness')
        return self

    def close(self):
        for proc in self.processes.values():
            if proc.poll() is None:proc.terminate()
        for proc in self.processes.values():
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill(); proc.wait()
        for stream in self.streams:stream.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', nargs='?', default='doctor', choices=['doctor','verify-assets','serve','baseline','benchmark','smoke'])
    p.add_argument('--assets', type=Path, default=Path('/assets'))
    p.add_argument('--output', type=Path, default=Path('/results/new-run'))
    p.add_argument('--phase-seconds', type=float, default=60)
    p.add_argument('--concurrent-seconds', type=float, default=180)
    p.add_argument('--idle-seconds', type=float, default=30)
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--warmup', type=int, default=10)
    args=p.parse_args()
    if args.command=='verify-assets':print(json.dumps(verify(args.assets), indent=2)); return
    environment=doctor()
    if args.command=='doctor':print(json.dumps(environment, indent=2)); return
    manifest=prepare_assets(args.assets)
    if args.command=='baseline':
        subprocess.run([sys.executable, str(ROOT/'baseline/benchmark.py'), '--deployment-root',
                        '/opt/pi05-deployment', '--checkpoint', str(args.assets/'pi05_libero'),
                        '--iterations',str(args.iterations),'--warmup',str(args.warmup),
                        '--output', str(args.output)], check=True, cwd=ROOT)
        return
    if args.command=='benchmark':
        from benchmark_container import run
        run(args, environment, manifest); return
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    logs=args.output/'service-logs'; models=Models(args.assets, logs)
    def stop(signum, frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        models.start()
        if args.command=='smoke':
            subprocess.run([sys.executable, str(ROOT/'dual_stack/smoke_test.py'), '--websocket',
                            '--check-changed-prompt', '--output', str(args.output/'api-smoke.json')],
                           check=True, cwd=ROOT)
            (args.output/'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
            return
        print(json.dumps({'ready':True, 'pids':{k:v.pid for k,v in models.processes.items()},
                          'vla_backend':'PyTorch BF16; FlashRT Thor not enabled'}), flush=True)
        while True:
            if any(proc.poll() is not None for proc in models.processes.values()):
                raise RuntimeError('A model child exited; inspect container logs '+str(logs))
            time.sleep(1)
    finally:
        models.close()


if __name__=='__main__':
    main()
