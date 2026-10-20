#!/usr/bin/env python3
"""Exercise candidate activation, one-PID APIs, fixed-task boundary and rollback."""
import argparse
import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'dual_stack'))
from clients import request_json


def states():
    return {unit: {key: subprocess.run(['systemctl', key, unit], capture_output=True,
                                     text=True).stdout.strip()
                   for key in ['is-active', 'is-enabled']}
            for unit in ['openpi-vla-vlm-shared', 'openpi-pi05-flashrt-dual',
                         'qwen35-vlm', 'openpi-pi05']}


def wait_ready():
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            for port in [8081, 8082]:
                request_json(f'http://127.0.0.1:{port}/health', timeout=2)
            with urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2) as response:
                assert response.status == 200
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError('Model API readiness')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    args = p.parse_args()
    e = args.results.resolve() / 'evidence'
    record = {'passed': False, 'before': states()}
    try:
        subprocess.run(['bash', str(REPO/'shared_context/activate.sh'),
                        str(args.results.resolve())], check=True, cwd=REPO)
        record['deployed'] = states()
        record['metadata'] = request_json('http://127.0.0.1:8081/metadata')
        record['listeners'] = subprocess.check_output(
            ['ss', '-lntp', '( sport = :8081 or sport = :8082 or sport = :8000 )'], text=True)
        pid = record['metadata']['pid']
        assert record['listeners'].count(f'pid={pid},') == 3
        record['children'] = subprocess.run(['ps', '--ppid', str(pid), '-o',
                                            'pid,ppid,comm,args'], capture_output=True, text=True).stdout
        subprocess.run([sys.executable, str(REPO/'dual_stack/smoke_test.py'), '--websocket',
                        '--output', str(e/'deployed-api-smoke.json')], check=True, cwd=REPO)
        import numpy as np
        from PIL import Image
        with np.load(REPO/'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz') as f:
            fixture = {k: f[k] for k in f.files}
        def encoded(value):
            stream=io.BytesIO(); Image.fromarray(value).save(stream, format='PNG')
            return base64.b64encode(stream.getvalue()).decode()
        body = {'image_base64': encoded(fixture['image']),
                'wrist_image_base64': encoded(fixture['wrist_image']),
                'prompt': 'pick up the blue cube'}
        req = urllib.request.Request('http://127.0.0.1:8081/infer',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(req, timeout=30)
            raise AssertionError('Unsupported prompt was silently accepted')
        except urllib.error.HTTPError as error:
            assert error.code == 409
            record['changed_prompt_boundary'] = {'status': error.code,
                                                'response': json.loads(error.read())}
    finally:
        subprocess.run(['bash', str(REPO/'shared_context/rollback.sh')], check=True, cwd=REPO)
        wait_ready()
        record['restored'] = states()
        for unit in ['openpi-pi05-flashrt-dual', 'qwen35-vlm', 'openpi-pi05']:
            assert record['before'][unit] == record['restored'][unit]
        subprocess.run([sys.executable, str(REPO/'dual_stack/smoke_test.py'), '--websocket',
                        '--check-changed-prompt', '--output', str(e/'restored-api-smoke.json')],
                       check=True, cwd=REPO)
    record['passed'] = True
    (e/'lifecycle-validation.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n',
                                              encoding='utf-8')
    print(json.dumps({'lifecycle_passed': True}))


if __name__ == '__main__':
    main()
