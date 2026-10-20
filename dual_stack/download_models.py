#!/usr/bin/env python3
"""Download only the two pinned public model files; resume and verify SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def parallel_download(url, dest, size, workers):
    parts = dest.with_name(dest.name + '.parts')
    parts.mkdir(exist_ok=True)
    chunk = 64 * 1024 * 1024
    def fetch(index):
        start, end = index * chunk, min((index + 1) * chunk, size) - 1
        path = parts / f'{index:04d}'
        expected = end - start + 1
        if path.is_file() and path.stat().st_size == expected:
            return expected
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--retry', '12', '--retry-all-errors', '--retry-delay', '2',
                        '--connect-timeout', '20', '--max-time', '300',
                        '--range', f'{start}-{end}', '--output', str(path),
                        url + f'&part={index}'], check=True)
        if path.stat().st_size != expected:
            raise RuntimeError(f'Range length mismatch: {index}')
        return expected
    count = (size + chunk - 1) // chunk
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, index) for index in range(count)]
        for done, job in enumerate(as_completed(jobs), 1):
            job.result()
            print(json.dumps({'event': 'download_parts', 'filename': dest.name, 'complete': done, 'total': count}), flush=True)
    temporary = dest.with_name(dest.name + '.assembled')
    with temporary.open('wb') as target:
        for index in range(count):
            with (parts / f'{index:04d}').open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    target.write(block)
    temporary.replace(dest)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--endpoint', default='https://huggingface.co', choices=['https://huggingface.co', 'https://hf-mirror.com'])
    args = p.parse_args()
    source = json.loads(Path(__file__).with_name('sources.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for item in source['files']:
        dest = args.output / item['filename']
        if not (dest.is_file() and dest.stat().st_size == item['size_bytes']):
            url = f"{args.endpoint}/{source['gguf_repository']}/resolve/{source['gguf_revision']}/{item['filename']}?download=true"
            print(json.dumps({'event': 'download_start', 'filename': item['filename'], 'bytes': item['size_bytes']}), flush=True)
            parallel_download(url, dest, item['size_bytes'], args.workers)
        actual = digest(dest)
        if dest.stat().st_size != item['size_bytes'] or actual != item['sha256']:
            raise RuntimeError(f"Model file identity mismatch: {item['filename']}")
        print(json.dumps({'event': 'verified', 'filename': item['filename'], 'sha256': actual}), flush=True)
    source['download_endpoint'] = args.endpoint
    (args.output / 'verified_sources.json').write_text(json.dumps(source, indent=2) + '\n')


if __name__ == '__main__':
    main()
