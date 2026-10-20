#!/usr/bin/env python3
"""Export only pinned model assets; never copy home directories or credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    pins = json.loads((HERE/'pins.json').read_text(encoding='utf-8'))
    result = {}
    for name, expected in pins['assets'].items():
        p = Path(root)/name
        actual = digest(p)
        if actual != expected:
            raise ValueError('Pinned asset hash mismatch: '+name)
        result[name] = {'bytes': p.stat().st_size, 'sha256': actual}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    v = sub.add_parser('verify'); v.add_argument('root', type=Path)
    e = sub.add_parser('export'); e.add_argument('--output', required=True, type=Path)
    e.add_argument('--pi05', type=Path, default=Path('/home/nvidia/pi05-orin/models/pi05_libero'))
    e.add_argument('--qwen', type=Path, default=Path('/home/nvidia/pi05-orin-dual/models/Qwen3.5-9B'))
    e.add_argument('--tokenizer', type=Path, default=Path('/home/nvidia/pi05-orin/cache/big_vision/paligemma_tokenizer.model'))
    args = p.parse_args()
    if args.command == 'verify':
        result = verify(args.root)
    else:
        if args.output.exists():
            raise FileExistsError('Use a new output directory: '+str(args.output))
        pins = json.loads((HERE/'pins.json').read_text(encoding='utf-8'))
        sources = {}
        for name, expected in pins['assets'].items():
            if name.startswith('pi05_libero/'):
                source = args.pi05/Path(name).relative_to('pi05_libero')
            elif name.startswith('Qwen3.5-9B/'):
                source = args.qwen/Path(name).relative_to('Qwen3.5-9B')
            else:
                source = args.tokenizer
            if digest(source) != expected:
                raise ValueError('Source asset hash mismatch: '+name)
            sources[name] = source
        args.output.mkdir(parents=True)
        for name, source in sources.items():
            target = args.output/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        result = verify(args.output)
        (args.output/'asset-manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'verified': True, 'files': result}, indent=2))


if __name__ == '__main__':
    main()
