#!/usr/bin/env python3
"""Separate diagnostic latency pass for FlashRT input/model/output boundaries.

This deliberately adds synchronization boundaries. Production latency remains
the result of flashrt/benchmark.py; these samples must not replace it.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flashrt-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=('bf16', 'int8'), default='bf16')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error('warmup must be nonnegative and iterations positive')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output must be empty to preserve measurements')
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flashrt'))
    import numpy as np
    import torch
    from adapter import LiberoFlashRT
    from benchmark import accuracy, statistics

    with np.load(args.fixture, allow_pickle=False) as archive:
        fixture = {key: archive[key] for key in archive.files}
    adapter = LiberoFlashRT(args.flashrt_root.resolve(), args.checkpoint.resolve(), args.precision)
    adapter.prepare(fixture)
    reference = adapter.infer(fixture['noise'])
    torch.cuda.synchronize()
    pair = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    raw = {'warmup': [], 'samples': [], 'interpretation': [
        'Separate synchronized diagnostic pass, not an untouched production policy measurement.',
        'Input preparation combines CPU RGB normalization/casts and noise/image transfers; it is not pure H2D.',
        'Prompt tokenization/embedding is persistent and was prepared before sampling, as in the FlashRT API.',
        'The original full production graph is replayed; no diagnostic stage graphs participate.',
        'Output download includes D2D staging, conversion, D2H, quantile unnormalization and seven-dimension slicing.',
        'CUDA elapsed time includes stream gaps and is not kernel busy time.',
    ]}
    for index in range(args.warmup + args.iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        adapter.stage_model_inputs(fixture['noise'], include_images=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        pair[0].record(adapter.model_stream)
        adapter.replay_model_graph()
        pair[1].record(adapter.model_stream)
        torch.cuda.synchronize()
        t2 = time.perf_counter_ns()
        actions = adapter.download_model_actions()
        torch.cuda.synchronize()
        t3 = time.perf_counter_ns()
        validation = accuracy(actions, reference, np)
        if not validation['exact_match']:
            raise AssertionError('Diagnostic boundary pass changed FlashRT output')
        sample = {'sample': index - args.warmup, 'timings': {
            'input_preparation_and_upload_wall_ms': (t1-t0)/1e6,
            'model_wall_ms': (t2-t1)/1e6,
            'model_cuda_stream_elapsed_ms': float(pair[0].elapsed_time(pair[1])),
            'download_and_postprocess_wall_ms': (t3-t2)/1e6,
            'diagnostic_policy_wall_ms': (t3-t0)/1e6,
        }, 'validation': validation}
        raw['warmup' if index < args.warmup else 'samples'].append(sample)
    summary = {key: statistics([row['timings'][key] for row in raw['samples']], np)
               for key in raw['samples'][0]['timings']}
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                  'precision': args.precision, 'warmup': args.warmup, 'iterations': args.iterations,
                  'tokenizer': adapter.tokenizer_provenance}
    for name, value in (('raw.json', raw), ('summary.json', summary), ('provenance.json', provenance)):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (out/'samples.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=('sample', 'metric', 'milliseconds'))
        writer.writeheader()
        for row in raw['samples']:
            for key, value in row['timings'].items():
                writer.writerow({'sample': row['sample'], 'metric': key, 'milliseconds': value})
    print(json.dumps({'event': 'pipeline_latency_complete', 'output': str(out), 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
