"""CUDA device selection uses ordinals relative to the supervisor's visible set."""
import os
from pathlib import Path


def profile(environ=None):
    env=os.environ if environ is None else environ
    target=env.get('BENCHMARK_TARGET','thor')
    if target not in ('thor','nvidia'):raise ValueError('Unknown BENCHMARK_TARGET: '+target)
    return target


def pins_path(environ=None):
    root=Path(__file__).resolve().parent
    return root/'nvidia/pins.json' if profile(environ)=='nvidia' else root/'pins.json'


def bindings(environ=None):
    env=os.environ if environ is None else environ
    mask=env.get('CUDA_VISIBLE_DEVICES')
    visible=None if mask is None else [x.strip() for x in mask.split(',')]
    if visible is not None and (not all(visible) or '-1' in visible):
        raise ValueError('CUDA_VISIBLE_DEVICES exposes no valid GPU')
    result={}
    for kind in ['vla','vlm']:
        raw=env.get(kind.upper()+'_GPU','0')
        if not raw.isdigit():raise ValueError(kind.upper()+'_GPU must be a nonnegative visible-device ordinal')
        index=int(raw)
        if visible is not None and index>=len(visible):raise ValueError(kind+' GPU is outside CUDA_VISIBLE_DEVICES')
        result[kind]={'visible_ordinal':index,'cuda_visible_devices':visible[index] if visible is not None else str(index)}
    return result


def child_environment(kind,environ=None):
    env=dict(os.environ if environ is None else environ)
    env['CUDA_VISIBLE_DEVICES']=bindings(env)[kind]['cuda_visible_devices']
    return env
