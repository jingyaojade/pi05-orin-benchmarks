"""Prepare recorded LIBERO images with reproducible, explicitly supplied noise."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--asset', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = []
    with np.load(a.legacy, allow_pickle=False) as f:
        legacy = {k: f[k].copy() for k in f.files}
    cases.append(dict(legacy, third_image=legacy['image'].copy(), name='legacy_synthetic'))
    frames = []
    with np.load(a.asset, allow_pickle=False) as f:
        for i in range(int(f['n'])):
            frames.append({'asset_frame': i, 'task': int(f[f'task_{i}']),
                           'global_index': int(f[f'global_index_{i}'])})
            for sample in range(2):
                seed = 2026103000 + i*10 + sample
                rng = np.random.default_rng(seed)
                cases.append({'image': f[f'img_{i}'].copy(),
                    'wrist_image': f[f'wrist_{i}'].copy(),
                    'third_image': f[f'img_{i}'].copy(),
                    'state': f[f'state_{i}'].copy(),
                    'noise': rng.standard_normal((10,32)).astype(np.float32),
                    'prompt': np.asarray('pick up the black bowl'),
                    'name': f'libero_frame_{i}_noise_{sample}'})
    values={k:np.stack([c[k] for c in cases]) for k in
            ('image','wrist_image','third_image','state','noise','prompt')}
    values['names']=np.asarray([c['name'] for c in cases])
    np.savez_compressed(a.output/'cases.npz', **values)
    manifest={'cases':len(cases), 'recorded_frames':frames,
        'assets':[{ 'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in (a.asset,a.legacy)],
        'noise_seeds':'2026103000 + frame_index*10 + sample_index (0,1)',
        'prompt':'pick up the black bowl',
        'scope':'Recorded LIBERO images with a fixed benchmark prompt; not task-specific closed-loop episodes.',
        'third_camera':'Duplicate of front image, used ONLY for three-valid-view shape/numerical tests; not an acquired third camera.',
        'first_case':'Preserved historical synthetic fixture.'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__': main()
