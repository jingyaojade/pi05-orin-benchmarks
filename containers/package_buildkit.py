#!/usr/bin/env python3
"""Create a source-only Docker build kit from committed Git blobs, with checksums."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PREFIXES=('containers/','baseline/','deployment/','dual_stack/','flashrt/','profiling/','reporting/')
EXACT={'README.md','summary.md','standalone_tokenizer.py',
       'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz',
       'reports/engine-comparison-2026-10-16/baseline-latency/reference_actions.npy'}


def include(name):
    path=Path(name)
    return (name.startswith(PREFIXES) or name in EXACT) and not any(
        part in ('.git','__pycache__','local-artifacts') or part.startswith('.env') for part in path.parts
        ) and path.suffix not in ('.safetensors','.gguf','.whl','.so','.pem','.key','.pyc')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    def git(*arguments):return subprocess.check_output(['git','-C',str(ROOT),*arguments])
    commit=git('rev-parse','HEAD').decode().strip()
    names=[x for x in git('ls-tree','-r','--name-only','HEAD').decode().splitlines() if include(x)]
    if 'containers/Dockerfile.thor' not in names:raise RuntimeError('Commit the build kit before exporting it')
    content={name:git('show','HEAD:'+name) for name in names}
    manifest={'source_commit':commit,'git_tree':git('rev-parse','HEAD^{tree}').decode().strip(),
              'status':'source build kit; Thor GPU build/validation not performed',
              'files_sha256':{name:hashlib.sha256(data).hexdigest() for name,data in content.items()}}
    content['BUILD_KIT_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w') as tar:
            for name,data in sorted(content.items()):
                info=tarfile.TarInfo('pi05-qwen-thor-buildkit/'+name);info.size=len(data)
                info.mode=0o755 if name.endswith('.sh') else 0o644;info.mtime=0
                tar.addfile(info,io.BytesIO(data))
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix+'.sha256').write_text(digest+'  '+args.output.name+'\n',encoding='utf-8')
    print(json.dumps({'source_commit':commit,'files':len(content),'bytes':args.output.stat().st_size,
                      'sha256':digest,'archive':str(args.output)}))


if __name__=='__main__':main()
