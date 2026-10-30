"""Read-only software, source/binary fingerprints and restored-service health."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/home/nvidia/pi05-orin-flashrt/vendor/FlashRT')
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from adapter import verify_patched_source
    import torch
    result={'created_unix':time.time(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
            'python':sys.version,'source_verification':verify_patched_source(root),'commands':{}}
    for name,command in {'nvpmodel':['sudo','-n','nvpmodel','-q'],
        'clocks':['sudo','-n','jetson_clocks','--show'],
        'nvcc':['/usr/local/cuda/bin/nvcc','--version'],
        'services':['systemctl','show','openpi-pi05-flashrt-dual','qwen35-vlm','-p','ActiveState','-p','MainPID'],
        'vendor_status':['git','-C',str(root),'status','--short']}.items():
        process=subprocess.run(command,text=True,capture_output=True,timeout=20)
        result['commands'][name]={'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
    result['native_extensions_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
                                        for f in (root/'flash_rt').glob('*.so')}
    import numpy as np
    new_reference=a.output.parent/'references/openpi_2valid_10steps.npy'
    old_reference=Path('/home/nvidia/pi05-orin-benchmarks/reports/engine-comparison-2026-10-16/baseline-latency/reference_actions.npy')
    if new_reference.is_file() and old_reference.is_file():
        new=np.load(new_reference,allow_pickle=False)[0]
        old=np.load(old_reference,allow_pickle=False)
        result['historical_reference_check']={'bitwise_equal':bool(np.array_equal(new,old)),
             'max_abs':float(np.max(np.abs(new.astype(np.float64)-old.astype(np.float64))))}
    result['health']={}
    deadline=time.monotonic()+180
    pending={8081,8082}
    while pending and time.monotonic()<deadline:
        for port in tuple(pending):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:
                    result['health'][str(port)]={'status':response.status,'body':response.read().decode()}
                pending.remove(port)
            except Exception as error:result['health'][str(port)]={'error':repr(error)}
        if pending:time.sleep(2)
    result['restored_services_healthy']=not pending
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'restored_services_healthy':not pending,'health':result['health']}),flush=True)
    if pending:raise RuntimeError('Original service health did not recover')


if __name__=='__main__':main()
