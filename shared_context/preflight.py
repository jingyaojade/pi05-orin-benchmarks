#!/usr/bin/env python3
"""Prove CUDA context identity using CUPTI callbacks on actual model launches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dual_stack'))
from clients import request_json
from support import preserve_services,stop_process,wait_ready,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    repo=Path(__file__).resolve().parents[1]
    process=None
    with preserve_services(args.output):
        with (args.output/'service.log').open('w') as log:
            try:
                process=subprocess.Popen([sys.executable,str(repo/'shared_context/service.py')],stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='6',NO_PROXY='127.0.0.1,localhost'))
                wait_ready(process,'http://127.0.0.1:8081',request_json)
                metadata=request_json('http://127.0.0.1:8081/metadata');write(args.output/'metadata.json',metadata)
                request_json('http://127.0.0.1:8081/audit/start')
                subprocess.run([sys.executable,str(repo/'dual_stack/smoke_test.py'),'--output',str(args.output/'api-smoke.json')],check=True)
                audit=request_json('http://127.0.0.1:8081/audit/stop')
                write(args.output/'context-audit.json',audit)
                by_role={role:{x['context'] for x in audit['events'] if x['role']==role} for role in [1,2]}
                assert by_role[1]==by_role[2]=={metadata['cuda_context']},by_role
                assert any('GraphLaunch' in x['api'] and x['role']==1 for x in audit['events'])
                assert any('Launch' in x['api'] and x['role']==2 for x in audit['events'])
                assert not audit['discarded']
                write(args.output/'validation.json',{'passed':True,'pid':process.pid,'cuda_context':metadata['cuda_context'],
                     'launches_by_role':{str(role):sum(x['role']==role for x in audit['events']) for role in [1,2]},
                     'unique_contexts':sorted({x['context'] for x in audit['events']}),
                     'method':'CUPTI runtime/driver API launch callbacks, not a before/after context assumption'})
                print('SHARED_CONTEXT_PROVEN',flush=True)
            finally:stop_process(process)


if __name__=='__main__':main()
