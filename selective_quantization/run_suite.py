"""Isolate the experiment, run children sequentially, and restore active services."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


UNITS=('openpi-pi05','openpi-pi05-flashrt-dual','qwen35-vlm','pi05-qwen-shared')


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def cmd(arguments):
    p=subprocess.run(arguments,text=True,capture_output=True,timeout=30)
    return {'command':arguments,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--flashrt-python',default='/home/nvidia/pi05-orin-flashrt/.venv/bin/python')
    p.add_argument('--baseline-python',default='/home/nvidia/pi05-orin/.venv/bin/python')
    a=p.parse_args();root=a.root;code=Path(__file__).resolve().parent
    results=root/'results';results.mkdir(parents=True,exist_ok=True)
    configs=root/'configs';configs.mkdir(exist_ok=True)
    state={unit:subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True).stdout.strip() for unit in UNITS}
    provenance={'started_unix':time.time(),'service_state_before':state,'commands':{},'runs':[],
        'notes':['Original services and vendor source are restored/preserved; no clock settings are changed.',
                 'Temperatures describe 30-second sustained tests, not thermal equilibrium.',
                 'Cosine is numerical agreement, not robot task success.']}
    for label,command in {'power_mode':['sudo','-n','nvpmodel','-q'],
        'clocks':['sudo','-n','jetson_clocks','--show'],
        'l4t':['cat','/etc/nv_tegra_release'],
        'vendor_git':['git','-C','/home/nvidia/pi05-orin-flashrt/vendor/FlashRT','rev-parse','HEAD'],
        'packages':['/home/nvidia/.local/bin/uv','pip','freeze','--python',a.flashrt_python]}.items():
        provenance['commands'][label]=cmd(command)
    provenance['experiment_code_sha256']={str(f.relative_to(code.parent)):hashlib.sha256(f.read_bytes()).hexdigest()
                                         for f in code.glob('*.py')}
    env=dict(os.environ,OMP_NUM_THREADS='6',PYTHONUNBUFFERED='1',JAX_PLATFORMS='cpu',
             OPENPI_DATA_HOME='/home/nvidia/pi05-orin/cache')
    write(results/'provenance.json',provenance)
    def child(name,arguments,required=True):
        logfile=results/(name+'.log')
        print(json.dumps({'event':'start','name':name,'unix':time.time()}),flush=True)
        with logfile.open('w') as f:
            run=subprocess.run(arguments,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=code.parent)
        record={'name':name,'returncode':run.returncode,'log':str(logfile),'finished_unix':time.time()}
        provenance['runs'].append(record);write(results/'provenance.json',provenance)
        print(json.dumps({'event':'finished',**record}),flush=True)
        if required and run.returncode:raise RuntimeError(f'{name} failed: see {logfile}')
        return run.returncode
    signatures={}
    def variant(config):
        name=config['name'];values=dict(config);values.pop('name')
        key=json.dumps(values,sort_keys=True)
        if key in signatures:
            provenance.setdefault('aliases',{})[name]=signatures[key]
            write(results/'provenance.json',provenance);return
        signatures[key]=name
        path=configs/(name+'.json');write(path,config)
        valid=config.get('valid_num_views',2)
        teacher=results/('bf16_3valid_968' if valid==3 else 'bf16_2valid_968')/'actions.npy'
        child(name,[a.flashrt_python,str(code/'worker.py'),'--config',str(path),
             '--inputs',str(root/'inputs/cases.npz'),'--references',str(results/'references'),
             '--teacher',str(teacher),'--output',str(results/name)],required=False)

    def interrupted(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        active=[u for u,s in state.items() if s in ('active','activating')]
        if active:subprocess.run(['sudo','-n','systemctl','stop',*active],check=True)
        provenance['isolated_processes']=cmd(['ps','-eo','pid,pcpu,pmem,args'])
        write(results/'provenance.json',provenance)
        child('references',[a.baseline_python,str(code/'reference.py'),
              '--inputs',str(root/'inputs/cases.npz'),'--output',str(results/'references')])
        base={'num_views':3,'valid_num_views':2,'max_prompt_len':200,'num_steps':10}
        variant(dict(base,name='bf16_2valid_968'))
        child('screening',[a.flashrt_python,str(code/'screen.py'),
              '--inputs',str(root/'inputs/cases.npz'),'--output',str(results/'screening')])
        screen=json.loads((results/'screening/screening.json').read_text())
        enc=screen['selected_layers']['encoder'];dec=screen['selected_layers']['decoder']
        all_layers=list(range(18));ffn=['ffn_gate','ffn_up','ffn_down']
        quant=[{'name':'int8_encoder_all','encoder_layers':all_layers},
               {'name':'int8_decoder_all','decoder_layers':all_layers},
               {'name':'int8_both_all','encoder_layers':all_layers,'decoder_layers':all_layers},
               {'name':'int8_encoder_ffn','encoder_layers':all_layers,'modules':ffn},
               {'name':'int8_decoder_ffn','decoder_layers':all_layers,'modules':ffn},
               {'name':'int8_encoder_screened','encoder_layers':enc},
               {'name':'int8_decoder_screened','decoder_layers':dec},
               {'name':'int8_both_screened','encoder_layers':enc,'decoder_layers':dec}]
        for q in quant:variant(dict(base,**q))
        variant(dict(base,name='bf16_2valid_text_compact_775',max_prompt_len=7))
        variant(dict(base,name='bf16_2valid_full_compact_519',num_views=2,max_prompt_len=7))
        variant(dict(base,name='bf16_2valid_steps8',num_steps=8))
        variant(dict(base,name='bf16_2valid_steps5',num_steps=5))
        variant(dict(base,name='bf16_3valid_968',valid_num_views=3))
        variant(dict(base,name='bf16_3valid_compact_775',valid_num_views=3,max_prompt_len=7))
        # Screened combination gets an independent 17-case evaluation in a
        # fresh process; a single-layer screen does not certify the combination.
        variant(dict(base,name='int8_screened_compact_519',encoder_layers=enc,
                     decoder_layers=dec,num_views=2,max_prompt_len=7))
        # Independent control at the end exposes temperature/tactic drift.
        signatures.clear();variant(dict(base,name='bf16_2valid_968_end_control'))
        provenance['completed']=True
    except BaseException as error:
        provenance['fatal_error']=repr(error);raise
    finally:
        provenance['restore']={}
        for unit,status in state.items():
            if status in ('active','activating'):
                provenance['restore'][unit]=cmd(['sudo','-n','systemctl','start',unit])
        provenance['service_state_after']={u:subprocess.run(['systemctl','is-active',u],
            capture_output=True,text=True).stdout.strip() for u in UNITS}
        provenance['finished_unix']=time.time();write(results/'provenance.json',provenance)


if __name__=='__main__':main()
