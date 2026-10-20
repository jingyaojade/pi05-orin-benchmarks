"""Shared experiment lifecycle helpers; only manage this project's model services."""
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
import time

UNITS=['openpi-pi05-flashrt-dual','qwen35-vlm','openpi-pi05']


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def states():
    return {unit:{kind:subprocess.run(['systemctl',kind,unit],text=True,capture_output=True).stdout.strip()
                  for kind in ['is-active','is-enabled']} for unit in UNITS}


@contextmanager
def preserve_services(output):
    previous=states();write(Path(output)/'original-services.json',previous)
    active=[u for u,v in previous.items() if v['is-active'] in ('active','activating')]
    def interrupted(signum,frame):raise InterruptedError(f'Experiment interrupted by {signum}')
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
    try:
        if active:subprocess.run(['sudo','-n','systemctl','stop',*active],check=True)
        yield previous
    finally:
        # Attempt all restores even if one unit fails. No unrelated services are touched.
        for unit in active:
            result=subprocess.run(['sudo','-n','systemctl','start',unit])
            if result.returncode:
                reason=subprocess.check_output(['systemctl','show',unit,'-p','Result','--value'],text=True).strip()
                if reason=='start-limit-hit':
                    subprocess.run(['sudo','-n','systemctl','reset-failed',unit],check=True)
                    subprocess.run(['sudo','-n','systemctl','start',unit],check=True)
        write(Path(output)/'restored-services.json',states())


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=30)
        except subprocess.TimeoutExpired:process.kill();process.wait()


def wait_ready(process,url,request_json,timeout=900):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError(f'Service exited {process.returncode}: {url}')
        try:return request_json(url+'/health',timeout=2)
        except Exception:time.sleep(.25)
    raise TimeoutError('Service startup: '+url)
