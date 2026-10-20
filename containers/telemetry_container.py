"""Best-effort read-only NVIDIA telemetry; unsupported sensors remain unavailable."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time


def number(value):
    try:return float(value.strip())
    except (ValueError,AttributeError):return None


def kib_fields(text):
    result={}
    for line in text.splitlines():
        parts=line.split()
        if len(parts)==3 and parts[2]=='kB' and parts[1].isdigit():
            result[parts[0].rstrip(':')]=int(parts[1])*1024
    return result


def sensors():
    result={}
    for folder in Path('/sys/class/hwmon').glob('hwmon*'):
        try:name=(folder/'name').read_text().strip()
        except OSError:continue
        for pattern in ['power*_input','in*_input','curr*_input','temp*_input']:
            for node in folder.glob(pattern):
                try:
                    label=node.with_name(node.name.replace('_input','_label'))
                    result[str(node)]={'device':name,'raw':node.read_text().strip(),
                                       'label':label.read_text().strip() if label.exists() else None}
                except OSError:pass
    # Do not sum rails without verifying this board's sensor topology and units.
    return result


class Monitor:
    def __init__(self, output, models):
        self.output=Path(output); self.models=models; self.phase='initial'; self.done=threading.Event()
        self.error=None; self.thread=None

    def sample(self):
        row={'monotonic_ns':time.monotonic_ns(),'unix_ns':time.time_ns(),'phase':self.phase,
             'host_meminfo_raw':Path('/proc/meminfo').read_text(),'processes':{},'hwmon':sensors()}
        for kind,proc in list(self.models.processes.items()):
            try:
                base=Path('/proc')/str(proc.pid)
                row['processes'][kind]={'pid':proc.pid,'status_raw':(base/'status').read_text(),
                                        'smaps_rollup_raw':(base/'smaps_rollup').read_text()}
            except OSError as error:row['processes'][kind]={'pid':proc.pid,'unavailable':str(error)}
        keys=['index','uuid','utilization.gpu','memory.used','memory.total','temperature.gpu','power.draw','clocks.sm','clocks.mem']
        if shutil.which('nvidia-smi'):
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu='+','.join(keys),'--format=csv,noheader,nounits'],
                                 capture_output=True,text=True,timeout=3)
                row['nvidia_smi']={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
                if p.returncode==0:
                    row['devices']=[dict(zip(keys,[v.strip() if k=='uuid' else number(v) for k,v in zip(keys,values)]))
                                    for values in csv.reader(p.stdout.splitlines()) if len(values)==len(keys)]
            except (OSError,subprocess.TimeoutExpired) as error:row['nvidia_smi']={'unavailable':str(error)}
        else:row['nvidia_smi']={'unavailable':'nvidia-smi not exposed by container runtime'}
        row['scope']='GPU device/global host sensors; process proc entries only for own children. No NvMap assumption.'
        return row

    def start(self):
        def worker():
            try:
                with (self.output/'telemetry.jsonl').open('w',encoding='utf-8') as f:
                    while not self.done.is_set():
                        begin=time.monotonic(); value=self.sample()
                        value['collection_ms']=(time.monotonic()-begin)*1000
                        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()
                        self.done.wait(max(0,1-(time.monotonic()-begin)))
            except Exception as error:self.error=repr(error)
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start()

    def stop(self):
        self.done.set()
        if self.thread:self.thread.join(timeout=10)
        if self.thread and self.thread.is_alive():self.error='Telemetry thread failed to stop'
