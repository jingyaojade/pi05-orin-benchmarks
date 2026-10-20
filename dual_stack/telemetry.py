"""Read-only 1 Hz Jetson telemetry. Accounting domains must not be summed."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from environment import oc_counters

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'profiling'))
from memory_common import parse_kib_fields, parse_nvmap


def thermal_zones():
    """tegrastats can omit GPU temperature if its sensor was unavailable at startup."""
    aliases={'cpu-thermal':'cpu','gpu-thermal':'gpu','tj-thermal':'tj','tboard-thermal':'tboard'}
    values,raw={},{}
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            kind=(zone/'type').read_text().strip()
            if kind not in aliases:continue
            value=(zone/'temp').read_text().strip()
            raw[str(zone)]={'type':kind,'raw_millicelsius':value}
            values[aliases[kind]]=int(value)/1000
        except (OSError,ValueError,TypeError) as error:
            raw[str(zone)]={'unavailable':str(error)}
    return values,raw


def parse_tegrastats(line):
    result = {'raw': line.strip()}
    ram = re.search(r'RAM (\d+)/(\d+)MB', line)
    if ram:
        result['ram_used_mib'], result['ram_total_mib'] = map(int, ram.groups())
    gpu = re.search(r'GR3D_FREQ (\d+)%', line)
    if gpu:
        result['gpu_load_percent'] = int(gpu[1])
    result['temperatures_c'] = {name: float(value) for name, value in re.findall(r'([A-Za-z0-9_]+)@(-?[\d.]+)C', line)}
    rails = {name: float(value)/1000 for name, value in re.findall(r'([A-Za-z0-9_]+) (\d+)mW/\d+mW', line)}
    result['power_rails_w'] = rails
    # AGX Orin: VDDQ is already included in VIN_SYS_5V0. Exclude it and SUMMATION.
    names = ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0')
    result['main_rails_sum_w'] = sum(rails[k] for k in names) if all(k in rails for k in names) else None
    return result


def process_memory(pid):
    try:
        status = parse_kib_fields(Path(f'/proc/{pid}/status').read_text())
        rollup = Path(f'/proc/{pid}/smaps_rollup').read_text()
        values = parse_kib_fields(rollup)
        return {'pid': pid, 'rss_bytes': values.get('Rss', status.get('VmRSS')),
                'pss_bytes': values.get('Pss'), 'uss_bytes': values.get('Private_Clean', 0)+values.get('Private_Dirty', 0),
                'lifetime_rss_hwm_bytes': status.get('VmHWM'), 'swap_bytes': values.get('Swap'),
                'smaps_raw': rollup}
    except (OSError, ValueError) as error:
        return {'pid': pid, 'unavailable': str(error)}


class Monitor:
    def __init__(self, output):
        self.output = Path(output)
        self.phase = 'initial'
        self.pids = {}
        self.lock = threading.Lock()
        self.rows = []
        self.stream = None
        self.error = None

    def start(self):
        self.stream = (self.output/'telemetry.jsonl').open('w')
        self.process = subprocess.Popen(['tegrastats', '--interval', '1000'], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if 'RAM ' not in line:
                    continue
                begin = time.monotonic_ns()
                with self.lock:
                    phase, pids = self.phase, dict(self.pids)
                sample = {'unix_ns': time.time_ns(), 'monotonic_ns': begin, 'phase': phase,
                          **parse_tegrastats(line), 'processes': {k: process_memory(pid) for k,pid in pids.items()},
                          'meminfo_raw': Path('/proc/meminfo').read_text()}
                sample['oc_counters'] = oc_counters()
                direct,sensor_raw=thermal_zones()
                sample['tegrastats_temperatures_c']=dict(sample['temperatures_c'])
                sample['thermal_sysfs_raw']=sensor_raw
                sample['temperatures_c'].update(direct)
                mem = parse_kib_fields(sample['meminfo_raw'])
                sample['mem_available_bytes'] = mem.get('MemAvailable')
                sample['system_swap_used_bytes'] = mem.get('SwapTotal',0)-mem.get('SwapFree',0)
                nv = subprocess.run(['sudo','-n','cat','/sys/kernel/debug/nvmap/iovmm/clients'],
                                    capture_output=True, text=True, timeout=3)
                if nv.returncode == 0:
                    sample['nvmap_raw'] = nv.stdout
                    for key,pid in pids.items():
                        sample['processes'][key]['nvmap_bytes'] = parse_nvmap(nv.stdout, pid)['self_bytes']
                else:
                    sample['nvmap_error'] = nv.stderr.strip()
                sample['collection_ms'] = (time.monotonic_ns()-begin)/1e6
                self.rows.append(sample)
                self.stream.write(json.dumps(sample, allow_nan=False)+'\n')
                self.stream.flush()
        except Exception as error:
            self.error = repr(error)

    def set(self, phase, pids):
        with self.lock:
            self.phase, self.pids = phase, dict(pids)

    def stop(self):
        if not hasattr(self, 'process'):
            if self.stream:
                self.stream.close()
            return
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.thread.join(timeout=5)
        if self.stream:
            self.stream.close()
