"""Read-only deployment provenance and hardware event counters."""
from pathlib import Path
import platform
import subprocess


def oc_counters():
    result = {}
    for path in Path('/sys/devices/platform/soctherm-oc-event/hwmon').glob('hwmon*/oc*_event_cnt'):
        try:
            result[path.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            pass
    return result


def snapshot():
    commands = {
        'power_mode': ['sudo', '-n', 'nvpmodel', '-q'],
        'clocks_read_only': ['sudo', '-n', 'jetson_clocks', '--show'],
        'l4t': ['cat', '/etc/nv_tegra_release'],
        'cuda_compiler': ['/usr/local/cuda/bin/nvcc', '--version'],
        'cmake': ['/home/nvidia/pi05-orin-flashrt/.venv/bin/cmake', '--version'],
        'flashrt_packages': ['/home/nvidia/.local/bin/uv', 'pip', 'freeze', '--python',
                             '/home/nvidia/pi05-orin-flashrt/.venv/bin/python'],
        'process_inventory': ['ps', '-eo', 'pid,comm,pcpu,pmem'],
    }
    result = {'platform': platform.platform(), 'oc_counters': oc_counters(), 'commands': {}}
    for name, command in commands.items():
        value = subprocess.run(command, capture_output=True, text=True, timeout=30)
        result['commands'][name] = {'command': command, 'returncode': value.returncode,
                                    'stdout': value.stdout, 'stderr': value.stderr}
    return result
