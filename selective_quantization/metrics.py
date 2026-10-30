"""Numerical agreement and read-only telemetry for the quantization study.

Cosine is numerical agreement with a reference action tensor, never task success.
RSS, NvMap and CUDA/Torch accounting domains overlap and must not be summed.
The sampler performs no CUDA calls; explicit Torch snapshots belong outside the
latency pass. Run the monitor only during the separate sustained-load pass.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time


def stats(values):
    """Latency statistics in milliseconds, with JSON-safe empty results."""
    import numpy as np
    values = np.asarray(list(values), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite timing sample")
    names = ("mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "std_ms")
    if values.size == 0:
        return {"n": 0, **{name: None for name in names}}
    return {"n": int(values.size), "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
            "min_ms": float(values.min()), "max_ms": float(values.max()),
            "std_ms": float(values.std())}


def _agreement(actual, expected):
    import numpy as np
    a, b = actual.ravel(), expected.ravel()
    delta = a - b
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(a @ b / denominator) if denominator else (1.0 if np.array_equal(a, b) else 0.0)
    return {"cosine": max(-1.0, min(1.0, cosine)),
            "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "p95_abs": float(np.percentile(np.abs(delta), 95)),
            "exact_match": bool(np.array_equal(a, b))}


def accuracy(actual, expected):
    """Compare final physical action tensors [10,7] or [N,10,7].

    The final coordinate is reported separately without assuming a robot's
    gripper convention. Threshold mismatches at 0 and 0.5 are merely numerical
    diagnostics; neither is labelled as grasp success or a correct decision.
    """
    import numpy as np
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.ndim not in (2, 3) or actual.shape[-2:] != (10, 7):
        raise ValueError(f"Expected matching [10,7] or [N,10,7], got {actual.shape}/{expected.shape}")
    if not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Action tensors must be nonempty and finite")
    result = {"shape": list(actual.shape), "all_finite": True,
              "scope": "numerical_action_agreement_not_robot_task_success",
              **_agreement(actual, expected)}
    aa = actual[None] if actual.ndim == 2 else actual
    bb = expected[None] if expected.ndim == 2 else expected
    result["per_case"] = [{"case_index": i, **_agreement(a, b)} for i, (a, b) in enumerate(zip(aa, bb))]
    result["per_action_dim"] = [{"dimension": i, **_agreement(aa[..., i], bb[..., i])} for i in range(7)]
    a, b = aa[..., 6], bb[..., 6]
    threshold_rows = {}
    for threshold in (0.0, 0.5):
        mismatches = (a > threshold) != (b > threshold)
        threshold_rows[str(threshold)] = {"count": int(mismatches.sum()),
                                          "fraction": float(mismatches.mean())}
    result["gripper"] = {"dimension": 6, **_agreement(a, b),
                         "threshold_mismatches": threshold_rows,
                         "threshold_scope": "diagnostic_only_no_robot_gripper_convention_assumed"}
    return result


def _read(path):
    try:
        # Some Jetson thermal sysfs nodes transiently return None/EAGAIN.
        # TextIOWrapper can raise TypeError while concatenating its decoder
        # buffer with that None. Read binary and handle absent data directly.
        with Path(path).open("rb", buffering=0) as stream:
            value = stream.read()
        return value.decode("utf-8", errors="replace").strip() if value is not None else None
    except (OSError, UnicodeError, TypeError):
        return None


def _kib(text):
    return {m[0]: int(m[1]) * 1024 for m in re.findall(r"^([A-Za-z_]+):\s+(\d+)\s+kB\s*$", text or "", re.M)}


def parse_nvmap(text, pid):
    if "PID" not in text or "SIZE" not in text:
        raise ValueError("Unrecognized NvMap clients table")
    clients = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\S+)\s+(.+?)\s+(\d+)\s+(\d+)([KMGT]?)\s*$", line)
        if match and int(match[3]) == pid:
            multiplier = 1024 ** (" KMGT".index(match[5]) if match[5] else 0)
            clients.append({"client": match[1], "process": match[2], "pid": pid,
                            "bytes": int(match[4]) * multiplier})
    return {"available": True, "self_bytes": sum(row["bytes"] for row in clients),
            "self_clients": clients, "raw": text}


def parse_tegrastats(line):
    result = {"raw": line.strip()}
    ram = re.search(r"RAM (\d+)/(\d+)MB", line)
    if ram:
        result.update(ram_used_bytes=int(ram[1]) * 1024 ** 2, ram_total_bytes=int(ram[2]) * 1024 ** 2)
    gpu = re.search(r"GR3D_FREQ (\d+)%", line)
    if gpu:
        result["gpu_load_percent"] = int(gpu[1])
    cpu = re.search(r"CPU \[([^]]+)\]", line)
    if cpu:
        result["cpu_core_load_percent"] = [int(m[1]) if (m := re.search(r"(\d+)%", part)) else None
                                             for part in cpu[1].split(",")]
    result["temperatures_c"] = {name: float(value) for name, value in re.findall(r"([A-Za-z0-9_]+)@(-?[\d.]+)C", line)}
    rails = {name: int(value) / 1000 for name, value in re.findall(r"([A-Za-z0-9_]+) (\d+)mW/\d+mW", line)}
    result["power_rails_w"] = rails
    names = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
    result["main_rails_sum_w"] = sum(rails[name] for name in names) if all(name in rails for name in names) else None
    return result


def _thermal(unavailable=None):
    result = {}
    aliases = {"gpu-thermal": "gpu", "cpu-thermal": "cpu", "tj-thermal": "tj", "tboard-thermal": "tboard"}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        kind = _read(zone / "type")
        # Unpowered CV and other unrelated thermal domains can have unreadable
        # temp nodes. Query only the four declared experiment sensors.
        if kind not in aliases:
            continue
        value = _read(zone / "temp")
        try:
            if kind is not None and value is not None:
                result[aliases.get(kind, kind)] = int(value) / 1000.0
            elif unavailable is not None:
                unavailable[aliases.get(kind, kind) if kind else str(zone)] = "sensor returned no data or could not be read"
        except (ValueError, TypeError) as error:
            if unavailable is not None:
                unavailable[aliases.get(kind, kind) if kind else str(zone)] = str(error)
    return result


def _oc_counters():
    result = {}
    for path in Path("/sys/devices/platform/soctherm-oc-event/hwmon").glob("hwmon*/oc*_event_cnt"):
        try:
            result[path.name] = int(path.read_text().strip())
        except (OSError, ValueError):
            pass
    return result


def _frequencies():
    values = {}
    paths = {"cpu0_hz": ("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000)}
    for path in Path("/sys/class/devfreq").glob("*ga10b*/cur_freq"):
        paths["gpu_hz"] = (path, 1)
    for name, (path, multiplier) in paths.items():
        value = _read(path)
        try:
            if value is not None:
                values[name] = int(value) * multiplier
        except ValueError:
            pass
    return values


def _distribution(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "start": None, "end": None}
    return {"n": len(values), "min": min(values), "max": max(values),
            "mean": sum(values) / len(values), "start": values[0], "end": values[-1]}


class MetricsMonitor:
    """Background OS-only sampler. Never calls Torch/CUDA from its threads."""
    def __init__(self, output, interval=0.5, nvmap_interval=1.0):
        if interval < 0.1 or nvmap_interval < interval:
            raise ValueError("Require interval>=0.1 and nvmap_interval>=interval")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.interval, self.nvmap_interval = interval, nvmap_interval
        self.pid = os.getpid()
        self.phase = "initial"
        self.rows, self.checkpoints, self.errors = [], [], []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest_tegrastats = None
        self._last_nvmap = {"available": False, "reason": "not_sampled"}
        self._last_nvmap_ns = None
        self._nvmap_command = None
        self._nvmap_direct = False
        self._started = False
        self._stopped = False

    def _probe_nvmap(self):
        path = "/sys/kernel/debug/nvmap/iovmm/clients"
        text = _read(path)
        if text is not None:
            try:
                parse_nvmap(text, self.pid)
                self._nvmap_direct = True
                return
            except ValueError:
                pass
        if os.name != "posix" or not shutil.which("sudo"):
            self._last_nvmap = {"available": False, "reason": "NvMap/sudo unavailable"}
            return
        try:
            command = ["sudo", "-n", "cat", path]
            result = subprocess.run(command, capture_output=True, timeout=2)
            if result.returncode:
                raise RuntimeError((result.stderr or b"").decode(errors="replace").strip())
            parse_nvmap((result.stdout or b"").decode(errors="replace"), self.pid)
            self._nvmap_command = command
        except (OSError, ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
            self._last_nvmap = {"available": False, "reason": str(error)}

    def _nvmap(self):
        try:
            if self._nvmap_direct:
                return parse_nvmap(Path("/sys/kernel/debug/nvmap/iovmm/clients").read_text(), self.pid)
            if self._nvmap_command:
                result = subprocess.run(self._nvmap_command, capture_output=True, timeout=2)
                if result.returncode:
                    raise RuntimeError((result.stderr or b"").decode(errors="replace").strip())
                return parse_nvmap((result.stdout or b"").decode(errors="replace"), self.pid)
        except (OSError, ValueError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
            return {"available": False, "reason": str(error)}
        return self._last_nvmap

    def start(self):
        if self._started:
            raise RuntimeError("Monitor already started")
        self._started = True
        self._journal = (self.output / "telemetry.jsonl").open("w", encoding="utf-8")
        self._probe_nvmap()
        self._tegrastats_process = None
        self._tegrastats_thread = None
        if shutil.which("tegrastats"):
            try:
                self._tegrastats_process = subprocess.Popen(
                    ["tegrastats", "--interval", str(round(self.interval * 1000))],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
                self._tegrastats_thread = threading.Thread(target=self._read_tegrastats, daemon=True)
                self._tegrastats_thread.start()
            except OSError as error:
                self.errors.append("tegrastats: " + str(error))
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def _read_tegrastats(self):
        try:
            with (self.output / "tegrastats.log").open("wb") as raw:
                while not self._stop.is_set():
                    line_bytes = self._tegrastats_process.stdout.readline()
                    if line_bytes is None:
                        # Defensive EAGAIN handling if an embedding host has
                        # changed descriptor flags; no TextIOWrapper is used.
                        self._stop.wait(0.02)
                        continue
                    if not line_bytes:
                        break
                    raw.write(line_bytes)
                    raw.flush()
                    line = line_bytes.decode("utf-8", errors="replace")
                    if "RAM " in line:
                        with self._lock:
                            self._latest_tegrastats = {"sample_monotonic_ns": time.monotonic_ns(), **parse_tegrastats(line)}
        except Exception as error:
            self.errors.append("tegrastats_reader: " + repr(error))

    def set_phase(self, phase):
        with self._lock:
            self.phase = str(phase)

    def _sample(self):
        begin = time.monotonic_ns()
        with self._lock:
            phase, tegra = self.phase, self._latest_tegrastats
        status, mem = _kib(_read("/proc/self/status")), _kib(_read("/proc/meminfo"))
        component_errors = {}

        def safe_component(name, function, default):
            try:
                return function()
            except Exception as error:
                component_errors[name] = repr(error)
                message = f"component {name}: {error!r}"
                if message not in self.errors:
                    self.errors.append(message)
                return default

        if self._last_nvmap_ns is None or (begin - self._last_nvmap_ns) / 1e9 >= self.nvmap_interval:
            self._last_nvmap = safe_component("nvmap", self._nvmap,
                {"available": False, "reason": "sampling component failed; see component_errors"})
            self._last_nvmap_ns = time.monotonic_ns()
        thermal_unavailable = {}
        temperatures = safe_component("thermal", lambda: _thermal(thermal_unavailable), {})
        row = {"created_utc": datetime.now(timezone.utc).isoformat(), "unix_ns": time.time_ns(),
               "monotonic_ns": begin, "phase": phase, "pid": self.pid,
               "temperatures_c": temperatures, "thermal_unavailable": thermal_unavailable,
               "oc_counters": safe_component("oc_counters", _oc_counters, {}),
               "frequencies": safe_component("frequencies", _frequencies, {}), "tegrastats": tegra,
               "process": {"rss_bytes": status.get("VmRSS"), "lifetime_rss_hwm_bytes": status.get("VmHWM"),
                           "swap_bytes": status.get("VmSwap")},
               "system_memory": {"total_bytes": mem.get("MemTotal"), "available_bytes": mem.get("MemAvailable"),
                                 "free_bytes": mem.get("MemFree"), "cached_bytes": mem.get("Cached"),
                                 "swap_total_bytes": mem.get("SwapTotal"), "swap_free_bytes": mem.get("SwapFree")},
               "nvmap": self._last_nvmap, "nvmap_sample_monotonic_ns": self._last_nvmap_ns,
               "component_errors": component_errors}
        row["collection_ms"] = (time.monotonic_ns() - begin) / 1e6
        self.rows.append(row)
        self._journal.write(json.dumps(row, allow_nan=False) + "\n")
        self._journal.flush()

    def _sample_loop(self):
        while not self._stop.is_set():
            begin = time.monotonic()
            try:
                self._sample()
            except Exception as error:
                self.errors.append("sampler: " + repr(error))
            self._stop.wait(max(0, self.interval - (time.monotonic() - begin)))

    def snapshot_torch(self, torch_module, label):
        """Call explicitly on the main thread outside timers; no synchronize."""
        result = {"label": str(label), "monotonic_ns": time.monotonic_ns(), "torch": None}
        if torch_module.cuda.is_initialized():
            result["torch"] = {"allocated_bytes": int(torch_module.cuda.memory_allocated()),
                               "reserved_bytes": int(torch_module.cuda.memory_reserved()),
                               "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
                               "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved())}
            free, total = torch_module.cuda.mem_get_info()
            result["cuda_device"] = {"free_bytes": int(free), "total_bytes": int(total), "used_bytes": int(total - free)}
        self.checkpoints.append(result)
        with (self.output / "memory_checkpoints.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        return result

    def stop(self):
        if not self._started or self._stopped:
            return
        self._stop.set()
        self._thread.join(timeout=6)
        if self._thread.is_alive():
            raise RuntimeError("Telemetry sampler failed to stop")
        process = self._tegrastats_process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if self._tegrastats_thread:
            self._tegrastats_thread.join(timeout=3)
        self._journal.close()
        self._stopped = True

    def summary(self, phase=None):
        rows = [row for row in self.rows if phase is None or row["phase"] == phase]
        sensors = sorted({name for row in rows for name in row["temperatures_c"]})
        result = {"phase": phase, "sample_count": len(rows), "interval_s": self.interval,
                  "nvmap_interval_s": self.nvmap_interval,
                  "duration_s": (rows[-1]["monotonic_ns"] - rows[0]["monotonic_ns"]) / 1e9 if len(rows) > 1 else 0,
                  "temperatures_c": {name: _distribution([row["temperatures_c"].get(name) for row in rows]) for name in sensors},
                  "process_rss_bytes": _distribution([row["process"].get("rss_bytes") for row in rows]),
                  "process_lifetime_rss_hwm_bytes": _distribution([row["process"].get("lifetime_rss_hwm_bytes") for row in rows]),
                  "nvmap_self_bytes": _distribution([row["nvmap"].get("self_bytes") for row in rows if row["nvmap"].get("available")]),
                  "system_available_bytes": _distribution([row["system_memory"].get("available_bytes") for row in rows]),
                  "tegrastats_ram_used_bytes": _distribution([(row["tegrastats"] or {}).get("ram_used_bytes") for row in rows]),
                  "gpu_load_percent": _distribution([(row["tegrastats"] or {}).get("gpu_load_percent") for row in rows]),
                  "main_rails_sum_w": _distribution([(row["tegrastats"] or {}).get("main_rails_sum_w") for row in rows]),
                  "collection_ms": _distribution([row["collection_ms"] for row in rows]),
                  "oc_counters_start": rows[0]["oc_counters"] if rows else {},
                  "oc_counters_end": rows[-1]["oc_counters"] if rows else {},
                  "memory_checkpoints": list(self.checkpoints), "errors": list(self.errors),
                  "accounting_note": "RSS, NvMap, Torch allocator and CUDA device usage overlap; do not sum them.",
                  "temperature_note": "Observed during this measurement window; not a thermal steady-state claim.",
                  "telemetry_note": "OS-only background samples; Torch/CUDA snapshots are explicit and outside timers."}
        return result
