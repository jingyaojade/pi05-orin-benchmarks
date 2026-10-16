"""Read-only memory accounting for a separate, intentionally intrusive pass.

All sizes are bytes. These accounting domains overlap on Jetson; never add
RSS/PSS, nvmap and CUDA device usage together.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time


def read_optional(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return None


def parse_kib_fields(text):
    result = {}
    for line in (text or "").splitlines():
        match = re.match(r"^([A-Za-z_]+):\s+(\d+)\s+kB$", line)
        if match:
            result[match[1]] = int(match[2]) * 1024
    return result


def proc_memory():
    rollup = read_optional("/proc/self/smaps_rollup")
    status = read_optional("/proc/self/status")
    smaps = parse_kib_fields(rollup)
    fields = parse_kib_fields(status)
    return {
        "rss_bytes": smaps.get("Rss", fields.get("VmRSS")),
        "pss_bytes": smaps.get("Pss"),
        "uss_bytes": (smaps.get("Private_Clean", 0) + smaps.get("Private_Dirty", 0)) if rollup else None,
        "lifetime_rss_high_watermark_bytes": fields.get("VmHWM"),
        "virtual_bytes": fields.get("VmSize"),
        "swap_bytes": smaps.get("Swap", fields.get("VmSwap")),
        "smaps_rollup_raw": rollup,
        # Process name and identity are useful; full status is not necessary.
        "status_memory_fields_bytes": fields,
    }


def parse_nvmap(text, pid):
    """Parse the L4T clients table; one process can have multiple clients.

    Example: ``user python 63196 7713456K``. Kernel K means 1024 bytes.
    The process name is allowed to contain whitespace; PID/SIZE are trailing.
    An absent PID in a successfully read table means zero nvmap clients.
    """
    rows = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\S+)\s+(.+?)\s+(\d+)\s+(\d+)([KMGT]?)\s*$", line)
        if match:
            client, process, row_pid, size, suffix = match.groups()
            multiplier = 1024 ** (" KMGT".index(suffix) if suffix else 0)
            rows.append({"client": client, "process": process, "pid": int(row_pid),
                         "bytes": int(size) * multiplier})
    if "PID" not in text or "SIZE" not in text:
        raise ValueError("Unrecognized nvmap clients table")
    own = [row for row in rows if row["pid"] == pid]
    return {"pid": pid, "self_bytes": sum(row["bytes"] for row in own),
            "self_clients": own, "all_clients": rows, "raw": text}


class NvmapReader:
    def __init__(self, disabled=False):
        self.command = None
        self.unavailable = "disabled" if disabled else None
        if disabled:
            return
        path = "/sys/kernel/debug/nvmap/iovmm/clients"
        for command in (["cat", path], ["sudo", "-n", "cat", path]):
            try:
                result = subprocess.run(command, text=True, capture_output=True, timeout=5, check=False)
                if result.returncode == 0:
                    parse_nvmap(result.stdout, os.getpid())
                    self.command = command
                    return
                self.unavailable = result.stderr.strip()
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                self.unavailable = str(error)

    def read(self):
        if self.command is None:
            return {"available": False, "reason": self.unavailable}
        try:
            result = subprocess.run(self.command, text=True, capture_output=True, timeout=5, check=False)
            if result.returncode:
                return {"available": False, "reason": result.stderr.strip()}
            return {"available": True, **parse_nvmap(result.stdout, os.getpid())}
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            return {"available": False, "reason": str(error)}


def flatten(snapshot):
    """Stable numeric fields for CSV and summaries; all values are bytes."""
    result = {}
    for group, names in {
        "process": ("rss", "pss", "uss", "lifetime_rss_high_watermark", "swap", "virtual"),
        "torch": ("allocated", "reserved", "peak_allocated", "peak_reserved"),
        "cuda_device": ("used", "free", "total"),
        "nvmap": ("self",),
    }.items():
        for name in names:
            key = name + "_bytes"
            value = snapshot.get(group, {}).get(key)
            if value is not None:
                result[f"{group}_{key}"] = value
    return result


class MemoryRecorder:
    def __init__(self, output, interval=0.1, disable_nvmap=False):
        self.output = Path(output)
        self.interval = interval
        self.nvmap = NvmapReader(disable_nvmap)
        self.torch = None
        self.records = []
        self.checkpoints = []
        self.active_stage = None
        self.next_id = 0
        self.journal = (self.output / "snapshots.jsonl").open("w", encoding="utf-8")
        self.lock = threading.Lock()

    def snapshot(self, label, *, cuda=True):
        start = time.monotonic_ns()
        value = {"label": label, "created_utc": datetime.now(timezone.utc).isoformat(),
                 "monotonic_ns": start, "pid": os.getpid(), "process": proc_memory(),
                 "nvmap": self.nvmap.read()}
        torch = self.torch
        if cuda and torch is not None and torch.cuda.is_initialized():
            free, total = torch.cuda.mem_get_info()
            value["torch"] = {"allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            value["cuda_device"] = {"free_bytes": free, "total_bytes": total, "used_bytes": total - free}
        value["collection_elapsed_ms"] = (time.monotonic_ns() - start) / 1e6
        with self.lock:
            value["snapshot_id"] = self.next_id
            self.next_id += 1
            self.journal.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            self.journal.flush()
        return value

    def checkpoint(self, label):
        if self.torch is not None and self.torch.cuda.is_initialized():
            self.torch.cuda.synchronize()
        value = self.snapshot(label)
        self.checkpoints.append(value)
        print(json.dumps({"event": "memory_checkpoint", "label": label, **flatten(value)}), flush=True)
        return value

    @contextmanager
    def measure(self, stage, *, sample=None, iteration=None, sample_host=False):
        """Only nonnested regions; peak counters cannot be reset by children."""
        if self.active_stage is not None:
            raise RuntimeError(f"Nested peak reset forbidden: {stage} inside {self.active_stage}")
        self.active_stage = stage
        torch = self.torch
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        before = self.snapshot(stage + ":before")
        samples = []
        stop = threading.Event()

        def worker():
            while not stop.wait(self.interval):
                # No runtime calls/synchronization from the sampling thread.
                samples.append(self.snapshot(stage + ":sampled", cuda=False))

        thread = threading.Thread(target=worker, daemon=True) if sample_host else None
        if thread:
            thread.start()
        try:
            yield
        finally:
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            stop.set()
            if thread:
                thread.join(timeout=6)
                if thread.is_alive():
                    raise RuntimeError("Memory sampler did not stop")
            after = self.snapshot(stage + ":after")
            before_flat, after_flat = flatten(before), flatten(after)
            observed = {}
            for value in [before, *samples, after]:
                for key, number in flatten(value).items():
                    observed[key] = max(observed.get(key, number), number)
            exact_peak = after.get("torch", {}).get("peak_allocated_bytes")
            allocated_start = before.get("torch", {}).get("allocated_bytes")
            record = {"stage": stage, "sample": sample, "iteration": iteration,
                      "before": before, "after": after,
                      "sampled_host_snapshot_ids": [value["snapshot_id"] for value in samples],
                      "sample_count": len(samples), "max_observed_bytes": observed,
                      "torch_peak_allocated_bytes": exact_peak,
                      "torch_peak_above_start_bytes": max(0, exact_peak - allocated_start)
                          if exact_peak is not None and allocated_start is not None else None,
                      "endpoint_delta_bytes": {key: after_flat[key] - number for key, number in before_flat.items()
                                               if key in after_flat and "peak_" not in key and "high_watermark" not in key}}
            self.records.append(record)
            self.active_stage = None

    def save(self, metadata):
        self.journal.flush()
        write_json(self.output / "raw.json", {"schema_version": 1, "checkpoints": self.checkpoints,
                    "measurements": self.records, "metadata": metadata})
        columns = set()
        rows = []
        for record in self.records:
            row = {"stage": record["stage"], "sample": record["sample"], "iteration": record["iteration"],
                   "sample_count": record["sample_count"],
                   "torch_peak_allocated_bytes": record["torch_peak_allocated_bytes"],
                   "torch_peak_above_start_bytes": record["torch_peak_above_start_bytes"]}
            for boundary in ("before", "after"):
                row.update({boundary + "_" + key: number for key, number in flatten(record[boundary]).items()})
            row.update({"max_observed_" + key: number for key, number in record["max_observed_bytes"].items()})
            rows.append(row)
            columns.update(row)
        write_csv(self.output / "samples.csv", rows, ["stage", "sample", "iteration"] +
                  sorted(columns - {"stage", "sample", "iteration"}))
        summary = {}
        for stage in sorted({row["stage"] for row in rows}):
            group = [row for row in rows if row["stage"] == stage]
            numeric = {}
            for column in columns - {"stage", "sample", "iteration", "sample_count"}:
                values = [row[column] for row in group if isinstance(row.get(column), (int, float))]
                if values:
                    numeric[column] = {"n": len(values), "mean_bytes": sum(values) / len(values),
                                       "min_bytes": min(values), "max_bytes": max(values)}
            summary[stage] = numeric
        write_json(self.output / "summary.json", {"schema_version": 1, "stages": summary,
                   "checkpoints": [{"label": item["label"], **flatten(item)} for item in self.checkpoints],
                   "metadata": metadata})
        self.journal.close()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def cuda_buffer_inventory(root, buffer_type):
    """Reachable persistent CudaBuffers only; deduplicate pointer aliases.

    Avoid tensors/ndarrays/modules and all arbitrary iterables. This is an
    allocation inventory, not total runtime memory: CUDA Graph/FA2/cuBLAS
    internals are outside it and are covered by nvmap process accounting.
    """
    visited = set()
    allocations = {}

    def walk(value, path, depth=0):
        if isinstance(value, buffer_type):
            pointer = value.ptr
            address = pointer.value if hasattr(pointer, "value") else int(pointer)
            if not address:
                return
            item = allocations.setdefault(address, {"pointer_hex": hex(address),
                "nbytes": int(value.nbytes), "managed": bool(getattr(value, "_managed", False)), "aliases": []})
            if item["nbytes"] != int(value.nbytes):
                raise RuntimeError("Aliased CudaBuffer pointers have inconsistent sizes")
            item["aliases"].append(path)
            return
        if depth > 14 or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, dict):
            children = [(str(key), child) for key, child in value.items()]
        elif isinstance(value, (list, tuple)):
            children = [(str(index), child) for index, child in enumerate(value)]
        elif type(value).__module__.startswith("flash_rt.") and hasattr(value, "__dict__"):
            children = list(vars(value).items())
        else:
            return
        for name, child in children:
            walk(child, path + "." + name, depth + 1)

    # Adapter lives outside the upstream package: seed its frontend explicitly.
    walk(root, "frontend")
    values = sorted(allocations.values(), key=lambda item: item["pointer_hex"])
    return {"unique_allocation_count": len(values), "unique_bytes": sum(item["nbytes"] for item in values),
            "allocations": values,
            "scope": "reachable persistent CudaBuffer objects, unique base pointers; excludes Torch allocations and library/graph internals"}
