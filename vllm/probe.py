#!/usr/bin/env python3
"""CPU-only, non-mutating vLLM / pi05 compatibility probe for the Orin stack.

Run with the deployment interpreter. Only a NEW empty venv is created for uv's
dependency dry-run; no package is installed, no CUDA context/model is created.
Source files are fetched at locked commits and hashed before import testing.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request


HERE = Path(__file__).resolve().parent


def command(argv, timeout=90, cwd=None):
    start = time.perf_counter()
    record = {"argv": [str(v) for v in argv]}
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, cwd=cwd,
                                   env={**os.environ, "UV_NO_PROGRESS": "1"})
        record.update(returncode=completed.returncode, stdout=completed.stdout,
                      stderr=completed.stderr)
    except subprocess.TimeoutExpired as exc:
        record.update(returncode=None, timeout_seconds=timeout,
                      stdout=(exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else exc.stdout,
                      stderr=(exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else exc.stderr)
    record["elapsed_seconds"] = time.perf_counter() - start
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--uv", type=Path, default=Path.home()/".local/bin/uv")
    ap.add_argument("--venv", type=Path, default=Path.home()/"pi05-orin-vllm/.venv-probe")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": "CPU/source/dependency preflight only; no model or CUDA allocation",
        "python": {"version": sys.version, "executable": sys.executable,
                   "machine": platform.machine(), "platform": platform.platform()},
        "installed": {}, "commands": {}, "sources": {},
        "inference_performed": False, "latency_ms": None,
        "inference_peak_memory_bytes": None,
    }
    for package in ["torch", "transformers", "vllm", "vllm-omni", "safetensors"]:
        try: report["installed"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: report["installed"][package] = None
    report["commands"]["nvcc_version"] = command(["/usr/local/cuda/bin/nvcc", "--version"])
    report["commands"]["l4t_release"] = command(["cat", "/etc/nv_tegra_release"])

    lock = json.loads((HERE/"sources.json").read_text())
    source_root = HERE/"vendor"
    for item in lock["files"]:
        target = source_root/item["local"]
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"url": item["url"], "expected_sha256": item["sha256"]}
        try:
            if not target.exists():
                target.write_bytes(urllib.request.urlopen(item["url"], timeout=30).read())
            data = target.read_bytes()
            record.update(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            if record["sha256"] != item["sha256"]:
                raise RuntimeError("Source digest mismatch")
            record["verified"] = True
        except Exception as exc:
            record.update(verified=False, exception=repr(exc))
        report["sources"][item["local"]] = record

    model_local = "vllm-omni/vllm_omni/diffusion/models/pi05/modeling_pi05.py"
    if report["sources"].get(model_local, {}).get("verified"):
        # Import the ORIGINAL official file, without replacing vllm's loader,
        # monkey-patching the module, or initializing any weights/CUDA context.
        source_path = source_root/model_local
        snippet = (
            "import importlib.util; "
            f"s=importlib.util.spec_from_file_location('official_pi05_probe', {str(source_path)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print('ORIGINAL_OFFICIAL_PI05_IMPORT_OK')"
        )
        report["commands"]["official_pi05_import"] = command([sys.executable, "-c", snippet],
                                                              cwd=args.output.resolve())

    pyproject = source_root/"vllm/pyproject.toml"
    if report["sources"].get("vllm/pyproject.toml", {}).get("verified"):
        import re
        from packaging.specifiers import SpecifierSet
        spec = re.search(r'^requires-python\s*=\s*"([^"]+)"', pyproject.read_text(), re.M).group(1)
        report["latest_source_python_requirement"] = {
            "requires_python": spec,
            "target_python": platform.python_version(),
            "satisfied": platform.python_version() in SpecifierSet(spec),
        }

    # uv's dry-run evaluates target Python/dependencies without installing them.
    # NVIDIA Torch is retained as an explicit constraint, rather than permitting
    # a resolver to replace it with an incompatible generic wheel.
    constraints = args.output/"constraints.txt"
    constraints.write_text("torch=="+report["installed"]["torch"]+"\n")
    if args.venv.exists():
        report["commands"]["create_venv"] = {"skipped": "pre-existing probe-only venv"}
    else:
        report["commands"]["create_venv"] = command([str(args.uv), "venv", "--python", sys.executable,
                                                    str(args.venv)])
    new_python = args.venv/"bin/python"
    if new_python.exists():
        report["commands"]["official_release_dry_run"] = command(
            [str(args.uv), "pip", "install", "--dry-run", "--python", str(new_python),
             "--constraint", str(constraints), "vllm==0.30.0", "vllm-omni==0.30.0"], 120)
        report["commands"]["current_release_dry_run"] = command(
            [str(args.uv), "pip", "install", "--dry-run", "--python", str(new_python),
             "--constraint", str(constraints), "vllm==0.31.0"], 120)

    # Publish complete raw records including failed commands, never an invented
    # timing/memory value for an engine that did not reach model inference.
    (args.output/"compatibility.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "installed": report["installed"],
                      "command_returncodes": {k:v.get("returncode") for k,v in report["commands"].items()},
                      "inference_performed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
