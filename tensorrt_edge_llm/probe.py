#!/usr/bin/env python3
"""CPU-only, non-mutating platform/API/package probe for native Edge-LLM pi05.

Creates only its output directory and a disposable venv there. Does not import
Torch, initialize CUDA, install packages, build engines, or alter the host SDK.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

PIN = "95515c2f87fba8982db5a519f9022277667b3cc9"  # v0.11.0
REPO = "https://github.com/NVIDIA/TensorRT-Edge-LLM"
SOURCE_FILES = {
    "native_onnx_contract": "tensorrt_edgellm/onnx/onnx_custom_schemas.py",
    "plugin_contract": "cpp/plugins/attentionPlugin/attentionPlugin.h",
    "runtime_contract": "experimental_models/pi05/cpp/common/pi05Common.cpp",
    "support_matrix": "docs/source/user_guide/getting_started/support-matrix.md",
    "pi05_example": "docs/source/user_guide/examples/vla/pi05.md",
    "pi05_design": "docs/source/developer_guide/models/pi05.md",
    "python_dependencies": "pyproject.toml",
}
API_CASES = {
    "control_trt_headers": "static_assert(NV_TENSORRT_MAJOR >= 8);",
    "native_attention": "using Required = decltype(&nvinfer1::INetworkDefinition::addAttention);",
    "native_rotary_embedding": "using Required = decltype(&nvinfer1::INetworkDefinition::addRotaryEmbedding);",
    "plugin_v3_build_v2": "using Required = nvinfer1::IPluginV3OneBuildV2;",
    "runtime_managed_memory": "using Required = decltype(&nvinfer1::ICudaEngine::getDeviceMemorySizeV2);",
}


def run(command: list[str], timeout: int = 30) -> dict:
    try:
        p = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return {"command": command, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": command, "error": str(e)}


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "pi05-orin-compatibility-probe"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return json.load(response)


def read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", value)[:3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path,
                        help="Existing official v0.11.0 source (no checkout changed).")
    parser.add_argument("--fetch-source", action="store_true",
                        help="Fetch and hash seven pinned official source files, not full repo.")
    parser.add_argument("--resolve-package", action="store_true",
                        help="Run uv pip install --dry-run in an isolated disposable venv.")
    parser.add_argument("--uv", default=str(Path.home() / ".local/bin/uv"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = Path(__file__).with_name("source_audit.json")
    expected_source_hashes = {}
    if audit_path.exists():
        expected_source_hashes = {item["path"]: item["sha256"]
                                  for item in json.loads(audit_path.read_text())["source_files"]}
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "TensorRT Edge-LLM native experimental pi0.5",
        "upstream": {"repository": REPO, "ref": "v0.11.0", "commit": PIN},
        "inference_executed": False,
        "latency_ms": None,
        "engine_memory_bytes": None,
        "measurement_note": "Compatibility probe only. No model engine was loaded or timed.",
        "platform": {"machine": platform.machine(), "platform": platform.platform(),
                     "glibc": platform.libc_ver(), "python": sys.version,
                     "l4t": read(Path("/etc/nv_tegra_release")),
                     "os_release": read(Path("/etc/os-release"))},
        "commands": {}, "source_evidence": {}, "api_compile_probes": {},
        "blockers": [],
    }
    for name, command in {
        "nvcc": ["/usr/local/cuda/bin/nvcc", "--version"],
        "tensorrt_python": ["/usr/bin/python3", "-c", "import tensorrt; print(tensorrt.__version__)"],
        "packages": ["dpkg-query", "-W", "libnvinfer8", "libnvinfer-dev", "nvidia-l4t-cuda"],
        "glibc": ["getconf", "GNU_LIBC_VERSION"],
        "docker": ["sudo", "-n", "docker", "version", "--format", "{{.Server.Version}}"],
    }.items():
        report["commands"][name] = run(command)

    for name, code in API_CASES.items():
        source = args.output / (name + ".cpp")
        source.write_text("#include <NvInfer.h>\n#include <NvInferVersion.h>\n" + code + "\n")
        result = run(["g++", "-std=c++17", "-fsyntax-only", "-I/usr/include/aarch64-linux-gnu",
                      "-I/usr/local/cuda/include", str(source)])
        report["api_compile_probes"][name] = result
        (args.output / (name + ".log")).write_text(result.get("stdout", "") + result.get("stderr", ""))

    for name, path in SOURCE_FILES.items():
        target = args.source_root / path if args.source_root else args.output / "source_evidence" / path
        try:
            if args.fetch_source and not args.source_root:
                target.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://raw.githubusercontent.com/NVIDIA/TensorRT-Edge-LLM/{PIN}/{path}"
                with urllib.request.urlopen(url, timeout=45) as response:
                    target.write_bytes(response.read())
            data = target.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            report["source_evidence"][name] = {
                "path": path, "sha256": digest,
                "url": f"{REPO}/blob/{PIN}/{path}",
            }
            if path in expected_source_hashes:
                matched = digest == expected_source_hashes[path]
                report["source_evidence"][name]["matches_pinned_audit"] = matched
                if not matched:
                    report["blockers"].append("Source evidence hash does not match pinned audit: " + path)
            if name == "native_onnx_contract":
                report["source_evidence"][name]["trt_minimum_statement_present"] = (
                    "consumed by TRT >= 10.15" in data.decode())
        except (OSError, urllib.error.URLError) as e:
            report["source_evidence"][name] = {"path": path, "error": str(e)}

    try:
        pypi = fetch_json("https://pypi.org/pypi/tensorrt-edgellm/0.11.0/json")
        wheels = [{"filename": x["filename"], "size_bytes": x["size"],
                   "sha256": x["digests"]["sha256"]} for x in pypi["urls"]]
        report["published_wheels"] = wheels
        report["wheel_floor"] = {"aarch64_glibc_minimum": "2.39",
                                 "installed_glibc": platform.libc_ver()[1]}
        if platform.machine() == "aarch64" and version_tuple(platform.libc_ver()[1]) < (2, 39):
            report["blockers"].append("Published aarch64 wheel requires glibc 2.39; host is older.")
    except Exception as e:
        report["published_wheel_query_error"] = str(e)

    if args.resolve_package:
        # Deliberately uses no --system, no package installation, no host mutations.
        venv = args.output / "resolver_venv"
        # uv does not require Debian's optional ensurepip/python3-venv package.
        creation = run([args.uv, "venv", "--python", "/usr/bin/python3", "--allow-existing", str(venv)])
        report["commands"]["create_isolated_resolver_venv"] = creation
        if creation.get("returncode") == 0:
            result = run([args.uv, "pip", "install", "--python", str(venv / "bin/python"),
                          "--dry-run", "--no-deps", "--index-url", "https://pypi.org/simple",
                          "tensorrt-edgellm==0.11.0"], timeout=150)
            report["commands"]["native_package_resolver_dry_run"] = result
            (args.output / "package_resolver.log").write_text(result.get("stdout", "") + result.get("stderr", ""))
            if result.get("returncode") not in (0, None):
                report["blockers"].append("Native package resolver rejected the host platform; see package_resolver.log.")

    failed = [name for name, result in report["api_compile_probes"].items()
              if name != "control_trt_headers" and result.get("returncode") != 0]
    control_ok = report["api_compile_probes"]["control_trt_headers"].get("returncode") == 0
    if control_ok and failed:
        report["blockers"].append("Installed TensorRT headers lack APIs used by the native pi05 path: " + ", ".join(failed))
    report["status"] = "blocked_on_current_stack" if report["blockers"] else "requires_runtime_validation"
    (args.output / "compatibility.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "blockers": report["blockers"],
                      "inference_executed": False}, ensure_ascii=False))
    return 0  # Probe completed; a blocked engine is a result, not a probe crash.


if __name__ == "__main__":
    raise SystemExit(main())
