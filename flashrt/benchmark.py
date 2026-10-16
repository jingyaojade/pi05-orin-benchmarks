#!/usr/bin/env python3
"""Compare a pinned, patched FlashRT CUDA Graph with the saved pi05 fixture."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from adapter import LiberoFlashRT, PIN


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashrt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True, help="Baseline fixed_inputs.npz")
    parser.add_argument("--reference", type=Path, help="Default: fixture sibling reference_actions.npy")
    parser.add_argument("--tokenizer", type=Path,
                        help="Official model; defaults to PI05_TOKENIZER_PATH / FLASHRT_PALIGEMMA_TOKENIZER / deployment cache")
    parser.add_argument("--precision", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/flashrt_bf16"))
    parser.add_argument("--min-cosine", type=float, help="Optional explicit numerical gate, not task-success validation")
    parser.add_argument("--max-rmse", type=float, help="Optional action-space RMSE numerical gate")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_optional(path):
    try:
        return Path(path).read_text(errors="replace").replace("\x00", "").strip()
    except OSError:
        return None


def run_readonly(command):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"command": command, "returncode": proc.returncode,
                "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def statistics(values, np):
    values = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "min_ms": float(values.min()),
            "max_ms": float(values.max()), "std_ms": float(values.std())}


def accuracy(actual, expected, np):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or actual.shape != (10, 7):
        raise ValueError(f"Action shape mismatch {actual.shape} / {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite predicted/reference actions")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(a @ b / denominator) if denominator else (1.0 if np.array_equal(a, b) else 0.0)
    return {"shape": list(actual.shape), "all_finite": True,
            "exact_match": bool(np.array_equal(actual, expected)), "cosine": cosine,
            "rmse": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta)))}


def timed_model_graph(adapter, noise, pair, torch):
    """GPU-ready inputs -> original full graph -> completed GPU output.

    Restoring the in-place noise input and all staging are before the timer.
    CPU output transfer, unnormalization and numerical checks are performed
    by the caller after this function returns.
    """
    adapter.stage_model_inputs(noise)
    torch.cuda.synchronize()
    stream = adapter.model_stream
    started = time.perf_counter_ns()
    pair[0].record(stream)
    adapter.replay_model_graph()
    pair[1].record(stream)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    return {"model_wall_ms": wall_ms,
            "model_cuda_stream_elapsed_ms": float(pair[0].elapsed_time(pair[1]))}


def main():
    args = arguments()
    import numpy as np
    import torch

    out = args.output.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    root = args.flashrt_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    fixture_path = args.fixture.expanduser().resolve()
    reference_path = (args.reference or fixture_path.with_name("reference_actions.npy")).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError("Run the baseline first; reference_actions.npy is required")
    with np.load(fixture_path, allow_pickle=False) as archive:
        fixture = {key: archive[key] for key in ("image", "wrist_image", "state", "noise", "prompt")}
    reference = np.load(reference_path, allow_pickle=False)
    np.savez(out / "fixed_inputs.npz", **fixture)

    print(json.dumps({"event": "loading_flashrt", "commit": PIN, "precision": args.precision}), flush=True)
    adapter = LiberoFlashRT(root, checkpoint, precision=args.precision, tokenizer_path=args.tokenizer)
    adapter.prepare(fixture)
    torch.cuda.synchronize()
    first = adapter.infer(fixture["noise"])
    torch.cuda.synchronize()
    comparison = accuracy(first, reference, np)
    np.save(out / "flashrt_reference_actions.npy", first)
    write_json(out / "accuracy.json", {"against": str(reference_path), "metrics": comparison,
        "scope": "one seeded synthetic observation; this is not robot-task success",
        "noise_contract": "same float32 fixture values; FlashRT stages them as BF16; deployed sampler uses FP32 state/update"})
    if args.min_cosine is not None and comparison["cosine"] < args.min_cosine:
        raise RuntimeError(f"Cosine gate failed: {comparison['cosine']} < {args.min_cosine}")
    if args.max_rmse is not None and comparison["rmse"] > args.max_rmse:
        raise RuntimeError(f"RMSE gate failed: {comparison['rmse']} > {args.max_rmse}")

    raw = {"phases": {}, "accuracy_against_openpi": comparison}
    production_graph = adapter.pipe.pipeline._graph

    def check_repeat(actions):
        metrics = accuracy(actions, first, np)
        if not metrics["exact_match"]:
            raise RuntimeError(f"FlashRT fixed-input repeat changed: {metrics}")
        return metrics

    phase = {"warmup_wall_ms": [], "samples": []}
    for index in range(args.warmup + args.iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = adapter.infer(fixture["noise"])
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        validation = check_repeat(result)
        if index < args.warmup:
            phase["warmup_wall_ms"].append(elapsed)
        else:
            phase["samples"].append({"sample": index - args.warmup,
                "timings": {"policy_wall_ms": elapsed}, "validation": validation})
    raw["phases"]["flashrt_policy_uninstrumented"] = phase
    print(json.dumps({"event": "production_phase_complete", "samples": args.iterations}), flush=True)

    # Prebuild GPU image inputs once. timed_model_graph restores fixture noise
    # before EVERY timer because the graph writes final actions into that slot.
    # This phase uses only the original full production graph, before any
    # diagnostic graphs are even captured.
    adapter.stage_model_inputs(fixture["noise"], include_images=True)
    model_pair = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    phase = {"warmup_wall_ms": [], "warmup_timings": [], "samples": []}
    for index in range(args.warmup + args.iterations):
        if adapter.pipe.pipeline._graph is not production_graph:
            raise RuntimeError("Model-only phase lost the original production graph")
        timings = timed_model_graph(adapter, fixture["noise"], model_pair, torch)
        # D2H, action decoding and bitwise validation are OUTSIDE both model
        # timing boundaries. The GPU final-action state is ready on return.
        result = adapter.download_model_actions()
        validation = check_repeat(result)
        if index < args.warmup:
            phase["warmup_wall_ms"].append(timings["model_wall_ms"])
            phase["warmup_timings"].append(timings)
        else:
            phase["samples"].append({"sample": index - args.warmup,
                "timings": timings, "validation": validation})
    raw["phases"]["model_uninstrumented"] = phase
    print(json.dumps({"event": "model_phase_complete", "samples": args.iterations}), flush=True)

    # Capture diagnostic stages only AFTER BOTH production measurements.
    # The production full graph and its nodes remain unmodified.
    graphs = adapter.stage_graphs()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    phase = {"warmup_wall_ms": [], "samples": []}
    for index in range(args.warmup + args.iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result, timings = adapter.run_stages(graphs, fixture["noise"], events)
        timings["diagnostic_chain_wall_ms"] = (time.perf_counter() - start) * 1000.0
        validation = check_repeat(result)
        if index < args.warmup:
            phase["warmup_wall_ms"].append(timings["diagnostic_chain_wall_ms"])
        else:
            phase["samples"].append({"sample": index - args.warmup,
                "timings": timings, "validation": validation})
    raw["phases"]["flashrt_diagnostic_stage_graphs"] = phase
    if adapter.pipe.pipeline._graph is not production_graph:
        raise RuntimeError("Production graph was replaced")
    raw["production_graph_restoration"] = check_repeat(adapter.infer(fixture["noise"]))

    caveats = [
        "FlashRT is locally patched; upstream source commit and patch hashes are recorded.",
        "All 3 cameras run SigLIP; encoder GEMMs retain 968 rows (768 image + 200 prompt capacity).",
        "Prompt K/V are compacted into the masked camera gap for FA2 seqused masking; padded attention keys are skipped.",
        "No image pooling, layer dropping, temporal KV reuse, or step reduction; 10 action rows and 10 ODE steps.",
        "FlashRT computes/accumulates the action state in BF16 and precomputes timestep modulations; OpenPI retains some FP32 work.",
        "Model-only times replay the original full production graph from CUDA-ready inputs to completed CUDA output; staging, D2H, action decoding and comparison are outside timing.",
        "Diagnostic stage graphs are separate launches, not a timing decomposition of one full production graph.",
        "CUDA events report stream elapsed intervals; they do not report kernel busy time or utilization.",
        "Accuracy uses a single synthetic fixture; INT8, if selected, is a separate experiment.",
    ]
    summary = {"phases": {}, "caveats": caveats}
    csv_rows = []
    for name, phase in raw["phases"].items():
        summary["phases"][name] = {}
        for metric in phase["samples"][0]["timings"]:
            values = [sample["timings"][metric] for sample in phase["samples"]]
            summary["phases"][name][metric] = statistics(values, np)
        for sample in phase["samples"]:
            for metric, value in sample["timings"].items():
                csv_rows.append({"phase": name, "sample": sample["sample"], "metric": metric, "milliseconds": value})
    write_json(out / "raw.json", raw)
    write_json(out / "summary.json", summary)
    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("phase", "sample", "metric", "milliseconds"))
        writer.writeheader()
        writer.writerows(csv_rows)

    versions = {}
    for package in ("torch", "numpy", "flash-rt", "pybind11", "ml_dtypes", "pyyaml", "safetensors", "sentencepiece"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    hashes = {"weights": sha256_file(checkpoint / "model.safetensors"),
              "fixture": sha256_file(fixture_path), "reference_actions": sha256_file(reference_path)}
    for path in checkpoint.glob("assets/**/norm_stats.json"):
        hashes[str(path.relative_to(checkpoint))] = sha256_file(path)
    from flash_rt import flash_rt_kernels, flash_rt_fa2
    for name, module in (("flash_rt_kernels", flash_rt_kernels), ("flash_rt_fa2", flash_rt_fa2)):
        hashes[name] = {"path": str(module.__file__), "sha256": sha256_file(Path(module.__file__))}
    benchmark_root = Path(__file__).resolve().parent
    for name in ("adapter.py", "benchmark.py", "prepare_patch.py", "standalone_tokenizer.py"):
        hashes[name] = sha256_file(benchmark_root / name)
    caches = {}
    for cache in root.glob("build*/CMakeCache.txt"):
        lines = cache.read_text(errors="replace").splitlines()
        caches[str(cache)] = {"sha256": sha256_file(cache),
            "configuration": [line for line in lines if line.startswith((
                "GPU_ARCH:", "FA2_", "FLASHRT_", "CMAKE_CUDA_COMPILER:",
                "CUDAToolkit_", "Python3_EXECUTABLE:", "CUTLASS_DIR:"))]}
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "flashrt_upstream_commit": PIN, "patch": adapter.source_manifest,
        "checkpoint": str(checkpoint), "fixture": str(fixture_path), "hashes": hashes,
        "versions": versions, "python": platform.python_version(), "platform": platform.platform(),
        "torch_cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "gpu_capability": list(torch.cuda.get_device_capability()),
        "l4t": read_optional("/etc/nv_tegra_release"), "model": read_optional("/proc/device-tree/model"),
        "nvpmodel": run_readonly(["nvpmodel", "-q"]), "jetson_clocks": run_readonly(["jetson_clocks", "--show"]),
        "nvcc": run_readonly(["nvcc", "--version"]), "git": run_readonly(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "precision": args.precision, "build_caches": caches,
        "tokenizer": adapter.tokenizer_provenance,
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("FVK_", "FLASHRT_")) or key in ("OMP_NUM_THREADS", "JAX_PLATFORMS")},
        "geometry": {"views_computed": 3, "valid_views": 2, "image_mask": [True, True, False],
                     "encoder_rows": 968, "prompt_capacity": 200, "valid_prompt_tokens": adapter.pipe.current_prompt_len,
                     "action_horizon": 10, "action_dim_internal": 32, "action_dim_returned": 7,
                     "denoise_steps": 10, "cache_frames": 1},
        "caveats": caveats}
    write_json(out / "provenance.json", provenance)
    print(json.dumps({"event": "complete", "output": str(out), "accuracy": comparison}), flush=True)


if __name__ == "__main__":
    main()
