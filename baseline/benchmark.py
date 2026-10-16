#!/usr/bin/env python3
"""Measure the deployed pi05-LIBERO baseline without changing its computation.

Run with the deployment's Python environment. This script loads serve.py but
does not start a server, change clocks, stop services, or edit the deployment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-root", type=Path, default=Path("/home/nvidia/pi05-orin"))
    parser.add_argument("--checkpoint", type=Path, help="Default: DEPLOYMENT_ROOT/models/pi05_libero")
    parser.add_argument("--iterations", type=int, default=30, help="Measured samples in EACH of three phases")
    parser.add_argument("--warmup", type=int, default=10, help="Excluded warmup samples before EACH phase")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup must be nonnegative")
    return args


def emit(event: str, **details: Any) -> None:
    print(json.dumps({"event": event, **details}, ensure_ascii=False), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_optional(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").replace("\x00", "").strip()
    except (OSError, PermissionError):
        return None


def run_readonly(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=15, check=False)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def safe_remote(value: str) -> str:
    # Never persist an HTTPS credential if a checkout happens to have one.
    if "://" not in value:
        return value
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def repository_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path)}
    if shutil.which("git"):
        for label, arguments in (
            ("commit", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain"]),
            ("remote", ["remote", "get-url", "origin"]),
        ):
            result = run_readonly(["git", "-C", str(path), *arguments])
            if result.get("returncode") == 0:
                value = result["stdout"]
                info[label] = safe_remote(value) if label == "remote" else value
            else:
                info[f"{label}_unavailable"] = result.get("stderr", result.get("error", "unknown"))
    return info


def load_policy(args: argparse.Namespace):
    # Import after argument parsing so --help works without the GPU environment.
    global np, torch, jax, model_module
    deployment = args.deployment_root.expanduser().resolve()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("OPENPI_DATA_HOME", str(deployment / "cache"))
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    for path in (deployment / "openpi" / "src", deployment / "openpi" / "packages" / "openpi-client" / "src"):
        if path.is_dir():
            sys.path.insert(0, str(path))
    helper = deployment / "serve.py"
    if not helper.is_file():
        helper = deployment / "scripts" / "deploy_orin" / "serve.py"
        for path in (deployment / "src", deployment / "packages" / "openpi-client" / "src"):
            if path.is_dir():
                sys.path.insert(0, str(path))
    if not helper.is_file():
        raise FileNotFoundError(f"Deployment serve.py not found under {deployment}")
    spec = importlib.util.spec_from_file_location("_pi05_baseline_deployment", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import numpy as np
    import torch
    import jax
    from openpi.models import model as model_module

    checkpoint = (args.checkpoint or deployment / "models" / "pi05_libero").expanduser().resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the deployed CUDA PyTorch environment")
    emit("loading_policy", checkpoint=str(checkpoint), helper=str(helper))
    policy = module.create_policy(checkpoint)
    config = policy._model.config
    required = {"pi05": True, "action_horizon": 10, "action_dim": 32,
                "discrete_state_input": False, "pytorch_compile_mode": None}
    for name, expected in required.items():
        if getattr(config, name) != expected:
            raise RuntimeError(f"Baseline requires {name}={expected!r}, got {getattr(config, name)!r}")
    if not policy._is_pytorch_model or policy._sample_kwargs.get("num_steps", 10) != 10:
        raise RuntimeError("Baseline requires PyTorch with the official 10 denoising steps")
    if policy._model.training:
        raise RuntimeError("Baseline requires model.eval()")
    return policy, deployment, checkpoint, helper


def make_fixture(seed: int) -> tuple[dict[str, Any], Any]:
    rng = np.random.default_rng(seed)
    observation = {
        "observation/state": rng.random(8).astype(np.float32),
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the black bowl",
    }
    # Explicit noise removes RNG generation and RNG drift from comparisons.
    noise = rng.standard_normal((10, 32)).astype(np.float32)
    return observation, noise


def prepare_cpu(policy, observation):
    return policy._input_transform(jax.tree.map(lambda value: value, observation))


def transfer_and_observe(policy, inputs, noise):
    # Exact operations and ordering from official Policy.infer, with the fixed
    # noise transfer included in this phase rather than hidden before the timer.
    device = policy._pytorch_device
    tensors = jax.tree.map(lambda value: torch.from_numpy(np.array(value)).to(device)[None, ...], inputs)
    sample_kwargs = dict(policy._sample_kwargs)
    noise_tensor = torch.from_numpy(noise).to(device)
    if noise_tensor.ndim == 2:
        noise_tensor = noise_tensor[None, ...]
    sample_kwargs["noise"] = noise_tensor
    observation = model_module.Observation.from_dict(tensors)
    return tensors, observation, sample_kwargs


def finish_cpu(policy, inputs, actions):
    outputs = {"state": inputs["state"], "actions": actions}
    outputs = jax.tree.map(lambda value: np.asarray(value[0, ...].detach().cpu()), outputs)
    return policy._output_transform(outputs)


def check_actions(result, reference=None) -> tuple[Any, dict[str, Any]]:
    actions = np.asarray(result["actions"])
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise AssertionError(f"Invalid actions: shape={actions.shape}, finite={np.isfinite(actions).all()}")
    validation = {"shape": [10, 7], "finite": True}
    if reference is not None:
        exact = bool(np.array_equal(actions, reference))
        maximum = float(np.max(np.abs(actions.astype(np.float64) - reference.astype(np.float64))))
        validation.update({"bitwise_equal_reference": exact, "max_abs_error": maximum})
        if not exact:
            raise AssertionError(f"Fixed-input actions changed; max absolute error={maximum:.9g}")
    return actions.copy(), validation


class EventRecorder:
    """Reusable CUDA event pairs; never synchronize between model stages."""

    def __init__(self):
        self.pool = []
        self.records = []
        self.stack = []
        self.denoise_count = 0
        self.image_count = 0
        self.iteration = None

    def begin_sample(self):
        self.records = []
        self.stack = []
        self.denoise_count = 0
        self.image_count = 0
        self.iteration = None

    @contextmanager
    def measure(self, stage, image=None):
        index = len(self.records)
        if index == len(self.pool):
            self.pool.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
        start, end = self.pool[index]
        record = {"stage": stage, "iteration": self.iteration, "image": image,
                  "parent": self.stack[-1] if self.stack else "model_total"}
        self.records.append((record, start, end))
        self.stack.append(stage)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.stack.pop()

    def collect(self):
        # The caller has already synchronized once at the entire model boundary.
        if self.image_count != 3 or self.denoise_count != 10:
            raise AssertionError(f"Expected 3 vision calls and 10 denoise calls, got {self.image_count}, {self.denoise_count}")
        counts = Counter(record["stage"] for record, _, _ in self.records)
        for stage, expected in (("model_preprocess", 1), ("prefix_embedding_total", 1), ("vision_encode", 3),
                                ("language_token_embedding", 1), ("vlm_prefix_kv_prefill", 1),
                                ("denoise_iteration", 10), ("action_expert_transformer", 10),
                                ("action_suffix_embedding", 10), ("time_mlp_total", 10)):
            if counts[stage] != expected:
                raise AssertionError(f"Expected {expected} {stage} intervals, got {counts[stage]}")
        return [{**record, "cuda_stream_elapsed_ms": float(start.elapsed_time(end))}
                for record, start, end in self.records]


class ModelInstrumentation:
    """Install instance-local wrappers and restore the original attributes."""

    def __init__(self, model, recorder):
        self.model = model
        self.recorder = recorder
        self.restore = []

    def replace(self, target, name, factory):
        was_local = name in target.__dict__
        local = target.__dict__.get(name)
        original = getattr(target, name)
        wrapper = factory(original)
        self.restore.append((target, name, was_local, local))
        setattr(target, name, wrapper)

    def fixed(self, target, name, stage):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                with self.recorder.measure(stage):
                    return original(*args, **kwargs)
            return wrapped
        self.replace(target, name, factory)

    def __enter__(self):
        recorder = self.recorder
        model = self.model

        def denoise_factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                previous = recorder.iteration
                recorder.denoise_count += 1
                recorder.iteration = recorder.denoise_count
                try:
                    with recorder.measure("denoise_iteration"):
                        return original(*args, **kwargs)
                finally:
                    recorder.iteration = previous
            return wrapped

        def image_factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_padded")
                index = recorder.image_count
                recorder.image_count += 1
                image = names[index] if index < len(names) else f"unexpected_image_{index}"
                with recorder.measure("vision_encode", image=image):
                    return original(*args, **kwargs)
            return wrapped

        def forward_factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                embeddings = kwargs.get("inputs_embeds", args[3] if len(args) > 3 else None)
                if embeddings is None:
                    raise AssertionError("Cannot classify PaliGemmaWithExpertModel.forward")
                if embeddings[1] is None:
                    stage = "vlm_prefix_kv_prefill"
                elif embeddings[0] is None:
                    stage = "action_expert_transformer"
                else:
                    raise AssertionError("Unexpected joint training forward during inference")
                with recorder.measure(stage):
                    return original(*args, **kwargs)
            return wrapped

        def checkpoint_factory(original):
            @functools.wraps(original)
            def wrapped(function, *args, **kwargs):
                # This named closure contains both time-MLP linear layers and
                # both SiLU activations in the upstream pi05 embed_suffix.
                if getattr(function, "__name__", None) == "time_mlp_func":
                    with recorder.measure("time_mlp_total"):
                        return original(function, *args, **kwargs)
                return original(function, *args, **kwargs)
            return wrapped

        try:
            self.fixed(model, "_preprocess_observation", "model_preprocess")
            self.fixed(model, "embed_prefix", "prefix_embedding_total")
            self.replace(model.paligemma_with_expert, "embed_image", image_factory)
            self.fixed(model.paligemma_with_expert, "embed_language_tokens", "language_token_embedding")
            self.replace(model.paligemma_with_expert, "forward", forward_factory)
            self.replace(model, "denoise_step", denoise_factory)
            self.fixed(model, "embed_suffix", "action_suffix_embedding")
            self.replace(model, "_apply_checkpoint", checkpoint_factory)
            self.fixed(model, "_prepare_attention_masks_4d", "attention_mask_prepare")
            for attribute, stage in (
                ("action_in_proj", "action_input_projection"),
                ("time_mlp_in", "time_mlp_in_linear"),
                ("time_mlp_out", "time_mlp_out_linear"),
                ("action_out_proj", "action_output_projection"),
            ):
                self.fixed(getattr(model, attribute), "forward", stage)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exception_type, exception, traceback):
        for target, name, was_local, local in reversed(self.restore):
            if was_local:
                setattr(target, name, local)
            else:
                delattr(target, name)
        self.restore.clear()


def timed_official(policy, observation, noise):
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    result = policy.infer(observation, noise=noise)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    return result, {"full_policy_wall_ms": elapsed_ms}


def timed_model(policy, observation_cuda, sample_kwargs, pair):
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    pair[0].record()
    actions = policy._sample_actions(policy._pytorch_device, observation_cuda, **sample_kwargs)
    pair[1].record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    return actions, {"model_wall_ms": wall_ms, "model_cuda_stream_elapsed_ms": float(pair[0].elapsed_time(pair[1]))}


def timed_decomposed(policy, observation, noise, recorder, model_pair, transfer_pair):
    torch.cuda.synchronize()
    full_start = time.perf_counter_ns()
    inputs_cpu = prepare_cpu(policy, observation)
    cpu_end = time.perf_counter_ns()

    transfer_pair[0].record()
    inputs, observation_cuda, sample_kwargs = transfer_and_observe(policy, inputs_cpu, noise)
    transfer_pair[1].record()
    torch.cuda.synchronize()
    transfer_end = time.perf_counter_ns()

    model_start = time.perf_counter_ns()
    recorder.begin_sample()
    model_pair[0].record()
    actions = policy._sample_actions(policy._pytorch_device, observation_cuda, **sample_kwargs)
    model_pair[1].record()
    torch.cuda.synchronize()
    model_end = time.perf_counter_ns()

    result = finish_cpu(policy, inputs, actions)
    post_end = time.perf_counter_ns()
    model_event_ms = float(model_pair[0].elapsed_time(model_pair[1]))
    stages = recorder.collect()
    top_level = {"model_preprocess", "prefix_embedding_total", "vlm_prefix_kv_prefill", "denoise_iteration"}
    covered_ms = sum(row["cuda_stream_elapsed_ms"] for row in stages if row["stage"] in top_level)
    timings = {
        "cpu_preprocess_wall_ms": (cpu_end - full_start) / 1e6,
        "h2d_and_observation_wall_ms": (transfer_end - cpu_end) / 1e6,
        "h2d_and_observation_cuda_stream_elapsed_ms": float(transfer_pair[0].elapsed_time(transfer_pair[1])),
        "model_wall_ms": (model_end - model_start) / 1e6,
        "model_cuda_stream_elapsed_ms": model_event_ms,
        "d2h_and_postprocess_wall_ms": (post_end - model_end) / 1e6,
        "decomposed_policy_wall_ms": (post_end - full_start) / 1e6,
        "model_remaining_cuda_stream_elapsed_ms": model_event_ms - covered_ms,
    }
    return result, timings, stages


def summarize(values) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, 0.95)), "min_ms": float(values.min()),
            "max_ms": float(values.max()), "std_ms": float(values.std(ddof=0))}


def phase_summary(samples):
    return {key: summarize([sample["timings"][key] for sample in samples]) for key in samples[0]["timings"]}


def collect_provenance(policy, deployment, checkpoint, helper, fixture, args):
    emit("collecting_provenance", checkpoint_sha256="reading full weights file")
    versions = {}
    for package in ("torch", "torchvision", "transformers", "safetensors", "jax", "jaxlib", "flax", "numpy",
                    "websockets", "sentencepiece"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    device = torch.cuda.get_device_properties(0)
    files = {"benchmark": Path(__file__).resolve(), "deployment_helper": helper,
             "policy": Path(inspect.getfile(type(policy))),
             "model": Path(inspect.getfile(type(policy._model))),
             "gemma": Path(inspect.getfile(type(policy._model.paligemma_with_expert)))}
    import transformers.models.gemma.modeling_gemma as modeling_gemma
    import transformers.models.siglip.modeling_siglip as modeling_siglip
    import transformers.models.paligemma.modeling_paligemma as modeling_paligemma
    for label, module in (("patched_gemma", modeling_gemma), ("patched_siglip", modeling_siglip),
                          ("patched_paligemma", modeling_paligemma)):
        files[label] = Path(inspect.getfile(module))
    source_files = {label: {"path": str(path), "sha256": sha256_file(path)} for label, path in files.items()}
    for name in ("openpi.transforms", "openpi.models.pi0_config", "openpi.models.tokenizer",
                 "openpi.models_pytorch.preprocessing_pytorch", "openpi.policies.libero_policy", "openpi.shared.normalize"):
        module = sys.modules.get(name)
        if module is not None:
            path = Path(inspect.getfile(module))
            source_files[name] = {"path": str(path), "sha256": sha256_file(path)}
    tokenizer_hash = None
    for transform in policy._input_transform.transforms:
        candidate = getattr(getattr(transform, "tokenizer", None), "_tokenizer", None)
        if candidate is not None and hasattr(candidate, "serialized_model_proto"):
            tokenizer_hash = hashlib.sha256(candidate.serialized_model_proto()).hexdigest()
    devfreq = []
    for directory in sorted(Path("/sys/class/devfreq").glob("*")):
        devfreq.append({"path": str(directory), **{name: read_optional(directory / name)
                       for name in ("name", "governor", "cur_freq", "min_freq", "max_freq")}})
    commands = []
    for executable, arguments in (("nvpmodel", ["-q"]), ("jetson_clocks", ["--show"]), ("nvidia-smi", [])):
        location = shutil.which(executable)
        if location:
            commands.append(run_readonly([location, *arguments]))
    dtype_counts = {}
    for parameter in policy._model.parameters():
        key = str(parameter.dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
    relevant_env = {name: os.environ.get(name) for name in
                    ("PYTHONPATH", "JAX_PLATFORMS", "OPENPI_DATA_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "LD_LIBRARY_PATH")}
    input_hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in fixture.items()
                    if isinstance(value, np.ndarray)}
    source_root = files["policy"].parents[3]
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "uname": list(platform.uname()),
        "versions": versions,
        "cuda": {"runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                 "device": device.name, "compute_capability": [device.major, device.minor],
                 "total_memory_bytes": device.total_memory, "multiprocessors": device.multi_processor_count,
                 "bf16_supported": torch.cuda.is_bf16_supported(),
                 "max_tensor_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "max_tensor_reserved_bytes": torch.cuda.max_memory_reserved()},
        "torch_settings": {"num_threads": torch.get_num_threads(), "num_interop_threads": torch.get_num_interop_threads(),
                           "float32_matmul_precision": torch.get_float32_matmul_precision(),
                           "cudnn_benchmark": torch.backends.cudnn.benchmark,
                           "cudnn_deterministic": torch.backends.cudnn.deterministic,
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "parameter_elements_by_dtype": dtype_counts,
        "environment": relevant_env,
        "hardware_files": {str(path): read_optional(path) for path in
                           (Path("/proc/device-tree/model"), Path("/etc/nv_tegra_release"), Path("/etc/os-release"),
                            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))},
        "devfreq": devfreq,
        "readonly_hardware_commands": commands,
        "repositories": {"openpi": repository_info(source_root), "benchmark": repository_info(Path(__file__).parent)},
        "source_files": source_files,
        "checkpoint": {"directory": str(checkpoint), "model_sha256": sha256_file(checkpoint / "model.safetensors"),
                       "size_bytes": (checkpoint / "model.safetensors").stat().st_size,
                       "norm_stats_sha256": sha256_file(checkpoint / "assets/physical-intelligence/libero/norm_stats.json"),
                       "tokenizer_model_sha256": tokenizer_hash},
        "deployment_directory": str(deployment),
        "fixture": {"seed": args.seed, "array_sha256": input_hashes, "prompt": fixture["prompt"]},
    }


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_results(output, raw, summary, provenance, fixture, reference):
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("raw.json", raw), ("summary.json", summary), ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    np.savez(output / "fixed_inputs.npz", **fixture)
    np.save(output / "reference_actions.npy", reference, allow_pickle=False)
    sample_rows = []
    for phase, samples in raw["phases"].items():
        for sample in samples:
            for metric, value in sample["timings"].items():
                sample_rows.append({"phase": phase, "sample": sample["sample"], "metric": metric, "elapsed_ms": value})
    write_csv(output / "samples.csv", sample_rows, ["phase", "sample", "metric", "elapsed_ms"])
    stage_rows = [{"sample": sample["sample"], **stage} for sample in raw["phases"]["instrumented_decomposed"]
                  for stage in sample["stages"]]
    write_csv(output / "stages.csv", stage_rows,
              ["sample", "stage", "iteration", "image", "parent", "cuda_stream_elapsed_ms"])
    summary_rows = []
    for phase, metrics in summary["phases"].items():
        for metric, statistics in metrics.items():
            summary_rows.append({"group": phase, "metric": metric, **statistics})
    for stage, statistics in summary["stage_request_totals"].items():
        summary_rows.append({"group": "instrumented_stage_request_total", "metric": stage, **statistics})
    write_csv(output / "summary.csv", summary_rows,
              ["group", "metric", "n", "mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms", "std_ms"])


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty to preserve earlier results: {args.output}")
    policy, deployment, checkpoint, helper = load_policy(args)
    observation, noise = make_fixture(args.seed)
    fixture = {"image": observation["observation/image"], "wrist_image": observation["observation/wrist_image"],
               "state": observation["observation/state"], "noise": noise, "prompt": observation["prompt"]}
    model_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    transfer_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    torch.cuda.reset_peak_memory_stats()
    raw = {"schema_version": 1, "settings": {"iterations_per_phase": args.iterations, "warmup_per_phase": args.warmup,
           "seed": args.seed, "fixed_noise": True, "denoising_steps": 10, "action_horizon": 10,
           "image_views_encoded": 3, "valid_image_views": 2, "batch_size": 1,
           "pytorch_compile_mode": None}, "warmup": {}, "phases": {}}
    reference = None

    # No model wrappers, hooks, or additional stage synchronizations in this phase.
    phase = "official_policy_uninstrumented"
    emit("phase_start", phase=phase, warmup=args.warmup, iterations=args.iterations)
    raw["warmup"][phase] = []
    samples = []
    for index in range(args.warmup + args.iterations):
        result, timings = timed_official(policy, observation, noise)
        actions, validation = check_actions(result, reference)
        if reference is None:
            reference = actions
        if index < args.warmup:
            raw["warmup"][phase].append(timings)
        else:
            samples.append({"sample": index - args.warmup, "timings": timings, "validation": validation})
            if len(samples) % 5 == 0 or len(samples) == args.iterations:
                emit("progress", phase=phase, measured=len(samples), **timings)
    raw["phases"][phase] = samples

    inputs, observation_cuda, sample_kwargs = transfer_and_observe(policy, prepare_cpu(policy, observation), noise)
    torch.cuda.synchronize()
    phase = "model_uninstrumented"
    emit("phase_start", phase=phase, warmup=args.warmup, iterations=args.iterations)
    raw["warmup"][phase] = []
    samples = []
    for index in range(args.warmup + args.iterations):
        actions, timings = timed_model(policy, observation_cuda, sample_kwargs, model_pair)
        _, validation = check_actions(finish_cpu(policy, inputs, actions), reference)
        if index < args.warmup:
            raw["warmup"][phase].append(timings)
        else:
            samples.append({"sample": index - args.warmup, "timings": timings, "validation": validation})
            if len(samples) % 5 == 0 or len(samples) == args.iterations:
                emit("progress", phase=phase, measured=len(samples), **timings)
    raw["phases"][phase] = samples
    del inputs, observation_cuda, sample_kwargs, actions

    phase = "instrumented_decomposed"
    emit("phase_start", phase=phase, warmup=args.warmup, iterations=args.iterations)
    raw["warmup"][phase] = []
    samples = []
    recorder = EventRecorder()
    with ModelInstrumentation(policy._model, recorder):
        for index in range(args.warmup + args.iterations):
            result, timings, stages = timed_decomposed(policy, observation, noise, recorder, model_pair, transfer_pair)
            _, validation = check_actions(result, reference)
            if index < args.warmup:
                raw["warmup"][phase].append(timings)
            else:
                samples.append({"sample": index - args.warmup, "timings": timings,
                                "stages": stages, "validation": validation})
                if len(samples) % 5 == 0 or len(samples) == args.iterations:
                    emit("progress", phase=phase, measured=len(samples), **timings)
    raw["phases"][phase] = samples

    # Verify restoration through the original public API after all wrappers exit.
    restored, _ = timed_official(policy, observation, noise)
    _, restored_validation = check_actions(restored, reference)
    raw["restoration_validation"] = restored_validation
    stage_names = sorted({stage["stage"] for sample in samples for stage in sample["stages"]})
    per_request = {name: summarize([sum(stage["cuda_stream_elapsed_ms"] for stage in sample["stages"]
                                      if stage["stage"] == name) for sample in samples]) for name in stage_names}
    per_call = {name: summarize([stage["cuda_stream_elapsed_ms"] for sample in samples for stage in sample["stages"]
                                if stage["stage"] == name]) for name in stage_names}
    summary = {
        "schema_version": 1,
        "phases": {name: phase_summary(values) for name, values in raw["phases"].items()},
        "stage_request_totals": per_request,
        "stage_per_call": per_call,
        "vision_per_image": {image: summarize([stage["cuda_stream_elapsed_ms"] for sample in samples
                                               for stage in sample["stages"]
                                               if stage["stage"] == "vision_encode" and stage["image"] == image])
                             for image in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb_padded")},
        "denoise_per_iteration": {str(iteration): summarize([stage["cuda_stream_elapsed_ms"] for sample in samples
                                                             for stage in sample["stages"]
                                                             if stage["stage"] == "denoise_iteration"
                                                             and stage["iteration"] == iteration])
                                  for iteration in range(1, 11)},
        "all_actions_bitwise_equal": True,
        "instrumented_model_wall_ratio":
            phase_summary(samples)["model_wall_ms"]["mean_ms"] /
            phase_summary(raw["phases"]["model_uninstrumented"])["model_wall_ms"]["mean_ms"],
        "interpretation": [
            "Official uninstrumented policy wall time is the primary baseline; it includes preprocessing, transfers, model and postprocessing, without network.",
            "Model wall time starts with CUDA-ready inputs and ends after device synchronization; it includes CPU launch/control overhead and GPU work.",
            "CUDA event intervals are elapsed time on the current CUDA stream, including possible host launch gaps; they are not summed kernel busy time or utilization.",
            "Instrumented stage intervals are inclusive and nested; vision is inside prefix embedding and expert transformer is inside denoise. Do not sum all stage rows.",
            "Instrumented decomposition adds explicit synchronization only at transfer and whole-model boundaries; wrappers never synchronize individual stages.",
            "Time MLP linear intervals exclude SiLU activations; time_mlp_total includes both SiLUs. Action_suffix_embedding additionally includes sinusoidal embedding and action input projection.",
            "All three images are encoded, including the padded right-wrist image whose attention mask is false; no model or execution optimization was applied.",
            "Each denoise iteration excludes the following Euler update and loop-condition synchronization; these appear in model remaining stream elapsed time.",
            "Fixed seeded synthetic observation and explicit fixed noise are reused for every sample. Ten actions are predicted using ten denoising iterations; these are separate concepts.",
            "Sequential phases may experience thermal or clock drift. Inspect provenance and compare instrumented model overhead before interpreting small differences.",
        ],
    }
    provenance = collect_provenance(policy, deployment, checkpoint, helper, fixture, args)
    save_results(args.output, raw, summary, provenance, fixture, reference)
    emit("complete", output=str(args.output),
         official_policy=summary["phases"]["official_policy_uninstrumented"]["full_policy_wall_ms"],
         model=summary["phases"]["model_uninstrumented"]["model_wall_ms"],
         all_actions_bitwise_equal=True)


if __name__ == "__main__":
    main()
