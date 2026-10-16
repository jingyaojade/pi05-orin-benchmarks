#!/usr/bin/env python3
"""Memory-only companion to the unchanged baseline/FlashRT latency harnesses.

Run each backend in a fresh process. Extra stage synchronizations and host
sampling deliberately invalidate latency comparisons from this script.
"""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from memory_common import MemoryRecorder, cuda_buffer_inventory, write_json


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("baseline", "flashrt"), required=True)
    parser.add_argument("--deployment-root", type=Path, default=Path("/home/nvidia/pi05-orin"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--flashrt-root", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--precision", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3, help="Memory samples, not latency repetitions")
    parser.add_argument("--sample-interval", type=float, default=0.1, help="Host/nvmap polling during full-model passes")
    parser.add_argument("--disable-nvmap", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.sample_interval < 0.02:
        parser.error("iterations>0, warmup>=0 and sample-interval>=0.02 are required")
    if args.backend == "flashrt" and args.flashrt_root is None:
        parser.error("--flashrt-root is required for flashrt")
    args.checkpoint = args.checkpoint or args.deployment_root / "models" / "pi05_libero"
    args.output = args.output.expanduser().resolve()
    return args


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_actions(actions, reference, np, *, exact):
    actions = np.asarray(actions)
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise AssertionError("Invalid action output")
    equal = bool(np.array_equal(actions, reference))
    delta = actions.astype(np.float64) - reference.astype(np.float64)
    if exact and not equal:
        raise AssertionError(f"Memory instrumentation changed actions, max abs={float(np.abs(delta).max())}")
    a, b = actions.astype(np.float64).ravel(), reference.astype(np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"exact_match": equal, "rmse": float(np.sqrt(np.mean(delta ** 2))),
            "max_abs": float(np.abs(delta).max()),
            "cosine": float(a @ b / denominator) if denominator else None}


class BaselineStages(AbstractContextManager):
    """Only disjoint stage wrappers. No parent scope resets the peak counter."""
    def __init__(self, model, memory):
        self.model, self.memory = model, memory
        self.restore = []
        self.sample = None
        self.vision = self.denoise = 0

    def begin(self, sample):
        self.sample = sample
        self.vision = self.denoise = 0

    def replace(self, target, name, factory):
        local = name in target.__dict__
        previous = target.__dict__.get(name)
        original = getattr(target, name)
        self.restore.append((target, name, local, previous))
        setattr(target, name, factory(original))

    def fixed(self, target, name, stage):
        def factory(original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                with self.memory.measure(stage, sample=self.sample):
                    return original(*args, **kwargs)
            return wrapper
        self.replace(target, name, factory)

    def __enter__(self):
        def vision_factory(original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                self.vision += 1
                with self.memory.measure("vision", sample=self.sample, iteration=self.vision):
                    return original(*args, **kwargs)
            return wrapper

        def denoise_factory(original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                self.denoise += 1
                with self.memory.measure("denoise_iteration", sample=self.sample, iteration=self.denoise):
                    return original(*args, **kwargs)
            return wrapper

        def forward_factory(original):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                embeddings = kwargs.get("inputs_embeds", args[3] if len(args) > 3 else None)
                if embeddings is None:
                    raise RuntimeError("Cannot classify PaliGemma forward")
                if embeddings[1] is None:
                    with self.memory.measure("prefill", sample=self.sample):
                        return original(*args, **kwargs)
                if embeddings[0] is None:
                    # Already inside the denoise_iteration region: no reset.
                    return original(*args, **kwargs)
                raise RuntimeError("Unexpected joint forward in inference")
            return wrapper

        try:
            self.fixed(self.model, "_preprocess_observation", "model_preprocess")
            self.replace(self.model.paligemma_with_expert, "embed_image", vision_factory)
            self.fixed(self.model.paligemma_with_expert, "embed_language_tokens", "language_embedding")
            self.replace(self.model.paligemma_with_expert, "forward", forward_factory)
            self.replace(self.model, "denoise_step", denoise_factory)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for target, name, local, previous in reversed(self.restore):
            if local:
                setattr(target, name, previous)
            else:
                delattr(target, name)
        self.restore.clear()


def run_baseline(args, memory, fixture, expected, np, torch, repo):
    module = load_module(repo / "baseline" / "benchmark.py", "_memory_baseline")
    with memory.measure("backend_load", sample_host=True):
        policy, deployment, checkpoint, helper = module.load_policy(args)
    memory.checkpoint("backend_loaded")
    observation = {"observation/image": fixture["image"], "observation/wrist_image": fixture["wrist_image"],
                   "observation/state": fixture["state"], "prompt": str(fixture["prompt"].item())}
    noise = fixture["noise"]
    first = np.asarray(policy.infer(observation, noise=noise)["actions"]).copy()
    validations = {"against_saved_baseline": compare_actions(first, expected, np, exact=True), "passes": []}
    for _ in range(args.warmup):
        compare_actions(policy.infer(observation, noise=noise)["actions"], first, np, exact=True)
    memory.checkpoint("steady_state_after_warmup")

    # A separate pass obtains an unbroken model-wide peak. Stage wrappers are
    # absent, so no nested scope can reset allocator peak counters.
    inputs_cpu = module.prepare_cpu(policy, observation)
    inputs, observation_cuda, sample_kwargs = module.transfer_and_observe(policy, inputs_cpu, noise)
    torch.cuda.synchronize()
    for index in range(args.iterations):
        with memory.measure("model_total", sample=index, sample_host=True):
            actions = policy._sample_actions(policy._pytorch_device, observation_cuda, **sample_kwargs)
        result = module.finish_cpu(policy, inputs, actions)
        validations["passes"].append(compare_actions(result["actions"], first, np, exact=True))
        del actions, result
    del inputs_cpu, inputs, observation_cuda, sample_kwargs

    with BaselineStages(policy._model, memory) as stages:
        for index in range(args.iterations):
            stages.begin(index)
            with memory.measure("cpu_preprocess", sample=index):
                inputs_cpu = module.prepare_cpu(policy, observation)
            with memory.measure("h2d_and_observation", sample=index):
                inputs, observation_cuda, sample_kwargs = module.transfer_and_observe(policy, inputs_cpu, noise)
            actions = policy._sample_actions(policy._pytorch_device, observation_cuda, **sample_kwargs)
            with memory.measure("d2h_and_postprocess", sample=index):
                result = module.finish_cpu(policy, inputs, actions)
            validations["passes"].append(compare_actions(result["actions"], first, np, exact=True))
            if stages.vision != 3 or stages.denoise != 10:
                raise RuntimeError(f"Unexpected stage counts {stages.vision}/{stages.denoise}")
            del inputs_cpu, inputs, observation_cuda, sample_kwargs, actions, result
    validations["wrapper_restoration"] = compare_actions(policy.infer(observation, noise=noise)["actions"], first, np, exact=True)
    memory.checkpoint("after_memory_passes")
    return validations, {"helper": str(helper), "backend_source": str(module.__file__)}


def run_flashrt(args, memory, fixture, expected, np, torch, repo):
    sys.path.insert(0, str(repo / "flashrt"))
    from adapter import LiberoFlashRT
    with memory.measure("backend_load", sample_host=True):
        adapter = LiberoFlashRT(args.flashrt_root, args.checkpoint, args.precision, args.tokenizer)
    memory.checkpoint("backend_loaded_before_prepare")
    with memory.measure("prepare_and_capture", sample_host=True):
        adapter.prepare(fixture)
    memory.checkpoint("backend_loaded_and_production_graph_captured")
    first = adapter.infer(fixture["noise"])
    validations = {"against_saved_baseline": compare_actions(first, expected, np, exact=False), "passes": []}
    for _ in range(args.warmup):
        compare_actions(adapter.infer(fixture["noise"]), first, np, exact=True)
    memory.checkpoint("steady_state_after_warmup")
    from flash_rt.core.cuda_buffer import CudaBuffer
    inventory = {"production_steady_state": cuda_buffer_inventory(adapter.pipe, CudaBuffer)}
    production_graph = adapter.pipe.pipeline._graph
    adapter.stage_model_inputs(fixture["noise"], include_images=True)
    torch.cuda.synchronize()
    for index in range(args.iterations):
        adapter.stage_model_inputs(fixture["noise"])
        torch.cuda.synchronize()
        with memory.measure("model_total", sample=index, sample_host=True):
            adapter.replay_model_graph()
        validations["passes"].append(compare_actions(adapter.download_model_actions(), first, np, exact=True))

    # Diagnostic graphs are captured only after the untouched production
    # full-model pass; they can allocate graph metadata, which is recorded.
    with memory.measure("diagnostic_graph_capture", sample_host=True):
        graphs = adapter.stage_graphs()
    memory.checkpoint("diagnostic_graphs_captured")
    inventory["after_diagnostic_capture"] = cuda_buffer_inventory(adapter.pipe, CudaBuffer)
    pipe, pipeline, stream = adapter.pipe, adapter.pipe.pipeline, adapter.model_stream
    for index in range(args.iterations):
        # Existing frontend routines combine CPU conversion with Torch staging.
        # Report these boundaries honestly instead of inventing CPU-only/H2D.
        with memory.measure("input_prepare_and_staging", sample=index):
            adapter.stage_model_inputs(fixture["noise"], include_images=True)
        for stage in ("vision", "prefill", "denoise"):
            with memory.measure(stage, sample=index):
                graphs[stage].replay(pipeline._graph_stream)
        with memory.measure("d2h_and_postprocess", sample=index):
            result = adapter.download_model_actions()
        validations["passes"].append(compare_actions(result, first, np, exact=True))
    if pipeline._graph is not production_graph:
        raise AssertionError("Diagnostic graphs replaced the production graph")
    validations["production_graph_restoration"] = compare_actions(adapter.infer(fixture["noise"]), first, np, exact=True)
    memory.checkpoint("after_memory_passes")
    write_json(args.output / "cuda_buffer_inventory.json", inventory)
    return validations, {"upstream_commit": "6cb2f80caec2f561a1331c554a6d3bbf95d2ed55",
                         "source_patch": adapter.source_manifest, "tokenizer": adapter.tokenizer_provenance}


def main():
    args = arguments()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    memory = MemoryRecorder(args.output, args.sample_interval, args.disable_nvmap)
    memory.checkpoint("python_before_numpy_torch_import")
    import numpy as np
    import torch
    memory.torch = torch
    torch.cuda.init()
    memory.checkpoint("cuda_initialized_before_backend_load")
    repo = Path(__file__).resolve().parent.parent
    fixture_path = args.fixture.expanduser().resolve()
    reference_path = (args.reference or fixture_path.with_name("reference_actions.npy")).expanduser().resolve()
    with np.load(fixture_path, allow_pickle=False) as archive:
        fixture = {key: archive[key] for key in ("image", "wrist_image", "state", "noise", "prompt")}
    expected = np.load(reference_path, allow_pickle=False)
    metadata = {"backend": args.backend, "precision": args.precision if args.backend == "flashrt" else "official_mixed_bf16_fp32",
        "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "iterations": args.iterations, "warmup": args.warmup, "sample_interval_seconds": args.sample_interval,
        "fixture": str(fixture_path), "fixture_sha256": sha256(fixture_path),
        "reference": str(reference_path), "reference_sha256": sha256(reference_path),
        "checkpoint": str(args.checkpoint), "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "python": platform.python_version(), "platform": platform.platform(), "gpu": torch.cuda.get_device_name(),
        "profiler_hashes": {name: sha256(Path(__file__).with_name(name)) for name in ("memory.py", "memory_common.py")},
        "caveats": [
            "Memory-only pass with synchronizations, peak resets and /proc/nvmap polling; its duration is not a latency benchmark.",
            "Jetson unified-memory accounting domains overlap. Never sum process RSS/PSS/USS with nvmap or CUDA device used bytes.",
            "RSS/PSS/USS are process host mappings (USS=Private_Clean+Private_Dirty); VmHWM is a process-lifetime RSS high-water mark.",
            "nvmap self_bytes sums all debugfs clients for this PID; it includes driver-accounted allocations, not a disjoint physical-memory total.",
            "Torch peaks are exact only for the Torch caching allocator; FlashRT cudaMalloc buffers, graph internals and libraries can be outside it.",
            "cudaMemGetInfo used bytes cover the entire device, including other processes; changes are not exact per-process allocations.",
            "max_observed process/nvmap values are sampled maxima or synchronized endpoints, hence lower bounds on transient peaks; no kernel allocation tracing is claimed.",
            "Stage before/after memory includes persistent weights/KV/scratch allocations shared by stages; stage values cannot be added.",
            "Model-total peak and detailed-stage peaks use separate passes; stage peak resets never occur inside the model-total pass.",
            "FlashRT vision/prefill/denoise stage memory replays diagnostic graphs sharing persistent arenas; the production graph is measured separately first.",
            "FlashRT input_prepare_and_staging combines CPU image/noise conversion and H2D/D2D staging from the unchanged frontend; no artificial CPU-only split.",
            "Profiler bookkeeping and sampling threads consume some host memory; all values include the profiler process overhead.",
        ]}
    write_json(args.output / "provenance.json", metadata)
    try:
        run = run_baseline if args.backend == "baseline" else run_flashrt
        validation, source = run(args, memory, fixture, expected, np, torch, repo)
        metadata["source"] = source
        metadata["validation"] = validation
        metadata["completed"] = True
    except BaseException as error:
        metadata["completed"] = False
        metadata["failure"] = {"type": type(error).__name__, "message": str(error)}
        memory.save(metadata)
        write_json(args.output / "provenance.json", metadata)
        raise
    memory.save(metadata)
    write_json(args.output / "provenance.json", metadata)
    print(json.dumps({"event": "memory_complete", "output": str(args.output), "backend": args.backend}), flush=True)


if __name__ == "__main__":
    main()
