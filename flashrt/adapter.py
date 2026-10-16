"""Patched FlashRT pi05-LIBERO adapter with the deployed input geometry.

CUDA imports are deferred so source checks and --help work on a CPU machine.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from standalone_tokenizer import NoStatePaligemmaTokenizer

PIN = "6cb2f80caec2f561a1331c554a6d3bbf95d2ed55"


def verify_patched_source(root: Path) -> dict:
    manifest = json.loads((Path(__file__).parent / "patches" / "manifest.json").read_text())
    if manifest["upstream_commit"] != PIN:
        raise RuntimeError("FlashRT patch manifest has an unexpected source pin")
    for name, hashes in manifest["files"].items():
        normalized = (root / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(normalized).hexdigest()
        if actual != hashes["patched_sha256_lf"]:
            raise RuntimeError(f"FlashRT pinned patched source does not match: {name}")
    return manifest


class LiberoFlashRT:
    def __init__(self, flashrt_root: Path, checkpoint: Path, precision: str = "bf16",
                 tokenizer_path: Path | None = None):
        self.source_manifest = verify_patched_source(flashrt_root)
        if precision not in ("bf16", "int8"):
            raise ValueError("precision must be bf16 or int8")
        # Precision is explicit; inherited experimental environment settings
        # cannot silently change the first BF16 comparison.
        os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1" if precision == "bf16" else "0"
        os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1" if precision == "int8" else "0"
        os.environ["FVK_PI05_RTX_INT8_ENCODER_ONLY"] = "0"
        os.environ["FVK_PI05_RTX_INT8_ENCODER_STATIC"] = "0"
        os.environ["FVK_PI05_RTX_INT8_VISION"] = "0"
        os.environ["FLASHRT_PI05_STATE_PROMPT_MODE"] = "fixed"
        os.environ["FLASHRT_PI05_PREFIX_PRECISION"] = "fp8"
        os.environ["FLASHRT_PI05_DECODER_PRECISION"] = "fp8"
        os.environ["FVK_RTX_FA2"] = "1"
        os.environ.pop("FVK_RTX_FA2_SITES", None)
        tokenizer_path = tokenizer_path or Path(os.environ.get("PI05_TOKENIZER_PATH",
            os.environ.get("FLASHRT_PALIGEMMA_TOKENIZER",
                "/home/nvidia/pi05-orin/cache/big_vision/paligemma_tokenizer.model")))
        self.tokenizer = NoStatePaligemmaTokenizer(tokenizer_path, max_len=200)
        sys.path.insert(0, str(flashrt_root))
        import numpy as np
        import torch
        from flash_rt.frontends.torch import pi05_rtx as frontend

        # Pin tokenization to this explicit model file instead of depending on
        # OpenPI imports or a fallback model locator in the isolated environment.
        # Populate the existing preferred-tokenizer protocol cache; no upstream
        # file or import hook changes. "openpi" is the upstream cache protocol
        # tag, not a claim that OpenPI is installed.
        frontend._TOKENIZERS[("openpi", 200)] = ("openpi", self.tokenizer)
        self._frontend_module = frontend

        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA-enabled NVIDIA PyTorch environment is required")
        if torch.cuda.get_device_capability() != (8, 7):
            raise RuntimeError("This adapter is qualified only for the Orin SM87 build")
        self.np, self.torch = np, torch
        self.precision = precision
        self.pipe = frontend.Pi05TorchFrontendRtx(
            checkpoint, hardware="rtx_sm87", num_views=3, valid_num_views=2,
            fixed_prompt_length=True, max_prompt_len=200, chunk_size=10,
            num_steps=10, vision_pool_factor=1, vision_num_layers=27,
            cache_frames=1, use_fp8=False, use_cuda_graph=True,
            state_prompt_mode="fixed", denoise_trace=False,
        )

    def prepare(self, fixture: dict) -> None:
        np = self.np
        for name in ("image", "wrist_image"):
            value = fixture[name]
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 (224, 224, 3)")
        if fixture["noise"].shape != (10, 32) or fixture["noise"].dtype != np.float32:
            raise ValueError("noise must be float32 (10, 32)")
        self.observation = {
            "images": [fixture["image"], fixture["wrist_image"], np.zeros_like(fixture["image"])],
        }
        # pi05_libero has discrete_state_input=False: no state is added to text.
        prompt = str(fixture["prompt"].item())
        self.tokenizer_provenance = self.tokenizer.provenance(prompt)
        actual_ids = self._frontend_module._prompt_token_ids(prompt, max_len=200, state=None)
        if actual_ids != self.tokenizer_provenance["valid_token_ids"]:
            raise RuntimeError("FlashRT prompt token IDs differ from the explicit no-state tokenizer")
        self.pipe.set_prompt(prompt)
        self.pipe.calibrate_with_real_data([self.observation])
        pipeline = self.pipe.pipeline
        actual = (pipeline.num_views, pipeline.valid_num_views, pipeline.encoder_seq_len,
                  pipeline.max_prompt_len, pipeline.chunk_size, pipeline.num_steps)
        if actual != (3, 2, 968, 200, 10, 10):
            raise RuntimeError(f"Unexpected comparison geometry: {actual}")
        if self.pipe.current_prompt_len != self.tokenizer_provenance["valid_token_count"]:
            raise RuntimeError("FlashRT embedded a different number of prompt tokens")
        if not pipeline._fixed_shape or self.pipe._cache_frames != 1:
            raise RuntimeError("Fixed padded prefix and cache_frames=1 are required")
        if self.precision == "bf16" and (pipeline.use_fp8 or pipeline.use_int8_encoder or pipeline.use_int8_decoder):
            raise RuntimeError("The BF16 comparison unexpectedly enabled quantization")

    def infer(self, noise):
        return self.pipe.infer(self.observation, noise=noise)["actions"]

    @property
    def model_stream(self):
        """Torch stream owning the original full graph's capture and replay."""
        return self.pipe._graph_torch_stream

    def stage_model_inputs(self, noise, *, include_images: bool = False) -> None:
        """Enqueue input restoration outside the model-only timed interval.

        The full graph mutates diffusion_noise in place, so each replay needs
        fresh fixture noise. Images are staged once before this phase because
        the graph only reads observation_images_normalized. The caller must
        synchronize these writes before starting its model timer/events.
        """
        pipe = self.pipe
        pipeline = pipe.pipeline
        stream = pipe._graph_torch_stream
        with self.torch.cuda.stream(stream):
            pipe._fill_noise(pipe._noise_buf, noise, None)
            pipe._copy_tensor_to_pipeline_buf_stream(
                pipe._noise_buf, pipeline.input_noise_buf, stream.cuda_stream)
            if include_images:
                pipe._fill_img_buf(self.observation)
                pipe._copy_tensor_to_pipeline_buf_stream(
                    pipe._img_buf, pipeline.input_images_buf, stream.cuda_stream)

    def replay_model_graph(self) -> None:
        """Launch the ORIGINAL production full graph, without a synchronization.

        CUDAGraph.replay calls cudaGraphLaunch. Unlike pipeline.forward, it
        does not synchronize internally, so the harness can put one CUDA event
        pair around this exact full-graph launch and then await GPU completion.
        No diagnostic stage graph participates in this path.
        """
        pipeline = self.pipe.pipeline
        pipeline._graph.replay(pipeline._graph_stream)

    def download_model_actions(self):
        """Download/decode GPU output after the model timer has stopped."""
        import ctypes
        from flash_rt.core.utils.actions import unnormalize_actions

        pipe = self.pipe
        pipeline = pipe.pipeline
        stream = pipe._graph_torch_stream
        # input_noise_buf is the public alias for bufs['diffusion_noise']; the
        # original full graph leaves its final action state in this same slot.
        with self.torch.cuda.stream(stream):
            pipe._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(pipe._noise_out.data_ptr()),
                pipeline.input_noise_buf.ptr, pipe._noise_out.numel() * 2, 3, stream.cuda_stream)
        stream.synchronize()
        raw = pipe._noise_out.float().cpu().numpy()
        return unnormalize_actions(raw, pipe.norm_stats)[:, :7]

    def stage_graphs(self) -> dict:
        """Capture diagnostic stages after the untouched production full graph.

        These extra graphs use the same persistent buffers, kernels, tactics,
        and stream. They do not replace or instrument the full graph.
        """
        from flash_rt.subgraphs.capture import capture_graph

        pipe = self.pipe
        pipeline = pipe.pipeline
        stream = pipe._graph_torch_stream
        handle = pipeline._graph_stream
        original = pipeline._graph
        with self.torch.cuda.stream(stream):
            vision = capture_graph(pipeline, handle, lambda: pipeline.vision_encoder(stream.cuda_stream))

            def prefill():
                pipeline._copy_lang_embeds_to_encoder_x(stream=stream.cuda_stream)
                pipeline.transformer_encoder(stream.cuda_stream)

            prefix = capture_graph(pipeline, handle, prefill)
        if pipeline._graph is not original:
            raise RuntimeError("Diagnostic stage capture replaced the production graph")
        return {"vision": vision, "prefill": prefix,
                "denoise": pipeline._decoder_only_graph}

    def run_stages(self, graphs: dict, noise, events: list) -> tuple:
        """Replay vision/prefill/denoise consecutively, then synchronize once."""
        import ctypes
        from flash_rt.core.utils.actions import unnormalize_actions

        pipe = self.pipe
        pipeline = pipe.pipeline
        stream = pipe._graph_torch_stream
        with self.torch.cuda.stream(stream):
            pipe._fill_noise(pipe._noise_buf, noise, None)
            pipe._copy_tensor_to_pipeline_buf_stream(pipe._noise_buf, pipeline.input_noise_buf, stream.cuda_stream)
            pipe._fill_img_buf(self.observation)
            pipe._copy_tensor_to_pipeline_buf_stream(pipe._img_buf, pipeline.input_images_buf, stream.cuda_stream)
            events[0].record(stream)
            for index, name in enumerate(("vision", "prefill", "denoise"), start=1):
                graphs[name].replay(pipeline._graph_stream)
                events[index].record(stream)
            pipe._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(pipe._noise_out.data_ptr()),
                pipeline.input_noise_buf.ptr, pipe._noise_out.numel() * 2, 3, stream.cuda_stream)
        stream.synchronize()
        raw = pipe._noise_out.float().cpu().numpy()
        actions = unnormalize_actions(raw, pipe.norm_stats)[:, :7]
        timings = {
            "vision_cuda_stream_elapsed_ms": events[0].elapsed_time(events[1]),
            "prefill_cuda_stream_elapsed_ms": events[1].elapsed_time(events[2]),
            "denoise_cuda_stream_elapsed_ms": events[2].elapsed_time(events[3]),
            "stage_chain_cuda_stream_elapsed_ms": events[0].elapsed_time(events[3]),
        }
        return actions, timings
