"""Isolated experiment adapter for selective INT8 and geometry candidates.

The deployed adapter and pinned vendor source remain unchanged.  All selected
GEMMs start on the BF16 model path, preserving its GELU and normalization
semantics; quantization.py may replace individual GEMMs before graph capture.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


_HERE = Path(__file__).resolve().parent
_DEPLOYED_DIR = _HERE.parent / "flashrt"
if str(_DEPLOYED_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOYED_DIR))
_SPEC = importlib.util.spec_from_file_location(
    "_pi05_deployed_flashrt_adapter", _DEPLOYED_DIR / "adapter.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load the deployed FlashRT adapter")
_DEPLOYED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEPLOYED)

PIN = _DEPLOYED.PIN
verify_patched_source = _DEPLOYED.verify_patched_source
NoStatePaligemmaTokenizer = _DEPLOYED.NoStatePaligemmaTokenizer
DEFAULT_MODULES = ("qkv", "o", "ffn_gate", "ffn_up", "ffn_down")


class LiberoExperiment(_DEPLOYED.LiberoFlashRT):
    """A fixed-geometry frontend; use one process per measured variant.

    `num_steps` controls ODE integration, while the action horizon remains 10.
    For two valid cameras, capacities 200/7 and slots 3/2 isolate text padding
    and masked-camera compaction. Three valid slots require `third_image`.
    """

    def __init__(self, flashrt_root: Path, checkpoint: Path,
                 precision: str = "bf16", tokenizer_path: Path | None = None,
                 *, num_steps: int = 10, num_views: int = 3,
                 valid_num_views: int = 2, max_prompt_len: int = 200,
                 encoder_layers=(), decoder_layers=(), modules=DEFAULT_MODULES,
                 encoder_modules=None, decoder_modules=None):
        self.source_manifest = verify_patched_source(Path(flashrt_root))
        if precision not in ("bf16", "selective_int8"):
            raise ValueError("Use precision='bf16' or 'selective_int8'; native INT8 changes GELU semantics")
        self.num_steps = int(num_steps)
        self.num_views = int(num_views)
        self.valid_num_views = int(valid_num_views)
        self.max_prompt_len = int(max_prompt_len)
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if not 1 <= self.valid_num_views <= self.num_views <= 3:
            raise ValueError("Require 1 <= valid_num_views <= num_views <= 3")
        if not 1 <= self.max_prompt_len <= 200:
            raise ValueError("This no-state experiment supports prompt capacity 1..200")
        self.encoder_layers = tuple(int(i) for i in encoder_layers)
        self.decoder_layers = tuple(int(i) for i in decoder_layers)
        self.modules = tuple(str(name) for name in modules)
        self.encoder_modules = (None if encoder_modules is None else
                                tuple(str(name) for name in encoder_modules))
        self.decoder_modules = (None if decoder_modules is None else
                                tuple(str(name) for name in decoder_modules))
        if any(i < 0 or i >= 18 for i in self.encoder_layers + self.decoder_layers):
            raise ValueError("Encoder and decoder layer indices must be in 0..17")
        if any(name not in DEFAULT_MODULES for name in self.modules):
            raise ValueError(f"Unknown GEMM module; choose from {DEFAULT_MODULES}")
        for override in (self.encoder_modules, self.decoder_modules):
            if override is not None and any(name not in DEFAULT_MODULES for name in override):
                raise ValueError(f"Unknown branch GEMM module; choose from {DEFAULT_MODULES}")
        if (self.encoder_layers or self.decoder_layers) and not self.modules:
            raise ValueError("Selected layers require at least one GEMM module")

        # Explicitly disable the native INT8/SwiGLU route and unrelated tiers.
        os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
        os.environ["FVK_PI05_RTX_FORCE_INT8"] = "0"
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
        self.tokenizer_200 = NoStatePaligemmaTokenizer(tokenizer_path, max_len=200)
        self.tokenizer = (self.tokenizer_200 if self.max_prompt_len == 200 else
                          NoStatePaligemmaTokenizer(tokenizer_path, max_len=self.max_prompt_len))
        sys.path.insert(0, str(Path(flashrt_root)))
        import numpy as np
        import torch
        from flash_rt.frontends.torch import pi05_rtx as frontend

        # Both keys are essential: shortened capacities must not select the
        # standalone engine's different fallback tokenization path.
        frontend._TOKENIZERS[("openpi", 200)] = ("openpi", self.tokenizer_200)
        frontend._TOKENIZERS[("openpi", self.max_prompt_len)] = ("openpi", self.tokenizer)
        self._frontend_module = frontend
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled NVIDIA PyTorch is required")
        if torch.cuda.get_device_capability() != (8, 7):
            raise RuntimeError("This experiment is qualified only for Orin SM87")
        self.np, self.torch = np, torch
        self.precision = ("selective_int8" if self.encoder_layers or self.decoder_layers else "bf16")
        self.quantization = None
        self._prepared = False
        self.pipe = frontend.Pi05TorchFrontendRtx(
            checkpoint, hardware="rtx_sm87", num_views=self.num_views,
            valid_num_views=self.valid_num_views, fixed_prompt_length=True,
            max_prompt_len=self.max_prompt_len, chunk_size=10,
            num_steps=self.num_steps, vision_pool_factor=1, vision_num_layers=27,
            cache_frames=1, use_fp8=False, use_cuda_graph=True,
            state_prompt_mode="fixed", denoise_trace=False,
        )

    def _set_fixture(self, fixture: dict) -> None:
        np = self.np
        names = ["image"]
        if self.num_views >= 2:
            names.append("wrist_image")
        if self.valid_num_views == 3:
            names.append("third_image")
        for name in names:
            if name not in fixture:
                raise ValueError(f"{name} is required for {self.valid_num_views} valid camera views")
            value = fixture[name]
            if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                raise ValueError(f"{name} must be uint8 (224,224,3)")
        if fixture["noise"].shape != (10, 32) or fixture["noise"].dtype != np.float32:
            raise ValueError("noise stays float32 (10,32), independently of denoise steps")
        images = [fixture["image"]]
        if self.num_views >= 2:
            images.append(fixture["wrist_image"])
        if self.num_views == 3:
            images.append(fixture["third_image"] if self.valid_num_views == 3 else
                          np.zeros_like(fixture["image"]))
        self.observation = {"images": images}
        prompt = str(np.asarray(fixture["prompt"]).item())
        provenance_200 = self.tokenizer_200.provenance(prompt)
        # Reject truncation before comparing IDs: a small text capacity must
        # only remove padding, never remove a valid token.
        cleaned = provenance_200["cleaned_prompt"]
        full_ids = (self.tokenizer_200._tokenizer.encode(cleaned, add_bos=True)
                    + self.tokenizer_200._tokenizer.encode("\n"))
        if len(full_ids) > self.max_prompt_len:
            raise ValueError(f"Prompt needs {len(full_ids)} tokens; capacity is {self.max_prompt_len}")
        provenance = self.tokenizer.provenance(prompt)
        valid_ids = provenance_200["valid_token_ids"]
        if full_ids != valid_ids or provenance["valid_token_ids"] != valid_ids:
            raise RuntimeError("Capacity change altered valid prompt token IDs")
        for capacity in (200, self.max_prompt_len):
            actual = self._frontend_module._prompt_token_ids(prompt, max_len=capacity, state=None)
            if actual != valid_ids:
                raise RuntimeError(f"Frontend token IDs differ at capacity {capacity}")
        self.tokenizer_provenance = dict(provenance)
        self.tokenizer_provenance["capacity_200_reference"] = provenance_200
        self.tokenizer_provenance["valid_ids_equal_capacity_200"] = True
        self.prompt = prompt
        self.pipe.set_prompt(prompt)

    def prepare(self, fixture: dict) -> None:
        """Select prompt, install GEMM dispatch, then calibrate and capture."""
        self._set_fixture(fixture)
        pipeline = self.pipe.pipeline
        self._check_geometry()
        if self._prepared:
            # Fixed-capacity prompt updates preserve the language buffer's
            # address; no graph rebuild/quantization reinstall is necessary.
            return
        if getattr(pipeline, "_graph", None) is not None or self.pipe.graph_recorded:
            raise RuntimeError("Selective GEMMs must be installed before calibration/graph capture")
        if self.encoder_layers or self.decoder_layers:
            if str(_HERE) not in sys.path:
                sys.path.insert(0, str(_HERE))
            from quantization import install_selective_quantization
            self.quantization = install_selective_quantization(
                self.pipe, encoder_layers=self.encoder_layers,
                decoder_layers=self.decoder_layers, modules=self.modules,
                encoder_modules=self.encoder_modules, decoder_modules=self.decoder_modules)
        self.pipe.calibrate_with_real_data([self.observation])
        self._check_geometry()
        if getattr(pipeline, "_graph", None) is None or not self.pipe.graph_recorded:
            raise RuntimeError("Calibration did not capture the original full production graph")
        self._prepared = True

    def _check_geometry(self) -> None:
        pipeline = self.pipe.pipeline
        actual = (pipeline.num_views, pipeline.valid_num_views, pipeline.encoder_seq_len,
                  pipeline.max_prompt_len, pipeline.chunk_size, pipeline.num_steps)
        expected = (self.num_views, self.valid_num_views, self.num_views * 256 + self.max_prompt_len,
                    self.max_prompt_len, 10, self.num_steps)
        if actual != expected:
            raise RuntimeError(f"Unexpected experiment geometry: {actual}, expected {expected}")
        if self.pipe.current_prompt_len != self.tokenizer_provenance["valid_token_count"]:
            raise RuntimeError("Frontend embedded a different number of prompt tokens")
        if not pipeline._fixed_shape or self.pipe._cache_frames != 1:
            raise RuntimeError("Experiment requires fixed graph geometry and fresh prefix per inference")
        if pipeline.use_fp8 or pipeline.use_int8_encoder or pipeline.use_int8_decoder:
            raise RuntimeError("Native low-precision paths must stay off; only the GEMM proxy may quantize")

    def metadata(self) -> dict:
        result = {
            "upstream_commit": PIN,
            "source_patch": self.source_manifest,
            "precision": self.precision,
            "num_views": self.num_views,
            "valid_num_views": self.valid_num_views,
            "max_prompt_len": self.max_prompt_len,
            "encoder_rows": self.num_views * 256 + self.max_prompt_len,
            "num_steps": self.num_steps,
            "action_horizon": 10,
            "action_dim": 32,
            "encoder_layers": list(self.encoder_layers),
            "decoder_layers": list(self.decoder_layers),
            "modules": list(self.modules),
            "encoder_modules": (None if self.encoder_modules is None else list(self.encoder_modules)),
            "decoder_modules": (None if self.decoder_modules is None else list(self.decoder_modules)),
            "state_in_model_conditioning": False,
            "cache_frames": 1,
            "tokenizer": getattr(self, "tokenizer_provenance", None),
        }
        if self.quantization is not None:
            method = getattr(self.quantization, "metadata", None)
            if callable(method):
                result["quantization"] = method()
        return result


# Explicit alias for callers that prefer the full experiment name.
LiberoSelectiveFlashRT = LiberoExperiment
