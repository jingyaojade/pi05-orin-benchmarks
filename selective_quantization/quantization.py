"""Selective W8A8 GEMM substitution on FlashRT's unmodified BF16 dataflow.

Install after ``frontend.set_prompt`` and before calibration / graph capture.
Only selected ``bf16_nn`` calls are replaced. Norms, BF16 materialization,
GELU, residuals, attention, and timestep integration remain on the BF16 path.
The original BF16 weights are retained: this is a precision/performance
experiment, not a memory-compacted deployment.

All kernels must run serially on one stream. Graph replays do not execute
Python, so submission counters describe preparation/capture calls, not GPU
replay counts. Keep the returned proxy alive for the lifetime of the graph.
"""

from __future__ import annotations

from collections import Counter
import time


MODULE_WEIGHTS = {
    "qkv": "attn_qkv_w",
    "o": "attn_o_w",
    "ffn_gate": "ffn_gate_w",
    "ffn_up": "ffn_up_w",
    "ffn_down": "ffn_down_w",
}


def _layers(values):
    if values is None:
        return ()
    if isinstance(values, str):
        if values.lower() in ("", "none"):
            return ()
        if values.lower() == "all":
            return tuple(range(18))
        expanded = []
        for part in values.split(","):
            if "-" in part:
                first, last = (int(x) for x in part.split("-", 1))
                if first > last:
                    raise ValueError(f"Invalid layer range: {part}")
                expanded.extend(range(first, last + 1))
            else:
                expanded.append(int(part))
        values = expanded
    result = tuple(sorted(set(int(x) for x in values)))
    if any(x < 0 or x >= 18 for x in result):
        raise ValueError("Layer indices must lie in [0, 17]")
    return result


def _modules(values):
    if isinstance(values, str):
        values = tuple(x.strip() for x in values.split(",") if x.strip())
    result = tuple(dict.fromkeys(values))
    invalid = set(result) - set(MODULE_WEIGHTS)
    if invalid:
        raise ValueError(f"Unknown GEMM modules: {sorted(invalid)}")
    return result


def _pointer(value):
    # Vendor pybind11 callers normally pass int; accepting c_void_p also makes
    # the adapter safe against harmless differences in pointer wrappers.
    return int(value.value if hasattr(value, "value") else value)


class SelectiveGemmProxy:
    """Delegate a GemmRunner, substituting selected weight-pointer GEMMs."""

    def __init__(self, frontend, *, encoder_layers=(), decoder_layers=(),
                 modules=tuple(MODULE_WEIGHTS), encoder_modules=None,
                 decoder_modules=None):
        import torch

        self._torch = torch
        self._pipeline = pipeline = frontend.pipeline
        if pipeline is None:
            raise RuntimeError("Call frontend.set_prompt before installing quantization")
        if getattr(pipeline, "_graph", None) is not None:
            raise RuntimeError("Install selective quantization before CUDA graph capture")
        if (pipeline.use_fp8 or pipeline.use_fp8_decoder
                or pipeline.use_int8_encoder or pipeline.use_int8_decoder):
            raise RuntimeError("Selective proxy requires the original global BF16 path")
        if isinstance(pipeline.gemm, SelectiveGemmProxy):
            raise RuntimeError("Selective quantization is already installed")

        self._delegate = pipeline.gemm
        self._fvk = pipeline.fvk
        self._by_pointer = {}
        self._keepalive = []
        self._counts = Counter()
        self._shapes = {}
        self._statuses = {}
        self._inactive = []
        self._encoder_layers = _layers(encoder_layers)
        self._decoder_layers = _layers(decoder_layers)
        common_modules = _modules(modules)
        self._encoder_modules = _modules(encoder_modules) if encoder_modules is not None else common_modules
        self._decoder_modules = _modules(decoder_modules) if decoder_modules is not None else common_modules

        selected = []
        for domain, layer_ids, module_ids in (
                ("encoder", self._encoder_layers, self._encoder_modules),
                ("decoder", self._decoder_layers, self._decoder_modules)):
            rows = int(pipeline.encoder_seq_len if domain == "encoder" else pipeline.chunk_size)
            for layer in layer_ids:
                for module in module_ids:
                    key = f"{domain}_{MODULE_WEIGHTS[module]}"
                    name = f"{key}_{layer}"
                    # Native Pi0.5 only needs last-prefix-layer K/V; allocating
                    # INT8 copies for its dead O/FFN calls would distort RAM.
                    if (domain == "encoder" and layer == 17 and module != "qkv"
                            and not getattr(pipeline, "prefix_export", False)):
                        self._inactive.append({"name": name, "reason": "last-prefix-layer O/FFN is not executed"})
                        continue
                    tensor = frontend._ckpt_bf16[key][layer]
                    if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                        raise ValueError(f"Expected contiguous BF16 [K,N] for {name}")
                    ptr = tensor.data_ptr()
                    if _pointer(pipeline.weights[key][layer]) != ptr:
                        raise RuntimeError(f"Tensor/production weight pointer mismatch: {name}")
                    if ptr in self._by_pointer or any(x[3] == ptr for x in selected):
                        raise RuntimeError(f"Selected weight aliases another substitution: {name}")
                    k, n = map(int, tensor.shape)
                    if k % 16 or n % 8:
                        raise ValueError(f"Unsupported CUTLASS alignment for {name}: K={k},N={n}")
                    selected.append((name, tensor, rows, ptr, k, n, domain, layer, module))

        torch.cuda.synchronize()
        started = time.perf_counter()
        before_allocated = int(torch.cuda.memory_allocated())
        before_reserved = int(torch.cuda.memory_reserved())
        max_elements = max((rows * k for _, _, rows, _, k, _, _, _, _ in selected), default=0)
        max_rows = max((rows for _, _, rows, _, _, _, _, _, _ in selected), default=0)
        self._act_int8 = torch.empty(max_elements, dtype=torch.int8, device="cuda")
        self._act_scales = torch.empty(max_rows, dtype=torch.float32, device="cuda")
        self._keepalive.extend((self._act_int8, self._act_scales))
        quantized_bytes = scale_bytes = source_bytes = 0
        with torch.no_grad():
            for name, tensor, rows, ptr, k, n, domain, layer, module in selected:
                # CUTLASS expects ColumnMajor [K,N], represented by contiguous
                # [N,K] storage. Each output channel receives its own scale.
                transposed = tensor.float().transpose(0, 1).contiguous()
                scale = torch.clamp(transposed.abs().amax(dim=1) / 127.0, min=1e-12).contiguous()
                quantized = torch.clamp(torch.round(transposed / scale[:, None]), -127, 127).to(torch.int8).contiguous()
                self._keepalive.extend((quantized, scale))
                self._by_pointer[ptr] = {
                    "name": name, "domain": domain, "layer": layer, "module": module,
                    "K": k, "N": n, "max_rows": rows,
                    "weight_int8_pointer": quantized.data_ptr(),
                    "weight_scale_pointer": scale.data_ptr(),
                    "source_weight_bytes": int(tensor.numel() * tensor.element_size()),
                    "weight_int8_bytes": int(quantized.numel()),
                    "weight_scale_bytes": int(scale.numel() * scale.element_size()),
                }
                quantized_bytes += quantized.numel()
                scale_bytes += scale.numel() * scale.element_size()
                source_bytes += tensor.numel() * tensor.element_size()
                # Do not retain preparation-only FP32 transposes or temporary
                # quantization expressions in the persistent memory accounting.
                del transposed, quantized, scale
        torch.cuda.synchronize()
        self._preparation = {
            "seconds": time.perf_counter() - started,
            "torch_allocated_before_bytes": before_allocated,
            "torch_allocated_after_bytes": int(torch.cuda.memory_allocated()),
            "torch_reserved_before_bytes": before_reserved,
            "torch_reserved_after_bytes": int(torch.cuda.memory_reserved()),
            "source_bf16_selected_bytes_retained": int(source_bytes),
            "persistent_weight_int8_bytes": int(quantized_bytes),
            "persistent_weight_scale_bytes": int(scale_bytes),
            "persistent_activation_scratch_bytes": int(max_elements + max_rows * 4),
        }
        self._preparation["torch_allocated_delta_bytes"] = (
            self._preparation["torch_allocated_after_bytes"] - before_allocated)
        self._preparation["torch_reserved_delta_bytes"] = (
            self._preparation["torch_reserved_after_bytes"] - before_reserved)
        self._active_names = {item["name"] for item in self._by_pointer.values()}
        self._active_encoder_layers = self._encoder_layers
        self._active_decoder_layers = self._decoder_layers
        self._active_encoder_modules = self._encoder_modules
        self._active_decoder_modules = self._decoder_modules
        self._active_inactive_requested = list(self._inactive)
        self._selection_generation = 0
        print(f"Selective W8A8 prepared {len(selected)} active weights; "
              f"kept BF16 norms/GELU, INT8 copies={quantized_bytes / 2**20:.1f} MiB; "
              f"scratch={(max_elements + max_rows * 4) / 2**20:.1f} MiB", flush=True)

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def bf16_nn(self, A, B, D, M, N, K, stream=0):
        spec = self._by_pointer.get(_pointer(B))
        if spec is None or spec["name"] not in self._active_names:
            return self._delegate.bf16_nn(A, B, D, M, N, K, stream=stream)
        M, N, K = int(M), int(N), int(K)
        if (N, K) != (spec["N"], spec["K"]) or M < 1 or M > spec["max_rows"]:
            raise RuntimeError(f"Unexpected selected GEMM geometry {spec['name']}: {(M, N, K)}")
        self._fvk.quantize_int8_rowwise(
            _pointer(A), self._act_int8.data_ptr(), self._act_scales.data_ptr(),
            M, K, stream=stream)
        status = int(self._fvk.cutlass_int8_rowwise_bf16out(
            self._act_int8.data_ptr(), spec["weight_int8_pointer"],
            self._act_scales.data_ptr(), spec["weight_scale_pointer"],
            _pointer(D), M, N, K, stream=stream))
        name = spec["name"]
        self._counts[name] += 1
        self._shapes.setdefault(name, set()).add((M, N, K))
        self._statuses[name] = status
        if status != 0:
            raise RuntimeError(f"Selective CUTLASS INT8 failed: {name} shape={(M, N, K)} status={status}")

    def reset_submission_counts(self):
        """Optional: call before capture to isolate captured graph submissions."""
        self._counts.clear()
        self._shapes.clear()
        self._statuses.clear()

    def set_active_selection(self, *, encoder_layers=(), decoder_layers=(),
                             encoder_modules=None, decoder_modules=None):
        """Select a subset of the prepared weights, then recapture the graph.

        No buffer is reallocated or moved. Existing CUDA graphs are immutable:
        replaying the old graph still runs its old precision selection. After
        this method, call ``pipeline.record_infer_graph`` on the same owned
        stream and use that newly captured graph for the next experiment.
        A synchronization here prevents changing selections while a previous
        serial experiment is still executing. Submission counts are reset so
        each screening case records its own warmup/capture submissions.
        """
        enc_layers, dec_layers = _layers(encoder_layers), _layers(decoder_layers)
        enc_modules = self._encoder_modules if encoder_modules is None else _modules(encoder_modules)
        dec_modules = self._decoder_modules if decoder_modules is None else _modules(decoder_modules)
        prepared_names = {spec["name"] for spec in self._by_pointer.values()}
        known_inactive = {item["name"]: item for item in self._inactive}
        requested, inactive = set(), []
        for domain, layers, module_ids in (
                ("encoder", enc_layers, enc_modules),
                ("decoder", dec_layers, dec_modules)):
            for layer in layers:
                for module in module_ids:
                    name = f"{domain}_{MODULE_WEIGHTS[module]}_{layer}"
                    if name in known_inactive:
                        inactive.append(known_inactive[name])
                    elif name not in prepared_names:
                        raise ValueError(f"Requested weight was not prepared for INT8: {name}")
                    else:
                        requested.add(name)
        self._torch.cuda.synchronize()
        self._active_names = requested
        self._active_encoder_layers = enc_layers
        self._active_decoder_layers = dec_layers
        self._active_encoder_modules = enc_modules
        self._active_decoder_modules = dec_modules
        self._active_inactive_requested = inactive
        self._selection_generation += 1
        self.reset_submission_counts()
        return self.metadata()

    def metadata(self):
        weights = []
        for spec in self._by_pointer.values():
            item = {key: value for key, value in spec.items() if not key.endswith("_pointer")}
            item["active"] = spec["name"] in self._active_names
            item["python_kernel_submissions"] = int(self._counts[spec["name"]])
            item["submitted_shapes"] = [list(x) for x in sorted(self._shapes.get(spec["name"], ()))]
            item["last_cutlass_status"] = self._statuses.get(spec["name"])
            weights.append(item)
        return {
            "method": "selective_W8A8_bf16_GEMM_proxy",
            "activation_quantization": "runtime_symmetric_per_row",
            "weight_quantization": "symmetric_per_output_channel_INT8",
            "activation_function": "original_BF16_GELU_unchanged",
            "bf16_weights_retained": True,
            "global_fp8_int8_flags": False,
            "encoder_layers": list(self._active_encoder_layers),
            "decoder_layers": list(self._active_decoder_layers),
            "encoder_modules": list(self._active_encoder_modules),
            "decoder_modules": list(self._active_decoder_modules),
            "prepared_encoder_layers": list(self._encoder_layers),
            "prepared_decoder_layers": list(self._decoder_layers),
            "prepared_encoder_modules": list(self._encoder_modules),
            "prepared_decoder_modules": list(self._decoder_modules),
            "prepared_weights": weights,
            "selected_active_weights": [item for item in weights if item["active"]],
            "inactive_requested_weights": list(self._active_inactive_requested),
            "prepared_inactive_requested_weights": list(self._inactive),
            "selection_generation": self._selection_generation,
            "selection_change_requires_full_graph_recapture": True,
            "resident_memory_scope": "All prepared quantized copies stay resident, including currently inactive weights. Screening memory is not fresh-deployment memory.",
            "prepared_weight_int8_bytes": sum(item["weight_int8_bytes"] for item in weights),
            "active_weight_int8_bytes": sum(item["weight_int8_bytes"] for item in weights if item["active"]),
            "preparation_memory": dict(self._preparation),
            "submission_counter_note": (
                "Counts Python kernel submissions during warmup/preparation/capture. "
                "CUDA graph replays do not reenter Python and do not increment these counters."),
            "stream_contract": "serial execution on one stream; shared quantization scratch",
        }


def install_selective_quantization(frontend, *, encoder_layers=(), decoder_layers=(),
                                   modules=tuple(MODULE_WEIGHTS), encoder_modules=None,
                                   decoder_modules=None):
    """Install and return a persistent proxy before calibration/graph capture."""
    proxy = SelectiveGemmProxy(
        frontend, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
        modules=modules, encoder_modules=encoder_modules, decoder_modules=decoder_modules)
    frontend.pipeline.gemm = proxy
    # An explicit reference on frontend documents ownership for CUDA graph
    # pointer lifetime and is useful to adapter provenance/report generation.
    frontend._selective_quantization_proxy = proxy
    return proxy
