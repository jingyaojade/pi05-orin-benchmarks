"""Serve the pi05-LIBERO PyTorch checkpoint without training dependencies.

The transform order matches pi05_libero in openpi.training.config, including
quantile normalization and the absence of an extra delta-action transform.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import time

# JAX is used by shared tree utilities; all model computation runs in PyTorch.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import safetensors.torch
import torch
import websockets.asyncio.server as websocket_server

from openpi import transforms
from openpi.models import pi0_config
from openpi.models import tokenizer
from openpi.models_pytorch import pi0_pytorch
from openpi.policies import libero_policy
from openpi.policies import policy as policy_module
from openpi.serving import websocket_policy_server
from openpi.shared import normalize

LOGGER = logging.getLogger("openpi.orin")


class OrinWebsocketPolicyServer(websocket_policy_server.WebsocketPolicyServer):
    """Keep slow, synchronous Orin inference from triggering ping timeouts."""

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
            process_request=websocket_policy_server._health_check,
        ) as server:
            LOGGER.info("Ready: ws://%s:%s (GET /healthz)", self._host, self._port)
            await server.serve_forever()


def create_policy(checkpoint: pathlib.Path) -> policy_module.Policy:
    """Load official weights and reproduce the official LIBERO policy pipeline."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; install the Jetson-compatible CUDA PyTorch build.")

    weight_path = checkpoint / "model.safetensors"
    norm_dir = checkpoint / "assets" / "physical-intelligence" / "libero"
    if not weight_path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {weight_path}")
    norm_stats = normalize.load(norm_dir)
    if not {"state", "actions"}.issubset(norm_stats):
        raise ValueError(f"LIBERO normalization requires state and actions statistics: {norm_dir}")

    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        pytorch_compile_mode=None,
    )
    LOGGER.info("Loading pi05_libero weights from %s", weight_path)
    model = pi0_pytorch.PI0Pytorch(config)
    safetensors.torch.load_model(model, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # This is the pi05_libero ModelTransformFactory path, written explicitly to
    # avoid importing training.config and its dataset dependencies.
    input_transforms = [
        transforms.InjectDefaultPrompt(None),
        libero_policy.LiberoInputs(model_type=config.model_type),
        transforms.Normalize(norm_stats, use_quantiles=True),
        transforms.InjectDefaultPrompt(None),
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(
            tokenizer.PaligemmaTokenizer(config.max_token_len),
            discrete_state_input=False,
        ),
        transforms.PadStatesAndActions(config.action_dim),
    ]
    output_transforms = [
        transforms.Unnormalize(norm_stats, use_quantiles=True),
        libero_policy.LiberoOutputs(),
    ]
    return policy_module.Policy(
        model,
        transforms=input_transforms,
        output_transforms=output_transforms,
        is_pytorch=True,
        pytorch_device="cuda",
        metadata={
            "model": "pi05_libero",
            "backend": "pytorch",
            "device": torch.cuda.get_device_name(0),
            "action_horizon": config.action_horizon,
            "action_dim": 7,
            "state_dim": 8,
            "normalization": "quantile",
            "discrete_state_input": False,
            "pytorch_compile_mode": None,
        },
    )


def warmup(policy: policy_module.Policy) -> None:
    """Verify complete preprocessing, CUDA inference, and action decoding."""
    rng = np.random.default_rng(0)
    observation = {
        "observation/state": np.zeros(8, dtype=np.float32),
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the black bowl",
    }
    LOGGER.info("Warming up on a synthetic observation before opening the port")
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = policy.infer(observation)
    torch.cuda.synchronize()
    actions = np.asarray(result["actions"])
    if actions.shape != (10, 7) or not np.isfinite(actions).all():
        raise RuntimeError(f"Warmup returned invalid actions: shape={actions.shape}")
    LOGGER.info(
        "Warmup passed: shape=%s, elapsed=%.3fs, peak CUDA tensor memory=%.2f GiB",
        actions.shape,
        time.perf_counter() - start,
        torch.cuda.max_memory_allocated() / 2**30,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    policy = create_policy(args.checkpoint.expanduser().resolve())
    warmup(policy)
    OrinWebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
