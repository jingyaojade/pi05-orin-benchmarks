#!/usr/bin/env python3
"""Convert the existing baseline fixture to the native pi05 CLI's two input modes.

Does not load weights or CUDA. Requires numpy, sentencepiece, Pillow. This is
the LIBERO/224x224 contract only; all derived artifact hashes are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import numpy as np
    from PIL import Image
    import sentencepiece

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.fixture, allow_pickle=False) as f:
        fixture = {key: f[key] for key in f.files}
    for key in ("image", "wrist_image"):
        if fixture[key].shape != (224, 224, 3) or fixture[key].dtype != np.uint8:
            raise ValueError(f"Expected exact baseline RGB uint8 224x224 fixture, got {key}: {fixture[key].shape}/{fixture[key].dtype}")
    noise = np.asarray(fixture["noise"], dtype="<f4")
    if noise.shape == (10, 32):
        noise = noise[None]
    if noise.shape != (1, 10, 32):
        raise ValueError(f"Wrong x0: {noise.shape}")
    prompt = str(fixture["prompt"].item())
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    sp = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    tokens = (sp.encode(cleaned, add_bos=True) + sp.encode("\n"))[:200]
    # Canonical path intentionally compacts the masked third view and text padding,
    # following the native upstream contract. No state text for pi05_libero.
    pixels = np.stack([fixture["image"], fixture["wrist_image"]]).astype(np.float32) / 255.0 * 2.0 - 1.0
    pixels = np.ascontiguousarray(pixels.transpose(0, 3, 1, 2), dtype="<f2")
    pixels.tofile(args.output / "pixel_values.bin")
    noise.tofile(args.output / "x0.bin")
    (args.output / "token_ids.csv").write_text(",".join(str(int(x)) for x in tokens) + "\n")
    Image.fromarray(fixture["image"]).save(args.output / "base.png")
    Image.fromarray(fixture["wrist_image"]).save(args.output / "wrist.png")
    request = {"task": prompt, "state": fixture["state"].tolist(),
               "cameras": {"observation/image": str((args.output / "base.png").resolve()),
                           "observation/wrist_image": str((args.output / "wrist.png").resolve())}}
    (args.output / "observation.json").write_text(json.dumps(request, indent=2) + "\n")
    record = {"schema_version": 1, "fixture_sha256": sha(args.fixture),
              "tokenizer_sha256": sha(args.tokenizer), "policy_config": "pi05_libero",
              "pixel_values": {"shape": list(pixels.shape), "dtype": "little-endian float16"},
              "noise": {"shape": list(noise.shape), "dtype": "little-endian float32"},
              "token_count": len(tokens), "prefix_len": 512 + len(tokens),
              "masked_padding_compacted": True,
              "precision_note": "Pixel cast to FP16 is required by upstream; baseline uses FP32 pixels then BF16 model.",
              "files": {file.name: sha(file) for file in args.output.iterdir() if file.is_file()}}
    (args.output / "input_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
