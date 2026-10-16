#!/usr/bin/env python3
"""Compare native pi05 output to the existing unnormalized 10x7 baseline chunk.

Only meaningful robot dimensions are compared. No threshold or robot task
success is assumed. Requires numpy; safetensors is needed for that input type.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actual", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-cosine", type=float)
    p.add_argument("--max-rmse", type=float)
    args = p.parse_args()
    reference = np.asarray(np.load(args.reference, allow_pickle=False), dtype=np.float64)
    if reference.shape != (10, 7):
        raise ValueError(f"Expected baseline LIBERO 10x7, got {reference.shape}")
    if args.actual.suffix == ".safetensors":
        from safetensors.numpy import load_file
        actual = load_file(str(args.actual))["actions"]
        normalized = True
    else:
        doc = json.loads(args.actual.read_text())
        normalized = "robot_actions" not in doc
        actual = np.asarray(doc["actions" if normalized else "robot_actions"])
    actual = np.asarray(actual, dtype=np.float64)
    if actual.shape == (1, 10, 32):
        actual = actual[0]
    if normalized:
        if actual.shape != (10, 32):
            raise ValueError(f"Expected normalized native 10x32, got {actual.shape}")
        stats = json.loads(args.norm_stats.read_text())
        stats = stats.get("norm_stats", stats)["actions"]
        q01 = np.asarray(stats["q01"], dtype=np.float64)[:7]
        q99 = np.asarray(stats["q99"], dtype=np.float64)[:7]
        actual = (actual[:, :7] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Action shape mismatch or non-finite output")
    delta = actual - reference
    denominator = np.linalg.norm(actual.ravel()) * np.linalg.norm(reference.ravel())
    cosine = float(np.dot(actual.ravel(), reference.ravel()) / denominator) if denominator else None
    record = {
        "schema_version": 1, "shape": [10, 7], "units": "LIBERO robot units",
        "comparison": "Native FP16 vs deployed openpi BF16, same fixture and x0; numerical comparison only.",
        "cosine_similarity": cosine, "rmse": float(np.sqrt(np.mean(delta**2))),
        "mae": float(np.mean(np.abs(delta))), "max_abs_error": float(np.max(np.abs(delta))),
        "files_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in [("actual", args.actual), ("reference", args.reference), ("norm_stats", args.norm_stats)]},
        "task_success_validated": False,
    }
    record["requested_gate_passed"] = (
        (args.min_cosine is None or cosine is not None and cosine >= args.min_cosine)
        and (args.max_rmse is None or record["rmse"] <= args.max_rmse)
    ) if args.min_cosine is not None or args.max_rmse is not None else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    if record["requested_gate_passed"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
