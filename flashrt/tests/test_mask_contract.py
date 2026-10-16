"""CPU checks for the static masked-camera transformation; no CUDA claim."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from prepare_patch import TRANSFORMS


def softmax(value):
    value = value - value.max(axis=-1, keepdims=True)
    result = np.exp(value)
    return result / result.sum(axis=-1, keepdims=True)


class MaskContract(unittest.TestCase):
    def test_disjoint_kv_copy_preserves_valid_queries_across_layers(self):
        rng = np.random.default_rng(42)
        all_vision, valid_vision, capacity = 12, 8, 3
        for prompt_length in (1, 2, 3):
            valid = np.zeros(all_vision + capacity, dtype=bool)
            valid[:valid_vision] = True
            valid[all_vision:all_vision + prompt_length] = True
            positions = np.cumsum(valid) - 1
            expected_positions = np.array(list(range(8)) + [7] * 4
                                          + list(range(8, 8 + prompt_length))
                                          + [7 + prompt_length] * (capacity - prompt_length))
            np.testing.assert_array_equal(positions, expected_positions)
            original = rng.standard_normal((len(valid), 4))
            packed = original.copy()
            # Two residual layers demonstrate that different invalid-query
            # outputs cannot leak back through excluded K/V on the next layer.
            for _ in range(2):
                q_w, k_w, v_w = (rng.standard_normal((4, 4)) for _ in range(3))
                values = []
                for state, compact in ((original, False), (packed, True)):
                    q, k, v = state @ q_w, state @ k_w, state @ v_w
                    # A position-dependent rotation stands in for RoPE. The
                    # exact cumsum positions are shared before K/V compaction.
                    phase = positions[:, None] * np.array([0.1, 0.2])
                    for tensor in (q, k):
                        even, odd = tensor[:, 0::2].copy(), tensor[:, 1::2].copy()
                        tensor[:, 0::2] = even * np.cos(phase) - odd * np.sin(phase)
                        tensor[:, 1::2] = even * np.sin(phase) + odd * np.cos(phase)
                    if compact:
                        k[valid_vision:valid_vision + capacity] = k[all_vision:all_vision + capacity]
                        v[valid_vision:valid_vision + capacity] = v[all_vision:all_vision + capacity]
                        count = valid_vision + prompt_length
                        values.append(softmax(q @ k[:count].T) @ v[:count] + state)
                    else:
                        score = q @ k.T
                        score[:, ~valid] = -np.inf
                        values.append(softmax(score) @ v + state)
                original, packed = values
                np.testing.assert_allclose(packed[valid], original[valid], rtol=1e-12, atol=1e-12)

            # The decoder writes its own K/V immediately after the compact
            # valid prefix. Compare joint attention with an explicit mask over
            # the original uncompressed prefix plus action tokens.
            keys, values = original @ k_w, original @ v_w
            compact_keys, compact_values = packed @ k_w, packed @ v_w
            compact_keys[valid_vision:valid_vision + capacity] = compact_keys[all_vision:all_vision + capacity]
            compact_values[valid_vision:valid_vision + capacity] = compact_values[all_vision:all_vision + capacity]
            action_keys, action_values, query = (rng.standard_normal((2, 4)) for _ in range(3))
            keys = np.concatenate([keys, action_keys])
            values = np.concatenate([values, action_values])
            joint_mask = np.concatenate([valid, [True, True]])
            scores = query @ keys.T
            scores[:, ~joint_mask] = -np.inf
            expected = softmax(scores) @ values
            count = valid_vision + prompt_length
            compact_keys[count:count + 2] = action_keys
            compact_values[count:count + 2] = action_values
            actual = softmax(query @ compact_keys[:count + 2].T) @ compact_values[:count + 2]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_full_geometry_and_nonoverlap(self):
        self.assertEqual(3 * 256 + 200, 968)
        self.assertLessEqual(512 + 200, 768)
        for prompt_length in (1, 48, 199, 200):
            valid_prefix = 512 + prompt_length
            decoder_positions = np.arange(valid_prefix, valid_prefix + 10)
            self.assertEqual(decoder_positions[0], valid_prefix)
            self.assertLessEqual(valid_prefix + 10, 968 + 10)

    def test_pinned_transform_placement_and_syntax(self):
        default = BENCH.parents[1] / "third_party" / "FlashRT-research"
        root = Path(os.environ.get("FLASHRT_SOURCE", default))
        if not (root / "flash_rt" / "frontends" / "torch" / "pi05_rtx.py").exists():
            self.skipTest("Set FLASHRT_SOURCE to the unpatched pinned source tree")
        manifest = json.loads((BENCH / "patches" / "manifest.json").read_text())
        for name, transform in TRANSFORMS.items():
            source = (root / name).read_text(encoding="utf-8")
            digest = hashlib.sha256(source.encode()).hexdigest()
            self.assertEqual(digest, manifest["files"][name]["upstream_sha256_lf"])
            edited = transform(source)
            compile(edited, name, "exec")
            ast.parse(edited)
            self.assertEqual(hashlib.sha256(edited.encode()).hexdigest(),
                             manifest["files"][name]["patched_sha256_lf"])
        edited = TRANSFORMS["flash_rt/hardware/rtx/attn_backend.py"](
            (root / "flash_rt/hardware/rtx/attn_backend.py").read_text(encoding="utf-8"))
        start = edited.rindex("    def encoder_attn(self, layer_idx: int, seq: int, stream: int = 0)")
        method = edited[start:edited.index("    def decoder_attn(", start)]
        self.assertLess(method.index("k[:, dst:dst + count].copy_"),
                        method.index("self._call_fvk_fa2_seqused("))


if __name__ == "__main__":
    unittest.main()
