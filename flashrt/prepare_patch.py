"""Generate/apply the reviewed FlashRT suffix-camera-mask patch for one source pin.

No GPU imports are needed. Original-file hashes reject a different upstream tree.
The default only writes the unified patch; --apply also changes the selected tree.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

PIN = "6cb2f80caec2f561a1331c554a6d3bbf95d2ed55"


def write_lf(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one upstream anchor, found {source.count(old)}: {old[:90]!r}")
    return source.replace(old, new, 1)


def transform_frontend(source: str) -> str:
    source = replace_once(source,
        '                 decoder_precision: Optional[str] = None):',
        '                 decoder_precision: Optional[str] = None,\n'
        '                 valid_num_views: Optional[int] = None,\n'
        '                 fixed_prompt_length: bool = False):')
    source = replace_once(source,
        '        self.max_prompt_len = int(max_prompt_len)\n',
        '        self.max_prompt_len = int(max_prompt_len)\n'
        '        # Local Orin comparison patch: retain padded camera compute,\n'
        '        # but exclude suffix-camera keys from prefix/joint attention.\n'
        '        self._valid_num_views = (self.num_views if valid_num_views is None\n'
        '                                 else int(valid_num_views))\n'
        '        self._fixed_prompt_length = bool(fixed_prompt_length)\n'
        '        if not 1 <= self._valid_num_views <= self.num_views:\n'
        '            raise ValueError("valid_num_views must be in [1, num_views]")\n'
        '        if self._valid_num_views < self.num_views and not self._fixed_prompt_length:\n'
        '            raise ValueError("masked suffix cameras require fixed_prompt_length=True")\n')
    source = replace_once(source,
        '        max_len = (PI05_STATE_PROMPT_MAX_LEN if state is not None\n'
        '                   else MAX_PROMPT_LEN_DEFAULT)',
        '        max_len = (self.max_prompt_len if self._fixed_prompt_length else\n'
        '                   PI05_STATE_PROMPT_MAX_LEN if state is not None else\n'
        '                   MAX_PROMPT_LEN_DEFAULT)')
    source = replace_once(source,
        '        if self._state_prompt_mode == "fixed" and state is not None:\n',
        '        if self._fixed_prompt_length or (self._state_prompt_mode == "fixed" and state is not None):\n')
    source = replace_once(source,
        '        self._ensure_prompt_capacity(PI05_STATE_PROMPT_MAX_LEN)\n'
        '        if self._fixed_pipeline is None:',
        '        fixed_capacity = (self.max_prompt_len if self._fixed_prompt_length\n'
        '                          else PI05_STATE_PROMPT_MAX_LEN)\n'
        '        self._ensure_prompt_capacity(fixed_capacity)\n'
        '        if self._fixed_pipeline is None:')
    source = replace_once(source,
        '                        PI05_STATE_PROMPT_MAX_LEN)\n'
        '            pipeline_weights = self._build_pipeline_weights()',
        '                        fixed_capacity)\n'
        '            pipeline_weights = self._build_pipeline_weights()')
    source = replace_once(source,
        '                max_prompt_len=PI05_STATE_PROMPT_MAX_LEN,\n',
        '                max_prompt_len=fixed_capacity,\n'
        '                valid_num_views=self._valid_num_views,\n')
    return source


def transform_pipeline(source: str) -> str:
    source = replace_once(source,
        '                 decoder_precision: str = "fp8"):',
        '                 decoder_precision: str = "fp8",\n'
        '                 valid_num_views: int | None = None):')
    source = replace_once(source,
        '        self.total_kv = self.encoder_seq_len + self.chunk_size\n',
        '        self.total_kv = self.encoder_seq_len + self.chunk_size\n'
        '        self.valid_num_views = (self.num_views if valid_num_views is None\n'
        '                                else int(valid_num_views))\n'
        '        if not 1 <= self.valid_num_views <= self.num_views:\n'
        '            raise ValueError("valid_num_views must be in [1, num_views]")\n'
        '        self.valid_vision_seq_enc = self.valid_num_views * VIS_SEQ_PER_VIEW // (pf * pf)\n'
        '        if self.valid_num_views < self.num_views:\n'
        '            if not self._fixed_shape or pf != 1 or self.prefix_export:\n'
        '                raise ValueError("suffix-camera masking requires fixed shape, pool=1, no prefix export")\n'
        '        self.attn.set_masked_view_layout(\n'
        '            self.vision_seq_enc, self.valid_vision_seq_enc, self.max_prompt_len)\n')
    source = replace_once(source,
        '        start = self.vision_seq_enc + prompt_len\n',
        '        start = self.valid_vision_seq_enc + prompt_len\n')
    source = replace_once(source,
        '        self._current_prompt_len = prompt_len\n\n'
        '        # Update decoder RoPE slice for this prompt length',
        '        self._current_prompt_len = prompt_len\n'
        '        # Match openpi cumsum(valid_mask)-1 in the ORIGINAL Q-row\n'
        '        # layout. K/V alone are compacted by the attention backend.\n'
        '        valid_mask = np.zeros(self.encoder_seq_len, dtype=np.int32)\n'
        '        valid_mask[:self.valid_vision_seq_enc] = 1\n'
        '        valid_mask[self.vision_seq_enc:self.vision_seq_enc + prompt_len] = 1\n'
        '        positions = np.cumsum(valid_mask) - 1\n'
        '        self.bufs["encoder_rope_weights"].upload(\n'
        '            np.ascontiguousarray(self._rope_table_np[positions]))\n\n'
        '        # Update decoder RoPE slice for this prompt length')
    source = replace_once(source,
        '            self.attn.set_fixed_valid_len(self.vision_seq_enc + prompt_len)',
        '            self.attn.set_fixed_valid_len(self.valid_vision_seq_enc + prompt_len)')
    return source


def transform_attention(source: str) -> str:
    source = replace_once(source,
        '        self._fixed_shape = False\n',
        '        self._fixed_shape = False\n'
        '        self._masked_view_layout = None\n')
    source = replace_once(source,
        '    def set_fixed_shape(self, enabled: bool) -> None:\n',
        '    def set_masked_view_layout(self, all_vision: int, valid_vision: int,\n'
        '                               prompt_capacity: int) -> None:\n'
        '        """Configure a static suffix-camera mask; Q/GEMM shapes stay full.\n'
        '\n'
        '        Prompt K/V move into the masked camera gap. Copy the entire\n'
        '        capacity so captured memcpy nodes survive prompt-length changes.\n'
        '        The disjoint source/destination ranges avoid overlapping copies.\n'
        '        """\n'
        '        gap = int(all_vision) - int(valid_vision)\n'
        '        if gap < 0 or (gap and int(prompt_capacity) > gap):\n'
        '            raise ValueError("prompt capacity must fit the masked camera gap")\n'
        '        self._masked_view_layout = (\n'
        '            (int(all_vision), int(valid_vision), int(prompt_capacity))\n'
        '            if gap else None)\n\n'
        '    def set_fixed_shape(self, enabled: bool) -> None:\n')
    source = replace_once(source,
        '        v = self.enc_V[layer_idx, :seq].unsqueeze(0)     # (1, seq, 1, 256)\n'
        '        if self._fa2_sites["encoder"]:',
        '        v = self.enc_V[layer_idx, :seq].unsqueeze(0)     # (1, seq, 1, 256)\n'
        '        if self._masked_view_layout is not None:\n'
        '            if not self._fixed_shape:\n'
        '                raise RuntimeError("masked camera keys require the fixed FA2 seqused path")\n'
        '            src, dst, count = self._masked_view_layout\n'
        '            # Both copy_ calls record static D2D memcpy nodes. Their\n'
        '            # source/destination views are disjoint and use the same\n'
        '            # torch stream as graph capture and all other FA2 work.\n'
        '            k[:, dst:dst + count].copy_(k[:, src:src + count])\n'
        '            v[:, dst:dst + count].copy_(v[:, src:src + count])\n'
        '        if self._fa2_sites["encoder"]:')
    return source


TRANSFORMS = {
    "flash_rt/frontends/torch/pi05_rtx.py": transform_frontend,
    "flash_rt/models/pi05/pipeline_rtx.py": transform_pipeline,
    "flash_rt/hardware/rtx/attn_backend.py": transform_attention,
}


def generate(root: Path, apply: bool = False) -> dict:
    out = Path(__file__).resolve().parent / "patches"
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    patch_chunks = []
    files = {}
    pending = []
    for name, transform in TRANSFORMS.items():
        raw = (root / name).read_bytes()
        source = raw.decode("utf-8").replace("\r\n", "\n")
        original_sha = hashlib.sha256(source.encode()).hexdigest()
        if manifest and original_sha != manifest["files"][name]["upstream_sha256_lf"]:
            raise ValueError(f"Pinned upstream source hash mismatch: {name}")
        edited = transform(source)
        compile(edited, name, "exec")
        patch_chunks.append("".join(difflib.unified_diff(
            source.splitlines(keepends=True), edited.splitlines(keepends=True),
            fromfile="a/" + name, tofile="b/" + name)))
        files[name] = {"upstream_sha256_lf": original_sha,
                       "patched_sha256_lf": hashlib.sha256(edited.encode()).hexdigest()}
        pending.append((root / name, edited))
    patch = "".join(patch_chunks)
    out.mkdir(parents=True, exist_ok=True)
    write_lf(out / "0001-orin-fixed-prompt-suffix-camera-mask.patch", patch)
    result = {"upstream_commit": PIN, "files": files,
              "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}
    write_lf(manifest_path, json.dumps(result, indent=2) + "\n")
    if apply:
        for target, edited in pending:
            write_lf(target, edited)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flashrt-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    options = parser.parse_args()
    print(json.dumps(generate(options.flashrt_root.resolve(), apply=options.apply), indent=2))
