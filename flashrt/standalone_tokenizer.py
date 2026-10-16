"""OpenPI-compatible no-state PaliGemma tokenization without OpenPI/JAX imports.

The tokenization body mirrors src/openpi/models/tokenizer.py:PaligemmaTokenizer.
State-conditioned prompts are deliberately rejected for the pi05_libero fixture.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path


class NoStatePaligemmaTokenizer:
    def __init__(self, model_path: Path, max_len: int = 200):
        import numpy as np
        import sentencepiece

        self.model_path = Path(model_path).expanduser().resolve()
        self._max_len = int(max_len)
        if self._max_len < 1:
            raise ValueError("max_len must be positive")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"PaliGemma tokenizer model not found: {self.model_path}. "
                "Use --tokenizer, PI05_TOKENIZER_PATH or FLASHRT_PALIGEMMA_TOKENIZER to select the deployed official model.")
        proto = self.model_path.read_bytes()
        self.model_sha256 = hashlib.sha256(proto).hexdigest()
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=proto)
        self.serialized_model_sha256 = hashlib.sha256(self._tokenizer.serialized_model_proto()).hexdigest()
        self.np = np

    def tokenize(self, prompt: str, state=None):
        if state is not None:
            raise ValueError("pi05_libero uses discrete_state_input=False; state prompts are unsupported here")
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # The newline is encoded separately. No lowercasing, Task: prefix,
        # state string, EOS, or joined text/newline encoding is introduced.
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently.")
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len
        return self.np.asarray(tokens), self.np.asarray(mask)

    def provenance(self, prompt: str) -> dict:
        tokens, mask = self.tokenize(prompt)
        return {"implementation": "standalone OpenPI-compatible no-state SentencePiece",
                "model_path": str(self.model_path), "model_sha256": self.model_sha256,
                "serialized_model_sha256": self.serialized_model_sha256,
                "max_len": self._max_len, "state_in_prompt": False,
                "prompt": prompt, "cleaned_prompt": prompt.strip().replace("_", " ").replace("\n", " "),
                "token_ids": tokens.tolist(), "token_mask": mask.tolist(),
                "valid_token_ids": tokens[mask].tolist(), "valid_token_count": int(mask.sum())}
