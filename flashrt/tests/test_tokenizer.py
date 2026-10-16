"""Compare standalone tokenization with the official OpenPI method body."""

from __future__ import annotations

import ast
import __future__
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parents[1]
sys.path.insert(0, str(BENCH))
from standalone_tokenizer import NoStatePaligemmaTokenizer


def extracted_functions(path, names, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name is not None:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError(f"Official source methods not found: {path}")
    module = ast.Module(body=selected, type_ignores=[])
    # Annotation evaluation is unrelated to the method body under test and
    # would otherwise require importing all of OpenPI/JAX on CPU Python 3.9.
    code = compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag)
    namespace = {"np": np, "logging": logging}
    exec(code, namespace)
    return namespace


class FakeSentencePiece:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_bos=False):
        self.calls.append((text, add_bos))
        return ([2] if add_bos else []) + [ord(character) + 10 for character in text]


class TokenizerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "src/openpi/models/tokenizer.py"
        cls.official_method = staticmethod(extracted_functions(path, {"tokenize"}, "PaligemmaTokenizer")["tokenize"])

    def compare(self, prompt, capacity, standalone):
        reference = SimpleNamespace(_tokenizer=standalone._tokenizer, _max_len=capacity)
        with patch("logging.warning"):
            actual_tokens, actual_mask = standalone.tokenize(prompt)
            expected_tokens, expected_mask = self.official_method(reference, prompt, state=None)
        np.testing.assert_array_equal(actual_tokens, expected_tokens)
        np.testing.assert_array_equal(actual_mask, expected_mask)
        self.assertEqual(actual_tokens.dtype, expected_tokens.dtype)
        self.assertEqual(actual_mask.dtype, expected_mask.dtype)
        self.assertEqual(actual_tokens.shape, (capacity,))

    def test_cleaning_padding_truncation_and_separate_newline_encoding(self):
        prompts = ("pick up the black bowl", "  Pick_up\nThe_bowl  ", "", "中文_test\n", "word_" * 300)
        for capacity in (1, 48, 200):
            for prompt in prompts:
                tokenizer = object.__new__(NoStatePaligemmaTokenizer)
                tokenizer._max_len = capacity
                tokenizer._tokenizer = FakeSentencePiece()
                tokenizer.np = np
                self.compare(prompt, capacity, tokenizer)
                cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
                self.assertEqual(tokenizer._tokenizer.calls[:2], [(cleaned, True), ("\n", False)])

    def test_real_official_sentencepiece_model_matches_openpi(self):
        local_deps = ROOT / "third_party/FlashRT-research/.test-deps"
        if importlib.util.find_spec("sentencepiece") is None and local_deps.exists():
            sys.path.insert(0, str(local_deps))
        if importlib.util.find_spec("sentencepiece") is None:
            self.skipTest("Install sentencepiece to test the real model")
        model_path = Path(os.environ.get("PI05_TOKENIZER_PATH",
            ROOT / "third_party/FlashRT-research/paligemma_tokenizer.model"))
        if not model_path.is_file():
            self.skipTest("Set PI05_TOKENIZER_PATH to the official model")
        for capacity in (1, 48, 200):
            tokenizer = NoStatePaligemmaTokenizer(model_path, capacity)
            for prompt in ("pick up the black bowl", "  Pick_up\nThe_bowl  ", "", "中文_test\n", "word_" * 300):
                self.compare(prompt, capacity, tokenizer)
        tokenizer = NoStatePaligemmaTokenizer(model_path, 200)
        provenance = tokenizer.provenance("pick up the black bowl")
        self.assertEqual(provenance["model_sha256"], hashlib.sha256(model_path.read_bytes()).hexdigest())
        self.assertEqual(provenance["serialized_model_sha256"],
                         hashlib.sha256(tokenizer._tokenizer.serialized_model_proto()).hexdigest())
        self.assertEqual(len(provenance["token_ids"]), 200)
        self.assertEqual(len(provenance["token_mask"]), 200)
        self.assertEqual(provenance["valid_token_count"], len(provenance["valid_token_ids"]))
        with self.assertRaises(ValueError):
            tokenizer.tokenize("pick up the black bowl", state=np.zeros(8))

    def test_pinned_frontend_cache_bypasses_optional_locator_imports(self):
        source = Path(os.environ.get("FLASHRT_SOURCE", ROOT / "third_party/FlashRT-research"))
        path = source / "flash_rt/frontends/torch/pi05_rtx.py"
        if not path.is_file():
            self.skipTest("Set FLASHRT_SOURCE to test the pinned frontend cache protocol")
        frontend = extracted_functions(path, {"_get_tokenizer", "_prompt_token_ids"})
        tokenizer = object.__new__(NoStatePaligemmaTokenizer)
        tokenizer._max_len = 200
        tokenizer._tokenizer = FakeSentencePiece()
        tokenizer.np = np
        frontend["_TOKENIZERS"] = {("openpi", 200): ("openpi", tokenizer)}
        prompt = "pick up the black bowl"
        expected, mask = tokenizer.tokenize(prompt)
        self.assertEqual(frontend["_prompt_token_ids"](prompt, 200), expected[mask].tolist())
        self.assertIs(frontend["_get_tokenizer"](200)[1], tokenizer)


if __name__ == "__main__":
    unittest.main()
