"""Verify production replay selection and timing boundaries without CUDA."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import LiberoFlashRT
from benchmark import timed_model_graph


class ModelTimingBoundary(unittest.TestCase):
    def test_staging_before_timer_and_gpu_completion_before_wall_stop(self):
        calls = []
        stream = object()
        noise = object()

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self, actual_stream):
                self.assert_stream = actual_stream
                calls.append(self.name)

            def elapsed_time(self, other):
                calls.append("event_elapsed")
                return 2.5

        class Adapter:
            model_stream = stream

            def stage_model_inputs(self, actual_noise):
                if actual_noise is not noise:
                    raise AssertionError("Fixture noise was replaced")
                calls.append("stage_inputs")

            def replay_model_graph(self):
                calls.append("replay_original_graph")

            def download_model_actions(self):
                raise AssertionError("D2H must be outside timed_model_graph")

        ticks = iter((1_000_000, 4_000_000))

        def clock():
            calls.append("clock")
            return next(ticks)

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            synchronize=lambda: calls.append("synchronize")))
        pair = [Event("event_start"), Event("event_end")]
        with patch("benchmark.time.perf_counter_ns", clock):
            result = timed_model_graph(Adapter(), noise, pair, fake_torch)
        self.assertEqual(calls, ["stage_inputs", "synchronize", "clock", "event_start",
            "replay_original_graph", "event_end", "synchronize", "clock", "event_elapsed"])
        self.assertIs(pair[0].assert_stream, stream)
        self.assertIs(pair[1].assert_stream, stream)
        self.assertEqual(result, {"model_wall_ms": 3.0, "model_cuda_stream_elapsed_ms": 2.5})

    def test_adapter_replays_full_graph_on_its_captured_stream(self):
        stream = object()
        calls = []
        original_graph = SimpleNamespace(replay=lambda handle: calls.append(("full_graph", handle)))

        def forbidden(_):
            raise AssertionError("Production timing must not replay decoder-only graph")

        adapter = object.__new__(LiberoFlashRT)
        adapter.pipe = SimpleNamespace(pipeline=SimpleNamespace(
            _graph=original_graph, _graph_stream=stream,
            _decoder_only_graph=SimpleNamespace(replay=forbidden)))
        adapter.replay_model_graph()
        self.assertEqual(calls, [("full_graph", stream)])


if __name__ == "__main__":
    unittest.main()
