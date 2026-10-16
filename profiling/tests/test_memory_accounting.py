"""CPU checks for accounting boundaries; no fake GPU performance data."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_common import MemoryRecorder, cuda_buffer_inventory, parse_kib_fields, parse_nvmap


class AccountingTests(unittest.TestCase):
    def test_nvmap_multiple_clients_pid_suffixes_and_process_name(self):
        table = """CLIENT PROCESS PID SIZE
user python 12 1024K
user python worker 12 2M
user python 123 99K
total 3195K
"""
        result = parse_nvmap(table, 12)
        self.assertEqual(result["self_bytes"], 3 * 1024 * 1024)
        self.assertEqual(len(result["self_clients"]), 2)
        self.assertEqual(parse_nvmap(table, 999)["self_bytes"], 0)
        with self.assertRaises(ValueError):
            parse_nvmap("permission denied", 12)

    def test_proc_kib_units_and_not_line_substrings(self):
        fields = parse_kib_fields("Rss: 100 kB\nPss: 80 kB\nPrivate_Dirty: 2 kB\nVmHWM: 200 kB\nThreads: 4\n")
        self.assertEqual(fields["Rss"], 102400)
        self.assertEqual(fields["Pss"], 81920)
        self.assertEqual(fields["VmHWM"], 204800)
        self.assertNotIn("Threads", fields)

    def test_cuda_inventory_deduplicates_aliases_and_does_not_walk_torch(self):
        class Pointer:
            def __init__(self, value): self.value = value
        class Buffer:
            def __init__(self, pointer, size):
                self.ptr, self.nbytes, self._managed = Pointer(pointer), size, False
        class Frontend:
            pass
        Frontend.__module__ = "flash_rt.frontends.torch.pi05_rtx"
        front = Frontend()
        front.bufs = {"weight": Buffer(10, 100), "weight_alias": Buffer(10, 100), "scratch": Buffer(20, 8)}
        front.self = front
        front.unrelated = object()
        result = cuda_buffer_inventory(front, Buffer)
        self.assertEqual(result["unique_allocation_count"], 2)
        self.assertEqual(result["unique_bytes"], 108)
        weight = next(item for item in result["allocations"] if item["pointer_hex"] == "0xa")
        self.assertEqual(len(weight["aliases"]), 2)

    def test_nested_peak_scope_rejected_without_corrupting_outer(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = MemoryRecorder(directory, disable_nvmap=True)
            with recorder.measure("outer"):
                with self.assertRaisesRegex(RuntimeError, "Nested peak reset forbidden"):
                    with recorder.measure("inner"):
                        pass
            self.assertEqual([row["stage"] for row in recorder.records], ["outer"])
            self.assertIsNone(recorder.active_stage)
            recorder.save({"test": True})
            self.assertTrue((Path(directory) / "samples.csv").is_file())


if __name__ == "__main__":
    unittest.main()
