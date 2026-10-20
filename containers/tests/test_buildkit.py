"""CPU checks for portable packaging and reporting, not GPU acceptance tests."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CONTAINERS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CONTAINERS))
import assets
from package_buildkit import include
from telemetry_container import kib_fields, number
from benchmark_container import summarize


class BuildKitTests(unittest.TestCase):
    def test_scope_excludes_large_assets_and_credentials(self):
        for name in ['.git/config','containers/local-artifacts/model.gguf','dual_stack/.env',
                     'flashrt/token.key','flashrt/compiled.so','models/model.safetensors',
                     'reports/shared-context-2026-10-20/telemetry.jsonl']:
            self.assertFalse(include(name),name)
        for name in ['containers/Dockerfile.thor','dual_stack/clients.py',
                     'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz']:
            self.assertTrue(include(name),name)

    def test_pins_match_existing_measured_assets(self):
        root=CONTAINERS.parent; pins=json.loads((CONTAINERS/'pins.json').read_text())
        source=json.loads((root/'dual_stack/sources.json').read_text())
        self.assertEqual(source['llama_cpp_commit'],pins['llama_cpp_commit'])
        for entry in source['files']:
            self.assertEqual(entry['sha256'],pins['assets']['Qwen3.5-9B/'+entry['filename']])
        fixture=root/'reports/engine-comparison-2026-10-16/baseline-latency/fixed_inputs.npz'
        self.assertEqual(assets.digest(fixture),pins['fixture_sha256'])

    def test_asset_export_checks_before_copy_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); sources=root/'sources'; pins=root/'pins';pins.mkdir()
            names=['pi05_libero/model.safetensors','pi05_libero/assets/physical-intelligence/libero/norm_stats.json',
                   'Qwen3.5-9B/model.gguf','tokenizer/paligemma_tokenizer.model']
            hashes={}
            for name in names:
                p=sources/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode())
                hashes[name]=assets.digest(p)
            (pins/'pins.json').write_text(json.dumps({'assets':hashes}))
            output=root/'exported'
            argv=['assets.py','export','--pi05',str(sources/'pi05_libero'),'--qwen',str(sources/'Qwen3.5-9B'),
                  '--tokenizer',str(sources/'tokenizer/paligemma_tokenizer.model'),'--output',str(output)]
            with patch.object(assets,'HERE',pins),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                assets.main();self.assertEqual(len(assets.verify(output)),4)
                with self.assertRaises(FileExistsError):assets.main()
                (output/'pi05_libero/model.safetensors').write_bytes(b'changed')
                with self.assertRaises(ValueError):assets.verify(output)
            output2=root/'must-not-exist';argv[-1]=str(output2)
            (sources/'pi05_libero/model.safetensors').write_bytes(b'changed')
            with patch.object(assets,'HERE',pins),patch.object(sys,'argv',argv):
                with self.assertRaises(ValueError):assets.main()
            self.assertFalse(output2.exists())

    def test_missing_sensor_is_not_zero(self):
        self.assertIsNone(number('[N/A]'));self.assertIsNone(number('Not Supported'))
        self.assertEqual(number(' 45.6 '),45.6)
        self.assertEqual(kib_fields('MemAvailable: 4096 kB\nName: test\n')['MemAvailable'],4194304)

    def test_summary_excludes_warmups_and_preserves_missing_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);phase={'name':'test','start_ns':0,'end_ns':1_000_000_000}
            rows=[]
            for warm,ms in [(False,10),(False,20),(True,100)]:
                timings={k:ms for k in ['cpu_preprocess_ms','input_prepare_upload_ms','model_wall_ms',
                                        'model_stream_ms','download_postprocess_ms']}
                rows.append({'phase':'test','kind':'vla','warmup':warm,'api_wall_ms':ms,
                             'response':{'timings':timings}})
            row={'monotonic_ns':1,'host_meminfo_raw':'MemAvailable: 1024 kB','processes':{},'devices':[]}
            (root/'telemetry.jsonl').write_text(json.dumps(row)+'\n')
            summarize(root,rows,[phase],{'backend':'CPU fixture only'})
            result=json.loads((root/'summary.json').read_text())['phases']['test']
            self.assertEqual(result['vla']['n'],2)
            self.assertEqual(result['vla']['model_wall_ms']['mean'],15)
            self.assertEqual(result['vla']['model_wall_ms']['p95'],19.5)
            self.assertIsNone(result['device_metrics']['power.draw']['mean'])
            self.assertIn('N/A',(root/'report_zh.md').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
