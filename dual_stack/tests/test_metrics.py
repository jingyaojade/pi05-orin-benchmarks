import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from telemetry import parse_tegrastats
from clients import content_delta
from report import stats


class MetricsTests(unittest.TestCase):
    def test_agx_rails_do_not_count_ddr_or_summation_twice(self):
        line='RAM 12657/62836MB GR3D_FREQ 50%@[816,816] cpu@44.5C gpu@43C tj@45C VDD_GPU_SOC 20000mW/18000mW VDD_CPU_CV 4000mW/3000mW VIN_SYS_5V0 5000mW/5000mW VDDQ_VDD2_1V8AO 800mW/800mW SUMMATION 24000mW/21000mW'
        value=parse_tegrastats(line)
        self.assertEqual(value['main_rails_sum_w'],29)
        self.assertEqual(value['temperatures_c']['gpu'],43)
        self.assertEqual(value['gpu_load_percent'],50)
        self.assertEqual(value['ram_used_mib'],12657)

    def test_missing_main_rail_is_unmeasured(self):
        self.assertIsNone(parse_tegrastats('VDD_GPU_SOC 1000mW/1000mW')['main_rails_sum_w'])

    def test_ttft_does_not_count_empty_role_or_reasoning(self):
        self.assertEqual(content_delta({'choices':[{'delta':{'role':'assistant','content':''}}]}),'')
        self.assertEqual(content_delta({'choices':[{'delta':{'reasoning_content':'thinking'}}]}),'')
        self.assertEqual(content_delta({'choices':[{'delta':{'content':'红色'}}]}),'红色')

    def test_percentiles_and_missing_values(self):
        self.assertEqual(stats([1,None,2,3])['mean'],2)
        self.assertAlmostEqual(stats([1,2,3])['p95'],2.9)
        self.assertIsNone(stats([])['mean'])


if __name__=='__main__':unittest.main()
