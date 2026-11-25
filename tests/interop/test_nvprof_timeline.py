import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[2] / "share" / "modules" / "metrics" / "nvprof_timeline.py"
spec = importlib.util.spec_from_file_location("nvprof_timeline", MODULE)
timeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline)

GPU = """==42== Profiling result:
Start,Duration,Grid Size,Block Size,Regs*,SSMem*,DSMem*,Size,Device,Context,Stream,Name,Correlation_ID
s,us,,,,KB,B,KB,,,,,
0.005,1.25,(2 3 1),(32 1 1),17,1.5,256,,Tesla K20c (0),1,7,"step(A*, B*) [11]",11
0.004,0.5,,,,,,2,Tesla K20c (0),1,7,[CUDA memcpy HtoD],10
0.006,0.5,,,,,,1,Tesla K20c (0),1,7,[CUDA memset],12
"""
API = """Start,Duration,Name
ms,us,
3.5,20,cudaMemcpy [10]
4.5,15,"cuLaunchKernel (step(A*, B*) [11])"
"""


class NvprofTimelineTest(unittest.TestCase):
    def test_trace_preserves_units_devices_and_api_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu, api = root / "gpu.csv", root / "api.csv"
            gpu.write_text(GPU)
            api.write_text(API)
            report = timeline.convert(gpu, api, "sm_35")
        data = report["trace_events"]
        kernel = data["RATELPROF_DOMAIN_KERNEL"][3]
        self.assertEqual(kernel["name"], "step(A*, B*)")
        self.assertEqual(kernel["corr_id"], 2)
        self.assertEqual(kernel["start"], 5_000_000)
        self.assertEqual(kernel["dur"], 1250)
        self.assertEqual(kernel["args"]["grd"], [2, 3, 1])
        self.assertEqual(kernel["args"]["group_segment_size"], 1792)
        self.assertEqual(data["RATELPROF_DOMAIN_CUDA_DRIVER"][2]["name"], "cuLaunchKernel")
        copy = data["RATELPROF_DOMAIN_COPY"][4]
        self.assertEqual(copy["corr_id"], 1)
        self.assertEqual(copy["args"]["src_agent"], timeline.HOST)
        self.assertEqual(copy["args"]["size"], 2048)
        self.assertEqual(data["RATELPROF_DOMAIN_CUDA_MEMSET"][5]["args"]["size"], 1024)
        self.assertEqual(report["gpu_props"][1]["arch"], "sm_35")

    def test_invalid_or_missing_units_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            gpu = Path(directory) / "gpu.csv"
            for source in (GPU.replace("s,us,", "s,bad,"), GPU.replace("s,us,,,,KB,B,KB,,,,,\n", "")):
                gpu.write_text(source)
                with self.assertRaises(ValueError):
                    timeline.convert(gpu, None, "sm_35")


if __name__ == "__main__":
    unittest.main()
