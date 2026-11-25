from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "share/modules/metrics/metrics.py"
NCU_FIXTURE = """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if "--export" in args:
    assert args[-2:] == ["./application path", "argument with spaces"]
    assert args[args.index("--kernel-name") + 1] == "regex:kernel"
    assert args[args.index("--launch-count") + 1] == "1"
    if os.environ.get("NCU_DENIED"):
        print("==ERROR== ERR_NVGPUCTRPERM: hardware counters require permission on this platform")
        sys.exit(1)
    Path(args[args.index("--export") + 1].replace("%%", "%")).write_bytes(b"native report")
elif args[args.index("--page") + 1] == "source":
    assert args[args.index("--print-source") + 1] == "sass"
    print('"Address","Source"')
    print('"0x10","IADD3 R0, R1, R2"')
else:
    print('"ID","Process ID","Kernel Name","sm__cycles_active.sum","dram__bytes.sum"')
    print('"","","","cycle","byte"')
    print('"0","81","kernel(int*, int)","9007199254740993","n/a"')
"""

# CSV version of the per-invocation metric table documented in section 3.2.4
# of the CUDA Profiler User's Guide. Values and kernel names are fixtures.
NVPROF_CSV = """==81== Profiling result:
"Device","Context","Stream","Kernel","Correlation_ID","ipc","dram_read_throughput"
"Tesla P100 (0)","1","7","kernel(int*, int)","43","0.625000","1.5GB/s"
"Tesla P100 (0)","1","7","kernel(int*, int)","49","N/A","2.5GB/s"
"Tesla P100 (0)","1","7","filtered(int*)","51","N/A","N/A"
"""
NVPROF_FIXTURE = """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

args = sys.argv[1:]
assert "--print-gpu-trace" in args and "--csv" in args
log = Path(args[args.index("--log-file") + 1].replace("%%", "%"))
if "--export-profile" in args:
    assert args[-2:] == ["./application path", "argument with spaces"]
    assert args[args.index("--kernels") + 1] == "kernel"
    assert args.index("--kernels") < args.index("--metrics")
    assert args[args.index("--replay-mode") + 1] == "application"
    assert args[args.index("--metrics") + 1] == "ipc,dram_read_throughput"
    Path(args[args.index("--export-profile") + 1].replace("%%", "%")).write_bytes(b"legacy report")
    if os.environ.get("NVPROF_DENIED"):
        log.write_text("==81== Warning: The user does not have permission to profile on the target device.\\n")
        sys.exit(0)
else:
    assert Path(args[args.index("--import-profile") + 1]).read_bytes() == b"legacy report"
log.write_text(""" + repr(NVPROF_CSV) + ")\n"


class MetricsTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.work = Path(work.name)
        self.ncu = self.work / "mock ncu"
        self.ncu.write_text(NCU_FIXTURE, encoding="utf-8")
        self.ncu.chmod(0o700)
        self.nvprof = self.work / "mock nvprof"
        self.nvprof.write_text(NVPROF_FIXTURE, encoding="utf-8")
        self.nvprof.chmod(0o700)

    def invoke(self, *args, extra_env=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--ncu", str(self.ncu), *args],
            cwd=self.work, env=dict(os.environ, **(extra_env or {})),
            capture_output=True, text=True,
        )

    def collect(self, *args, extra_env=None):
        return self.invoke("--kernel", "kernel", "--launch-count", "1", *args,
                           "--", "./application path", "argument with spaces", extra_env=extra_env)

    def test_collection_and_source_export_preserve_values(self):
        result = self.collect("--source", "--output", "out%1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.work / "out%1.ncu-rep").read_bytes(), b"native report")
        self.assertIn("IADD3", (self.work / "out%1.source.csv").read_text())
        summary = json.loads((self.work / "out%1.metrics.json").read_text())
        self.assertEqual(summary["launch_count"], 1)
        self.assertEqual(summary["measurement"]["replay_mode"], "kernel")
        table, = summary["tables"]
        self.assertEqual(table["units"][-2:], ["cycle", "byte"])
        self.assertEqual(table["rows"][0][-3:], ["kernel(int*, int)", "9007199254740993", "n/a"])

    def test_permission_error_leaves_no_partial_report(self):
        result = self.collect(extra_env={"NCU_DENIED": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERR_NVGPUCTRPERM", result.stderr)
        self.assertEqual(list(self.work.glob("rprof-metrics.*")), [])

    def test_import_preserves_report_and_rejects_collision(self):
        native = self.work / "existing.ncu-rep"
        native.write_bytes(b"original")
        result = self.invoke("--import", native.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(native.read_bytes(), b"original")
        summary = json.loads((self.work / "existing.metrics.json").read_text())
        self.assertEqual(summary["measurement"]["mode"], "import")
        self.assertIsNone(summary["measurement"]["replay_mode"])
        repeated = self.invoke("--import", native.name)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("already exists", repeated.stderr)
        self.assertEqual(native.read_bytes(), b"original")

    def collect_legacy(self, *args, extra_env=None):
        return self.invoke("--arch", "sm_60", "--nvprof", str(self.nvprof),
                           "--kernel", "kernel", "--metrics", "ipc,dram_read_throughput",
                           "--replay-mode", "application", *args, "--", "./application path",
                           "argument with spaces", extra_env=extra_env)

    def test_legacy_architecture_selects_nvprof_and_reimports_native_profile(self):
        result = self.collect_legacy("-o", "dir%1/pass")
        self.assertEqual(result.returncode, 0, result.stderr)
        native = self.work / "dir%1/pass.nvprof"
        self.assertEqual(native.read_bytes(), b"legacy report")
        summary = json.loads((native.parent / "pass.metrics.json").read_text())
        self.assertEqual(summary["schema"], "ratelprof.metrics.v1")
        self.assertEqual(summary["profiler"], "nvprof")
        self.assertEqual(summary["launch_count"], 2)
        self.assertEqual(summary["measurement"]["replay_mode"], "application")
        table, = summary["tables"]
        self.assertIsNone(table["units"])
        self.assertEqual(table["rows"][0][-4:], ["kernel(int*, int)", "43", "0.625000", "1.5GB/s"])
        self.assertEqual(table["rows"][1][-2:], ["N/A", "2.5GB/s"])

        imported = self.invoke("--nvprof", str(self.nvprof), "--import", str(native),
                               "--metrics", "ipc", "-o", "imported")
        self.assertEqual(imported.returncode, 0, imported.stderr)
        data = json.loads((self.work / "imported.metrics.json").read_text())
        self.assertEqual(data["launch_count"], 1)
        self.assertEqual(data["measurement"]["mode"], "import")
        self.assertEqual(data["tables"][0]["columns"][-1], "ipc")

    def test_legacy_csv_import_requires_no_vendor_executable(self):
        native = self.work / "recorded.csv"
        native.write_text(NVPROF_CSV, encoding="utf-8")
        result = self.invoke("--nvprof", "absent", "--import", native.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads((self.work / "recorded.metrics.json").read_text())
        self.assertEqual(data["report"], native.name)
        self.assertEqual(data["launch_count"], 2)
        self.assertEqual(native.read_text(), NVPROF_CSV)

        spec = spec_from_file_location("measurements", SCRIPT.parents[1] / "webgl/measurements.py")
        measurements = module_from_spec(spec)
        spec.loader.exec_module(measurements)
        stats = measurements.metrics_table(self.work / "recorded.metrics.json")["cuda_metrics"]
        self.assertEqual(stats["NAME"], "nvprof")
        self.assertIn("Imported from an existing nvprof recording.", stats["MSG"])
        header, *rows = stats["DATA"]
        self.assertEqual(header[-1], "Correlation ID")
        self.assertEqual({row[1] for row in rows}, {"ipc", "dram_read_throughput"})
        self.assertEqual({row[-1] for row in rows}, {"43", "49"})

    def test_legacy_permission_error_leaves_no_partial_report(self):
        result = self.collect_legacy(extra_env={"NVPROF_DENIED": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERR_NVGPUCTRPERM", result.stderr)
        self.assertEqual(list(self.work.glob("rprof-metrics.*")), [])

    def test_legacy_rejects_ncu_only_flags_before_collection(self):
        result = self.collect_legacy("--source")
        self.assertEqual(result.returncode, 2)
        self.assertIn("require Nsight Compute", result.stderr)


if __name__ == "__main__":
    unittest.main()
