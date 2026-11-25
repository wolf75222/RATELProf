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


class MetricsTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.work = Path(work.name)
        self.ncu = self.work / "mock ncu"
        self.ncu.write_text(NCU_FIXTURE, encoding="utf-8")
        self.ncu.chmod(0o700)

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


if __name__ == "__main__":
    unittest.main()
