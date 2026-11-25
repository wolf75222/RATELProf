"""Compile a generated C wrapper whose SDK also declares a C++ overload."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


GENERATOR = Path(__file__).resolve().parents[2] / "src" / "cuda" / "generate_host_wrappers.py"


@unittest.skipUnless(shutil.which("clang++"), "Clang is required to generate host wrappers")
class HostWrappersTest(unittest.TestCase):
    def test_c_signature_with_cpp_overload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sdk.h").write_text(
                'extern "C" int cublasExample(int count) noexcept;\n'
                'inline int cublasExample(float count) { return cublasExample(int(count)); }\n'
            )
            output = root / "wrappers.inc"
            subprocess.run(
                [sys.executable, str(GENERATOR), "--clang", "clang++", "--domain", "Cublas",
                 "--prefix", "cublas", "--header", "sdk.h", "--include", directory,
                 "--output", str(output)],
                check=True, capture_output=True,
            )
            self.assertIn('"count", "int"', output.read_text())
            (root / "wrappers.cpp").write_text(
                '#include "sdk.h"\n'
                '#include "argument_values.h"\n'
                'enum HostDomain { Cublas };\n'
                'template<class T> T resolve(const char *symbol);\n'
                'template<class Call, class Read> auto traced(HostDomain, const char *, const void *,\n'
                '                                             Call call, Read read) { read(); return call(); }\n'
                '#include "wrappers.inc"\n'
            )
            subprocess.run(
                ["clang++", "-std=c++17", "-fsyntax-only", "-I", str(GENERATOR.parent),
                 str(root / "wrappers.cpp")],
                check=True, capture_output=True,
            )


if __name__ == "__main__":
    unittest.main()
