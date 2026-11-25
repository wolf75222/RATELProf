"""Exercise the pre-release 1.1.1 reader without CUDA or ROCm."""
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LUA = shutil.which("luajit") or shutil.which("lua")


def uint(value):
    if value < 128:
        return bytes([value])
    for tag, code in enumerate("BHIQ", 0xCC):
        if value < 1 << (8 * struct.calcsize(code)):
            return bytes([tag]) + struct.pack("<" + code, value)
    raise ValueError(value)


def numbers(*values):
    return b"".join(map(uint, values))


def string(value):
    payload = value.encode()
    if len(payload) < 32:
        return bytes([0xA0 + len(payload)]) + payload
    return b"\xda" + struct.pack("<H", len(payload)) + payload


def references(items):
    return b"".join(items) + struct.pack("<Q", len(items))


def event(payload):
    size = uint(len(payload))
    # Size bytes precede the type tag and retain their little-endian order.
    return payload + size[1:] + size[:1] + b"\xc1"


def fixture():
    strings = ["hipFake", "char *", "message", "fixture", "entry", "fixture.c", "Memset", "gpuFake"]
    globals_ = (b"\xff" + string("host") + numbers(1, 4, 0, 1700000000000000000, 10, 900, 1000, 0, 1)
                + string("fixture") + numbers(1, 360, 1))
    location = references([numbers(0x400400, 0x400000, 3, 4, 5, 20)])
    api = references([numbers(0, 1, 1, 2)])
    kernel = references([numbers(7, 1234, 48, 16)])
    hip = event(numbers(1, 40, 15, 0, 0) + string("x" * 319))
    fill = event(numbers(2, 90, 60) + b"\xc6" + numbers(1, 6, 1, 22, 128, 80, 64))
    kern = event(numbers(3, 180, 75, 0, 23, 150) + b"\x93" + numbers(64, 1, 1)
                 + b"\x93" + numbers(256, 1, 1, 43981))
    offsets = numbers(3, 1, 10, 1, 11, 1, 4, 0, 1, 0,
                      1, 360, 1, 42, 2, 10, len(hip), 1, 1,
                      11, len(hip) + len(fill), 1, 2, (1 << 4) | (1 << 10) | (1 << 11))
    sections = [globals_, references(list(map(string, strings))), location, b"\0" * 4, api,
                kernel, references([numbers(1, 0, len(hip) - 1)]), offsets, hip + fill + kern]
    header = b"RATELProf\0\x01\x01\x01"
    offset = len(header) + 16 * len(sections)
    for section in sections:
        header += struct.pack("<QQ", offset, len(section))
        offset += len(section)
    return header + b"".join(sections)


@unittest.skipUnless(LUA, "LuaJIT or Lua required")
class SectionedReportTest(unittest.TestCase):
    def run_decoder(self, data, lua):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "profile.rprof-rep"
            report.write_bytes(data)
            script = Path(tmp) / "decode.lua"
            script.write_text(
                'package.path = "bin/lua/?.lua;" .. package.path\n'
                'local Reader = require("utils.SectionedReport")\n'
                'local file = assert(io.open(arg[1], "rb"))\n'
                'local payload = file:read("*a")\n'
                'file:close()\n' + lua
            )
            subprocess.run([LUA, str(script), str(report)], cwd=ROOT, check=True)

    def test_import_preserves_large_api_event_and_gpu_events(self):
        self.run_decoder(fixture(), """
            local report = Reader.decode(payload)
            local t = report.trace_events
            local hip = t.RATELPROF_DOMAIN_HIP[1]
            assert(report.mpi_rank == -1 and report.hostname == "host")
            assert(report.node_id[360] == 1 and report.lifecycle[3] == 1000)
            assert(hip.args.message.value == string.rep("x", 319))
            assert(report.locations[hip.return_address].sline == 20)
            assert(t.RATELPROF_DOMAIN_MEMORY_FILL[2].corr_id == 1)
            assert(t.RATELPROF_DOMAIN_MEMORY_FILL[2].args.queue_id == 42)
            assert(t.RATELPROF_DOMAIN_KERNEL[3].args.wrg[1] == 64)
            assert(t.RATELPROF_DOMAIN_KERNEL[3].args.private_segment_size == 16)
        """)

    def test_rejects_bad_ranges_versions_and_event_markers(self):
        self.run_decoder(fixture(), """
            local function rejects(contents, reason)
                local ok, err = pcall(Reader.decode, contents)
                assert(not ok and err:find(reason, 1, true), tostring(err))
            end
            rejects(payload:sub(1, -2), "invalid EVENTS range")
            rejects(payload:sub(1, -2) .. "\\0", "missing event marker")
            rejects(payload:sub(1, 12) .. "\\2" .. payload:sub(14), "unsupported sectioned report version")
        """)


if __name__ == "__main__":
    unittest.main()
