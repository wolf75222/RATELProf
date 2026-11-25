"""Build and load a C plugin without CUDA to test the versioned callback ABI."""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


CUDA_SOURCE = Path(__file__).resolve().parents[2] / "src" / "cuda"
CC = os.environ.get("CC") or shutil.which("cc")
CXX = os.environ.get("CXX") or shutil.which("c++")

PLUGIN = r"""
#include "ratelprof_cuda_plugin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int append(const char *line)
{
    FILE *log = fopen(getenv("RATELPROF_TEST_PLUGIN_LOG"), "a");
    if (!log) return 1;
    int failed = fputs(line, log) < 0;
    return fclose(log) != 0 || failed;
}

__attribute__((constructor)) static void opened(void)
{
    append("open\n");
}

__attribute__((destructor)) static void closed(void)
{
    append("close\n");
}

int ratelprof_cuda_plugin_process_activity_v1(const ratelprof_cuda_activity_v1 *event)
{
    if (event->abi_version != 1 || event->struct_size != sizeof(*event))
        return 2;
    if (event->mpi_rank != 4 || event->process_id != 99 ||
        strcmp(event->domain, "MPI") != 0 || strcmp(event->name, "MPI_Test") != 0)
        return 3;
    if (event->argument_count != 2 ||
        event->arguments[0].value_kind != RATELPROF_CUDA_VALUE_DIM3 ||
        event->arguments[0].value.dim3[2] != 64 ||
        event->arguments[1].value_kind != RATELPROF_CUDA_VALUE_STRING ||
        strcmp(event->arguments[1].value.string, "copied") != 0)
        return 4;
    if (event->sequence == 1)
        return append("first\n");
    if (event->sequence == 2)
        return append("second\n");
    return 5;
}

int ratelprof_cuda_plugin_process_report_v1(const ratelprof_cuda_report_v1 *report)
{
    if (report->abi_version != 1 || report->struct_size != sizeof(*report) ||
        report->mpi_rank != 4 || report->process_id != 99 || report->exit_code != 6)
        return 6;
    return append("report\n");
}
"""

RUNNER = r"""
#include "output_plugin.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

static std::atomic<unsigned> flushes{0};

static bool flush()
{
    ++flushes;
    return true;
}

static bool first_callback_finished()
{
    std::ifstream file(std::getenv("RATELPROF_TEST_PLUGIN_LOG"));
    std::string content{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    return content.find("first\n") != std::string::npos;
}

int main(int argc, char **argv)
{
    if (argc != 2)
        return 1;
    if (std::string(argv[1]) == "report") {
        std::ifstream before(std::getenv("RATELPROF_TEST_PLUGIN_LOG"));
        if (before.good())
            return 2;
        ratelprof_cuda_run_output_plugin("published.rprof-rep", 4, 99, 6);
        return 0;
    }
    CudaActivityPlugin plugin;
    plugin.open();
    if (!plugin.captures_activity())
        return 3;
    plugin.set_periodic_flush(flush);

    PluginActivity packet;
    packet.activity.abi_version = 1;
    packet.activity.struct_size = sizeof(packet.activity);
    packet.activity.mpi_rank = 4;
    packet.activity.process_id = 99;
    packet.activity.domain = "MPI";
    packet.activity.sequence = 1;
    packet.name = "MPI_Test";
    packet.arguments.push_back({"dimensions", "dim3", std::array<uint64_t, 3>{1, 2, 64}});
    packet.arguments.push_back({"label", "char *", std::string("copied")});
    plugin.enqueue(packet);
    packet.activity.sequence = 2;
    plugin.enqueue(std::move(packet));

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while ((!first_callback_finished() || !flushes) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!first_callback_finished() || !flushes)
        return 4;

    plugin.drain();
    plugin.process_report("published.rprof-rep", 4, 99, 6);
    plugin.close();
    return 0;
}
"""


@unittest.skipUnless(CC and CXX, "C and C++ compilers required")
class LivePluginTest(unittest.TestCase):
    def test_live_worker_and_legacy_report_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plugin = directory / "plugin.c"
            plugin.write_text(PLUGIN)
            runner = directory / "runner.cpp"
            runner.write_text(RUNNER)

            library = directory / "plugin.so"
            subprocess.run([CC, "-std=c11", "-shared", "-fPIC", "-I", CUDA_SOURCE, plugin,
                            "-o", library], check=True)
            binary = directory / "runner"
            subprocess.run([CXX, "-std=c++17", "-pthread", "-I", CUDA_SOURCE, runner,
                            CUDA_SOURCE / "output_plugin.cpp", "-ldl", "-o", binary], check=True)

            for mode, expected in [("live", "open\nfirst\nsecond\nreport\nclose\n"),
                                   ("report", "open\nreport\nclose\n")]:
                with self.subTest(mode=mode):
                    log = directory / mode
                    env = dict(os.environ, RATELPROF_CUDA_PLUGIN=str(library),
                               RATELPROF_TEST_PLUGIN_LOG=str(log))
                    result = subprocess.run([binary, mode], env=env, check=True, timeout=10,
                                            capture_output=True, text=True)
                    self.assertEqual(result.stderr, "")
                    self.assertEqual(log.read_text(), expected)


if __name__ == "__main__":
    unittest.main()
