"""Check the exported smoke report on a machine with CUDA."""

import json
import sys
from pathlib import Path


def values(table):
    return list(table.values()) if isinstance(table, dict) else table


path = Path(sys.argv[1])
data = json.loads(path.read_text())
ranks = values(data["trace_events"])
assert len(ranks) == 1
trace = ranks[0]


def events(domain):
    return values(trace["RATELPROF_DOMAIN_" + domain])


kernels = events("KERNEL")
assert len(kernels) == 10
assert all(event["args"]["kernel_name"] == "_Z9incrementPi" for event in kernels)
assert all(event["args"]["grd"] == [1, 1, 1] for event in kernels)
assert all(event["args"]["wrg"] == [32, 1, 1] for event in kernels)
assert all(event["args"]["registers_per_thread"] > 0 for event in kernels)
assert all(event["corr_id"] > 0 and event["dur"] > 0 for event in kernels)

copy, = events("COPY")
assert copy["args"]["size"] == 128
assert copy["args"]["src_type"] == 1 and copy["args"]["dst_type"] == 0
memset, = events("CUDA_MEMSET")
assert memset["args"]["size"] == 128
assert any("Event synchronize" == event["name"] for event in events("CUDA_SYNC"))

markers = {event["name"]: event["dur"] for event in events("NVTX")}
assert markers["Smoke NVTX range"] > 0 and markers["Smoke NVTX mark"] == 0
runtime = events("CUDA_RUNTIME")
assert sum(event["name"].startswith("cudaLaunchKernel") for event in runtime) == 10
assert any(event["name"].startswith("cudaMalloc") and event["args"]["bytes"]["value"] == 128
           for event in runtime)
assert events("CUDA_DRIVER")
assert data["main_data"] == [0, "./smoke", "argument with 'quotes' and spaces"]
assert any(device["arch"].startswith("sm_") and device["wavefront_size"] == 32
           for device in values(data["gpu_props"]))
print("Verified CUDA activities, CUDA arguments, NVTX, and GPU metadata")
