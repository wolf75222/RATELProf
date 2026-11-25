# RATELProf

## NVIDIA CUDA backend

This port writes RATELProf `master`-format reports (`8e71397`) with CUDA
domains. It also reads AMD sectioned `.rprof-rep` reports (version 1.1.1)
from `pre-release` through the export, analysis and viewer commands.

Build the CUDA backend on Linux with CMake 3.18+, Python 3.8+, LuaJIT or Lua
5.1, and the CUDA 12.9, 13.0 or 13.1 toolkit, including CUPTI and the CUDA
binary utilities.
Use a separate install prefix for each backend.

```sh
cmake -S . -B build-cuda -DRATELPROF_BACKEND=cuda \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local/ratelprof-cuda"
cmake --build build-cuda --parallel
cmake --install build-cuda
export PATH="$HOME/.local/ratelprof-cuda/bin:$PATH"
```

If several toolkits are installed, pass `-DCUDAToolkit_ROOT=/path/to/cuda` to
CMake and put that toolkit's `bin` directory on `PATH` for inspection. The
default build remains ROCm. For full host signature generation, put Clang on
`PATH` before configuring. CMake discovers optional MPI, OpenMP, cuBLAS, and
NCCL headers. Set `MPI_C_COMPILER`, `RATELPROF_OMPT_INCLUDE_DIR`, or `NCCL_ROOT`
if discovery needs a hint. MPI tracing uses the same MPI implementation and
ABI as the application.

```sh
ratelprof profile --trace gpu,cuda-runtime,cuda-driver,nvtx -o run.rprof-rep ./application -- --app-option
ratelprof stats -r gpu_kern_sum,cuda_runtime_sum,cuda_sync_sum run.rprof-rep
ratelprof breakdown -r gpu_breakdown run.rprof-rep
ratelprof summarize run.rprof-rep
ratelprof analyze run.rprof-rep
ratelprof visualize run.rprof-rep
ratelprof export --type json run.rprof-rep
ratelprof export --type arg-info run.rprof-rep
ratelprof inspect --info gpu
ratelprof inspect --linked --archs sm_90 --save-json --sass --sass-summary ./application
```

- `profile` records CUDA Runtime and Driver calls, concurrent kernels, memory
  copies, peer copies, memsets, synchronization and NVTX ranges and marks.
  Kernel launches retain grid dimensions in blocks, block dimensions in
  threads, registers per thread and shared memory. CUDA synchronization events
  include the recorded stream and the waiting stream when CUPTI supplies their
  matching IDs. CUDA 13 can emit device timestamps for event markers.
- The whitelist and blacklist use the existing format. For example,
  `CUDA_RUNTIME: cudaMalloc cudaMemcpyAsync` selects those Runtime API calls.
  `--api-args basic` (default) traces the curated CUDA API set, `--api-args all`
  traces the enabled Runtime and Driver callback domains and captures the
  typed parameters available in the installed CUPTI headers, and
  `--api-args none` keeps the curated trace with return codes only. Typed values
  include scalars and common CUDA geometry structures. Pointers are recorded
  as addresses, and unsupported by-value structures carry their size.
- `inspect` reads kernel resources with `cuobjdump` and uses `nvdisasm` for
  SASS when available. `cuobjdump --dump-sass` is the fallback. The
  `RATELPROF_NVDISASM` environment variable can point to a compatible disassembler.
  `--sass-summary` counts static instructions by kernel and opcode in a
  `.sass.json` file. `--linked` searches recursive ELF dependencies without
  running the application. Use `--archs sm_80,sm_90` to restrict the images.
  `profile --dump-cubins DIR` also captures modules loaded into CUDA, including
  code generated from PTX at runtime. Inspect accepts executables, shared
  libraries, and standalone cubins. GPU queries follow CUDA device visibility.
- `--trace mpi` records the C MPI interface. With
  `--prefix='mpirun -np 2'`, each rank writes its own `rank_<n>_*.rprof-rep`.
  Pass those files together to the report commands. `--trace omp` captures
  OpenMP API calls and the OMPT callbacks the runtime supports, including target
  regions, memory transfers, submits, mappings, and device events.
- `--trace cublas,nccl` records cuBLAS, cuBLASLt, and NCCL calls. With Clang and
  SDK headers, CMake generates interposition for eligible C declarations in
  these libraries, MPI, and OpenMP. `--api-args all` adds their signature
  arguments. Inline, variadic, and unexported calls are outside that
  interposition. A hand-written baseline is available without Clang or with
  `-DRATELPROF_CUDA_FULL_HOST_WRAPPERS=OFF`.
- `--plugin stdout` emits text traces. `--plugin ./my-plugin.so` runs a custom
  output hook after the report is written. Implement the C ABI in the installed
  `include/ratelprof_cuda_plugin.h` header.
- `breakdown` reports active time by rank and by PID or GPU. It uses the
  union of observed event intervals to account for overlapping work and can
  export the tables as CSV or TSV.

The collector writes the report at normal process exit. CUDA uses event and
stream synchronization, while HSA barrier AND/OR packets remain specific to
the AMD backend. Existing ROCm live plugins keep their AMD ABI.

### Hardware counters and PC sampling

`metrics` uses Nsight Compute CLI for hardware counters, including DRAM and
cache traffic, achieved occupancy, and warp stalls. `--source` also exports
the source/SASS page when the report contains it. Keep the `.ncu-rep` file to
re-export it without running the application again.

```sh
ratelprof metrics -o counters --section SpeedOfLight -- ./application
ratelprof metrics --import existing.ncu-rep -o counters --source
ratelprof sample --period 10 --dump-cubins cubins -o pc.json ./application
ratelprof inspect --pc-samples pc.json -o kernel.csv cubins/module.cubin
```

Compile with `-lineinfo` to make file and line correlation available when the
cubin includes this information. `sample` uses CUPTI to collect instruction
addresses and warp stall reasons in a dedicated run. `inspect --pc-samples`
uses `nvdisasm` and joins by cubin CRC, kernel symbol, and instruction offset.
It writes per-instruction sample counts into the `.sass.json` output for
matched cubins. Substitute the actual cubin file saved in `cubins/`.

Live hardware counters and PC sampling require performance counter access
from the NVIDIA driver. Tool availability and supported metrics depend on
the GPU and toolkit. GH200 Hopper is the validated hardware target. On ROMEO,
the driver currently blocks live performance counter and PC sampling access.
Offline import of NVIDIA Nsight Compute sample measurements works there.

### RATELProf-Visualize WebGL viewer

The built-in HTML viewer is available with `ratelprof visualize`. To generate
the layout expected by the separate RATELProf-Visualize project, use Python
3.9 or newer and pass its checkout to `--viewer`.

```sh
git clone https://github.com/Kassouley/RATELProf-Visualize.git
ratelprof visualize --viewer ./RATELProf-Visualize -o webgl-report run.rprof-rep
python3 -m http.server --directory webgl-report 8000
```

Open `http://localhost:8000`. The export includes the Summary, Timeline,
Breakdown, Statistics and Analyze views. Pass multiple rank reports to the
same command to use the rank selector. The viewer loads deck.gl from its
configured CDN and requires WebGL in the browser. The export was tested with
RATELProf-Visualize revision `5a2d509`.

To display the measurements in the Statistics panel, copy the sidecar files
from the profiling host along with the trace report and add them to the export.
The tables retain the identifiers of their own profiling runs.

```sh
ratelprof visualize --viewer ./RATELProf-Visualize -o webgl-report \
    --metrics-json counters.metrics.json --source-csv counters.source.csv \
    --pc-samples pc.json run.rprof-rep
```

### CUDA smoke test

```sh
nvcc -arch=sm_90 tests/cuda/smoke.cu -ldl -o smoke
ratelprof profile -t all -o smoke.rprof-rep ./smoke -- "argument with 'quotes' and spaces"
ratelprof export --type json smoke.rprof-rep
python3 tests/cuda/verify.py smoke.json
```

Set the architecture to match the test GPU. The tests also contain separate
fixtures for peer copies, MPI and OpenMP target offload.

RATELProf is a powerful and lightweight profiler designed specifically for AMD GPU applications. It enables detailed profiling by tracing GPU activities, and runtime API calls from HSA, HIP, and OpenMP. This tool is ideal for developers and researchers aiming to optimize GPU-accelerated applications.

##  Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Comparison with ROCprof v3](#comparison-with-rocprof-v3)
- [Future Work and Improvement](#future-work-and-improvement)
- [Project Structure](#project-structure)
- [Contribution](#contribution)

---

##  Features

RATELProf provides a comprehensive suite of tools to profile, analyze, and optimize AMD GPU applications. Key features include:

#### **GPU Activity Profiling**
- Monitor kernel dispatches, barriers and memory transfers to identify bottlenecks and optimize GPU performance.

#### **Runtime API Tracing**
- Trace calls from HSA, HIP, MPI, and OpenMP runtimes, offering deep insights into application behavior.

#### **Commands**
RATELProf includes four core commands to streamline your profiling workflow:

1. **`profile`**  
   - Profiles the application and generates a detailed .rprof-rep report.  
   - This command captures kernel execution, memory transfers, and runtime API details.

2. **`stats`**  
   - Computes statistical summaries from a .rprof-rep report created by the `profile` command.  
   - Output includes detailed metrics similar to the `stat` command from NVIDIA's Nsight Systems.

3. **`analyze`**  
   - Analyze profiling results from a .rprof-rep report created by the `profile` command.  
   - Output advices to optimize your CPU/GPU code.

4. **`summarize`** 
   - Output global metrics and plot to get insight of your application.

5. **`visualize`**  
   - Generates an interactive HTML timeline report for the .rprof-rep profile report created by the `profile` command.  
   - Perfect for visualizing application details.

6. **`inspect`**  
   - Inspects the application binary and outputs a CSV/JSON report containing detailed kernel information.  
   - Use this command to analyze static kernel properties.

7. **`export`**  
   - Export the .rprof-rep report to another type of report (json, arg-info, ...).

---

## **Getting Started**

Installing RATELProf is simple and requires running the provided `set_install.lua` script.

### Prerequisites

Ensure you have the following installed on your system before proceeding:
- **CMake** (version 3.10 or later)
- **Lua** (version 5.1)
- **AMD ROCm** (download from [ROCm's official site](https://github.com/ROCm/ROCm))

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Kassouley/RATELProf
cd RATELProf
```

2. Run the installation script:

```bash
./sett_install.lua
```
By default, the tool will be installed to $HOME/.local. If you want to install it to a custom directory, specify it in the `sett.config` file:

```bash
./sett_install.lua /path/to/sett.config
```

##  Comparison with ROCprof v3

| Attribute                         | ROCprof v3                                      | RATELProf                                    |
|-----------------------------------|-------------------------------------------------|----------------------------------------------|
| **GPU Architecture Support**      | AMD RDNA, CDNA (ROCm-compatible GPUs)           | AMD RDNA, CDNA (ROCm-compatible GPUs)        |
| **HIP Tracing**                   | ✅                                              | ✅                                           |
| **HSA Tracing**                   | ✅                                              | ✅                                           |
| **rocBLAS Tracing**               | ❌                                              | ❌ But can be easily implemented with GILDA  |
| **RCCL Tracing**                  | ✅                                              | ❌ But can be easily implemented with GILDA  |
| **Marker Tracing**                | ✅ (ROCTx)                                      | ✅ (ROCTx)                                   |
| **OpenMP Routine Tracing**        | ❌                                              | ✅                                           |
| **OpenMP Target RTL Tracing**     | ❌                                              | ✅                                           |
| **OMPT Integration**              | ✅                                              | ✅                                           |
| **MPI Tracing**                   | ❌                                              | ✅                                           |
| **Scratch Memory Tracing**        | ✅                                              | ❌                                           |
| **Memory transfers Profiling**    | ✅                                              | ✅                                           |
| **Kernel dispatch Profiling**     | ✅                                              | ✅                                           |
| **Barrier dispatch Profiling**    | ❌                                              | ✅                                           |
| **Trace filtering**               | ❌                                              | ✅                                           |
| **PC Sampling**                   | ✅ (Beta)                                       | ❌                                           |
| **HW Counter**                    | ✅                                              | ❌ (WIP)                                     |
| **Statistical post processing**   | ✅ but really simple post processing            | ✅                                           |
| **Post processing analysis**      | ❌                                              | ✅                                           |
| **Output Formats**                | CSV, JSON                                       | Binary (rprof-rep), CSV, TSV, JSON, TXT      |
| **Output Size**                   | Large                                           | Small (msgpack binary format)                |
| **Visualization Tools**           | External (Perfetto)                             | Integrated                                   |
| **Ease of Use**                   | Medium (requires scripting for deeper analysis) | Easy, run and play                           |


##  Future Work and Improvement

While the current version provides a functional profiling workflow, there are several areas identified for future enhancement:

- **Hardware Counter Support**: Support for hardware performance counters. These metrics are crucial for low-level performance analysis and are planned to be implemented.

- **Barrier Dispatch Reliability**: Certain applications may encounter issues with barrier dispatch tracking. Investigating edge cases is on the roadmap.

- **Documentation**: A full and detailed documentation is work in progress.

Community feedback and contributions are welcome to help guide and accelerate these improvements.


##  Project Structure

```sh
└── RATELProf/
    ├── CMakeLists.txt
    ├── install.sh
    ├── README.md
    ├── bin/
    │   ├── lua/
    │   ├── ratelprof.sh
    ├── share/
    │   ├── modules/
    │   │      ├── lua/
    │   │      ├── html/
    │   ├── visualize/
    └── src/
        ├── lua/
        ├── tools/
        ├── core/
        ├── ext/
        ├── wrappers/
        ├── common/
        └── plugins/
```

- **`bin/`**  
  Contains executable scripts and lua command scripts:
  - **`lua/`**: Lua scripts used for tooling.
  - **`ratelprof.sh`**: Main shell script to launch RATELProf.

- **`share/`**  
  Contains shared assets used by the tool:
  - **`modules/`**: Modular components of the tool.
    - **`lua/`**: Built-in and user Lua modules for Analyze and Stats scripts.
    - **`html/`**: Minified and unified HTML report from visualize directory.
  - **`visualize/`**: HTML code used for the visualizing report.

- **`src/`**  
  Core source code of RATELProf:
  - **`lua/`**: C Stub for Lua modules.
  - **`tools/`**: Main tool src file.
  - **`core/`**: Core logic of the tool (auto generated by GILDA).
  - **`ext/`**: Extension logic of the tool (source of GPU profiling logic).
  - **`wrappers/`**: API wrappers for hooking into applications or libraries (e.g., HIP, HSA, ...).
  - **`plugins/`**: Built-in plugins for callbacks definition.
  - **`common/`**: Common source code to lua stub libraries and RATELProf

##  Contribution

Contributions to RATELProf are welcome and appreciated! Whether you're fixing bugs, improving documentation, adding new features, or optimizing performance, your input helps make this tool better for everyone.

### Guidelines

- Follow existing code style and structure.
- Keep changes focused and well-documented.
- For major changes, open an issue to discuss first.

Thanks for contributing!
