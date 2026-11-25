"""Collect or export NVIDIA kernel counters with Nsight Compute or nvprof."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


DEFAULT_SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "WarpStateStats",
)
PERMISSION_ERROR = "ERR_NVGPUCTRPERM"
LEGACY_DEFAULT_METRICS = "achieved_occupancy,ipc"
LEGACY_SUFFIXES = {".csv", ".nvprof", ".prof", ".nvvp"}


class MetricsError(Exception):
    pass


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def nonempty(text):
    if not text.strip():
        raise argparse.ArgumentTypeError("expected a nonempty value")
    return text


def architecture(text):
    match = re.fullmatch(r"sm_([1-9]\d{1,2})[af]?", text)
    if not match:
        raise argparse.ArgumentTypeError("expected a CUDA architecture such as sm_61 or sm_90")
    return int(match[1])


def select_backend(options):
    if options.backend != "auto":
        return options.backend
    if options.report:
        return "nvprof" if options.report.suffix.lower() in LEGACY_SUFFIXES else "ncu"
    if options.arch is not None:
        return "nvprof" if options.arch < 75 else "ncu"
    if shutil.which(options.ncu) is None and shutil.which(options.nvprof):
        return "nvprof"
    return "ncu"


def parse_options(argv):
    parser = argparse.ArgumentParser(
        prog="ratelprof metrics",
        description=__doc__,
        epilog="Collection is a separate run. The profiler may replay kernels or the application.",
    )
    parser.add_argument("-o", "--output", type=Path, help="output prefix")
    parser.add_argument("--ncu", default=os.environ.get("RATELPROF_NCU", "ncu"),
                        help="Nsight Compute CLI executable (default: ncu on PATH)")
    parser.add_argument("--nvprof", default=os.environ.get("RATELPROF_NVPROF", "nvprof"),
                        help="legacy NVIDIA profiler executable (default: nvprof on PATH)")
    parser.add_argument("--backend", choices=("auto", "ncu", "nvprof"), default="auto",
                        help="counter profiler (auto uses the architecture or imported report type)")
    parser.add_argument("--arch", type=architecture, metavar="sm_NN",
                        help="target architecture for backend selection, for example sm_61")
    parser.add_argument("--import", dest="report", type=Path,
                        help="import an .ncu-rep, nvprof profile, or nvprof per-launch trace CSV")
    parser.add_argument("--replay-mode", choices=("kernel", "application"),
                        help="profiler replay mode (default: kernel)")
    parser.add_argument("--kernel", type=nonempty, help="regular expression selecting kernel names")
    parser.add_argument("--launch-count", type=positive_int, help="limit the number of matching launches")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--section", action="append", type=nonempty,
                           help="Nsight Compute section identifier (repeatable)")
    selection.add_argument("--metrics", type=nonempty, help="comma-separated metric names for the selected profiler")
    parser.add_argument("--source", action="store_true",
                        help="also export the SASS-correlated source page as CSV")
    parser.add_argument("application", nargs=argparse.REMAINDER,
                        help="executable and arguments, usually after --")
    options = parser.parse_args(argv)
    if options.application[:1] == ["--"]:
        options.application.pop(0)
    if options.report:
        if options.application or options.section or options.kernel or options.launch_count or options.replay_mode:
            parser.error("--import accepts --metrics and --source without application or collection filters")
        if not options.report.is_file():
            parser.error(f"cannot read the profiler report {options.report}")
    elif not options.application:
        parser.error("provide an application after --, or use --import REPORT")

    if options.metrics:
        metric_names = [name.strip() for name in options.metrics.split(",")]
        if not all(metric_names):
            parser.error("--metrics must contain a comma-separated list of nonempty metric names")
        options.metrics = ",".join(metric_names)
    options.backend = select_backend(options)
    if options.backend == "nvprof":
        if options.section or options.source or options.launch_count:
            parser.error("--section, --source and --launch-count require Nsight Compute. "
                         "Use --metrics and --kernel with nvprof.")
        if options.kernel and ":" in options.kernel:
            parser.error("nvprof reserves ':' for context and stream selectors. "
                         r"Use \x3a to match a literal colon in --kernel.")
    if options.output is None:
        options.output = options.report.with_suffix("") if options.report else Path("rprof-metrics")
    return options


def find_ncu(executable):
    ncu = shutil.which(executable)
    if ncu:
        return str(Path(ncu).absolute())
    raise MetricsError(f"Nsight Compute executable {executable!r} is unavailable. Use --ncu PATH or load its toolkit.")


def find_nvprof(executable):
    nvprof = shutil.which(executable)
    if nvprof:
        return str(Path(nvprof).absolute())
    raise MetricsError(f"nvprof executable {executable!r} is unavailable. "
                       "Use --nvprof PATH from a compatible CUDA toolkit through 12.x, "
                       "or import an existing nvprof CSV.")


def collect(ncu, options, native_report):
    mode = options.replay_mode or "kernel"
    sections = [] if options.metrics else list(options.section or DEFAULT_SECTIONS)
    # ncu expands percent macros in --export paths.
    report_argument = str(native_report).replace("%", "%%")
    command = [ncu, "--config-file", "off", "--export", report_argument, "--replay-mode", mode]
    if options.kernel:
        command += ["--kernel-name", f"regex:{options.kernel}"]
    if options.launch_count:
        command += ["--launch-count", str(options.launch_count)]
    if options.metrics:
        command += ["--metrics", options.metrics]
    for section in sections:
        command += ["--section", section]
    command += options.application

    permission_denied = False
    unsupported_gpu = False
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace") as run:
        for line in run.stdout:
            sys.stderr.write(line)
            permission_denied |= line.startswith("==ERROR==") and PERMISSION_ERROR in line
            unsupported_gpu |= line.startswith("==ERROR==") and (
                "ERR_NVGPU:" in line or "unsupported gpu" in line.lower()
            )
        status = run.wait()
    if permission_denied:
        raise MetricsError(
            f"{PERMISSION_ERROR}: the NVIDIA driver denied access to GPU performance counters on this host"
        )
    if unsupported_gpu:
        raise MetricsError("this GPU is not supported by the selected Nsight Compute version. "
                           "For pre-Turing GPUs, select --backend nvprof with a compatible CUDA toolkit.")
    if status:
        raise MetricsError(f"Nsight Compute collection exited with status {status} during {mode} replay")
    if not native_report.is_file() or native_report.stat().st_size == 0:
        raise MetricsError("Nsight Compute produced no report. Check that a CUDA kernel matches the launch filter.")
    return {"mode": "collect", "replay_mode": mode, "sections": sections or None,
            "kernel_filter": options.kernel, "launch_count": options.launch_count}


def export_csv(ncu, report, destination, metrics, source=False):
    page = "source" if source else "raw"
    command = [ncu, "--config-file", "off", "--import", str(report.absolute()),
               "--page", page, "--csv", "--print-units", "base"]
    if metrics:
        command += ["--metrics", metrics]
    if source:
        command += ["--print-source", "sass"]
    with destination.open("wb") as output:
        run = subprocess.run(command, stdout=output, stderr=subprocess.PIPE)
    if run.stderr:
        sys.stderr.write(run.stderr.decode(errors="replace"))
    if run.returncode:
        raise MetricsError(f"Nsight Compute {page} export exited with status {run.returncode}")
    if destination.stat().st_size == 0:
        raise MetricsError(f"Nsight Compute produced an empty {page} export")


def parse_raw_csv(path):
    """Read raw wide CSV tables, preserving units and metric values losslessly."""
    tables = []
    table = None
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source):
            if not row or row[0].startswith("==") or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0] == "ID" and "Kernel Name" in row:
                table = {"columns": row, "units": None, "rows": []}
                tables.append(table)
                continue
            if table is None:
                continue
            if len(row) != len(table["columns"]):
                raise MetricsError("unexpected row size in the Nsight Compute raw CSV export")
            if not row[0] and table["units"] is None and not table["rows"]:
                table["units"] = row
                continue
            if not row[0].isdigit():
                raise MetricsError("unexpected launch ID in the Nsight Compute raw CSV export")
            table["rows"].append(row)
    tables = [table for table in tables if table["rows"]]
    if not tables:
        raise MetricsError("no kernel launches in the Nsight Compute raw CSV export")
    return tables


def nvprof_permission_denied(line):
    return any(indicator in line for indicator in (
        PERMISSION_ERROR, "CUPTI_ERROR_INSUFFICIENT_PRIVILEGES",
        "does not have permission to profile on the target device",
    ))


def run_nvprof(nvprof, options, native_report, log):
    # nvprof expands percent macros in both output paths. The import pathname
    # is literal, and the profiled application's output is excluded from CSV.
    command = [nvprof, "--print-gpu-trace", "--csv", "--log-file", str(log).replace("%", "%%")]
    mode = options.replay_mode or "kernel"
    metrics = options.metrics or LEGACY_DEFAULT_METRICS
    if options.report:
        command += ["--import-profile", str(options.report.absolute())]
    else:
        command += ["--export-profile", str(native_report).replace("%", "%%"), "--replay-mode", mode]
        if options.kernel:
            command += ["--kernels", options.kernel]
        command += ["--metrics", metrics, *options.application]

    permission_denied = False
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace") as run:
        for line in run.stdout:
            sys.stderr.write(line)
            permission_denied |= nvprof_permission_denied(line)
        status = run.wait()
    if log.is_file():
        with log.open(encoding="utf-8-sig", errors="replace") as diagnostics:
            for line in diagnostics:
                if line.startswith("==") or line.startswith(("Warning:", "Error:")):
                    permission_denied |= nvprof_permission_denied(line)
                    if any(level in line for level in ("Error:", "Warning:", "ERR_")) or status:
                        sys.stderr.write(line)
    if permission_denied:
        raise MetricsError(f"{PERMISSION_ERROR}: the NVIDIA driver denied access to GPU performance counters")
    if status:
        operation = "profile import" if options.report else f"{mode} replay"
        raise MetricsError(f"nvprof exited with status {status} during {operation}")
    if not options.report and (not native_report.is_file() or native_report.stat().st_size == 0):
        raise MetricsError("nvprof produced no native profile. Check the toolkit, device and kernel filter.")
    if not log.is_file():
        raise MetricsError("nvprof did not produce a CSV trace")
    if options.report:
        return {"mode": "import", "replay_mode": None}
    return {"mode": "collect", "replay_mode": mode, "sections": None,
            "kernel_filter": options.kernel, "launch_count": None}


def parse_nvprof_csv(path, metrics):
    """Read per-launch metric tables without casting vendor values or inventing units."""
    metadata = {"Device", "Context", "Stream", "Kernel", "Correlation_ID", "Correlation ID"}
    labels = {"Kernel": "Kernel Name", "Correlation_ID": "Correlation ID"}
    missing = {"", "-", "n/a", "na"}
    selected = set(metrics.split(",")) if metrics and metrics != "all" else None
    tables = []
    table = None
    launch_id = 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source):
            if not row or not any(row) or row[0].startswith("==") or row[0].startswith(("Warning:", "Error:")):
                continue
            if row[:3] == ["Device", "Context", "Stream"] and "Kernel" in row[3:5]:
                kernel_column = row.index("Kernel")
                metric_columns = [i for i, name in enumerate(row) if name not in metadata
                                  and (selected is None or name in selected)]
                metadata_columns = [i for i, name in enumerate(row) if name in metadata]
                columns = [*metadata_columns, *metric_columns]
                table = {"columns": ["ID", *(labels.get(row[i], row[i]) for i in columns)],
                         "units": None, "rows": []}
                tables.append(table)
                width = len(row)
                continue
            if table is None or not metric_columns:
                continue
            if len(row) != width:
                raise MetricsError("unexpected row size in the nvprof kernel metric CSV")
            if not row[0] and not row[kernel_column] and table["units"] is None and not table["rows"]:
                table["units"] = ["", *(row[i] for i in columns)]
                continue
            if not row[kernel_column] or all(row[i].strip().lower() in missing for i in metric_columns):
                continue
            table["rows"].append([str(launch_id), *(row[i] for i in columns)])
            launch_id += 1
    tables = [table for table in tables if table["rows"]]
    if not tables:
        raise MetricsError("no per-launch counters found in the nvprof CSV. "
                           "Export with --print-gpu-trace --csv and select metrics present in the profile.")
    return tables


def write_metric_tables(path, tables):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        for table in tables:
            writer.writerow(table["columns"])
            if table["units"]:
                writer.writerow(table["units"])
            writer.writerows(table["rows"])


def publish_outputs(pairs):
    # New hard links avoid overwriting reports in a concurrent invocation.
    published = []
    try:
        for source, destination in pairs:
            os.link(source, destination)
            published.append((source, destination))
    except OSError:
        for source, destination in published:
            if destination.exists() and destination.samefile(source):
                destination.unlink()
        raise


def metrics(options):
    legacy = options.backend == "nvprof"
    native_suffix = ".nvprof" if legacy else ".ncu-rep"
    prefix = options.output
    suffixes = [".metrics.csv", ".metrics.json"]
    if not options.report:
        suffixes.insert(0, native_suffix)
    if options.source:
        suffixes.insert(-1, ".source.csv")
    outputs = [Path(f"{prefix}{suffix}") for suffix in suffixes]
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise MetricsError(f"output already exists: {output}. Select a new prefix with --output")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ratelprof-metrics-", dir=prefix.parent) as work:
        stem = Path(work).absolute() / "report"
        native_report = Path(f"{stem}{native_suffix}")
        raw_csv = Path(f"{stem}.metrics.csv")
        measurement = {"mode": "import", "replay_mode": None}
        selected_metrics = options.metrics
        if legacy:
            if options.report and options.report.suffix.lower() == ".csv":
                vendor_csv = options.report
            else:
                vendor_csv = Path(f"{stem}.nvprof.csv")
                nvprof = find_nvprof(options.nvprof)
                measurement = run_nvprof(nvprof, options, native_report, vendor_csv)
            if not options.report:
                selected_metrics = options.metrics or LEGACY_DEFAULT_METRICS
            tables = parse_nvprof_csv(vendor_csv, selected_metrics)
            write_metric_tables(raw_csv, tables)
        else:
            ncu = find_ncu(options.ncu)
            if not options.report:
                measurement = collect(ncu, options, native_report)
            report = options.report or native_report
            export_csv(ncu, report, raw_csv, options.metrics)
            tables = parse_raw_csv(raw_csv)
            if options.source:
                export_csv(ncu, report, Path(f"{stem}.source.csv"), options.metrics, source=True)

        measurement["metrics"] = selected_metrics.split(",") if selected_metrics else None
        result = {
            "schema": "ratelprof.metrics.v1",
            "profiler": "nvprof" if legacy else "Nsight Compute",
            "report": str(options.report or Path(f"{prefix}{native_suffix}")),
            "measurement": measurement,
            "launch_count": sum(len(table["rows"]) for table in tables),
            "tables": tables,
        }
        raw_json = Path(f"{stem}.metrics.json")
        raw_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        publish_outputs(zip((Path(f"{stem}{suffix}") for suffix in suffixes), outputs))
    print(f"RATELProf metrics: {result['launch_count']} launch(es)")
    for output in outputs:
        print(f"  {output}")


def main(argv=None):
    options = parse_options(argv)
    try:
        metrics(options)
    except (MetricsError, OSError, csv.Error, UnicodeError) as error:
        print(f"RATELProf metrics: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
