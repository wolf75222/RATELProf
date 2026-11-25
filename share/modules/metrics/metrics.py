"""Collect or export per-launch NVIDIA counters via Nsight Compute."""

import argparse
import csv
import json
import os
from pathlib import Path
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


def parse_options(argv):
    parser = argparse.ArgumentParser(
        prog="ratelprof metrics",
        description=__doc__,
        epilog="Collection is a separate run and Nsight Compute may replay kernels or the application.",
    )
    parser.add_argument("-o", "--output", type=Path, help="output prefix")
    parser.add_argument("--ncu", default=os.environ.get("RATELPROF_NCU", "ncu"),
                        help="Nsight Compute CLI executable (default: ncu on PATH)")
    parser.add_argument("--import", dest="report", type=Path,
                        help="export counters from an existing .ncu-rep, without running the application")
    parser.add_argument("--replay-mode", choices=("kernel", "application"),
                        help="Nsight Compute replay mode (default: kernel)")
    parser.add_argument("--kernel", type=nonempty, help="regular expression selecting kernel names")
    parser.add_argument("--launch-count", type=positive_int, help="limit the number of matching launches")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--section", action="append", type=nonempty,
                           help="Nsight Compute section identifier (repeatable)")
    selection.add_argument("--metrics", type=nonempty, help="comma-separated Nsight Compute metric names")
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
            parser.error(f"cannot read the Nsight Compute report {options.report}")
    elif not options.application:
        parser.error("provide an application after --, or use --import REPORT")

    if options.metrics:
        metric_names = [name.strip() for name in options.metrics.split(",")]
        if not all(metric_names):
            parser.error("--metrics must contain a comma-separated list of nonempty metric names")
        options.metrics = ",".join(metric_names)
    if options.output is None:
        options.output = options.report.with_suffix("") if options.report else Path("rprof-metrics")
    return options


def find_ncu(executable):
    ncu = shutil.which(executable)
    if ncu:
        return str(Path(ncu).absolute())
    raise MetricsError(f"Nsight Compute executable {executable!r} is unavailable. Use --ncu PATH or load its toolkit.")


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
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace") as run:
        for line in run.stdout:
            sys.stderr.write(line)
            permission_denied |= line.startswith("==ERROR==") and PERMISSION_ERROR in line
        status = run.wait()
    if permission_denied:
        raise MetricsError(
            f"{PERMISSION_ERROR}: the NVIDIA driver denied access to GPU performance counters on this host"
        )
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
    ncu = find_ncu(options.ncu)
    prefix = options.output
    suffixes = [".metrics.csv", ".metrics.json"]
    if not options.report:
        suffixes.insert(0, ".ncu-rep")
    if options.source:
        suffixes.insert(-1, ".source.csv")
    outputs = [Path(f"{prefix}{suffix}") for suffix in suffixes]
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise MetricsError(f"output already exists: {output}. Select a new prefix with --output")

    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ratelprof-metrics-", dir=prefix.parent) as work:
        stem = Path(work).absolute() / "report"
        native_report = Path(f"{stem}.ncu-rep")
        if options.report:
            measurement = {"mode": "import", "replay_mode": None}
            report = options.report
        else:
            measurement = collect(ncu, options, native_report)
            report = native_report

        raw_csv = Path(f"{stem}.metrics.csv")
        export_csv(ncu, report, raw_csv, options.metrics)
        tables = parse_raw_csv(raw_csv)
        if options.source:
            export_csv(ncu, report, Path(f"{stem}.source.csv"), options.metrics, source=True)

        measurement["metrics"] = options.metrics.split(",") if options.metrics else None
        result = {
            "schema": "ratelprof.metrics.v1",
            "profiler": "Nsight Compute",
            "report": str(options.report or Path(f"{prefix}.ncu-rep")),
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
