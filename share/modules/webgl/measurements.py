"""Read independent CUDA measurement reports for the WebGL Statistics tab."""

import csv
import json
from pathlib import Path


LAUNCH_FIELDS = (
    "Process ID", "Process Name", "Host Name", "Device", "CC",
    "Context", "Stream", "Block Size", "Grid Size",
)


def read_json(path, schema):
    with path.open(encoding="utf-8") as source:
        report = json.load(source)
    if not isinstance(report, dict) or report.get("schema") != schema:
        raise ValueError(f"expected schema {schema}")
    return report


def text_row(row, width):
    return isinstance(row, list) and len(row) == width and all(isinstance(cell, str) for cell in row)


def metrics_table(path):
    report = read_json(path, "ratelprof.metrics.v1")
    measurement = report.get("measurement")
    tables = report.get("tables")
    if not isinstance(tables, list) or not tables or not isinstance(measurement, dict):
        raise ValueError("missing Nsight Compute tables or measurement metadata")
    rows = [["Kernel Name", "Metric", "Unit", "Value", "Report Table", "Launch ID", *LAUNCH_FIELDS]]
    for number, table in enumerate(tables, 1):
        if not isinstance(table, dict) or not isinstance(table.get("columns"), list):
            raise ValueError(f"missing Nsight Compute headers in table {number}")
        columns = table["columns"]
        if not text_row(columns, len(columns)) or not {"ID", "Kernel Name"}.issubset(columns):
            raise ValueError(f"invalid Nsight Compute headers in table {number}")
        units = table.get("units")
        if units is None:
            units = [""] * len(columns)
        launches = table.get("rows")
        if not text_row(units, len(columns)) or not isinstance(launches, list):
            raise ValueError(f"invalid Nsight Compute units or launches in table {number}")
        metrics = [(i, name, units[i]) for i, name in enumerate(columns)
                   if name not in (*LAUNCH_FIELDS, "ID", "Kernel Name")]
        for values in launches:
            if not text_row(values, len(columns)):
                raise ValueError(f"invalid Nsight Compute launch in table {number}")
            launch = dict(zip(columns, values))
            identity = [str(number), launch["ID"], *(launch.get(field, "") for field in LAUNCH_FIELDS)]
            rows.extend([launch["Kernel Name"], metric, unit, values[i], *identity]
                        for i, metric, unit in metrics)
    message = [f"Nsight Compute report {path.name}. Launch IDs refer to this report."]
    replay = measurement.get("replay_mode")
    if replay:
        message.append(f"Collection used {replay} replay.")
    elif measurement.get("mode") == "import":
        message.append("Imported from an existing Nsight Compute recording.")
    if len(rows) == 1:
        message.append("This report contains no metric values.")
    return {"cuda_metrics": {
        "NAME": "Nsight Compute", "TYPE": "Hardware Counters",
        "DATA": rows if len(rows) > 1 else [], "MSG": "\n".join(message),
    }}


def is_count(value):
    return type(value) is int and value >= 0


def pc_samples_table(path):
    report = read_json(path, "ratelprof-pcsampling-v1")
    samples = report.get("samples")
    counts = ("contexts", "records", "stall_samples", "dropped_samples")
    if report.get("complete") is not True or not isinstance(samples, list):
        raise ValueError("expected a complete CUPTI PC sampling report")
    if not all(is_count(report.get(key)) for key in counts) or report["records"] != len(samples):
        raise ValueError("invalid CUPTI PC sampling report counts")
    rows = [["Function", "PC Offset", "Stall Reason", "Samples", "Source File", "Source Line",
             "Context", "Range", "Cubin CRC", "Function Index"]]
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("invalid CUPTI PC sample")
        reasons = sample.get("stall_reasons")
        source = sample.get("source", {"file": "", "line": ""})
        if (not all(is_count(sample.get(key)) for key in ("context", "range", "pc_offset", "function_index"))
                or not all(isinstance(sample.get(key), str) for key in ("function", "cubin_crc"))
                or not isinstance(reasons, dict) or not all(is_count(count) for count in reasons.values())
                or not isinstance(source, dict) or not isinstance(source.get("file"), str)
                or (source.get("line") != "" and not is_count(source.get("line")))):
            raise ValueError("invalid CUPTI PC sample")
        rows.extend([sample["function"], hex(sample["pc_offset"]), reason, count, source["file"],
                     str(source["line"]), str(sample["context"]), str(sample["range"]),
                     sample["cubin_crc"], str(sample["function_index"])]
                    for reason, count in reasons.items())
    period = report.get("period_exponent")
    interval = f"every 2^{period} cycles" if is_count(period) else "at the CUPTI default rate"
    message = [f"CUPTI PC sampling report {path.name}, sampled {interval}.",
               f"{report['contexts']} contexts, {report['records']} PC records, "
               f"{report['stall_samples']} samples, {report['dropped_samples']} dropped.",
               "PC offsets refer to the function and cubin in this recording."]
    if len(rows) == 1:
        message.append("This report contains no sampled stalls.")
    return {"cuda_pc_sampling": {
        "NAME": "CUPTI", "TYPE": "PC Sampling",
        "DATA": rows if len(rows) > 1 else [], "MSG": "\n".join(message),
    }}


def source_tables(path):
    tables = {}
    name, header, rows, number = "", None, None, 0
    with path.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source, strict=True):
            if not row or (len(row) == 1 and not row[0].strip()) or row[0].startswith("=="):
                continue
            if row[0] == "Kernel Name" and len(row) >= 2:
                name, header = row[1], None
                continue
            if row[:2] == ["Address", "Source"]:
                header = tuple(row)
                rows = tables.setdefault(header, [["Kernel Name", "Source Page", *row]])
                number += 1
                continue
            if header is None or len(row) != len(header):
                raise ValueError("unexpected row in Nsight Compute SASS source CSV")
            rows.append([name, str(number), *row])
    if not tables:
        raise ValueError("no SASS source table in Nsight Compute CSV")
    return {f"cuda_sass_source_{index}": {
        "NAME": "Nsight Compute", "TYPE": f"SASS Source {index}" if len(tables) > 1 else "SASS Source",
        "DATA": rows if len(rows) > 1 else [],
        "MSG": f"SASS instructions and per-instruction values from {path.name}. "
               "Addresses refer to this Nsight Compute recording.",
    } for index, rows in enumerate(tables.values(), 1)}


def measurement_tables(paths):
    tables = {}
    for option, parse in (("metrics-json", metrics_table), ("pc-samples", pc_samples_table),
                          ("source-csv", source_tables)):
        if option in paths:
            try:
                tables.update(parse(Path(paths[option])))
            except (OSError, ValueError, csv.Error) as error:
                raise ValueError(f"--{option} {paths[option]}: {error}") from error
    return tables
