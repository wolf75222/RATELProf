"""Translate NVIDIA nvprof GPU/API trace CSV into a normalized RATELProf rank."""

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


HOST = 1 << 32
NANOSECONDS = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}
BYTES = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def key(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def rows(source):
    columns = units = None
    for row in csv.reader(source):
        if not row or row[0].startswith("==") or not any(row):
            continue
        names = [key(column) for column in row]
        if {"start", "duration", "name"}.issubset(names):
            columns, units = names, None
            continue
        if columns is None:
            continue
        if len(row) != len(columns):
            raise ValueError("unexpected column count in nvprof trace CSV")
        values = dict(zip(columns, (cell.strip() for cell in row)))
        if values["start"] in NANOSECONDS:
            units = values
        elif units:
            yield values, units
        else:
            raise ValueError("nvprof trace CSV omits its timestamp units row")
    if columns is None:
        raise ValueError("expected a trace CSV with Start, Duration and Name columns")


def scaled(value, unit, scales):
    if not value or value == "-":
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\s*([A-Za-z]+))?", value)
    if not match or (match[2] or unit) not in scales:
        raise ValueError(f"invalid nvprof measurement {value!r} with unit {unit!r}")
    return round(Decimal(match[1]) * scales[match[2] or unit])


def value(record, *names, default=""):
    return next((record[name] for name in names if record.get(name) not in (None, "", "-")), default)


def integer(record, *names, default=None):
    field = value(record, *names)
    return int(field) if field else default


def dimensions(row, prefix):
    parts = [value(row, prefix + coordinate) for coordinate in "xyz"]
    if not all(parts):
        parts = re.findall(r"\d+", value(row, prefix + "size"))
    return list(map(int, parts)) if len(parts) == 3 else [0, 0, 0]


def identity(row):
    cid = integer(row, "correlationid", "correlation")
    match = re.search(r"\s+\[(\d+)\](?:\))?$", row["name"])
    if match:
        fallback = None if "parentid" in row else int(match[1])
        return row["name"][:match.start()], cid if cid is not None else fallback
    return row["name"], cid


def gpu_record(row, units, device):
    name, correlation = identity(row)
    if device is None:
        raise ValueError(f"nvprof GPU activity {name!r} has no visible device")
    args = {
        "gpu_id": device, "queue_id": integer(row, "stream", default=-1),
        "context_id": integer(row, "context", default=-1),
    }
    copy = re.fullmatch(r"\[CUDA (?:memcpy|memcopy) (HtoD|DtoH|DtoD|HtoH|PtoP)\]", name, re.I)
    if copy:
        mode = copy[1].lower()
        endpoints = list(map(int, re.findall(r"\((\d+)\)", value(row, "device"))))
        src = (endpoints[0] if endpoints else device) if mode.startswith(("d", "p")) else HOST
        dst = (endpoints[-1] if endpoints else device) if mode.endswith(("d", "p")) else HOST
        if mode == "ptop" and len(endpoints) < 2:
            raise ValueError("nvprof peer copy does not specify both devices")
        count = scaled(value(row, "size"), units.get("size", ""), BYTES)
        args.update(src_agent=src, dst_agent=dst, src_type=int(src != HOST), dst_type=int(dst != HOST),
                    size=count or 0, engine_id=args["queue_id"])
        return "COPY", name, args, correlation
    if name.lower() == "[cuda memset]":
        count = scaled(value(row, "size"), units.get("size", ""), BYTES)
        args["size"] = count or 0
        return "CUDA_MEMSET", name, args, correlation
    if name.startswith("[CUDA"):
        raise ValueError(f"unrecognized nvprof GPU activity {name!r}")
    static = scaled(value(row, "staticsmem", "ssmem"), value(units, "staticsmem", "ssmem"), BYTES)
    dynamic = scaled(value(row, "dynamicsmem", "dsmem"), value(units, "dynamicsmem", "dsmem"), BYTES)
    args.update(kernel_name=name, wrg=dimensions(row, "block"), grd=dimensions(row, "grid"),
                grid_unit="blocks", group_segment_size=(static or 0) + (dynamic or 0),
                private_segment_size="---")
    registers = integer(row, "registersperthread", "regs")
    if registers is not None:
        args["registers_per_thread"] = registers
    return "KERNEL", name, args, correlation


def api_record(row):
    name, correlation = identity(row)
    function = name.split(" ", 1)[0].split("(", 1)[0]
    domain = "CUDA_RUNTIME" if function.startswith("cuda") else "CUDA_DRIVER"
    if not function.startswith("cu"):
        raise ValueError(f"unrecognized nvprof CUDA API activity {name!r}")
    return domain, function, {}, correlation


def convert(gpu, api, arch):
    events, nodes, devices, ordinals, correlations = {}, {HOST: HOST}, {}, {}, {}
    end = 0
    next_id = 1
    for csv_file, is_gpu in ((api, False), (gpu, True)):
        if csv_file is None:
            continue
        with open(csv_file, encoding="utf-8-sig", newline="") as stream:
            for row, units in rows(stream):
                start = scaled(row["start"], units["start"], NANOSECONDS)
                duration = scaled(row["duration"], units["duration"], NANOSECONDS)
                if start is None or duration is None:
                    raise ValueError("nvprof trace row omits start or duration")
                if is_gpu:
                    label = value(row, "device")
                    ordinal = re.search(r"\((\d+)\)", label)
                    if ordinal:
                        device = int(ordinal[1])
                    elif label:
                        if label not in ordinals:
                            ordinals[label] = max(nodes.keys() - {HOST}, default=-1) + 1
                        device = ordinals[label]
                    else:
                        device = None
                    domain, name, args, cid = gpu_record(row, units, device)
                    for gpu_id in {device, args.get("src_agent"), args.get("dst_agent")} - {None, HOST}:
                        node = gpu_id + 1
                        nodes[gpu_id] = node
                        devices.setdefault(node, {"vendor": "cuda", "device_type_name": "GPU",
                                                  "ordinal": gpu_id, "node": node,
                                                  "name": label if gpu_id == device else f"CUDA GPU {gpu_id}",
                                                  "uuid": "", "arch": arch or "unknown"})
                    corr_id = correlations.get(cid, 0)
                    if domain == "KERNEL":
                        args["dispatch_time"] = start
                else:
                    domain, name, args, cid = api_record(row)
                    corr_id = 0
                    if cid is not None:
                        correlations[cid] = next_id
                event = {"name": name, "start": start, "dur": duration, "corr_id": corr_id,
                         "pid": integer(row, "processid", "pid", default=0),
                         "tid": integer(row, "threadid", "tid", default=0), "phase": 1, "args": args}
                if cid is not None:
                    event["cupti_correlation"] = cid
                events.setdefault("RATELPROF_DOMAIN_" + domain, {})[next_id] = event
                end = max(end, start + duration)
                next_id += 1
    if not events:
        raise ValueError("nvprof trace CSV contains no activities")
    return {"version": [1, 1, 0], "mpi_rank": -1,
            "experiment_start_epoch": 0, "lifecycle": [0, end, end],
            "main_data": [0, "nvprof import"], "node_id": nodes,
            "locations": {}, "gpu_props": devices, "trace_events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu")
    parser.add_argument("--api")
    parser.add_argument("--arch")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--nvprof", default=os.environ.get("RATELPROF_NVPROF", "nvprof"))
    args = parser.parse_args()
    if args.arch and not re.fullmatch(r"sm_\d+[af]?", args.arch):
        parser.error("--arch expects a CUDA architecture such as sm_30")
    if bool(args.profile) == bool(args.gpu or args.api):
        parser.error("provide --profile or one or both of --gpu and --api")
    try:
        if args.profile:
            with tempfile.TemporaryDirectory(prefix="ratelprof-nvprof-") as folder:
                base = Path(folder)
                gpu, api = base / "gpu.csv", base / "api.csv"
                for trace, path in (("gpu", gpu), ("api", api)):
                    subprocess.run([args.nvprof, "--import-profile", str(args.profile.resolve()),
                                    f"--print-{trace}-trace", "--csv", "--log-file", str(path)],
                                   check=True, stdout=subprocess.DEVNULL)
                report = convert(gpu, api, args.arch)
        else:
            report = convert(args.gpu, args.api, args.arch)
        json.dump(report, sys.stdout)
    except (OSError, ValueError, InvalidOperation, csv.Error, subprocess.CalledProcessError) as error:
        parser.exit(1, f"nvprof trace import: {error}\n")


if __name__ == "__main__":
    main()
