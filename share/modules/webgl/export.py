"""Write the RATELProf-Visualize WebGL layout from a RATELProf report."""

import argparse
import base64
import heapq
import html
import json
import math
import shutil
import struct
from collections import defaultdict
from itertools import islice
from pathlib import Path

from measurements import measurement_tables


BUCKET_SIZE = 100_000
HISTOGRAM_BARS = 1_000
GPU_DOMAINS = {"KERNEL", "COPY", "CUDA_MEMSET", "CUDA_EVENT", "MEMORY_FILL", "BARRIERAND", "BARRIEROR"}
DOMAIN_PREFIX = "RATELPROF_DOMAIN_"


def indexed(table):
    if isinstance(table, list):
        return enumerate(table, 1)
    return ((int(index), value) for index, value in table.items())


def json_text(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def write_js(path, name, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{name}={json_text(value)};\n", encoding="utf-8")


def pack(value):
    # RATELProf-Visualize's reader uses little-endian multibyte values.
    if isinstance(value, bool):
        return b"\xc3" if value else b"\xc2"
    if isinstance(value, int):
        if -32 <= value < 128:
            return bytes((value & 0xFF,))
        if -(2**31) <= value < -32:
            return struct.pack("<Bi", 0xD2, value)
        return struct.pack("<Bq" if value < 0 else "<BQ", 0xD3 if value < 0 else 0xCF, value)
    if isinstance(value, float):
        return struct.pack("<Bd", 0xCB, value)
    if isinstance(value, str):
        data = value.encode("utf-8")
        return struct.pack("<BI", 0xDB, len(data)) + data
    if isinstance(value, list):
        return struct.pack("<BI", 0xDD, len(value)) + b"".join(map(pack, value))
    if value is None:
        return b"\xc0"
    # Named fields in nested values remain inspectable as JSON in the viewer.
    return pack(json_text(value))


def b64(content):
    return base64.b64encode(content).decode("ascii")


def interval_time(intervals):
    total = stop = 0
    for begin, end in sorted(intervals):
        total += max(0, end - max(begin, stop))
        stop = max(stop, end)
    return total


def event_name(domain, event):
    args = event.get("args", {})
    if domain == "KERNEL":
        return args.get("kernel_name", "Kernel")
    if domain == "COPY":
        src = "Device" if args.get("src_type") == 1 else "Host"
        dst = "Device" if args.get("dst_type") == 1 else "Host"
        return f"Copy {src} to {dst}"
    if domain == "CUDA_MEMSET":
        return "CUDA memset"
    if domain == "MEMORY_FILL":
        return args.get("operation_name", "Memory fill")
    return event.get("name", domain)


def format_arguments(event):
    arguments = event.get("args", {})
    if not isinstance(arguments, dict):
        return ()
    result = []
    for name, value in sorted(arguments.items()):
        if isinstance(value, dict) and "value" in value:
            result.append((name, value.get("type", "value"), value["value"]))
        else:
            result.append((name, type(value).__name__, value))
    return result


def track_key(domain, event, nodes):
    args = event.get("args", {})
    if domain not in GPU_DOMAINS:
        return ("PID", event.get("pid", 0), "TID", event.get("tid", 0))
    controller = args.get("gpu_id")
    if domain == "COPY" and controller is None:
        controller = args["dst_agent"] if args.get("dst_type") == 1 else args["src_agent"]
    gpu = nodes.get(controller, 0)
    context = args.get("context_id")
    if context is not None:
        label = f"Stream (context {context})"
    else:
        label = "Engine" if domain == "COPY" and "queue_id" not in args else "Queue"
    return ("GPU", gpu, label, args.get("queue_id", args.get("engine_id", 0)))


class RankExport:
    def __init__(self, path, lifecycle, nodes, labels, locations, start, stop):
        self.path = path
        self.lifecycle = lifecycle
        self.nodes = nodes
        self.labels = labels
        self.locations = locations
        self.start = start
        self.stop = stop
        self.strings = {}
        self.api = {}
        self.source = {}
        self.rows = []
        self.group_data = []
        self.hist_width = max(1, math.ceil(lifecycle[2] / HISTOGRAM_BARS))
        self.worker = Path(__file__).with_name("bucket-worker.js").read_text(encoding="utf-8")

    def intern(self, text):
        return self.strings.setdefault(text, len(self.strings))

    def metadata(self, name, event):
        args = format_arguments(event)
        signature = (name, tuple((key, label) for key, label, _ in args))
        api_id = self.api.setdefault(signature, len(self.api))
        address = int(event.get("return_address") or 0)
        loc_id = self.source.setdefault(address, len(self.source))
        return b64(b"".join(map(pack, [api_id, loc_id, len(args), *(value for _, _, value in args)])))

    def add_group(self, unit_label, unit, domain, tracks):
        group = len(self.group_data)
        histogram = defaultdict(lambda: defaultdict(int))
        payload = bytearray()
        track_info = []
        for track, ((label, value), events) in enumerate(tracks.items()):
            occupied = []
            free = []
            width = 0
            for event_id, event in sorted(events, key=lambda item: (item[1]["start"], item[0])):
                start, dur = event["start"], event["dur"]
                end = start + dur
                while occupied and occupied[0][0] <= start:
                    _, layer = heapq.heappop(occupied)
                    heapq.heappush(free, layer)
                if free:
                    layer = heapq.heappop(free)
                else:
                    layer = width
                    width += 1
                heapq.heappush(occupied, (end, layer))

                name = event_name(domain, event)
                function = self.intern(name)
                row = (group, track, layer, function, start, dur, event_id, event.get("corr_id", 0))
                self.rows.append((row, self.metadata(name, event)))
                first = start // self.hist_width
                last = min(HISTOGRAM_BARS - 1, (end - 1) // self.hist_width)
                if layer == 0:
                    for i in range(first, last + 1):
                        overlap = min(end, (i + 1) * self.hist_width) - max(start, i * self.hist_width)
                        histogram[i + 1][function] += overlap
            track_info.append((track, label, value, width))

        # Mode 2 is the viewer's typed argument record and carries CUDA metadata
        # without borrowing AMD-specific completion signals or queue objects.
        for item in (group, unit_label, self.labels.get(DOMAIN_PREFIX + domain, domain), 2, unit, len(track_info)):
            payload.extend(pack(item))
        for info in track_info:
            payload.extend(b"".join(map(pack, info)))
        payload.extend(pack(len(histogram)))
        width = self.hist_width * len(tracks)
        for index, segments in sorted(histogram.items()):
            payload.extend(pack(index) + pack(len(segments)))
            for function, elapsed in segments.items():
                payload.extend(pack(function) + pack(elapsed / width))
        self.group_data.append(payload)

    def write_sections(self):
        groups = pack(len(self.group_data)) + b"".join(self.group_data)
        write_js(self.path / "groups.js", "window.groups", b64(groups))

        apis = bytearray()
        for name, args in self.api:
            apis.extend(pack(self.intern(name)) + pack(len(args)))
            for key, label in args:
                apis.extend(pack(self.intern(label)) + pack(self.intern(key)))
        write_js(self.path / "api_data.js", "window.api_data", b64(apis + struct.pack("<Q", len(self.api))))
        write_js(self.path / "kernel_data.js", "window.kernel_data", b64(struct.pack("<Q", 0)))

        locations = bytearray()
        for address in self.source:
            loc = self.locations.get(address, {})
            fields = [address, max(0, address - loc.get("offset", 0)),
                      self.intern(loc.get("ofile", "Unknown")), self.intern(loc.get("sfun", "Unknown")),
                      self.intern(loc.get("sfile", loc.get("source", "Unknown"))), loc.get("sline", 0)]
            locations.extend(b"".join(map(pack, fields)))
        write_js(self.path / "locations.js", "window.locations", b64(locations + struct.pack("<Q", len(self.source))))
        write_js(self.path / "strings.js", "window.strings", list(self.strings))

    def write_buckets(self):
        buckets = {}
        ordered = iter(sorted(self.rows, key=lambda item: (item[0][4], item[0][6])))
        for index in range(math.ceil(len(self.rows) / BUCKET_SIZE)):
            rows, metadata = zip(*islice(ordered, BUCKET_SIZE))
            script = f"const rows={json_text(rows)};\n" + self.worker
            path = self.path / f"bucket_{index}.js"
            path.write_text(
                f"window.metadata={json_text(metadata)};\n"
                f"window.worker=new Worker(URL.createObjectURL(new Blob([{json_text(script)}],"
                "{type:'application/javascript'})));\n",
                encoding="utf-8",
            )
            buckets[index] = {"minStart": rows[0][4], "maxStop": max(row[4] + row[5] for row in rows), "count": len(rows)}
        data = {"mainTime": self.lifecycle[1] - self.lifecycle[0], "maxTime": self.lifecycle[2], "bucketList": buckets}
        write_js(self.path / "buckets.js", "window.buckets", data)
        return buckets

    def write(self, domains):
        tracks = defaultdict(lambda: defaultdict(list))
        duration_by_domain = {}
        for key, events in (domains or {}).items():
            domain = key.removeprefix(DOMAIN_PREFIX)
            intervals = []
            for event_id, event in indexed(events):
                if event["start"] < self.start or event["start"] + event["dur"] > self.stop:
                    continue
                unit, rank_unit, track, value = track_key(domain, event, self.nodes)
                tracks[(unit, rank_unit, domain)][(track, value)].append((event_id, event))
                intervals.append((event["start"], event["start"] + event["dur"]))
            duration_by_domain[domain] = (interval_time(intervals), len(intervals))
        for (unit, value, domain), group_tracks in tracks.items():
            self.add_group(unit, value, domain, group_tracks)
        self.write_sections()
        return duration_by_domain, self.write_buckets()


def write_timeline(output, filenames, bounds, lifecycles):
    write_js(output, "const __TIMELINE_DATA__", {"FILENAMES": filenames, "MAX_BUCKET_SIZE": BUCKET_SIZE})
    start, stop = bounds or (0, max((life[2] for life in lifecycles.values()), default=1))
    stop = max(start + 1, stop)
    # The viewer's normal default is the first millisecond of the process.
    # Start at the trace on first visit while preserving saved zoom and URL ranges.
    with output.open("a", encoding="utf-8") as file:
        file.write(
            'try {\n'
            '    const key = document.title + "timelineConfig";\n'
            '    if (localStorage.getItem(key) === null) {\n'
            f'        localStorage.setItem(key, JSON.stringify({{initialView: [{start}, {stop}]}}));\n'
            '    }\n'
            '} catch {}\n'
        )


def write_breakdown(output, counts, labels, lifecycles, gpu):
    domains = sorted({domain for by_domain in counts.values() for domain in by_domain
                      if (domain in GPU_DOMAINS) == gpu})
    table = [["Rank", "Duration (ns)", "Events", *(labels.get(DOMAIN_PREFIX + name, name) + " (%)" for name in domains)]]
    for rank, info in counts.items():
        span = lifecycles[rank][2]
        active = [info.get(domain, (0, 0)) for domain in domains]
        table.extend([[f"Rank {rank}", span, sum(count for _, count in active),
                       *(round(100 * time / max(1, span), 6) for time, _ in active)], []])
    write_js(output, "const __BREAKDOWN_DATA__", table)
    return {
        "xLabel": "Rank",
        "yLabel": "Active Time (%)",
        "labels": [row[0] for row in table[1::2]],
        "datasets": [{"label": label, "data": [row[i] for row in table[1::2]]}
                     for i, label in enumerate(table[0][3:], 3)],
    }


def write_tables(output, kind, reports):
    index = {}
    for report_id, report in (reports or {}).items():
        rows = report.get("DATA")
        message = (report.get("MSG") or "").strip()
        if report.get("skip") or (not rows and not message):
            continue
        entry = {"name": report["NAME"] + " " + report["TYPE"]}
        if rows:
            file = f"{report_id}.js"
            write_js(output / "csv" / kind / file, "window.currentCSV", rows)
            entry["file"] = file
        if message:
            entry["dataRendering"] = "renderAdvice"
            entry["data"] = {"advice": html.escape(message).replace("\n", "<br>"),
                             "desc": html.escape(report["NAME"])}
        index[report_id] = entry
    write_js(output / f"{kind}.js", "const __CSV_DATA__", index)


def export(payload, viewer, output):
    assets = viewer / "share" / "html"
    if not (assets / "timeline" / "js" / "RProfVis.js").is_file():
        raise ValueError("Expected the root of a RATELProf-Visualize checkout with share/html.")
    if assets == output.resolve() or assets in output.resolve().parents:
        raise ValueError("Choose an output directory outside the viewer's source assets.")
    statistics = dict(payload["statistics"] or {})
    statistics.update(measurement_tables(payload.get("measurements") or {}))
    shutil.copytree(assets, output, dirs_exist_ok=True)
    selector = output / "timeline" / "js" / "SelectInput.js"
    if selector.is_file():
        script = selector.read_text(encoding="utf-8")
        # The viewer supplies rank options as objects with a display name.
        updated = script.replace("o.toLowerCase().includes(value)", "o.name.toLowerCase().includes(value)")
        if updated != script:
            selector.write_text(updated, encoding="utf-8")
    data_dir = output / "data"
    report = payload["report"]
    loc = dict(indexed(report["locations"]))
    lifecycles = dict(indexed(report["lifecycle"]))
    rank_nodes = dict(indexed(report.get("cuda_node_id_by_rank", {})))
    labels = payload["domains"]

    filenames = []
    counts = {}
    bounds = None
    for rank, domains in sorted(indexed(report["trace_events"])):
        basename = f"rank_{rank}"
        rank_export = RankExport(data_dir / "traces" / basename, lifecycles[rank],
                                 dict(indexed(rank_nodes.get(rank, report["node_id"]))),
                                 labels, loc, payload.get("start", 0), payload.get("stop", math.inf))
        counts[rank], buckets = rank_export.write(domains)
        if buckets:
            start = min(bucket["minStart"] for bucket in buckets.values())
            stop = max(bucket["maxStop"] for bucket in buckets.values())
            bounds = (min(bounds[0], start), max(bounds[1], stop)) if bounds else (start, stop)
        filenames.append({"basename": basename, "rank": rank})
    write_timeline(data_dir / "timeline.js", filenames, bounds, lifecycles)

    gpu_plot = write_breakdown(data_dir / "gpu_breakdown.js", counts, labels, lifecycles, True)
    cpu_plot = write_breakdown(data_dir / "cpu_breakdown.js", counts, labels, lifecycles, False)
    profiling = payload["summary"]["profilingSummary"]
    if profiling["rows"]:
        plot = gpu_plot if gpu_plot["datasets"] else cpu_plot
        if plot["datasets"]:
            profiling["rows"][0].update({"onClick": "renderHistogram", "data": plot, "defaultPlot": True})

    gpus = {}
    for node, properties in indexed(report["gpu_props"]):
        name = f"gpu_{node}.js"
        write_js(data_dir / "gpus" / "Report" / name, "window.currentGPU", properties)
        gpus[node] = name
    experiment = payload["summary"]["experimentInfo"]
    experiment["rows"].append({"cells": ["Machine(s)", "Report"]})
    write_js(data_dir / "summary.js", "const __SUMMARY_DATA__", {
        "PROFILING": profiling, "EXPERIMENT": experiment, "GPU": {"Report": gpus},
    })
    write_tables(data_dir, "statistics", statistics)
    write_tables(data_dir, "analyze", payload["analyze"])
    write_js(data_dir / "misc.js", "document.title", "RATELProf - " + output.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("viewer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        export(json.loads(args.report.read_text(encoding="utf-8")), args.viewer.resolve(), args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"RATELProf WebGL export: {error}\n")


if __name__ == "__main__":
    main()
