"""Validate CUDA dependency and all-arguments output from event_dependency.cu."""

import json
import sys
from pathlib import Path


def values(table):
    return list(table.values()) if isinstance(table, dict) else table


report = json.loads(Path(sys.argv[1]).read_text())
(trace,) = values(report["trace_events"])

def events(domain):
    return values(trace["RATELPROF_DOMAIN_" + domain])

waits = [event["args"] for event in events("CUDA_SYNC") if event["name"] == "Stream wait event"]
assert len(waits) == 2
assert waits[0]["event_id"]["value"] == waits[1]["event_id"]["value"]
assert len({wait["event_sync_id"]["value"] for wait in waits}) == 2
assert all(wait["stream_id"]["value"] != wait["record_stream_id"]["value"] for wait in waits)
assert all(wait["record_context_id"]["value"] == wait["context_id"]["value"] for wait in waits)

api_waits = [event["args"] for event in events("CUDA_RUNTIME")
             if event["name"].startswith("cudaStreamWaitEvent")]
assert len(api_waits) == 2
assert all(api["flags"]["value"] == 0 for api in api_waits)
assert all(api["event"]["value"] != 0 and api["stream"]["value"] != 0 for api in api_waits)

if "RATELPROF_DOMAIN_CUDA_EVENT" in trace:
    records = {event["args"]["event_sync_id"] for event in events("CUDA_EVENT")}
    assert records.issuperset(wait["event_sync_id"]["value"] for wait in waits)

print("Verified two records of the same CUDA event, stream dependencies, and SDK arguments")
