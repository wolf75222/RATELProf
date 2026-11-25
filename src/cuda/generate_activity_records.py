"""Select the CUPTI activity record types documented by the installed SDK."""

import argparse
from pathlib import Path
import re


RECORDS = {
    "CONCURRENT_KERNEL": "KernelRecord",
    "CONTEXT": "ContextRecord",
    "MEMCPY": "MemcpyRecord",
    "MEMCPY2": "PeerCopyRecord",
    "MEMSET": "MemsetRecord",
    "SYNCHRONIZATION": "SyncRecord",
    "CUDA_EVENT": "CudaEventRecord",
    "DEVICE": "DeviceRecord",
    "MARKER": "MarkerRecord",
}

OPTIONAL_FIELDS = {
    "KernelRecord": {"submitted": "KERNEL_SUBMITTED"},
    "SyncRecord": {"cudaEventSyncId": "SYNC_EVENT_ID", "returnValue": "SYNC_RESULT"},
    "CudaEventRecord": {"deviceTimestamp": "EVENT_DEVICE_TIMESTAMP"},
    "DeviceRecord": {
        "isCudaVisible": "DEVICE_VISIBLE",
        "migUuid": "DEVICE_MIG_UUID",
        "uuid": "DEVICE_UUID",
    },
}

OPTIONAL_SYMBOLS = {
    "cuptiActivityRegisterTimestampCallback": "TIMESTAMP_CALLBACK",
    "cuptiActivityEnableLatencyTimestamps": "LATENCY_TIMESTAMPS",
    "cuptiActivityEnableCudaEventDeviceTimestamps": "REQUEST_EVENT_TIMESTAMPS",
    "CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC": "DEVICE_STATIC",
    "CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC": "MANAGED_STATIC",
}


def generate(include_dir):
    activity = (include_dir / "cupti_activity.h").read_text(encoding="utf-8")
    deprecated = include_dir / "cupti_activity_deprecated.h"
    source = activity + (deprecated.read_text(encoding="utf-8") if deprecated.is_file() else "")
    declarations = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    docs = {
        kind: comment for comment, kind in re.findall(
            r"(/\*\*(?:(?!\*/).)*\*/)\s*CUPTI_ACTIVITY_KIND_(\w+)\s*=\s*\d+",
            activity, flags=re.DOTALL,
        )
    }
    lines = ["// Generated from the CUPTI Activity API header.", ""]
    features = set()
    for kind, alias in RECORDS.items():
        types = set(re.findall(r"\bCUpti_Activity\w+\b", docs.get(kind, "")))
        if len(types) != 1:
            raise ValueError(f"CUPTI header does not identify a single record for {kind}: {sorted(types)}")
        record = types.pop()
        lines.append(f"using {alias} = {record};")
        end = re.search(rf"}}\s*{record}\s*;", declarations)
        if not end:
            raise ValueError(f"CUPTI record {record} is not declared in the installed headers")
        start = declarations.rfind("typedef struct", 0, end.start())
        body = declarations[start:end.start()]
        for field, feature in OPTIONAL_FIELDS.get(alias, {}).items():
            if re.search(rf"\b{field}\b", body):
                features.add(feature)
    for symbol, feature in OPTIONAL_SYMBOLS.items():
        if re.search(rf"\b{symbol}\b", declarations):
            features.add(feature)
    lines.extend(f"#define RATELPROF_CUPTI_{name}" for name in sorted(features))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("include_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        args.output.write_text(generate(args.include_dir), encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
