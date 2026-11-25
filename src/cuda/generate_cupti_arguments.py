"""Generate typed CUDA callback argument readers from the installed CUPTI ABI."""

import argparse
import json
import re
from pathlib import Path


PARAMETERS = re.compile(r"\btypedef\s+struct\s+\w+\s*\{([^{}]*)\}\s*(\w+_params)\s*;")
FIELD = re.compile(r"(.+?)(\b\w+)(?:\s*\[[^\]]*\])?$")

DOMAINS = (
    ("DRIVER", "cupti_driver_cbid.h", "generated_cuda_meta.h"),
    ("RUNTIME", "cupti_runtime_cbid.h", "generated_cuda_runtime_api_meta.h"),
)

# Preserve the keys used by reports written before full argument capture was
# available. The full argument list also retains the spelling from the SDK.
ALIASES = {
    "cudaMalloc_v3020": (("bytes", "size_t", "size"),),
    "cudaMemcpy_v3020": (("bytes", "size_t", "count"),),
    "cudaMemcpyAsync_v3020": (("bytes", "size_t", "count"),),
    "cudaMemsetAsync_v3020": (("bytes", "size_t", "count"),),
}


def field_declarations(body):
    fields = []
    for declaration in body.split(";"):
        declaration = " ".join(declaration.split())
        if not declaration:
            continue
        field = FIELD.fullmatch(declaration)
        if not field:
            raise ValueError(f"unrecognized CUPTI parameter declaration: {declaration}")
        datatype, name = field.groups()
        fields.append((name, datatype.rstrip(), name))
    return fields


def sdk_readers(directory, domain, enum_file, parameters_file):
    enum = (directory / enum_file).read_text()
    meta = (directory / parameters_file).read_text()
    strips_comments = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
    records = {name: body for body, name in PARAMETERS.findall(strips_comments.sub("", meta))}
    prefix = f"CUPTI_{domain}_TRACE_CBID_"
    callbacks = re.findall(r"\b(" + prefix + r"\w+)\s*=", enum)
    readers = []
    for callback in callbacks:
        function = callback[len(prefix):]
        params = function + "_params"
        if params in records:
            fields = field_declarations(records[params])
            readers.append((callback, params, fields + list(ALIASES.get(function, ()))))
    return readers


def generate(directory):
    yield "// Generated from the CUPTI headers selected by CMake."
    for domain, enum, meta in DOMAINS:
        readers = sdk_readers(directory, domain, enum, meta)
        if not readers:
            raise ValueError(f"no {domain} callback metadata found in {directory}")
        print(f"CUPTI {domain}: {len(readers)} callback signatures")
        yield f"    if (domain == CUPTI_CB_DOMAIN_{domain}_API) {{"
        yield "        switch (callback) {"
        for callback, params, fields in readers:
            yield f"        case {callback}: {{"
            if fields:
                yield "            static constexpr FieldReader fields[] = {"
                for name, datatype, member in fields:
                    yield (f"                {{{json.dumps(name)}, {json.dumps(datatype)}, "
                           f"offsetof({params}, {member}), read_value<decltype({params}::{member})>}},")
                yield "            };"
                yield "            return {fields, sizeof(fields) / sizeof(*fields)};"
            else:
                yield "            return {};"
            yield "        }"
        yield "        default: return {};"
        yield "        }"
        yield "    }"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("headers", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text("\n".join(generate(args.headers)) + "\n")


if __name__ == "__main__":
    main()
