#!/usr/bin/env bash
set -euo pipefail

ratelprof=$(realpath "$(command -v "${1:-ratelprof}")")
source_tree=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
cd "$scratch"

mkdir -p lib/nested
cat > linked.cu <<'SOURCE'
extern "C" __global__ void linked_kernel(int *value) { *value = 73; }
extern "C" int leaf_anchor(void) { return 73; }
SOURCE
cat > bridge.c <<'SOURCE'
int leaf_anchor(void);
int bridge_anchor(void) { return leaf_anchor(); }
SOURCE
cat > application.c <<'SOURCE'
#include <stdio.h>
int bridge_anchor(void);
int main(void)
{
    FILE *sentinel = fopen("application-executed", "w");
    if (sentinel) fclose(sentinel);
    return bridge_anchor() != 73;
}
SOURCE

nvcc -shared -Xcompiler=-fPIC -Xlinker=-soname -Xlinker=liblinked.so \
    -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 \
    linked.cu -o lib/nested/liblinked.so
cc -shared -fPIC bridge.c -Llib/nested -llinked -Wl,-soname,libbridge.so \
    -Wl,-rpath,'${ORIGIN}/nested' -o lib/libbridge.so
cc application.c -Llib -lbridge -Wl,-rpath,'$ORIGIN/lib' -Wl,-rpath-link,lib/nested -o application

"$ratelprof" inspect --linked --archs sm_80,sm_90 --save-json -o linked.csv ./application
"$ratelprof" inspect --linked --archs sm_90 --save-json --sass --sass-summary \
    -o linked-90.csv ./application
test ! -e application-executed
python3 - <<'PYTHON'
import json
from pathlib import Path

both = json.loads(Path("linked.json").read_text())
selected = json.loads(Path("linked-90.json").read_text())
summary = json.loads(Path("linked-90.sass.json").read_text())
assert not both["unresolved_dependencies"], both["unresolved_dependencies"]
assert {row["architecture"] for row in both["kernels"]} == {"sm_80", "sm_90"}
assert all(row["source"].endswith("/liblinked.so") for row in both["kernels"])
assert {row["architecture"] for row in selected["kernels"]} == {"sm_90"}
assert {row["architecture"] for row in summary["kernels"]} == {"sm_90"}
assert any(row["symbol"] == "linked_kernel" and row["instruction_count"] > 0 for row in summary["kernels"])
PYTHON

nvcc -gencode arch=compute_90,code=compute_90 "$source_tree/tests/cuda/smoke.cu" -ldl -o jit
"$ratelprof" profile --trace gpu,cuda-runtime,nvtx --dump-cubins modules ./jit -- "argument with 'quotes' and spaces"
found_kernel=false
for image in modules/*.cubin; do
    test -s "$image"
    if "$ratelprof" inspect --archs sm_90,sm_90a --sass-summary -o jit-module.csv "$image" \
        && grep -q '_Z9incrementPi' jit-module.sass.json; then
        found_kernel=true
        break
    fi
done
test "$found_kernel" = true

# Synthetic counts identify an actual captured cubin and its first SASS instruction.
lua=$(command -v luajit || command -v lua)
lua_libraries=$(dirname -- "$ratelprof")/../lib/lua
crc=$(LUA_CPATH="$lua_libraries/?.so;;" "$lua" - "$image" <<'LUA'
print(assert(require("ratelprof_cubin").get_crc(arg[1])))
LUA
)
python3 - "$crc" <<'PYTHON'
import json
import sys
from pathlib import Path

fixture = {"cubin_crc": sys.argv[1], "function": "_Z9incrementPi", "pc_offset": 0,
           "stall_reasons": {"synthetic": 7}, "source": {"file": "test.cu", "line": 1}}
wrong_crc = fixture | {"cubin_crc": hex(int(sys.argv[1], 16) ^ 1), "stall_reasons": {"unexpected": 1}}
wrong_symbol = fixture | {"function": "_Z10differentPi", "stall_reasons": {"unexpected": 1}}
wrong_offset = fixture | {"pc_offset": 1, "stall_reasons": {"unexpected": 1}}
for path, rows in [("samples.json", [fixture, fixture, wrong_crc, wrong_symbol, wrong_offset]),
                   ("unmatched.json", [wrong_crc, wrong_symbol, wrong_offset])]:
    Path(path).write_text(json.dumps({"schema": "ratelprof-pcsampling-v1", "samples": rows}))
PYTHON
"$ratelprof" inspect --pc-samples samples.json -o annotated.csv "$image"
if "$ratelprof" inspect --pc-samples unmatched.json -o unmatched.csv "$image" > unmatched.log 2>&1; then
    printf 'The inspector unexpectedly matched unrelated PC samples.\n' >&2
    exit 1
fi
grep -q 'No PC samples matched' unmatched.log
python3 - "$crc" <<'PYTHON'
import json
import sys
from pathlib import Path

kernels = json.loads(Path("annotated.sass.json").read_text())["kernels"]
kernel, = (row for row in kernels if row["symbol"] == "_Z9incrementPi")
assert kernel["cubin_crc"] == sys.argv[1]
instruction, = kernel["sampled_instructions"]
assert instruction["pc_offset"] == 0
assert instruction["stall_reasons"] == {"synthetic": 14}
assert instruction["source"] == {"file": "test.cu", "line": 1}
PYTHON
printf 'Verified linked sm_80/sm_90 CUDA images, runtime JIT cubin and synthetic SASS annotations.\n'
