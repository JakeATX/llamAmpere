#!/bin/bash
set -euo pipefail
# Invoke inside the normal low-weight, locked FNPREFETCH build unit.
python3 - <<'PY'
from pathlib import Path
cgroup=Path('/proc/self/cgroup').read_text()
assert '/llama.cpp-fnprefetch-build-' in cgroup, 'use an owned locked build unit'
PY
cd /mnt/models/fnprefetch
baseline_commit=3f372a1ba
baseline_lib=/mnt/models/fnship/releases/$baseline_commit/bin
headers=/mnt/models/fnprefetch/baseline-headers-$baseline_commit
mkdir -p "$headers" protocol
git archive "$baseline_commit" common include ggml/include | tar -x -C "$headers"
c++ -std=c++17 -O2 -I"$headers/common" -I"$headers/include" -I"$headers/ggml/include" tests/test-moe-prefetch-model.cpp -o protocol/llama.cpp-prefetch-baseline-3f -L"$baseline_lib" -Wl,-rpath,"$baseline_lib" -lllama-common -lllama -lggml -lggml-base -lggml-cpu -pthread
python3 - <<'PY'
import json,struct
from pathlib import Path
ids=json.loads(Path('/mnt/models/stratamatch-home/prompt-ids.json').read_text())
Path('protocol/golden-ids.bin').write_bytes(struct.pack('<2080i',*ids[:2080]))
PY
