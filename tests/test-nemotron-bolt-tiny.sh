#!/bin/bash
# Smoke test for the Nemotron 3.5 Bolt layout (per-layer expert counts/widths) on a tiny random model.
# usage: tests/test-nemotron-bolt-tiny.sh BIN_DIR ["KV types", default "f16 q8_0"] [extra llama args, default "-ngl 0"]
# Checks: the model loads and generates with each KV type, temp-0 output is repeatable, and a llama-server
# slot save -> erase -> restore round trip reproduces the same temp-0 continuation (prompt caching must work).
set -euo pipefail
BIN=${1:?usage: $0 BIN_DIR [KV types] [extra args]}
KVS=${2:-"f16 q8_0"}
EXTRA=${3:-"-ngl 0"}
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'kill ${SP:-} 2>/dev/null || true; [ -n "${KEEP:-}" ] || rm -rf "$TMP"' EXIT
python3 "$HERE/nemotron_bolt_tiny_gguf.py" "$TMP/tiny.gguf" >/dev/null
PROMPT="The quick brown fox jumps over the lazy dog. Once upon a time, in a land far away, there lived"
fail=0
for kv in $KVS; do
    run() { "$BIN/llama-completion" -m "$TMP/tiny.gguf" -p "$PROMPT" -n 300 --temp 0 -fa on -c 1024 \
            -ctk "$kv" -ctv "$kv" $EXTRA -no-cnv --no-warmup --simple-io 2>"$TMP/$kv.err"; }
    a=$(run) || { echo "FAIL load/generate kv=$kv"; tail -5 "$TMP/$kv.err"; fail=1; continue; }
    b=$(run)
    if [ "$a" != "$b" ] || [ ${#a} -le ${#PROMPT} ]; then echo "FAIL repeatable kv=$kv"; fail=1; else echo "ok generate kv=$kv (${#a} chars)"; fi
done
# slot save/restore round trip on the first KV type
kv=${KVS%% *}
PORT=$((18000 + RANDOM % 2000))
mkdir -p "$TMP/slots"
"$BIN/llama-server" -m "$TMP/tiny.gguf" -c 2048 -np 1 -fa on -ctk "$kv" -ctv "$kv" $EXTRA --port $PORT \
    --slot-save-path "$TMP/slots" >"$TMP/server.log" 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -sf "localhost:$PORT/health" >/dev/null && break; sleep 0.5; done
python3 - "$PORT" "$PROMPT" <<'PY' || fail=1
import json, sys, urllib.request
port, prompt = sys.argv[1], sys.argv[2] * 8
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
gen = lambda: post("/completion", {"prompt": prompt, "n_predict": 64, "temperature": 0, "top_k": 1, "cache_prompt": True, "id_slot": 0})
a = gen()
post("/slots/0?action=save", {"filename": "t.bin"})
post("/slots/0?action=erase", {})
post("/slots/0?action=restore", {"filename": "t.bin"})
b = gen()
ok = a["content"] == b["content"]
print(("ok" if ok else "FAIL") + f" slot save/restore (prompt_n first {a['timings']['prompt_n']}, after restore {b['timings']['prompt_n']})")
sys.exit(0 if ok else 1)
PY
[ $fail -eq 0 ] && echo "ALL OK" || { echo "SOME FAILED"; exit 1; }
