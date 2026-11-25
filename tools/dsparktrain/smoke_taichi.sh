#!/bin/bash
# DSPARKTRAIN smoke on Taichi GPU1 (run through /mnt/data/inkspec2/h/gpu1run.sh, under gpu1.lock), hard cap 590 s.
# Order = priority: server calibration (exact, mb) -> calib dumps + parity -> offline eval -> train profile ->
# gen throughput -> train dump throughput -> export + server load check of the exported drafter.
set -u
[[ ${PARENT_HOLDS_GPU_LOCK:-0} = 1 ]] || { echo "run under gpu1run.sh"; exit 9; }
for v in $(env | grep -oE '^(GGML|LLAMA|TURBO)_[A-Z0-9_]*'); do unset "$v"; done
export CUDA_VISIBLE_DEVICES=1 LLAMA_ARG_NO_CACHE_DISK=1
T=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-/mnt/data/inkspec2/wt-train/build/bin}; PY=${PY:-/mnt/data/inkflags/venv/bin/python}
M=${MODEL:-/mnt/data/inkflags/enc/Inkling-2-Mini-A_all_ait4u-fold.gguf}
DQ=/mnt/data/inkling2/Inkling-2-Mini-DSpark-Q8_0.gguf; DF=/mnt/data/inkling2/Inkling-2-Mini-DSpark-Q8_0-fold.gguf
DB=/mnt/data/inkling2/Inkling-2-Mini-DSpark-BF16.gguf; W0=${W0:-/mnt/data/inkspec2/w0}
O=${OUT:-/mnt/data/inkspec2/smoke}; mkdir -p $O; cd $O
export LD_LIBRARY_PATH=$BIN PYTHONPATH=$T/../../gguf-py
T0=$(date +%s)
st() { echo "STEP $1 t=$(( $(date +%s) - T0 ))s $(date -u +%H:%M:%S)"; }
left() { echo $(( 590 - ($(date +%s) - T0) )); }
run() { local lim=$(left); (( lim > 5 )) || { echo "SKIP $* (time cap)"; return 1; }; timeout $lim "$@"; }
nvidia-smi --id=1 --query-gpu=name,power.limit,memory.used --format=csv,noheader; free -g | head -2; df -h / | tail -1
DUMP="$BIN/llama-dspark-dump -m $M -ngl 99 -fa on -t 8"

st calib_exact; run $PY $T/spec_check.py --bin $BIN --model $M --drafter $DF --out $O/cal_exact --exact --npred 2048 --port 18641 | tail -2
st calib_mb;    run $PY $T/spec_check.py --bin $BIN --model $M --drafter $DF --out $O/cal_mb --npred 2048 --port 18642 | tail -2
st dump_calib
for c in exact mb; do
  [[ -f $O/cal_$c/calib.seq ]] || continue
  run $DUMP -c 8192 -b 2048 -ub 512 --ds-in $O/cal_$c/calib.seq --ds-out $O/cal_$c/d --ds-dtype f32 2>&1 | grep -E "dspark-dump|rror" | tail -2
done
st parity
run $DUMP -md $DF -ngld 99 -c 8192 -b 2048 -ub 512 --ds-in $O/cal_exact/calib.seq --ds-out $O/par --ds-dtype f32 --ds-parity 48 2>&1 | grep -E "parity done|rror" | tail -2
st parity_py
run $PY $T/parity.py --par $O/par --w0 $W0 --drafter $DF --bf16 $DB --seq $O/cal_exact/calib.seq --refold --out $O/parity.json > $O/parity.log 2>&1; tail -c 300 $O/parity.log; echo
st eval_calib
for c in exact mb; do
  [[ -f $O/cal_$c/d.idx ]] || continue
  run $PY $T/train.py --w0 $W0 --drafter $DF --bf16 $DB --out $O/ev_$c --eval-only --rep $O/cal_$c/d:$O/cal_$c/calib.seq --shipped-fold | tail -1
  run $PY $T/train.py --w0 $W0 --drafter $DF --bf16 $DB --out $O/ev_${c}_noact --eval-only --rep $O/cal_$c/d:$O/cal_$c/calib.seq --shipped-fold --no-act-q | tail -1
done
st gen
if [[ $(left) -gt 200 ]]; then
  $BIN/llama-server -m $M --host 127.0.0.1 --port 18643 -c 65536 -np 16 -ngl 99 -fa on -t 8 -tb 8 --jinja --no-webui > $O/gen_server.log 2>&1 &
  SP=$!
  for i in $(seq 120); do curl -sf http://127.0.0.1:18643/health >/dev/null && break; sleep 1; done
  run timeout 75 $PY $T/gen.py --prompts /mnt/data/inkspec2/data/prompts.jsonl --out $O/gens.jsonl --port 18643 --workers 16 --npred 1024 --splits train --limit 48 --budget 40000 | tail -3
  kill $SP; wait $SP 2>/dev/null
fi
st dump_train
if [[ -s $O/gens.jsonl ]]; then
  $PY $T/mkseq.py $O/gens.jsonl $O/seq
  [[ -f $O/seq/train.seq ]] && run $DUMP -c 8192 -b 2048 -ub 512 --ds-in $O/seq/train.seq --ds-out $O/tr --ds-dtype bf16 2>&1 | grep -E "dspark-dump: [0-9]+/|rror" | tail -2
fi
TRS=$O/cal_exact/d:$O/cal_exact/calib.seq
[[ -f $O/tr.idx ]] && TRS=$O/tr:$O/seq/train.seq
st train_profile
run $PY $T/train.py --w0 $W0 --drafter $DF --bf16 $DB --out $O/prof --train $TRS --profile 6 --seqs-per-step 2 --anchors-per-seq 512 --lr 3e-5 --warmup 1 | tail -8
st train_profile_bf16base
run $PY $T/train.py --w0 $W0 --drafter $DF --bf16 $DB --out $O/prof_bf --train $TRS --profile 3 --seqs-per-step 2 --anchors-per-seq 512 --bf16-base --lr 3e-5 --warmup 1 | tail -2
st export
[[ -f $O/prof/last.pt ]] && run $PY $T/export.py $O/prof/last.pt $DQ $DB $W0 $O/exp | tail -1
st export_server
[[ -f $O/exp-Q8_0-fold.gguf ]] && run $PY $T/spec_check.py --bin $BIN --model $M --drafter $O/exp-Q8_0-fold.gguf --out $O/cal_exp --npred 512 --port 18644 | tail -1
st end
