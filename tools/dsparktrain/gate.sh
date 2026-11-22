#!/bin/bash
# DSPARKQAT server gate: shipped vs retrained DSpark drafter (both Q8_0 fold, 453,522,464 B), one card, one bin.
# Rows: 100K fill = HEADLINE, then 32K and 3K (secondary). Main KV SJ-KVaRN 4/4 (explicit), drafter KV = bin default.
# Protocol: inkbench.py speed mode, chat template, temp 1 / top_k 20 / top_p 0.95, >= 5,120-token reply (follow-up turns
# on EOS), one boot per (arm, seed), ABBA over seeds 1..6, DSpark d4 (--spec-draft-n-max 4, the Inkling mb default),
# LLAMA_DEEPCTX_LADDER=1, -b 4096, -ub = highest of 4096/2048/1024 that boots AND prefills that row (probed once per row,
# same ub for both arms; the effective ub is printed from the server log).
# usage: gate.sh OUTDIR SHIP.gguf NEW.gguf [rows (default "100k 32k 3k")]
set -u
O=$1; S=$2; N=$3; ROWS=${4:-100k 32k 3k}
R=${R:-/root/dst}; BIN=${BIN:-$R/bin}; M=${M:-$R/model/Inkling-2-Mini-A_all_ait4u-fold.gguf}
PY=${PY:-/mnt/data/inkflags/venv/bin/python}; IB=${IB:-$R/wt/inkbench.py}; SEEDS=${SEEDS:-1 2 3 4 5 6}
KVF=${KVF:--ctk sjkvarn4 -ctv sjkvarn4}; G=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$G INK_GPU=$G
mkdir -p $O
for f in $S $N; do sz=$(stat -c %s $f); [ "$sz" = 453522464 ] || { echo "GATE_ABORT $f is $sz B, not the shipped 453,522,464 B Q8_0 fold"; exit 3; }; done
declare -A PLEN=([100k]=100000 [32k]=32000 [3k]=3000) CTX=([100k]=110592 [32k]=40960 [3k]=16384)
ENVS="LLAMA_ARG_NO_CACHE_DISK=1 LLAMA_DEEPCTX_LADDER=1 LLAMA_SPEC_ACCPOS=1"
ext() { echo "$KVF -ub $1 -b 4096 --model-draft $2 --spec-type draft-dspark -ngld 99 -ubd 64 --spec-draft-n-max 4"; }
nvidia-smi --id=$G --query-gpu=name,power.limit,driver_version --format=csv,noheader | sed 's/^/GPU /'; date -u
for row in $ROWS; do
  ub=""
  for u in 4096 2048 1024; do   # probe: boot + full cold prefill + 64 tokens with the shipped drafter
    $PY $IB --bin $BIN --model $M --out $O/probe_${row}_ub$u --mode speed --plen ${PLEN[$row]} --npred 64 --ctx ${CTX[$row]} --seeds 1 \
        --port 18791 --reuse --env "$ENVS" --extra "$(ext $u $S)" > $O/probe_${row}_ub$u.log 2>&1
    if grep -q '"error": null' $O/probe_${row}_ub$u/result.json 2>/dev/null; then ub=$u; break; fi
    echo "PROBE $row ub $u failed: $(grep -a -m1 -iE 'out of memory|failed to allocate|cudaMalloc|error' $O/probe_${row}_ub$u/server.log | cut -c1-200)"
  done
  [ -n "$ub" ] || { echo "GATE_ABORT row $row boots at no ub"; exit 4; }
  eff=$(grep -a -m1 -oE 'n_ubatch *= *[0-9]+' $O/probe_${row}_ub$ub/server.log)
  echo "ROWCFG $row plen ${PLEN[$row]} ctx ${CTX[$row]} -ub $ub -b 4096 (server log: $eff; ladder: $(grep -a -m1 -oiE 'ladder[^,;]*' $O/probe_${row}_ub$ub/server.log | cut -c1-120))"
  for s in $SEEDS; do
    if (( s % 2 )); then ord="S N"; else ord="N S"; fi
    for a in $ord; do
      f=$S; [ $a = N ] && f=$N
      d=$O/$row.$a.s$s
      $PY $IB --bin $BIN --model $M --out $d --mode speed --plen ${PLEN[$row]} --npred 5120 --ctx ${CTX[$row]} --seeds $s --port 18792 \
          --reuse --env "$ENVS" --extra "$(ext $ub $f)" > $d.log 2>&1
      echo "G $row $a s$s $(grep -E '^ROW|^FAIL' $d.log | tr '\n' ' ') $($PY -c "import json; r=json.load(open('$d/result.json')); s=r.get('spec') or {}; print('tok/round', round(s.get('tok_per_round') or 0, 3), 'acc', round(s.get('acceptance_rate') or 0, 4), 'err', r.get('error'))" 2>&1 | tail -1)"
    done
  done
done
$PY $R/wt/tools/dsparktrain/gate_sum.py $O | tee $O/summary.txt
date -u; echo GATE_DONE
