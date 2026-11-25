#!/bin/bash
# MTPRETRAIN QAT-from-step-1 run (Jake 11-22: train at ship precision). Reuses gens.jsonl from the first rental; re-dumps
# trunk states + targets (those were not kept). Phases resumable via $R/<phase>.done. Layout as run_rental.sh.
#   q5: shipped types (Q5_0 matrices, Q8_0 k/v), 2 epochs from the in-file head  -> mtp-qat-q5.gguf (296.0 MB)
#   q8: Q5_0 matrices retyped to Q8_0, 1 epoch from the in-file head             -> mtp-qat-q8.gguf (451.3 MB)
# Selection and every reported number use the fake-quantized forward (= the exported bytes), heldout_sel to pick,
# heldout_rep to report.
set -uo pipefail
R=${R:-/root/mtp}; T=$R/wt/tools/mtpretrain; D=$R/data; M=$R/model/P76.gguf; PY=$R/venv/bin/python3
export LD_LIBRARY_PATH=$R/bin:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
done_or() { [ -e $R/$1.done ] && { log "skip $1"; return 0; }; return 1; }
run() { local ph=$1; shift; done_or $ph && return 0; "$@" && touch $R/$ph.done && log "$ph ok" || { log "PHASE $ph FAILED"; exit 1; }; }

run extract bash -c "$PY $T/extract.py $M $R/w0 > $R/extract.log 2>&1"
run dump bash -c "$PY $T/mkseq.py $D/gens.jsonl $D > $R/mkseq.log 2>&1 && $R/bin/llama-mtp-dump-h -m $M -c 6144 -b 2048 -ub 2048 -fa on \
  -ctk sjkvarn4 -ctv sjkvarn4 -ngl 99 --mtp-in $D/seq.bin --mtp-out $D/h.bin > $R/dump.log 2>&1"
run targets bash -c "$PY $T/mtp_train.py targets --w $R/w0 --data $D > $R/targets.log 2>&1"
run base_rep bash -c "MTP_QAT=1 $PY $T/mtp_train.py eval --w $R/w0 --data $D --split heldout_rep --json-out $R/eval_base_rep.json > $R/eval_base_rep.log 2>&1"

# lr: every weight starts at the centre of its quant bucket and Adam moves ~lr per step, so the coarse Q5_0 grid
# (step ~4e-3) gets 6e-5 to let codes change within 136 steps; Q8_0 (8x finer) keeps 3e-5. codes_changed is logged.
for v in q5:q5_0:2:6e-5:  q8:q8_0:1:3e-5:--retype=q5_0=q8_0; do
  IFS=: read -r tag mats ep lr retype <<< "$v"
  [ -e $R/train_$tag.done ] || rm -rf $R/run_$tag  # a crashed train restarts at step 0: never keep its old head_best.pt
  run train_$tag bash -c "MTP_QAT=1 MTP_QAT_MATS=$mats $PY $T/mtp_train.py train --w $R/w0 --data $D --out $R/run_$tag --lr $lr \
    --epochs $ep --eval-every 25 > $R/train_$tag.log 2>&1"
  [ -e $R/run_$tag/head_best.pt ] || { log "$tag NO_IMPROVEMENT over the in-file head"; continue; }
  run export_$tag bash -c "$PY $T/export_overlay.py --w $R/w0 --head $R/run_$tag/head_best.pt --out $R/mtp-qat-$tag.gguf $retype > $R/export_$tag.log 2>&1"
  # report number from the exported file itself (dequantized), not the fake-quant forward: they must agree
  run rep_$tag bash -c "$PY $T/mtp_train.py eval --w $R/w0 --data $D --split heldout_rep --head-gguf $R/mtp-qat-$tag.gguf \
    --json-out $R/eval_qat_${tag}_rep.json > $R/eval_qat_${tag}_rep.log 2>&1"
  log "rep $tag $(tail -1 $R/eval_qat_${tag}_rep.log | cut -c1-400)"
done
log QATFULL_DONE
