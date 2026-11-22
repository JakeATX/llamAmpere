#!/bin/bash
# MTPRETRAIN rental runner (one 3090-class box). Phases are resumable: each writes $R/<phase>.done.
# Layout: $R/bin (v0.7.2.1 release llama-server + libs + llama-mtp-dump-h), $R/model/P76.gguf, $R/wt/tools/mtpretrain (this dir),
# $R/wt/gguf-py, $R/data (prompts.jsonl, s24.txt, s64.txt). Outputs: $R/data (gens, seq, h, targets), $R/w0, $R/run.
# usage: run_rental.sh [phase ...]   (default: all phases in order)
set -uo pipefail
R=${R:-/root/mtp}; T=$R/wt/tools/mtpretrain; D=$R/data; M=$R/model/P76.gguf; PY=${PY:-$R/venv/bin/python3}
BUDGET=${BUDGET:-2000000}; NPRED=${NPRED:-2048}; NP=${NP:-8}; PORT=18480
LR=${LR:-3e-5}; EPOCHS=${EPOCHS:-2}
export LD_LIBRARY_PATH=$R/bin:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
done_or() { [ -e $R/$1.done ] && { log "skip $1"; return 0; }; return 1; }

ph_extract() { done_or extract && return; $PY $T/extract.py $M $R/w0 > $R/extract.log 2>&1 && touch $R/extract.done && log extract ok; }

ph_gen() {
  done_or gen && return
  LLAMA_ARG_NO_CACHE_DISK=1 $R/bin/llama-server -m $M --preset 24dense --parallel $NP -c $((NP * 8192)) --no-spec-mb-mtp ${GEN_EXTRA:-} \
      --port $PORT --host 127.0.0.1 --no-webui > $R/gen_server.log 2>&1 &
  local sp=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:$PORT/health > /dev/null || { log 'gen server did not come up'; kill $sp; return 1; }
  nvidia-smi --query-gpu=name,memory.used,power.limit --format=csv,noheader >> $R/run.log
  local rc=1
  for try in 1 2 3; do  # gen.py resumes; a retry only redoes failed prompts
    $PY $T/gen.py --prompts $D/prompts.jsonl --out $D/gens.jsonl --port $PORT --workers $NP --npred $NPRED --budget $BUDGET >> $R/gen.log 2>&1
    rc=$?; [ $rc = 0 ] && break; log "gen try $try rc $rc"
  done
  kill $sp; wait $sp 2>/dev/null
  [ $rc = 0 ] && grep -q GEN_DONE $R/gen.log && touch $R/gen.done && log "gen ok $(tail -1 $R/gen.log)"
}

ph_dump() {
  done_or dump && return
  $PY $T/mkseq.py $D/gens.jsonl $D > $R/mkseq.log 2>&1 || return 1
  $R/bin/llama-mtp-dump-h -m $M -c 6144 -b 2048 -ub 2048 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -ngl 99 \
      --mtp-in $D/seq.bin --mtp-out $D/h.bin > $R/dump.log 2>&1 && touch $R/dump.done && log "dump ok $(tail -1 $R/dump.log)"
}

ph_targets() { done_or targets && return; $PY $T/mtp_train.py targets --w $R/w0 --data $D > $R/targets.log 2>&1 && touch $R/targets.done && log "targets ok $(tail -1 $R/targets.log)"; }

ph_base() { done_or base && return; $PY $T/mtp_train.py eval --w $R/w0 --data $D --split heldout_sel --json-out $R/eval_base.json > $R/eval_base.log 2>&1 && touch $R/base.done && log "base $(tail -1 $R/eval_base.log)"; }

ph_train() {
  done_or train && return
  $PY $T/mtp_train.py train --w $R/w0 --data $D --out $R/run --lr $LR --epochs $EPOCHS --eval-every 25 > $R/train.log 2>&1 \
      && touch $R/train.done && log "train ok $(grep BEST $R/run/train.log | tail -1)"
}

ph_export() {
  done_or export && return
  [ -e $R/run/head_best.pt ] || { log 'NO_IMPROVEMENT: no checkpoint beat step 0 on heldout_sel'; touch $R/export.done; return 0; }
  $PY $T/export_overlay.py --w $R/w0 --head $R/run/head_best.pt --out $R/mtp-otp.gguf > $R/export.log 2>&1 || return 1
  # go/no-go numbers: requantized new head vs the in-file head on heldout_rep (never used for checkpoint selection)
  $PY $T/mtp_train.py eval --w $R/w0 --data $D --split heldout_rep --json-out $R/eval_base_rep.json > $R/eval_base_rep.log 2>&1 || return 1
  $PY $T/mtp_train.py eval --w $R/w0 --data $D --split heldout_rep --head-gguf $R/mtp-otp.gguf --json-out $R/eval_otp_rep.json > $R/eval_otp_rep.log 2>&1 \
      || return 1
  log "rep base $(tail -1 $R/eval_base_rep.log)"; log "rep otp  $(tail -1 $R/eval_otp_rep.log)"
  # the overlay must load in the release server (15 tensors replaced) and draft
  LLAMA_MODEL_OVERLAY_PATCH=$R/mtp-otp.gguf LLAMA_ARG_NO_CACHE_DISK=1 timeout 600 $R/bin/llama-server -m $M --preset 24dense -c 16384 -lv 4 \
      --port $PORT --host 127.0.0.1 --no-webui > $R/ovl_server.log 2>&1 &
  local sp=$!
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
  curl -sf http://127.0.0.1:$PORT/completion -d '{"prompt":"Write a short Python function that reverses a list.","n_predict":128}' > $R/ovl_reply.json
  kill $sp; wait $sp 2>/dev/null
  grep -q 'overlay patch .*: 15 tensors replaced' $R/ovl_server.log || { log 'OVERLAY NOT LOADED'; return 1; }
  touch $R/export.done; log "export ok; overlay loads (15 tensors replaced)"
}

PHASES=${*:-extract gen dump targets base train export}
for p in $PHASES; do ph_$p || { log "PHASE $p FAILED"; exit 1; }; done
log ALL_DONE
