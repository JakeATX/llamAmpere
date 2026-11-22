#!/bin/bash
# DSPARKTRAIN rental runner (one 24 GB 3090-class box). Phases resume: each writes $R/<phase>.done.
# Layout (mkbundle.sh): $R/bin (shimmed llama-server, llama-dspark-dump), $R/wt/tools/dsparktrain, $R/wt/gguf-py,
#   $R/wt/ink_dspark_fold.py, $R/data/{prompts.jsonl,wiki.test.raw}, $R/w0, $R/model/{target,Q8_0,Q8_0-fold,BF16}.gguf;
#   python at /mnt/data/inkflags/venv (same absolute path as Taichi).
# Outputs: $R/data (gens, seq, dumps), $R/run (train log, best.pt), $R/out (eval JSONs, exported drafter, server checks).
# usage: run_rental.sh [phase ...]   (default: check gen seq dump base train rep export server)
set -uo pipefail
R=${R:-/root/dst}; T=$R/wt/tools/dsparktrain; D=$R/data; O=$R/out
PY=${PY:-/mnt/data/inkflags/venv/bin/python}
M=$R/model/Inkling-2-Mini-A_all_ait4u-fold.gguf
DQ=$R/model/Inkling-2-Mini-DSpark-Q8_0.gguf; DF=$R/model/Inkling-2-Mini-DSpark-Q8_0-fold.gguf; DB=$R/model/Inkling-2-Mini-DSpark-BF16.gguf
BUDGET=${BUDGET:-2000000}; NPRED=${NPRED:-4096}; MAXP=${MAXP:-2048}; NP=${NP:-10}; PORT=18480
STEPS=${STEPS:-1000}; LR=${LR:-3e-5}; LOSS=${LOSS:-e4}; KL=${KL:-0.1}; SPS=${SPS:-4}; APS=${APS:-512}
EVAL_EVERY=${EVAL_EVERY:-100}; EVAL_STRIDE=${EVAL_STRIDE:-8}; DTYPE=${DTYPE:-bf16}   # train dump dtype; sel/rep are f32
# KVF: target KV flags for gen, dumps and server checks (empty = default KV, the config of the 134.1 tok/s baseline;
# set e.g. KVF="-ctk sjkvarn4 -ctv sjkvarn4" to train/evaluate at the Inkling gate KV)
KVF=${KVF:-}
export PYTHONPATH=$R/wt/gguf-py LLAMA_ARG_NO_CACHE_DISK=1
mkdir -p $D $O $R/run
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
done_or() { [ -e $R/$1.done ] && { log "skip $1"; return 0; }; return 1; }
TRAINARGS="--w0 $R/w0 --drafter $DF --bf16 $DB"
DUMP="$R/bin/llama-dspark-dump -m $M -ngl 99 -fa on -t 8 -c 8192 -b 2048 -ub 512 $KVF"

ph_check() {
  done_or check && return
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader | tee -a $R/run.log
  $R/bin/llama-server --version >> $R/run.log 2>&1 || { log 'llama-server does not start (shim)'; return 1; }
  $PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" >> $R/run.log 2>&1 \
      || { log 'torch CUDA not usable'; return 1; }
  ( cd $R/w0 && sha256sum -c SHA256SUMS ) >> $R/run.log 2>&1 || { log 'w0 checksum mismatch'; return 1; }
  df -h $R | tail -1 | tee -a $R/run.log
  touch $R/check.done; log 'check ok'
}

ph_gen() {
  done_or gen && return
  $R/bin/llama-server -m $M --host 127.0.0.1 --port $PORT -c $((NP * (MAXP + NPRED + 64))) -np $NP -ngl 99 -fa on -t 8 -tb 8 --jinja --no-webui $KVF \
      > $R/gen_server.log 2>&1 &
  local sp=$!
  for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
  curl -sf http://127.0.0.1:$PORT/health > /dev/null || { log 'gen server did not come up'; kill $sp; return 1; }
  local rc=1
  for try in 1 2 3; do   # gen.py resumes; a retry only redoes failed prompts
    $PY $T/gen.py --prompts $D/prompts.jsonl --out $D/gens.jsonl --port $PORT --workers $NP --npred $NPRED --max-prompt $MAXP \
        --budget $BUDGET >> $R/gen.log 2>&1
    rc=$?; [ $rc = 0 ] && break; log "gen try $try rc $rc"
  done
  kill $sp; wait $sp 2>/dev/null
  [ $rc = 0 ] && touch $R/gen.done && log "gen ok $(tail -1 $R/gen.log)"
}

ph_seq() { done_or seq && return; $PY $T/mkseq.py $D/gens.jsonl $D/seq > $R/seq.log 2>&1 && touch $R/seq.done && log "seq $(tr '\n' ' ' < $R/seq.log)"; }

ph_dump() {
  done_or dump && return
  local ntok=$($PY -c "import json, glob; print(sum(m['len'] for f in glob.glob('$D/seq/*.seq.json') for m in json.load(open(f))))")
  local need=$(( ntok * 4 * 2304 * 2 / 1000000000 * 115 / 100 + 5 ))   # bf16 train features (+ f32 sel/rep, top-64 rows)
  local free=$(df -BG --output=avail $D | tail -1 | tr -dc 0-9)
  (( free > need )) || { log "dump needs ~${need} GB, ${free} GB free"; return 1; }
  for s in sel rep train; do
    [ -e $D/$s.dump.done ] && continue
    local dt=f32; [ $s = train ] && dt=$DTYPE
    $DUMP --ds-in $D/seq/$s.seq --ds-out $D/$s --ds-dtype $dt > $R/dump_$s.log 2>&1 || { log "dump $s failed"; return 1; }
    touch $D/$s.dump.done; log "dump $s $(grep -E 'seqs,' $R/dump_$s.log | tail -1)"
  done
  touch $R/dump.done
}

ph_base() {
  done_or base && return
  $PY $T/train.py $TRAINARGS --out $O/base --eval-only --sel $D/sel:$D/seq/sel.seq --rep $D/rep:$D/seq/rep.seq --shipped-fold > $R/base.log 2>&1 \
      && touch $R/base.done && log "base $(grep '"eval"' $R/base.log | tr '\n' ' ' | cut -c1-600)"
}

ph_train() {
  done_or train && return
  $PY $T/train.py $TRAINARGS --out $R/run --train $D/train:$D/seq/train.seq --sel $D/sel:$D/seq/sel.seq --steps $STEPS --lr $LR --loss $LOSS \
      --kl $KL --seqs-per-step $SPS --anchors-per-seq $APS --eval-every $EVAL_EVERY --eval-stride $EVAL_STRIDE > $R/train.log 2>&1 \
      && touch $R/train.done && log "train ok $(grep saved $R/run/log.jsonl | tail -1)"
}

ph_rep() {
  # report numbers: best checkpoint vs shipped on the rep split (never used for selection), every anchor
  done_or rep && return
  grep -q saved $R/run/log.jsonl || { log 'NO_IMPROVEMENT: no checkpoint beat step 0 on sel'; touch $R/rep.done; return 0; }
  $PY $T/train.py $TRAINARGS --out $O/best --eval-only --init $R/run/best.pt --rep $D/rep:$D/seq/rep.seq > $R/rep.log 2>&1 \
      && touch $R/rep.done && log "rep best $(tail -1 $R/rep.log | cut -c1-600)"
}

ph_export() {
  done_or export && return
  [ -e $R/run/best.pt ] && grep -q saved $R/run/log.jsonl || { log 'nothing to export'; touch $R/export.done; return 0; }
  $PY $T/export.py $R/run/best.pt $DQ $DB $R/w0 $O/dspark-new --fold-script $R/wt/ink_dspark_fold.py > $R/export.log 2>&1 \
      && touch $R/export.done && log "export $(tail -1 $R/export.log | cut -c1-400)"
}

ph_server() {
  # quick server check (not a gate: 2 seeds x 3,072 tokens; the gate is ABBA >= 6 seeds x 5,120): bench prompt (3K chat),
  # shipped vs new, exact and mb, depth 4, each with the offline metric on the same sequences
  done_or server && return
  local arms="ship:$DF"; [ -e $O/dspark-new-Q8_0-fold.gguf ] && arms="$arms new:$O/dspark-new-Q8_0-fold.gguf"
  for a in $arms; do
    local n=${a%%:*} f=${a#*:}
    for mode in exact mb; do
      local x=""; [ $mode = exact ] && x=--exact
      $PY $T/spec_check.py --bin $R/bin --model $M --drafter $f --out $O/srv_${n}_$mode $x --seeds 1,2 --npred 3072 --wiki $D/wiki.test.raw \
          --port $((PORT + 1)) --extra="$KVF" > $O/srv_${n}_$mode.log 2>&1
      log "server $n $mode $(tail -1 $O/srv_${n}_$mode.log | cut -c1-400)"
      # calibration: the offline metric on exactly the sequences this server run produced (f32 features)
      $DUMP --ds-in $O/srv_${n}_$mode/calib.seq --ds-out $O/srv_${n}_$mode/d --ds-dtype f32 > $O/srv_${n}_$mode/dump.log 2>&1
      local ev="--shipped-fold"; [ $n = new ] && ev="--init $R/run/best.pt"
      $PY $T/train.py $TRAINARGS --out $O/cal_${n}_$mode --eval-only $ev --rep $O/srv_${n}_$mode/d:$O/srv_${n}_$mode/calib.seq > $O/cal_${n}_$mode.log 2>&1
      log "calib $n $mode server mean_len $($PY -c "import json; print(json.load(open('$O/srv_${n}_$mode/result.json'))['mean_len'])") offline $(tail -1 $O/cal_${n}_$mode.log | $PY -c "import json,sys; d=json.loads(sys.stdin.read()); print('tok_round_sim', d['tok_round_sim'], 'mb', d['tok_round_sim_mb'], 'E', d['E'][-1], 'E_mb', d['E_mb'][-1])")"
    done
  done
  # parity of the exported drafter against torch, on the first server sequence
  if [ -e $O/dspark-new-Q8_0-fold.gguf ]; then
    $DUMP -md $O/dspark-new-Q8_0-fold.gguf -ngld 99 --ds-in $O/srv_new_exact/calib.seq --ds-out $O/par_new --ds-dtype f32 --ds-parity 48 > $O/par_new.log 2>&1
    $PY $T/parity.py --par $O/par_new --w0 $R/w0 --drafter $O/dspark-new-Q8_0-fold.gguf --bf16 $O/dspark-new-BF16.gguf \
        --seq $O/srv_new_exact/calib.seq --out $O/parity_new.json > /dev/null 2>&1
    log "parity new $($PY -c "import json; d=json.load(open('$O/parity_new.json'))['shipped_fold_actq1']; print({k: d[k] for k in ('meandiff', 'argmax_agree', 'B_id_agree')})")"
  fi
  touch $R/server.done
}

PHASES=${*:-check gen seq dump base train rep export server}
for p in $PHASES; do ph_$p || { log "PHASE $p FAILED"; exit 1; }; done
log ALL_DONE
