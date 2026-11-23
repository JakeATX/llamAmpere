#!/bin/bash
# DSPARKQAT long-context retrain runner (one Colab L4 / 24 GB box). Target = NQ (24 GB ship file), main KV SJ-KVaRN 4/4,
# -np 1 -b 2048 -ub 1024 (24ink values), ladder on. Phases resume: each writes $R/L.<phase>.done.
#   check   : bins, torch, w0, NQ boots (via selrep)
#   selrep  : bench prompts (inkbench TASK + wikitext-2 test, as the 3090 gate) with the shipped drafter, d4 mb, seeds >= 101
#             (the gate uses 1..6): sel = 100K x2 + 32K x2, rep = 100K, 32K, 3K x2. Never used for training.
#   lgen    : ~40 long-context replies (~92-99K fill) on LongBench-v2 docs (prep_long.py), NQ target + shipped drafter in
#             EXACT mode (= the target's own sampling law), 3,072 tokens each
#   lseq    : mkseq -> data/lseq/train.seq
#   dump    : NQ features: long train (bf16), short train subset (bf16, the earlier heal-trace replies), sel/rep (f32)
#   base    : shipped drafter offline on sel + each rep row; cal = offline round-sim vs server tok/round (sel 100K, 32K)
#   profile : 3 train steps (memory / speed on 100K sequences)
#   train   : mixed sampler (long with p=FRAC2, else short), select E_d mb on sel (32K + 100K bench samples)
#   rep     : best vs shipped on rep100 / rep32 / rep3 ; export : Q8_0 fold gguf (453,522,464 B)
# usage: run_long.sh [phase ...]   (default: check selrep dumpsel profile lgen lseq dump base train rep export)
set -uo pipefail
R=${R:-/root/dst}; T=$R/wt/tools/dsparktrain; D=$R/data; O=$R/out
PY=${PY:-/mnt/data/inkflags/venv/bin/python}
M=$R/model/Inkling-2-Mini-NVFP4-native-q8d.gguf
DQ=$R/model/Inkling-2-Mini-DSpark-Q8_0.gguf; DF=$R/model/Inkling-2-Mini-DSpark-Q8_0-fold.gguf; DB=$R/model/Inkling-2-Mini-DSpark-BF16.gguf
KVF="-ctk sjkvarn4 -ctv sjkvarn4"; UBF="-b 2048 -ub 1024"; PORT=18480
STEPS=${STEPS:-600}; LR=${LR:-3e-5}; LOSS=${LOSS:-e4}; KL=${KL:-0.1}; SPS=${SPS:-2}; APS=${APS:-1024}; FRAC2=${FRAC2:-0.6}
EVAL_EVERY=${EVAL_EVERY:-50}; EVAL_STRIDE=${EVAL_STRIDE:-8}; NPRED=${NPRED:-3072}; CALTOL=${CALTOL:-3.0}
export PYTHONPATH=$R/wt/gguf-py LLAMA_ARG_NO_CACHE_DISK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $D $O $R/run
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $R/run.log; }
done_or() { [ -e $R/L.$1.done ] && { log "skip $1"; return 0; }; return 1; }
TRAINARGS="--w0 $R/w0 --drafter $DF --bf16 $DB"
DUMPS="$R/bin/llama-dspark-dump -m $M -ngl 99 -fa on -t 8 -c 12288 -b 2048 -ub 512 $KVF"
DUMPL="$R/bin/llama-dspark-dump -m $M -ngl 99 -fa on -t 8 -c 110592 $UBF $KVF"
SPEC="--env LLAMA_DEEPCTX_LADDER=1 --timeout 3600"

ph_check() {
  done_or check && return
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader | tee -a $R/run.log
  $R/bin/llama-server --version >> $R/run.log 2>&1 || { log 'llama-server does not start (shim)'; return 1; }
  $PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" >> $R/run.log 2>&1 \
      || { log 'torch CUDA not usable'; return 1; }
  ( cd $R/w0 && sha256sum -c SHA256SUMS ) >> $R/run.log 2>&1 || { log 'w0 checksum mismatch'; return 1; }
  for f in $M $DQ $DF $DB $D/long_docs.jsonl $D/short_train.seq $D/wiki.test.raw; do [ -s $f ] || { log "missing $f"; return 1; }; done
  df -h $R | tail -1 | tee -a $R/run.log
  local free=$(df -BG --output=avail $R | tail -1 | tr -dc 0-9); (( free >= 110 )) || { log "only ${free} GB free, need >= 110"; return 1; }
  touch $R/L.check.done; log 'check ok'
}

ph_selrep() {
  done_or selrep && return
  # name plen ctx seeds
  for row in "sel100 100000 110592 101,102" "sel32 32000 40960 101,102" "rep100 100000 110592 103" "rep32 32000 40960 103" "rep3 3000 16384 101,102"; do
    set -- $row; local c=$O/sr_$1
    [ -e $c/result.json ] && continue
    local ru=""; [[ $1 == sel* ]] && ru=--reuse   # sel: one prefill per row; rep rows: fresh prefill per seed (honest server cal)
    $PY $T/spec_check.py --bin $R/bin --model $M --drafter $DF --out $c --seeds $4 --plen $2 --ctx $3 --npred 5120 --wiki $D/wiki.test.raw \
        --port $((PORT + 2)) $SPEC $ru --extra="$KVF $UBF" > $c.log 2>&1 || { log "selrep $1 failed: $(tail -2 $c.log | tr '\n' ' ')"; return 1; }
    log "selrep $1 $(grep DONE $c.log | cut -c1-300)"
  done
  $PY - $O $D <<'PY' || { log 'selrep concat failed'; return 1; }
import struct, sys
O, D = sys.argv[1:]
def rd(p):
    b = open(p, 'rb').read(); n = struct.unpack_from('<i', b, 0)[0]; o = 4; out = []
    for _ in range(n):
        L, npr = struct.unpack_from('<ii', b, o); o += 8; out.append((npr, list(struct.unpack_from(f'<{L}i', b, o)))); o += 4 * L
    return out
def wr(p, seqs):
    with open(p, 'wb') as f:
        f.write(struct.pack('<i', len(seqs)))
        for npr, s in seqs:
            f.write(struct.pack('<ii', len(s), npr)); f.write(struct.pack(f'<{len(s)}i', *s))
wr(f'{D}/sel.seq', rd(f'{O}/sr_sel100/calib.seq') + rd(f'{O}/sr_sel32/calib.seq'))
for r in ('rep100', 'rep32', 'rep3'):
    wr(f'{D}/{r}.seq', rd(f'{O}/sr_{r}/calib.seq'))
PY
  touch $R/L.selrep.done
}

ph_lgen() {
  done_or lgen && return
  LLAMA_DEEPCTX_LADDER=1 LLAMA_SPEC_ACCPOS=1 $R/bin/llama-server -m $M --host 127.0.0.1 --port $PORT -c 110592 -np 1 -ngl 99 -fa on -t 8 -tb 8 \
      --jinja --no-webui $UBF $KVF --model-draft $DF --spec-type draft-dspark -ngld 99 -ubd 64 --spec-draft-n-max 4 --no-spec-mb-mtp \
      > $R/lgen_server.log 2>&1 &
  local sp=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
  curl -sf http://127.0.0.1:$PORT/health > /dev/null || { log 'lgen server did not come up'; kill $sp; return 1; }
  local rc=1
  for try in 1 2; do
    $PY $T/gen_long.py --docs $D/long_docs.jsonl --out $D/lgens.jsonl --port $PORT --npred $NPRED >> $R/lgen.log 2>&1; rc=$?
    [ $rc = 0 ] && break; log "lgen try $try rc $rc"
  done
  kill $sp; wait $sp 2>/dev/null
  [ $rc = 0 ] && touch $R/L.lgen.done && log "lgen ok $(tail -1 $R/lgen.log)"
}

ph_lseq() { done_or lseq && return; $PY $T/mkseq.py $D/lgens.jsonl $D/lseq --max-len 110592 > $R/lseq.log 2>&1 && touch $R/L.lseq.done && log "lseq $(tr '\n' ' ' < $R/lseq.log)"; }

ph_dumpsel() {   # sel first: the profile phase trains on it (2 x 100K + 2 x 32K real sequences) before lgen spends hours
  done_or dumpsel && return
  [ -e $D/sel.dump.done ] || { $DUMPL --ds-in $D/sel.seq --ds-out $D/sel --ds-dtype f32 > $R/dump_sel.log 2>&1 && touch $D/sel.dump.done; } \
      || { log "dump sel failed: $(tail -2 $R/dump_sel.log | tr '\n' ' ')"; return 1; }
  log "dump sel $(grep -E 'seqs,' $R/dump_sel.log | tail -1)"
  dd if=$D/sel.feat of=/dev/null bs=16M count=64 iflag=direct 2>&1 | tail -1 | sed 's/^/COLD_READ /' | tee -a $R/run.log
  touch $R/L.dumpsel.done
}

ph_dump() {
  done_or dump && return
  for s in "ltrain $D/lseq/train.seq bf16 L" "strain $D/short_train.seq bf16 S" "rep100 $D/rep100.seq f32 L" "rep32 $D/rep32.seq f32 L" "rep3 $D/rep3.seq f32 S"; do
    set -- $s; [ -e $D/$1.dump.done ] && continue
    local cmd=$DUMPL; [ $4 = S ] && cmd=$DUMPS
    $cmd --ds-in $2 --ds-out $D/$1 --ds-dtype $3 > $R/dump_$1.log 2>&1 || { log "dump $1 failed: $(tail -2 $R/dump_$1.log | tr '\n' ' ')"; return 1; }
    touch $D/$1.dump.done; log "dump $1 $(grep -E 'seqs,' $R/dump_$1.log | tail -1)"
  done
  touch $R/L.dump.done
}

ph_base() {
  done_or base && return
  $PY $T/train.py $TRAINARGS --out $O/base --eval-only --shipped-fold --sel $D/sel:$D/sel.seq > $R/base_sel.log 2>&1 || { log 'base sel failed'; return 1; }
  for r in rep100 rep32 rep3; do
    $PY $T/train.py $TRAINARGS --out $O/base --eval-only --shipped-fold --rep $D/$r:$D/$r.seq > $R/base_$r.log 2>&1 || { log "base $r failed"; return 1; }
  done
  # cal: offline round-sim (shipped) vs the server's own tok/round on the same sequences (sel100 / sel32 / rep rows)
  $PY - $O $R $CALTOL <<'PY' > $R/cal.txt || { log 'cal compare failed'; return 1; }
import json, sys
O, R, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def ev(f):
    return json.loads([l for l in open(f) if '"eval"' in l][-1])
for r, f in (('rep100', 'base_rep100.log'), ('rep32', 'base_rep32.log'), ('rep3', 'base_rep3.log')):
    e = ev(f'{R}/{f}'); s = json.load(open(f'{O}/sr_{r}/result.json'))
    srv = sum(s['mean_len']) / len(s['mean_len']); off = e['tok_round_sim_mb']
    print(f"CAL {r} prompt {s['prompt_tokens']} server_tok_round {srv:.3f} (per request {s['mean_len']}) offline_mb {off:.3f} diff {100 * (off / srv - 1):+.2f}% "
          f"srv_acc_pos {s['acc_per_pos']} BAD {int(abs(100 * (off / srv - 1)) > tol)}")
e = ev(f'{R}/base_sel.log'); print('BASE sel E_mb', e['E_mb'], 'round_sim_mb', e['tok_round_sim_mb'], 'alpha_mb', e['alpha_mb'])
PY
  cat $R/cal.txt | tee -a $R/run.log
  grep -q 'rep100 .*BAD 1' $R/cal.txt && { log "CAL_STOP: offline vs server differ > ${CALTOL}% at 100K on NQ"; return 1; }
  touch $R/L.base.done
}

TARGS() { echo "$TRAINARGS --train $D/strain:$D/short_train.seq --train2 $D/ltrain:$D/lseq/train.seq --frac2 $FRAC2 --sel $D/sel:$D/sel.seq \
  --lr $LR --loss $LOSS --kl $KL --seqs-per-step $SPS --anchors-per-seq $APS --chunk-long 32"; }

ph_profile() {
  done_or profile && return
  $PY $T/train.py $(TARGS) --out $R/prof --profile 4 --train2 $D/sel:$D/sel.seq --frac2 1.0 > $R/profile.log 2>&1 || { log "profile failed: $(tail -3 $R/profile.log | tr '\n' ' ')"; return 1; }
  log "profile $(grep -E 'mem_gib|mean_step' $R/profile.log | tail -2 | tr '\n' ' ' | cut -c1-400)"
  touch $R/L.profile.done
}

ph_train() {
  done_or train && return
  rm -f $R/run/log.jsonl $R/run/best.pt
  $PY $T/train.py $(TARGS) --out $R/run --steps $STEPS --eval-every $EVAL_EVERY --eval-stride $EVAL_STRIDE > $R/train.log 2>&1 \
      && touch $R/L.train.done && log "train ok $(grep saved $R/run/log.jsonl | tail -1)"
}

ph_rep() {
  done_or rep && return
  grep -q saved $R/run/log.jsonl || { log 'NO_IMPROVEMENT: no checkpoint beat step 0 on sel (bench 32K+100K) -> DROP, nothing to gate'; touch $R/L.rep.done; return 0; }
  for r in rep100 rep32 rep3; do
    $PY $T/train.py $TRAINARGS --out $O/best --eval-only --init $R/run/best.pt --rep $D/$r:$D/$r.seq > $R/rep_$r.log 2>&1 || { log "rep $r failed"; return 1; }
    log "rep $r best $(tail -1 $R/rep_$r.log | cut -c1-500)"
    log "rep $r base $(grep '"eval"' $R/base_$r.log | tail -1 | cut -c1-500)"
  done
  touch $R/L.rep.done
}

ph_export() {
  done_or export && return
  [ -e $R/run/best.pt ] && grep -q saved $R/run/log.jsonl || { log 'nothing to export'; touch $R/L.export.done; return 0; }
  $PY $T/export.py $R/run/best.pt $DQ $DB $R/w0 $O/dspark-long --fold-script $R/wt/ink_dspark_fold.py > $R/export.log 2>&1 \
      && touch $R/L.export.done && log "export $(tail -1 $R/export.log | cut -c1-400)"
}

PHASES=${*:-check selrep dumpsel profile lgen lseq dump base train rep export}
for p in $PHASES; do
  log "PHASE $p"
  ph_$p || { log "PHASE $p FAILED"; exit 1; }
done
log ALL_DONE
