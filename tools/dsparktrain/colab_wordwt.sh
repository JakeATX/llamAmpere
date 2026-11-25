#!/bin/bash
# DRAFT-WORDWT: one Colab A100 session = WORD-weighted drafter QAT at the shipped size (Q8_0 fold STE from step 1, init =
# shipped drafter, never full size). WRITTEN, NOT RUN: needs a COORD go (est. 2.7-3.5 A100-h, ~24-31 CU at ~8.9 CU/h).
# Runs on Taichi; drives session $SES over ssh. Same pipeline as the long-ctx round 2 (run_long.sh), with:
#   WORDWT=1.0 (WORD slots count 2x in the loss: E_d term + KL), LR 1e-5 (round 2 at 3e-5 left the optimum by step 50),
#   STEPS 200, sel eval every 25, STOPNG 100 (no sel gain over the shipped drafter by step 100 -> stop, nothing exported),
#   20 of round 2's 40 long seqs (LongBench-v2 ~94-99K prompts, NQ target replies; lgen skipped, lseq built here) + the
#   200 short heal seqs, FRAC2 0.6. Selection = unweighted E_d mb on bench sel (100K x2 + 32K x2, seeds >= 101).
# Phases on the box: check dumpsel dump base train rep export (selrep/lgen/lseq outputs are staged from round 2).
# After: pull run.log, log.jsonl, rep logs, exported gguf -> $J/ww; then the 24ink 100K replay (replay_ww.sh) on Taichi.
# Spend guards: stop the session on any phase FAIL, on NO_GAIN, at ALL_DONE, or at CAP_H hours (default 4).
# Read-out: ship-vs-new is decided by the replay/gate. Credit the WORD weighting itself only if sel alpha_mb_word rises
# more than alpha_mb_other at the same k (this run also changes lr and data vs round 2; no unweighted lr-1e-5 control).
# usage: colab_wordwt.sh [prep|stage|run|watch|pull|stop|all]   (all = prep stage run watch pull stop)
set -uo pipefail
SES=${SES:-dsqww}; J=${J:-/mnt/data/dsparkqat/ww}; CAP_H=${CAP_H:-4}
W=$(cd "$(dirname "$0")/../.." && pwd); COLAB=/home/jakek/.local/bin/colab; PULL=/home/jakek/qwen38-bench/.locks/relay/pull.sh
LT_MAX=${LT_MAX:-20}; LPY=/mnt/data/inkflags/venv/bin/python   # Taichi venv (numpy, gguf-py deps)
ENVRUN="WORDWT=${WORDWT:-1.0} LR=${LR:-1e-5} STEPS=${STEPS:-200} EVAL_EVERY=${EVAL_EVERY:-25} STOPNG=${STOPNG:-100} FRAC2=${FRAC2:-0.6} SPS=${SPS:-2} APS=${APS:-1024}"
mkdir -p $J
cat > $J/ssh_config <<CFG
Host $SES
  HostName $SES
  User root
  ProxyCommand $COLAB ssh --proxy-mode -s $SES
  UserKnownHostsFile $J/known_hosts_colab
  StrictHostKeyChecking accept-new
  BatchMode yes
  ServerAliveInterval 30
  LogLevel ERROR
CFG
SSHA="-F $J/ssh_config $SES"
BX() { ssh $SSHA "$@" < /dev/null; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $J/drive.log; }

prep() {   # local, before any paid session: 20 long train seqs from round 2's replies, the word mask
  python3 -I $W/tools/dsparktrain/mkseq.py /mnt/data/dsparkqat/colab_long/final/lgens.jsonl $J/lseq_all --max-len 110592 > $J/lseq.log
  python3 -I - $J/lseq_all/train.seq $J/train.seq $LT_MAX <<'PY'
import struct, sys
b = open(sys.argv[1], 'rb').read(); n = struct.unpack_from('<i', b, 0)[0]; k = min(n, int(sys.argv[3])); o = 4; out = []
for _ in range(n):
    L, npr = struct.unpack_from('<ii', b, o); out.append(b[o:o + 8 + 4 * L]); o += 8 + 4 * L
open(sys.argv[2], 'wb').write(struct.pack('<i', k) + b''.join(out[:k]))
print('lseq', k, 'of', n, 'seqs')
PY
  [ -s $J/train.seq ] || { log 'PREP_FAIL lseq'; return 1; }
  $LPY -I $W/tools/dsparktrain/mkwordmask.py /mnt/data/inkling2/Inkling-2-Mini-NVFP4-native-q8d.gguf $J/wordmask.npy | tee -a $J/drive.log
  [ ${PIPESTATUS[0]} = 0 ] && [ -s $J/wordmask.npy ] || { log 'PREP_FAIL wordmask'; return 1; }
  log PREP_OK
}

stage() {
  [ -s $J/train.seq ] && [ -s $J/wordmask.npy ] || { log 'run prep first'; return 1; }
  $COLAB new -s $SES --gpu A100 2>&1 | tail -2 | tee -a $J/drive.log
  date -u +%s > $J/session_start
  BX "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; df -h / | tail -1" | tee -a $J/drive.log
  # big files: bundle (bins + w0) and the 3 drafter files via the relay; NQ over the session ssh (4 dd streams)
  for f in dsparkqat-bundle.tar.zst Inkling-2-Mini-DSpark-Q8_0.gguf Inkling-2-Mini-DSpark-Q8_0-fold.gguf Inkling-2-Mini-DSpark-BF16.gguf; do
    d=/root/dst/model/$f; [ $f = dsparkqat-bundle.tar.zst ] && d=/root/$f
    r=$($PULL "$SSHA" $f $d 2>&1 | tail -1); log "$f $r"; [[ $r == "RELAY rc=0" ]] || { log STAGE_FAIL; return 1; }
  done
  BX 'set -e; mkdir -p /root/dst && cd /root/dst && (command -v zstd >/dev/null || (apt-get -qq update && apt-get -qq install -y zstd >/dev/null)) && zstd -dc /root/dsparkqat-bundle.tar.zst | tar xf - && rm /root/dsparkqat-bundle.tar.zst
    sed -i "s#  /mnt/data/dsparkqat/rent/bundle/w0/#  #" /root/dst/w0/SHA256SUMS
    mkdir -p /mnt/data/inkflags/data /mnt/data/inkflags/venv/bin /root/dst/data/lseq; cp /root/dst/data/wiki.test.raw /mnt/data/inkflags/data/; ln -sf $(command -v python3) /mnt/data/inkflags/venv/bin/python
    python3 -c "import yaml" 2>/dev/null || pip -q install pyyaml
    pip -q install "nvidia-cuda-runtime-cu12==12.4.*" "nvidia-cublas-cu12==12.4.*" 2>&1 | tail -1
    P=$(python3 -c "import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))")/nvidia
    for x in $P/cuda_runtime/lib/libcudart.so.12 $P/cublas/lib/libcublas.so.12 $P/cublas/lib/libcublasLt.so.12 /usr/lib64-nvidia/libcuda.so* /usr/lib64-nvidia/libnvidia-ptxjitcompiler.so* /usr/lib64-nvidia/libnvidia-ml.so*; do ln -sf $x /root/dst/bin/; done
    /root/dst/bin/llama-dspark-dump --version 2>&1 | grep -m1 version
    python3 -c "import torch, numpy, yaml; assert torch.cuda.is_available(); print(\"PY_OK\", torch.__version__, torch.cuda.get_device_name(0))"' > $J/setup.log 2>&1
  local src=$?; tail -3 $J/setup.log | tee -a $J/drive.log
  [ $src = 0 ] && grep -q PY_OK $J/setup.log || { log 'BOX_SETUP_FAIL (before the NQ upload)'; return 1; }
  # small delta: this worktree's tools, round-2 data (bench sel/rep seqs + their server baselines), long seqs, mask
  tar -C $W -cf - --exclude __pycache__ tools/dsparktrain | ssh $SSHA 'cd /root/dst/wt && tar xf -' || { log DELTA_FAIL; return 1; }
  tar -C /mnt/data/dsparkqat/long -cf - long_docs.jsonl short_train.seq | ssh $SSHA 'cd /root/dst/data && tar xf -' || { log DELTA_FAIL; return 1; }
  tar -C /mnt/data/dsparkqat/colab_long/keep -cf - sel.seq rep100.seq rep32.seq rep3.seq | ssh $SSHA 'cd /root/dst/data && tar xf -' || { log DELTA_FAIL; return 1; }
  tar -C /mnt/data/dsparkqat/colab_long/keep -cf - out | ssh $SSHA 'cd /root/dst && tar xf -' || { log DELTA_FAIL; return 1; }
  tar -C $J -cf - train.seq | ssh $SSHA 'cd /root/dst/data/lseq && tar xf -' || { log DELTA_FAIL; return 1; }
  tar -C $J -cf - wordmask.npy | ssh $SSHA 'cd /root/dst/data && tar xf -' || { log DELTA_FAIL; return 1; }
  BX 'cd /root/dst && touch L.selrep.done L.lgen.done L.lseq.done && ls -la data data/lseq | head -20' | tee -a $J/drive.log
  NQ=/mnt/data/inkling2/Inkling-2-Mini-NVFP4-native-q8d.gguf; NS=4; BS=$((64 * 1024 * 1024))
  NB=$(( ($(stat -c %s $NQ) + BS - 1) / BS )); PER=$(( (NB + NS - 1) / NS )); log "NQ upload start ($NB x 64 MiB)"
  local pids="" ok=1
  for k in $(seq 0 $((NS - 1))); do
    ( dd if=$NQ bs=$BS skip=$((k * PER)) count=$PER status=none | ssh $SSHA "dd of=/root/dst/model/$(basename $NQ) bs=$BS seek=$((k * PER)) conv=notrunc status=none iflag=fullblock" ) &
    pids="$pids $!"
  done
  for p in $pids; do wait $p || ok=0; done
  (( ok )) || { log NQ_UPLOAD_FAIL; return 1; }
  r=$(BX "sha256sum /root/dst/model/$(basename $NQ) | cut -d' ' -f1"); l=$(sha256sum $NQ | cut -d' ' -f1)
  [ "$r" = "$l" ] || { log "NQ_SHA_MISMATCH $r $l"; return 1; }
  log STAGE_OK
}

run() {   # selrep/lgen/lseq are staged; the rest runs detached on the box
  BX "cd /root/dst && env $ENVRUN CL=8 nohup bash wt/tools/dsparktrain/run_long.sh check dumpsel dump base train rep export > /root/dst/long.out 2>&1 &" && log "RUN started: $ENVRUN"
}

watch() {   # every 5 min: phase + last eval; stop the session at the end or at CAP_H
  while :; do
    s=$(BX "tail -1 /root/dst/run.log; pgrep -f run_long.sh >/dev/null || echo RUNNER_GONE; grep -c '\"eval\"' /root/dst/run/log.jsonl 2>/dev/null; grep '\"eval\"' /root/dst/run/log.jsonl 2>/dev/null | tail -1 | cut -c1-200" 2>&1)
    log "WATCH $(echo "$s" | tr '\n' ' ' | cut -c1-400)"
    echo "$s" | grep -qE 'ALL_DONE|FAILED|NO_GAIN|NO_IMPROVEMENT|RUNNER_GONE' && { log "WATCH end: $(echo "$s" | head -1)"; return 0; }
    (( $(date -u +%s) - $(cat $J/session_start) > CAP_H * 3600 )) && { log "CAP ${CAP_H} h reached"; return 0; }
    sleep 300
  done
}

pull() {
  mkdir -p $J/final
  for f in run.log long.out cal.txt train.log base_sel.log base_rep100.log base_rep32.log base_rep3.log rep_rep100.log rep_rep32.log rep_rep3.log export.log run/log.jsonl; do
    ssh $SSHA "cat /root/dst/$f" > $J/final/$(basename $f) 2>/dev/null < /dev/null
  done
  ssh $SSHA "cat /root/dst/run/best.pt" > $J/final/best.pt 2>/dev/null < /dev/null   # export can be redone on Taichi (export.py)
  BX 'ls /root/dst/out/dspark-long-Q8_0-fold.gguf /root/dst/out/dspark-long-export.json 2>/dev/null' | while read -r g; do ssh $SSHA "cat $g" > $J/final/$(basename $g) < /dev/null; done
  ls -la $J/final | tee -a $J/drive.log
}

stop() { $COLAB stop -s $SES 2>&1 | tail -1 | tee -a $J/drive.log; log "session $SES stopped; A100-h $(python3 -c "import time; print(round((time.time() - $(cat $J/session_start)) / 3600, 2))")"; }

for ph in ${*:-all}; do
  case $ph in
    all) prep || exit 1; stage && run && watch; pull; stop ;;
    prep) prep || exit 1 ;;
    stage|run|watch|pull|stop) $ph || { log "$ph FAILED"; stop; exit 1; } ;;
    *) echo "unknown $ph"; exit 2 ;;
  esac
done
