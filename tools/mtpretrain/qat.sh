#!/bin/bash
# QAT polish: start from run/head_best.pt, train through the exact Q5_0/Q8_0 round trip, then export + heldout_rep eval
cd /root/mtp; PY=venv/bin/python3; T=wt/tools/mtpretrain
log() { echo "$(date -u +%FT%TZ) $*" | tee -a run.log; }
MTP_QAT=1 $PY $T/mtp_train.py train --w w0 --data data --out run_qat --head run/head_best.pt --lr 3e-5 --epochs 1 --eval-every 17 > train_qat.log 2>&1 || { log "PHASE qat FAILED"; exit 1; }
log "qat ok $(grep BEST run_qat/train.log | tail -1)"
[ -e run_qat/head_best.pt ] || { log "QAT NO_IMPROVEMENT"; exit 0; }
$PY $T/export_overlay.py --w w0 --head run_qat/head_best.pt --out mtp-otp-qat.gguf > export_qat.log 2>&1 || { log "PHASE qat-export FAILED"; exit 1; }
$PY $T/mtp_train.py eval --w w0 --data data --split heldout_rep --head-gguf mtp-otp-qat.gguf --json-out eval_otpqat_rep.json > eval_otpqat_rep.log 2>&1
log "rep otpqat $(tail -1 eval_otpqat_rep.log)"; log QAT_DONE
