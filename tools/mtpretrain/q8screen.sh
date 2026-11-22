#!/bin/bash
# Q8_0 screen (COORD-taichi ask): heldout_rep only, Taichi GPU1 under gpu1.lock; dump + targets + 4 evals
D=/mnt/data/mtpretrain/q8screen; T=/mnt/data/mtpretrain/wt/tools/mtpretrain; PY=/mnt/data/inkflags/venv/bin/python3; W=/mnt/data/mtpretrain/w0
B=/mnt/data/rel0721/build-76c662548/bin; M=/home/jakek/qwen38-bench/models/multi4lsa-p76/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf
export CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$B
cd $D
[ -e h.bin.idx ] && [ $(wc -l < h.bin.idx) = 57 ] || /mnt/data/mtpretrain/bin/llama-mtp-dump-h -m $M -c 6144 -b 2048 -ub 2048 -fa on -ctk sjkvarn4 -ctv sjkvarn4 -ngl 99 --mtp-in seq.bin --mtp-out h.bin > dump.log 2>&1 || exit 1
$PY $T/mtp_train.py targets --w $W --data $D > targets.log 2>&1 || exit 1
for v in base:- q5qat:--head-gguf=/mnt/data/mtpretrain/rental_out/mtp-otp-qat.gguf fp32qat:--head=/mnt/data/mtpretrain/rental_out/run_qat/head_best.pt q8qat:--head-gguf=$D/mtp-otp-qat-q8.gguf; do
  n=${v%%:*}; arg=${v#*:}; [ "$arg" = - ] && arg=
  $PY $T/mtp_train.py eval --w $W --data $D --split heldout_rep $arg --json-out eval_$n.json > eval_$n.log 2>&1 || exit 1
  echo "$n $(tail -1 eval_$n.log)"
done
echo SCREEN_DONE
