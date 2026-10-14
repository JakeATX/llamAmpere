#!/bin/bash
# AI+ span verification chain: 3 GPU holds (<= 20 min each) + one CPU repack between holds 2 and 3.
set -u
L=/mnt/models/bl1013/AIBEST/logs
V=$L/llama.cpp-aibest-verify-s.sh
Q=/mnt/models/bl1013/AIBEST/build-bl/bin/llama-ai4-quantize
hold() {
  echo "$(date '+%F %T') waiting for gpu.lock ($1)" >> $L/s_summary.log
  flock /home/jake-k/qwen38-bench/.locks/gpu.lock systemd-run --user --wait --collect -p RuntimeMaxSec=1200 \
    -E PARENT_HOLDS_GPU_LOCK=1 $V "$1"
  echo "hold $1 rc=$?" >> $L/s_summary.log
}
hold kern
hold enc
if [ -s $L/s_ov_u.gguf ]; then
  $Q --repack-s $L/s_ov_u.gguf --out $L/s_ov_s.gguf --repack-verify-all > $L/s_repack.log 2>&1
  echo "repack rc=$? u=$(stat -c%s $L/s_ov_u.gguf) s=$(stat -c%s $L/s_ov_s.gguf 2>/dev/null)" >> $L/s_summary.log
  hold kl
else
  echo "repack SKIPPED: no s_ov_u.gguf" >> $L/s_summary.log
fi
echo "chain done $(date '+%F %T')" >> $L/s_summary.log
