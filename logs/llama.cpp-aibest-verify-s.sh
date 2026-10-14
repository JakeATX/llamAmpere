#!/bin/bash
# AI+ span (AIk_S) verification. usage: llama.cpp-aibest-verify-s.sh <stage: kern|enc|kl>
# Writes its own logs (systemd-run sends stdout to the journal). Each stage is one GPU hold under 20 min.
set -u
B=/mnt/models/bl1013/AIBEST/build-bl/bin
L=/mnt/models/bl1013/AIBEST/logs
T=/home/jake-k/qwen38-bench/models/d2_d003/Qwen3.8-27B-D003-D2.gguf
IM=/home/jake-k/Documents/Codex/2026-09-12/there/work/agnes3/reference-imatrix/imatrix_unsloth.gguf
BF=/mnt/models/blr/aiq_models/bf16_ffn8.gguf
IN=/home/jake-k/qwen38-bench/models/rq4_kl_inputs
S=$L/s_summary.log
stage=${1:-kern}
run() { local n=$1; shift; local t0=$SECONDS; "$@" > $L/s_$n.log 2>&1; echo "$n rc=$? $((SECONDS-t0))s" >> $S; }

echo "== stage $stage $(date '+%F %T')" >> $S
case "$stage" in
kern)
  run ratetest  $B/llama-ai4-rate-test
  run ucheck    $B/llama-ai4-kern-bench ucheck
  run bops_mm   $B/test-backend-ops -o MUL_MAT    -p 'type_a=ai[23456]s'
  run bops_mmid $B/test-backend-ops -o MUL_MAT_ID -p 'type_a=ai[23456]s'
  run ubench    $B/llama-ai4-kern-bench ubench 100
  ;;
enc)
  run self_s4 $B/llama-ai4-quantize --template $T --bf16 $BF --imatrix $IM \
    --tensor-type 'blk\.8\.ffn_gate\.weight=ai4s' --selftest --scale-mode span
  run self_s5 $B/llama-ai4-quantize --template $T --bf16 $BF --imatrix $IM \
    --tensor-type 'blk\.8\.ffn_gate\.weight=ai5s' --selftest --scale-mode span
  run self_s2 $B/llama-ai4-quantize --template $T --bf16 $BF --imatrix $IM \
    --tensor-type 'blk\.8\.ffn_gate\.weight=ai2s' --selftest --scale-mode span
  run self_u4 $B/llama-ai4-quantize --template $T --bf16 $BF --imatrix $IM \
    --tensor-type 'blk\.8\.ffn_gate\.weight=ai4u' --selftest --scale-mode row
  run enc_u   $B/llama-ai4-quantize --template $T --bf16 $BF --imatrix $IM \
    --tensor-type 'blk\.8\.(ffn_gate|ffn_up|ffn_down)\.weight=ai4u' --scale-mode row --overlay-only --werr \
    --out $L/s_ov_u.gguf
  ;;
kl)
  for a in u s; do
    env LLAMA_OVERLAY_GGUF=$L/s_ov_$a.gguf LLAMA_OVERLAY_RE='.*' KLD_MODE=read KLD_BASE=$IN/atx4xs.coding_gate.kldb \
        KLD_STRIDE=1 KLD_OUT=$L/s_kl_$a.csv \
      $B/llama-kld-depth -m $T -f $IN/atx4xs.coding_gate.txt -c 16384 -b 1024 -ub 1024 -t 8 -ngl 99 -fa on \
        -ctk f16 -ctv f16 > $L/s_kl_$a.log 2>&1
    echo "kl_$a rc=$?" >> $S
  done
  if cmp -s $L/s_kl_u.csv $L/s_kl_s.csv; then echo "kl csv IDENTICAL ($(wc -l < $L/s_kl_u.csv) lines)" >> $S
  else echo "kl csv DIFFER" >> $S; fi
  grep -hE "^ALL " $L/s_kl_u.log $L/s_kl_s.log >> $S
  ;;
esac
echo "== stage $stage done $(date '+%F %T')" >> $S
