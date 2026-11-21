#!/bin/bash
# dry compute-buffer probe (caller holds gpu1.lock): $1=ctx $2=ub [extra args...]
C=$1; U=$2; shift 2
L=/mnt/data/inklsa/runs/cbq_$$.log
env CUDA_VISIBLE_DEVICES=1 LLAMA_ARG_NO_CACHE_DISK=1 /mnt/data/inklsa/src/build/bin/llama-server -m /mnt/data/inkflags/enc/Inkling-2-Mini-A_all_ait4u-fold.gguf \
  -c $C -ub $U -b 2048 -ngl 99 -fa on -cmoe --no-op-offload --port 18799 -v "$@" > $L 2>&1 &
P=$!; for i in $(seq 120); do grep -q 'CUDA0 compute buffer size' $L && break; sleep 1; done
echo "ctx $C ub $U $*: $(grep -m1 'CUDA0 compute buffer size' $L | sed 's/.*= *//')"; kill $P 2>/dev/null; wait $P 2>/dev/null; rm -f $L
