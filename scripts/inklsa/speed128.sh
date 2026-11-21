#!/bin/bash
# protocol speed, Taichi GPU 1 (RTX 3090), Mini IQ4_NL, chat prompt ~116,736 tokens, chained seeds (fill 117K -> 132K),
# temp 1, >= 5,120-token reply per seed, ABBA order
cd /mnt/data/inklsa
M=/mnt/data/inkling2/Inkling-2-Mini-IQ4_NL.gguf
X="-ub 64 -b 2048 --slot-save-path /mnt/data/inklsa/speed/slots"
run() { # tag env
  flock /home/jakek/qwen38-bench/.locks/gpu1.lock /mnt/data/grown9b/venv/bin/python tools/inkbench.py --bin src/build/bin --model $M \
    --out speed/$1 --mode speed --plen 116736 --ctx 133120 --seeds 1,2,3 --npred 5120 --warm --chain --slot p116736.bin --port 18731 --extra "$X" --env "$2" 2>&1 | grep -E 'ROW|DONE|FAIL'
}
run A1_dense "LLAMA_ARG_NO_CACHE_DISK=1"
run B1_k4096 "LLAMA_ARG_NO_CACHE_DISK=1 LLAMA_INKLSA=1 LLAMA_INKLSA_KEEP=4096"
run C1_k16384 "LLAMA_ARG_NO_CACHE_DISK=1 LLAMA_INKLSA=1 LLAMA_INKLSA_KEEP=16384"
run C2_k16384 "LLAMA_ARG_NO_CACHE_DISK=1 LLAMA_INKLSA=1 LLAMA_INKLSA_KEEP=16384"
run B2_k4096 "LLAMA_ARG_NO_CACHE_DISK=1 LLAMA_INKLSA=1 LLAMA_INKLSA_KEEP=4096"
run A2_dense "LLAMA_ARG_NO_CACHE_DISK=1"
echo ALLDONE
