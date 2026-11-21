#!/bin/bash
# routing trace (all 48 visits) over Mini's own replies + coding/agentic text, Taichi GPU 1
D=/mnt/data/inklsa/trace/out; rm -rf $D; mkdir -p $D
cd /mnt/data/inklsa/src
flock /home/jakek/qwen38-bench/.locks/gpu1.lock env CUDA_VISIBLE_DEVICES=1 LLAMA_ARG_NO_CACHE_DISK=1 INKDUMP_DIR=$D INKDUMP_TOPK=1 INKDUMP_LAYERS=999 \
  build/bin/llama-inklsa-dump -m /mnt/data/inkling2/Inkling-2-Mini-IQ4_NL.gguf -f /mnt/data/inklsa/trace/corpus.txt -c 81920 -b 2048 -ub 128 -ngl 99 -fa on -t 8
