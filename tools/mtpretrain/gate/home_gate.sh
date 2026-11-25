#!/bin/bash
# MTPRETRAIN on HOME cardq: gate.sh then copy the summary to V07/MTPRETRAIN/
V=/home/jake-k/qwen38-bench/kernel_cache_investigation/frontier/SHIP_20261019/V07/MTPRETRAIN; mkdir -p $V
bash /mnt/overflow/mtpretrain/gate/gate.sh > /mnt/overflow/mtpretrain/gate/gate-run.log 2>&1; rc=$?
{ echo "MTPRETRAIN gate on HOME $(nvidia-smi --query-gpu=name --format=csv,noheader), finished $(date -u +%FT%H:%MZ), gate.sh rc=$rc"; cat /mnt/overflow/mtpretrain/gate/summary.txt; echo "raw: /mnt/overflow/mtpretrain/gate/out"; } > $V/gate.txt 2>&1
