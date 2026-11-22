#!/bin/bash
# MTPRETRAIN speed gate on HOME cardq (owner MTPRETRAIN): old head (A) vs retrained head (B, overlay patch), 27B P76 AI4,
# bare --preset 24dense -c 262144, chat template (q88 coding trajectory + follow-up turn), temp 1 / top_k 20 / top_p 0.95,
# 5,120-token reply, LLAMA_ARG_NO_CACHE_DISK=1, ABBA 3-seed cells -> 6 seeds per arm per depth, at ~8K and ~100K prompts.
# Then identity cells (greedy 512 tokens at ~8K): exact MTP A vs B, and MTP off A vs B.
W=/mnt/overflow/mtpretrain/gate
sha256sum /mnt/overflow/mtpretrain/mtp-otp.gguf > $W/out-overlay.sha256
bash $W/cell2.sh $W/cfg_gate.txt g100 100000 "A:7300,7301,7302 B:7300,7301,7302 B:7303,7304,7305 A:7303,7304,7305"
bash $W/cell2.sh $W/cfg_gate.txt g8 8000 "A:7300,7301,7302 B:7300,7301,7302 B:7303,7304,7305 A:7303,7304,7305"
# identity cells log at -lv 4 so the loader's 'overlay patch ...: 15 tensors replaced' line proves B loaded the new head
# (speed cells stay at default verbosity; they use the same DS_ENV as the proven B cells)
NPRED=512 DS_EXTRA=--greedy SERVER_EXTRA="-lv 4" bash $W/cell2.sh $W/cfg_gate.txt id8 8000 "AX:7300 BX:7300"
NPRED=512 DS_EXTRA=--greedy SERVER_EXTRA="-lv 4 --spec-type none" bash $W/cell2.sh $W/cfg_gate.txt id8off 8000 "A:7300 B:7300"
python3 $W/summ.py $W/out > $W/summary.txt 2>&1
echo GATE_DONE
