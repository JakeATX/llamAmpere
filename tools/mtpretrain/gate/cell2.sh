#!/bin/bash
# MTPRETRAIN copy of REL0721 copy of REL072 copy of MOEPLACE copy of REL071RC copy of SPEEDREG generic cell runner (do not edit while a job runs; add new configs in a new cfg file).
# usage: cell2.sh CFGFILE TAG DEPTH "L1:SEEDS L2:SEEDS ..."   (cells run in the order given)
# CFGFILE: lines "LABEL ARM BUNDLE CTX DS_ENV"  (DS_ENV ';'-separated, '-' for none)
set -u
CFGF=$1; TAG=$2; DEPTH=$3; LIST=$4
W=/mnt/overflow/mtpretrain/gate; H=/mnt/overflow/rel0721/extern/harness
MODEL=/mnt/models/lsadepth-ship/qwen3.8-27b-Swift1.5-Uncensored-ATX-AI4.gguf
T=/mnt/overflow/rel0721/extern/traj/q88.coding_gate.s7300
export DS_LOCKFILE=/home/jake-k/qwen38-bench/.locks/gpu.lock SERVER_EXTRA="${SERVER_EXTRA:-}"
declare -A CFG; while read -r L rest; do [[ -z $L || $L == \#* ]] && continue; CFG[$L]=$rest; done < "$CFGF"
[[ ${PARENT_HOLDS_GPU_LOCK:-0} = 1 ]] || { echo "run under cardq"; exit 9; }
shm=$(du -sm /dev/shm 2>/dev/null | cut -f1); [ "${shm:-0}" -lt 16384 ] || { echo "shm ${shm} MB"; exit 8; }
[[ $(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits) = 350.00 ]] || { echo "power limit"; exit 7; }
python3 -c "import shutil; assert shutil.disk_usage('/').free >= 45*2**30, 'root < 45G'" || exit 6
for item in $LIST; do
  L=${item%%:*}; S=${item#*:}; read -r arm B ctx env <<<"${CFG[$L]}"; [[ $env == - ]] && env=""
  OUT=$W/out/$TAG/$L.s${S//,/-}
  [ -e "$OUT/DONE" ] && { echo "skip $OUT"; continue; }
  rm -rf -- "${OUT:?}"; mkdir -p "$OUT"
  { echo "label=$L arm=$arm ctx=$ctx depth=$DEPTH seeds=$S ds_env=$env bundle=$B"; head -1 "$B/COMMIT"; date -u +%FT%TZ
    nvidia-smi --query-gpu=name,power.limit,memory.used,temperature.gpu --format=csv,noheader; } > "$OUT/arm_info.txt"
  nvidia-smi --query-gpu=timestamp,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader --loop-ms=500 > "$OUT/resources.csv" &
  mon=$!
  LD_LIBRARY_PATH=$B DS_ENV="$env" python3 $H/depth_speed.py --bin $B/llama-server --model $MODEL --map auto --out "$OUT" --traj $T \
     --arms $arm --depths $DEPTH --seeds $S --npred ${NPRED:-5120} --ctx $ctx --port ${PORT:-18497} --require-all-valid --warm ${DS_EXTRA:-} > "$OUT/depth_speed.log" 2>&1
  rc=$?; kill $mon 2>/dev/null; wait $mon 2>/dev/null
  echo "rc=$rc end=$(date -u +%FT%TZ)" >> "$OUT/arm_info.txt"; [ $rc = 0 ] && touch "$OUT/DONE"
  echo "$TAG $L s$S rc=$rc"
done
echo "LIST_DONE $TAG"
