#!/bin/bash
# Re-run the frozen e8a769ddf arena gate. Recheck free RAM after the card
# lock is acquired. The previous attempt checked RAM first, then waited, and
# the copy finished under the 10 GiB floor.
set -euo pipefail
LOCK=/home/jake-k/qwen38-bench/.locks/gpu.lock
# 52 GiB. Arena7 started at 51.11 GiB and crossed under 10 GiB as the copy pinned.
NEED_KIB=$((52 * 1024 * 1024))
BUNDLE=/mnt/models/fnarena/releases/e8a769ddf
CELL=arena8
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

if [ -e "/mnt/models/fnarena/artifacts/${CELL}/result.json" ]; then
    log "refusing to overwrite ${CELL}"
    exit 2
fi

while true; do
    exec 9>"$LOCK"
    flock 9
    kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$kb" -ge "$NEED_KIB" ]; then
        log "card acquired with ${kb} KiB available"
        break
    fi
    log "card acquired with only ${kb} KiB available; releasing"
    flock -u 9
    exec 9>&-
    sleep 5
done

export PARENT_HOLDS_GPU_LOCK=1
export XDG_RUNTIME_DIR=/run/user/1000
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
export SHIP_PROMPT=100000
export SHIP_PREDICT=16
export SHIP_RESERVE_MB=1024
export SHIP_CONFIG="${BUNDLE}/harness/output-file.json"
set +e
timeout --signal=TERM --kill-after=5 590 \
    python3 "${BUNDLE}/harness/combined.py" "$BUNDLE" "$CELL"
rc=$?
set -e
log "gate exit ${rc}"
exit "$rc"
