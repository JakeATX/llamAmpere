#!/bin/bash
# Rerun only the arena side of the 2K KL. The ship side already passed.
# The frozen harness stops at 10 GiB. This run uses the worktree harness,
# which stops at 7 GiB, and the same frozen e8 server. The card is released
# when free RAM is under 52 GiB, because that server still refuses the pin
# unless 10 GiB will remain.
set -euo pipefail
ROOT=/mnt/models/fnarena
CELL=arena8kl-candidate
NEED=$((52 * 1024 * 1024))
if [ -e "$ROOT/artifacts/$CELL" ]; then
    printf 'refusing to overwrite %s\n' "$CELL"
    exit 2
fi
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }

while true; do
    set +e
    bash "$ROOT/scripts/fnarena/ram-locked-run.sh" "$NEED" \
        systemd-run --user --wait --collect \
        --unit="llama.cpp-fnarena-${CELL}" \
        -p CPUWeight=10 -p MemoryMax=50G -p TimeoutStopSec=10 -p LimitCORE=0 \
        -p WorkingDirectory="$ROOT/scripts/fnarena" \
        env \
        XDG_RUNTIME_DIR=/run/user/1000 \
        DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
        SHIP_SERVER="$ROOT/releases/e8a769ddf/bin/llama.cpp-server" \
        SHIP_LIB="$ROOT/releases/e8a769ddf/bin" \
        SHIP_CELL="$CELL" \
        SHIP_PROFILE="$ROOT/data/moe/flash-next.profile" \
        SHIP_PROMPT=2048 \
        SHIP_PREDICT=32 \
        SHIP_RELOAD=0 \
        SHIP_LIMIT=580 \
        SHIP_RESERVE_MB=2560 \
        PARENT_HOLDS_GPU_LOCK=1 \
        SHIP_CONFIG="$ROOT/releases/e8a769ddf/harness/output-file.json" \
        ARENA_MODE=on \
        ARENA_TEACHER=1 \
        SHIP_PREVIOUS_KL="$ROOT/artifacts/arena8kl-quality-previous/logprobs.npy" \
        timeout --signal=TERM --kill-after=10 590 \
        python3 "$ROOT/scripts/fnarena/gate.py" flashnext
    rc=$?
    set -e
    if [ "$rc" -eq 75 ]; then
        log "free RAM under 52 GiB after the card lock; releasing"
        sleep 5
        continue
    fi
    log "candidate exit ${rc}"
    exit "$rc"
done
