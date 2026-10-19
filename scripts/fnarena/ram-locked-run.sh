#!/bin/bash
# Hold the GPU lock only when host RAM is high enough for a pinned arena.
# Exit 75 without keeping the lock when memory is short, so other card work can proceed.
set -euo pipefail
need_kib="${1:?need kib}"
shift
lock=/home/jake-k/qwen38-bench/.locks/gpu.lock
exec 9>"$lock"
flock 9
if [ "$need_kib" -gt 0 ]; then
    kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt "$need_kib" ]; then
        printf 'short %s KiB available, need %s\n' "$kb" "$need_kib"
        exit 75
    fi
    printf 'card acquired with %s KiB available\n' "$kb"
fi
export PARENT_HOLDS_GPU_LOCK=1
exec "$@"
