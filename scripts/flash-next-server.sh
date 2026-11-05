#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/data/moe/flash-next.env"
server=${LLAMA_SERVER_BIN:-$repo/build/bin/llama-server}
exact=()
if [[ ${1-} == --exact ]]; then
    exact=(--moe-skip off --moe-skip-budget 0 --moe-skip-rank-floor 0 --no-spec-mb-mtp)
    shift
fi
depth=()
# a depth in the caller's own arguments wins outright
if [[ -n $FLASH_NEXT_DRAFT_MAX && " $* " != *" --spec-draft-n-"* ]]; then
    depth=(--spec-draft-n-min-adaptive $(( FLASH_NEXT_DRAFT_MAX < 3 ? FLASH_NEXT_DRAFT_MAX : 3 )) --spec-draft-n-max "$FLASH_NEXT_DRAFT_MAX")
    # --spec-type adds to the list, so only pick adaptive MTP when the caller names no type
    [[ " $* " == *" --spec-type"* ]] || depth=(--spec-type draft-mtp-adaptive "${depth[@]}")
fi
# one slot unless the caller sets the count: the server default (4 slots) gives SJ-KVaRN one KV stream per slot, and
# the default --kv-gpu-tokens needs a single stream, so the model would refuse to load
slots=()
[[ " $* " == *" -np "* || " $* " == *" --parallel"* || " $* " == *" -np="* ]] || slots=(--parallel 1)
# the shape our speed numbers were measured with; the caller's own arguments come later, so they win. A 12 GB card still
# reads the prompt with -ub 6144 (the engine lowers it), and experts fall back to unpinned file pages when RAM is short.
shape=(-ngl 999 -fa on -b 131072 -ub 8192 --fit off --moe-arena on --moe-cache on --ctx-checkpoints 4)
[[ " $* " == *" -c "* || " $* " == *" --ctx-size"* ]] || shape+=(-c 110592)
# 8 CPU threads as measured; smaller machines keep the engine's own count (physical cores)
(( $(nproc --all) < 16 )) || shape+=(-t 8 -tb 8)
# a long prompt's temporary output (about 4 GiB at -c 110592) lives in an unlinked file, as measured. In RAM it is pinned,
# costs seconds to allocate and pushes the prompt's read cache out. Needs a real disk with room; otherwise it stays in RAM.
if [[ -z ${LLAMA_MOE_ARENA_OUTPUT_DIR-} ]]; then
    output_dir=${XDG_CACHE_HOME:-$HOME/.cache}/llamampere/moe-output
    if mkdir -p "$output_dir" 2>/dev/null && [[ ! $(stat -f -c %T "$output_dir") =~ ^(tmpfs|ramfs)$ ]] &&
            (( $(df --output=avail -k "$output_dir" | tail -1) >= 16 * 1024 * 1024 )); then
        export LLAMA_MOE_ARENA_OUTPUT_DIR=$output_dir
    else
        echo "flash-next-server: $output_dir needs a real disk with 16 GiB free; keeping prompt output in RAM (slower prompts)" >&2
    fi
fi
exec "$server" -md "$LLAMA_ARG_MODEL_DRAFT" --moe-profile "$GGML_MOE_CACHE_PROFILE_LOAD" \
    -ctk sjkvarn4 -ctv sjkvarn4 --moe-skip \
    "${shape[@]}" "${slots[@]}" "${depth[@]}" "$@" "${exact[@]}"
