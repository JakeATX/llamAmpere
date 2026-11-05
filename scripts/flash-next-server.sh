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
exec "$server" -md "$LLAMA_ARG_MODEL_DRAFT" --moe-profile "$GGML_MOE_CACHE_PROFILE_LOAD" \
    -ot token_embd.weight=CUDA0 -ctk sjkvarn4 -ctv sjkvarn4 --moe-skip \
    "${slots[@]}" "${depth[@]}" "$@" "${exact[@]}"
