#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/data/moe/flash-next.env"
server=${LLAMA_SERVER_BIN:-$repo/build/bin/llama-server}
depth=()
# a depth in the caller's own arguments wins outright
if [[ -n $FLASH_NEXT_DRAFT_MAX && " $* " != *" --spec-draft-n-"* ]]; then
    depth=(--spec-draft-n-min-adaptive $(( FLASH_NEXT_DRAFT_MAX < 3 ? FLASH_NEXT_DRAFT_MAX : 3 )) --spec-draft-n-max "$FLASH_NEXT_DRAFT_MAX")
    # --spec-type adds to the list, so only pick adaptive MTP when the caller names no type
    [[ " $* " == *" --spec-type"* ]] || depth=(--spec-type draft-mtp-adaptive "${depth[@]}")
fi
exec "$server" -md "$LLAMA_ARG_MODEL_DRAFT" --moe-profile "$GGML_MOE_CACHE_PROFILE_LOAD" "${depth[@]}" "$@"
