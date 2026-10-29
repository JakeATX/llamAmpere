#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/data/moe/flash-next.env"
server=${LLAMA_SERVER_BIN:-$repo/build/bin/llama-server}
depth=()
[[ -z $FLASH_NEXT_DRAFT_MAX ]] || depth=(--spec-draft-n-max "$FLASH_NEXT_DRAFT_MAX")
exec "$server" -md "$LLAMA_ARG_MODEL_DRAFT" --moe-profile "$GGML_MOE_CACHE_PROFILE_LOAD" "${depth[@]}" "$@"
