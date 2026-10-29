#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/data/moe/flash-next.env"
server=${LLAMA_SERVER_BIN:-$repo/build/bin/llama-server}
exec "$server" -md "$LLAMA_ARG_MODEL_DRAFT" --moe-profile "$GGML_MOE_CACHE_PROFILE_LOAD" "$@"
