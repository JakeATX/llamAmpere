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
slots=()
slot_count=${LLAMA_ARG_N_PARALLEL-1}
slot_explicit=${LLAMA_ARG_N_PARALLEL+x}
slot_next=0
for arg in "$@"; do
    if (( slot_next )); then slot_count=$arg; slot_next=0; fi
    case $arg in
        -np|--parallel) slot_next=1; slot_explicit=1 ;;
        -np=*|--parallel=*) slot_count=${arg#*=}; slot_explicit=1 ;;
    esac
done
[[ -n $slot_explicit ]] || slots=(--parallel 1)
# Shared-expert stream joins can be crossed by fused multi-slot graphs.
# Keep those experts on the main stream until the multi-slot join is safe.
if [[ $slot_count != 1 ]]; then
    export GGML_CUDA_SHEXP_SIDE=${GGML_CUDA_SHEXP_SIDE-0}
fi
# The caller's own shape arguments come later, so they win.
shape=(-ngl 999 -fa on -b 131072 -ub 8192 --fit off --moe-arena on --moe-cache on --ctx-checkpoints 4)
# The resident QSA window supports one stream; several slots use the existing per-sequence GPU cache instead.
# Explicit CLI values come last, and an explicit environment value also wins.
[[ $slot_count == 1 || -n ${LLAMA_ARG_KV_GPU_TOKENS+x} ]] || shape+=(--kv-gpu-tokens 0)
# Small cards use less host carry RAM. The server lists devices in CUDA order, including visibility remapping.
device_spec=${LLAMA_ARG_DEVICE-}
device_next=0
for arg in "$@"; do
    if (( device_next )); then device_spec=$arg; device_next=0; fi
    case $arg in
        -dev|--device) device_next=1 ;;
        -dev=*|--device=*) device_spec=${arg#*=} ;;
    esac
done
devices=$("$server" --list-devices 2>/dev/null) || devices=
while IFS= read -r device; do
    [[ $device =~ ^[[:space:]]*(CUDA[0-9]+):.*\(([0-9]+)\ MiB,\ ([0-9]+)\ MiB\ free\) ]] || continue
    name=${BASH_REMATCH[1],,}; card_mib=${BASH_REMATCH[2]}; free_mib=${BASH_REMATCH[3]}
    [[ -z $device_spec || ,${device_spec,,}, == *,"$name",* ]] || continue
    if (( card_mib < 23552 || free_mib < 14336 )); then
        shape+=(-b 32768 -ub 4096)
        break
    fi
done <<< "$devices"
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
