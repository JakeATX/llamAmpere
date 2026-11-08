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
        export GGML_CUDA_MOE_CACHE_RESERVE_MB=${GGML_CUDA_MOE_CACHE_RESERVE_MB-1536}
        export LLAMA_PROMPT_TEMP_HEADROOM=${LLAMA_PROMPT_TEMP_HEADROOM-1}
        export LLAMA_B1_ADAPTIVE_PASSES=${LLAMA_B1_ADAPTIVE_PASSES:-1}
        shape+=(-ub 4096)
        # Adaptive B1 bounds each pass, so the full prompt can stay in one batch.
        (( LLAMA_B1_ADAPTIVE_PASSES > 0 )) || shape+=(-b 32768)
        # Keep the small-card graph pad; the larger pad was measured on 24 GB cards.
        if (( card_mib < 23552 )); then
            export LLAMA_B1_TAIL_MIN_TOKENS=${LLAMA_B1_TAIL_MIN_TOKENS-1024}
            export LLAMA_B1_TAIL_UBATCH=${LLAMA_B1_TAIL_UBATCH-512}
        fi
        break
    fi
done <<< "$devices"
[[ " $* " == *" -c "* || " $* " == *" --ctx-size"* ]] || shape+=(-c 110592)
threads=0
(( $(nproc --all) < 16 )) || threads=8
if [[ -r /sys/devices/cpu_core/cpus ]]; then
    p_cores=$(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, -v cpus="$(cat /sys/devices/cpu_core/cpus)" '
        BEGIN {
            n = split(cpus, ranges, ",")
            for (i = 1; i <= n; ++i) {
                split(ranges[i], ends, "-")
                last = ends[2] == "" ? ends[1] : ends[2]
                for (cpu = ends[1]; cpu <= last; ++cpu) keep[cpu] = 1
            }
        }
        !/^#/ && ($1 in keep) { cores[$2 "," $3] = 1 }
        END { print length(cores) }
    ') || p_cores=0
    # Count each P-core once, even when its two threads are listed.
    [[ ! $p_cores =~ ^[1-9][0-9]*$ ]] || threads=$(( p_cores < 8 ? p_cores : 8 ))
fi
(( threads == 0 )) || shape+=(-t "$threads" -tb "$threads")
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
