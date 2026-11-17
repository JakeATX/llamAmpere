#!/usr/bin/env bash
# CPU-only checks of the card choice and the busy-card planner switch in scripts/flash-next-server.sh (#96). A stub
# server prints a fake device list, then the arguments and the tuning environment it was started with; no GPU, model
# or real server is used. With a second launcher (for example the previous release's), idle-card cases must give
# byte-identical arguments and environment with both.
# Usage: tests/test-flash-next-launcher.sh [launcher [previous-launcher]]
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
launcher=${1:-$repo/scripts/flash-next-server.sh}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
previous=
if [[ -n ${2-} ]]; then
    # run the previous launcher from a copy of this tree's layout, so both read the same data files
    mkdir -p "$tmp/prev/scripts"; cp "$2" "$tmp/prev/scripts/flash-next-server.sh"; ln -s "$repo/data" "$tmp/prev/data"
    previous=$tmp/prev/scripts/flash-next-server.sh
fi
cat > "$tmp/server" <<'STUB'
#!/usr/bin/env bash
if [[ ${1-} == --list-devices ]]; then printf '%b' "${FAKE_DEVICES-}"; exit 0; fi
printf '%s\n' "$@"
env | grep -E '^(GGML_CUDA_MOE_CACHE_RESERVE_MB|LLAMA_PROMPT_TEMP_HEADROOM|LLAMA_B1_[A-Z_]+)=' | sort
STUB
chmod +x "$tmp/server"
card() { printf '  %s: NVIDIA GeForce RTX 3090 (%s MiB, %s MiB free)\\n' "$1" "$2" "$3"; }
fails=0
# run NAME DEVICES [VAR=VALUE ...] -- [launcher args]; sets $out (argv, one per line) and $err (stderr)
run() {
    local name=$1 devs=$2; shift 2
    local vars=()
    while [[ $1 != -- ]]; do vars+=("$1"); shift; done
    shift
    rc=0
    out=$(env -u LLAMA_ARG_DEVICE -u FLASH_NEXT_MULTI_GPU -u FLASH_NEXT_FIT_MIN_MIB LLAMA_SERVER_BIN="$tmp/server" \
          LLAMA_MOE_ARENA_OUTPUT_DIR="$tmp" FAKE_DEVICES="$devs" "${vars[@]}" bash "$launcher" "$@" 2> "$tmp/err") || rc=$?
    err=$(cat "$tmp/err")
    case_name=$name
    if [[ -n $previous && $name == same-* ]]; then
        local old old_err
        old=$(env -u LLAMA_ARG_DEVICE -u FLASH_NEXT_MULTI_GPU -u FLASH_NEXT_FIT_MIN_MIB LLAMA_SERVER_BIN="$tmp/server" \
              LLAMA_MOE_ARENA_OUTPUT_DIR="$tmp" FAKE_DEVICES="$devs" "${vars[@]}" bash "$previous" "$@" 2> "$tmp/old_err")
        old_err=$(cat "$tmp/old_err")
        old=${old//"$tmp/prev"/$repo}
        check '[[ $out == "$old" && $err == "$old_err" ]]'
    fi
}
# last value given for an option, as the server's parser keeps the last one
last() { awk -v o="$1" 'p { v = $0; p = 0 } $0 == o { p = 1 } END { print v }' <<< "$out"; }
has_err() { [[ $err == *"$1"* ]]; }
check() {
    if eval "$1"; then echo "PASS $case_name: $1"; else echo "FAIL $case_name: $1"; fails=$((fails + 1)); fi
}
idle24=$(card CUDA0 24576 24100)
idle24b=$(card CUDA1 24576 23900)
busy0=$(card CUDA0 24576 3469)
busy1=$(card CUDA1 24576 3387)

run same-one-idle-24 "$idle24" --
check '[[ $(last -ngl) == 999 && $(last --fit) == off && $(last --device) == "" && -z $err ]]'
run same-one-idle-12 "$(card CUDA0 12288 11800)" --
check '[[ $(last -ngl) == 999 && $(last --fit) == off && $(last -ub) == 4096 && -z $err ]]'
run same-no-devices "" --
check '[[ $(last -ngl) == 999 && $(last --fit) == off && $(last --device) == "" && -z $err ]]'
run two-idle "$idle24$idle24b" --
check '[[ $(last --device) == CUDA0 && $(last -ngl) == 999 && $(last --fit) == off ]] && has_err "2 CUDA cards visible; using CUDA0"'
check '! has_err "other programs" && ! has_err "memory planner" && ! has_err "experimental"'
run reporter "$busy0$busy1" --
check '[[ $(last --device) == CUDA0 && $(last -ngl) == auto && $(last --fit) == on && $(last -c) == "" && $(last --moe-skip) == off ]]'
check 'has_err "CUDA0 has 3469 of 24576 MiB free; other programs hold about 21107 MiB" && ! has_err "CUDA1 has" && has_err "memory planner"'
run busy-first "$busy0$idle24b" --
check '[[ $(last --device) == CUDA1 && $(last -ngl) == 999 && $(last --fit) == off && $(last --moe-skip) != off ]] && ! has_err "memory planner"'
run multi-idle "$idle24$idle24b" FLASH_NEXT_MULTI_GPU=1 --
check '[[ $(last --device) == "" && $(last -ngl) == 999 && $(last --fit) == off && $(last -c) == 110592 ]] && has_err "two-card Flash-Next is experimental" && ! has_err "memory planner"'
run multi-reporter "$busy0$busy1" FLASH_NEXT_MULTI_GPU=1 --
check '[[ $(last --device) == "" && $(last -ngl) == auto && $(last --fit) == on && $(last --moe-skip) == off ]] && has_err "CUDA0 has 3469" && has_err "CUDA1 has 3387" && has_err "on CUDA0 CUDA1;"'
run explicit-busy "$idle24$busy1" -- --device CUDA1
check '[[ $(grep -c "^--device$" <<< "$out") == 1 && $(last --device) == CUDA1 && $(last -ngl) == auto && $(last --fit) == on ]]'
run same-env-device "$busy0$idle24b" LLAMA_ARG_DEVICE=CUDA1 --
check '[[ $(last --device) == "" && $(last -ngl) == 999 && $(last --fit) == off && -z $err ]]'
run fit-min-off "$busy0$busy1" FLASH_NEXT_FIT_MIN_MIB=0 --
check '[[ $(last --device) == CUDA0 && $(last -ngl) == 999 && $(last --fit) == off && $(last --moe-skip) != off ]] && has_err "other programs" && ! has_err "memory planner"'
run caller-wins "$busy0$busy1" -- --fit off -ngl 999 --moe-skip on
check '[[ $(last -ngl) == 999 && $(last --fit) == off && $(last --moe-skip) == on ]]'
run same-one-idle-10 "$(card CUDA0 10240 9700)" --
check '[[ $(last -ngl) == 999 && $(last --fit) == off && $(last -c) == 110592 && -z $err ]]'
run same-explicit-idle "$idle24$idle24b" -- --device CUDA0
check '[[ $(grep -c "^--device$" <<< "$out") == 1 && $(last -ngl) == 999 && -z $err ]]'
run same-dev-lower "$idle24$idle24b" -- -dev cuda1
check '[[ $(last -dev) == cuda1 && $(last --device) == "" && $(last -ngl) == 999 && -z $err ]]'
run same-dev-none "$busy0$busy1" -- -dev none
check '[[ $(last -ngl) == 999 && $(last --fit) == off && -z $err ]]'
run same-cuda-vulkan "$idle24  Vulkan0: AMD Radeon (8192 MiB, 100 MiB free)\\n" --
check '[[ $(last --device) == "" && $(last -ngl) == 999 && -z $err ]]'
run device-eq-busy "$busy0$idle24b" -- --device=CUDA0
check '[[ $(last --device) == "" && $(last -ngl) == auto && $(last --fit) == on ]] && has_err "CUDA0 has 3469"'
run list-one-busy "$idle24$busy1" -- --device CUDA0,CUDA1
check '[[ $(grep -c "^--device$" <<< "$out") == 1 && $(last -ngl) == auto && $(last --fit) == on ]] && has_err "on CUDA1;" && ! has_err "CUDA cards visible" && has_err "two-card Flash-Next is experimental"'
run caller-ctx "$busy0$busy1" -- -c 32768
check '[[ $(last -c) == 32768 && $(grep -c "^-c$" <<< "$out") == 1 && $(last -ngl) == auto ]]'
run bad-fit-min "$busy0$busy1" FLASH_NEXT_FIT_MIN_MIB=8G --
check '[[ $rc == 2 && -z $out ]] && has_err "FLASH_NEXT_FIT_MIN_MIB must be"'
if (( fails )); then echo "$fails launcher check(s) failed"; exit 1; fi
echo "all launcher checks passed"
