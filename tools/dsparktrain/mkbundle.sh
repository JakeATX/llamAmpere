#!/bin/bash
# DSPARKTRAIN: assemble the rental upload on Taichi and list every staged file with its size.
#   $B/bin       : llama-server + llama-dspark-dump of this worktree behind the INKSTACK glibc shim (wrapper + .real +
#                  sys/ from /mnt/data/inkstack/rent/stack1/sys) + libs + libnccl.so.2
#   $B/wt        : tools/dsparktrain + gguf-py (this worktree), ink_dspark_fold.py
#   $B/data      : prompts.jsonl (prep_prompts.py), wiki.test.raw (calibration prompt only)
#   $B/w0        : target.pt, lm_truth.bf16, lm_truth.gguf (prep_weights.py; 2.8 GB)
#   models (uploaded / relayed separately, same names on the box): target AI4 fold gguf, drafter Q8_0, Q8_0-fold, BF16
#   python: /mnt/data/inkflags/venv (5.4 GB) + its interpreter /home/jakek/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu,
#           rsynced to the same absolute paths on the box (torch cu132 wheels: the box driver must support CUDA 13.2)
# usage: mkbundle.sh [B=/mnt/data/inkspec2/rent/dsparktrain]
set -eu
W=$(cd "$(dirname "$0")/../.." && pwd)
B=${1:-/mnt/data/inkspec2/rent/dsparktrain}
SRC=${SRC:-$W/build/bin}; SHIM=/mnt/data/inkstack/rent/stack1/sys
mkdir -p $B/bin $B/wt/tools $B/data $B/w0
cp -a $SRC/lib*.so* $B/bin/
cp -a $SHIM $B/bin/
[[ -f $B/bin/libnccl.so.2 ]] || cp /mnt/data/inkspec/build/bin/libnccl.so.2 $B/bin/
for x in llama-server llama-dspark-dump; do
  cp $SRC/$x $B/bin/$x.real
  cat > $B/bin/$x <<'EOF'
#!/bin/sh
# run a Taichi-built binary (glibc 2.43) on an older-glibc rental with the shipped loader + libc
D=$(dirname "$(readlink -f "$0")")
exec "$D/sys/ld-linux-x86-64.so.2" --library-path "$D:$D/sys:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu" "$D/$(basename "$0").real" "$@"
EOF
  chmod +x $B/bin/$x
done
rsync -a --delete --exclude __pycache__ $W/tools/dsparktrain $B/wt/tools/
rsync -a --delete --exclude __pycache__ $W/gguf-py $B/wt/
cp /mnt/data/inkflags/h/ink_dspark_fold.py $B/wt/
cp /mnt/data/inkspec2/data/prompts.jsonl /mnt/data/inkflags/data/wiki.test.raw $B/data/
for f in target.pt lm_truth.bf16 lm_truth.gguf prep_check.json; do cp -u /mnt/data/inkspec2/w0/$f $B/w0/; done
( cd $B && find . -type f -printf '%s\t%p\n' | sort -k2 ) > $B/STAGED.txt
{
  for f in /mnt/data/inkflags/enc/Inkling-2-Mini-A_all_ait4u-fold.gguf /mnt/data/inkling2/Inkling-2-Mini-DSpark-Q8_0.gguf \
           /mnt/data/inkling2/Inkling-2-Mini-DSpark-Q8_0-fold.gguf /mnt/data/inkling2/Inkling-2-Mini-DSpark-BF16.gguf; do
    printf '%s\t%s\n' "$(stat -c %s $f)" "$f"
  done
  printf '%s\t%s\n' "$(du -sb /mnt/data/inkflags/venv | cut -f1)" /mnt/data/inkflags/venv/
  UVP=$(readlink -f /home/jakek/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu); printf "%s\t%s (symlink target of the uv python; rsync both)\n" "$(du -sb $UVP | cut -f1)" "$UVP/"
} > $B/STAGED_MODELS.txt
echo "bundle $B: $(du -sh $B | cut -f1); + models/python:"; cat $B/STAGED_MODELS.txt
sha256sum $B/w0/target.pt $B/w0/lm_truth.bf16 > $B/w0/SHA256SUMS
