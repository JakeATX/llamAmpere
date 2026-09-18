# llamAmpere v0.3.1 release notes

## Summary

v0.3.1 is a format release. v0.3 made one quant fast; v0.3.1 widens the fork to three more weight
formats and keeps the v0.3 decode stack underneath them unchanged.

Three things are new. EXL3 (exllamav3 trellis) weights are now GGUF-native types with an SM86 CUDA
decode path, so an exllamav3 checkpoint can be repacked and served by `llama-server` with no Python and
no second runtime. Prism ML's Ternary Bonsai 2 27B runs on CPU and CUDA, with SM86 decode kernels for
both of its ternary containers. The IQ3 MMVQ path gains an opt-in shared-memory codebook. The branch is
also a full upstream catch-up: 772 commits of upstream llama.cpp and TurboQuant ahead of the v0.3 base.

Nothing in this release changes the ATX IQ4_XS configuration from v0.3. The flags in
[QWEN_AMPERE.md](../../QWEN_AMPERE.md#build-and-run) are the same, and all v0.3 numbers stand.

## What is new

### EXL3 weights as GGUF-native types

Seven ggml types, `GGML_TYPE_EXL3_2` through `GGML_TYPE_EXL3_8` (ids 51 to 57), a CPU reference decode,
a CUDA trellis-direct GEMV for up to 16 activation columns, a reconstruct-plus-cuBLAS path for prefill,
and a converter, verifier and numpy reference decoder under `scripts/exl3/`. The `.suh`/`.svh` side
vectors and the 128-block Hadamard glue ride on the matmul node itself, so a whole EXL3 matmul is one
graph node. The Qwen3.5/3.8 loader and graph also understand the fused-group layout (`attn_qkv`,
`ffn_up = [gate | up]`) that the converter's `--fuse` writes, which turns two or three GEMVs per block
into one.

Measured on one RTX 3090 Ti with Qwen3.8-27B EXL3 4.0 bpw on the release build: 40.6 tok/s single-token
(3 seeds, sd 0.15) and 81.9 tok/s (3 seeds, sd 0.82) at a 20K prompt with the model's MTP head at depth 4,
temperature 1.0, 20,480-token cap, every run ending on EOS; 73.72 (sd 0.59) at a 50K prompt on the
development build. Peak whole-card VRAM 17,154 MiB at 20K with the drafter and 17,893 MiB at 50K. KL
divergence to the reference implementation on the same weights is 0.000221 ± 0.000057 with the same top
token on 99.6 % of positions, and the converter reconstructs its checked tensors with `max|Δ| = 0`.

Full documentation, including conversion, verification and the known limits: [docs/exl3.md](../exl3.md).

### Ternary Bonsai 2: PTQ1_0 and PQ2_0 SM86 decode kernels

CPU and CUDA inference for Prism ML's Ternary Bonsai 2 27B, plus decode kernels for both containers.
`PQ2_0` (2-bit codes) and `PTQ1_0` (5 trits per byte) hold bit-identical ternary values and scales; they
differ only in packing. At 16K context on a 3090 Ti, `PQ2_0` decodes at 69.7 tok/s single-token and 104.8
tok/s with the MTP drafter, `PTQ1_0` at 62.8 and 59.1 tok/s, for about 1.1 GB less peak VRAM (release
build, one 16K rag prompt, two seeds, full runs of at least 5,000 generated tokens at temperature 1.0;
the PTQ1_0 drafter does not pay at temperature 1.0 at any depth from 1 to 3, so run it single-token). Both are
bit-exact against the Prism reference: the 16-token greedy hash is unchanged.

Model, build, validation and credits: [docs/bonsai2.md](../bonsai2.md).

### IQ3 shared-memory codebook, opt-in

`GGML_CUDA_SM86_IQ3_SMEM_GRID=1` stages the IQ3_XXS/IQ3_S grid into shared memory once per block instead
of reading it from global memory per lookup. Same values, same accumulation order, bit-identical output
(verified byte for byte on both types). Off by default, and only taken on cc 8.6.

### Upstream sync

This release branches from `ws/v0.3.1-upstream`: two true merge commits on top of the v0.3 base
(`617f92f23`), no rebase, no fork commit left behind.

- `96087172f` rolls TurboQuant forward to `407f3237b` (41 commits).
- `ef9ed337f` merges ggml-org/master at `b49650adb` (728 commits; 99 conflicted files, resolved hunk by hunk).

Upstream work that is now in the fork: the GDN normalization fix (PR 28068), the branchless Q4_K/Q5_K
unpack with L2 prefetch, the f16 flash-attention divergent-barrier fix, DFlash2, draft-mtp with embeddings,
synthetic speculative acceptance, and the per-hardware MMVQ to MMQ crossover tables.

Kept ours where it matters on this card: the SM86 MMVQ crossover is unchanged (upstream's new tables
only touch Ada, Blackwell, GB10, Orin and GCN); the exact p/q rejection loop stays the default; a slot
context above `n_ctx_train` warns instead of being capped; the CUDA-graph shape-key cache and its kill
switch; and the 21 turbo K/V flash-attention instances are always compiled regardless of
`GGML_CUDA_FA_QUANTS`.

Dropped: the fork's Metal TurboQuant kernels and the Vulkan TurboQuant host wiring. Upstream
restructured both backends, so re-attaching them is a port rather than a merge. v0.3.1 is a CUDA SM86
release; Metal or Vulkan users of TurboQuant should stay on v0.3 until that port lands (the removed code
is preserved as patches for it). The Metal and Vulkan moe-cache registrations are intact.

One upstream change alters model output, and it is kept on purpose. PR 28068 (`5fdfa6282`) makes the gated
delta net q/k normalization match the reference flash-linear-attention definition, `x * rsqrt(sum(x*x) + eps)`,
where every GDN model in the tree had been calling `ggml_l2_norm`, which is `x / max(sqrt(sum(x*x)), eps)` and at
these magnitudes never applies the epsilon at all. Qwen3.8 is a GDN model, so v0.3.1 normalizes those layers the way
the model was trained and v0.3 did not. Greedy continuations therefore differ from v0.3 after the first near-tie: on the
100K coding fixture the v0.3 build reproduces its documented hash `70d92f3ff9eac692` and v0.3.1 gives
`02d91ab519c9ecba`, at the same decode rate and draft acceptance (94.3 versus 94.9 tok/s, 0.85 versus 0.88 on that
run). A control build of the v0.3.1 tree with only that helper reverted reproduces `70d92f3ff9eac692` exactly, so
nothing else in the sync or the new formats changed the ATX path bit for bit. The exact p/q speculative path is
unaffected: on every format tested, the MTP output equals that build's single-token output.

One defect came in with the merge and is fixed here: `gguf-py/gguf/constants.py` had eight `MODEL_TENSOR` members defined twice
(`HC_ATTN_NORM/DOWN/UP/INJECT`, `A_ENC_SE_CONV1/2`, `A_ENC_ASP_ATTN/TDNN`), which made `import gguf`
fail outright with "already defined". Each is now declared once, as upstream master has it.

One behaviour note carried over from the merge: `common_ngram_cache_save` takes a vocabulary id. The
parameter defaults to "unknown", so existing callers are source-compatible, but `lookup-create` and
`lookup` now stamp a real vocabulary id into the caches they write.

## Per-format speed-up blurbs

**IQ4_XS, the ATX path (from v0.3, unchanged here).** IQ4_XS is the fastest format on SM86 at the widths
speculative verification actually runs, so the ATX recipe builds on it and the kernel work targets it.
Two changes carry the speed. The MMVQ launch table became per-width instead of one entry for all widths,
and `rows_per_cuda_block` is now 8 at the verify widths, so eight output rows share each activation load
rather than each reloading it. Cross-column weight reuse means a block is unpacked once and dotted
against every column in flight instead of being re-unpacked per column. The launch geometry is worth
+8.40 % on the kernel and +4.88 % at model level, and it is bit-exact.

**IQ3_XXS and IQ3_S, shared-memory codebook (new, opt-in).** These types are issue-bound rather than
bandwidth-bound: every weight costs a lookup into a codebook table that lives in global memory, and the
dependent load is what the kernel waits on. The grid is small enough for both types that a block can
cooperatively copy it into shared memory once and then read it at shared-memory latency for the rest of
its work. The values and the accumulation order are unchanged, so output is bit-identical. In an
env-toggled A/B in one binary at m=4096, k=14336, the win concentrates exactly where MTP verifies: IQ3_S
is +10.07 % at width 4 and +9.13 % at width 2, IQ3_XXS +5.52 % and +5.25 %, and the win is gone by width
5 to 8. It ships off by default because there is no end-to-end model-level measurement yet.

**PTQ1_0, weight reuse and dp4a on digits (new).** PTQ1_0 packs five trits into a byte, so unpacking is
a multiply-by-three digit extraction and the GEMV is ALU-bound rather than bandwidth-bound. Three
changes. Verify widths 2 to 8 now run a cross-column reuse kernel: a 128-weight block is split over four
lanes, each lane unpacks its quarter once and dots every column, where the upstream path re-unpacked the
block per column per row and spilled 540 to 1260 bytes per thread, which made a width-4 verify cost as
much as 4.4 single-token steps. Rows per block at widths 2 to 4 drop from 8 to 2 to pay for the
registers that buys. And the dp4a products now run on the unsigned digits 0 to 2, with the exact
per-sub-block activation sum subtracted once at the end, instead of a byte-wise correction on every
four-weight group; that alone cut the fused width-1 kernel from 3,224 to 2,520 SASS instructions and took single-token decode at 16K from 55.5 to 64.3 tok/s (2K check, same fixture and seed family). The
width-1 path is bit-exact: the 16-token greedy hash is unchanged and `test-backend-ops` reports zero
failures. The reuse kernel at widths 2 to 8 accumulates in a different order from the width-1 kernel, so a
greedy speculative continuation can part from the single-token continuation at a near-tie; on a 256-token
greedy check it did once, and with `GGML_CUDA_SM86_PTQ1_REUSE=0` the speculative text matched the
single-token text again. At temperature 1.0 this is invisible; it is noted here so a greedy diff is not
mistaken for a defect. PQ2_0 has no such path and its speculative and single-token texts agree.

**PQ2_0, the byte-permute unpack (new).** PQ2_0 stores the same ternary values and the same block scales
as PTQ1_0 in a 2-bit container, so it trades about 1.1 GB of VRAM for an unpack that is a single byte
permute per four weights instead of a multiply chain. That moves the kernel off the integer pipe and
back onto bandwidth, which is where a decode kernel wants to be. It matters most under speculative
decoding, because a width-4 verify re-reads the same weights and the byte-permute path carries almost no
re-unpack cost: on the same 16K fixture in full temperature-1.0 runs on the release build, PQ2_0 goes
from 69.7 tok/s single-token to 104.8 tok/s with the MTP drafter, where PTQ1_0 goes 62.8 to 59.1 (the
2K development checks read 64 to 59 after the dp4a rewrite and 55 to 60 before it): on PTQ1_0 a width-4
verify pass still costs about 2.85 single-token steps, so the drafter loses slightly there. The two containers were checked tensor by tensor and are a lossless
re-container of each other, so this is pure packing, not a quality trade.

**EXL3, the trellis GEMV (new).** The first working path reconstructed each whole weight matrix to f16
and called cuBLAS, which is correct and slow: 20.1 tok/s. The shipped kernel decodes straight from the
trellis stream. At 4 bits and under, a 16x16 tile is at most 32 words, so one lane loads one word and
the second word covering its eight bit windows arrives by shuffle, turning four scattered loads per lane
into one coalesced 128 B request per warp per tile. The decoded pairs then feed `mma.m16n8k16` tensor-core
instructions, because the trellis tile's lane order already is the B fragment layout, which makes the
cost per tile independent of the verify width. Finally the scale and Hadamard glue moved into two small
kernels attached to the matmul node instead of five graph ops, and the output glue reduces four split-K
rows in flight rather than one serial chain, taking it from 12.9 to 2.57 microseconds at width 5. Net:
20.1 to 41.3 tok/s single-token and 56.7 to 76.1 in an MTP smoke, with byte-identical greedy output at
every step.

## Deferred to v0.4

- **W58, the width 5 to 8 MMVQ launch tables.** Measured and tuned, not on this branch. The launch
  geometry upstream drifted (`calc_nwarps` gained parameters), so the tables have to be re-expressed
  against the new signature rather than applied as written.
- **The n-gram speculative path.** The persistent lookup cache, the CLI options, the splice policy and
  the draft-length statistics stay out of v0.3.1. Every n-gram policy measured so far loses to MTP
  because verify widths 5 to 9 cost 2.2 to 3.1x on the current MMVQ, so this is gated on W58 landing.
  The v0.4 target for it is narrower than "beat MTP": an option for 8, 10 and 12 GB cards to skip the
  MTP head entirely when context is tight, where the head's weights and its draft KV do not fit.
- **PTQ1_0 verify widths 2 to 4.** The dp4a rewrite made single-token PTQ1_0 decode faster than the
  MTP path at 16K on this container; the width 2 to 4 kernels are the next PTQ1_0 target so the drafter
  pays off there too.
- **A 6-bit TurboQuant K cache (TQ6) to pair with TQ3 V.** The idea is K at roughly 8-bit quality in
  6 bits for tight-VRAM configurations (tq6/tq3), to be explored and measured before anything ships.
- **MTP depth 4 as a product default.** Depth 4 with `p-min 0` won the ship corpus and is the
  recommended setting, but `common_params_speculative::n_max` is still 3 in the code. Changing the
  default is a v0.4 change.
- **Fused norm / FWHT / quantize on the ternary decode path.** The ternary decode issues about 2,063
  kernel launches per token; the FWHT is 3.0 % of GPU time and `quantize_q8_1` 2.2 %, so fusing them
  caps at roughly 5 % and is only worth doing bundled with a wider launch-count pass.
- **PTQ1_0 MMQ J=128 prefill tiles.** `mmq.cuh` has J=128 tiles for PQ2_0 but PTQ1_0 stops at J=64:
  843.5 against 1561 tok/s on pp512, which at 100K context is roughly 125 s of time-to-first-token
  against 65 s.
- **EXL3 prefill.** Anything wider than 16 activation columns still reconstructs the weight and calls
  cuBLAS, 2.8x slower than the ATX path. A trellis-direct GEMM is the fix.
- **EXL3 `test-backend-ops` coverage, and EXL3 MoE.** No cases exist for the new types, and
  `GGML_OP_MUL_MAT_ID` already uses `src[2]`, which is where the EXL3 side tensors live.
- **IQ3 shared-memory codebook on by default,** and the same idea for IQ2. Both need an end-to-end
  measurement first.

## Attribution and licenses

**EXL3.** The format, the "mul1" codebook arithmetic, the tile layout and the trellis GEMV design are
Turboderp's, from [exllamav3](https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c)
2025 Turboderp. The license text is in `licenses/LICENSE-exllamav3` and the attribution is repeated in
the header of every EXL3 source file. EXL3 is Turboderp's streamlined variant of QTIP (Tseng, Sun, Hou,
De Sa, NeurIPS 2024, arXiv:2406.11235). The code in this fork is an independent ggml/CUDA
reimplementation written with the exllamav3 sources open as the reference; a line-level audit found zero
verbatim or renamed functions. No other project is credited for this format, and none has a claim on it.

**Ternary Bonsai 2.** The model is Prism ML's (weights Apache 2.0). The PQ2_0 and PTQ1_0 formats, the
folded signed-Hadamard runtime and the CPU/CUDA kernels this port builds on are adapted from Prism ML's
llama.cpp fork (MIT, same license text as this repository) at the pinned commit listed in
[docs/bonsai2.md](../bonsai2.md), which also carries the citation Prism ML asks for.

**Upstream.** llama.cpp is ggml-org's, MIT. The KV cache and native MTP speculative decoding this fork
was built on come from TurboQuant. Both are merged, not vendored.

## Known TBDs

Numbers that are not in this document because no measurement supports them yet:

- End-to-end model-level gain of `GGML_CUDA_SM86_IQ3_SMEM_GRID`: **TBD**. Only kernel microbenchmark
  deltas exist, and the sweep that produced them ran a configuration production does not.
- Peak VRAM at 100K and 200K context, per format, on the v0.3.1 build: measured (whole card, context
  filled to the stated depth, q8_0 K / turbo3 V, MTP drafter loaded). 100K: ATX IQ4_XS 19,670 MiB,
  EXL3 4.0 18,859, PQ2_0 12,351, PTQ1_0 11,165. 200K: ATX 22,801, EXL3 22,032, PQ2_0 15,486,
  PTQ1_0 14,342. All four sit under the 23,552 MiB budget at 200K; see HIGHLIGHTS.md section 4.
- A v0.3.1 ladder against stock llama.cpp and TurboQuant, in the form v0.3 reports: **TBD**. No
  ship-corpus run has been made on this branch.
- PQ2_0 and PTQ1_0 on the production ship corpus (coding, agentic, rag, 3 seeds, 20K generated,
  temperature 1.0): the 16K rag cells are full runs on the release build (quoted above, two seeds each).
  The coding and agentic 16K fixtures end under 5,000 generated tokens on every format, ternary and ATX
  alike, so no ship number exists for them. At 100K the primary rag prompt ends under 5,000 tokens on
  every seed for both containers (1,500 to 4,750 generated), and at 64K on every seed but three PQ2_0
  ones, so those cells use whatever reached the floor: PQ2_0 64K no drafter 59.8 tok/s (two seeds), PQ2_0 64K MTP-3 88.4 (one seed of
  five), PQ2_0 100K on the fixture set's second prompt 54.5 no drafter and 82.9 MTP-3 (two seeds
  each, acceptance 0.53 to 0.55, peak 13,015 MiB). PTQ1_0 has no full run at 64K without the drafter
  on either prompt (ten runs, 2,135 to 4,427 tokens, 55.1 to 56.3 tok/s); with the drafter one seed of
  ten reached the floor (56.6 tok/s, second prompt); at 100K on the second prompt it decodes at 51.2
  tok/s without the drafter (two seeds, sd 0.02) and 52.7 with MTP-3 (two seeds, sd 0.41, acceptance
  0.54), so at 100K the drafter is a wash on PTQ1_0 rather than a loss.
- PQ2_0 prefill on the v0.3.1 build: **TBD**. The 1,418 tok/s figure on record is a 2K check on the
  earlier ternary build.
- EXL3 at bit widths other than 4.0 (and the 6-bit head): **TBD**. Types 2, 3, 5, 7 and 8 are
  implemented and parity-checked, but no end-to-end model has been converted or timed at those widths.
- File size and bits per weight of every shipped quant in one table: **TBD** beyond ATX-4-XS
  (14.5 GiB, 4.56 bpw), EXL3 4.0 bpw (13.7 GiB), PTQ1_0 (5,946,648,928 B, 1.75 bpw) and PQ2_0
  (7,206,168,928 B).
