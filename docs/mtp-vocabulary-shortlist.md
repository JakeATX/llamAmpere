# Draft-only vocabulary shortlist for the MTP drafter

`--spec-draft-vocab-map /path/to/map.txt` (shipped maps for Qwen3.8-27B ATX-IQ4_XS-M: `docs/mtp-vocab/atx_65536.txt`, the production choice, and `docs/mtp-vocab/atx_32768.txt`) (env `LLAMA_ARG_SPEC_DRAFT_VOCAB_MAP`; quick-test env
`LLAMA_SPEC_DRAFT_VOCAB` read by MTP draft contexts) restricts the **draft** context's output head to a
shortlist of token rows. The target context, its verification batches and the final sampler are unchanged.
Without a map the draft path is byte-for-byte the stock full-vocabulary path.

Applies to the sequential `draft-mtp` / `draft-mtp-adaptive` drafters of the Qwen3.5/3.6/3.8 dense
series (`llama_model_qwen35::graph_mtp`). The chained graph (`LLAMA_SPEC_CHAIN`) ignores the map.

## Map format

```text
llama-mtp-vocab-v1 VOCABULARY_SIZE SHORTLIST_SIZE
TOKEN_ID
TOKEN_ID
...
```

Sorted, unique token ids of the model's own tokenizer. The loader rejects a missing file, a bad header,
duplicate or out-of-range ids, missing or trailing data, and a vocabulary size that differs from the model.
A matching vocabulary size does not prove tokenizer identity: build the map with the same tokenizer.

## What runs

1. At draft-context creation the map is parsed once and uploaded to a persistent `I32 [n_sel]` tensor on
   the buffer type of the output head (128 KiB for 32K entries, 256 KiB for 64K). No per-eval upload, fixed
   shape and pointer, so CUDA graphs stay valid.
2. The draft graph views the existing quantized head (`output.weight` or `nextn.shared_head_head`) as
   one-row experts and computes `mul_mat_id(head, h, map)`: only the shortlisted rows are read, there is no
   second copy of the head. On CUDA this is a dedicated MMVQ specialization (`mul_mat_vec_q_indexed_rows`),
   ~0.19 ms for 32,768 Q6_K rows of width 5120 versus ~1.1-1.5 ms for the full 248,320-row head.
3. The compact logits `[n_sel, n_outputs]` feed the backend draft sampler directly. The sampler's candidate
   list is pre-seeded with the map, so `top_k` (and `temp`/`dist`) emit real token ids; greedy verify rows
   are mapped the same way. The host side (`common/sampling.cpp`) sees ordinary token ids and logits.

The shortlist is only taken when the fast path exists: head on the default CUDA device buffer, a supported
quantized type (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K..Q6_K, IQ*), no LoRA on the head, no per-row output scale, and
every output row of the ubatch owned by a sequence with a backend sampler (`--spec-draft-backend-sampling`,
the default). Otherwise the graph silently uses the full head, which keeps every output correct.

## Adaptive tail (`--spec-draft-vocab-hot`)

`--spec-draft-vocab-hot N` (env `LLAMA_ARG_SPEC_DRAFT_VOCAB_HOT`, default 0) declares that the **last N
entries** of the loaded map are *hot slots*: the server may repoint them at token ids seen in recent
traffic. `0 <= N < n_sel` is required; `N = 0` (the default) leaves the map fully static. The first
`n_sel - N` ids are the static part and can never be evicted; the initial contents of the hot slots are
whatever the file has, so a 71,680-id map with `N = 6144` starts out as a plain 71,680-id static map.

The shortlist size never changes, so the graph, the tensor and the sampler's candidate list are the same
objects as before: a refresh is one `ggml_backend_tensor_set` over the hot region (24 KiB for 6,144
slots). There is no per-decode-step work.

Ranking (`llama_mtp_hot_vocab` in `src/llama-mtp-vocab.h`, unit tested by `test-draft-vocab-hot`):

* `observe(toks)` is called once per request boundary - with the prompt just before it is decoded, and
  with the generated tail when the request finishes. Ids that are already in the static part are ignored.
* Each distinct id gets `score = score * 0.5^(dreq/half_life) + min(count, cap)` with `half_life = 8`
  requests and `cap = 8`. The cap is what makes recurrence across requests beat a single long document:
  one 5,000-token file contributes at most 8, an id seen once in each of three recent requests wins
  after ~14 requests of decay.
* `refresh()` keeps the `N` best-scoring ids, evicts the rest and fills the freed slots with the best
  non-resident candidates. Ids that stay keep their slot, so churn is minimal; ties are resolved in
  favour of what is already resident. The file-provided hot ids start at score 0 and go first, in slot
  order. The candidate table is pruned of decayed entries above `4*N` ids.
* The state is per context and per process. Nothing is written to disk, and nothing is shared between
  server restarts.

The server logs one line per request at INFO:
`slot update_slots: id  0 | task 3 | draft vocab hot: 12/6144 slots replaced, 8431 candidates`.

Offline coverage of the emitted-token distribution (`W3/data/adaptive_tail_coverage.json`): adding the
current prompt's ids to the 64K shortlist lifts coding coverage from 98.5% to 99.95%.

## Probability threshold (`--spec-draft-p-min`)

**Default `0` — the gate is off, and measurement says leave it off.** It is a draft-side EMISSION
gate: when the top candidate's probability falls below `p_min` the drafter stops extending the chain,
so the token is never proposed. That is independent of the p/q rejection test the target applies to
tokens that *were* proposed (`common/sampling.cpp:927`); p/q cannot rescue a token the drafter never
emitted.

**Which distribution `p` is measured against depends on the drafting path, and it changed when PQ1
shipped.** Since `3207e7fd7` (2026-09-10) exact p/q drafting is default-on, so the live chain builds
its candidates with `pq_sampler` (`common/speculative.cpp:1924`, selected at `:2235`) — the
**target-mirrored** sampler, running the request's own temperature / `top_k` / `top_p`. The gate at
`:2265` therefore reads `p` from that distribution.

Only under `LLAMA_SPEC_PQ=0` does the older description apply: the drafter samples on the backend with
a `top_k(10)` chain and the host softmaxes those ten logits, so `p` is renormalised over the
shortlist's top-10. With a shortlist the ten candidates are the top ten **of the shortlist**; when the
shortlist contains the full-vocabulary top-10 (the common case for a well-built map) `p` is identical
to the full-head value, and when it does not, the drafter proposes the best shortlisted token with a
`p` that slightly overstates its full-vocabulary probability. Either way the target verifies every
draft against its own full head, so this affects only how eagerly the drafter continues, never the
output distribution.

**What it costs and buys (H2, 2026-09-11, seven arms on one binary, temp 1.0, all three ship
fixtures).** `p_min 0.45` against `p_min 0` scores **G = −0.04%** on the weighted ship rule
(0.4 coding + 0.4 agentic + 0.2 rag) against a `2·SD_G` of 0.73% — indistinguishable. But it is not
uniform, and the per-fixture split is the useful part:

| fixture | tok/s vs `p_min 0` | mechanism |
|---|---|---|
| coding | −0.98% | drafts are already good; the gate only suppresses proposals that would have been accepted |
| agentic | −0.26% | as above |
| rag | **+2.29%** | `tok/pass` +2.53% at flat kernel, acceptance 0.827 vs 0.735 — the gate suppresses bad drafts on the workload where drafts are worst |

So the guidance is: **keep the shipped default at `0`**; a rag-only deployment is the one case with a
real reason to raise it. Do not describe `p_min` as a throughput win — on the ship mix it is a wash,
and on the two heaviest-weighted fixtures it is slightly negative.

Full write-up: `kernel_cache_investigation/frontier/H2_h1_acceptance_recovery/RESULT.md`.

## Memory

Indexed rows add only the map (<= 256 KiB) plus the smaller compact logits/sampler buffers. A packed
contiguous copy of the rows (131 MiB @32K, 262 MiB @64K for Q6_K) was benchmarked on the research branch at
203 us versus 192 us for indexed reads, so it is not built.

## Validation

```sh
build/bin/test-mtp-vocab
build/bin/test-draft-vocab-hot
build/bin/test-backend-ops test -o MUL_MAT_ID -p 'n_mats=17,n_used=.*b=1,m=1,n=1,k=512'
build/bin/test-backend-ops test -o MUL_MAT_ID
build/bin/test-backend-ops test -o ARGMAX
```

Server flags (Qwen3.8-27B example):

```sh
llama-server ... --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45 \
    --spec-draft-vocab-map /path/to/map_32768.txt
```

With the adaptive tail (65,536 static ids + 6,144 hot slots):

```sh
llama-server ... --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45 \
    --spec-draft-vocab-map /path/to/atx_65536_hot6144.txt --spec-draft-vocab-hot 6144
```

Two log lines confirm the map is loaded: `common_speculative_init: MTP draft context uses the draft-only
vocabulary shortlist '<path>'` is printed at the server's default verbosity, and libllama's
`draft vocabulary shortlist: 32768 of 248320 tokens ...` (tensor size and buffer) only with `--verbose`.
The target's greedy (temperature 0) trajectory does not depend on the drafter, so a greedy run with the map must
reproduce the full-head greedy output; sampled outputs (temperature > 0) are not byte-identical because a
different acceptance pattern changes how the sampler's RNG stream is consumed. Compare acceptance
(`draft_n_accepted/draft_n`), tokens per second and the quality protocol, and treat any repetition or restart
as a bug.
