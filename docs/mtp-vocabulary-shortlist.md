# Draft-only vocabulary shortlist for the MTP drafter

`--spec-draft-vocab-map /path/to/map.txt` (env `LLAMA_ARG_SPEC_DRAFT_VOCAB_MAP`; quick-test env
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

## Probability threshold (`--spec-draft-p-min`)

The production drafter samples on the backend with a `top_k(10)` chain and the host softmaxes those ten
logits; `p_min` is applied to the top-1 of that renormalised distribution. With a shortlist the ten
candidates are the top ten **of the shortlist**, so `p` is renormalised over the shortlist's top-10. When the
shortlist contains the full-vocabulary top-10 (the common case for a well-built map) `p` is identical to the
full-head value; when it does not, the drafter proposes the best shortlisted token with a `p` that slightly
overstates its full-vocabulary probability. The target still verifies every draft against its own full
head, so this only affects how eagerly the drafter continues, never the output distribution.

## Memory

Indexed rows add only the map (<= 256 KiB) plus the smaller compact logits/sampler buffers. A packed
contiguous copy of the rows (131 MiB @32K, 262 MiB @64K for Q6_K) was benchmarked on the research branch at
203 us versus 192 us for indexed reads, so it is not built.

## Validation

```sh
build/bin/test-mtp-vocab
build/bin/test-backend-ops test -o MUL_MAT_ID -p 'n_mats=17,n_used=.*b=1,m=1,n=1,k=512'
build/bin/test-backend-ops test -o MUL_MAT_ID
build/bin/test-backend-ops test -o ARGMAX
```

Server flags (Qwen3.8-27B example):

```sh
llama-server ... --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45 \
    --spec-draft-vocab-map /path/to/map_32768.txt
```

Two log lines confirm the map is loaded: `common_speculative_init: MTP draft context uses the draft-only
vocabulary shortlist '<path>'` is printed at the server's default verbosity, and libllama's
`draft vocabulary shortlist: 32768 of 248320 tokens ...` (tensor size and buffer) only with `--verbose`.
The target's greedy (temperature 0) trajectory does not depend on the drafter, so a greedy run with the map must
reproduce the full-head greedy output; sampled outputs (temperature > 0) are not byte-identical because a
different acceptance pattern changes how the sampler's RNG stream is consumed. Compare acceptance
(`draft_n_accepted/draft_n`), tokens per second and the quality protocol, and treat any repetition or restart
as a bug.
