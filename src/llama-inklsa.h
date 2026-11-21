#pragma once

// INKLSA (opt-in, lossy): latent sparse attention for the full-attention visits of Inkling 2 (inkling2_loop).
//
// Off unless LLAMA_INKLSA=1. For each full-attention visit the KV cache keeps a training-free index next to K:
// the mean key of every block of B cells (F16 [n_head_kv*head_dim, kv_size/B]). It is refreshed for the blocks each
// ubatch writes (re-read whole from the K cache after the store) and rebuilt from K after anything changes K outside
// the graph (state restore, K shift: llama_lsa_idx_invalidate calls llama_inklsa_invalidate).
//
// Decode/verify graphs (T <= LLAMA_INKLSA_MAX_T rows, n_kv >= LLAMA_INKLSA_MIN_KV) attend over a compact set only:
//   fixed cells  = the first SINKS cells + every visible cell that is not inside an eligible block (the window of
//                  >= WINDOW cells before the first row, which holds every cell with a relative-position bias)
//   picked blocks= the KEEP/B eligible blocks with the highest score, one pick per visit shared by all heads and rows:
//                  score_b = sum over q heads h and rows t of softmax_b(scale * q_ht . mean_k_{g(h), b})
// Eligible block: every cell non-empty, at distance >= WINDOW from every row and past the sinks. The model's bias is
// zero past the relative extent E (<= WINDOW), so a picked block's mask is the per-(head,row) constant of the
// padded bias row. Prefill and short contexts stay dense (the index is still refreshed).
//
// Requires a plain F16 K/V cache, flash attention, one sequence (cells == positions are checked per ubatch).

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

struct llama_inklsa_config {
    bool    on      = false; // LLAMA_INKLSA
    int64_t keep    = 4096;  // LLAMA_INKLSA_KEEP: middle cells kept (rounded up to a multiple of 256)
    int64_t block   = 64;    // LLAMA_INKLSA_BLOCK: cells per block (divides 256)
    int64_t window  = 1024;  // LLAMA_INKLSA_WINDOW: >= relative extent
    int64_t sinks   = 128;   // LLAMA_INKLSA_SINKS: multiple of BLOCK
    int64_t min_kv  = 16384; // LLAMA_INKLSA_MIN_KV: below this n_kv the visit stays dense
    int64_t max_t   = 16;    // LLAMA_INKLSA_MAX_T: wider ubatches stay dense
    int     reuse_pass = 0;  // LLAMA_INKLSA_REUSE_PASS=1: loop passes 1 and 2 reuse pass 0's pick of the same layer
    bool    stats   = false; // LLAMA_INKLSA_STATS: log the configuration once
};

const llama_inklsa_config & llama_inklsa_cfg();

// fixed-cell capacity of a sparse graph with up to max_t rows: sinks + window + one partial block + rows, to 256
int64_t llama_inklsa_n_fix();
// kept middle cells per sparse graph (multiple of 256 and of the block)
int64_t llama_inklsa_n_keep();

struct llama_inklsa_idx {
    ggml_tensor * t     = nullptr; // F16 [d, kv_size/B]
    bool          valid = false;   // false: this graph rebuilds every block from K
    uint64_t      epoch = 0;
};
// index of one K cache tensor (allocated on buft at first use, valid at creation: the cache starts empty)
bool llama_inklsa_idx_get(const ggml_tensor * kcache, int64_t d, int64_t n_blk_cap, ggml_backend_buffer_type_t buft,
        llama_inklsa_idx * out);
void llama_inklsa_idx_commit(const ggml_tensor * kcache, uint64_t epoch);
// nullptr / n == 0: every index
void llama_inklsa_invalidate(const ggml_tensor * const * kcaches, size_t n);
// cache tensors destroyed
void llama_inklsa_forget(const ggml_tensor * const * kcaches, size_t n);

// per-graph inputs of the INKLSA path, shared by every full-attention visit of the graph (llama-graph.cpp builds
// them, llama_kv_cache::set_input_inklsa fills them)
struct llama_inklsa_inputs {
    bool          sparse   = false;   // this graph attends over the compact set (else: dense, index refresh only)
    int64_t       B        = 0;
    int64_t       n_blk    = 0;       // n_kv / B
    int64_t       n_fix    = 0;
    int64_t       n_u      = 0;       // blocks the ubatch may write
    ggml_tensor * fix_ids  = nullptr; // I32 [n_fix]           fixed cells (sinks + window), padding = cell 0
    ggml_tensor * fix_rel  = nullptr; // I32 [n_fix, T]        bias-table row of each fixed cell per query row
    ggml_tensor * fix_mask = nullptr; // F32 [n_fix, T]        0 visible, -inf padding / future / other sequence
    ggml_tensor * elig     = nullptr; // F32 [n_blk]           0 eligible block, -inf otherwise
    ggml_tensor * u_blk    = nullptr; // I32 [n_u]             blocks this ubatch writes (refresh), padded with the last
    ggml_tensor * u_blk64  = nullptr; // I64 [n_u]             same, for set_rows
    ggml_tensor * onehot   = nullptr; // F32 [1, n_head_kv, n_head, 1]  KV group of each q head
};
