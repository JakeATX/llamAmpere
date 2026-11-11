#pragma once

// LSA fast path (llama-lsa.h, LLAMA_LSA_FAST=1): attention reads only the selected KV cells.
//
// Instead of masking the unselected cells of the full cache, the selected cells are copied out of the K and V caches
// into a small contiguous buffer (raw bytes, so quantized cells are not decoded or requantized), the KQ mask columns
// of those cells are gathered the same way, and the normal attention runs over the gathered cells only. With the
// same selection this gives the same result as attention over the full cache with every unselected cell masked to
// -inf, up to summation order. Header-only and ggml-only so test-backend-ops checks the exact graph the engine builds.

#include "ggml.h"

#include <cstdint>

// t: one layer's cache view [d, n_head_kv, n_kv] (heads of a cell packed back to back, cells nb[2] apart).
// ids: I32 [n_sel] cell indices. Returns [d, n_head_kv, n_sel] of t's type, contiguous. nullptr if t's layout does not
// allow a raw byte gather (caller stays on the masked path).
static inline ggml_tensor * llama_lsa_gather_cells(ggml_context * ctx, ggml_tensor * t, ggml_tensor * ids) {
    const size_t row_bytes  = ggml_row_size(t->type, t->ne[0]);
    const size_t cell_bytes = row_bytes*(size_t) t->ne[1];
    if (t->ne[3] != 1 || t->nb[0] != ggml_type_size(t->type) || t->nb[1] != row_bytes || t->nb[2] < cell_bytes ||
            cell_bytes % sizeof(int32_t) != 0 || t->nb[2] % sizeof(int32_t) != 0 ||
            ids->type != GGML_TYPE_I32 || !ggml_is_contiguous(ids) || ids->ne[1] != 1) {
        return nullptr;
    }
    // reinterpret each cell as cell_bytes/4 I32 words; GET_ROWS on I32 copies the words unchanged
    ggml_tensor * words = ggml_view_2d(ctx, t, t->ne[0], t->ne[2], t->nb[2], 0);
    words->type  = GGML_TYPE_I32;
    words->ne[0] = (int64_t) (cell_bytes/sizeof(int32_t));
    words->ne[1] = t->ne[2];
    words->nb[0] = sizeof(int32_t);
    words->nb[1] = t->nb[2];
    words->nb[2] = t->nb[2]*t->ne[2];
    words->nb[3] = words->nb[2];

    ggml_tensor * g = ggml_get_rows(ctx, words, ids); // I32 [cell_bytes/4, n_sel], contiguous

    ggml_tensor * r = ggml_view_2d(ctx, g, g->ne[0], g->ne[1], g->nb[1], 0);
    r->type  = t->type;
    r->ne[0] = t->ne[0];
    r->ne[1] = t->ne[1];
    r->ne[2] = ids->ne[0];
    r->ne[3] = 1;
    r->nb[0] = t->nb[0];
    r->nb[1] = row_bytes;
    r->nb[2] = cell_bytes;
    r->nb[3] = cell_bytes*(size_t) ids->ne[0];
    return r;
}

// mask: F32 [n_kv, T] (the causal/sequence mask of the full cache). Returns F32 [n_sel, T] = mask[ids[j], t].
static inline ggml_tensor * llama_lsa_gather_mask(ggml_context * ctx, ggml_tensor * mask, ggml_tensor * ids) {
    const int64_t n_kv = mask->ne[0];
    const int64_t T    = mask->ne[1];
    ggml_tensor * mt = T == 1 ? ggml_reshape_2d(ctx, ggml_cont(ctx, mask), 1, n_kv)
                              : ggml_cont(ctx, ggml_transpose(ctx, mask));             // [T, n_kv]
    ggml_tensor * g  = ggml_get_rows(ctx, mt, ids);                                    // [T, n_sel]
    return T == 1 ? ggml_reshape_2d(ctx, g, ids->ne[0], 1) : ggml_cont(ctx, ggml_transpose(ctx, g));
}

// Dense reference for the same selection: -inf everywhere except the selected cells, plus the original mask.
// mask: F32 [n_kv, T]; ids: I32 [n_sel] (one selection shared by all T rows). Returns F32 [n_kv, T].
static inline ggml_tensor * llama_lsa_dense_mask(ggml_context * ctx, ggml_tensor * mask, ggml_tensor * ids) {
    const int64_t n_kv = mask->ne[0];
    const int64_t T    = mask->ne[1];
    ggml_tensor * m = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_kv), -INFINITY); // one cell per row
    ggml_tensor * z = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, ids->ne[0]), 0.0f);
    m = ggml_set_rows(ctx, m, z, ids);
    m = ggml_reshape_2d(ctx, m, n_kv, 1);
    return ggml_add(ctx, ggml_repeat_4d(ctx, m, n_kv, T, 1, 1), mask);
}
