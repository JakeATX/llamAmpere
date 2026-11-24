#pragma once

// INKATTN (fork): relative-position bias and output gate taken inside FLASH_ATTN_EXT (Inkling 2 decode).
//
// A node marked with ggml_flash_attn_ext_set_inkrel adds to every scaled logit of (q head h, key c, query row t)
//     f16(mask[c, t] + table[h, rel_pos[c, t], t])
// (the same f16 value the dense path's per-head mask holds), so the mask stays a broadcast [n_kv, T] F16 mask and the
// [n_kv, T, n_head] bias is never built. With a gate, the output is multiplied by sigmoid(gate[d + D*h, t]).
//   table   F32 [n_head, n_rel_rows, T]   (contiguous)
//   rel_pos I32 [n_kv, T]                  (row index into table dim 1, every value < n_rel_rows)
//   gate    F32 [D*n_head, T] or NULL      (contiguous)
// Slots: src[7] table, src[8] rel_pos, src[5] gate, op_params[15] = GGML_INKREL_MAGIC. One sequence (Q ne[3] == 1),
// no ALiBi / softcap, no LSE, no kv_rows, no SJ-KVaRN / LSA attachments. Backends: CPU (reference), CUDA (INKATTN).
// Header-only so that ggml.h stays untouched.

#include "ggml.h"

#include <string.h>

#define GGML_INKREL_PARAM 15
#define GGML_INKREL_MAGIC 0x494b52

static inline bool ggml_flash_attn_ext_is_inkrel(const struct ggml_tensor * a) {
    return a->op == GGML_OP_FLASH_ATTN_EXT && a->op_params[GGML_INKREL_PARAM] == GGML_INKREL_MAGIC;
}

static inline void ggml_flash_attn_ext_set_inkrel(struct ggml_tensor * a, struct ggml_tensor * table,
        struct ggml_tensor * rel_pos, struct ggml_tensor * gate) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT && !ggml_flash_attn_ext_is_inkrel(a));
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        GGML_ASSERT(a->src[i] == NULL && "INKREL: FA extras (SJ-KVaRN, LSA, kv_rows, LSE) are not combined with it");
    }
    for (int i = 5; i < GGML_MAX_OP_PARAMS / (int) sizeof(int32_t); ++i) {
        GGML_ASSERT(a->op_params[i] == 0);
    }
    const struct ggml_tensor * q    = a->src[0];
    const struct ggml_tensor * k    = a->src[1];
    const struct ggml_tensor * mask = a->src[3];
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) a->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) a->op_params + 2, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f && softcap == 0.0f);
    GGML_ASSERT(q->ne[3] == 1 && mask != NULL && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(table->type == GGML_TYPE_F32 && ggml_is_contiguous(table) && table->ne[0] == q->ne[2] &&
                table->ne[2] == q->ne[1] && table->ne[3] == 1);
    GGML_ASSERT(rel_pos->type == GGML_TYPE_I32 && rel_pos->ne[0] == k->ne[1] && rel_pos->ne[1] == q->ne[1] &&
                rel_pos->ne[2] == 1 && rel_pos->ne[3] == 1 && rel_pos->nb[0] == sizeof(int32_t));
    if (gate) {
        GGML_ASSERT(gate->type == GGML_TYPE_F32 && ggml_is_contiguous(gate) && gate->ne[0] == a->src[2]->ne[0]*q->ne[2] &&
                    gate->ne[1] == q->ne[1] && gate->ne[2] == 1 && gate->ne[3] == 1);
    }
    a->src[5] = gate;
    a->src[7] = table;
    a->src[8] = rel_pos;
    a->op_params[GGML_INKREL_PARAM] = GGML_INKREL_MAGIC;
}

static inline const struct ggml_tensor * ggml_flash_attn_ext_inkrel_table(const struct ggml_tensor * a) {
    return ggml_flash_attn_ext_is_inkrel(a) ? a->src[7] : NULL;
}

static inline const struct ggml_tensor * ggml_flash_attn_ext_inkrel_pos(const struct ggml_tensor * a) {
    return ggml_flash_attn_ext_is_inkrel(a) ? a->src[8] : NULL;
}

static inline const struct ggml_tensor * ggml_flash_attn_ext_inkrel_gate(const struct ggml_tensor * a) {
    return ggml_flash_attn_ext_is_inkrel(a) ? a->src[5] : NULL;
}

// INKREL factor form (fork, INKGQA-3; the graph builds it under LLAMA_INKFA_RELF=1, CUDA runs it in inkgqa.cu):
// the bias table is not materialised. Same node as above except
//   src[7] rf      F32 [16, extent + n_head*T] (contiguous): rows [0, extent) = the relative embedding E (row d =
//                  the 16 factors of distance d), row extent + t*n_head + h = the 16 factors r of query row t, head h
//   src[4] sinks   F32 [n_head, T] (contiguous): one sink logit per (head, row)
// and every logit adds f16(mask[c, t] + (d < extent ? r . E[d] : 0)) with d = rel_pos[c, t] (extent = the far row).
// With the per-row sink logit s this is the same softmax as the table form's f16(mask + r . E[d] - s) with a zero sink
// (up to the f16 rounding of the bias). op_params[GGML_INKREL_F_PARAM] = GGML_INKREL_F_MAGIC marks it.
#define GGML_INKREL_F_PARAM 14
#define GGML_INKREL_F_MAGIC 0x494b46

static inline bool ggml_flash_attn_ext_is_inkrel_factor(const struct ggml_tensor * a) {
    return ggml_flash_attn_ext_is_inkrel(a) && a->op_params[GGML_INKREL_F_PARAM] == GGML_INKREL_F_MAGIC;
}

// extent of a factor node (rows of E in rf)
static inline int64_t ggml_flash_attn_ext_inkrel_factor_extent(const struct ggml_tensor * a) {
    return a->src[7]->ne[1] - a->src[0]->ne[2]*a->src[0]->ne[1];
}

static inline void ggml_flash_attn_ext_set_inkrel_factor(struct ggml_tensor * a, struct ggml_tensor * rf,
        struct ggml_tensor * rel_pos, struct ggml_tensor * gate, struct ggml_tensor * row_sinks) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT && !ggml_flash_attn_ext_is_inkrel(a) && a->src[4] == NULL);
    for (int i = 5; i < GGML_MAX_SRC; ++i) {
        GGML_ASSERT(a->src[i] == NULL && "INKREL: FA extras (SJ-KVaRN, LSA, kv_rows, LSE) are not combined with it");
    }
    for (int i = 5; i < GGML_MAX_OP_PARAMS / (int) sizeof(int32_t); ++i) {
        GGML_ASSERT(a->op_params[i] == 0);
    }
    const struct ggml_tensor * q    = a->src[0];
    const struct ggml_tensor * k    = a->src[1];
    const struct ggml_tensor * mask = a->src[3];
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) a->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) a->op_params + 2, sizeof(float));
    GGML_ASSERT(max_bias == 0.0f && softcap == 0.0f);
    GGML_ASSERT(q->ne[3] == 1 && mask != NULL && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(rf->type == GGML_TYPE_F32 && ggml_is_contiguous(rf) && rf->ne[0] == 16 && rf->ne[1] > q->ne[2]*q->ne[1] &&
                rf->ne[2] == 1 && rf->ne[3] == 1);
    GGML_ASSERT(row_sinks->type == GGML_TYPE_F32 && ggml_is_contiguous(row_sinks) && row_sinks->ne[0] == q->ne[2] &&
                row_sinks->ne[1] == q->ne[1] && row_sinks->ne[2] == 1 && row_sinks->ne[3] == 1);
    GGML_ASSERT(rel_pos->type == GGML_TYPE_I32 && rel_pos->ne[0] == k->ne[1] && rel_pos->ne[1] == q->ne[1] &&
                rel_pos->ne[2] == 1 && rel_pos->ne[3] == 1 && rel_pos->nb[0] == sizeof(int32_t));
    if (gate) {
        GGML_ASSERT(gate->type == GGML_TYPE_F32 && ggml_is_contiguous(gate) && gate->ne[0] == a->src[2]->ne[0]*q->ne[2] &&
                    gate->ne[1] == q->ne[1] && gate->ne[2] == 1 && gate->ne[3] == 1);
    }
    a->src[4] = row_sinks;
    a->src[5] = gate;
    a->src[7] = rf;
    a->src[8] = rel_pos;
    a->op_params[GGML_INKREL_PARAM]   = GGML_INKREL_MAGIC;
    a->op_params[GGML_INKREL_F_PARAM] = GGML_INKREL_F_MAGIC;
}
