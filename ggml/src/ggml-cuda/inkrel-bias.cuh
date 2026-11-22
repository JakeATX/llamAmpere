#pragma once

// INKPREFILL / INKKV shared helper: Inkling 2 relative-position attention bias read in-kernel.
// Interface version 1 (INK_REL_BIAS_VERSION). Additions keep old fields and meaning; a change of meaning bumps the version.
//
// What it computes
//   bias(h, c, t) = table[rel[c, t]] for head h and query row t
// where
//   table  = the per-query rel table the graph builds in build_attn_rel: (rel embedding x R) - sinks, F32.
//            Two layouts occur and both work through the strides:
//              [H, E+1, T]  (after CONT(PERMUTE(1,0,2,3)), what the GET_ROWS of the mask chain reads; s_h = 1)
//              [E+1, H, T]  (the MUL_MAT/SUB output before that permute; s_e = 1)
//   rel    = the table row for KV cell c (the mask's column) and query row t. Two sources, same result:
//            rel mode: rel_pos [n_kv, T] I32 as set_input_relative_pos fills it today.
//            pos mode: rel = d = qpos[t] - kpos[c] if 0 <= d < extent, else extent (kpos[c] = -1 for an empty cell).
//                      Same formula as set_input_relative_pos; it needs only T + n_kv ints instead of the
//                      [n_kv, T] tensor (1 GB at 128K cells x 2048 rows, filled on the CPU per ubatch).
//   c      = KV cell index in the FA K/V view, 0 .. n_kv-1 (same column as the FA mask).
//   t      = flattened query row: ts + Ts*s, with Ts = Q->ne[1] (rows per stream) and s the stream (Q->ne[3] index).
//   h      = query head, 0 .. H-1 (Q->ne[2]); the H query heads of one KV head (GQA) read their own bias.
//
// Where it goes
//   The bias is added where the FA mask is added: to the scaled (and softcapped) logit, before the max/exp:
//     logit = scale*q.k [-> softcap] + mask[c, t] + bias(h, c, t)
//   mask[c, t] is the shared causal/SWA base mask (ne[2] == 1, -inf for masked cells). Sinks stay as today (src[4]).
//   The sum is formed in F32. The materialised path instead rounds (bias + mask) to F16 before FA, so an in-kernel
//   read is not bit-identical to it (it is the more exact one); gate it with KLD, not identity.
//
// Host side: ink_rel_bias_make() validates the two tensors and fills the POD below; pass it by value to the kernel.

#include "common.cuh"

#define INK_REL_BIAS_VERSION 1

struct ink_rel_bias {
    const float   * table; // table + r*s_e + h*s_h + t*s_t
    int64_t         s_e;   // element strides of the table (not bytes)
    int64_t         s_h;
    int64_t         s_t;
    const int32_t * rel;   // rel mode: rel + c + t*r_t (nullptr = pos mode)
    int64_t         r_t;   // element stride of rel between query rows
    const int32_t * qpos;  // pos mode: qpos[t], kpos[c] (-1 = empty cell), extent
    const int32_t * kpos;
    int32_t         extent;
    int32_t         n_e;   // rows of the table (E+1): every rel value is in [0, n_e)
    int32_t         H;     // query heads
    int64_t         n_kv;  // KV cells covered by rel (rel->ne[0])
    int64_t         T;     // query rows (rel->ne[1] == table T)
};

// Row index for KV cell c of query row t: load once per (c, t), then ink_rel_bias_at() per head.
static __device__ __forceinline__ int32_t ink_rel_bias_row(const ink_rel_bias & b, const int64_t c, const int64_t t) {
    if (b.rel != nullptr) {
        return __ldg(b.rel + c + t*b.r_t);
    }
    const int32_t kp = __ldg(b.kpos + c);
    const int32_t d  = __ldg(b.qpos + t) - kp;
    return kp >= 0 && d >= 0 && d < b.extent ? d : b.extent;
}

static __device__ __forceinline__ float ink_rel_bias_at(const ink_rel_bias & b, const int32_t r, const int h, const int64_t t) {
    return __ldg(b.table + r*b.s_e + h*b.s_h + t*b.s_t);
}

static __device__ __forceinline__ float ink_rel_bias_load(const ink_rel_bias & b, const int h, const int64_t c, const int64_t t) {
    return ink_rel_bias_at(b, ink_rel_bias_row(b, c, t), h, t);
}

// Host: fill from the graph tensors (rel mode). table_h_fast = true for the [H, E+1, T] layout, false for [E+1, H, T].
// Returns false (and leaves out untouched) when types, contiguity of dim 0, or shapes do not match.
static inline bool ink_rel_bias_make(const ggml_tensor * table, const ggml_tensor * rel, const bool table_h_fast,
        ink_rel_bias & out) {
    if (table == nullptr || rel == nullptr || table->type != GGML_TYPE_F32 || rel->type != GGML_TYPE_I32 ||
            table->data == nullptr || rel->data == nullptr || table->ne[3] != 1 || rel->ne[2] != 1 || rel->ne[3] != 1 ||
            table->nb[0] != sizeof(float) || rel->nb[0] != sizeof(int32_t) ||
            table->nb[1] % sizeof(float) || table->nb[2] % sizeof(float) || rel->nb[1] % sizeof(int32_t) ||
            table->ne[2] != rel->ne[1]) {
        return false;
    }
    const int64_t H = table_h_fast ? table->ne[0] : table->ne[1];
    const int64_t E = table_h_fast ? table->ne[1] : table->ne[0];
    if (H < 1 || H > 256 || E < 1 || E > INT32_MAX) {
        return false;
    }
    ink_rel_bias b;
    b.table = (const float *) table->data;
    const int64_t s0 = 1, s1 = table->nb[1]/sizeof(float);
    b.s_h  = table_h_fast ? s0 : s1;
    b.s_e  = table_h_fast ? s1 : s0;
    b.s_t  = table->nb[2]/sizeof(float);
    b.rel  = (const int32_t *) rel->data;
    b.r_t  = rel->nb[1]/sizeof(int32_t);
    b.qpos = b.kpos = nullptr;
    b.extent = 0;
    b.n_e  = (int32_t) E;
    b.H    = (int32_t) H;
    b.n_kv = rel->ne[0];
    b.T    = rel->ne[1];
    out = b;
    return true;
}

// Host, pos mode: qpos [T] I32 (positions of the query rows), kpos [n_kv] I32 (cell positions, -1 = empty),
// extent = the table's far row (E = n_e - 1).
static inline bool ink_rel_bias_make_pos(const ggml_tensor * table, const ggml_tensor * qpos, const ggml_tensor * kpos,
        const int32_t extent, const bool table_h_fast, ink_rel_bias & out) {
    if (table == nullptr || qpos == nullptr || kpos == nullptr || table->type != GGML_TYPE_F32 ||
            qpos->type != GGML_TYPE_I32 || kpos->type != GGML_TYPE_I32 || table->data == nullptr ||
            qpos->data == nullptr || kpos->data == nullptr || table->ne[3] != 1 || table->nb[0] != sizeof(float) ||
            table->nb[1] % sizeof(float) || table->nb[2] % sizeof(float) ||
            !ggml_is_contiguous(qpos) || !ggml_is_contiguous(kpos) || ggml_nelements(qpos) != table->ne[2]) {
        return false;
    }
    const int64_t H = table_h_fast ? table->ne[0] : table->ne[1];
    const int64_t E = table_h_fast ? table->ne[1] : table->ne[0];
    if (H < 1 || H > 256 || E < 1 || E > INT32_MAX || extent != E - 1) {
        return false;
    }
    ink_rel_bias b;
    b.table  = (const float *) table->data;
    const int64_t s0 = 1, s1 = table->nb[1]/sizeof(float);
    b.s_h    = table_h_fast ? s0 : s1;
    b.s_e    = table_h_fast ? s1 : s0;
    b.s_t    = table->nb[2]/sizeof(float);
    b.rel    = nullptr;
    b.r_t    = 0;
    b.qpos   = (const int32_t *) qpos->data;
    b.kpos   = (const int32_t *) kpos->data;
    b.extent = extent;
    b.n_e    = (int32_t) E;
    b.H      = (int32_t) H;
    b.n_kv   = ggml_nelements(kpos);
    b.T      = table->ne[2];
    out = b;
    return true;
}
