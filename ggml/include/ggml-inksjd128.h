#pragma once

// INKPREFILL (fork): marks an SJ-KVaRN FLASH_ATTN_EXT node whose 256-wide heads are two packed 128-wide heads, as
// build_attn_rel_sj (src/llama-graph.cpp) builds them for Inkling 2: KV head 2i in lanes [0, 128) and 2i+1 in
// [128, 256) of cache head i; q head 2g*i + c holds [q, 0] for c < g and [0, q] for c >= g (g = q heads per 128-wide
// KV head), and only that half of each output row is read. A backend may then run the two real D 128 heads and leave the
// other half of each output row zero (CUDA: ggml-cuda/inksjd128.cu). Other backends ignore the flag.
// Header-only so that ggml.h stays untouched.

#include "ggml.h"

#define GGML_INK_SJ_D128_PARAM 13
#define GGML_INK_SJ_D128_MAGIC 0x494b44

static inline void ggml_flash_attn_ext_set_ink_sj_d128(struct ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT && a->src[6] != NULL && a->op_params[GGML_INK_SJ_D128_PARAM] == 0);
    a->op_params[GGML_INK_SJ_D128_PARAM] = GGML_INK_SJ_D128_MAGIC;
}
