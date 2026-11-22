#pragma once

// INKKV (fork): Inkling 2 attention on the SJ-KVaRN 4/4 cache, 128-wide heads packed in pairs.
//
// A node marked with ggml_flash_attn_ext_set_inksj is an SJ-KVaRN FA node (src[6] set, D 256) whose 256-wide cache
// head i holds the 128-wide KV heads 2i (lanes [0, 128)) and 2i+1 (lanes [128, 256)), and whose Q packs each 128-wide
// q head into the half of its KV head: with r = n_head/n_head_kv (even), q head h holds [q, 0] when h % r < r/2 and
// [0, q] otherwise. Every logit is then the 128-wide one (the zero half meets the other KV head).
// Contract: Q is zero in the other half, and only the real half of each output row is defined (a backend may leave the
// other half zero instead of the other KV head's values). The LSE (src[9]), mask and sinks are unchanged.
// Slot: op_params[15] = GGML_INKSJ_MAGIC (free on SJ-KVaRN nodes; distinct from GGML_INKREL_MAGIC). Backends that do
// not read the marker compute the full packed attention, which agrees on the real halves.
// Header-only so that ggml.h stays untouched.

#include "ggml.h"

#define GGML_INKSJ_PARAM 15
#define GGML_INKSJ_MAGIC 0x494b53

static inline bool ggml_flash_attn_ext_is_inksj(const struct ggml_tensor * a) {
    return a->op == GGML_OP_FLASH_ATTN_EXT && a->src[6] != NULL && a->op_params[GGML_INKSJ_PARAM] == GGML_INKSJ_MAGIC;
}

static inline void ggml_flash_attn_ext_set_inksj(struct ggml_tensor * a) {
    GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT && a->src[6] != NULL && a->op_params[GGML_INKSJ_PARAM] == 0);
    const struct ggml_tensor * q = a->src[0];
    const struct ggml_tensor * k = a->src[1];
    GGML_ASSERT(q->ne[0] == 256 && k->ne[0] == 256 && a->src[2]->ne[0] == 256 && q->ne[2] % k->ne[2] == 0 && (q->ne[2]/k->ne[2]) % 2 == 0);
    a->op_params[GGML_INKSJ_PARAM] = GGML_INKSJ_MAGIC;
}

// the lanes [128*half, 128*half + 128) of q head h hold its real values (r = n_head/n_head_kv)
static inline int ggml_inksj_half(int h, int r) {
    return h % r >= r/2 ? 1 : 0;
}// INKKV far visible end: bits 8..31 of the fork flags entry (desc[15], GGML_SJKVARN_DESC_FLAGS_FORK; bits 0-1 are the
// TILE_SKIP / STAGE_ALL flags) of a far node's desc. > 0: no query of the node sees a mask column >= the end, and the
// SJ-KVaRN stream decode kernel skips the ring strips past it (fully masked, so exact). 0: no bound.
#define GGML_INKSJ_DESC_FLAGS    15
#define GGML_INKSJ_FAR_END_SHIFT 8


