// FLASH_ATTN_EXT src[9] is shared by upstream kv_rows (I32), the LSA LSE output (F32) and the LSA virtual-key V means
// (F16). The accessors must tell them apart: a kv_rows node has no LSE, an LSE node has no kv_rows.
#include "ggml.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

int main() {
    ggml_init_params ip = { 16*1024*1024, nullptr, /*no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    CHECK(ctx != nullptr);

    const int64_t D = 64, NQ = 4, H = 2, NKV = 32;
    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, NQ, H, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, H, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, NKV, H, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, NKV, NQ, 1, 1);

    // upstream kv_rows node
    ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, NKV, 1);
    ggml_tensor * fr   = ggml_flash_attn_ext_rows(ctx, q, k, v, mask, rows, 0.125f, 0.0f, 0.0f);
    CHECK(fr->src[9] == rows);
    CHECK(ggml_flash_attn_ext_get_kv_rows(fr) == rows);
    CHECK(ggml_flash_attn_ext_get_lse(fr) == nullptr);

    // plain node: neither
    ggml_tensor * fp = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    CHECK(ggml_flash_attn_ext_get_kv_rows(fp) == nullptr);
    CHECK(ggml_flash_attn_ext_get_lse(fp) == nullptr);

    // LSA LSE node (compensation): LSE, no kv_rows
    ggml_tensor * fl  = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    ggml_tensor * lse = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H, NQ, 1, 1);
    ggml_flash_attn_ext_set_lse(fl, lse);
    CHECK(fl->src[9] == lse);
    CHECK(ggml_flash_attn_ext_get_lse(fl) == lse);
    CHECK(ggml_flash_attn_ext_get_kv_rows(fl) == nullptr);

    // not an FA node
    CHECK(ggml_flash_attn_ext_get_kv_rows(q) == nullptr);

    ggml_free(ctx);
    printf("OK: FA src[9] accessors (kv_rows / LSE / plain)\n");
    return 0;
}
