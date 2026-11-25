#pragma once

#include "common.cuh"

#include <cstdlib>

// [FUSE KVSTORE] (GGML_CUDA_FUSE_KVSTORE=1, default off): the SJ-KVaRN cache writes of an Inkling full-attention visit,
// TURBO_WHT(Q, group 256) and the K and V SET_ROWS into the TQ6_0 cache with the 256-point write rotation, in one
// launch instead of three. Each block runs one slice of one of them with the arithmetic of the unfused kernel
// (k_turbo_wht_f32_plain256, one group per warp; k_set_rows_tq6_sj_kvarn_rot<256>, one slice per block), so every
// output is the bytes of the unfused run.
static inline bool ggml_cuda_fuse_kvstore_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FUSE_KVSTORE");
        return e && *e && atoi(e) != 0;
    }();
    return on;
}

// can the fused launch take this TURBO_WHT (group 256, no scale) / this SET_ROWS (SJ-KVaRN 256-point rotation)
bool ggml_cuda_kvstore_wht_ok(const ggml_tensor * wht);
bool ggml_cuda_kvstore_store_ok(const ggml_tensor * st);

// wht, sk and sv passed the checks above; sk and sv use the same index type. false (launches nothing): too many blocks
bool ggml_cuda_kvstore_launch(ggml_backend_cuda_context & ctx, ggml_tensor * wht, ggml_tensor * sk, ggml_tensor * sv);
