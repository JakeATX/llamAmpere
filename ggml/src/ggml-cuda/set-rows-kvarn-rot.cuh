#pragma once

#include "common.cuh"

// KVarN fused write rotation (llamAmpere #139, GGML_KVARN_FUSED_ROT): SET_ROWS into a TQ6_0 cache that
// applies the KVarN Hadamard to the source rows itself (see ggml_set_rows_kvarn_rot in ggml.h).
// Upstream builds drop this file together with the commits that add it.

static inline bool ggml_cuda_set_rows_is_kvarn_rot(const ggml_tensor * dst) {
    const int32_t mode = ggml_get_op_params_i32(dst, 1);
    return mode == GGML_SET_ROWS_KVARN_ROT256 || mode == GGML_SET_ROWS_KVARN_ROT128;
}

void ggml_cuda_set_rows_kvarn_rot(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
