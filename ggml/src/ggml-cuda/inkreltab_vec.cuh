#pragma once

#include "common.cuh"

// [T5-GLUE] G3: Vectorized T-reusing relative-position bias table kernel for Inkling-2-Mini (n_rel=16, E=1024, H=24, T=1..8).
// Loads each bias[e, 0..15] row ONCE via a single coalesced 64-byte warp read and broadcasts via __shfl_sync across all H heads
// and all T tokens (120x fewer global reads of bias vs k_ink_rel_table), with zero integer division and bank-conflict-free SMEM.
// 100% bit-identical ascending fmaf chain.
// Kill switch: GGML_CUDA_INK_RELTAB_VEC=0 (default 1 = on).

bool ggml_cuda_ink_rel_table_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, ggml_tensor * dst);
