#pragma once

#include "common.cuh"
#include "ggml.h"

#include <cstdint>

// NEONKERN: opt-in fast DSpark folded Markov head + exact 8-bit radix top-K / Gumbel-max kernels
// (GGML_CUDA_NK_DSPARK_HEAD=1, default OFF). Bit-identical numerics to base dspark-head.cu.

bool ggml_cuda_nk_dspark_head_enabled();
int64_t ggml_cuda_nk_dspark_head_count();

bool ggml_cuda_nk_dspark_head_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * fold_w,
        const ggml_tensor * latent,
        const ggml_tensor * fold_bias,
        const ggml_tensor * base,
        const ggml_tensor * ids,
        const ggml_tensor * noise,
        ggml_tensor       * cand_out,
        ggml_tensor       * vals_out,
        ggml_tensor       * prev_out,
        int64_t             n_eval,
        int                 sk,
        float               softcap,
        float               inv_temp);
