#pragma once

#include "common.cuh"

// INKPREFILL: faster k_ink_rel_table (same inputs, bit-identical output). GGML_CUDA_INK_RELTAB2=0 = the inkreltab.cu kernel.
bool ggml_cuda_ink_rel_table2_enabled();
void ggml_cuda_ink_rel_table2(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, ggml_tensor * dst);
