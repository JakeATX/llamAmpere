#pragma once

#include "common.cuh"

// GGML_OP_LOOP_GATE at prompt widths (T > 8 rows): the same F32 math as loop-gate.cu as two tiled kernels, so a prompt
// ubatch runs the whole gate in 2 launches instead of the unfused BF16 graph (rms_norm x2, concat, 3 GEMMs with
// their converts, 2 tanh and 4 broadcast ops). Rank R <= 64 (Inkling 2: R = 64).
bool ggml_cuda_ink_loop_gate_pp_supported(const ggml_tensor * op);
void ggml_cuda_ink_loop_gate_pp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
