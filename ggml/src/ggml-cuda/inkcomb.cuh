#pragma once
#include "common.cuh"

// [INKGLUE WCONV] the Inkling MoE combine in one launch: the weighted expert sum (MUL -> CONT(PERMUTE) -> SUM_ROWS), the
// planned short-conv ring of the FFN post conv with its windows, and the epilogue SCALE(ADD(sum, conv)) + residual.
// e: [n_embd, n_used, 1] expert outputs, w: [1, n_used, 1] weights, sum: the SUM_ROWS output (the ring's new row, also
// written), out: the residual ADD output. false: the ring of `conv` is not planned or the shapes are not supported
bool ggml_cuda_ink_wsum_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w,
        ggml_tensor * sum, const ggml_tensor * conv, ggml_tensor * out, float scale, float bias, const ggml_tensor * res);
