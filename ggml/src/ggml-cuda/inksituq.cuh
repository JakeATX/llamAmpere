#pragma once

#include "common.cuh"
#include "inkglue.cuh"

// [INKSITUQ] SITU (inkglue) + the H128 activation quantize of its AI+ MUL_MAT_ID consumer in one launch. The kernel writes
// the SITU f32 output as k_ink_situ does and, from the same registers, quantize_act128's int8 + scales into a per-context
// one-shot buffer keyed to the consumer node; ggml_cuda_mul_mat_id_ai4 takes it instead of running quantize_act128.
// GGML_CUDA_INKSITUQ=0 (or GGML_CUDA_INKGLUE=0) = off.

// true: launched (out written, buffer handed to down); false: nothing launched, run the plain SITU kernel
bool ggml_cuda_ink_situq_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out,
        const ink_situ_params & p, const ggml_tensor * down);

// consumer side, one-shot: the prequantized [rows, K] activation for dst (its src1 unchanged since the producer ran)
bool ggml_cuda_ink_situq_take(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, int64_t rows, int K,
        const int8_t ** q, const float ** s);

// cumulative counters: 0 = produced, 1 = consumed
int ggml_cuda_ink_situq_count(int which);
