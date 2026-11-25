#pragma once

#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

// Opt-in (GGML_CPU_AI4_AVX2=1) SIMD path for the AI+ unit MUL_MAT / MUL_MAT_ID reference ops: activations rotated once
// into the weights' basis, then the host expert-split slab kernels (1MAD types) or the AVX2 pair-table kernel (AIT2P_U).
// Unset keeps the dequantize + f32 dot reference path. Valid-math change (float summation order), not bit-exact.

// true when the flag is set and the op's weight type and shape are covered; work_size is then the op's wsize
bool   ggml_cpu_ai4_avx2_use(const struct ggml_tensor * dst);
size_t ggml_cpu_ai4_avx2_work_size(const struct ggml_tensor * dst, int nth);
// false (nothing written) when the op is not covered or params->wsize is short: the caller runs the reference path
bool   ggml_cpu_ai4_avx2_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
bool   ggml_cpu_ai4_avx2_mul_mat_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
