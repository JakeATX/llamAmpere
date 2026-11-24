#pragma once
// NEONKERN: sm_86 int8 tensor-core Q8_0 MoE decode/verify kernels (T = 1..8, K = 2304, F = 1152, n_used = 8).
// Default OFF; enable with GGML_CUDA_NK_Q8_0_TC=1.
// Optional L2 evict-first weight policy toggle: GGML_CUDA_NK_Q8_0_TC_L2=0 (default 1 when NK_Q8_0_TC=1).

#include "nvfp4moe.cuh"

bool    ggml_cuda_nk_q8_0_tc_enabled();
bool    ggml_cuda_nk_q8_0_tc_supported(int cc, int64_t K, int64_t F, int64_t n_used);
bool    ggml_cuda_nk_q8_0_tc_accepts(const nvfp4moe_params & P);
bool    ggml_cuda_nk_q8_0_tc_launch(const nvfp4moe_params & P, cudaStream_t stream);
int64_t ggml_cuda_nk_q8_0_tc_count();
