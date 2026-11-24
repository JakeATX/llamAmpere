#pragma once
// NEONKERN: sm_86 int8 tensor-core NVFP4 MoE decode/verify kernels (T = 1..8, K = 2304, F = 1152, n_used = 8).
// Default OFF; enable with GGML_CUDA_NK_NVFP4_TC=1.
// Optional L2 evict-first weight policy toggle: GGML_CUDA_NK_NVFP4_TC_L2=0 (default 1 when NK_NVFP4_TC=1).

#include "nvfp4moe.cuh"

bool ggml_cuda_nk_nvfp4_tc_enabled();
bool ggml_cuda_nk_nvfp4_tc_accepts(const nvfp4moe_params & P);
bool ggml_cuda_nk_nvfp4_tc_launch(const nvfp4moe_params & P, cudaStream_t stream);
