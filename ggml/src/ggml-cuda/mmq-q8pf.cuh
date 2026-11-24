#pragma once

#include "common.cuh"

struct mmq_args;

// PFG-4 (GGML_CUDA_MMQ_Q8PF=1, default off): dense Q8_0 x q8_1 prompt GEMM for Ampere/Ada (int8 mma m16n8k32).
// Same products and the same fp32 operation order per output as the MMQ Q8_0 vec dot (no K split); only the
// int32 -> fp32 conversion is done with an exact bias trick instead of I2F (quarter rate on sm_86).
bool ggml_cuda_should_use_mmq_q8pf(const mmq_args & args);
void ggml_cuda_mul_mat_q8pf(const mmq_args & args, cudaStream_t stream);
