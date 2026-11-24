#pragma once
// INKNVFP4K NVFP4PF: NVFP4 MoE prefill (MUL_MAT_ID with >= 512 tokens) on int8 tensor cores, ported from Gemini's S8-K64
// engine (exp/ampere-nextgen 82183ff0b); kernels and numerics in nvfp4pf-kern.cuh. Runs from the fused MUL_MAT_ID +
// per-expert scale site (the scale is applied in the epilogue). Not bit-exact with MMQ: KLD-gated.
// GGML_CUDA_NVFP4PF = k64 (default: int8 per 64) | 0 (MMQ) | k32 (int8 per 32, lower loss).
// GGML_CUDA_NVFP4PF_LOG=1 logs every call (shape, mode); the first call always logs once.

#include "common.cuh"

// 0 off, 32 or 64: the int8 block width
int ggml_cuda_nvfp4pf_mode();

// dst = MUL_MAT_ID(src0 NVFP4 [K, M, E], src1 f32 [K, ne11, T], ids [n_used, T]) [* s_exp[e]] when the shape fits
// (T >= 512, sm_80 / sm_86, K % 128 == 0, M % 16 == 0, contiguous src1 / dst); false = nothing done
bool ggml_cuda_nvfp4pf_try(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * s_exp);

// nk_override: -1 = env GGML_CUDA_NK_PF_MM (default OFF), 0 = force base, 1 = force NK_PF_MM
bool ggml_cuda_nvfp4pf_try_mode(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * s_exp, int nk_override);
