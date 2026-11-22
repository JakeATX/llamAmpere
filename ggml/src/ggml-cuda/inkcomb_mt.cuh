#pragma once

#include "common.cuh"

// [T5-GLUE] G2: Multi-token (T = 1..8) coalesced MoE weighted sum + short-conv ring + residual epilogue
// (replaces k_ink_moe_wsum + ssm_conv_ring_f32 across all 48 Inkling-2-Mini visits during SpecDecode T=5,
// saving 48 launches/round and fixing 32x uncoalesced scatter in k_ink_moe_wsum; 100% bit-identical).
// Kill switch: GGML_CUDA_INK_WSUM_CONV_MT=0 (default 1 = on).

bool ggml_cuda_ink_wsum_conv_mt_enabled();

bool ggml_cuda_ink_wsum_conv_mt(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w,
        ggml_tensor * sum, const ggml_tensor * conv, ggml_tensor * out, float scale, float bias, const ggml_tensor * res);

// Coalesced standalone MoE weighted expert sum (replaces uncoalesced 1-warp-per-element k_ink_moe_wsum; 100% bit-identical).
bool ggml_cuda_ink_moe_wsum_coalesced(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w, ggml_tensor * dst);
