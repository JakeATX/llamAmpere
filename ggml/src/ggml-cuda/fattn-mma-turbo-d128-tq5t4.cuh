#pragma once

#include "common.cuh"

// Opt-in fused D=128 TQ5_0 K / TURBO4_0 V MMA flash attention (GGML_CUDA_FATTN_D128_TQ5T4=1).
// Eliminates full-cache F16 K/V dequant scratch and passes on D=128 drafter caches (e.g. DSpine).
// When ctx is nullptr, probes support only (used by alloc-size and packed-mask queries).
bool ggml_cuda_flash_attn_ext_d128_tq5t4_try(ggml_backend_cuda_context * ctx, ggml_tensor * dst, int device);
