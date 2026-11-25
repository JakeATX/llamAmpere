#pragma once

#include "common.cuh"

// Single-pass compact SJ-KVaRN 4/4 attention for verify widths up to 16 (GGML_SJKVARN_V16=1).
bool ggml_cuda_flash_attn_ext_sj_kvarn_v16_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_sj_kvarn_v16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
