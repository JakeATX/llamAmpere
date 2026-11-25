#pragma once

#include "common.cuh"

// Spill-free compact SJ-KVaRN 4/4 verify attention for widths 7..8 at GQA 6 (GGML_SJKVARN_W8: unset/0 = off, 2 = on, 1 = raw-K;
// common/arg.cpp sets 2 for draft-dflash runs only).
bool ggml_cuda_flash_attn_ext_sj_kvarn_w8_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_sj_kvarn_w8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
