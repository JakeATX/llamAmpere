#pragma once

#include "common.cuh"

// Verify-tuned SJ-KVaRN 4/4 kernels (default OFF unless opt-in env switches are set):
//   - GGML_SJKVARN_STREAM_PV16=1   -> spill-free HMMA.16816.F16 (142T) stream path (n_q <= 8)
//   - GGML_SJKVARN_DIRECT_VERIFY=1 -> verify-tuned IMMA.16832.U8.S8 + U8.U8 path (n_q >= 4..64, or >=1 with =2 / MIN_T=1)

bool ggml_cuda_flash_attn_ext_sj_kvarn_stream_pv16_applies(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_sj_kvarn_stream_pv16_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_sj_kvarn_direct_verify_applies(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_sj_kvarn_direct_verify_run(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
