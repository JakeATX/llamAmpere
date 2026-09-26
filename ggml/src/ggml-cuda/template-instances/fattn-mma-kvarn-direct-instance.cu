// This file has been hand-created (KVarN fragment-direct decode kernel). Do NOT run generate_cu_files.py
// over it — that script deletes all *.cu including the turbo VEC instances.

#include "../fattn-kvarn-direct.cuh"

bool ggml_cuda_flash_attn_ext_kvarn_direct_supported(const ggml_tensor * dst) {
    return ggml_cuda_flash_attn_ext_kvarn_direct_supported_impl(dst);
}

void ggml_cuda_flash_attn_ext_kvarn_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_kvarn_direct_impl(ctx, dst);
}
