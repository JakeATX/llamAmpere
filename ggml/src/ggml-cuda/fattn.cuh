#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);

// [#22/#29] true when bounded prefill is on (GGML_CUDA_PREFILL_KV_MIB unset = 256 MiB, 0 or off = off) and this op runs
// the bounded f16 prefill plan (the same decision the allocator and the executor use); CUDA graphs are not captured
// for graphs holding such an op
bool ggml_cuda_flash_attn_ext_bounded_prefill_applies(int device, const ggml_tensor * dst);
