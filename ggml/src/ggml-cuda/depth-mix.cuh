#include "common.cuh"

// GGML_OP_DEPTH_MIX on CUDA (fork, Inkling 2 attention over depth; see ggml.h): one block per token.
void ggml_cuda_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op);

// [INKGLUE] depth mix (one query) + RMS_NORM(neps) + MUL(w) into ndst; false: not this shape
// aq / as (optional): also the AI+ H256 activation of the normed rows (rq4::quantize_act layout), n % 256 == 0
// nvfp4_kf = 32 / 64 (needs aq / as): the NVFP4PF activation (k_nvfp4pf_xq layout) instead, n % nvfp4_kf == 0
bool ggml_cuda_depth_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * w, ggml_tensor * ndst, float neps,
        int8_t * aq = nullptr, float * as = nullptr, int nvfp4_kf = 0);
