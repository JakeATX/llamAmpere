#include "common.cuh"

// GGML_OP_DEPTH_MIX on CUDA (fork, Inkling 2 attention over depth; see ggml.h): one block per token.
void ggml_cuda_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op);

// [INKGLUE] depth mix (one query) + RMS_NORM(neps) + MUL(w) into ndst; false: not this shape
// aq / as (optional): also the AI+ H256 activation of the normed rows (rq4::quantize_act layout), n % 256 == 0
// prefill_mode (0 = legacy AI4 MIXQ, 1 = prefill AI4 H256 from SMEM, 232/264 = NVFP4PF KF=32/64 from SMEM; write_f32=false elides ndst store when SMEM active)
bool ggml_cuda_depth_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * w, ggml_tensor * ndst, float neps,
        int8_t * aq = nullptr, float * as = nullptr, int prefill_mode = 0, bool write_f32 = true);
