#include "common.cuh"

// GGML_OP_DEPTH_MIX on CUDA (fork, Inkling 2 attention over depth; see ggml.h): one block per token.
void ggml_cuda_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op);

// [INKGLUE] depth mix (one query) + RMS_NORM(neps) + MUL(w) into ndst; false: not this shape
bool ggml_cuda_depth_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * w, ggml_tensor * ndst, float neps);
