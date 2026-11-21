#include "common.cuh"

// GGML_OP_DEPTH_MIX on CUDA (fork, Inkling 2 attention over depth; see ggml.h): one block per token.
void ggml_cuda_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op);
