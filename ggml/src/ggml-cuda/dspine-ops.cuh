#pragma once

#include "common.cuh"

extern int64_t ggml_cuda_dspine_fuse_count;

// Fused DSpine draft/lattice kernels (GGML_CUDA_DSPINE_FUSE=1, default 0).
// ctx == nullptr: plan-only match for graph_optimize (returns extra nodes spanned, 0 if no match).
// ctx != nullptr: verifies memory ranges and launches the fused kernel on ctx->stream().
int ggml_cuda_try_fuse_dspine(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i);
