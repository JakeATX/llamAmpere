#pragma once

#include "common.cuh"

// Returns the number of following graph nodes consumed, or -1 when the graph uses the ordinary path.
int ggml_cuda_fuse_nemotron_moe(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int index);
