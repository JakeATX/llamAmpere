#pragma once

#include "common.cuh"

struct ggml_backend_graph_optimize_params;

bool ggml_cuda_nemotron_ssm_enabled();
void ggml_cuda_nemotron_ssm_prepare(ggml_cgraph * graph, ggml_backend_graph_optimize_params * params);
int ggml_cuda_fuse_nemotron_ssm(ggml_backend_cuda_context & ctx, ggml_cgraph * graph, int node_idx);
