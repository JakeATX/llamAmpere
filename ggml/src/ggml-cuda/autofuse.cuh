#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

struct ggml_backend_cuda_context;

#define GGML_CUDA_AUTOFUSE_MAX_OPS 8

struct ggml_cuda_autofuse_plan {
    uint64_t uid = 0;
    uint64_t topology_hash = 0;
    std::vector<uint64_t> topology;
    std::vector<uint8_t> lengths;
    int64_t histogram[GGML_CUDA_AUTOFUSE_MAX_OPS + 1] = {};
    bool reported = false;
};

bool ggml_cuda_autofuse_enabled();
ggml_cuda_autofuse_plan * ggml_cuda_autofuse_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph);
int ggml_cuda_autofuse_launch(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph, int i, ggml_cuda_autofuse_plan & plan);
void ggml_cuda_autofuse_report(ggml_backend_cuda_context & ctx, ggml_cuda_autofuse_plan & plan);
