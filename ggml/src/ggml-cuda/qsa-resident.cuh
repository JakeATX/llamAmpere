#pragma once

#include "common.cuh"

struct ggml_cuda_qsa_resident_map {
    const char * host;
    char * card;
    int32_t * tags;
    uint32_t slots;
    size_t stride;
};

bool ggml_cuda_qsa_resident_is_buft(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_cuda_qsa_resident_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots);
size_t ggml_backend_cuda_qsa_resident_device_bytes(ggml_backend_buffer_type_t buft, const ggml_tensor * body);
const char * ggml_cuda_qsa_resident_body(const ggml_tensor * body);
void ggml_cuda_qsa_resident_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_qsa_resident_invalidate(ggml_backend_cuda_context & ctx, const ggml_tensor * body);
