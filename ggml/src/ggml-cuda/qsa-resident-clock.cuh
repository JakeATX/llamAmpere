#pragma once

#include "common.cuh"

// Four-token pages keep the scalar codec's record metadata resident without changing its bytes.
constexpr int GGML_QSA_PAGE_CELLS = 4;
constexpr int GGML_QSA_PAGE_HEAD_BYTES = 1048;
constexpr int GGML_QSA_META_HEAD_BYTES = 1536;

struct ggml_cuda_qsa_clock_map {
    const char * host;
    char * card;
    char * meta;
    int32_t * table;
    int32_t * tags;
    uint32_t * stamp;
    int32_t * ref;
    int32_t * miss_page;
    int32_t * miss_slot;
    uint32_t * ctl; // epoch, CLOCK hand, current misses, active, overflow; u64 totals at 8,10,12
    uint32_t slots;
    uint32_t records;
    uint32_t group;
    uint32_t heads;
    size_t stride;
};

bool ggml_cuda_qsa_clock_is_buft(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_cuda_qsa_clock_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots);
size_t ggml_backend_cuda_qsa_clock_device_bytes(ggml_backend_buffer_type_t buft, const ggml_tensor * body);
const char * ggml_cuda_qsa_clock_body(const ggml_tensor * body);
void ggml_cuda_qsa_clock_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_qsa_clock_refresh(ggml_backend_cuda_context & ctx, const ggml_tensor * body);
