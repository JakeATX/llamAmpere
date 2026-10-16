#pragma once

#include "ggml-cpu.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

bool ggml_cpu_exl3_enabled(void);
GGML_BACKEND_API bool ggml_cpu_exl3_supported(const struct ggml_tensor * dst);
GGML_BACKEND_API size_t ggml_cpu_exl3_work_size(const struct ggml_tensor * dst, int nth);
GGML_BACKEND_API bool ggml_cpu_exl3_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_cpu_exl3_mul_mat_ref(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// Internal test entry point: row-major states and rounded weights for one native tile.
GGML_BACKEND_API bool ggml_cpu_exl3_decode_tile(const void * data, int bits, uint16_t * states, float * weights);

#ifdef __cplusplus
}
#endif
