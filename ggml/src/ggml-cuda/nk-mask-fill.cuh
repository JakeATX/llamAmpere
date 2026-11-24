#pragma once

#include "ggml-backend.h"

#include <cstdint>

bool    ggml_backend_cuda_nk_fill_prefix_mask(ggml_backend_t backend, ggml_tensor * dst,
                                              const int32_t * n_visible, int64_t n_rows, int64_t n_kv);
int64_t ggml_cuda_nk_dev_mask_count();
