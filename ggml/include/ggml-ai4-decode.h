#pragma once

// AI+ (ggml/src/ggml-ai4.h) host-side slab decoder for code outside ggml that needs plain rows of an AI+ tensor, such
// as the compact draft head (src/llama-draft-vocab-compact.h). AI+ types have no to_float: their bytes are 16-row slabs
// (AI4: 1024 slab, AIk_U: unit-128, AIk_S: span-1024), so the decoder works one slab at a time. Kept out of ggml.h so
// adding it does not rebuild every translation unit.

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

    // Bytes of one 16-row slab of an AI+ tensor with row length K (ggml_row_size(type, K) * 16), 0 for a non-AI+ type.
    GGML_API size_t ggml_ai4_slab_nbytes(enum ggml_type type, int64_t K);

    // Decodes slab g (rows 16g .. 16g + 15) of the AI+ tensor `data` (row length K) into y[r*K + k] as f32 in the
    // ORIGINAL basis (the offline rotation undone), the weights the CUDA reference path and the CPU backend multiply.
    // Returns false (y untouched) for a non-AI+ type or a K the type's layout does not allow.
    GGML_API bool ggml_ai4_dequantize_slab(enum ggml_type type, const void * data, int64_t K, int64_t g, float * y);

#ifdef __cplusplus
}
#endif
