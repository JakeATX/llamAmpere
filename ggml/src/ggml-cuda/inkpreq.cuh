#pragma once
#include "common.cuh"

// [INKGLUE MIXQ] A pre-quantized AI+ activation for the MUL_MAT_ID `consumer` that reads x (set by the fused producer, taken
// once by ai4.cu, which then skips its own quantize launch). The buffers (alloc) are one fixed device buffer per backend
// context, freed with the context (ggml_cuda_ink_ai4_preq_free), so the consumer may run several graph nodes after the
// producer and a captured graph may replay them. xsh: 1 = H256 (rq4::quantize_act), 0 = H128.
void ggml_cuda_ink_ai4_preq_set(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t * q,
        const float * s);
// buffers for a preq that outlives the producer node: q [rows*K], s [rows*(K >> (7 + xsh))]; false = too big
bool ggml_cuda_ink_ai4_preq_alloc(ggml_backend_cuda_context & ctx, int64_t rows, int K, int xsh, int8_t ** q, float ** s);
void ggml_cuda_ink_ai4_preq_release();
void ggml_cuda_ink_ai4_preq_free(const ggml_backend_cuda_context * ctx);
bool ggml_cuda_ink_ai4_preq_take(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t ** q,
        const float ** s);
