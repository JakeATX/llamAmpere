#pragma once
#include "common.cuh"

// [INKGLUE MIXQ] A pre-quantized AI+ activation for the MUL_MAT_ID `consumer` that reads x (set by the fused producer, taken
// once by ai4.cu, which then skips its own quantize launch). Owned buffers (alloc) stay allocated until the next alloc or
// release, so the consumer may run several graph nodes after the producer. xsh: 1 = H256 (rq4::quantize_act), 0 = H128.
void ggml_cuda_ink_ai4_preq_set(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t * q,
        const float * s);
// pool buffers for a preq that outlives the producer node: q [rows*K], s [rows*(K >> (7 + xsh))]
void ggml_cuda_ink_ai4_preq_alloc(ggml_backend_cuda_context & ctx, int64_t rows, int K, int xsh, int8_t ** q, float ** s);
void ggml_cuda_ink_ai4_preq_release();
bool ggml_cuda_ink_ai4_preq_take(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t ** q,
        const float ** s);
