#pragma once

#include "common.cuh"

struct mmq_args;

// #51 (GGML_CUDA_MMQ_SK_LASTCTA=1, default off): dense Q8_0 stream-K MMQ with the K-split fixup done in the matmul
// kernel by the last CTA to finish each split tile, instead of the separate mul_mat_q_stream_k_fixup launch. Same grid,
// same K split points and the same summation order as the fixup kernel, so the result is bit-identical.
// Returns false (nothing launched) when the flag is off or the case is not covered; the caller then runs stock MMQ.
bool ggml_cuda_mmq_sklast(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream);
