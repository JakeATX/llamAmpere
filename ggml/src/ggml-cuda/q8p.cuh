#pragma once
// [INKDENSE PORTMAP row 1] planar Q8_0 dense weights for decode (GGML_CUDA_Q8P=1, default off).
//
// At graph compute, a Q8_0 weight that this graph only uses as src0 of MUL_MAT is repacked in place (same bytes) from
// 34-byte blocks into two planes: int8 values [M][K], then FP16 block scales [M][K/32]. A warp reads 128 values with one
// coalesced 4-byte load per lane, and the activations stay F32 (no q8_1 quantize launch). Decode (N <= 8) runs
// q8p_mmv; one launch may cover several weights (each with its own activations), e.g. an Inkling visit's
// q / k / v / rel / sink / gate. Larger batches unpack the weight to Q8_0 blocks in the pool and take the usual path,
// so prefill results do not change. A weight is unpacked back (and never repacked) as soon as any graph uses it in
// another way. Results are rounding-level different from the q8_1 path (F32 activations instead of int8).

#include "common.cuh"

#define GGML_CUDA_Q8P_FLAG      (1 << 30)   // tensor->flags: the data is planar
#define GGML_CUDA_Q8P_FLAG_DENY (1 << 29)   // tensor->flags: used in another way, keep the Q8_0 blocks
#define GGML_CUDA_Q8P_GRP_MAX   8

bool ggml_cuda_q8p_enabled();

static inline bool ggml_cuda_q8p_is(const ggml_tensor * t) {
    return t != nullptr && (t->flags & GGML_CUDA_Q8P_FLAG) != 0;
}

// the tensor or the base it views is planar
static inline bool ggml_cuda_q8p_any(const ggml_tensor * t) {
    return t != nullptr && (ggml_cuda_q8p_is(t) || ggml_cuda_q8p_is(t->view_src));
}

// bumped by every in-place pack / unpack: a captured CUDA graph that read the weight in the other layout must re-capture
uint64_t ggml_cuda_q8p_generation();

// shape test of a weight (no buffer test)
bool ggml_cuda_q8p_shape_ok(const ggml_tensor * w);

// Q8_0 blocks <-> planes, in place (synchronous on stream; row chunks, so the scratch is the scale plane plus one chunk)
void ggml_cuda_q8p_pack_inplace(ggml_tensor * w, cudaStream_t stream);
void ggml_cuda_q8p_unpack_inplace(ggml_tensor * w, cudaStream_t stream);
// planes of w -> Q8_0 blocks at dst (async)
void ggml_cuda_q8p_unpack_to(const ggml_tensor * w, void * dst, cudaStream_t stream);

// one launch for n MUL_MATs with planar src0, F32 src1 of at most 8 columns (the same count for all), F32 dst
bool ggml_cuda_q8p_mmv(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n);
