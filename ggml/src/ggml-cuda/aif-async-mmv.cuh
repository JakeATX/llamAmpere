#pragma once
#include "common.cuh"

// [OPT10] cp.async fractional AI+ matvec for decode / verify widths (GGML_CUDA_AIF_ASYNC=1; unset or 0 = off, and then
// ai4.cu calls nothing in this TU). Types AIF25_U, AIF25_U2, AIF25_U2_512, AIF175_U2 with the same template arguments
// as their magic2::mmv / magic2::mmq call. The kernel is the magic2::mmq unit loop (one 16-row strip per block, NB
// 8-token blocks per tile) with the weight units staged by a 2-stage cp.async ring in shared memory instead of the
// register prefetch, the partial-sum buffer aliased onto the dead tiles, and the exact int32 -> float magic
// conversion. Same bytes, same exact IMMA sums, same float expression and order: y is bit-identical to magic2::mmv.

bool ggml_cuda_aif_async_enabled();

// nb = 8-token blocks per tile (1..4): nb = ceil(N / 8) with N <= 32 is the verify-width (narrow) tile, one decode per
// weight unit for all N tokens; nb = 1 is the mmv tiling (one tile per 8 tokens) for any N. Returns false, without
// launching, for any other type / xsh / nb, or for nb > 1 with N > 8 * nb (nb > 1 is a single tile; nb = 1 splits N into
// launches of 8 * 65535 tokens, so grid.y stays <= 65535); the caller then falls back.
bool ggml_cuda_aif_async_mm(ggml_type type, int xsh, int nb, const uint8_t * w, const int8_t * q, const float * scales,
                            float * y, int M, int K, int64_t N, cudaStream_t stream);
