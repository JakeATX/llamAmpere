#pragma once
#include "common.cuh"

// [PP27] Exact AIF25_U prompt kernel (magic2::mmq, bit-identical to magic2::mmv). xsh: 1 = one activation scale per 256
// columns, 0 = per 128. tile = token tile (32, 64 or 128), strips = 16-row strips per block (1 or 2; tile 128 takes 1).
// Returns false, without launching, when (tile, strips) is not a built variant (warns once) or N needs more than 65535
// token tiles; the caller then runs mmv.
bool ggml_cuda_aif25_mmq(int xsh, int tile, int strips, const uint8_t * w, const int8_t * q, const float * scales, float * y,
                         int M, int K, int64_t N, cudaStream_t stream);
