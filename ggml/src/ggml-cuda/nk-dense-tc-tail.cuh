#pragma once

#include "common.cuh"

// 4-block tail launcher for head_grid_dp4a when K % 256 == 128 (nb % 8 == 4, nb >= 4).
// Kept in nk-dense-tc-tail.cu so k_nk_head_grid_dp4a (K % 256 == 0) codegen stays unchanged.
void ggml_cuda_nk_head_grid_tail4_launch(
        const char  * vx,
        const int2  * qx_q,
        const float * qx_d,
        float       * dst,
        int64_t       srow,
        int64_t       sd,
        int           nr,
        int           nb,
        int           T,
        bool          l2_ef,
        cudaStream_t  stream);
