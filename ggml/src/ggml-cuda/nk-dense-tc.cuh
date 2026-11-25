#pragma once

#include "common.cuh"

// GGML_CUDA_NK_DENSE_TC (default ON, =0 is the kill switch): Ampere int8 kernel for dense Q8_0 decode/verify
// projections (0 = exact base q8a16/mmvq).
// GGML_CUDA_NK_DENSE_TC_MIN_ROWS: minimum weight rows (default 16385 = head-only;
// set 0 to include attention-sized and grouped projections).
// GGML_CUDA_NK_DENSE_TC_MINT: minimum batch width T (default 4 so T=1..3 stay on base q8a16).
// GGML_CUDA_NK_DENSE_TC_HEAD_PATH: auto | tc | grid (default auto; selects per T at T>=4).
enum nk_dense_tc_head_path {
    NK_HEAD_PATH_AUTO = 0,
    NK_HEAD_PATH_TC   = 1,
    NK_HEAD_PATH_GRID = 2,
};

bool    ggml_cuda_nk_dense_tc_enabled();
int     ggml_cuda_nk_dense_tc_min_rows();
bool    ggml_cuda_nk_dense_tc_takes_rows(int64_t nrows);
int64_t ggml_cuda_nk_dense_tc_count();
int64_t ggml_cuda_nk_dense_tc_head_grid_count();
int64_t ggml_cuda_nk_dense_tc_head_tc_count();
int64_t ggml_cuda_nk_dense_tc_dense_count();

bool ggml_cuda_nk_dense_tc_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor       * dst,
    bool                force_enable  = false,
    int                 path_override = -1,
    bool                respect_mint  = true);

bool ggml_cuda_nk_dense_tc_group(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * const       * dsts,
    int                         n,
    bool                        check_only,
    bool                        force_enable = false);
