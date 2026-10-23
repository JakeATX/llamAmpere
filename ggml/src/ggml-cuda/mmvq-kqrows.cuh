#pragma once

#include "common.cuh"

// [KQROWS] Q4_K / Q5_K / Q6_K x q8_1 at 2..8 columns (speculative check widths) as a row kernel: per weight block each
// thread loads every column's activation fragment once, then decodes each row's weights once and applies them to every
// column. (In the MMVQ kernel nvcc already merges the repeated weight decode across columns at these widths: same LDG
// and IDP counts in SASS. The gains are elsewhere: on Ampere the K-quants leave MMVQ for the MMQ tile kernel at widths
// 5..8, and small matrices want fewer rows per CTA than MMVQ's 8 to fill the card.) All loads of a K step are issued
// before the first dot product (no per-row branch between them), as nvcc schedules MMVQ.
//
// Exactness: with GGML_CUDA_KQ_ROWS_NW=0 (the default) each CTA uses the MMVQ table's warp count for that width, so every
// thread visits the same K blocks in the same order, adds the same per-block values (same integer dot products, same
// float expression) and joins the warps in the same order as the MMVQ kernel: the output is bit-identical to MMVQ at
// that width. The rows per CTA only decide which rows share a CTA, never how one row is summed. Against the MMQ tile
// kernel (the Ampere default at widths 5..8) the sums are grouped differently: valid math, not bit-identical.
//
// Default off. GGML_CUDA_KQ_ROWS=1 turns it on for the (type, width, row count) cells of the default table in
// mmvq-kqrows.cu, the cells where it measured faster than the current route (MMVQ at 2..4, the MMQ tile at 5..8);
// the router (ggml_cuda_mul_mat_route) then sends those products to MMVQ, which hands them to the row kernel.
// GGML_CUDA_KQ_ROWS_R = 1|2|4|8 forces that many rows per CTA for EVERY cell (sweeps); GGML_CUDA_KQ_ROWS_NW = 0 (table)
// | 2 | 4 | 8; GGML_CUDA_KQ_ROWS_TYPES = comma list of q4_K,q5_K,q6_K (default all three); GGML_CUDA_KQ_ROWS_MIN_N /
// _MAX_N (default 2 / 8).

// True when the row kernel takes this product (switch on, type and width enabled, a table cell or a forced row count).
bool ggml_cuda_kqrows_wants(ggml_type type, int cc, int64_t ncols_x, int64_t nrows_x, int64_t ncols_dst);

// Runs the row kernel on q8_1 activations when ggml_cuda_kqrows_wants() holds; false = caller keeps its path.
// Plain 2D product only: one channel, one sample, no ids, no fusion.
bool ggml_cuda_kqrows_try(ggml_type type, const void * vx, const void * vy_q8_1, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst,
        int cc, cudaStream_t stream);

// per-type launchers (mmvq-kqrows-<type>.cu)
template <ggml_type type>
void ggml_cuda_kqrows_launch(const void * vx, const void * vy_q8_1, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst,
        int rows, int nwarps, cudaStream_t stream);
