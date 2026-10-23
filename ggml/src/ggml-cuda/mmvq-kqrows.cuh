#pragma once

#include "common.cuh"

// [KQROWS] Q4_K / Q5_K / Q6_K x q8_1 at 2..8 columns (speculative check widths) as a row kernel: per weight block each
// thread loads every column's activation fragment once, then decodes each row's weights once and applies them to every
// column. The default MMVQ kernel decodes the weights once per (row, column) pair, so it does ncols times the decode
// work, and on Ampere the K-quants leave MMVQ for the MMQ tile kernel at widths 5..8.
//
// Exactness: with GGML_CUDA_KQ_ROWS_NW=0 (the default) each CTA uses the MMVQ table's warp count for that width, so every
// thread visits the same K blocks in the same order, adds the same per-block values (same integer dot products, same
// float expression) and joins the warps in the same order as the MMVQ kernel: the output is bit-identical to MMVQ at
// that width. The rows per CTA only decide which rows share a CTA, never how one row is summed. Against the MMQ tile
// kernel (the Ampere default at widths 5..8) the sums are grouped differently: valid math, not bit-identical.
//
// Default off. GGML_CUDA_KQ_ROWS=1 turns it on (and lets MMVQ keep the K-quants up to KQ_ROWS_MAX_N on Ampere);
// GGML_CUDA_KQ_ROWS_R = 1|2|4|8 rows per CTA (default per width, see mmvq-kqrows.cu); GGML_CUDA_KQ_ROWS_NW = 0 (MMVQ
// table, bit-identical) | 2 | 4 | 8; GGML_CUDA_KQ_ROWS_TYPES = comma list of q4_K,q5_K,q6_K (default all three);
// GGML_CUDA_KQ_ROWS_MIN_N / _MAX_N (default 2 / 8).

// Widest batch the row kernel takes for this type, 0 when it is off for the type (used by ggml_cuda_should_use_mmvq).
int ggml_cuda_kqrows_max_n(ggml_type type);

// Runs the row kernel on q8_1 activations when the switch, type, width and layout allow it; false = caller keeps its path.
// Plain 2D product only: one channel, one sample, no ids, no fusion.
bool ggml_cuda_kqrows_try(ggml_type type, const void * vx, const void * vy_q8_1, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst,
        int cc, cudaStream_t stream);

// per-type launchers (mmvq-kqrows-<type>.cu)
template <ggml_type type>
void ggml_cuda_kqrows_launch(const void * vx, const void * vy_q8_1, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst,
        int rows, int nwarps, cudaStream_t stream);
