#pragma once

#include "common.cuh"
#include "topk-moe.cuh"

#include <cstdlib>

// [FUSE ROUTERT] (GGML_CUDA_FUSE_ROUTERT=1, default off; needs INKROUTER on): the INKROUTER router (BF16 gate matvec +
// the topk-moe chain in one launch) at 2..8 rows (drafter verify widths). Each row's logits are the sums
// mul_mat_vec_f computes for one column (block of 256, the same per-thread accumulation and reductions for every
// column), i.e. the bits a one-row decode step gets for that row; the unfused verify path runs mul_mat_f there, so
// the logits differ from it by rounding. The last block runs topk_moe_cuda's per-row device code, one warp per row.
#define GGML_CUDA_ROUTERT_MAX_ROWS 8

static inline bool ggml_cuda_fuse_routert_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FUSE_ROUTERT");
        return e && *e && atoi(e) != 0;
    }();
    return on;
}

// logits = the MUL_MAT node [n_expert, n_rows] (written), weights [1, n_expert_used, n_rows], ids [n_expert_used, n_rows]
// with row stride n_expert, clamp / scale / bias as for ggml_cuda_op_topk_moe. Returns false (launches nothing) when no
// completion-counter slot is free or the shape is not instantiated; the caller then runs the unfused nodes.
bool ggml_cuda_op_router_t(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             w,
                           const ggml_tensor *             y,
                           ggml_tensor *                   logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);
