#pragma once

#include "common.cuh"

// INKGLUE PREP: the per-head glue between the attention projections and the attention of an Inkling 2 decode visit in
// one launch (one warp per head row): the q / k / r clips ([MUL(m) ->] L2_NORM -> SCALE [-> MUL(tau)]), the v RMS_NORM,
// and optionally the k / v SET_ROWS into an F16 / F32 cache. Each job repeats its unfused kernel's arithmetic in the
// same order (k_ink_clip, rms_norm_rows_f32 PLAIN, the tau MUL, set_rows' ggml_cuda_cast), so outputs are bit-identical.
#define INK_PREP_MAX_JOBS 6

struct ink_prep_job {
    const float * x;       // rows [ncols, nrows, nch], strides in floats
    const float * m;       // clip weight [ncols, nrows] or nullptr
    const float * tau;     // per-channel factor (after the clip) or nullptr
    float       * dst;     // contiguous [ncols, nrows, nch]
    float       * dst2;    // the tau product (contiguous, same shape) when tau != nullptr
    char        * cache;   // set_rows target (row r = idx[ch] of a [ncols*nrows]-wide row) or nullptr
    const int64_t * idx;   // set_rows indices [nch]
    int64_t       cache_nb1;
    int           cache_f16;
    int           kind;    // 0 clip, 1 rms
    int           ncols, nrows, nch;
    int64_t       sx1, sx2;
    float         eps, s, b;
    int           row0;    // first global row of this job
};

struct ink_prep_args {
    ink_prep_job job[INK_PREP_MAX_JOBS];
    int          n;
};

void ggml_cuda_ink_prep(ggml_backend_cuda_context & ctx, const ink_prep_args & a, int total_rows);
