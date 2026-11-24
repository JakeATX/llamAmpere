#pragma once

#include "common.cuh"

#include <cstdlib>

// [FUSE RELMERGE] (default on; GGML_CUDA_FUSE_RELMERGE=0 = off): the Inkling far / near attention merge of a
// full-attention visit at decode widths in one launch. The graph (llama-graph.cpp inkrel_merge, INKKV far/near split)
// runs it as unpack CONCAT, ADD (far LSE + table constant), CONCAT, FILL, CONCAT, SOFT_MAX over (far, near, 0), 2x
// CONT, 2x MUL, ADD and the output gate (SIGMOID, MUL): 12 kernels. One block per (q head, token), one thread per dv,
// the same expressions in the same order (soft_max_f32's max / exp / sum / reciprocal, then the products and the sum).
static inline bool ggml_cuda_fuse_relmerge_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FUSE_RELMERGE");
        return !e || !*e || atoi(e) != 0;
    }();
    return on;
}

struct ggml_cuda_relmerge_args {
    // far output o_a [dv, H, T]: either a contiguous tensor (a1 == nullptr), or the unpack CONCAT along dim 1 of two
    // 4D views a0 / a1 [dv, g, P, T] (merged head h = i1 + 2g*i2, i1 < g from a0, else a1 at i1 - g)
    const char * a0; const char * a1;
    int64_t a0_nb1, a0_nb2, a0_nb3;
    int64_t a1_nb1, a1_nb2, a1_nb3;
    int g;
    // far LSE [H, T] and the per-(head, token) table constant added to it (element strides in floats)
    const float * lf; int64_t lf_t;
    const float * cf; int64_t cf_h, cf_t;
    // near output o_b [dv, H, T] (contiguous rows of dv) and its LSE [H, T]
    const float * ob; int64_t ob_t;
    const float * ln; int64_t ln_t;
    // output gate (nullptr = none) [dv*H, T], sigmoid(gate) * merged
    const float * gate; int64_t gate_t;
    float * dst; int64_t dst_t;
    int dv, H, T;
};

void ggml_cuda_relmerge_launch(ggml_backend_cuda_context & ctx, const ggml_cuda_relmerge_args & args);
