#pragma once

#include "common.cuh"

#include <cstdlib>

// [FUSE KVCONV] (default on; GGML_CUDA_FUSE_KVCONV=0 = off): the K and V short convs of an Inkling sliding-window visit
// (two planned conv rings with the SCALE(ADD(x, conv)) epilogue, back to back in the graph) in one launch. The blocks
// of the first unit run the first ring's arithmetic, the rest the second's: the expressions, order and window copies
// of ssm_conv_ring_f32<false, 4, n_t, epi> for each, so both outputs and all windows are the same bytes. The caller
// checks that neither unit reads or writes what the other writes.
static inline bool ggml_cuda_fuse_kvconv_on() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_FUSE_KVCONV");
        return !e || !*e || atoi(e) != 0;
    }();
    return on;
}

// conv1 / conv2: planned SSM_CONV nodes (ctx.conv_rings), x: the ADD operand (the ring's new rows), out: the SCALE
// node. false (launches nothing): a ring is not planned or a shape is outside the ring kernel's
bool ggml_cuda_op_kvconv_pair(ggml_backend_cuda_context & ctx,
        ggml_tensor * conv1, const ggml_tensor * x1, ggml_tensor * out1, float scale1, float bias1,
        ggml_tensor * conv2, const ggml_tensor * x2, ggml_tensor * out2, float scale2, float bias2);
