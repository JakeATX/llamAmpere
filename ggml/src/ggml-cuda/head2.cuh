#pragma once
// [HEAD2] exact two-stage output head (LLAMA_HEAD2, default off; see head2.cu and head2-ref.h).
#include "common.cuh"

// true once a head is registered (never with the flag unset)
extern bool g_ggml_cuda_head2_on;
static inline bool ggml_cuda_head2_active() {
    return g_ggml_cuda_head2_on;
}

// MUL_MAT on a registered head: computes dst and returns true. Any other op that reads a registered head aborts
// (its bytes are relaid out). Returns false for every other node.
bool ggml_cuda_head2_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// true when node reads a registered head (the graph evaluator must not fuse it)
bool ggml_cuda_head2_touches(const ggml_tensor * node);

// relayout t (Q8_0, on this device or in a host buffer) into planes and register it.
//   mode 1: planes in place (t on the device)
//   mode 2: hi planes on the device, lo planes in pinned host memory; a host-resident t moves its hi planes into a
//           new device buffer (frees 8/34 of the head's VRAM against mode 1)
//   top_k > 0: rows that can reach the top_k of each column are exact, others -inf
//   top_k = 0: rows whose softmax mass after softcap can exceed V * e^-delta are exact, others their upper bound
bool ggml_cuda_head2_prepare(ggml_tensor * t, int mode, int device, int top_k, float delta, float softcap);
