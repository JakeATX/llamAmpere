#pragma once
#include "common.cuh"

// [INKGLUE DYN] the Inkling dynamic short conv of a full-attention visit (k and v) at decode, in one launch.
// The graph (src/models/inkling2.cpp convolve() with `dynamic`) is, per unit (k, then v):
//   C = CONCAT(past [3, ch], TRANSPOSE(x) [1, ch]); CPY(VIEW(C, cols 1..3), state view)  (the conv state store)
//   y = SSM_CONV(C, w); for each lag: y = ADD(y, MUL(CONT(TRANSPOSE(VIEW(C, col))) [hs, nh], VIEW(tap) [1, nh]))
//   out = SCALE(ADD(x, y), s, b)
// One thread per channel of one unit; the arithmetic is the unfused nodes' (ssm_conv_f32 sum, rounded products and adds
// of the binbcast MUL / ADD, the epilogue v*s + b), so the output and state bytes match.
#define INK_DYN_MAX_UNITS 2
#define INK_DYN_MAX_TERMS 4

struct ink_dyn_unit_args {
    const float * keep;      // past [3, ch], channel stride keep_nb1 (floats)
    int64_t       keep_nb1;
    const float * x;         // new row, channel stride x_nb1 (floats)
    int64_t       x_nb1;
    const float * w;         // conv weight [4, ch], channel stride w_nb1 (floats)
    int64_t       w_nb1;
    float *       store;     // state window, contiguous [3, ch]
    int           store_col; // first concat column the window copies
    int           n_terms;
    int           col[INK_DYN_MAX_TERMS];
    const float * tap[INK_DYN_MAX_TERMS];   // tap of head h at tap[t][h*tap_nb1[t]]
    int64_t       tap_nb1[INK_DYN_MAX_TERMS];
    int           hs;        // channels per head
    float *       out;       // contiguous [ch]
    float         scale;
    float         bias;
};

struct ink_dyn_args {
    int               n_units;
    int               ch;
    ink_dyn_unit_args u[INK_DYN_MAX_UNITS];
};

void ggml_cuda_ink_dyn_conv(ggml_backend_cuda_context & ctx, const ink_dyn_args & args);
