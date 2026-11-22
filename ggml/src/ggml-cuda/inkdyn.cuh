#pragma once
#include "common.cuh"

// [INKGLUE DYN] the Inkling dynamic short conv of a full-attention visit (k and v) at decode / verify widths (M <= 8
// tokens of one sequence), in one launch. The graph (src/models/inkling2.cpp convolve() with `dynamic`) is, per unit:
//   C = CONCAT(past [3, ch], TRANSPOSE(x) [M, ch]); CPY(VIEW(C, cols s..s+2), state view) per stored window (the conv
//   state store, one per rollback slot with SLOTCPY)
//   y = SSM_CONV(C, w); for each lag: y = ADD(y, MUL(CONT(TRANSPOSE(VIEW(C, col.. col+M-1))) [hs, nh, M], VIEW(tap) [1, nh, M]))
//   out = SCALE(ADD(x, y), s, b)
// One thread per channel of one unit; the arithmetic is the unfused nodes' (ssm_conv_f32 sum, rounded products and adds
// of the binbcast MUL / ADD, the epilogue v*s + b), so the output and state bytes match.
#define INK_DYN_MAX_UNITS 2
#define INK_DYN_MAX_TERMS 4
#define INK_DYN_MAX_M     8
#define INK_DYN_MAX_STORE 8

struct ink_dyn_unit_args {
    const float * keep;      // past [3, ch], channel stride keep_nb1 (floats)
    int64_t       keep_nb1;
    const float * x;         // new rows: token t of channel c at x[t*x_nb0 + c*x_nb1] (floats)
    int64_t       x_nb0;
    int64_t       x_nb1;
    const float * w;         // conv weight [4, ch], channel stride w_nb1 (floats)
    int64_t       w_nb1;
    int           n_store;
    float *       store[INK_DYN_MAX_STORE];     // state windows, contiguous [3, ch], in graph order
    int           store_col[INK_DYN_MAX_STORE]; // first concat column each window copies (0 .. M)
    int           n_terms;
    int           col[INK_DYN_MAX_TERMS];
    const float * tap[INK_DYN_MAX_TERMS];   // tap of head h, token i at tap[t][h*tap_nb1[t] + i*tap_nb2[t]]
    int64_t       tap_nb1[INK_DYN_MAX_TERMS];
    int64_t       tap_nb2[INK_DYN_MAX_TERMS];
    int           hs;        // channels per head
    float *       out;       // contiguous [ch, M]
    float         scale;
    float         bias;
};

struct ink_dyn_args {
    int               n_units;
    int               ch;
    int               M;
    ink_dyn_unit_args u[INK_DYN_MAX_UNITS];
};

void ggml_cuda_ink_dyn_conv(ggml_backend_cuda_context & ctx, const ink_dyn_args & args);
