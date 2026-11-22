#include "inkdyn.cuh"

// one thread per (channel, unit); every thread reads its channel's operands before it writes anything
static __global__ void k_ink_dyn_conv(const ink_dyn_args args) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= args.ch || (int) blockIdx.y >= args.n_units) {
        return;
    }
    const ink_dyn_unit_args & u = args.u[blockIdx.y];

    float x[4];
    float w[4];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        x[j] = u.keep[c*u.keep_nb1 + j];
    }
    x[3] = u.x[c*u.x_nb1];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = u.w[c*u.w_nb1 + j];
    }
    const int h = c / u.hs;
    float tp[INK_DYN_MAX_TERMS];
#pragma unroll
    for (int t = 0; t < INK_DYN_MAX_TERMS; ++t) {
        tp[t] = t < u.n_terms ? u.tap[t][h*u.tap_nb1[t]] : 0.0f;
    }

    // the conv state store (CPY of concat columns store_col .. store_col + 2)
    float * st = u.store + (int64_t) c*3;
#pragma unroll
    for (int q = 0; q <= 1; ++q) {
        if (q == u.store_col) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                st[j] = x[q + j];
            }
        }
    }

    // ssm_conv_f32: sumf = 0, += x[j]*w[j], += bias (none: 0)
    float v = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v += x[j] * w[j];
    }
    v += 0.0f;
    // the dynamic taps: y = y + window*tap, each product and sum rounded on its own (separate MUL and ADD nodes)
#pragma unroll
    for (int t = 0; t < INK_DYN_MAX_TERMS; ++t) {
        if (t < u.n_terms) {
            float xc = 0.0f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (q == u.col[t]) {
                    xc = x[q];
                }
            }
            v = __fadd_rn(v, __fmul_rn(xc, tp[t]));
        }
    }
    // the epilogue: ADD(x, y), then SCALE (v*s + b, as scale_f32 / elem_chain)
    v = x[3] + v;
    v = u.scale * v + u.bias;
    u.out[c] = v;
}

void ggml_cuda_ink_dyn_conv(ggml_backend_cuda_context & ctx, const ink_dyn_args & args) {
    const int bs = 128;
    const dim3 grid((unsigned) ((args.ch + bs - 1)/bs), (unsigned) args.n_units);
    k_ink_dyn_conv<<<grid, bs, 0, ctx.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}
