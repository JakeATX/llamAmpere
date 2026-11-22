#include "inkdyn.cuh"

// one thread per (channel, unit); every thread reads its channel's operands before it writes anything
template <int M>
static __global__ void k_ink_dyn_conv(const ink_dyn_args args) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= args.ch || (int) blockIdx.y >= args.n_units) {
        return;
    }
    const ink_dyn_unit_args & u = args.u[blockIdx.y];

    float x[3 + M];
    float w[4];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        x[j] = u.keep[c*u.keep_nb1 + j];
    }
#pragma unroll
    for (int i = 0; i < M; ++i) {
        x[3 + i] = u.x[i*u.x_nb0 + c*u.x_nb1];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = u.w[c*u.w_nb1 + j];
    }
    const int h = c / u.hs;

    // the conv state stores (CPY of concat columns store_col .. store_col + 2), in graph order
    for (int k = 0; k < u.n_store; ++k) {
        float * st = u.store[k] + (int64_t) c*3;
#pragma unroll
        for (int q = 0; q <= M; ++q) {
            if (q == u.store_col[k]) {
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    st[j] = x[q + j];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < M; ++i) {
        // ssm_conv_f32: sumf = 0, += x[i + j]*w[j], += bias (none: 0)
        float v = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v += x[i + j] * w[j];
        }
        v += 0.0f;
        // the dynamic taps: y = y + window*tap, each product and sum rounded on its own (separate MUL and ADD nodes)
#pragma unroll
        for (int t = 0; t < INK_DYN_MAX_TERMS; ++t) {
            if (t < u.n_terms) {
                const float tp = u.tap[t][h*u.tap_nb1[t] + i*u.tap_nb2[t]];
                float xc = 0.0f;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    if (q == u.col[t]) {
                        xc = x[i + q];
                    }
                }
                v = __fadd_rn(v, __fmul_rn(xc, tp));
            }
        }
        // the epilogue: ADD(x, y), then SCALE (v*s + b, as scale_f32 / elem_chain)
        v = x[3 + i] + v;
        v = u.scale * v + u.bias;
        u.out[(int64_t) i*args.ch + c] = v;
    }
}

void ggml_cuda_ink_dyn_conv(ggml_backend_cuda_context & ctx, const ink_dyn_args & args) {
    const int bs = 128;
    const dim3 grid((unsigned) ((args.ch + bs - 1)/bs), (unsigned) args.n_units);
    cudaStream_t st = ctx.stream();
    switch (args.M) {
        case 1: k_ink_dyn_conv<1><<<grid, bs, 0, st>>>(args); break;
        case 2: k_ink_dyn_conv<2><<<grid, bs, 0, st>>>(args); break;
        case 3: k_ink_dyn_conv<3><<<grid, bs, 0, st>>>(args); break;
        case 4: k_ink_dyn_conv<4><<<grid, bs, 0, st>>>(args); break;
        case 5: k_ink_dyn_conv<5><<<grid, bs, 0, st>>>(args); break;
        case 6: k_ink_dyn_conv<6><<<grid, bs, 0, st>>>(args); break;
        case 7: k_ink_dyn_conv<7><<<grid, bs, 0, st>>>(args); break;
        case 8: k_ink_dyn_conv<8><<<grid, bs, 0, st>>>(args); break;
        default: GGML_ABORT("ink_dyn_conv: M");
    }
    CUDA_CHECK(cudaGetLastError());
}
