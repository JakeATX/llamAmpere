#include "fuse-kvconv.cuh"

struct kvconv_unit {
    ggml_cuda_conv_ring_args args;
    const float * w;
    int           w_nb1;
    float *       dst;
    int           dst_nb1;
    int           dst_nb2;
    float         epi_scale;
    float         epi_bias;
    const float * bias;    // conv bias: nullptr (the ring epilogue call passes none), read as the ring kernel does
    int           n_blk;   // blocks of 128 channels
};

// ssm_conv_ring_f32<apply_silu = false, d_conv = 4, n_t, epi = true> with no conv bias and no residual, for block row
// by of unit u (one thread per channel, sequence 0)
template <int n_t>
static __device__ __forceinline__ void kvconv_body(const kvconv_unit & u, const int by) {
    constexpr int d_conv = 4;
    constexpr int n_keep = d_conv - 1;
    constexpr int n_col  = n_keep + n_t;

    const int     tid = threadIdx.x;
    const int     s   = blockIdx.x;
    const int64_t c   = (int64_t) by * blockDim.x + tid;

    float x[n_col];
    float w[d_conv];

    const float * xs = (const float *) ((const char *) u.args.state + s * u.args.state_nb2 + c * u.args.state_nb1);
#pragma unroll
    for (int j = 0; j < n_keep; ++j) {
        x[j] = xs[j];
    }
    const char * xn = (const char *) u.args.x + s * u.args.x_nb2 + c * u.args.x_nb1;
#pragma unroll
    for (int i = 0; i < n_t; ++i) {
        x[n_keep + i] = *(const float *) (xn + i * u.args.x_nb0);
    }
    const int stride_w = u.w_nb1 / sizeof(float);
    if (stride_w == 4 && (((uintptr_t) u.w) & 15) == 0) {
        const float4 wv = *reinterpret_cast<const float4 *>(u.w + c * 4);
        w[0] = wv.x; w[1] = wv.y; w[2] = wv.z; w[3] = wv.w;
    } else {
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            w[j] = u.w[c * stride_w + j];
        }
    }
    const float b = u.bias != nullptr ? u.bias[c] : 0.0f;

    // windows in graph order (a later window wins a shared address)
#pragma unroll
    for (int k = 0; k < GGML_CUDA_CONV_RING_MAX_WIN; ++k) {
        if (k < u.args.n_win) {
            const ggml_cuda_conv_ring_window & win = u.args.win[k];
            const int64_t row = win.rows != nullptr ? (int64_t) win.rows[s] : (int64_t) s;
            float * out = win.dst + row * win.row_stride + c * n_keep;
#pragma unroll
            for (int q = 0; q <= n_t; ++q) {
                if (q == win.s_idx) {
#pragma unroll
                    for (int j = 0; j < n_keep; ++j) {
                        out[j] = x[q + j];
                    }
                }
            }
        }
    }

    float * y = (float *) ((char *) u.dst + s * u.dst_nb2) + c;
    const int stride_y = u.dst_nb1 / sizeof(float);
#pragma unroll
    for (int i = 0; i < n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; j++) {
            sumf += x[i + j] * w[j];
        }
        sumf += b;
        float v = sumf;
        v = x[n_keep + i] + v;
        v = u.epi_scale * v + u.epi_bias;
        y[i * stride_y] = v;
    }
}

template <int n_t>
static __global__ void k_fuse_kvconv(const kvconv_unit u0, const kvconv_unit u1) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    if ((int) blockIdx.y < u0.n_blk) {
        kvconv_body<n_t>(u0, blockIdx.y);
    } else {
        kvconv_body<n_t>(u1, blockIdx.y - u0.n_blk);
    }
}

// the checks of ggml_cuda_op_ssm_conv_ring_epi (no residual) for one unit; e: its plan entry
static bool kvconv_unit_prep(ggml_backend_cuda_context & ctx, ggml_tensor * conv, const ggml_tensor * x, ggml_tensor * out,
        const float scale, const float bias, kvconv_unit & u, ggml_backend_cuda_context::conv_ring_entry *& ep) {
    const ggml_tensor * src0 = conv->src[0];
    const ggml_tensor * src1 = conv->src[1];
    const int64_t nc  = src1->ne[0];
    const int64_t nr  = src0->ne[1];
    const int64_t n_t = conv->ne[1];
    const int64_t n_s = conv->ne[2];
    if (nc != 4 || n_s != 1 || n_t < 1 || n_t > GGML_CUDA_CONV_RING_MAX_T || src1->type != GGML_TYPE_F32 ||
            src1->nb[0] != sizeof(float) || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) ||
            ggml_nelements(out) != nr*n_t || out->ne[0] != nr || nr % 128 != 0 ||
            x->type != GGML_TYPE_F32 || x->ne[0] != nr || ggml_nrows(x) != n_t || x->nb[0] != sizeof(float)) {
        return false;
    }
    auto overlaps = [&](const ggml_tensor * t) {
        if (!t || !t->data) {
            return false;
        }
        const ggml_tensor * b = t->view_src ? t->view_src : t;
        const char * a0 = (const char *) out->data, * a1 = a0 + ggml_nbytes(out);
        const char * b0 = (const char *) b->data,   * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    for (int k = 0; k < GGML_MAX_SRC && src0->src[k]; ++k) {
        if (overlaps(src0->src[k])) {
            return false;
        }
    }
    if (overlaps(src0)) {
        return false;
    }
    for (auto & e : ctx.conv_rings) {
        if (e.conv != conv) {
            continue;
        }
        if (e.args.x != (const float *) x->data || e.args.x_nb1 != (int64_t) sizeof(float) ||
                (n_t > 1 && e.args.x_nb0 != (int64_t) x->nb[1])) {
            return false;
        }
        u.args      = e.args;
        u.w         = (const float *) src1->data;
        u.w_nb1     = (int) src1->nb[1];
        u.dst       = (float *) out->data;
        u.dst_nb1   = (int) out->nb[1];
        u.dst_nb2   = (int) (n_t*out->nb[1]);
        u.epi_scale = scale;
        u.epi_bias  = bias;
        u.bias      = nullptr;
        u.n_blk     = (int) (nr / 128);
        ep = &e;
        return true;
    }
    return false;
}

bool ggml_cuda_op_kvconv_pair(ggml_backend_cuda_context & ctx,
        ggml_tensor * conv1, const ggml_tensor * x1, ggml_tensor * out1, float scale1, float bias1,
        ggml_tensor * conv2, const ggml_tensor * x2, ggml_tensor * out2, float scale2, float bias2) {
    kvconv_unit u0, u1;
    ggml_backend_cuda_context::conv_ring_entry * e0 = nullptr, * e1 = nullptr;
    if (conv1->ne[1] != conv2->ne[1] || !kvconv_unit_prep(ctx, conv1, x1, out1, scale1, bias1, u0, e0) ||
            !kvconv_unit_prep(ctx, conv2, x2, out2, scale2, bias2, u1, e1)) {
        return false;
    }
    const int n_t = (int) conv1->ne[1];
    const ggml_cuda_kernel_launch_params lp(dim3(1, u0.n_blk + u1.n_blk, 1), dim3(128, 1, 1), 0, ctx.stream());
    switch (n_t) {
        case 1: ggml_cuda_kernel_launch(k_fuse_kvconv<1>, lp, u0, u1); break;
        case 2: ggml_cuda_kernel_launch(k_fuse_kvconv<2>, lp, u0, u1); break;
        case 3: ggml_cuda_kernel_launch(k_fuse_kvconv<3>, lp, u0, u1); break;
        case 4: ggml_cuda_kernel_launch(k_fuse_kvconv<4>, lp, u0, u1); break;
        case 5: ggml_cuda_kernel_launch(k_fuse_kvconv<5>, lp, u0, u1); break;
        case 6: ggml_cuda_kernel_launch(k_fuse_kvconv<6>, lp, u0, u1); break;
        case 7: ggml_cuda_kernel_launch(k_fuse_kvconv<7>, lp, u0, u1); break;
        case 8: ggml_cuda_kernel_launch(k_fuse_kvconv<8>, lp, u0, u1); break;
        default: GGML_ABORT("kvconv: n_t %d outside 1..8", n_t);
    }
    e0->used = true;
    e1->used = true;
    ctx.fusion_stats.conv_ring += 2;
    return true;
}
