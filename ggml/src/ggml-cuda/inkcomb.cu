#include "inkcomb.cuh"

// One thread per channel c (one token, one sequence).
// 1) The weighted sum, bit for bit as MUL then SUM_ROWS: each product is rounded (the MUL writes it to memory), then
//    the 32 lane values (products, zero past n_used) are added in the butterfly order of warp_reduce_sum as seen by
//    lane 0 (reduce_rows_f32 with one column per thread, starting from 0.0f), here in registers of one thread.
// 2) The ring conv of ssm_conv_ring_f32<false, 4, 1, epi>: kept state columns then the new value, the windows in graph
//    order, sumf = 0, += x[j]*w[j], += 0 (no bias), then v = x + sumf, v = scale*v + bias, v = res + v.
template <int n_used_max>
static __global__ void k_ink_wsum_conv(const float * __restrict__ e, const int64_t e_nb1, const float * __restrict__ wt,
        const int64_t w_nb1, const int n_used, float * __restrict__ sum_out, const ggml_cuda_conv_ring_args args,
        const float * __restrict__ cw, const int cw_nb1, float * __restrict__ dst, const float epi_scale,
        const float epi_bias, const float * __restrict__ res, const int n_embd) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n_embd) {
        return;
    }
    float r[WARP_SIZE];
#pragma unroll
    for (int k = 0; k < WARP_SIZE; ++k) {
        r[k] = 0.0f;
        if (k < n_used_max && k < n_used) {
            const float ev = *(const float *) ((const char *) e + k*e_nb1 + c*sizeof(float));
            const float wv = *(const float *) ((const char *) wt + k*w_nb1);
            r[k] = 0.0f + __fmul_rn(ev, wv);
        }
    }
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
#pragma unroll
        for (int l = 0; l < off; ++l) {
            r[l] = r[l] + r[l + off];
        }
    }
    const float xn = r[0];
    *(float *) ((char *) sum_out + c*sizeof(float)) = xn;

    constexpr int d_conv = 4;
    constexpr int n_keep = d_conv - 1;
    float x[n_keep + 1];
    float w[d_conv];
    const float * xs = (const float *) ((const char *) args.state + c*args.state_nb1);
#pragma unroll
    for (int j = 0; j < n_keep; ++j) {
        x[j] = xs[j];
    }
    x[n_keep] = xn;
    const int stride_w = cw_nb1 / sizeof(float);
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        w[j] = cw[c*stride_w + j];
    }
#pragma unroll
    for (int k = 0; k < GGML_CUDA_CONV_RING_MAX_WIN; ++k) {
        if (k < args.n_win) {
            const ggml_cuda_conv_ring_window & win = args.win[k];
            const int64_t row = win.rows != nullptr ? (int64_t) win.rows[0] : 0;
            float * o = win.dst + row*win.row_stride + c*n_keep;
#pragma unroll
            for (int q = 0; q <= 1; ++q) {
                if (q == win.s_idx) {
#pragma unroll
                    for (int j = 0; j < n_keep; ++j) {
                        o[j] = x[q + j];
                    }
                }
            }
        }
    }
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; j++) {
        sumf += x[j] * w[j];
    }
    sumf += 0.0f;
    float v = sumf;
    v = x[n_keep] + v;
    v = epi_scale * v + epi_bias;
    if (res != nullptr) {
        v = res[c] + v;
    }
    dst[c] = v;
}

bool ggml_cuda_ink_wsum_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w,
        ggml_tensor * sum, const ggml_tensor * conv, ggml_tensor * out, float scale, float bias, const ggml_tensor * res) {
    const ggml_tensor * cw = conv->src[1];
    const int64_t n_embd = e->ne[0];
    const int     n_used = (int) e->ne[1];
    if (e->type != GGML_TYPE_F32 || e->nb[0] != sizeof(float) || e->ne[2] != 1 || e->ne[3] != 1 || n_used < 1 ||
            n_used > WARP_SIZE || w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != n_used ||
            sum->type != GGML_TYPE_F32 || !ggml_is_contiguous(sum) || ggml_nelements(sum) != n_embd ||
            cw->type != GGML_TYPE_F32 || cw->ne[0] != 4 || cw->nb[0] != sizeof(float) || cw->ne[1] != n_embd ||
            conv->ne[0] != n_embd || conv->ne[1] != 1 || conv->ne[2] != 1 ||
            out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) != n_embd ||
            (res && (res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || ggml_nelements(res) != n_embd)) ||
            n_embd > INT_MAX) {
        return false;
    }
    for (auto & r : ctx.conv_rings) {
        if (r.conv != conv) {
            continue;
        }
        // the ring's new row must be the sum itself (channel stride 4 bytes)
        if (r.args.x != (const float *) sum->data || r.args.x_nb1 != (int64_t) sizeof(float)) {
            return false;
        }
        const int bs = 128;
        const dim3 grid((unsigned) ((n_embd + bs - 1)/bs));
        k_ink_wsum_conv<WARP_SIZE><<<grid, bs, 0, ctx.stream()>>>((const float *) e->data, e->nb[1],
            (const float *) w->data, w->nb[1], n_used, (float *) sum->data, r.args, (const float *) cw->data,
            (int) cw->nb[1], (float *) out->data, scale, bias, res ? (const float *) res->data : nullptr, (int) n_embd);
        CUDA_CHECK(cudaGetLastError());
        r.used = true;
        ctx.fusion_stats.conv_ring++;
        return true;
    }
    return false;
}
