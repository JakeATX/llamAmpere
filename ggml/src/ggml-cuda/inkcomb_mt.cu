// [T5-GLUE] G2: Multi-token (T = 1..8) coalesced MoE weighted expert sum + short-conv ring + residual epilogue,
// plus coalesced standalone k_ink_moe_wsum_coalesced. 100% bit-identical to unfused k_ink_moe_wsum + ssm_conv_ring_f32.
// Kill switch: GGML_CUDA_INK_WSUM_CONV_MT=0 (default 1 = on).

#include "inkcomb_mt.cuh"

#include <cstdlib>

bool ggml_cuda_ink_wsum_conv_mt_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_INK_WSUM_CONV_MT");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// Exact butterfly reduction of k = 0..n_used-1 matching warp_reduce_sum (reduce_rows_f32 with 1 col/lane).
template <int N_USED_MAX>
static __device__ __forceinline__ float wsum_butterfly_reduce(float (&r)[N_USED_MAX]) {
#pragma unroll
    for (int off = N_USED_MAX / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int l = 0; l < off; ++l) {
            r[l] = r[l] + r[l + off];
        }
    }
    return r[0];
}

// Multi-token fused MoE weighted sum + d_conv=4 ring conv + epilogue + residual add.
// 1 thread per channel c (100% coalesced reads of e[t, k, c] and coalesced stores of sum/dst, float4 load of cw[c]).
template <int N_T, int N_USED_MAX>
static __global__ void __launch_bounds__(128) k_ink_wsum_conv_mt(
        const char * __restrict__ e, const int64_t e_nb1, const int64_t e_nb2,
        const char * __restrict__ wt, const int64_t w_nb1, const int64_t w_nb2,
        const int n_used, float * __restrict__ sum_out,
        const ggml_cuda_conv_ring_args args, const float * __restrict__ cw, const int cw_nb1,
        float * __restrict__ dst, const int dst_nb1,
        const float epi_scale, const float epi_bias, const float * __restrict__ res, const int n_embd) {
    // Load router weights w[t, k] once per block into SMEM (tiny: <= 8 * 32 * 4 = 1 KB)
    __shared__ float sw[N_T][N_USED_MAX];
    const int tid = threadIdx.x;
    for (int idx = tid; idx < N_T * N_USED_MAX; idx += blockDim.x) {
        const int t = idx / N_USED_MAX;
        const int k = idx % N_USED_MAX;
        sw[t][k] = k < n_used ? *(const float *) (wt + t * w_nb2 + k * w_nb1) : 0.0f;
    }
    __syncthreads();

    const int c = blockIdx.x * blockDim.x + tid;
    if (c >= n_embd) {
        return;
    }

    constexpr int d_conv = 4;
    constexpr int n_keep = d_conv - 1;
    float x[n_keep + N_T];

    // Read kept state columns [0..2]
    const float * xs = (const float *) ((const char *) args.state + (int64_t) c * args.state_nb1);
#pragma unroll
    for (int j = 0; j < n_keep; ++j) {
        x[j] = xs[j];
    }

    // Compute bit-identical weighted expert sum for each token i = 0..N_T-1 (100% coalesced across c!)
#pragma unroll
    for (int i = 0; i < N_T; ++i) {
        float r[N_USED_MAX];
        const char * e_tok = e + (int64_t) i * e_nb2 + (int64_t) c * sizeof(float);
#pragma unroll
        for (int k = 0; k < N_USED_MAX; ++k) {
            if (k < n_used) {
                const float ev = *(const float *) (e_tok + (int64_t) k * e_nb1);
                r[k] = 0.0f + __fmul_rn(ev, sw[i][k]);
            } else {
                r[k] = 0.0f;
            }
        }
        const float xn = wsum_butterfly_reduce<N_USED_MAX>(r);
        x[n_keep + i] = xn;
        sum_out[(int64_t) i * n_embd + c] = xn;
    }

    // Load 4 conv weights (vectorized float4 when stride == 4 and aligned)
    float w[d_conv];
    const int stride_w = cw_nb1 / (int) sizeof(float);
    if (stride_w == 4 && (((uintptr_t) cw) & 15) == 0) {
        const float4 wv = *reinterpret_cast<const float4 *>(cw + (int64_t) c * 4);
        w[0] = wv.x; w[1] = wv.y; w[2] = wv.z; w[3] = wv.w;
    } else {
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            w[j] = cw[(int64_t) c * stride_w + j];
        }
    }

    // Write conv-state snapshot windows in exact graph order (bit-identical to ssm_conv_ring_f32)
#pragma unroll
    for (int k = 0; k < GGML_CUDA_CONV_RING_MAX_WIN; ++k) {
        if (k < args.n_win) {
            const ggml_cuda_conv_ring_window & win = args.win[k];
            const int64_t row = win.rows != nullptr ? (int64_t) win.rows[0] : 0;
            float * o = win.dst + row * win.row_stride + (int64_t) c * n_keep;
#pragma unroll
            for (int q = 0; q <= N_T; ++q) {
                if (q == win.s_idx) {
#pragma unroll
                    for (int j = 0; j < n_keep; ++j) {
                        o[j] = x[q + j];
                    }
                }
            }
        }
    }

    // Short-conv + Inkling residual epilogue (exact ssm_conv_ring_f32<false, 4, N_T, true> order)
    const int stride_y = dst_nb1 / (int) sizeof(float);
#pragma unroll
    for (int i = 0; i < N_T; ++i) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sumf += x[i + j] * w[j];
        }
        sumf += 0.0f;
        float v = x[n_keep + i] + sumf;
        v = epi_scale * v + epi_bias;
        if (res != nullptr) {
            v = res[c + (int64_t) i * stride_y] + v;
        }
        dst[c + (int64_t) i * stride_y] = v;
    }
}

template <int N_T>
static void launch_wsum_conv_mt(
        const char * e, int64_t e_nb1, int64_t e_nb2,
        const char * wt, int64_t w_nb1, int64_t w_nb2,
        int n_used, float * sum_out,
        const ggml_cuda_conv_ring_args & args, const float * cw, int cw_nb1,
        float * dst, int dst_nb1, float epi_scale, float epi_bias, const float * res, int n_embd,
        cudaStream_t stream) {
    const int bs = 128;
    const dim3 grid((unsigned) ((n_embd + bs - 1) / bs));
    if (n_used <= 8) {
        k_ink_wsum_conv_mt<N_T, 8><<<grid, bs, 0, stream>>>(
            e, e_nb1, e_nb2, wt, w_nb1, w_nb2, n_used, sum_out, args, cw, cw_nb1, dst, dst_nb1, epi_scale, epi_bias, res, n_embd);
    } else {
        k_ink_wsum_conv_mt<N_T, WARP_SIZE><<<grid, bs, 0, stream>>>(
            e, e_nb1, e_nb2, wt, w_nb1, w_nb2, n_used, sum_out, args, cw, cw_nb1, dst, dst_nb1, epi_scale, epi_bias, res, n_embd);
    }
}

bool ggml_cuda_ink_wsum_conv_mt(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w,
        ggml_tensor * sum, const ggml_tensor * conv, ggml_tensor * out, float scale, float bias, const ggml_tensor * res) {
    if (!ggml_cuda_ink_wsum_conv_mt_enabled()) {
        return false;
    }
    const ggml_tensor * cw = conv->src[1];
    const int64_t n_embd = e->ne[0];
    const int     n_used = (int) e->ne[1];
    const int     n_t    = (int) e->ne[2];
    if (e->type != GGML_TYPE_F32 || e->nb[0] != sizeof(float) || n_t < 1 || n_t > GGML_CUDA_CONV_RING_MAX_T || e->ne[3] != 1 ||
            n_used < 1 || n_used > WARP_SIZE ||
            w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != n_used || w->ne[2] != n_t || w->ne[3] != 1 ||
            sum->type != GGML_TYPE_F32 || !ggml_is_contiguous(sum) || ggml_nelements(sum) != n_embd * n_t ||
            cw->type != GGML_TYPE_F32 || cw->ne[0] != 4 || cw->nb[0] != sizeof(float) || cw->ne[1] != n_embd ||
            conv->ne[0] != n_embd || conv->ne[1] != n_t || conv->ne[2] != 1 ||
            out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || ggml_nelements(out) != n_embd * n_t ||
            (res && (res->type != GGML_TYPE_F32 || !ggml_is_contiguous(res) || ggml_nelements(res) != n_embd * n_t)) ||
            n_embd > INT_MAX) {
        return false;
    }
    const int dst_nb1 = (int) (n_embd * sizeof(float));
    for (auto & r : ctx.conv_rings) {
        if (r.conv != conv) {
            continue;
        }
        // The ring's new rows must be sum itself: channel stride 4B, token stride n_embd*4B (when n_t > 1)
        if (r.args.x != (const float *) sum->data || r.args.x_nb1 != (int64_t) sizeof(float) ||
                (n_t > 1 && r.args.x_nb0 != (int64_t) (n_embd * sizeof(float)))) {
            return false;
        }
        cudaStream_t stream = ctx.stream();
        const char * ep = (const char *) e->data;
        const char * wp = (const char *) w->data;
        const float * rp = res ? (const float *) res->data : nullptr;
        switch (n_t) {
            case 1: launch_wsum_conv_mt<1>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            case 2: launch_wsum_conv_mt<2>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            case 3: launch_wsum_conv_mt<3>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            case 4: launch_wsum_conv_mt<4>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            case 5: launch_wsum_conv_mt<5>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            case 6: launch_wsum_conv_mt<6>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            case 7: launch_wsum_conv_mt<7>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            case 8: launch_wsum_conv_mt<8>(ep, e->nb[1], e->nb[2], wp, w->nb[1], w->nb[2], n_used, (float *) sum->data, r.args, (const float *) cw->data, (int) cw->nb[1], (float *) out->data, dst_nb1, scale, bias, rp, (int) n_embd, stream); break;
            default: return false;
        }
        CUDA_CHECK(cudaGetLastError());
        r.used = true;
        ctx.fusion_stats.conv_ring++;
        return true;
    }
    return false;
}

// Standalone coalesced MoE weighted sum (1 thread per output (c, t), 100% coalesced reads & writes, exact butterfly sum).
// Replaces the 32x uncoalesced 1-warp-per-element k_ink_moe_wsum whenever WCONV does not fuse.
template <int N_USED_MAX>
static __global__ void __launch_bounds__(256) k_ink_moe_wsum_coalesced(
        const char * __restrict__ e, const char * __restrict__ w, float * __restrict__ dst,
        const int n_embd, const int n_used,
        const int64_t e_nb1, const int64_t e_nb2, const int64_t w_nb1, const int64_t w_nb2) {
    __shared__ float sw[N_USED_MAX];
    const int t = blockIdx.y;
    if (threadIdx.x < N_USED_MAX) {
        sw[threadIdx.x] = threadIdx.x < n_used ? *(const float *) (w + (int64_t) t * w_nb2 + (int64_t) threadIdx.x * w_nb1) : 0.0f;
    }
    __syncthreads();

    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_embd) {
        return;
    }
    const char * e_base = e + (int64_t) t * e_nb2 + (int64_t) c * sizeof(float);
    float r[N_USED_MAX];
#pragma unroll
    for (int k = 0; k < N_USED_MAX; ++k) {
        if (k < n_used) {
            const float ev = *(const float *) (e_base + (int64_t) k * e_nb1);
            r[k] = 0.0f + __fmul_rn(ev, sw[k]);
        } else {
            r[k] = 0.0f;
        }
    }
    dst[(int64_t) t * n_embd + c] = wsum_butterfly_reduce<N_USED_MAX>(r);
}

bool ggml_cuda_ink_moe_wsum_coalesced(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w, ggml_tensor * dst) {
    if (!ggml_cuda_ink_wsum_conv_mt_enabled()) {
        return false;
    }
    const int n_embd = (int) e->ne[0];
    const int n_used = (int) e->ne[1];
    const int n_tok  = (int) e->ne[2];
    if (n_used < 1 || n_used > WARP_SIZE || n_tok < 1 || n_tok > 65535 || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int bs = 256;
    const dim3 grid((unsigned) ((n_embd + bs - 1) / bs), (unsigned) n_tok, 1);
    if (n_used <= 8) {
        k_ink_moe_wsum_coalesced<8><<<grid, bs, 0, ctx.stream()>>>(
            (const char *) e->data, (const char *) w->data, (float *) dst->data,
            n_embd, n_used, e->nb[1], e->nb[2], w->nb[1], w->nb[2]);
    } else {
        k_ink_moe_wsum_coalesced<WARP_SIZE><<<grid, bs, 0, ctx.stream()>>>(
            (const char *) e->data, (const char *) w->data, (float *) dst->data,
            n_embd, n_used, e->nb[1], e->nb[2], w->nb[1], w->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
