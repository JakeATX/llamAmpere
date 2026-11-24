// GGML_OP_LOOP_GATE at prompt widths, in F32 like the CPU op and loop-gate.cu (only the summation order differs).
// k_lgp_down: raw dots down_half . row and the row sum of squares, per (row tile, half, k split), into part.
// k_lgp_apply: g = rs_x*gx + rs_m*gm from part, then out = (x - tanh(F g) x) + tanh(G g) m per (row tile, column tile).
#include "inklg_pp.cuh"
#include "convert.cuh"

#define LGP_TT   32   // rows per block (both kernels)
#define LGP_RMAX 64   // largest rank
#define LGP_KT   32   // k tile of the down kernel
#define LGP_KS   2    // k splits per half in the down kernel
#define LGP_NI   64   // output columns per apply block
#define LGP_DOWN_THREADS  128
#define LGP_APPLY_THREADS 256

bool ggml_cuda_ink_loop_gate_pp_supported(const ggml_tensor * op) {
    const ggml_type wt = op->src[2]->type;
    return (wt == GGML_TYPE_F32 || wt == GGML_TYPE_F16 || wt == GGML_TYPE_BF16) && op->src[3]->type == wt &&
        op->src[4]->type == wt && op->src[2]->ne[1] >= 1 && op->src[2]->ne[1] <= LGP_RMAX && op->src[0]->ne[0] >= 1;
}

template <typename TW>
static __global__ void __launch_bounds__(LGP_DOWN_THREADS) k_lgp_down(
        const char * __restrict__ x, const int64_t x_nb1, const char * __restrict__ m, const int64_t m_nb1,
        const TW * __restrict__ down, float * __restrict__ part, const int n, const int R, const int T, const int kchunk) {
    __shared__ __align__(16) float xs[LGP_TT][LGP_KT + 4];   // rows x k (stride 36 keeps float4 reads aligned)
    __shared__ float ws[LGP_KT][LGP_RMAX + 1];   // k x r (stride 65: conflict-free on both store and load)
    static_assert(LGP_KT == WARP_SIZE && LGP_DOWN_THREADS == 4*WARP_SIZE && LGP_RMAX == 2*WARP_SIZE, "loader and tile shapes");

    const int tid   = threadIdx.x;
    const int lane  = tid % WARP_SIZE;
    const int warp  = tid / WARP_SIZE;            // rows warp*8 .. warp*8+7 of the tile
    const int t0    = blockIdx.x*LGP_TT;
    const int half  = blockIdx.y;
    const int split = blockIdx.z;
    const int kb    = split*kchunk;
    const int ke    = min(n, kb + kchunk);

    const char   * src = half ? m : x;
    const int64_t  nb1 = half ? m_nb1 : x_nb1;
    const TW     * w   = down + (int64_t) half*n;  // row r of this half: w + r*2n

    // loaders: k = lane, rows warp + 4q (one row per warp per step, coalesced); weights: k = lane, r = warp + 4q
    float acc_a[8] = {0.0f};   // r = lane
    float acc_b[8] = {0.0f};   // r = lane + 32
    float ss[8]    = {0.0f};   // row warp + 4q

    for (int k0 = kb; k0 < ke; k0 += LGP_KT) {
        const int k = k0 + lane;
#pragma unroll
        for (int q = 0; q < LGP_TT/4; ++q) {
            const int lt = warp + 4*q;
            const float v = t0 + lt < T && k < ke ? ((const float *) (src + (int64_t) (t0 + lt)*nb1))[k] : 0.0f;
            ss[q] += v*v;
            xs[lt][lane] = v;
        }
#pragma unroll
        for (int q = 0; q < LGP_RMAX/4; ++q) {
            const int r = warp + 4*q;
            ws[lane][r] = r < R && k < ke ? ggml_cuda_cast<float>(w[(int64_t) r*2*n + k]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < LGP_KT; kk += 4) {
            float wa[4], wb[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                wa[u] = ws[kk + u][lane];
                wb[u] = ws[kk + u][lane + 32];
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float4 v = *(const float4 *) &xs[warp*8 + j][kk];
                acc_a[j] += wa[0]*v.x + wa[1]*v.y + wa[2]*v.z + wa[3]*v.w;
                acc_b[j] += wb[0]*v.x + wb[1]*v.y + wb[2]*v.z + wb[3]*v.w;
            }
        }
        __syncthreads();
    }

    float * p = part + (int64_t) (split*2 + half)*T*(R + 1);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int t = t0 + warp*8 + j;
        if (t < T) {
            if (lane < R) {
                p[(int64_t) t*(R + 1) + lane] = acc_a[j];
            }
            if (lane + 32 < R) {
                p[(int64_t) t*(R + 1) + lane + 32] = acc_b[j];
            }
        }
    }
#pragma unroll
    for (int q = 0; q < LGP_TT/4; ++q) {
        const float v = warp_reduce_sum(ss[q]);
        const int t = t0 + warp + 4*q;
        if (lane == 0 && t < T) {
            p[(int64_t) t*(R + 1) + R] = v;
        }
    }
}

template <typename TW>
static __global__ void __launch_bounds__(LGP_APPLY_THREADS) k_lgp_apply(
        const char * __restrict__ x, const int64_t x_nb1, const char * __restrict__ m, const int64_t m_nb1,
        const TW * __restrict__ gain, const TW * __restrict__ forget, const float * __restrict__ part,
        float * __restrict__ dst, const int n, const int R, const int T, const int nsplit, const float eps) {
    __shared__ float wg[LGP_RMAX][LGP_NI + 1];   // r x column (stride 65)
    __shared__ float wf[LGP_RMAX][LGP_NI + 1];
    __shared__ __align__(16) float gs[LGP_TT][LGP_RMAX + 4];   // rows x r (stride 68 keeps float4 reads aligned)
    __shared__ float rs[2][LGP_TT];

    const int tid = threadIdx.x;
    const int i0  = blockIdx.x*LGP_NI;
    const int t0  = blockIdx.y*LGP_TT;
    const int RP  = (R + 3) & ~3;
    const int64_t pstride = (int64_t) T*(R + 1);

    if (tid < 2*LGP_TT) {
        const int t = tid % LGP_TT;
        const int h = tid / LGP_TT;
        float s = 0.0f;
        if (t0 + t < T) {
            for (int sp = 0; sp < nsplit; ++sp) {
                s += part[(sp*2 + h)*pstride + (int64_t) (t0 + t)*(R + 1) + R];
            }
        }
        rs[h][t] = rsqrtf(s/n + eps);
    }
    // weights of this block's columns, r = tid%64 (coalesced along r), column = tid/64 + 4q
    {
        const int r = tid % LGP_RMAX;
        for (int c = tid / LGP_RMAX; c < LGP_NI; c += LGP_APPLY_THREADS/LGP_RMAX) {
            const int i = i0 + c;
            const bool ok = r < R && i < n;
            wg[r][c] = ok ? ggml_cuda_cast<float>(gain  [(int64_t) i*R + r]) : 0.0f;
            wf[r][c] = ok ? ggml_cuda_cast<float>(forget[(int64_t) i*R + r]) : 0.0f;
        }
    }
    __syncthreads();
    for (int idx = tid; idx < LGP_TT*LGP_RMAX; idx += LGP_APPLY_THREADS) {
        const int t = idx / LGP_RMAX;
        const int r = idx % LGP_RMAX;
        float g = 0.0f;
        if (r < R && t0 + t < T) {
            float gx = 0.0f;
            float gm = 0.0f;
            const int64_t o = (int64_t) (t0 + t)*(R + 1) + r;
            for (int sp = 0; sp < nsplit; ++sp) {
                gx += part[(sp*2 + 0)*pstride + o];
                gm += part[(sp*2 + 1)*pstride + o];
            }
            g = rs[0][t]*gx + rs[1][t]*gm;
        }
        gs[t][r] = g;
    }
    __syncthreads();

    const int c  = tid % LGP_NI;
    const int tq = tid / LGP_NI;                  // rows tq*8 .. tq*8+7
    float ag[8] = {0.0f};
    float af[8] = {0.0f};
    for (int r = 0; r < RP; r += 4) {
        float a[4], b[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            a[u] = wg[r + u][c];
            b[u] = wf[r + u][c];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 g = *(const float4 *) &gs[tq*8 + j][r];
            ag[j] += a[0]*g.x + a[1]*g.y + a[2]*g.z + a[3]*g.w;
            af[j] += b[0]*g.x + b[1]*g.y + b[2]*g.z + b[3]*g.w;
        }
    }
    const int i = i0 + c;
    if (i >= n) {
        return;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int t = t0 + tq*8 + j;
        if (t < T) {
            const float xi = ((const float *) (x + (int64_t) t*x_nb1))[i];
            const float mi = ((const float *) (m + (int64_t) t*m_nb1))[i];
            dst[(int64_t) t*n + i] = (xi - tanhf(af[j])*xi) + tanhf(ag[j])*mi;
        }
    }
}

template <typename TW>
static void lgp_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * m = dst->src[1];
    const int n = (int) x->ne[0];
    const int T = (int) x->ne[1];
    const int R = (int) dst->src[2]->ne[1];
    // k chunk per split, a multiple of the k tile; a short row gets one split
    const int ntile  = (n + LGP_KT - 1)/LGP_KT;
    const int nsplit = ntile >= 2*LGP_KS ? LGP_KS : 1;
    const int kchunk = ((ntile + nsplit - 1)/nsplit)*LGP_KT;

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nsplit*2*T*(R + 1));
    cudaStream_t stream = ctx.stream();

    k_lgp_down<TW><<<dim3((T + LGP_TT - 1)/LGP_TT, 2, nsplit), LGP_DOWN_THREADS, 0, stream>>>(
        (const char *) x->data, x->nb[1], (const char *) m->data, m->nb[1], (const TW *) dst->src[2]->data, part.get(),
        n, R, T, kchunk);
    k_lgp_apply<TW><<<dim3((n + LGP_NI - 1)/LGP_NI, (T + LGP_TT - 1)/LGP_TT), LGP_APPLY_THREADS, 0, stream>>>(
        (const char *) x->data, x->nb[1], (const char *) m->data, m->nb[1], (const TW *) dst->src[3]->data,
        (const TW *) dst->src[4]->data, part.get(), (float *) dst->data, n, R, T, nsplit, ggml_get_op_params_f32(dst, 0));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_ink_loop_gate_pp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_ink_loop_gate_pp_supported(dst));
    if (dst->src[0]->ne[1] == 0) {
        return;
    }
    switch (dst->src[2]->type) {
        case GGML_TYPE_F16:  lgp_launch<half>(ctx, dst);        break;
        case GGML_TYPE_BF16: lgp_launch<nv_bfloat16>(ctx, dst); break;
        default:             lgp_launch<float>(ctx, dst);       break;
    }
}
