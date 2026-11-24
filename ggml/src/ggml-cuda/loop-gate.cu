#include "loop-gate.cuh"
#include "inklg_vec.cuh"
#include "inklg_pp.cuh"
#include "convert.cuh"

#define LOOP_GATE_CHUNKS 8   // chunks per input half (x, m); one warp each
#define LOOP_GATE_WARPS  4
#define LOOP_GATE_MAX_T  8
#define LOOP_GATE_MAX_R  128

// part[t][c][r] = dot(down[r, half c], chunk c of x or m) for c < 2*CHUNKS; part[t][c][R] = its sum of squares
template <typename TW, int RB>
static __global__ void k_loop_gate_down(
        const char * __restrict__ x, const int64_t x_nb1, const char * __restrict__ m, const int64_t m_nb1,
        const TW * __restrict__ down, float * __restrict__ part, const int n, const int R, const int T) {
    // RB consecutive rows r per block (LGTUNE, GGML_CUDA_LG_RB); each (r, chunk) partial is bit-identical for any RB
    const int r0   = blockIdx.x*RB;
    const int c    = blockIdx.y*LOOP_GATE_WARPS + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int half = c / LOOP_GATE_CHUNKS;
    const int cl   = (n + LOOP_GATE_CHUNKS - 1)/LOOP_GATE_CHUNKS;
    const int i0   = (c % LOOP_GATE_CHUNKS)*cl;
    const int i1   = min(n, i0 + cl);

    const char   * src = half ? m : x;
    const int64_t  nb1 = half ? m_nb1 : x_nb1;
    const TW     * w   = down + (int64_t) r0*2*n + half*n;

    float d[RB][LOOP_GATE_MAX_T] = {{0.0f}};
    float ss[LOOP_GATE_MAX_T]    = {0.0f};
#pragma unroll 4
    for (int i = i0 + lane; i < i1; i += WARP_SIZE) {
        float wi[RB];
#pragma unroll
        for (int b = 0; b < RB; ++b) {
            wi[b] = r0 + b < R ? ggml_cuda_cast<float>(w[(int64_t) b*2*n + i]) : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < LOOP_GATE_MAX_T; ++t) {
            if (t < T) {
                const float v = ((const float *) (src + t*nb1))[i];
#pragma unroll
                for (int b = 0; b < RB; ++b) {
                    d[b][t] += wi[b]*v;
                }
                ss[t] += v*v;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < LOOP_GATE_MAX_T; ++t) {
        if (t < T) {
            if (r0 == 0) {
                ss[t] = warp_reduce_sum(ss[t]);
            }
            float * p = part + ((int64_t) t*2*LOOP_GATE_CHUNKS + c)*(R + 1);
#pragma unroll
            for (int b = 0; b < RB; ++b) {
                d[b][t] = warp_reduce_sum(d[b][t]);
                if (lane == 0 && r0 + b < R) {
                    p[r0 + b] = d[b][t];
                }
            }
            if (lane == 0 && r0 == 0) {
                p[R] = ss[t];
            }
        }
    }
}

// g = rms(x)*down_x.x + rms(m)*down_m.m from the chunk partials (once per block, in smem), then a warp per output row i:
// lanes over r (coalesced weight reads), out = x - tanh(F g) x + tanh(G g) m
#define LOOP_GATE_APPLY_WARPS 8
#define LOOP_GATE_APPLY_ROWS  2   // rows per warp
template <typename TW, int ROWS>
static __global__ void k_loop_gate_apply(
        const char * __restrict__ x, const int64_t x_nb1, const char * __restrict__ m, const int64_t m_nb1,
        const TW * __restrict__ gain, const TW * __restrict__ forget, const float * __restrict__ part,
        float * __restrict__ dst, const int n, const int R, const int T, const float eps) {
    __shared__ float g[LOOP_GATE_MAX_T*LOOP_GATE_MAX_R];
    __shared__ float rs[2*LOOP_GATE_MAX_T];

    for (int j = threadIdx.x; j < 2*T; j += blockDim.x) {
        const float * p = part + ((int64_t) (j/2)*2*LOOP_GATE_CHUNKS + (j % 2)*LOOP_GATE_CHUNKS)*(R + 1);
        float s = 0.0f;
        for (int c = 0; c < LOOP_GATE_CHUNKS; ++c) {
            s += p[c*(R + 1) + R];
        }
        rs[j] = rsqrtf(s/n + eps);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < T*R; j += blockDim.x) {
        const int t = j / R;
        const int r = j % R;
        const float * p = part + (int64_t) t*2*LOOP_GATE_CHUNKS*(R + 1) + r;
        float gx = 0.0f;
        float gm = 0.0f;
        for (int c = 0; c < LOOP_GATE_CHUNKS; ++c) {
            gx += p[c*(R + 1)];
            gm += p[(LOOP_GATE_CHUNKS + c)*(R + 1)];
        }
        g[t*R + r] = rs[2*t]*gx + rs[2*t + 1]*gm;
    }
    __syncthreads();

    const int lane = threadIdx.x % WARP_SIZE;
    const int row0 = (blockIdx.x*LOOP_GATE_APPLY_WARPS + threadIdx.x/WARP_SIZE)*ROWS;
#pragma unroll
    for (int k = 0; k < ROWS; ++k) {
        const int i = row0 + k;
        if (i >= n) {
            return;
        }
        float ag[LOOP_GATE_MAX_T] = {0.0f};
        float af[LOOP_GATE_MAX_T] = {0.0f};
        const TW * wg = gain   + (int64_t) i*R;
        const TW * wf = forget + (int64_t) i*R;
        for (int r = lane; r < R; r += WARP_SIZE) {
            const float a = ggml_cuda_cast<float>(wg[r]);
            const float b = ggml_cuda_cast<float>(wf[r]);
#pragma unroll
            for (int t = 0; t < LOOP_GATE_MAX_T; ++t) {
                if (t < T) {
                    ag[t] += a*g[t*R + r];
                    af[t] += b*g[t*R + r];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < LOOP_GATE_MAX_T; ++t) {
            if (t < T) {
                ag[t] = warp_reduce_sum(ag[t]);
                af[t] = warp_reduce_sum(af[t]);
                if (lane == t) {
                    const float xi = ((const float *) (x + t*x_nb1))[i];
                    const float mi = ((const float *) (m + t*m_nb1))[i];
                    dst[(int64_t) t*n + i] = (xi - tanhf(af[t])*xi) + tanhf(ag[t])*mi;
                }
            }
        }
    }
}

bool ggml_cuda_loop_gate_supported(const ggml_tensor * op) {
    const ggml_type wt = op->src[2]->type;
    return (wt == GGML_TYPE_F32 || wt == GGML_TYPE_F16 || wt == GGML_TYPE_BF16) && op->src[3]->type == wt &&
        op->src[0]->ne[1] <= LOOP_GATE_MAX_T && op->src[2]->ne[1] <= LOOP_GATE_MAX_R;
}

bool ggml_cuda_loop_gate_op_supported(const ggml_tensor * op) {
    return op->src[0]->ne[1] > LOOP_GATE_MAX_T ? ggml_cuda_ink_loop_gate_pp_supported(op) : ggml_cuda_loop_gate_supported(op);
}

// LGTUNE for the single loop gate (k_loop_gate_down/apply): its own shape knobs, GGML_CUDA_LG1_RB (1/2/4, default 1) and
// GGML_CUDA_LG1_ROWS (2/4/8, default 2). nsys rc5 (3090, AI4 fold): RB=4 ROWS=4, the best shape for the multi-gate
// k_ink_lg_*, made this one slower (apply 5.6 -> 7.9 us, down 3.9 -> 4.3 us). Launch shape only, bit-identical.
static int loop_gate_tune_rb() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_LG1_RB"); const int k = e && *e ? atoi(e) : 1; return k == 2 || k == 4 ? k : 1; }();
    return v;
}
static int loop_gate_tune_rows() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_LG1_ROWS"); const int k = e && *e ? atoi(e) : 2; return k == 4 || k == 8 ? k : 2; }();
    return v;
}

template <typename TW, int RB, int ROWS>
static void loop_gate_launch_t(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * m = dst->src[1];
    const int n = (int) x->ne[0];
    const int T = (int) x->ne[1];
    const int R = (int) dst->src[2]->ne[1];

    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) T*2*LOOP_GATE_CHUNKS*(R + 1));
    cudaStream_t stream = ctx.stream();

    k_loop_gate_down<TW, RB><<<dim3((R + RB - 1)/RB, 2*LOOP_GATE_CHUNKS/LOOP_GATE_WARPS), LOOP_GATE_WARPS*WARP_SIZE, 0, stream>>>(
        (const char *) x->data, x->nb[1], (const char *) m->data, m->nb[1], (const TW *) dst->src[2]->data, part.get(), n, R, T);
    const int rows_per_block = LOOP_GATE_APPLY_WARPS*ROWS;
    k_loop_gate_apply<TW, ROWS><<<(n + rows_per_block - 1)/rows_per_block, LOOP_GATE_APPLY_WARPS*WARP_SIZE, 0, stream>>>(
        (const char *) x->data, x->nb[1], (const char *) m->data, m->nb[1], (const TW *) dst->src[3]->data,
        (const TW *) dst->src[4]->data, part.get(), (float *) dst->data, n, R, T, ggml_get_op_params_f32(dst, 0));
    CUDA_CHECK(cudaGetLastError());
}

template <typename TW, int RB>
static void loop_gate_launch_r(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (loop_gate_tune_rows()) {
        case 4:  loop_gate_launch_t<TW, RB, 4>(ctx, dst); break;
        case 8:  loop_gate_launch_t<TW, RB, 8>(ctx, dst); break;
        default: loop_gate_launch_t<TW, RB, 2>(ctx, dst); break;
    }
}

template <typename TW>
static void loop_gate_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (loop_gate_tune_rb()) {
        case 2:  loop_gate_launch_r<TW, 2>(ctx, dst); break;
        case 4:  loop_gate_launch_r<TW, 4>(ctx, dst); break;
        default: loop_gate_launch_r<TW, 1>(ctx, dst); break;
    }
}

void ggml_cuda_loop_gate(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (dst->src[0]->ne[1] > LOOP_GATE_MAX_T) {
        ggml_cuda_ink_loop_gate_pp(ctx, dst);
        return;
    }
    GGML_ASSERT(ggml_cuda_loop_gate_supported(dst));
    if (dst->src[0]->ne[1] == 0) {
        return;
    }
    ggml_tensor * g1[1] = { dst };
    if (ggml_cuda_ink_loop_gate_vec(ctx, g1, 1)) {
        return;
    }
    switch (dst->src[2]->type) {
        case GGML_TYPE_F16:  loop_gate_launch<half>(ctx, dst);        break;
        case GGML_TYPE_BF16: loop_gate_launch<nv_bfloat16>(ctx, dst); break;
        default:             loop_gate_launch<float>(ctx, dst);       break;
    }
}
