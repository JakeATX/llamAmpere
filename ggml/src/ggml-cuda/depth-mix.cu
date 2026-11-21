#include "depth-mix.cuh"

#define DEPTH_MIX_THREADS 1024
#define DEPTH_MIX_MAX_ROWS 64

template <int W>
static __device__ __forceinline__ void depth_mix_load(float (&a)[W], const float * p) {
    if constexpr (W == 4) {
        const float4 v = *(const float4 *) p;
        a[0] = v.x; a[1] = v.y; a[2] = v.z; a[3] = v.w;
    } else {
        a[0] = *p;
    }
}

// scores: each state row is split into c chunks so that the (k+1)*c (row, chunk) items cover the warps once; a warp
// scores its chunk against the nq queries (plus the sum of squares, used for cur's rms norm) and the chunk partials are
// summed per row. Then the softmax over the rows and the weighted sum of the rows; the normed cur row is never written.
// W = 4: float4 loads (rows 16-byte aligned, n % 4 == 0).
template <int NQ, int W>
static __global__ void __launch_bounds__(DEPTH_MIX_THREADS) k_depth_mix(
        const char * __restrict__ stack, const int64_t s_nb1, const int64_t s_nb2, const int k,
        const char * __restrict__ cur, const int64_t c_nb1, const char * __restrict__ query, const int64_t q_nb1,
        float * __restrict__ dst, const int n, const float eps, const float scale) {
    __shared__ float part[DEPTH_MIX_MAX_ROWS*(NQ + 1)];
    __shared__ float p[DEPTH_MIX_MAX_ROWS*NQ];
    __shared__ float s_r;

    const int t      = blockIdx.x;
    const int lane   = threadIdx.x % WARP_SIZE;
    const int warp   = threadIdx.x / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    const int nv     = n / W;
    const int nc     = max(1, nwarps/(k + 1));
    const int cl     = (nv + nc - 1)/nc;

    const float * x = (const float *) (cur + t*c_nb1);
    const char  * s = stack + t*s_nb2;

    for (int it = warp; it < (k + 1)*nc; it += nwarps) {
        const int j = it / nc;
        const int c = it % nc;
        const float * row = j < k ? (const float *) (s + j*s_nb1) : x;
        float d[NQ] = {0.0f};
        float ss = 0.0f;
        for (int v = c*cl + lane; v < min(nv, (c + 1)*cl); v += WARP_SIZE) {
            float a[W];
            depth_mix_load<W>(a, row + v*W);
#pragma unroll
            for (int w = 0; w < W; ++w) {
                ss += a[w]*a[w];
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float b[W];
                depth_mix_load<W>(b, (const float *) (query + q*q_nb1) + v*W);
#pragma unroll
                for (int w = 0; w < W; ++w) {
                    d[q] += a[w]*b[w];
                }
            }
        }
        ss = warp_reduce_sum(ss);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            d[q] = warp_reduce_sum(d[q]);
        }
        if (lane == 0) {
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                part[it*(NQ + 1) + q] = d[q];
            }
            part[it*(NQ + 1) + NQ] = ss;
        }
    }
    __syncthreads();

    if (threadIdx.x <= k) {
        const int j = threadIdx.x;
        float d[NQ + 1] = {0.0f};
        for (int c = 0; c < nc; ++c) {
#pragma unroll
            for (int q = 0; q <= NQ; ++q) {
                d[q] += part[(j*nc + c)*(NQ + 1) + q];
            }
        }
        float r = 1.0f;
        if (j == k) {
            r   = rsqrtf(d[NQ]/n + eps);
            s_r = r;
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            p[j*NQ + q] = scale*r*d[q];
        }
    }
    __syncthreads();

    if (threadIdx.x < NQ) {
        const int q = threadIdx.x;
        float mx = -INFINITY;
        for (int j = 0; j <= k; ++j) {
            mx = fmaxf(mx, p[j*NQ + q]);
        }
        float sum = 0.0f;
        for (int j = 0; j <= k; ++j) {
            const float e = expf(p[j*NQ + q] - mx);
            p[j*NQ + q] = e;
            sum += e;
        }
        const float inv = 1.0f/sum;
        for (int j = 0; j <= k; ++j) {
            p[j*NQ + q] *= inv;
        }
    }
    __syncthreads();

    const float r = s_r;
    float * out = dst + (int64_t) t*NQ*n;
    for (int v = threadIdx.x; v < nv; v += blockDim.x) {
        float acc[NQ][W] = {{0.0f}};
        for (int j = 0; j < k; ++j) {
            float a[W];
            depth_mix_load<W>(a, (const float *) (s + j*s_nb1) + v*W);
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
#pragma unroll
                for (int w = 0; w < W; ++w) {
                    acc[q][w] += p[j*NQ + q]*a[w];
                }
            }
        }
        float a[W];
        depth_mix_load<W>(a, x + v*W);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
#pragma unroll
            for (int w = 0; w < W; ++w) {
                out[q*n + v*W + w] = acc[q][w] + p[k*NQ + q]*(a[w]*r);
            }
        }
    }
}

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * stack = op->src[2];
    const int64_t k = stack ? stack->ne[1] : 0;
    return op->src[1]->ne[1] >= 1 && op->src[1]->ne[1] <= 4 && k + 1 <= DEPTH_MIX_MAX_ROWS && ggml_is_contiguous(op);
}

void ggml_cuda_depth_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_depth_mix_supported(dst));
    const ggml_tensor * cur   = dst->src[0];
    const ggml_tensor * query = dst->src[1];
    const ggml_tensor * stack = dst->src[2];

    const float eps   = ggml_get_op_params_f32(dst, 0);
    const float scale = ggml_get_op_params_f32(dst, 1);

    const int     n  = (int) cur->ne[0];
    const int64_t T  = cur->ne[1];
    const int     k  = stack ? (int) stack->ne[1] : 0;
    const char *  sd = stack ? (const char *) stack->data : nullptr;
    const int64_t s1 = stack ? stack->nb[1] : 0;
    const int64_t s2 = stack ? stack->nb[2] : 0;

    if (T == 0) {
        return;
    }
    cudaStream_t stream = ctx.stream();
    const bool vec = n % 4 == 0 && (s1 | s2 | cur->nb[1] | query->nb[1]) % 16 == 0 &&
        ((uintptr_t) sd | (uintptr_t) cur->data | (uintptr_t) query->data | (uintptr_t) dst->data) % 16 == 0;
#define DEPTH_MIX_LAUNCH(NQ) (vec ? k_depth_mix<NQ, 4><<<T, DEPTH_MIX_THREADS, 0, stream>>>(sd, s1, s2, k, \
        (const char *) cur->data, cur->nb[1], (const char *) query->data, query->nb[1], (float *) dst->data, n, eps, scale) : \
    k_depth_mix<NQ, 1><<<T, DEPTH_MIX_THREADS, 0, stream>>>(sd, s1, s2, k, \
        (const char *) cur->data, cur->nb[1], (const char *) query->data, query->nb[1], (float *) dst->data, n, eps, scale))
    switch (query->ne[1]) {
        case 1: DEPTH_MIX_LAUNCH(1); break;
        case 2: DEPTH_MIX_LAUNCH(2); break;
        case 3: DEPTH_MIX_LAUNCH(3); break;
        case 4: DEPTH_MIX_LAUNCH(4); break;
        default: GGML_ABORT("depth_mix: nq > 4");
    }
#undef DEPTH_MIX_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
