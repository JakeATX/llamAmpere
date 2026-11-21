#include "depth-mix.cuh"

#define DEPTH_MIX_THREADS 512
#define DEPTH_MIX_MAX_ROWS 64

// one warp per state row scores it against the nq queries (cur also gets its sum of squares for the rms norm),
// then the softmax over the rows and the weighted sum of the rows; the normed cur row is never written
template <int NQ>
static __global__ void k_depth_mix(
        const char * __restrict__ stack, const int64_t s_nb1, const int64_t s_nb2, const int k,
        const char * __restrict__ cur, const int64_t c_nb1, const char * __restrict__ query, const int64_t q_nb1,
        float * __restrict__ dst, const int n, const float eps, const float scale) {
    __shared__ float p[DEPTH_MIX_MAX_ROWS*NQ];
    __shared__ float s_r;

    const int t      = blockIdx.x;
    const int lane   = threadIdx.x % WARP_SIZE;
    const int warp   = threadIdx.x / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;

    const float * x = (const float *) (cur + t*c_nb1);
    const char  * s = stack + t*s_nb2;

    for (int j = warp; j <= k; j += nwarps) {
        const float * row = j < k ? (const float *) (s + j*s_nb1) : x;
        float d[NQ] = {0.0f};
        float ss = 0.0f;
        for (int i = lane; i < n; i += WARP_SIZE) {
            const float v = row[i];
            ss += v*v;
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                d[q] += v*((const float *) (query + q*q_nb1))[i];
            }
        }
        ss = warp_reduce_sum(ss);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            d[q] = warp_reduce_sum(d[q]);
        }
        if (lane == 0) {
            float r = 1.0f;
            if (j == k) {
                r   = rsqrtf(ss/n + eps);
                s_r = r;
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                p[j*NQ + q] = scale*r*d[q];
            }
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
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc[NQ] = {0.0f};
        for (int j = 0; j < k; ++j) {
            const float v = ((const float *) (s + j*s_nb1))[i];
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                acc[q] += p[j*NQ + q]*v;
            }
        }
        const float v = x[i]*r;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            out[q*n + i] = acc[q] + p[k*NQ + q]*v;
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
#define DEPTH_MIX_LAUNCH(NQ) k_depth_mix<NQ><<<T, DEPTH_MIX_THREADS, 0, stream>>>(sd, s1, s2, k, \
        (const char *) cur->data, cur->nb[1], (const char *) query->data, query->nb[1], (float *) dst->data, n, eps, scale)
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
