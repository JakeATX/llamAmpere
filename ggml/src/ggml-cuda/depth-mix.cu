#include "depth-mix.cuh"
#include "rq4.cuh"
#include "prefill-glue-dev.cuh"

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
// [INKGLUE] NORM (NQ == 1): then the rms norm of the mixed row times nw into ndst, with rms_norm_f32<1024>'s scalar sum
// (col = tid, += 1024), block_reduce and expressions (bit-identical to the RMS_NORM + MUL nodes it replaces)
// [PREFILL-GLUE] KF = 32 / 64: aq / as take the NVFP4PF activation (k_nvfp4pf_xq layout) instead of the AI+ H256 one
template <int NQ, int W, bool NORM = false, bool SMEM_OUT = false, int KF = 0>
static __global__ void __launch_bounds__(DEPTH_MIX_THREADS) k_depth_mix(
        const char * __restrict__ stack, const int64_t s_nb1, const int64_t s_nb2, const int k,
        const char * __restrict__ cur, const int64_t c_nb1, const char * __restrict__ query, const int64_t q_nb1,
        float * dst, const int n, const int nq_all, const float eps, const float scale,
        const float * __restrict__ nw = nullptr, float * __restrict__ ndst = nullptr, const float neps = 0.0f,
        int8_t * __restrict__ aq = nullptr, float * __restrict__ as = nullptr) {
    extern __shared__ __align__(16) float s_out[];
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

    query += blockIdx.y*NQ*q_nb1;

    const float * x = (const float *) (cur + t*c_nb1);
    const char  * s = stack + t*s_nb2;

    for (int it = warp; it < (k + 1)*nc; it += nwarps) {
        const int j = it / nc;
        const int c = it % nc;
        const float * row = j < k ? (const float *) (s + j*s_nb1) : x;
        float d[NQ] = {0.0f};
        float ss = 0.0f;
#pragma unroll 4
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
    float * out = dst + ((int64_t) t*nq_all + blockIdx.y*NQ)*n;
    for (int v = threadIdx.x; v < nv; v += blockDim.x) {
        float acc[NQ][W] = {{0.0f}};
        // 4 state rows in flight before any of them is used
        for (int j0 = 0; j0 < k; j0 += 4) {
            float a[4][W];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (j0 + u < k) {
                    depth_mix_load<W>(a[u], (const float *) (s + (j0 + u)*s_nb1) + v*W);
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (j0 + u < k) {
#pragma unroll
                    for (int q = 0; q < NQ; ++q) {
#pragma unroll
                        for (int w = 0; w < W; ++w) {
                            acc[q][w] += p[(j0 + u)*NQ + q]*a[u][w];
                        }
                    }
                }
            }
        }
        float a[W];
        depth_mix_load<W>(a, x + v*W);
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if constexpr (W == 4) {
                const float pk = p[k*NQ + q];
                const float4 ov = make_float4(
                    acc[q][0] + pk*(a[0]*r),
                    acc[q][1] + pk*(a[1]*r),
                    acc[q][2] + pk*(a[2]*r),
                    acc[q][3] + pk*(a[3]*r));
                *reinterpret_cast<float4 *>(out + q*n + v*4) = ov;
                if constexpr (SMEM_OUT && NQ == 1) {
                    *reinterpret_cast<float4 *>(s_out + v*4) = ov;
                }
            } else {
                const float ov = acc[q][0] + p[k*NQ + q]*(a[0]*r);
                out[q*n + v] = ov;
                if constexpr (SMEM_OUT && NQ == 1) {
                    s_out[v] = ov;
                }
            }
        }
    }
    if constexpr (NORM) {
        static_assert(NQ == 1, "mix norm: one query");
        __shared__ float s_red[32];
        __syncthreads();
        const float * norm_src = SMEM_OUT ? s_out : out;
        float tmp = 0.0f;
        for (int col = threadIdx.x; col < n; col += DEPTH_MIX_THREADS) {
            const float xi = norm_src[col];
            tmp += xi * xi;
        }
        tmp = block_reduce<block_reduce_method::SUM, DEPTH_MIX_THREADS>(tmp, s_red);
        const float mean = tmp / n;
        const float sc   = rsqrtf(mean + neps);
        float * nd = ndst + (int64_t) t*n;
        for (int col = threadIdx.x; col < n; col += DEPTH_MIX_THREADS) {
            nd[col] = sc * norm_src[col] * nw[col];
        }
        // [INKGLUE MIXQ] then the AI+ H256 activation of the normed row, by rq4::quantize_act_body itself (one warp per 256
        // block; the warps of this block cover n <= 32*256), from the row this block just wrote
        if (aq != nullptr) {
            __syncthreads();
            if constexpr (KF == 0) {
                rq4::quantize_act_body<false, false, false>(ndst, n, aq, as, n, (int) gridDim.x, 0, (int) t);
            } else {
                pf_quantize_row_nvfp4pf_from_ptr<KF>(nd, aq, as, n, (int) t, DEPTH_MIX_THREADS/32);
            }
        }
    }
}

bool ggml_cuda_depth_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * stack = op->src[2];
    const int64_t k = stack ? stack->ne[1] : 0;
    const int64_t nq = op->src[1]->ne[1];
    return nq >= 1 && (nq <= 4 || (nq % 4 == 0 && nq/4 <= 65535)) && k + 1 <= DEPTH_MIX_MAX_ROWS && ggml_is_contiguous(op);
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
    // more than 4 queries: groups of 4, one block row each (all groups read the same states)
    const int  nq = (int) query->ne[1];
    const dim3 grid((unsigned) T, nq > 4 ? nq/4 : 1);
#define DEPTH_MIX_LAUNCH(NQ) (vec ? k_depth_mix<NQ, 4><<<grid, DEPTH_MIX_THREADS, 0, stream>>>(sd, s1, s2, k, \
        (const char *) cur->data, cur->nb[1], (const char *) query->data, query->nb[1], (float *) dst->data, n, nq, eps, scale) : \
    k_depth_mix<NQ, 1><<<grid, DEPTH_MIX_THREADS, 0, stream>>>(sd, s1, s2, k, \
        (const char *) cur->data, cur->nb[1], (const char *) query->data, query->nb[1], (float *) dst->data, n, nq, eps, scale))
    switch (nq > 4 ? 4 : nq) {
        case 1: DEPTH_MIX_LAUNCH(1); break;
        case 2: DEPTH_MIX_LAUNCH(2); break;
        case 3: DEPTH_MIX_LAUNCH(3); break;
        case 4: DEPTH_MIX_LAUNCH(4); break;
        default: GGML_ABORT("depth_mix: nq > 4");
    }
#undef DEPTH_MIX_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

// [INKGLUE] depth mix (one query) + RMS_NORM(eps) + MUL(w) of the mixed rows into ndst; false: not this shape
bool ggml_cuda_depth_mix_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * w, ggml_tensor * ndst, float neps,
        int8_t * aq, float * as, int nvfp4_kf) {
    if (!ggml_cuda_depth_mix_supported(dst) || dst->src[1]->ne[1] != 1) {
        return false;
    }
    if ((aq == nullptr) != (as == nullptr) || (nvfp4_kf != 0 && (aq == nullptr || (nvfp4_kf != 32 && nvfp4_kf != 64)))) {
        return false;
    }
    if (aq && (dst->src[0]->ne[0] % (nvfp4_kf ? nvfp4_kf : 256) != 0 || dst->src[0]->ne[0] > 32*256 || !ggml_is_contiguous(ndst) ||
            ((uintptr_t) ndst->data & 15) != 0)) {
        return false;
    }
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
        return true;
    }
    cudaStream_t stream = ctx.stream();
    const bool vec = n % 4 == 0 && (s1 | s2 | cur->nb[1] | query->nb[1]) % 16 == 0 &&
        ((uintptr_t) sd | (uintptr_t) cur->data | (uintptr_t) query->data | (uintptr_t) dst->data) % 16 == 0;
    // [T5-GLUE] G4: cache mixed output in SMEM (9.2 KB for n=2304) so NORM reads SMEM instead of 2 global round-trips.
    // Kill switch: GGML_CUDA_DEPTH_MIX_SMEM=0
    static const bool use_smem_env = [] {
        const char * e = getenv("GGML_CUDA_DEPTH_MIX_SMEM");
        return e == nullptr || atoi(e) != 0;
    }();
    const bool use_smem = use_smem_env && n <= 4096;
    const size_t smem_bytes = use_smem ? (size_t) n * sizeof(float) : 0;
    const dim3 grid((unsigned) T, 1);
    if (nvfp4_kf != 0) {
        // [PREFILL-GLUE] opt-in only (GGML_CUDA_PREFILL_FUSE_NORM_ACTQ): same mix + norm, ndst always written, then the
        // NVFP4PF activation quantized from the row this block just wrote
#define DEPTH_MIX_NORM_KF(W_, SMEM_, KF_) \
        k_depth_mix<1, W_, true, SMEM_, KF_><<<grid, DEPTH_MIX_THREADS, SMEM_ ? smem_bytes : 0, stream>>>(sd, s1, s2, k, \
            (const char *) cur->data, cur->nb[1], (const char *) query->data, query->nb[1], (float *) dst->data, n, 1, eps, scale, \
            (const float *) w->data, (float *) ndst->data, neps, aq, as)
        if (vec && use_smem) {
            if (nvfp4_kf == 32) { DEPTH_MIX_NORM_KF(4, true, 32); } else { DEPTH_MIX_NORM_KF(4, true, 64); }
        } else if (vec) {
            if (nvfp4_kf == 32) { DEPTH_MIX_NORM_KF(4, false, 32); } else { DEPTH_MIX_NORM_KF(4, false, 64); }
        } else {
            if (nvfp4_kf == 32) { DEPTH_MIX_NORM_KF(1, false, 32); } else { DEPTH_MIX_NORM_KF(1, false, 64); }
        }
#undef DEPTH_MIX_NORM_KF
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    if (vec) {
        if (use_smem) {
            k_depth_mix<1, 4, true, true><<<grid, DEPTH_MIX_THREADS, smem_bytes, stream>>>(sd, s1, s2, k, (const char *) cur->data, cur->nb[1],
                (const char *) query->data, query->nb[1], (float *) dst->data, n, 1, eps, scale, (const float *) w->data, (float *) ndst->data, neps, aq, as);
        } else {
            k_depth_mix<1, 4, true, false><<<grid, DEPTH_MIX_THREADS, 0, stream>>>(sd, s1, s2, k, (const char *) cur->data, cur->nb[1],
                (const char *) query->data, query->nb[1], (float *) dst->data, n, 1, eps, scale, (const float *) w->data, (float *) ndst->data, neps, aq, as);
        }
    } else {
        k_depth_mix<1, 1, true, false><<<grid, DEPTH_MIX_THREADS, 0, stream>>>(sd, s1, s2, k, (const char *) cur->data, cur->nb[1],
            (const char *) query->data, query->nb[1], (float *) dst->data, n, 1, eps, scale, (const float *) w->data, (float *) ndst->data, neps, aq, as);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
