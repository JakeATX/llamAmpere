// [T5-GLUE] G1: Vectorized SM 8.6 loop-gate kernels (down + apply) for G=1..4 gates.
// Targets Inkling-2-Mini looped layers: n=2304, R=64, T=1..8 (SpecDecode T=5), BF16 weights.
// Kill switch: GGML_CUDA_INK_LG_VEC=0 (default 1 = on).

#include "inklg_vec.cuh"
#include "convert.cuh"

#include <cstdlib>

#define LGV_CHUNKS      8
#define LGV_DOWN_WARPS  4
#define LGV_MAX_T       8
#define LGV_MAX_R       128
#define LGV_APPLY_WARPS 8   // 256 threads = 32 quarter-warps (sub-warps of 8 lanes)

struct ink_lgv_args {
    const char * x[INK_MULTI_MAX];
    const char * m[INK_MULTI_MAX];
    int64_t      x_nb1[INK_MULTI_MAX];
    int64_t      m_nb1[INK_MULTI_MAX];
    const void * down[INK_MULTI_MAX];
    const void * gain[INK_MULTI_MAX];
    const void * forget[INK_MULTI_MAX];
    float      * dst[INK_MULTI_MAX];
    float        eps[INK_MULTI_MAX];
};

bool ggml_cuda_ink_lg_vec_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_INK_LG_VEC");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

template <typename TW>
static __device__ __forceinline__ void lgv_unpack8(uint4 raw, float (&out)[8]);

template <>
__device__ __forceinline__ void lgv_unpack8<nv_bfloat16>(uint4 raw, float (&out)[8]) {
    const nv_bfloat162 b0 = *reinterpret_cast<const nv_bfloat162 *>(&raw.x);
    const nv_bfloat162 b1 = *reinterpret_cast<const nv_bfloat162 *>(&raw.y);
    const nv_bfloat162 b2 = *reinterpret_cast<const nv_bfloat162 *>(&raw.z);
    const nv_bfloat162 b3 = *reinterpret_cast<const nv_bfloat162 *>(&raw.w);
    const float2 f0 = __bfloat1622float2(b0);
    const float2 f1 = __bfloat1622float2(b1);
    const float2 f2 = __bfloat1622float2(b2);
    const float2 f3 = __bfloat1622float2(b3);
    out[0] = f0.x; out[1] = f0.y;
    out[2] = f1.x; out[3] = f1.y;
    out[4] = f2.x; out[5] = f2.y;
    out[6] = f3.x; out[7] = f3.y;
}

template <>
__device__ __forceinline__ void lgv_unpack8<half>(uint4 raw, float (&out)[8]) {
    const half2 h0 = *reinterpret_cast<const half2 *>(&raw.x);
    const half2 h1 = *reinterpret_cast<const half2 *>(&raw.y);
    const half2 h2 = *reinterpret_cast<const half2 *>(&raw.z);
    const half2 h3 = *reinterpret_cast<const half2 *>(&raw.w);
    const float2 f0 = __half22float2(h0);
    const float2 f1 = __half22float2(h1);
    const float2 f2 = __half22float2(h2);
    const float2 f3 = __half22float2(h3);
    out[0] = f0.x; out[1] = f0.y;
    out[2] = f1.x; out[3] = f1.y;
    out[4] = f2.x; out[5] = f2.y;
    out[6] = f3.x; out[7] = f3.y;
}

// Vectorized down projection: 128-bit uint4 loads of 8 BF16/F16 weights + 2x float4 loads of 8 F32 activations.
// Requires n % 8 == 0, 16-byte aligned rows.
template <typename TW, int RB, int T_CAP>
static __global__ void __launch_bounds__(LGV_DOWN_WARPS * WARP_SIZE) k_ink_lg_down_vec(
        const ink_lgv_args a, float * __restrict__ part_all, const int n, const int R, const int T) {
    const int g = blockIdx.z;
    const char * __restrict__ x = a.x[g];
    const char * __restrict__ m = a.m[g];
    const int64_t x_nb1 = a.x_nb1[g];
    const int64_t m_nb1 = a.m_nb1[g];
    const TW * __restrict__ down = (const TW *) a.down[g];
    float * __restrict__ part = part_all + (int64_t) g * T * 2 * LGV_CHUNKS * (R + 1);

    const int r0   = blockIdx.x * RB;
    const int c    = blockIdx.y * LGV_DOWN_WARPS + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int half = c / LGV_CHUNKS;

    const int nv  = n / 8;
    const int clv = (nv + LGV_CHUNKS - 1) / LGV_CHUNKS;
    const int v0  = (c % LGV_CHUNKS) * clv;
    const int v1  = min(nv, v0 + clv);

    const char *  src = half ? m : x;
    const int64_t nb1 = half ? m_nb1 : x_nb1;
    const TW *    w   = down + (int64_t) r0 * 2 * n + (int64_t) half * n;

    float d[RB][T_CAP];
    float ss[T_CAP];
#pragma unroll
    for (int t = 0; t < T_CAP; ++t) {
        ss[t] = 0.0f;
#pragma unroll
        for (int b = 0; b < RB; ++b) {
            d[b][t] = 0.0f;
        }
    }

    for (int v = v0 + lane; v < v1; v += WARP_SIZE) {
        float wi[RB][8];
#pragma unroll
        for (int b = 0; b < RB; ++b) {
            if (r0 + b < R) {
                const uint4 raw = *reinterpret_cast<const uint4 *>(w + (int64_t) b * 2 * n + v * 8);
                lgv_unpack8<TW>(raw, wi[b]);
            } else {
#pragma unroll
                for (int u = 0; u < 8; ++u) { wi[b][u] = 0.0f; }
            }
        }
#pragma unroll
        for (int t = 0; t < T_CAP; ++t) {
            if (t < T) {
                const float * sp = (const float *) (src + (int64_t) t * nb1) + v * 8;
                const float4 x0 = *reinterpret_cast<const float4 *>(sp);
                const float4 x1 = *reinterpret_cast<const float4 *>(sp + 4);
                const float xv[8] = { x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w };
                if (r0 == 0) {
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        ss[t] = fmaf(xv[u], xv[u], ss[t]);
                    }
                }
#pragma unroll
                for (int b = 0; b < RB; ++b) {
#pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        d[b][t] = fmaf(wi[b][u], xv[u], d[b][t]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < T_CAP; ++t) {
        if (t < T) {
            if (r0 == 0) {
                ss[t] = warp_reduce_sum(ss[t]);
            }
            float * p = part + ((int64_t) t * 2 * LGV_CHUNKS + c) * (R + 1);
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

// Vectorized apply kernel:
// - 8 warps = 256 threads = 32 quarter-warps (sub-warps of 8 lanes).
// - Each quarter-warp processes 1 row i per tile (32 consecutive rows per tile x RT tiles per block).
// - Each lane loads 1 uint4 (8 BF16/F16 weights = 16 B) per 64-rank tile: for R=64, exactly 1 LDG.E.128 per matrix!
// - 3-step sub-warp reduction (xor 4, 2, 1) -> SMEM staging -> 100% coalesced reads of x/m and writes of dst!
template <typename TW, int T_CAP, int RT>
static __global__ void __launch_bounds__(LGV_APPLY_WARPS * WARP_SIZE) k_ink_lg_apply_vec(
        const ink_lgv_args a, const float * __restrict__ part_all, const int n, const int R, const int T) {
    const int g = blockIdx.y;
    const char * __restrict__ x = a.x[g];
    const char * __restrict__ m = a.m[g];
    const int64_t x_nb1 = a.x_nb1[g];
    const int64_t m_nb1 = a.m_nb1[g];
    const TW * __restrict__ gain   = (const TW *) a.gain[g];
    const TW * __restrict__ forget = (const TW *) a.forget[g];
    const float * __restrict__ part = part_all + (int64_t) g * T * 2 * LGV_CHUNKS * (R + 1);
    float * __restrict__ dst = a.dst[g];
    const float eps = a.eps[g];

    __shared__ float gs[LGV_MAX_T * LGV_MAX_R];
    __shared__ float rs[2 * LGV_MAX_T];
    // SMEM staging for 32 consecutive rows x T_CAP tokens (+1 padding to avoid bank conflicts)
    __shared__ float s_ag[T_CAP][33];
    __shared__ float s_af[T_CAP][33];

    for (int j = threadIdx.x; j < 2 * T; j += blockDim.x) {
        const float * p = part + ((int64_t) (j / 2) * 2 * LGV_CHUNKS + (j % 2) * LGV_CHUNKS) * (R + 1);
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < LGV_CHUNKS; ++c) {
            s += p[c * (R + 1) + R];
        }
        rs[j] = rsqrtf(s / n + eps);
    }
    __syncthreads();

    for (int j = threadIdx.x; j < T * R; j += blockDim.x) {
        const int t = j / R;
        const int r = j % R;
        const float * p = part + (int64_t) t * 2 * LGV_CHUNKS * (R + 1) + r;
        float gx = 0.0f;
        float gm = 0.0f;
#pragma unroll
        for (int c = 0; c < LGV_CHUNKS; ++c) {
            gx += p[c * (R + 1)];
            gm += p[(LGV_CHUNKS + c) * (R + 1)];
        }
        gs[t * R + r] = rs[2 * t] * gx + rs[2 * t + 1] * gm;
    }
    __syncthreads();

    const int subwarp = threadIdx.x >> 3;   // 0..31 (32 consecutive output rows per tile)
    const int sublane = threadIdx.x & 7;    // 0..7  (each lane owns 8 consecutive rank dims -> 64 per iteration)
    const int warp_id = threadIdx.x >> 5;   // 0..7  (used for coalesced token-parallel epilogue store)
    const int lane_id = threadIdx.x & 31;   // 0..31 (maps directly to 32 consecutive channels!)

    const int base_row = blockIdx.x * (32 * RT);

#pragma unroll
    for (int tile = 0; tile < RT; ++tile) {
        const int row_tile = base_row + tile * 32;
        const int i = row_tile + subwarp;

        float ag[T_CAP];
        float af[T_CAP];
#pragma unroll
        for (int t = 0; t < T_CAP; ++t) {
            ag[t] = 0.0f;
            af[t] = 0.0f;
        }

        if (i < n) {
            const TW * wg = gain   + (int64_t) i * R;
            const TW * wf = forget + (int64_t) i * R;
            for (int r = sublane * 8; r < R; r += 64) {
                const uint4 raw_g = *reinterpret_cast<const uint4 *>(wg + r);
                const uint4 raw_f = *reinterpret_cast<const uint4 *>(wf + r);
                float fg[8], ff[8];
                lgv_unpack8<TW>(raw_g, fg);
                lgv_unpack8<TW>(raw_f, ff);
#pragma unroll
                for (int t = 0; t < T_CAP; ++t) {
                    if (t < T) {
                        const float4 g0 = *reinterpret_cast<const float4 *>(&gs[t * R + r]);
                        const float4 g1 = *reinterpret_cast<const float4 *>(&gs[t * R + r + 4]);
                        ag[t] = fmaf(fg[0], g0.x, ag[t]); af[t] = fmaf(ff[0], g0.x, af[t]);
                        ag[t] = fmaf(fg[1], g0.y, ag[t]); af[t] = fmaf(ff[1], g0.y, af[t]);
                        ag[t] = fmaf(fg[2], g0.z, ag[t]); af[t] = fmaf(ff[2], g0.z, af[t]);
                        ag[t] = fmaf(fg[3], g0.w, ag[t]); af[t] = fmaf(ff[3], g0.w, af[t]);
                        ag[t] = fmaf(fg[4], g1.x, ag[t]); af[t] = fmaf(ff[4], g1.x, af[t]);
                        ag[t] = fmaf(fg[5], g1.y, ag[t]); af[t] = fmaf(ff[5], g1.y, af[t]);
                        ag[t] = fmaf(fg[6], g1.z, ag[t]); af[t] = fmaf(ff[6], g1.z, af[t]);
                        ag[t] = fmaf(fg[7], g1.w, ag[t]); af[t] = fmaf(ff[7], g1.w, af[t]);
                    }
                }
            }
        }

        // 3-step quarter-warp butterfly reduction (width = 8)
#pragma unroll
        for (int t = 0; t < T_CAP; ++t) {
            if (t < T) {
                ag[t] += __shfl_xor_sync(0xffffffff, ag[t], 4, 8);
                ag[t] += __shfl_xor_sync(0xffffffff, ag[t], 2, 8);
                ag[t] += __shfl_xor_sync(0xffffffff, ag[t], 1, 8);
                af[t] += __shfl_xor_sync(0xffffffff, af[t], 4, 8);
                af[t] += __shfl_xor_sync(0xffffffff, af[t], 2, 8);
                af[t] += __shfl_xor_sync(0xffffffff, af[t], 1, 8);
                if (sublane == 0) {
                    s_ag[t][subwarp] = ag[t];
                    s_af[t][subwarp] = af[t];
                }
            }
        }
        __syncthreads();

        // 100% coalesced epilogue: warp w handles token t = w across 32 consecutive channels row_tile + lane_id!
        if (warp_id < T) {
            const int col = row_tile + lane_id;
            if (col < n) {
                const float xi = ((const float *) (x + (int64_t) warp_id * x_nb1))[col];
                const float mi = ((const float *) (m + (int64_t) warp_id * m_nb1))[col];
                const float fv = s_af[warp_id][lane_id];
                const float gv = s_ag[warp_id][lane_id];
                dst[(int64_t) warp_id * n + col] = (xi - tanhf(fv) * xi) + tanhf(gv) * mi;
            }
        }
        if (tile + 1 < RT) {
            __syncthreads();
        }
    }
}

template <typename TW, int T_CAP>
static void ink_lgv_launch_t(ggml_backend_cuda_context & ctx, const ink_lgv_args & a, int G, int n, int R, int T) {
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) G * T * 2 * LGV_CHUNKS * (R + 1));
    cudaStream_t stream = ctx.stream();
    // For G=1 (attn loop_gate): RB=2 -> 32*4=128 blocks; for G>=2 (QKV): RB=4 -> 16*4*3=192 blocks (saturates 84 SMs)
    if (G == 1) {
        k_ink_lg_down_vec<TW, 2, T_CAP><<<dim3((R + 1) / 2, 2 * LGV_CHUNKS / LGV_DOWN_WARPS, 1), LGV_DOWN_WARPS * WARP_SIZE, 0, stream>>>(
            a, part.get(), n, R, T);
        // RT=1 -> 72 blocks on 84 SMs for n=2304 (1 wave!)
        const int rpb = 32;
        k_ink_lg_apply_vec<TW, T_CAP, 1><<<dim3((n + rpb - 1) / rpb, 1, 1), LGV_APPLY_WARPS * WARP_SIZE, 0, stream>>>(
            a, part.get(), n, R, T);
    } else {
        k_ink_lg_down_vec<TW, 4, T_CAP><<<dim3((R + 3) / 4, 2 * LGV_CHUNKS / LGV_DOWN_WARPS, G), LGV_DOWN_WARPS * WARP_SIZE, 0, stream>>>(
            a, part.get(), n, R, T);
        // RT=2 -> 36*3 = 108 blocks on 84 SMs for n=2304, G=3
        const int rpb = 64;
        k_ink_lg_apply_vec<TW, T_CAP, 2><<<dim3((n + rpb - 1) / rpb, G, 1), LGV_APPLY_WARPS * WARP_SIZE, 0, stream>>>(
            a, part.get(), n, R, T);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename TW>
static void ink_lgv_launch(ggml_backend_cuda_context & ctx, const ink_lgv_args & a, int G, int n, int R, int T) {
    if (T == 1) {
        ink_lgv_launch_t<TW, 1>(ctx, a, G, n, R, T);
    } else if (T <= 5) {
        ink_lgv_launch_t<TW, 5>(ctx, a, G, n, R, T);
    } else {
        ink_lgv_launch_t<TW, LGV_MAX_T>(ctx, a, G, n, R, T);
    }
}

bool ggml_cuda_ink_loop_gate_vec(ggml_backend_cuda_context & ctx, ggml_tensor * const * gates, int G) {
    if (!ggml_cuda_ink_lg_vec_enabled() || G < 1 || G > INK_MULTI_MAX) {
        return false;
    }
    const ggml_tensor * d0 = gates[0];
    const int n = (int) d0->src[0]->ne[0];
    const int T = (int) d0->src[0]->ne[1];
    const int R = (int) d0->src[2]->ne[1];
    const ggml_type wt = d0->src[2]->type;
    if (T < 1 || T > LGV_MAX_T || R < 8 || R > LGV_MAX_R || (R & 7) != 0 || (n & 7) != 0) {
        return false;
    }
    if (wt != GGML_TYPE_BF16 && wt != GGML_TYPE_F16) {
        return false;
    }
    ink_lgv_args a{};
    for (int k = 0; k < G; ++k) {
        const ggml_tensor * d = gates[k];
        if (d->op != GGML_OP_LOOP_GATE || d->src[0]->ne[0] != n || d->src[0]->ne[1] != T || d->src[2]->ne[1] != R ||
                d->src[2]->type != wt || d->src[3]->type != wt || d->src[4]->type != wt ||
                d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d)) {
            return false;
        }
        // Check 16-byte alignment for uint4 / float4 vectorized loads
        if (((uintptr_t) d->src[0]->data & 15) != 0 || ((uintptr_t) d->src[1]->data & 15) != 0 ||
            ((uintptr_t) d->src[2]->data & 15) != 0 || ((uintptr_t) d->src[3]->data & 15) != 0 ||
            ((uintptr_t) d->src[4]->data & 15) != 0 ||
            (d->src[0]->nb[1] & 15) != 0 || (d->src[1]->nb[1] & 15) != 0) {
            return false;
        }
        a.x[k] = (const char *) d->src[0]->data;  a.x_nb1[k] = d->src[0]->nb[1];
        a.m[k] = (const char *) d->src[1]->data;  a.m_nb1[k] = d->src[1]->nb[1];
        a.down[k] = d->src[2]->data; a.gain[k] = d->src[3]->data; a.forget[k] = d->src[4]->data;
        a.dst[k] = (float *) d->data;
        a.eps[k] = ggml_get_op_params_f32(d, 0);
    }
    if (wt == GGML_TYPE_BF16) {
        ink_lgv_launch<nv_bfloat16>(ctx, a, G, n, R, T);
    } else {
        ink_lgv_launch<half>(ctx, a, G, n, R, T);
    }
    return true;
}
