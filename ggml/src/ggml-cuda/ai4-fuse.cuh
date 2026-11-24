#pragma once
// AmperInteger (AI+) drop-in activation & dense-FFN fusion kernels (SM 8.0+ / Ampere SM 8.6):
//   1. k_rms_norm_mul_quantize_act_ai4: RMSNorm(x, eps) * gamma -> H256/H128 -> int8 (optional f32 write)
//   2. k_swiglu_quantize_act_ai4:       GLU(gate, up)           -> H256/H128 -> int8 (optional f32 write)
//   3. k_mmv_grp_spec:                  Grouped Q/K/V SpecDecode (N=9..32, BN=16/32) for ALL 11 AI+ unit types
//   4. k_dense_gate_up_glu:             Fused dense Gate + Up + GLU (N=1..32, BN=8/16/32) for ALL 11 AI+ unit types
//   5. k_dense_gate_up_glu_q8:          Fused dense Gate + Up + GLU + H256/H128 + int8 bridge (N=1..32)
//
// Header has no ggml-cuda backend context dependencies so standalone nvcc SASS audit tools and
// ggml/src/ggml-cuda/ai4-fuse.cu compile the exact same device code.

#include "ai4-kern.cuh"

namespace ai4fuse {

// Matches ai4_prep_variant in ai4.cu
constexpr int PREP_PLAIN  = 0;
constexpr int PREP_SWAP8  = 1;
constexpr int PREP_XPERM  = 2;
constexpr int PREP_LINEAR = 3;

// Lightweight GLU op tag (no ggml.h dependency required in standalone SASS audit)
constexpr int GLU_SWIGLU       = 0;
constexpr int GLU_GEGLU        = 1;
constexpr int GLU_SWIGLU_OAI   = 2;
constexpr int GLU_SWIGLU_CLAMP = 3;

template <int GLU_OP>
static __device__ __forceinline__ float eval_glu(float gate, float up, float alpha, float limit) {
    if constexpr (GLU_OP == GLU_SWIGLU) {
        return up * (gate / (1.0f + expf(-gate)));
    } else if constexpr (GLU_OP == GLU_GEGLU) {
        constexpr float GELU_COEF_A    = 0.044715f;
        constexpr float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        return up * (0.5f * gate * (1.0f + tanhf(SQRT_2_OVER_PI * gate * (1.0f + GELU_COEF_A * gate * gate))));
    } else if constexpr (GLU_OP == GLU_SWIGLU_OAI) {
        const float x = fminf(gate, limit);
        const float g = fmaxf(fminf(up, limit), -limit);
        return (x / (1.0f + expf(-x * alpha))) * (1.0f + g);
    } else {
        // GLU_SWIGLU_CLAMP
        const float x = fminf(gate, limit);
        const float g = fmaxf(fminf(up, limit), -limit);
        return g * (x / (1.0f + expf(-x)));
    }
}

static __device__ __forceinline__ float eval_glu_rt(int glu_op, float gate, float up, float alpha, float limit) {
    if (glu_op == GLU_SWIGLU) {
        return eval_glu<GLU_SWIGLU>(gate, up, alpha, limit);
    } else if (glu_op == GLU_GEGLU) {
        return eval_glu<GLU_GEGLU>(gate, up, alpha, limit);
    } else if (glu_op == GLU_SWIGLU_OAI) {
        return eval_glu<GLU_SWIGLU_OAI>(gate, up, alpha, limit);
    } else {
        return eval_glu<GLU_SWIGLU_CLAMP>(gate, up, alpha, limit);
    }
}

// -------------------------------------------------------------------------------------------------
// Shared in-register Hadamard + int8 quantize helpers (bit-exact with rq4::quantize_act_body and
// ai4k::quantize_act128_body).
// -------------------------------------------------------------------------------------------------

template <int VARIANT>
static __device__ __forceinline__ void rotate_quantize_256(
        float (&v)[8], int8_t * __restrict__ q, float * __restrict__ s, int K, int hb, int tok) {
    const int lane = threadIdx.x & 31;
    const uint32_t sw = rq4::d_sign_bits[lane >> 2] >> ((lane & 3) * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if ((sw >> e) & 1) {
            v[e] = -v[e];
        }
    }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; i += 2 * h) {
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (lane & m) ? (o - v[e]) : (v[e] + o);
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= (1.0f / 16.0f);
        amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    uint32_t w[2] = { 0u, 0u };
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int qi = __float2int_rn(v[e] * id);
        w[e >> 2] |= ((uint32_t) (qi & 0xff)) << ((e & 3) * 8);
    }
    const int c  = lane >> 4;
    const int j  = (lane & 15) >> 2;
    const int h  = (lane & 3) >> 1;
    const int t0 = (lane & 1) * 2;
    if constexpr (VARIANT == PREP_LINEAR) {
        *(uint2 *) (q + (int64_t) tok * K + hb * RQ4_HB + lane * 8) = make_uint2(w[0], w[1]);
    } else if constexpr (VARIANT == PREP_XPERM) {
        int8_t * qr = q + (int64_t) tok * K + hb * RQ4_HB + c * RQ4_CHUNK + h * 64 + (t0 >> 1) * 32 + j * 8;
        *(uint2 *) qr = make_uint2(w[0], w[1]);
    } else {
        constexpr bool SWAP8 = (VARIANT == PREP_SWAP8);
        int8_t * qr = q + (int64_t) tok * K + hb * RQ4_HB + c * RQ4_CHUNK +
                      (j ^ (SWAP8 ? ((tok >> 1) & 1) : 0)) * 8 + h * 4;
        *(uint32_t *) (qr + (t0 + 0) * 32) = w[0];
        *(uint32_t *) (qr + (t0 + 1) * 32) = w[1];
    }
    if (lane == 0) {
        s[(int64_t) tok * (K / RQ4_HB) + hb] = d;
    }
}

template <int VARIANT>
static __device__ __forceinline__ void rotate_quantize_128(
        float (&v)[8], int8_t * __restrict__ q, float * __restrict__ s, int K, int cb, int tok, bool live) {
    const int lane = threadIdx.x & 31;
    const int l    = lane & 15;
    const uint32_t sw = rq4::d_sign_bits[l >> 2] >> ((l & 3) * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if ((sw >> e) & 1) {
            v[e] = -v[e];
        }
    }
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; i += 2 * h) {
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int m = 1; m < 16; m <<= 1) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (l & m) ? (o - v[e]) : (v[e] + o);
        }
    }
    float amax = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        v[e] *= 0.08838834764831845f; // 1 / sqrt(128)
        amax = fmaxf(amax, fabsf(v[e]));
    }
#pragma unroll
    for (int m = 8; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    }
    if (!live) {
        return;
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    uint32_t w[2] = { 0u, 0u };
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int qi = __float2int_rn(v[e] * id);
        w[e >> 2] |= ((uint32_t) (qi & 0xff)) << ((e & 3) * 8);
    }
    const int j  = l >> 2;
    const int h  = (l & 3) >> 1;
    const int t0 = (l & 1) * 2;
    if constexpr (VARIANT == PREP_LINEAR) {
        *(uint2 *) (q + (int64_t) tok * K + cb * 128 + l * 8) = make_uint2(w[0], w[1]);
    } else if constexpr (VARIANT == PREP_XPERM) {
        int8_t * qr = q + (int64_t) tok * K + cb * 128 + h * 64 + (t0 >> 1) * 32 + j * 8;
        *(uint2 *) qr = make_uint2(w[0], w[1]);
    } else {
        constexpr bool SWAP8 = (VARIANT == PREP_SWAP8);
        int8_t * qr = q + (int64_t) tok * K + cb * 128 +
                      (j ^ (SWAP8 ? ((tok >> 1) & 1) : 0)) * 8 + h * 4;
        *(uint32_t *) (qr + (t0 + 0) * 32) = w[0];
        *(uint32_t *) (qr + (t0 + 1) * 32) = w[1];
    }
    if (l == 0) {
        s[(int64_t) tok * (K / 128) + cb] = d;
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 1: Fused RMSNorm(x, eps) * gamma -> Hadamard H256/H128 -> int8 quantize
// One CTA per token (grid-stride if N > 65535). REG_CACHE=1 (K <= BLOCK_WARPS*256) or
// REG_CACHE=2 (K <= BLOCK_WARPS*512) holds x in registers -> single HBM read of x!
// REG_CACHE=0 streams pass 2 from L1/L2 for K > 8192 (e.g. K = 17408).
// -------------------------------------------------------------------------------------------------

template <int XSH, int VARIANT, bool WRITE_F32, int BLOCK_WARPS = 16, int REG_CACHE = 2>
static __global__ void __launch_bounds__(BLOCK_WARPS * 32) k_rms_norm_mul_quantize_act_ai4(
        const float * __restrict__ x, int64_t sx,
        const float * __restrict__ gamma,
        float       * __restrict__ dst_f32,
        int8_t      * __restrict__ q,
        float       * __restrict__ s,
        int K, int N, float eps) {
    static_assert(BLOCK_WARPS == 8 || BLOCK_WARPS == 16, "BLOCK_WARPS");
    static_assert(REG_CACHE >= 0 && REG_CACHE <= 2, "REG_CACHE");
    __shared__ float s_sum[BLOCK_WARPS];
    __shared__ float s_inv_rms;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwb  = (K + 255) >> 8; // 256-float warp-blocks per token

    for (int tok = blockIdx.x; tok < N; tok += gridDim.x) {
        const float * xrow = x + (int64_t) tok * sx;
        float sum_sq = 0.0f;

        if constexpr (REG_CACHE > 0) {
            float vx[REG_CACHE][8];
#pragma unroll
            for (int it = 0; it < REG_CACHE; ++it) {
                const int wb = warp + it * BLOCK_WARPS;
                const int off = wb * 256 + lane * 8;
                const bool live = off < K;
                if (live) {
                    const float4 u0 = *(const float4 *) (xrow + off);
                    const float4 u1 = *(const float4 *) (xrow + off + 4);
                    vx[it][0] = u0.x; vx[it][1] = u0.y; vx[it][2] = u0.z; vx[it][3] = u0.w;
                    vx[it][4] = u1.x; vx[it][5] = u1.y; vx[it][6] = u1.z; vx[it][7] = u1.w;
                    sum_sq += u0.x * u0.x + u0.y * u0.y + u0.z * u0.z + u0.w * u0.w
                            + u1.x * u1.x + u1.y * u1.y + u1.z * u1.z + u1.w * u1.w;
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { vx[it][e] = 0.0f; }
                }
            }
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, m);
            }
            if (lane == 0) {
                s_sum[warp] = sum_sq;
            }
            __syncthreads();
            if (warp == 0) {
                float t = lane < BLOCK_WARPS ? s_sum[lane] : 0.0f;
#pragma unroll
                for (int m = BLOCK_WARPS >> 1; m > 0; m >>= 1) {
                    t += __shfl_xor_sync(0xffffffffu, t, m);
                }
                if (lane == 0) {
                    s_inv_rms = rsqrtf(t / (float) K + eps);
                }
            }
            __syncthreads();
            const float inv_rms = s_inv_rms;

#pragma unroll
            for (int it = 0; it < REG_CACHE; ++it) {
                const int wb = warp + it * BLOCK_WARPS;
                if (wb >= nwb) {
                    break;
                }
                const int off = wb * 256 + lane * 8;
                const bool live = XSH ? true : (off < K);
                float v[8];
                if (live) {
                    const float4 g0 = *(const float4 *) (gamma + off);
                    const float4 g1 = *(const float4 *) (gamma + off + 4);
                    v[0] = (inv_rms * vx[it][0]) * g0.x;
                    v[1] = (inv_rms * vx[it][1]) * g0.y;
                    v[2] = (inv_rms * vx[it][2]) * g0.z;
                    v[3] = (inv_rms * vx[it][3]) * g0.w;
                    v[4] = (inv_rms * vx[it][4]) * g1.x;
                    v[5] = (inv_rms * vx[it][5]) * g1.y;
                    v[6] = (inv_rms * vx[it][6]) * g1.z;
                    v[7] = (inv_rms * vx[it][7]) * g1.w;
                    if constexpr (WRITE_F32) {
                        float * dr = dst_f32 + (int64_t) tok * K + off;
                        *(float4 *) dr       = make_float4(v[0], v[1], v[2], v[3]);
                        *(float4 *) (dr + 4) = make_float4(v[4], v[5], v[6], v[7]);
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { v[e] = 0.0f; }
                }
                if constexpr (XSH == 1) {
                    rotate_quantize_256<VARIANT>(v, q, s, K, wb, tok);
                } else {
                    rotate_quantize_128<VARIANT>(v, q, s, K, wb * 2 + (lane >> 4), tok, live);
                }
            }
        } else {
            // Streaming pass for large K (> BLOCK_WARPS * 512, e.g. K = 17408)
            for (int wb = warp; wb < nwb; wb += BLOCK_WARPS) {
                const int off = wb * 256 + lane * 8;
                if (XSH || off < K) {
                    const float4 u0 = *(const float4 *) (xrow + off);
                    const float4 u1 = *(const float4 *) (xrow + off + 4);
                    sum_sq += u0.x * u0.x + u0.y * u0.y + u0.z * u0.z + u0.w * u0.w
                            + u1.x * u1.x + u1.y * u1.y + u1.z * u1.z + u1.w * u1.w;
                }
            }
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) {
                sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, m);
            }
            if (lane == 0) {
                s_sum[warp] = sum_sq;
            }
            __syncthreads();
            if (warp == 0) {
                float t = lane < BLOCK_WARPS ? s_sum[lane] : 0.0f;
#pragma unroll
                for (int m = BLOCK_WARPS >> 1; m > 0; m >>= 1) {
                    t += __shfl_xor_sync(0xffffffffu, t, m);
                }
                if (lane == 0) {
                    s_inv_rms = rsqrtf(t / (float) K + eps);
                }
            }
            __syncthreads();
            const float inv_rms = s_inv_rms;

            for (int wb = warp; wb < nwb; wb += BLOCK_WARPS) {
                const int off = wb * 256 + lane * 8;
                const bool live = XSH ? true : (off < K);
                float v[8];
                if (live) {
                    const float4 u0 = *(const float4 *) (xrow + off);
                    const float4 u1 = *(const float4 *) (xrow + off + 4);
                    const float4 g0 = *(const float4 *) (gamma + off);
                    const float4 g1 = *(const float4 *) (gamma + off + 4);
                    v[0] = (inv_rms * u0.x) * g0.x;
                    v[1] = (inv_rms * u0.y) * g0.y;
                    v[2] = (inv_rms * u0.z) * g0.z;
                    v[3] = (inv_rms * u0.w) * g0.w;
                    v[4] = (inv_rms * u1.x) * g1.x;
                    v[5] = (inv_rms * u1.y) * g1.y;
                    v[6] = (inv_rms * u1.z) * g1.z;
                    v[7] = (inv_rms * u1.w) * g1.w;
                    if constexpr (WRITE_F32) {
                        float * dr = dst_f32 + (int64_t) tok * K + off;
                        *(float4 *) dr       = make_float4(v[0], v[1], v[2], v[3]);
                        *(float4 *) (dr + 4) = make_float4(v[4], v[5], v[6], v[7]);
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { v[e] = 0.0f; }
                }
                if constexpr (XSH == 1) {
                    rotate_quantize_256<VARIANT>(v, q, s, K, wb, tok);
                } else {
                    rotate_quantize_128<VARIANT>(v, q, s, K, wb * 2 + (lane >> 4), tok, live);
                }
            }
        }
        if (tok + (int) gridDim.x < N) {
            __syncthreads();
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 2: Fused SwiGLU(gate, up) -> Hadamard H256/H128 -> int8 quantize (bit-exact with unary.cu
// + rq4::quantize_act / ai4k::quantize_act128). Works for both dense FFN (MUL_MAT) and MoE experts
// (MUL_MAT_ID where N = n_used * n_tok), and for both split-tensor and two-tensor GLU inputs.
// -------------------------------------------------------------------------------------------------

template <int XSH, int VARIANT, bool WRITE_F32, int GLU_OP = GLU_SWIGLU>
static __global__ void __launch_bounds__(128) k_swiglu_quantize_act_ai4(
        const float * __restrict__ gate, int64_t s_gate,
        const float * __restrict__ up,   int64_t s_up,
        float       * __restrict__ dst_f32,
        int8_t      * __restrict__ q,
        float       * __restrict__ s,
        int K, int N, float alpha, float limit) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    for (int tok = blockIdx.y; tok < N; tok += gridDim.y) {
        if constexpr (XSH == 1) {
            const int hb = blockIdx.x * 4 + warp;
            if (hb * RQ4_HB >= K) {
                return;
            }
            const int off = hb * RQ4_HB + lane * 8;
            const float * gr = gate + (int64_t) tok * s_gate + off;
            const float * ur = up   + (int64_t) tok * s_up   + off;
            const float4 g0 = *(const float4 *) gr;
            const float4 g1 = *(const float4 *) (gr + 4);
            const float4 u0 = *(const float4 *) ur;
            const float4 u1 = *(const float4 *) (ur + 4);
            float v[8] = {
                eval_glu<GLU_OP>(g0.x, u0.x, alpha, limit),
                eval_glu<GLU_OP>(g0.y, u0.y, alpha, limit),
                eval_glu<GLU_OP>(g0.z, u0.z, alpha, limit),
                eval_glu<GLU_OP>(g0.w, u0.w, alpha, limit),
                eval_glu<GLU_OP>(g1.x, u1.x, alpha, limit),
                eval_glu<GLU_OP>(g1.y, u1.y, alpha, limit),
                eval_glu<GLU_OP>(g1.z, u1.z, alpha, limit),
                eval_glu<GLU_OP>(g1.w, u1.w, alpha, limit),
            };
            if constexpr (WRITE_F32) {
                float * dr = dst_f32 + (int64_t) tok * K + off;
                *(float4 *) dr       = make_float4(v[0], v[1], v[2], v[3]);
                *(float4 *) (dr + 4) = make_float4(v[4], v[5], v[6], v[7]);
            }
            rotate_quantize_256<VARIANT>(v, q, s, K, hb, tok);
        } else {
            const int l  = lane & 15;
            const int cb = (blockIdx.x * 4 + warp) * 2 + (lane >> 4);
            const bool live = cb * 128 < K;
            float v[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            if (live) {
                const int off = cb * 128 + l * 8;
                const float * gr = gate + (int64_t) tok * s_gate + off;
                const float * ur = up   + (int64_t) tok * s_up   + off;
                const float4 g0 = *(const float4 *) gr;
                const float4 g1 = *(const float4 *) (gr + 4);
                const float4 u0 = *(const float4 *) ur;
                const float4 u1 = *(const float4 *) (ur + 4);
                v[0] = eval_glu<GLU_OP>(g0.x, u0.x, alpha, limit);
                v[1] = eval_glu<GLU_OP>(g0.y, u0.y, alpha, limit);
                v[2] = eval_glu<GLU_OP>(g0.z, u0.z, alpha, limit);
                v[3] = eval_glu<GLU_OP>(g0.w, u0.w, alpha, limit);
                v[4] = eval_glu<GLU_OP>(g1.x, u1.x, alpha, limit);
                v[5] = eval_glu<GLU_OP>(g1.y, u1.y, alpha, limit);
                v[6] = eval_glu<GLU_OP>(g1.z, u1.z, alpha, limit);
                v[7] = eval_glu<GLU_OP>(g1.w, u1.w, alpha, limit);
                if constexpr (WRITE_F32) {
                    float * dr = dst_f32 + (int64_t) tok * K + off;
                    *(float4 *) dr       = make_float4(v[0], v[1], v[2], v[3]);
                    *(float4 *) (dr + 4) = make_float4(v[4], v[5], v[6], v[7]);
                }
            }
            rotate_quantize_128<VARIANT>(v, q, s, K, cb, tok, live);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 3: Grouped Q/K/V SpecDecode (N = 9..32, BN = 16 or 32) across ALL 11 AI+ unit layouts.
// Zero static SMEM (except AIT2P_U 2 KB pair_tab_s): warp reduction aliases dead cp.async smem.
// J_MAJOR 4-register A-fragment streaming keeps register count low (~70-88 regs, 0 spills).
// -------------------------------------------------------------------------------------------------

template <int BN, int LAY>
constexpr __host__ __device__ int grp_spec_stage_bytes() {
    return ai4k::ch_bytes<LAY>() + 64 + BN * 4;
}

template <int BN, int LAY, int WARPS>
constexpr __host__ __device__ int grp_spec_smem_bytes() {
    constexpr int stage_smem = WARPS * 2 * grp_spec_stage_bytes<BN, LAY>();
    constexpr int red_smem   = (WARPS - 1) * (BN / 8) * 4 * 32 * (int) sizeof(float);
    return stage_smem > red_smem ? stage_smem : red_smem;
}

template <int BN, int LAY, int XSH, int WARPS = 4>
static __global__ void __launch_bounds__(WARPS * 32, BN <= 16 ? 4 : 3) k_mmv_grp_spec(
        const ai4k::mmv_group grp,
        const int8_t * __restrict__ Xq,
        const float  * __restrict__ Xs,
        int N, int K) {
    static_assert(BN == 16 || BN == 32, "grp_spec BN");
    static_assert(WARPS == 4, "grp_spec WARPS");
    constexpr int NTI = BN / 8;
    constexpr int CHB = ai4k::ch_bytes<LAY>();
    constexpr int ST  = grp_spec_stage_bytes<BN, LAY>();
    extern __shared__ __align__(16) uint8_t smem[];

    const int b = blockIdx.x;
    int mem = 0;
#pragma unroll
    for (int j = 1; j < ai4k::MMV_GRP_MAX; ++j) {
        if (j < grp.n && b >= grp.s0[j]) {
            mem = j;
        }
    }
    const uint8_t * W = grp.W[mem];
    float         * Y = grp.Y[mem];
    const int       M = grp.M[mem];
    const int    slab = b - grp.s0[mem];
    const int      n0 = blockIdx.y * BN;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    if constexpr (ai4k::lay_pair<LAY>()) {
        ai4k::pair_tab_fill(threadIdx.x, WARPS * 32);
        __syncthreads();
    }

    const ai4k::k5win wa = ai4k::lay_win<LAY>(g, t & 1);
    const ai4k::k5win wb = ai4k::lay_win<LAY>(g + 8, t & 1);
    const int nk  = K / 128;
    const int xsn = K >> (7 + XSH);
    const uint8_t * ws = W + (int64_t) slab * ai4k::slab_bytes_l<LAY>(K);
    uint8_t * ring = smem + warp * 2 * ST;
    const int nmy  = (nk - warp + WARPS - 1) / WARPS;

    auto load = [&](int i, int slot) {
        const int kc = warp + i * WARPS;
        uint8_t * st = ring + slot * ST;
        const uint8_t * tiles = ws + ai4k::chunk_off<LAY>(kc);
        constexpr int CHP = CHB / 16;
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                ai4k::cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        if (lane < 8) {
            ai4k::cp_async4(st + CHB + lane * 4, ws + ai4k::sc_off<LAY>(kc) + lane * 4, true);
        }
        if (lane < BN) {
            const bool live = n0 + lane < N;
            const float * src = live ? Xs + (int64_t) (n0 + lane) * xsn + (kc >> XSH) : Xs;
            ai4k::cp_async4(st + CHB + 64 + lane * 4, src, live);
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }

    if (nmy > 0) {
        load(0, 0);
    }
    ai4k::cp_async_commit();

    for (int i = 0; i < nmy; ++i) {
        const int slot = i & 1;
        ai4k::cp_async_wait<0>();
        __syncwarp();
        if (i + 1 < nmy) {
            load(i + 1, 1 - slot);
        }
        ai4k::cp_async_commit();

        const int kc = warp + i * WARPS;
        const uint8_t * st  = ring + slot * ST;
        const uint32_t * tw = (const uint32_t *) st;
        const half  * sd    = (const half  *) (st + CHB);
        const float * dx    = (const float *) (st + CHB + 64);
        const float d0 = __half2float(sd[g]);
        const float d1 = __half2float(sd[g + 8]);

        int p[NTI][4];
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            p[nt][0] = p[nt][1] = p[nt][2] = p[nt][3] = 0;
        }

        // Two half-chunk passes (half=0 -> j=0,1; half=1 -> j=2,3) keep only NTI uint4 live
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            uint4 xv[NTI];
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
                const int c = n0 + nt * 8 + g;
                if (c < N) {
                    const int8_t * xp = Xq + (int64_t) c * K + kc * 128 + t * 32 + half * 16;
                    xv[nt] = __ldg((const uint4 *) xp);
                } else {
                    xv[nt] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = half * 2 + jj;
                uint32_t a[4];
                ai4k::a_regs_lay<LAY>(tw, j, g, t, wa, wb, a);
#pragma unroll
                for (int nt = 0; nt < NTI; ++nt) {
                    ai4k::mma_s8(p[nt], a, jj ? xv[nt].z : xv[nt].x, jj ? xv[nt].w : xv[nt].y);
                }
            }
        }

#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            const float dx0 = dx[nt * 8 + 2 * t];
            const float dx1 = dx[nt * 8 + 2 * t + 1];
            acc[nt][0] += ai4k::magic_i2f(p[nt][0]) * (d0 * dx0);
            acc[nt][1] += ai4k::magic_i2f(p[nt][1]) * (d0 * dx1);
            acc[nt][2] += ai4k::magic_i2f(p[nt][2]) * (d1 * dx0);
            acc[nt][3] += ai4k::magic_i2f(p[nt][3]) * (d1 * dx1);
        }
    }

    ai4k::cp_async_wait<0>();
    __syncthreads();

    // Alias warp-reduction scratch onto dead cp.async smem -> 0 static SMEM!
    float * red = (float *) smem;
    if (warp > 0) {
        float * dst_w = red + (warp - 1) * NTI * 4 * 32;
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                dst_w[(nt * 4 + e) * 32 + lane] = acc[nt][e];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < WARPS - 1; ++w) {
            const float * src_w = red + w * NTI * 4 * 32;
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc[nt][e] += src_w[(nt * 4 + e) * 32 + lane];
                }
            }
        }
        const int r = slab * 16 + g;
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            const int c = n0 + nt * 8 + 2 * t;
            if (c < N) {
                Y[(int64_t) c * M + r]     = acc[nt][0];
                Y[(int64_t) c * M + r + 8] = acc[nt][2];
            }
            if (c + 1 < N) {
                Y[(int64_t) (c + 1) * M + r]     = acc[nt][1];
                Y[(int64_t) (c + 1) * M + r + 8] = acc[nt][3];
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 4: Fused Dense Gate + Up + GLU (N = 1..32, BN = 8/16/32) across ALL 11 AI+ unit layouts.
// One CTA owns 1 slab (16 rows) of BOTH W_gate and W_up. Interleaved 2-slot cp.async ring overlaps
// W_up HBM fetch with W_gate decode+IMMA and W_gate(next) fetch with W_up decode+IMMA!
// Zero static SMEM (except AIT2P_U pair_tab_s). Eliminates gate/up FP32 writes & GLU kernel launch.
// -------------------------------------------------------------------------------------------------

template <int BN, int LAY>
constexpr __host__ __device__ int dense_glu_wslot_bytes() {
    return ai4k::ch_bytes<LAY>() + 64; // tiles + 32 B scale header (+ 32 B pad for 16B align)
}

template <int BN, int LAY, int WARPS>
constexpr __host__ __device__ int dense_glu_smem_bytes() {
    constexpr int dx_bytes   = ((BN * 4 + 15) & ~15); // 16B aligned per dx slot
    constexpr int per_warp   = 2 * dense_glu_wslot_bytes<BN, LAY>() + 2 * dx_bytes;
    constexpr int stage_smem = WARPS * per_warp;
    constexpr int red_smem   = 2 * (WARPS - 1) * (BN / 8) * 4 * 32 * (int) sizeof(float);
    return stage_smem > red_smem ? stage_smem : red_smem;
}

template <int BN, int LAY, int XSH, int WARPS = 4>
static __global__ void __launch_bounds__(WARPS * 32, BN <= 8 ? 5 : BN <= 16 ? 4 : 2) k_dense_gate_up_glu(
        const uint8_t * __restrict__ W_gate,
        const uint8_t * __restrict__ W_up,
        const int8_t  * __restrict__ Xq,
        const float   * __restrict__ Xs,
        float         * __restrict__ Y_glu,
        int M, int N, int K, int glu_op, float alpha, float limit) {
    static_assert(BN == 8 || BN == 16 || BN == 32, "dense_glu BN");
    static_assert(WARPS == 4, "dense_glu WARPS");
    constexpr int NTI   = BN / 8;
    constexpr int CHB   = ai4k::ch_bytes<LAY>();
    constexpr int WST   = dense_glu_wslot_bytes<BN, LAY>();
    constexpr int DXB   = ((BN * 4 + 15) & ~15);
    constexpr int PER_W = 2 * WST + 2 * DXB;
    extern __shared__ __align__(16) uint8_t smem[];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    if constexpr (ai4k::lay_pair<LAY>()) {
        ai4k::pair_tab_fill(threadIdx.x, WARPS * 32);
        __syncthreads();
    }

    const int slab = blockIdx.x;
    const int n0   = blockIdx.y * BN;
    const int nk   = K / 128;
    const int xsn  = K >> (7 + XSH);
    const int64_t sb_off = (int64_t) slab * ai4k::slab_bytes_l<LAY>(K);
    const uint8_t * wg = W_gate + sb_off;
    const uint8_t * wu = W_up   + sb_off;

    const ai4k::k5win wa = ai4k::lay_win<LAY>(g, t & 1);
    const ai4k::k5win wb = ai4k::lay_win<LAY>(g + 8, t & 1);

    uint8_t * wbase  = smem + warp * PER_W;
    uint8_t * slot_g = wbase;
    uint8_t * slot_u = wbase + WST;
    float   * dx_buf = (float *) (wbase + 2 * WST);
    const int nmy = (nk - warp + WARPS - 1) / WARPS;

    auto load_w = [&](const uint8_t * ws, uint8_t * st, int kc) {
        const uint8_t * tiles = ws + ai4k::chunk_off<LAY>(kc);
        constexpr int CHP = CHB / 16;
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                ai4k::cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        if (lane < 8) {
            ai4k::cp_async4(st + CHB + lane * 4, ws + ai4k::sc_off<LAY>(kc) + lane * 4, true);
        }
    };
    auto load_dx = [&](int dx_slot, int kc) {
        if (lane < BN) {
            const bool live = n0 + lane < N;
            const float * src = live ? Xs + (int64_t) (n0 + lane) * xsn + (kc >> XSH) : Xs;
            ai4k::cp_async4((uint8_t *) dx_buf + dx_slot * DXB + lane * 4, src, live);
        }
    };

    float acc_g[NTI][4];
    float acc_u[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        acc_g[nt][0] = acc_g[nt][1] = acc_g[nt][2] = acc_g[nt][3] = 0.0f;
        acc_u[nt][0] = acc_u[nt][1] = acc_u[nt][2] = acc_u[nt][3] = 0.0f;
    }

    if (nmy > 0) {
        load_w(wg, slot_g, warp);
        load_dx(0, warp);
    }
    ai4k::cp_async_commit();

    for (int i = 0; i < nmy; ++i) {
        const int kc = warp + i * WARPS;
        const int dxs = i & 1;

        // Wait for W_gate(i) + dx(i), start W_up(i) prefetch immediately
        ai4k::cp_async_wait<0>();
        __syncwarp();
        load_w(wu, slot_u, kc);
        ai4k::cp_async_commit();

        const float * dx = (const float *) ((const uint8_t *) dx_buf + dxs * DXB);

        if constexpr (NTI == 1) {
            // BN = 8: keep Xq in 8 registers across BOTH W_gate and W_up (0 second LDG!)
            uint4 x0 = make_uint4(0u, 0u, 0u, 0u);
            uint4 x1 = make_uint4(0u, 0u, 0u, 0u);
            if (n0 + g < N) {
                const int8_t * xp = Xq + (int64_t) (n0 + g) * K + kc * 128 + t * 32;
                x0 = __ldg((const uint4 *) xp);
                x1 = __ldg((const uint4 *) (xp + 16));
            }
            const float dx0 = dx[2 * t];
            const float dx1 = dx[2 * t + 1];

            // Compute W_gate(i)
            {
                const uint32_t * tw = (const uint32_t *) slot_g;
                const half * sd = (const half *) (slot_g + CHB);
                const float d0 = __half2float(sd[g]);
                const float d1 = __half2float(sd[g + 8]);
                int p[4] = { 0, 0, 0, 0 };
                int q[4] = { 0, 0, 0, 0 };
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t a[4];
                    ai4k::a_regs_lay<LAY>(tw, j, g, t, wa, wb, a);
                    const uint4 & xv = j < 2 ? x0 : x1;
                    if (j < 2) { ai4k::mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                    else       { ai4k::mma_s8(q, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                }
                acc_g[0][0] += ai4k::magic_i2f(p[0] + q[0]) * (d0 * dx0);
                acc_g[0][1] += ai4k::magic_i2f(p[1] + q[1]) * (d0 * dx1);
                acc_g[0][2] += ai4k::magic_i2f(p[2] + q[2]) * (d1 * dx0);
                acc_g[0][3] += ai4k::magic_i2f(p[3] + q[3]) * (d1 * dx1);
            }

            // Wait for W_up(i), start W_gate(i+1) + dx(i+1) prefetch
            ai4k::cp_async_wait<0>();
            __syncwarp();
            if (i + 1 < nmy) {
                const int kc_next = warp + (i + 1) * WARPS;
                load_w(wg, slot_g, kc_next);
                load_dx(1 - dxs, kc_next);
            }
            ai4k::cp_async_commit();

            // Compute W_up(i) reusing x0, x1, dx0, dx1 in registers!
            {
                const uint32_t * tw = (const uint32_t *) slot_u;
                const half * sd = (const half *) (slot_u + CHB);
                const float d0 = __half2float(sd[g]);
                const float d1 = __half2float(sd[g + 8]);
                int p[4] = { 0, 0, 0, 0 };
                int q[4] = { 0, 0, 0, 0 };
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t a[4];
                    ai4k::a_regs_lay<LAY>(tw, j, g, t, wa, wb, a);
                    const uint4 & xv = j < 2 ? x0 : x1;
                    if (j < 2) { ai4k::mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                    else       { ai4k::mma_s8(q, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                }
                acc_u[0][0] += ai4k::magic_i2f(p[0] + q[0]) * (d0 * dx0);
                acc_u[0][1] += ai4k::magic_i2f(p[1] + q[1]) * (d0 * dx1);
                acc_u[0][2] += ai4k::magic_i2f(p[2] + q[2]) * (d1 * dx0);
                acc_u[0][3] += ai4k::magic_i2f(p[3] + q[3]) * (d1 * dx1);
            }
        } else {
            // BN = 16 or 32: process in 2-tile (16-token) sub-batches to cap live xv/p regs -> 0 spills on all 11 layouts!
            auto run_mat = [&](const uint8_t * st, float (&acc)[NTI][4]) {
                const uint32_t * tw = (const uint32_t *) st;
                const half * sd = (const half *) (st + CHB);
                const float d0 = __half2float(sd[g]);
                const float d1 = __half2float(sd[g + 8]);
                constexpr int NTG = 2; // 2 n8 tiles (16 tokens) per sub-batch
#pragma unroll
                for (int nt0 = 0; nt0 < NTI; nt0 += NTG) {
                    int p[NTG][4];
#pragma unroll
                    for (int k = 0; k < NTG; ++k) {
                        p[k][0] = p[k][1] = p[k][2] = p[k][3] = 0;
                    }
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        uint4 xv[NTG];
#pragma unroll
                        for (int k = 0; k < NTG; ++k) {
                            const int c = n0 + (nt0 + k) * 8 + g;
                            if (c < N) {
                                const int8_t * xp = Xq + (int64_t) c * K + kc * 128 + t * 32 + half * 16;
                                xv[k] = __ldg((const uint4 *) xp);
                            } else {
                                xv[k] = make_uint4(0u, 0u, 0u, 0u);
                            }
                        }
#pragma unroll
                        for (int jj = 0; jj < 2; ++jj) {
                            const int j = half * 2 + jj;
                            uint32_t a[4];
                            ai4k::a_regs_lay<LAY>(tw, j, g, t, wa, wb, a);
#pragma unroll
                            for (int k = 0; k < NTG; ++k) {
                                ai4k::mma_s8(p[k], a, jj ? xv[k].z : xv[k].x, jj ? xv[k].w : xv[k].y);
                            }
                        }
                    }
#pragma unroll
                    for (int k = 0; k < NTG; ++k) {
                        const int nt = nt0 + k;
                        const float dx0 = dx[nt * 8 + 2 * t];
                        const float dx1 = dx[nt * 8 + 2 * t + 1];
                        acc[nt][0] += ai4k::magic_i2f(p[k][0]) * (d0 * dx0);
                        acc[nt][1] += ai4k::magic_i2f(p[k][1]) * (d0 * dx1);
                        acc[nt][2] += ai4k::magic_i2f(p[k][2]) * (d1 * dx0);
                        acc[nt][3] += ai4k::magic_i2f(p[k][3]) * (d1 * dx1);
                    }
                }
            };

            run_mat(slot_g, acc_g);

            ai4k::cp_async_wait<0>();
            __syncwarp();
            if (i + 1 < nmy) {
                const int kc_next = warp + (i + 1) * WARPS;
                load_w(wg, slot_g, kc_next);
                load_dx(1 - dxs, kc_next);
            }
            ai4k::cp_async_commit();

            run_mat(slot_u, acc_u);
        }
    }

    ai4k::cp_async_wait<0>();
    __syncthreads();

    // Alias both warp-reduction buffers onto dead cp.async smem -> 0 static SMEM!
    constexpr int RED_PER_WARP = NTI * 4 * 32;
    float * red_g = (float *) smem;
    float * red_u = red_g + (WARPS - 1) * RED_PER_WARP;
    if (warp > 0) {
        float * dg = red_g + (warp - 1) * RED_PER_WARP;
        float * du = red_u + (warp - 1) * RED_PER_WARP;
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                dg[(nt * 4 + e) * 32 + lane] = acc_g[nt][e];
                du[(nt * 4 + e) * 32 + lane] = acc_u[nt][e];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int w = 0; w < WARPS - 1; ++w) {
            const float * sg = red_g + w * RED_PER_WARP;
            const float * su = red_u + w * RED_PER_WARP;
#pragma unroll
            for (int nt = 0; nt < NTI; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc_g[nt][e] += sg[(nt * 4 + e) * 32 + lane];
                    acc_u[nt][e] += su[(nt * 4 + e) * 32 + lane];
                }
            }
        }
        const int r = slab * 16 + g;
#pragma unroll
        for (int nt = 0; nt < NTI; ++nt) {
            const int c = n0 + nt * 8 + 2 * t;
            if (c < N) {
                Y_glu[(int64_t) c * M + r]     = eval_glu_rt(glu_op, acc_g[nt][0], acc_u[nt][0], alpha, limit);
                Y_glu[(int64_t) c * M + r + 8] = eval_glu_rt(glu_op, acc_g[nt][2], acc_u[nt][2], alpha, limit);
            }
            if (c + 1 < N) {
                Y_glu[(int64_t) (c + 1) * M + r]     = eval_glu_rt(glu_op, acc_g[nt][1], acc_u[nt][1], alpha, limit);
                Y_glu[(int64_t) (c + 1) * M + r + 8] = eval_glu_rt(glu_op, acc_g[nt][3], acc_u[nt][3], alpha, limit);
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 5: Full FFN Bridge — Fused Dense Gate + Up + GLU + H256/H128 + Int8 Quantize!
// Slab-per-warp (SPW): 16 warps (= 256 output rows for XSH_OUT=1) or 8 warps (= 128 output rows
// for XSH_OUT=0) per CTA. Each warp scans ALL K/128 chunks of its own 16-row slab (zero cross-warp
// K-reduction!), writes GLU outputs transposed into dead smem, then warps run H256/H128 + int8
// quantize straight into ai_prep_cache for W_down! Eliminates 3 kernel launches & 6*N*F*4 B VRAM!
// -------------------------------------------------------------------------------------------------

template <int BN, int LAY, int XSH_OUT>
constexpr __host__ __device__ int dense_glu_q8_smem_bytes() {
    constexpr int NWARPS     = XSH_OUT ? 16 : 8;
    constexpr int RB         = XSH_OUT ? 256 : 128;
    constexpr int RB_PAD     = RB + 8; // +32 B stride padding for conflict-free transpose
    constexpr int dx_bytes   = ((BN * 4 + 15) & ~15);
    constexpr int per_warp   = 2 * dense_glu_wslot_bytes<BN, LAY>() + 2 * dx_bytes;
    constexpr int stage_smem = NWARPS * per_warp;
    constexpr int trans_smem = BN * RB_PAD * (int) sizeof(float);
    return stage_smem > trans_smem ? stage_smem : trans_smem;
}

template <int BN, int LAY, int XSH_IN, int XSH_OUT>
static __global__ void __launch_bounds__((XSH_OUT ? 16 : 8) * 32, 1) k_dense_gate_up_glu_q8(
        const uint8_t * __restrict__ W_gate,
        const uint8_t * __restrict__ W_up,
        const int8_t  * __restrict__ Xq_in,
        const float   * __restrict__ Xs_in,
        float         * __restrict__ Y_glu_f32,
        int8_t        * __restrict__ Xq_out,
        float         * __restrict__ Xs_out,
        int M, int N, int K, int glu_op, float alpha, float limit) {
    static_assert(BN == 8 || BN == 16 || BN == 32, "dense_glu_q8 BN");
    constexpr int NWARPS = XSH_OUT ? 16 : 8;
    constexpr int RB     = XSH_OUT ? 256 : 128;
    constexpr int RB_PAD = RB + 8;
    constexpr int NTI    = BN / 8;
    constexpr int CHB    = ai4k::ch_bytes<LAY>();
    constexpr int WST    = dense_glu_wslot_bytes<BN, LAY>();
    constexpr int DXB    = ((BN * 4 + 15) & ~15);
    constexpr int PER_W  = 2 * WST + 2 * DXB;
    extern __shared__ __align__(16) uint8_t smem[];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    if constexpr (ai4k::lay_pair<LAY>()) {
        ai4k::pair_tab_fill(threadIdx.x, NWARPS * 32);
        __syncthreads();
    }

    const int blk  = blockIdx.x;                  // H256 or H128 block index along M (= K_down)
    const int slab = blk * NWARPS + warp;         // 16-row slab owned by this warp
    const int n0   = blockIdx.y * BN;
    const int n_live = min(BN, N - n0);
    const int nk   = K / 128;
    const int xsn  = K >> (7 + XSH_IN);
    const int64_t sb_off = (int64_t) slab * ai4k::slab_bytes_l<LAY>(K);
    const uint8_t * wg = W_gate + sb_off;
    const uint8_t * wu = W_up   + sb_off;

    const ai4k::k5win wa = ai4k::lay_win<LAY>(g, t & 1);
    const ai4k::k5win wb = ai4k::lay_win<LAY>(g + 8, t & 1);

    uint8_t * wbase  = smem + warp * PER_W;
    uint8_t * slot_g = wbase;
    uint8_t * slot_u = wbase + WST;
    float   * dx_buf = (float *) (wbase + 2 * WST);

    auto load_w = [&](const uint8_t * ws, uint8_t * st, int kc) {
        const uint8_t * tiles = ws + ai4k::chunk_off<LAY>(kc);
        constexpr int CHP = CHB / 16;
#pragma unroll
        for (int r = 0; r < (CHP + 31) / 32; ++r) {
            if (CHP % 32 == 0 || lane + 32 * r < CHP) {
                ai4k::cp_async16(st + r * 512 + lane * 16, tiles + r * 512 + lane * 16, true);
            }
        }
        if (lane < 8) {
            ai4k::cp_async4(st + CHB + lane * 4, ws + ai4k::sc_off<LAY>(kc) + lane * 4, true);
        }
    };
    auto load_dx = [&](int dx_slot, int kc) {
        if (lane < BN) {
            const bool live = lane < n_live;
            const float * src = live ? Xs_in + (int64_t) (n0 + lane) * xsn + (kc >> XSH_IN) : Xs_in;
            ai4k::cp_async4((uint8_t *) dx_buf + dx_slot * DXB + lane * 4, src, live);
        }
    };

    float acc_g[NTI][4];
    float acc_u[NTI][4];
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        acc_g[nt][0] = acc_g[nt][1] = acc_g[nt][2] = acc_g[nt][3] = 0.0f;
        acc_u[nt][0] = acc_u[nt][1] = acc_u[nt][2] = acc_u[nt][3] = 0.0f;
    }

    if (nk > 0) {
        load_w(wg, slot_g, 0);
        load_dx(0, 0);
    }
    ai4k::cp_async_commit();

    for (int kc = 0; kc < nk; ++kc) {
        const int dxs = kc & 1;
        ai4k::cp_async_wait<0>();
        __syncwarp();
        load_w(wu, slot_u, kc);
        ai4k::cp_async_commit();

        const float * dx = (const float *) ((const uint8_t *) dx_buf + dxs * DXB);

        if constexpr (NTI == 1) {
            uint4 x0 = make_uint4(0u, 0u, 0u, 0u);
            uint4 x1 = make_uint4(0u, 0u, 0u, 0u);
            if (g < n_live) {
                const int8_t * xp = Xq_in + (int64_t) (n0 + g) * K + kc * 128 + t * 32;
                x0 = __ldg((const uint4 *) xp);
                x1 = __ldg((const uint4 *) (xp + 16));
            }
            const float dx0 = dx[2 * t];
            const float dx1 = dx[2 * t + 1];

            {
                const uint32_t * tw = (const uint32_t *) slot_g;
                const half * sd = (const half *) (slot_g + CHB);
                const float d0 = __half2float(sd[g]);
                const float d1 = __half2float(sd[g + 8]);
                int p[4] = { 0, 0, 0, 0 };
                int q[4] = { 0, 0, 0, 0 };
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t a[4];
                    ai4k::a_regs_lay<LAY>(tw, j, g, t, wa, wb, a);
                    const uint4 & xv = j < 2 ? x0 : x1;
                    if (j < 2) { ai4k::mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                    else       { ai4k::mma_s8(q, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                }
                acc_g[0][0] += ai4k::magic_i2f(p[0] + q[0]) * (d0 * dx0);
                acc_g[0][1] += ai4k::magic_i2f(p[1] + q[1]) * (d0 * dx1);
                acc_g[0][2] += ai4k::magic_i2f(p[2] + q[2]) * (d1 * dx0);
                acc_g[0][3] += ai4k::magic_i2f(p[3] + q[3]) * (d1 * dx1);
            }

            ai4k::cp_async_wait<0>();
            __syncwarp();
            if (kc + 1 < nk) {
                load_w(wg, slot_g, kc + 1);
                load_dx(1 - dxs, kc + 1);
            }
            ai4k::cp_async_commit();

            {
                const uint32_t * tw = (const uint32_t *) slot_u;
                const half * sd = (const half *) (slot_u + CHB);
                const float d0 = __half2float(sd[g]);
                const float d1 = __half2float(sd[g + 8]);
                int p[4] = { 0, 0, 0, 0 };
                int q[4] = { 0, 0, 0, 0 };
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t a[4];
                    ai4k::a_regs_lay<LAY>(tw, j, g, t, wa, wb, a);
                    const uint4 & xv = j < 2 ? x0 : x1;
                    if (j < 2) { ai4k::mma_s8(p, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                    else       { ai4k::mma_s8(q, a, (j & 1) ? xv.z : xv.x, (j & 1) ? xv.w : xv.y); }
                }
                acc_u[0][0] += ai4k::magic_i2f(p[0] + q[0]) * (d0 * dx0);
                acc_u[0][1] += ai4k::magic_i2f(p[1] + q[1]) * (d0 * dx1);
                acc_u[0][2] += ai4k::magic_i2f(p[2] + q[2]) * (d1 * dx0);
                acc_u[0][3] += ai4k::magic_i2f(p[3] + q[3]) * (d1 * dx1);
            }
        } else {
            // BN = 16 or 32: process in 2-tile (16-token) sub-batches -> caps live regs below 128 (0 spills on 16-warp CTA!)
            auto run_mat = [&](const uint8_t * st, float (&acc)[NTI][4]) {
                const uint32_t * tw = (const uint32_t *) st;
                const half * sd = (const half *) (st + CHB);
                const float d0 = __half2float(sd[g]);
                const float d1 = __half2float(sd[g + 8]);
                constexpr int NTG = 2;
#pragma unroll
                for (int nt0 = 0; nt0 < NTI; nt0 += NTG) {
                    int p[NTG][4];
#pragma unroll
                    for (int k = 0; k < NTG; ++k) {
                        p[k][0] = p[k][1] = p[k][2] = p[k][3] = 0;
                    }
#pragma unroll
                    for (int half = 0; half < 2; ++half) {
                        uint4 xv[NTG];
#pragma unroll
                        for (int k = 0; k < NTG; ++k) {
                            const int c = (nt0 + k) * 8 + g;
                            if (c < n_live) {
                                const int8_t * xp = Xq_in + (int64_t) (n0 + c) * K + kc * 128 + t * 32 + half * 16;
                                xv[k] = __ldg((const uint4 *) xp);
                            } else {
                                xv[k] = make_uint4(0u, 0u, 0u, 0u);
                            }
                        }
#pragma unroll
                        for (int jj = 0; jj < 2; ++jj) {
                            const int j = half * 2 + jj;
                            uint32_t a[4];
                            ai4k::a_regs_lay<LAY>(tw, j, g, t, wa, wb, a);
#pragma unroll
                            for (int k = 0; k < NTG; ++k) {
                                ai4k::mma_s8(p[k], a, jj ? xv[k].z : xv[k].x, jj ? xv[k].w : xv[k].y);
                            }
                        }
                    }
#pragma unroll
                    for (int k = 0; k < NTG; ++k) {
                        const int nt = nt0 + k;
                        const float dx0 = dx[nt * 8 + 2 * t];
                        const float dx1 = dx[nt * 8 + 2 * t + 1];
                        acc[nt][0] += ai4k::magic_i2f(p[k][0]) * (d0 * dx0);
                        acc[nt][1] += ai4k::magic_i2f(p[k][1]) * (d0 * dx1);
                        acc[nt][2] += ai4k::magic_i2f(p[k][2]) * (d1 * dx0);
                        acc[nt][3] += ai4k::magic_i2f(p[k][3]) * (d1 * dx1);
                    }
                }
            };

            run_mat(slot_g, acc_g);

            ai4k::cp_async_wait<0>();
            __syncwarp();
            if (kc + 1 < nk) {
                load_w(wg, slot_g, kc + 1);
                load_dx(1 - dxs, kc + 1);
            }
            ai4k::cp_async_commit();

            run_mat(slot_u, acc_u);
        }
    }

    ai4k::cp_async_wait<0>();
    __syncthreads();

    // Epilogue Phase 1: write GLU(acc_g, acc_u) transposed into dead smem[tok][row_in_block]
    float * s_glu = (float *) smem;
    const int r_loc = warp * 16 + g;
    const int r_glb = slab * 16 + g;
#pragma unroll
    for (int nt = 0; nt < NTI; ++nt) {
        const int c = nt * 8 + 2 * t;
        if (c < n_live) {
            const float v0 = eval_glu_rt(glu_op, acc_g[nt][0], acc_u[nt][0], alpha, limit);
            const float v2 = eval_glu_rt(glu_op, acc_g[nt][2], acc_u[nt][2], alpha, limit);
            s_glu[c * RB_PAD + r_loc]     = v0;
            s_glu[c * RB_PAD + r_loc + 8] = v2;
            if (Y_glu_f32 != nullptr) {
                Y_glu_f32[(int64_t) (n0 + c) * M + r_glb]     = v0;
                Y_glu_f32[(int64_t) (n0 + c) * M + r_glb + 8] = v2;
            }
        }
        if (c + 1 < n_live) {
            const float v1 = eval_glu_rt(glu_op, acc_g[nt][1], acc_u[nt][1], alpha, limit);
            const float v3 = eval_glu_rt(glu_op, acc_g[nt][3], acc_u[nt][3], alpha, limit);
            s_glu[(c + 1) * RB_PAD + r_loc]     = v1;
            s_glu[(c + 1) * RB_PAD + r_loc + 8] = v3;
            if (Y_glu_f32 != nullptr) {
                Y_glu_f32[(int64_t) (n0 + c + 1) * M + r_glb]     = v1;
                Y_glu_f32[(int64_t) (n0 + c + 1) * M + r_glb + 8] = v3;
            }
        }
    }
    __syncthreads();

    // Epilogue Phase 2: Hadamard H256 / H128 + int8 quantize directly into Xq_out / Xs_out (PLAIN order)
    if constexpr (XSH_OUT == 1) {
        for (int c = warp; c < n_live; c += NWARPS) {
            const float * row = s_glu + c * RB_PAD + lane * 8;
            float v[8] = { row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7] };
            rotate_quantize_256<PREP_PLAIN>(v, Xq_out, Xs_out, M, blk, n0 + c);
        }
    } else {
        const int l = lane & 15;
        for (int c_base = warp * 2; c_base < n_live; c_base += NWARPS * 2) {
            const int c = c_base + (lane >> 4);
            const bool live = c < n_live;
            float v[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            if (live) {
                const float * row = s_glu + c * RB_PAD + l * 8;
                v[0] = row[0]; v[1] = row[1]; v[2] = row[2]; v[3] = row[3];
                v[4] = row[4]; v[5] = row[5]; v[6] = row[6]; v[7] = row[7];
            }
            rotate_quantize_128<PREP_PLAIN>(v, Xq_out, Xs_out, M, blk, n0 + c, live);
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Typed C++ launch helpers (implemented in ai4-fuse.cu for clean separate TU compilation)
// -------------------------------------------------------------------------------------------------

void launch_rms_norm_mul_quantize_act_ai4(
        const float * x, int64_t sx, const float * gamma, float * dst_f32,
        int8_t * q, float * s, int K, int N, float eps,
        int xsh, int variant, bool write_f32, cudaStream_t stream);

void launch_swiglu_quantize_act_ai4(
        const float * gate, int64_t s_gate, const float * up, int64_t s_up, float * dst_f32,
        int8_t * q, float * s, int K, int N,
        int xsh, int variant, bool write_f32, int glu_op, float alpha, float limit, cudaStream_t stream);

void launch_mmv_grp_spec_ai4(
        int lay, int xsh, const ai4k::mmv_group & grp,
        const int8_t * Xq, const float * Xs, int N, int K, cudaStream_t stream);

void launch_dense_gate_up_glu_ai4(
        int lay, int xsh, int glu_op,
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq, const float * Xs, float * Y_glu,
        int M, int N, int K, float alpha, float limit, cudaStream_t stream);

void launch_dense_gate_up_glu_q8_ai4(
        int lay, int xsh_in, int xsh_out, int glu_op, bool write_f32,
        const uint8_t * W_gate, const uint8_t * W_up,
        const int8_t * Xq_in, const float * Xs_in,
        float * Y_glu_f32, int8_t * Xq_out, float * Xs_out,
        int M, int N, int K, float alpha, float limit, cudaStream_t stream);

} // namespace ai4fuse
