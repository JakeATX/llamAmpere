#pragma once
// [PREFILL-GLUE] Shared device inline quantizers (SMEM/row -> AI4 / NVFP4PF) for prefill-glue.cu and depth-mix.cu.

#include "common.cuh"
#include "rq4.cuh"
#include "ai4-kern.cuh"

static constexpr float PF_NVFP4PF_MIN_SCALE = 1e-30f;

static __device__ __forceinline__ int pf_nvfp4pf_f2i(const float x) {
    return __float2int_rn(x);
}

// Quantize one row `row_src[0..K-1]` (may be shared memory `s_out` or global) into AI+ H256 (xsh=1) or H128 (xsh=0)
// using all `nwarps` warps of the block (`blockDim.x` threads, multiple of 32).
template <int XSH, bool SWAP8 = false, bool XPERM = false, bool LINEAR = false>
static __device__ __forceinline__ void pf_quantize_row_ai4_from_ptr(
        const float * __restrict__ row_src, int8_t * __restrict__ q, float * __restrict__ s,
        int K, int tok, int nwarps) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if constexpr (XSH == 1) {
        const int nb = K / RQ4_HB;
        for (int hb = warp; hb < nb; hb += nwarps) {
            const float * xr = row_src + hb * RQ4_HB + lane * 8;
            const float4 u0 = *(const float4 *) xr;
            const float4 u1 = *(const float4 *) (xr + 4);
            float v[8] = { u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w };
            rq4::quantize_act_regs<SWAP8, XPERM, LINEAR>(v, hb, q, s, K, tok);
        }
    } else {
        const int nb = K / 128;
        const int l  = lane & 15;
        const int hw = lane >> 4;
        for (int cb0 = warp * 2; cb0 < nb; cb0 += nwarps * 2) {
            const int cb = cb0 + hw;
            const bool live = cb < nb;
            float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (live) {
                const float * xr = row_src + cb * 128 + l * 8;
                const float4 u0 = *(const float4 *) xr;
                const float4 u1 = *(const float4 *) (xr + 4);
                v[0] = u0.x; v[1] = u0.y; v[2] = u0.z; v[3] = u0.w;
                v[4] = u1.x; v[5] = u1.y; v[6] = u1.z; v[7] = u1.w;
            }
            ai4k::quantize_act128_regs<SWAP8, XPERM, LINEAR>(v, live, l, cb, q, s, K, tok);
        }
    }
}

// Quantize one row `row_src[0..K-1]` (SMEM or global) into NVFP4PF layout (KF = 64 or 32) using all `nwarps` warps.
template <int KF>
static __device__ __forceinline__ void pf_quantize_row_nvfp4pf_from_ptr(
        const float * __restrict__ row_src, int8_t * __restrict__ xq, float * __restrict__ xs,
        int K, int tok, int nwarps) {
    static_assert(KF == 32 || KF == 64, "NVFP4PF: KF 32 or 64");
    constexpr int V = KF / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nb   = K / KF;
    for (int kb = warp; kb < nb; kb += nwarps) {
        const float * xp = row_src + kb * KF + V * lane;
        float v[V];
        bool bad = false;
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            v[i] = xp[i];
            bad  = bad || !isfinite(v[i]);
            v[i] = isfinite(v[i]) ? v[i] : 0.0f;
            amax = fmaxf(amax, fabsf(v[i]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o, 32));
        }
        bad = __any_sync(0xffffffff, bad);
        const bool  ok  = amax >= PF_NVFP4PF_MIN_SCALE;
        const float inv = ok ? 127.0f / amax : 0.0f;
        int8_t * q = xq + (int64_t) tok * K + kb * KF + V * lane;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            q[i] = (int8_t) max(-127, min(127, pf_nvfp4pf_f2i(v[i] * inv)));
        }
        if (lane == 0) {
            xs[(int64_t) tok * nb + kb] = bad ? __int_as_float(0x7fc00000) : ok ? amax * (1.0f / 127.0f) : 0.0f;
        }
    }
}
