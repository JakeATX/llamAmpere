#pragma once
// AI+ unit types (AITk_U): rate-k tail-biting bitshift trellis, s8 levels, RQ4 rotated basis. Format: ggml/src/ggml-ai4.h.
// Header-only device helpers for the ggml CUDA backend.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "../ggml-ait.h"   // AIT2P_TAB_INIT (AIT2P_U levels)

// RQ4 rotation signs (ggml-rq4.h RQ4_SIGN_WORDS == ggml-ai4.h ai4_sign_bits)
static __device__ __constant__ uint32_t ai4_dev_sign_bits[8] = {
    0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5ced2a1bu,
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au };

// level of a 16-bit state: ((bytesum(state * C) + 2) >> 2) - 128, as in aik_level_c. C is the per-rate trellis constant
// of the AI+ type (ggml_ai4_trellis_mul).
static __device__ __forceinline__ int aik_dev_level_c(const uint32_t state, const uint32_t C) {
    const uint32_t x = state * C;
    return (int) ((__dp4a(x, 0x01010101u, 2u)) >> 2) - 128;
}

// In-warp 256-point FWHT (Sylvester order, unnormalized): lane holds elements lane*8 + q, q = 0..7.
static __device__ __forceinline__ void ai4_dev_fwht256(float (&v)[8]) {
#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            if ((q & h) == 0) {
                const float a = v[q], b = v[q + h];
                v[q] = a + b;
                v[q + h] = a - b;
            }
        }
    }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float o = __shfl_xor_sync(0xffffffffu, v[q], m);
            v[q] = (lane & m) ? (o - v[q]) : (v[q] + o);
        }
    }
}

static __device__ __forceinline__ bool ai4_dev_sign_neg(const int i) {
    return (ai4_dev_sign_bits[i >> 5] >> (i & 31)) & 1;
}

// AI+ unit layout at rate KB (ggml-ai4.h "AI+ unit layout at rate k"). Tile = 128*KB-bit MSB-first stream in
// 4*KB words; state of symbol n = the 16 stream bits ending at bit KB*(n+1) (mod 128*KB). Unit = 32 + 256*KB bytes.
template <int KB>
static __device__ __forceinline__ uint32_t aik_dev_state(const uint32_t * __restrict__ tile, const int n) {
    constexpr int NB = 128 * KB, NW = 4 * KB;
    int b0 = KB * n + KB - 16;
    b0 += b0 < 0 ? NB : 0;
    const int w = b0 >> 5, sh = b0 & 31;
    const int w1 = w + 1 == NW ? 0 : w + 1;
    const uint64_t c = ((uint64_t) tile[w] << 32) | tile[w1];
    return (uint32_t) (c >> (48 - sh)) & 0xFFFFu;
}

// AIT2P_U (ggml-ai4.h): pair p = symbols 2p, 2p+1 of a rate-2 tile; window = the 20 stream bits ending at bit 4p+4.
// level(2p + ce) = aik_pair_dev_tab[2 entry + ce]. The reference path reads the table from global memory.
static __device__ const int8_t aik_pair_dev_tab[2048] = AIT2P_TAB_INIT;

static __device__ __forceinline__ int aik_dev_pair_entry(const uint32_t w) {
    return (int) (((w * (w + 1u)) >> 10) & 1023u);
}

static __device__ __forceinline__ int aik_dev_pair_level(const uint32_t * __restrict__ tile, const int n) {
    constexpr int NB = 256, NW = 8;
    int b0 = 4 * (n >> 1) + 4 - 20;
    b0 += b0 < 0 ? NB : 0;
    const int w = b0 >> 5, sh = b0 & 31;
    const int w1 = w + 1 == NW ? 0 : w + 1;
    const uint64_t c = ((uint64_t) tile[w] << 32) | tile[w1];
    return aik_pair_dev_tab[2 * aik_dev_pair_entry((uint32_t) (c >> (44 - sh))) + (n & 1)];
}

// Reconstruct W[M][K] from a rate-KB unit tensor. C == 0 (KB == 2): AIT2P_U pair-table levels. UNROT = true: original basis; false: rotated basis (the values the
// IMMA path multiplies with rotated activations). Block = 512 threads = 16 warps = the 16 rows of one slab, grid =
// (ceil(K/256), M/16); lane l of warp r decodes unit 2*blockIdx.x + l/16, column step l%16 (8 K columns), then the warp
// (RB = 256) or each half-warp (RB = 128) unrotates its block.
template <typename T, int KB, int RB, bool UNROT>
static __global__ void __launch_bounds__(512) k_aik_reconstruct(const uint8_t * __restrict__ src, T * __restrict__ W,
                                                               const int64_t K, const uint32_t C, const bool row_scale) {
    constexpr int UB = 32 + 256 * KB;
    const int r = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    const int64_t nu = K / 128;
    const int64_t u  = (int64_t) blockIdx.x * 2 + (l >> 4);
    const bool live  = u < nu;
    float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (live) {
        // U2 shares one scale header across five consecutive 128-column chunks.
        const int64_t slab_bytes = row_scale ? (nu / 5) * (32 + 5 * 256 * KB) : nu * UB;
        const uint8_t * slab = src + (int64_t) blockIdx.y * slab_bytes;
        const uint8_t * blk = slab + (row_scale ? (u / 5) * (32 + 5 * 256 * KB) : u * UB);
        const int64_t chunk = row_scale ? (u % 5) * 256 * KB : 0;
        const uint32_t * tile = (const uint32_t *) (blk + 32 + chunk) + (l & 15) * (4 * KB);
        const float d = __half2float(__ushort_as_half(((const uint16_t *) blk)[r]));
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            if (KB == 2 && C == 0) {
                v[q] = d * (float) aik_dev_pair_level(tile, r * 8 + q);
            } else {
                v[q] = d * (float) aik_dev_level_c(aik_dev_state<KB>(tile, r * 8 + q), C);
            }
        }
    }
    if constexpr (UNROT) {
#pragma unroll
        for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                if ((q & h) == 0) {
                    const float a = v[q], b = v[q + h];
                    v[q] = a + b;
                    v[q + h] = a - b;
                }
            }
        }
#pragma unroll
        for (int m = 1; m < RB / 8; m <<= 1) {
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float o = __shfl_xor_sync(0xffffffffu, v[q], m);
                v[q] = (l & m) ? (o - v[q]) : (v[q] + o);
            }
        }
    }
    if (!live) {
        return;
    }
    const float nrm = RB == 256 ? 1.0f / 16.0f : 0.08838834764831845f;
    T * out = W + ((int64_t) blockIdx.y * 16 + r) * K + u * 128 + (l & 15) * 8;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const float s = UNROT ? (ai4_dev_sign_neg((l * 8 + q) % RB) ? -nrm : nrm) : 1.0f;
        if constexpr (sizeof(T) == 2) {
            out[q] = __float2half(v[q] * s);
        } else {
            out[q] = v[q] * s;
        }
    }
}

template <typename T, int KB>
static void aik_reconstruct_launch(const void * src, T * W, const int64_t K, const int64_t M, const bool unrot, cudaStream_t stream,
                                   const uint32_t C, const bool row_scale) {
    const dim3 grid((unsigned) ((K / 128 + 1) / 2), (unsigned) (M / 16));
    if (K % 256 == 0) {
        if (unrot) k_aik_reconstruct<T, KB, 256, true ><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C, row_scale);
        else       k_aik_reconstruct<T, KB, 256, false><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C, row_scale);
    } else {
        if (unrot) k_aik_reconstruct<T, KB, 128, true ><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C, row_scale);
        else       k_aik_reconstruct<T, KB, 128, false><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C, row_scale);
    }
}

// Reference reconstruct at any AI+ unit rate (2, 3, 4, 5). Returns false for an unknown rate.
template <typename T>
static bool aik_reconstruct_rate(const int rate, const void * src, T * W, const int64_t K, const int64_t M, const bool unrot,
                                 cudaStream_t stream, const uint32_t C, const bool row_scale = false) {
    GGML_ASSERT(!row_scale || K % 640 == 0);
    switch (rate) {
        case 2: aik_reconstruct_launch<T, 2>(src, W, K, M, unrot, stream, C, row_scale); return true;
        case 3: aik_reconstruct_launch<T, 3>(src, W, K, M, unrot, stream, C, row_scale); return true;
        case 4: aik_reconstruct_launch<T, 4>(src, W, K, M, unrot, stream, C, row_scale); return true;
        case 5: aik_reconstruct_launch<T, 5>(src, W, K, M, unrot, stream, C, row_scale); return true;
        default: return false;
    }
}

// Activation emulation for the rate-k reference path: X [N][K] -> rotated (per RB block, as the weights) and int8
// round-tripped with one scale amax/127 per (token, RB block), the arithmetic of the W4A8 path's activation quantizer.
// grid (K/RB, N), 32 threads; lane holds RB/32 consecutive elements.
template <int RB>
static __global__ void __launch_bounds__(32) k_aik_act_emul(const float * __restrict__ X, float * __restrict__ Y, const int64_t K) {
    constexpr int VPT = RB / 32;
    const int l = threadIdx.x;
    const int64_t off = (int64_t) blockIdx.y * K + (int64_t) blockIdx.x * RB + l * VPT;
    float v[VPT];
#pragma unroll
    for (int q = 0; q < VPT; ++q) {
        v[q] = ai4_dev_sign_neg(l * VPT + q) ? -X[off + q] : X[off + q];
    }
#pragma unroll
    for (int h = 1; h < VPT; h <<= 1) {
#pragma unroll
        for (int q = 0; q < VPT; ++q) {
            if ((q & h) == 0) {
                const float a = v[q], b = v[q + h];
                v[q] = a + b;
                v[q + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int q = 0; q < VPT; ++q) {
            const float o = __shfl_xor_sync(0xffffffffu, v[q], m);
            v[q] = (l & m) ? (o - v[q]) : (v[q] + o);
        }
    }
    const float nrm = RB == 256 ? 1.0f / 16.0f : 0.08838834764831845f;
    float amax = 0.0f;
#pragma unroll
    for (int q = 0; q < VPT; ++q) {
        v[q] *= nrm;
        amax = fmaxf(amax, fabsf(v[q]));
    }
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
#pragma unroll
    for (int q = 0; q < VPT; ++q) {
        Y[off + q] = d * (float) __float2int_rn(v[q] * id);
    }
}

static void aik_act_emul_launch(const float * X, float * Y, const int64_t K, const int64_t N, cudaStream_t stream) {
    for (int64_t r0 = 0; r0 < N; r0 += 65535) {
        const unsigned nr = (unsigned) (N - r0 < 65535 ? N - r0 : 65535);
        if (K % 256 == 0) {
            k_aik_act_emul<256><<<dim3((unsigned) (K / 256), nr), 32, 0, stream>>>(X + r0 * K, Y + r0 * K, K);
        } else {
            k_aik_act_emul<128><<<dim3((unsigned) (K / 128), nr), 32, 0, stream>>>(X + r0 * K, Y + r0 * K, K);
        }
    }
}
