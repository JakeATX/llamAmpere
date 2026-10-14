#pragma once
// AI4 (Ampere INT4, formerly D2): 4-bit tail-biting bitshift trellis, s8 levels, RQ4 rotated basis. Format: ggml/src/ggml-ai4.h.
// Header-only device helpers so tools/ai4 (the GPU encoder) and the ggml CUDA backend share one decoder.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define AI4_DEV_SB_BYTES    8448   // slab super block: 16 rows x 1024 K
#define AI4_DEV_SCALE_BYTES 256

// RQ4 rotation signs (ggml-rq4.h RQ4_SIGN_WORDS == ggml-ai4.h ai4_sign_bits)
static __device__ __constant__ uint32_t ai4_dev_sign_bits[8] = {
    0x9e3779b9u, 0x7f4a7c15u, 0xf39cc060u, 0x5ced2a1bu,
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au };

// level of a 16-bit state: ((bytesum(state * C) + 2) >> 2) - 128, as in aik_level_c. C is part of the AI+ type
// (ggml_ai4_trellis_mul): 0x83DCD12D for AI4 / AI4_U / AI3_U / AI2_U / AI5_U, 0xA1EEE321 for AI6_U.
#define AI4_DEV_MUL 0x83DCD12Du
static __device__ __forceinline__ int aik_dev_level_c(const uint32_t state, const uint32_t C) {
    const uint32_t x = state * C;
    return (int) ((__dp4a(x, 0x01010101u, 2u)) >> 2) - 128;
}

static __device__ __forceinline__ int ai4_dev_level(const uint32_t state) {
    return aik_dev_level_c(state, AI4_DEV_MUL);
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

// Reconstruct (original basis) W[M][K] row-major from AI4 bytes. Block = 512 threads = 16 warps = the 16 rows of
// one slab; grid = (K/256, M/16). Lane l of warp r decodes row r, K columns hb*256 + l*8 .. +7 (= one tile column
// step), then the warp unrotates the 256 block.
template <typename T>
static __global__ void __launch_bounds__(512) k_ai4_reconstruct(const uint8_t * __restrict__ src, T * __restrict__ W,
                                                               const int64_t K) {
    const int hb   = blockIdx.x;
    const int slab = blockIdx.y;
    const int r    = threadIdx.x >> 5;
    const int l    = threadIdx.x & 31;
    const int64_t nsb = K / 1024;
    const uint8_t * blk = src + ((int64_t) slab * nsb + hb / 4) * AI4_DEV_SB_BYTES;
    const uint16_t  dh  = ((const uint16_t *) blk)[r * 8 + (hb % 4) * 2 + (l >> 4)];
    const float     d   = __half2float(__ushort_as_half(dh));
    const uint32_t * tile = (const uint32_t *) (blk + AI4_DEV_SCALE_BYTES) + ((hb % 4) * 32 + l) * 16;
    const uint64_t c = ((uint64_t) tile[(r + 15) & 15] << 32) | tile[r];
    float v[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const uint32_t st = (uint32_t) (c >> (4 * (7 - q))) & 0xFFFFu;
        v[q] = d * (float) ai4_dev_level(st);
    }
    ai4_dev_fwht256(v);
    T * out = W + ((int64_t) slab * 16 + r) * K + (int64_t) hb * 256 + l * 8;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const float s = ai4_dev_sign_neg(l * 8 + q) ? -1.0f / 16.0f : 1.0f / 16.0f;
        if constexpr (sizeof(T) == 2) {
            out[q] = __float2half(v[q] * s);
        } else {
            out[q] = v[q] * s;
        }
    }
}

template <typename T>
static void ai4_reconstruct_launch(const void * src, T * W, const int64_t K, const int64_t M, cudaStream_t stream) {
    const dim3 grid((unsigned) (K / 256), (unsigned) (M / 16));
    k_ai4_reconstruct<T><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K);
}

// AI4_U reconstruct (original basis): unit-128 layout, rotation block RB = 256 (K % 256 == 0) or 128. Block = 512 threads
// = 16 warps = the 16 rows of one slab, grid = (ceil(K/256), M/16); lane l of warp r decodes unit 2*blockIdx.x + l/16,
// column step l%16 (8 K columns), then the warp (RB = 256) or each half-warp (RB = 128) unrotates its block.
template <typename T, int RB>
static __global__ void __launch_bounds__(512) k_ai4u_reconstruct(const uint8_t * __restrict__ src, T * __restrict__ W,
                                                                const int64_t K) {
    const int r = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    const int64_t nu = K / 128;
    const int64_t u  = (int64_t) blockIdx.x * 2 + (l >> 4);
    const bool live  = u < nu;
    float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (live) {
        const uint8_t * blk = src + ((int64_t) blockIdx.y * nu + u) * 1056;
        const float d = __half2float(__ushort_as_half(((const uint16_t *) blk)[r]));
        const uint32_t * tile = (const uint32_t *) (blk + 32) + (l & 15) * 16;
        const uint64_t c = ((uint64_t) tile[(r + 15) & 15] << 32) | tile[r];
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const uint32_t st = (uint32_t) (c >> (4 * (7 - q))) & 0xFFFFu;
            v[q] = d * (float) ai4_dev_level(st);
        }
    }
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
    if (!live) {
        return;
    }
    const float nrm = RB == 256 ? 1.0f / 16.0f : 0.08838834764831845f;
    T * out = W + ((int64_t) blockIdx.y * 16 + r) * K + u * 128 + (l & 15) * 8;
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const float s = ai4_dev_sign_neg((l * 8 + q) % RB) ? -nrm : nrm;
        if constexpr (sizeof(T) == 2) {
            out[q] = __float2half(v[q] * s);
        } else {
            out[q] = v[q] * s;
        }
    }
}

template <typename T>
static void ai4u_reconstruct_launch(const void * src, T * W, const int64_t K, const int64_t M, cudaStream_t stream) {
    const dim3 grid((unsigned) ((K / 128 + 1) / 2), (unsigned) (M / 16));
    if (K % 256 == 0) {
        k_ai4u_reconstruct<T, 256><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K);
    } else {
        k_ai4u_reconstruct<T, 128><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K);
    }
}

// AI+ unit layout at rate KB (AI5_U: KB = 5; ggml-ai4.h "AI+ unit layout at rate k"). Tile = 128*KB-bit MSB-first stream in
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

// Reconstruct W[M][K] from a rate-KB unit tensor. UNROT = true: original basis (as k_ai4u_reconstruct); false: rotated
// basis (the values the IMMA path multiplies with rotated activations). Same thread layout as k_ai4u_reconstruct.
// SPAN = true: the AIk_S span layout (16 rows x 1024 K blocks of 32 + 2048*KB bytes, one scale per row and span, the
// 8 units' tiles back to back) instead of AIk_U's unit-128 blocks; everything else is identical.
template <typename T, int KB, int RB, bool UNROT, bool SPAN = false>
static __global__ void __launch_bounds__(512) k_aik_reconstruct(const uint8_t * __restrict__ src, T * __restrict__ W,
                                                               const int64_t K, const uint32_t C) {
    constexpr int UB = 32 + 256 * KB;
    constexpr int SB = 32 + 2048 * KB;
    const int r = threadIdx.x >> 5;
    const int l = threadIdx.x & 31;
    const int64_t nu = K / 128;
    const int64_t u  = (int64_t) blockIdx.x * 2 + (l >> 4);
    const bool live  = u < nu;
    float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (live) {
        const uint8_t  * blk  = SPAN ? src + ((int64_t) blockIdx.y * (nu >> 3) + (u >> 3)) * SB
                                     : src + ((int64_t) blockIdx.y * nu + u) * UB;
        const uint32_t * tile = (const uint32_t *) (blk + 32) + (SPAN ? (int) (u & 7) * 16 + (l & 15) : (l & 15)) * (4 * KB);
        const float d = __half2float(__ushort_as_half(((const uint16_t *) blk)[r]));
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            v[q] = d * (float) aik_dev_level_c(aik_dev_state<KB>(tile, r * 8 + q), C);
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

template <typename T, int KB, bool SPAN = false>
static void aik_reconstruct_launch(const void * src, T * W, const int64_t K, const int64_t M, const bool unrot, cudaStream_t stream,
                                   const uint32_t C = AI4_DEV_MUL) {
    const dim3 grid((unsigned) ((K / 128 + 1) / 2), (unsigned) (M / 16));
    if (K % 256 == 0) {
        if (unrot) k_aik_reconstruct<T, KB, 256, true , SPAN><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C);
        else       k_aik_reconstruct<T, KB, 256, false, SPAN><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C);
    } else if constexpr (!SPAN) {   // the span layout requires K % 1024 == 0
        if (unrot) k_aik_reconstruct<T, KB, 128, true ><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C);
        else       k_aik_reconstruct<T, KB, 128, false><<<grid, 512, 0, stream>>>((const uint8_t *) src, W, K, C);
    }
}

// Reference reconstruct at any implemented AI+ rate (2, 3, 4, 5, 6). span: the AIk_S layout. Returns false for an
// unknown rate (or a span tensor whose K is not a multiple of 1024).
template <typename T>
static bool aik_reconstruct_rate(const int rate, const void * src, T * W, const int64_t K, const int64_t M, const bool unrot,
                                 cudaStream_t stream, const uint32_t C, const bool span = false) {
    if (span) {
        if (K % 1024 != 0) {
            return false;
        }
        switch (rate) {
            case 2: aik_reconstruct_launch<T, 2, true>(src, W, K, M, unrot, stream, C); return true;
            case 3: aik_reconstruct_launch<T, 3, true>(src, W, K, M, unrot, stream, C); return true;
            case 4: aik_reconstruct_launch<T, 4, true>(src, W, K, M, unrot, stream, C); return true;
            case 5: aik_reconstruct_launch<T, 5, true>(src, W, K, M, unrot, stream, C); return true;
            case 6: aik_reconstruct_launch<T, 6, true>(src, W, K, M, unrot, stream, C); return true;
            default: return false;
        }
    }
    switch (rate) {
        case 2: aik_reconstruct_launch<T, 2>(src, W, K, M, unrot, stream, C); return true;
        case 3: aik_reconstruct_launch<T, 3>(src, W, K, M, unrot, stream, C); return true;
        case 4: aik_reconstruct_launch<T, 4>(src, W, K, M, unrot, stream, C); return true;
        case 5: aik_reconstruct_launch<T, 5>(src, W, K, M, unrot, stream, C); return true;
        case 6: aik_reconstruct_launch<T, 6>(src, W, K, M, unrot, stream, C); return true;
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
