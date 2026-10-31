#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace magic2 {
template<int B, int V> constexpr int tile_words = 4 * B / V;
template<int B, int V> constexpr int unit_bytes = 32 + 64 * tile_words<B, V>;

template<uint32_t C>
static __device__ __forceinline__ uint32_t level_byte(uint32_t state) {
    return __dp4a(state * C, 0x40404040u, 128u - 32768u);
}

template<int B, int V, uint32_t C>
static __device__ __forceinline__ uint32_t fragment(const uint32_t * tile, int row, int half) {
    static_assert(V == 1 || V == 2, "computed scalar or pair");
    static_assert(4 * B % V == 0 && B * (4 / V - 1) + 16 <= 32, "four weights fit one window");
    constexpr int bits = 128 * B / V, words = tile_words<B, V>;
    int start = B * ((8 * row + 4 * half + 4) / V) - 32;
    if (start < 0) start += bits;
    const int word = start >> 5, next = word + 1 == words ? 0 : word + 1;
    const uint32_t f = __funnelshift_l(tile[next], tile[word], start & 31);
    uint32_t q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t state = (f >> (B * ((3 - j) / V))) & 0xffffu;
        if constexpr (V == 2) {
            q[j] = (j & 1) ? level_byte<uint32_t(uint64_t(C) * C)>(state) : level_byte<C>(state);
        } else {
            q[j] = level_byte<C>(state);
        }
    }
    return __byte_perm(__byte_perm(q[0], q[1], 0x0051), __byte_perm(q[2], q[3], 0x0051), 0x5410);
}

static __device__ __forceinline__ void mma(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Rotated, quantized activations; per-token scales are indexed by 128-column unit.
template<int B, int V, uint32_t C, int XSH = 0>
static __global__ void __launch_bounds__(128) mmv(const uint8_t * weights, const int8_t * x,
        const float * xs, float * y, int M, int K, int N) {
    static_assert(XSH == 0 || XSH == 1, "activation scale covers 128 or 256 columns");
    constexpr int UW = unit_bytes<B, V> / 4, TW = tile_words<B, V>;
    __shared__ uint32_t tiles[4][UW];
    __shared__ float partial[4][8][16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = lane >> 2, col = lane & 3, nu = K / 128;
    const int64_t nbase = int64_t(blockIdx.y) * 8, mbase = int64_t(blockIdx.x) * 16;
    float sum[4] = {0, 0, 0, 0};
    for (int u = warp; u < nu; u += 4) {
        const uint32_t * source = reinterpret_cast<const uint32_t *>(weights) + (int64_t(blockIdx.x) * nu + u) * UW;
        for (int i = lane; i < UW; i += 32) tiles[warp][i] = __ldg(source + i);
        __syncwarp();
        int c[4] = {0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t * a0 = tiles[warp] + 8 + (4 * j + (col >> 1)) * TW;
            const uint32_t * a1 = a0 + 2 * TW;
            const uint32_t a[4] = {fragment<B,V,C>(a0, row, col & 1), fragment<B,V,C>(a0, row + 8, col & 1),
                fragment<B,V,C>(a1, row, col & 1), fragment<B,V,C>(a1, row + 8, col & 1)};
            uint32_t b0 = 0, b1 = 0;
            if (nbase + row < N) {
                const int8_t * p = x + (nbase + row) * K + int64_t(u) * 128 + j * 32 + col * 4;
                b0 = __ldg(reinterpret_cast<const uint32_t *>(p));
                b1 = __ldg(reinterpret_cast<const uint32_t *>(p + 16));
            }
            mma(c, a, b0, b1);
        }
        const __half * scales = reinterpret_cast<const __half *>(tiles[warp]);
        const float d0 = __half2float(scales[row]), d1 = __half2float(scales[row + 8]);
        const float s0 = nbase + 2 * col < N ? xs[(nbase + 2 * col) * (nu >> XSH) + (u >> XSH)] : 0;
        const float s1 = nbase + 2 * col + 1 < N ? xs[(nbase + 2 * col + 1) * (nu >> XSH) + (u >> XSH)] : 0;
        sum[0] += c[0] * d0 * s0; sum[1] += c[1] * d0 * s1;
        sum[2] += c[2] * d1 * s0; sum[3] += c[3] * d1 * s1;
        __syncwarp();
    }
    partial[warp][2 * col][row] = sum[0]; partial[warp][2 * col + 1][row] = sum[1];
    partial[warp][2 * col][row + 8] = sum[2]; partial[warp][2 * col + 1][row + 8] = sum[3];
    __syncthreads();
    const int n = threadIdx.x / 16, r = threadIdx.x % 16;
    if (nbase + n < N) y[(nbase + n) * M + mbase + r] =
        (partial[0][n][r] + partial[1][n][r]) + (partial[2][n][r] + partial[3][n][r]);
}
}
