#pragma once
// INKNVFP4K Q8TC: dense Q8_0 decode/verify matmul at T = 4..8 columns on f16 tensor cores (mma.m16n8k16, f32 accumulate, sm_80+).
// Weights stay in the stock 34-byte block_q8_0 layout (no repack) and convert to f16 exactly in registers; the f32
// activation is rounded to f16 per fragment (HILO: plus its f16 residual through a second mma, ~f32 precision). So the
// only rounding beyond the f32-activation Q8A16 path is the f16 activation (HILO: none at f32 level) and the order of the
// f32 sums: no activation quantization kernel, no q8_1.
//
// Staging: a chunk = 8 blocks = 4 block pairs = 272 bytes per row, 16-byte aligned when the row stride and base are. A warp
// copies the 16 rows of its tile (272 cp.async of 16 bytes over the 32 lanes, fully coalesced) into its own shared buffer,
// row stride 68 words, so rows gid and gid + 8 of one mma read banks 4*gid + tig (+ const), conflict-free. In a block pair
// (17 words) words 1..8 shifted by 2 bytes hold the even block's int8 (one byte_perm per word), words 9..16 the odd block's
// aligned; the scales are the low half of word 0 and the high half of word 8.
// CTA: one 16-row tile x NW warps; warp w takes chunks w, w + NW, ...; the NW partial 16 x 8 tiles are summed in shared
// memory in fixed warp order (deterministic). Grid: the INKGLUE member layout of q8a16_args (block b runs member k's 16-row
// tile b - blk_end[k-1]); every member reads its own x.

#include "common.cuh"
#include "q8a16-kern.cuh"

#define Q8TC_CHUNK_BYTES 272
static __device__ __forceinline__ void q8tc_cp16(void * smem, const void * gmem) {
#ifdef AMPERE_MMA_AVAILABLE
    const uint32_t sa = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(gmem));
#else
    GGML_UNUSED_VARS(smem, gmem);
#endif
}

// v3: f16 tensor cores (mma.m16n8k16, f32 accumulate), no activation quantization kernel. Weights int8 -> f16 exactly in
// registers (0x6400 | (q ^ 0x80) is 1024 + 128 + q; one byte_perm + one HSUB2 per 2 weights). The f32 activation is read
// per fragment (float4, L1/L2) and rounded to f16 (HILO: plus the f16 residual in a second mma, ~f32 precision).
// Same k freedom as v2: for k step s of a block, thread tig uses int8 word P = tig + 4*s of the block, bytes 0,1 -> a0/b0
// (k = 2*tig, +1) and bytes 2,3 -> a2/b1 (k = 8 + 2*tig, +1), for both the weights and the activation.
// CTA: 16 rows x NW warps, warp w takes chunks w, w + NW, ... (single stage); dynamic shared memory NW*16*68 words.
static __device__ __forceinline__ void q8tc_mma16(float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2,
        const uint32_t a3, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// int8 word q -> (f16x2 of bytes 0,1, f16x2 of bytes 2,3), exact
static __device__ __forceinline__ void q8tc_i8_f16(const uint32_t q, uint32_t & lo, uint32_t & hi) {
    const uint32_t u = q ^ 0x80808080u;
    const half2 m = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));   // 1152 = 1024 + 128
    const uint32_t pl = __byte_perm(u, 0x64646464u, 0x4140), ph = __byte_perm(u, 0x64646464u, 0x4342);
    half2 l = __hsub2(*(const half2 *) &pl, m);
    half2 h = __hsub2(*(const half2 *) &ph, m);
    lo = *(uint32_t *) &l;
    hi = *(uint32_t *) &h;
}

static __device__ __forceinline__ uint32_t q8tc_h2(const float a, const float b) {
    half2 h = __floats2half2_rn(a, b);
    return *(uint32_t *) &h;
}

template <int NW, bool HILO, int MINB>
static __global__ void __launch_bounds__(32*NW, MINB) k_q8tc3(const q8a16_args a, const int nb, const int T) {
#ifdef AMPERE_MMA_AVAILABLE
    extern __shared__ __align__(16) uint32_t sw3[];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gid = lane >> 2, tig = lane & 3;
    int k = 0;
    while (k < a.n - 1 && blockIdx.x >= a.blk_end[k]) {
        ++k;
    }
    const int tile = blockIdx.x - (k == 0 ? 0 : a.blk_end[k - 1]);
    const int nr = a.nrows[k];
    const char * vx = a.vx[k] + (int64_t) tile*16*a.srow[k];
    const int64_t srow = a.srow[k];
    const int rmax = nr - 1 - tile*16;
    const int nch = nb/8;
    const bool colok = gid < T;
    const float * xc = a.x[k] + (int64_t) (colok ? gid : 0)*a.sx[k] + 4*tig;
    uint32_t * S = sw3 + w*16*68;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int ch = w; ch < nch; ch += NW) {
#pragma unroll
        for (int i0 = 0; i0 < 9; ++i0) {
            const int i = i0*32 + lane;
            if (i < 16*17) {
                const int r = i / 17, c = i - r*17;
                q8tc_cp16(&S[r*68 + c*4], vx + (int64_t) min(r, rmax)*srow + (int64_t) ch*Q8TC_CHUNK_BYTES + c*16);
            }
        }
        asm volatile("cp.async.commit_group;");
        // activation fragments, words tig and tig + 4 of each block: pair 0 while the copies fly, pair j + 1 during pair j
        auto ldx = [&](float4 (&v)[2][2], const int j) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const float * xb = xc + (8*ch + 2*j + e)*QK8_0;
                v[e][0] = colok ? *(const float4 *) xb        : make_float4(0.f, 0.f, 0.f, 0.f);
                v[e][1] = colok ? *(const float4 *) (xb + 16) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        };
        float4 xv[2][2][2];
        ldx(xv[0], 0);
        asm volatile("cp.async.wait_group 0;");
        __syncwarp();
        const uint32_t * S0 = S + gid*68;
        const uint32_t * S1 = S + (gid + 8)*68;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < 3) {
                ldx(xv[(j + 1) & 1], j + 1);
            }
            const uint32_t * W0 = S0 + 17*j;
            const uint32_t * W1 = S1 + 17*j;
            uint32_t q0[2][2], q1[2][2];   // [block even/odd][word tig / tig + 4]
            q0[0][0] = __byte_perm(W0[tig], W0[tig + 1], 0x5432); q0[0][1] = __byte_perm(W0[tig + 4], W0[tig + 5], 0x5432);
            q1[0][0] = __byte_perm(W1[tig], W1[tig + 1], 0x5432); q1[0][1] = __byte_perm(W1[tig + 4], W1[tig + 5], 0x5432);
            q0[1][0] = W0[9 + tig]; q0[1][1] = W0[13 + tig];
            q1[1][0] = W1[9 + tig]; q1[1][1] = W1[13 + tig];
            const float dw0[2] = { __half2float(((const half *) W0)[0]), __half2float(((const half *) W0)[17]) };
            const float dw1[2] = { __half2float(((const half *) W1)[0]), __half2float(((const half *) W1)[17]) };
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int st = 0; st < 2; ++st) {
                    uint32_t a0, a2, a1, a3;
                    q8tc_i8_f16(q0[e][st], a0, a2);
                    q8tc_i8_f16(q1[e][st], a1, a3);
                    const float4 v = xv[j & 1][e][st];
                    const uint32_t b0 = q8tc_h2(v.x, v.y), b1 = q8tc_h2(v.z, v.w);
                    q8tc_mma16(c, a0, a1, a2, a3, b0, b1);
                    if constexpr (HILO) {
                        const half2 h0 = *(const half2 *) &b0, h1 = *(const half2 *) &b1;
                        const uint32_t l0 = q8tc_h2(v.x - __low2float(h0), v.y - __high2float(h0));
                        const uint32_t l1 = q8tc_h2(v.z - __low2float(h1), v.w - __high2float(h1));
                        q8tc_mma16(c, a0, a1, a2, a3, l0, l1);
                    }
                }
                acc[0] += c[0]*dw0[e];
                acc[1] += c[1]*dw0[e];
                acc[2] += c[2]*dw1[e];
                acc[3] += c[3]*dw1[e];
            }
        }
        __syncwarp();
    }

    if constexpr (NW > 1) {
        __syncthreads();
        float * red = (float *) sw3;   // the chunk buffers are free now: [NW][32][4]
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            red[(w*32 + lane)*4 + i] = acc[i];
        }
        __syncthreads();
        if (w > 0) {
            return;
        }
        for (int v = 1; v < NW; ++v) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] += red[(v*32 + lane)*4 + i];
            }
        }
    }
    float * dst = a.dst[k];
    const int64_t sd = a.sd[k];
    const int r0 = tile*16 + gid, r1 = r0 + 8, c0 = 2*tig, c1 = 2*tig + 1;
    if (r0 < nr) {
        if (c0 < T) { dst[c0*sd + r0] = acc[0]; }
        if (c1 < T) { dst[c1*sd + r0] = acc[1]; }
    }
    if (r1 < nr) {
        if (c0 < T) { dst[c0*sd + r1] = acc[2]; }
        if (c1 < T) { dst[c1*sd + r1] = acc[3]; }
    }
#else
    GGML_UNUSED_VARS(a, nb, T);
    NO_DEVICE_CODE;
#endif
}
