#pragma once
// INKNVFP4K NVFP4PF: NVFP4 MoE prefill (MUL_MAT_ID, >= 512 tokens) on int8 tensor cores (mma m16n8k32 s8), sm_80 / sm_86.
// Port of the S8 MoE engine of exp/ampere-nextgen 82183ff0b (Gemini, GEMINIGATE-reviewed), same math:
//   - x rows -> int8 per KF (64 or 32) values with one f32 scale (amax / 127)
//   - each weight row's 36-byte NVFP4 block (64 weights, 4 UE4M3 sub-scales) -> int8 against the largest sub-scale of
//     its KF span: w8 = round(e2m1 * sub / max * 127 / 12) through a 4096-step fixed-point table, block scale max * 12 / 127
//   - acc += i32(dot over KF) * (w scale * x scale), f32, in KF order; out = acc [* per-expert / per-tensor scale]
// Not bit-exact with MMQ (q8_1 per 32 against the exact FP4 values): gate on KLD. Non-finite x gives a NaN block scale,
// so a NaN / Inf input still poisons its outputs (the original zeroed them).
// Routing runs on the device (one CTA builds expert -> column lists), so there is no host sync and CUDA graphs keep working.
// No weight scratch: a pair of warps decodes each 16-row slab of a 128-wide K step into shared memory.

#include "common.cuh"

#define NVFP4PF_MAX_EXPERTS 512

// device-side routing tables in one int buffer (sizes in nvfp4pf_work_ints)
struct nvfp4pf_map {
    const int * hdr;    // [0] active experts, [1] tiles
    const int * exps;   // [n_expert] active expert ids
    const int * off;    // [n_expert + 1] start of each active expert's pairs
    const int * rows;   // [n_pairs] x row of each pair
    const int * cols;   // [n_pairs] output column of each pair
    const int * tiles;  // [2 * max_tiles] (active expert, first pair in it)
    int64_t     nb02;   // bytes between experts in W
};

static inline int nvfp4pf_max_tiles(const int n_expert, const int64_t n_pairs, const int bn) {
    return (int) (n_expert + (n_pairs + bn - 1)/bn);
}

static inline size_t nvfp4pf_work_ints(const int n_expert, const int64_t n_pairs, const int max_tiles) {
    return (size_t) (2 + n_expert + (n_expert + 1) + 2*n_pairs + 2*max_tiles);
}

static inline nvfp4pf_map nvfp4pf_map_of(const int * work, const int n_expert, const int64_t n_pairs, const int64_t nb02) {
    nvfp4pf_map m;
    m.hdr   = work;
    m.exps  = work + 2;
    m.off   = m.exps + n_expert;
    m.rows  = m.off + n_expert + 1;
    m.cols  = m.rows + n_pairs;
    m.tiles = m.cols + n_pairs;
    m.nb02  = nb02;
    return m;
}

static constexpr float NVFP4PF_MIN_SCALE = 1e-30f;  // smaller block maxima count as zero (no 1 / subnormal = Inf)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define NVFP4PF_NO_SM80 1  // never launched there (ggml_cuda_nvfp4pf_try checks cc), the bodies only have to compile
#endif

static __device__ __forceinline__ void nvfp4pf_cp16(void * s, const void * g, const bool pred) {
#ifdef NVFP4PF_NO_SM80
    *(int4 *) s = pred ? *(const int4 *) g : make_int4(0, 0, 0, 0);
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(g), "r"(pred ? 16 : 0) : "memory");
#endif
}
static __device__ __forceinline__ void nvfp4pf_cp4(void * s, const void * g, const bool pred) {
#ifdef NVFP4PF_NO_SM80
    *(float *) s = pred ? *(const float *) g : 0.0f;
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sa), "l"(g), "r"(pred ? 4 : 0) : "memory");
#endif
}
static __device__ __forceinline__ void nvfp4pf_commit() {
#ifndef NVFP4PF_NO_SM80
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
template <int N>
static __device__ __forceinline__ void nvfp4pf_wait() {
#ifndef NVFP4PF_NO_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}
// barrier of the 64 threads (2 warps) of one slab
static __device__ __forceinline__ void nvfp4pf_bar64(const int id) {
#ifdef NVFP4PF_NO_SM80
    (void) id;
    __syncthreads();
#else
    asm volatile("bar.sync %0, 64;\n" :: "r"(id) : "memory");
#endif
}
static __device__ __forceinline__ void nvfp4pf_mma(int (&d)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#ifdef NVFP4PF_NO_SM80
    (void) d; (void) a; (void) b0; (void) b1;
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// UE4M3 -> f32 without a table (ggml_cuda_ue4m3_to_fp32 values, the 0.5 factor included; 0x7F -> 0)
static __device__ __forceinline__ float nvfp4pf_ue4m3(const uint32_t u) {
    const uint32_t x  = u & 0x7Fu;
    const bool     sn = x < 8u;
    const float    v  = __fsub_rn(__uint_as_float((sn ? 0x3C000000u : 0x3B800000u) + (x << 20)), sn ? 0.0078125f : 0.0f);
    return u == 0x7Fu ? 0.0f : v;
}
// exact int <-> float for |x| < 2^22 (no I2F / F2I)
static __device__ __forceinline__ float nvfp4pf_i2f(const int x) {
    return __fsub_rn(__int_as_float(x + 0x4B400000), 12582912.0f);
}
static __device__ __forceinline__ int nvfp4pf_f2i(const float x) {
    return __float_as_int(__fadd_rn(x, 12582912.0f)) - 0x4B400000;
}
// 1 / a in two Newton steps, 0 below NVFP4PF_MIN_SCALE
static __device__ __forceinline__ float nvfp4pf_rcp(const float a) {
    float y = __uint_as_float(0x7EF311C2u - __float_as_uint(a));
    y = y*fmaf(-a, y, 2.0f);
    y = y*fmaf(-a, y, 2.0f);
    return a >= NVFP4PF_MIN_SCALE ? y : 0.0f;
}

// int8 table of the 8 positive E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} * 2 scaled by sub / max: R = sub * inv in
// 1/4096 steps (inv = 4096 * 127/12 / max), entry v = round(m * R / 4096); l2 / l3 = the negated bytes
static __device__ __forceinline__ void nvfp4pf_lut(const float sub, const float inv, uint32_t & l0, uint32_t & l1,
        uint32_t & l2, uint32_t & l3) {
    const uint32_t R  = (uint32_t) min(max(nvfp4pf_f2i(sub*inv), 0), 43349);
    const uint32_t R3 = 2*R + R, R6 = R3 << 1;
    const uint32_t v1 = (R + 2048u) >> 12, v2 = (2*R + 2048u) >> 12, v3 = (R3 + 2048u) >> 12, v4 = (4*R + 2048u) >> 12;
    const uint32_t v6 = (R6 + 2048u) >> 12, v8 = (8*R + 2048u) >> 12, v12 = ((R6 << 1) + 2048u) >> 12;
    l0 = (v1 << 8) | (v2 << 16) | (v3 << 24);
    l1 = v4 | (v6 << 8) | (v8 << 16) | (v12 << 24);
    l2 = (0x80808080u - l0) ^ 0x80808080u;
    l3 = (0x80808080u - l1) ^ 0x80808080u;
}
// 8 E2M1 nibbles -> 8 int8 through the table: .x = even nibbles, .y = odd nibbles
static __device__ __forceinline__ int2 nvfp4pf_dec8(const uint32_t q4, const uint32_t l0, const uint32_t l1,
        const uint32_t l2, const uint32_t l3) {
    const uint32_t idx = q4 & 0x77777777u;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t m0  = __byte_perm(__byte_perm(l0, l1, idx), __byte_perm(l2, l3, idx), sel);
    const uint32_t m1  = __byte_perm(__byte_perm(l0, l1, idx >> 16), __byte_perm(l2, l3, idx >> 16), sel >> 16);
    return make_int2((int) __byte_perm(m0, m1, 0x6420u), (int) __byte_perm(m0, m1, 0x7531u));
}

// routing: one CTA of 256 threads. Counts the pairs (slot u of token t, valid expert) per expert, lays the active
// experts out in expert order with BN-pair tiles, fills rows / cols (order inside an expert is atomic-arrival order;
// every output column is computed alone, so that order does not change any value). Pairs with an invalid id get 0.
static __global__ void __launch_bounds__(256) k_nvfp4pf_route(const char * __restrict__ ids, const int64_t ids_nb0,
        const int64_t ids_nb1, const int n_used, const int n_tok, const int ne11, const int n_expert, const int bn,
        const int max_tiles, int * __restrict__ work, float * __restrict__ y, const int m) {
    __shared__ int cnt[NVFP4PF_MAX_EXPERTS];
    __shared__ int pos[NVFP4PF_MAX_EXPERTS];
    const int tid = threadIdx.x;
    const int64_t n_pairs = (int64_t) n_used*n_tok;
    int * hdr   = work;
    int * exps  = work + 2;
    int * off   = exps + n_expert;
    int * rows  = off + n_expert + 1;
    int * cols  = rows + n_pairs;
    int * tiles = cols + n_pairs;
    auto id_of = [&](const int64_t p) {
        return *(const int32_t *) (ids + (p % n_used)*ids_nb0 + (p / n_used)*ids_nb1);
    };

    for (int e = tid; e < n_expert; e += 256) {
        cnt[e] = 0;
    }
    __syncthreads();
    for (int64_t p = tid; p < n_pairs; p += 256) {
        const int e = id_of(p);
        if (e >= 0 && e < n_expert) {
            atomicAdd(&cnt[e], 1);
        }
    }
    __syncthreads();
    if (tid == 0) {
        int na = 0, o = 0, nt = 0;
        for (int e = 0; e < n_expert; ++e) {
            const int c = cnt[e];
            pos[e] = c > 0 ? o : -1;
            if (c == 0) {
                continue;
            }
            exps[na] = e;
            off[na]  = o;
            for (int t0 = 0; t0 < c; t0 += bn, ++nt) {
                if (nt < max_tiles) {
                    tiles[2*nt + 0] = na;
                    tiles[2*nt + 1] = t0;
                }
            }
            o += c;
            ++na;
        }
        off[na] = o;
        hdr[0]  = na;
        hdr[1]  = min(nt, max_tiles);
    }
    __syncthreads();
    for (int64_t p = tid; p < n_pairs; p += 256) {
        const int e = id_of(p);
        if (e >= 0 && e < n_expert) {
            const int i = atomicAdd(&pos[e], 1);
            rows[i] = (int) (p / n_used)*ne11 + (int) (p % n_used) % ne11;
            cols[i] = (int) p;
        } else {
            for (int r = 0; r < m; ++r) {
                y[p*m + r] = 0.0f;
            }
        }
    }
}

// x [n_rows][K] f32 -> xq [n_rows][K] int8 + xs [n_rows][K/KF] f32 (amax / 127). grid (K/KF/4, n_rows), 128 threads.
// A non-finite value in a block gives its scale NaN (outputs that use it become NaN, as with MMQ).
template <int KF>
static __global__ void __launch_bounds__(128) k_nvfp4pf_xq(const float * __restrict__ x, const int64_t ldx,
        int8_t * __restrict__ xq, float * __restrict__ xs, const int K, const int n_rows) {
    static_assert(KF == 32 || KF == 64, "NVFP4PF: KF 32 or 64");
    constexpr int V = KF/32;  // values per lane
    const int lane = threadIdx.x & 31;
    const int kb = blockIdx.x*4 + (threadIdx.x >> 5), r = blockIdx.y, nb = K/KF;
    if (kb >= nb || r >= n_rows) {
        return;
    }
    const float * xp = x + (int64_t) r*ldx + (int64_t) kb*KF + V*lane;
    float v[V];
    bool bad = false;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        v[i] = xp[i];
        bad = bad || !isfinite(v[i]);
        v[i] = isfinite(v[i]) ? v[i] : 0.0f;
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o, 32));
    }
    bad = __any_sync(0xffffffff, bad);
    const bool  ok  = amax >= NVFP4PF_MIN_SCALE;
    const float inv = ok ? 127.0f/amax : 0.0f;
    int8_t * q = xq + (int64_t) r*K + (int64_t) kb*KF + V*lane;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        q[i] = (int8_t) max(-127, min(127, nvfp4pf_f2i(v[i]*inv)));
    }
    if (lane == 0) {
        xs[(int64_t) r*nb + kb] = bad ? __int_as_float(0x7fc00000) : ok ? amax*(1.0f/127.0f) : 0.0f;
    }
}

template <int KF, int BN>
constexpr __host__ __device__ int nvfp4pf_stage_bytes() {
    return BN*128 + BN*(128/KF)*4;  // x int8 of 128 K per column + its scales
}
template <int KF, int BN>
constexpr __host__ __device__ int nvfp4pf_smem_bytes() {
    return 2*nvfp4pf_stage_bytes<KF, BN>() + 4*4*32*16 + 4*(128/KF)*16*4;  // 2 stages + decoded A + weight scales
}

// y[col][r] = sum_k w[e][r][k] * x[row][k] (* scale), col / row / e from the routing. grid (M/64, max tiles), 256 threads:
// 4 slabs of 16 weight rows (warp pair mw), each pair splits a tile's BN columns (nw); per 128-wide K step the pair decodes
// its slab (nw 0: block 0, nw 1: block 1) into shared memory, then both run the mma over their columns. x streams through
// a 2-stage cp.async ring (128-byte column rows, 16-byte pieces xor-swizzled by column).
// s_exp: per-expert output scale [n_expert] or nullptr; s_all: per-tensor output scale (1 = none)
template <int KF, int BN>
static __global__ void __launch_bounds__(256, 2) k_nvfp4pf_mm(const char * __restrict__ w_all, const int64_t stride_blk,
        const int8_t * __restrict__ xq, const float * __restrict__ xs, float * __restrict__ y, const int64_t ldy,
        const int M, const int K, const nvfp4pf_map map, const float * __restrict__ s_exp, const float s_all) {
    static_assert(KF == 32 || KF == 64, "NVFP4PF: KF 32 or 64");
    static_assert(BN == 64 || BN == 128, "NVFP4PF: BN 64 or 128");
    constexpr int WN = BN/2, NTI = WN/8, SUBS = 128/KF, JPS = KF/32, STAGE = nvfp4pf_stage_bytes<KF, BN>();
    extern __shared__ __align__(16) uint8_t smem_pf[];
    __shared__ int srow[BN];
    __shared__ int scol[BN];

    if ((int) blockIdx.y >= map.hdr[1]) {
        return;  // CTA-uniform, before any barrier
    }
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int mw = warp >> 1, nw = warp & 1, g = lane >> 2, t = lane & 3;
    const int nk = K >> 7, nbx = K/KF;
    const int act = map.tiles[2*blockIdx.y + 0], toff = map.tiles[2*blockIdx.y + 1], off = map.off[act];
    const int nt = min(BN, map.off[act + 1] - off - toff);
    const int e  = map.exps[act];
    const char * W = w_all + (int64_t) e*map.nb02;
    const float oscale = s_exp ? s_exp[e]*s_all : s_all;

    for (int i = tid; i < BN; i += 256) {
        srow[i] = i < nt ? map.rows[off + toff + i] : 0;
        scol[i] = i < nt ? map.cols[off + toff + i] : 0;
    }
    __syncthreads();

    uint4 * adec = (uint4 *) (smem_pf + 2*STAGE) + mw*128;                     // [4 steps][32 lanes] decoded slab
    float * wsc  = (float *) (smem_pf + 2*STAGE + 4*4*32*16) + mw*(SUBS*16);   // [SUBS][16 rows] weight scales

    auto load = [&](const int kc, const int slot) {
        uint8_t * st = smem_pf + slot*STAGE;
#pragma unroll
        for (int it = 0; it < BN*8/256; ++it) {
            const int i = tid + it*256, c = i >> 3, p = i & 7;
            const bool ok = c < nt;
            nvfp4pf_cp16(st + c*128 + (p ^ (c & 7))*16, xq + (ok ? (int64_t) srow[c]*K + kc*128 + p*16 : 0), ok);
        }
        float * sx = (float *) (st + BN*128);
#pragma unroll
        for (int it = 0; it < (BN*SUBS + 255)/256; ++it) {
            const int i = tid + it*256;
            if ((BN*SUBS) % 256 == 0 || i < BN*SUBS) {
                const int c = i / SUBS, sb = i % SUBS;
                const bool ok = c < nt;
                nvfp4pf_cp4(sx + i, xs + (ok ? (int64_t) srow[c]*nbx + kc*SUBS + sb : 0), ok);
            }
        }
    };

    float acc[NTI][4];
#pragma unroll
    for (int i = 0; i < NTI; ++i) {
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    }
    load(0, 0);
    nvfp4pf_commit();

    const bool slab_ok = (blockIdx.x*4 + mw) < (M >> 4);
    const int  r0 = (blockIdx.x*4 + mw)*16 + g, r1 = r0 + 8;
    const int  ncols = nt - nw*WN;  // columns of this warp's half (may be <= 0)
    constexpr float QW = 4096.0f*(127.0f/12.0f);

    for (int kc = 0; kc < nk; ++kc) {
        nvfp4pf_wait<0>();
        __syncthreads();
        if (kc + 1 < nk) {
            load(kc + 1, (kc + 1) & 1);
        }
        nvfp4pf_commit();
        const uint8_t * st = smem_pf + (kc & 1)*STAGE;

        if (slab_ok) {
            const char * b0 = W + ((int64_t) r0*stride_blk + kc*2 + nw)*36;
            const char * b1 = W + ((int64_t) r1*stride_blk + kc*2 + nw)*36;
            const uint32_t d0 = __ldg((const uint32_t *) b0), d1 = __ldg((const uint32_t *) b1);
            float s0[4], s1[4];
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                s0[s] = nvfp4pf_ue4m3((d0 >> 8*s) & 0xFFu);
                s1[s] = nvfp4pf_ue4m3((d1 >> 8*s) & 0xFFu);
            }
            // span maxima: one per block (KF 64) or per half block (KF 32); h = half of the block
            float m0[2], m1[2];
            if constexpr (KF == 64) {
                m0[0] = m0[1] = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s0[2], s0[3]));
                m1[0] = m1[1] = fmaxf(fmaxf(s1[0], s1[1]), fmaxf(s1[2], s1[3]));
            } else {
                m0[0] = fmaxf(s0[0], s0[1]); m0[1] = fmaxf(s0[2], s0[3]);
                m1[0] = fmaxf(s1[0], s1[1]); m1[1] = fmaxf(s1[2], s1[3]);
            }
            if (t == 0) {
#pragma unroll
                for (int h = 0; h < SUBS/2; ++h) {
                    wsc[(nw*(SUBS/2) + h)*16 + g]     = m0[h] >= NVFP4PF_MIN_SCALE ? m0[h]*(12.0f/127.0f) : 0.0f;
                    wsc[(nw*(SUBS/2) + h)*16 + g + 8] = m1[h] >= NVFP4PF_MIN_SCALE ? m1[h]*(12.0f/127.0f) : 0.0f;
                }
            }
            const int  boff = 4 + (t & 1)*4;
            const bool hi   = t >= 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float i0 = QW*nvfp4pf_rcp(m0[h]), i1 = QW*nvfp4pf_rcp(m1[h]);
                uint32_t l0, l1, l2, l3;
                nvfp4pf_lut(s0[2*h], i0, l0, l1, l2, l3);
                const int2 a0 = nvfp4pf_dec8(__ldg((const uint32_t *) (b0 + boff + 16*h)), l0, l1, l2, l3);
                nvfp4pf_lut(s0[2*h + 1], i0, l0, l1, l2, l3);
                const int2 a1 = nvfp4pf_dec8(__ldg((const uint32_t *) (b0 + boff + 16*h + 8)), l0, l1, l2, l3);
                nvfp4pf_lut(s1[2*h], i1, l0, l1, l2, l3);
                const int2 c0 = nvfp4pf_dec8(__ldg((const uint32_t *) (b1 + boff + 16*h)), l0, l1, l2, l3);
                nvfp4pf_lut(s1[2*h + 1], i1, l0, l1, l2, l3);
                const int2 c1 = nvfp4pf_dec8(__ldg((const uint32_t *) (b1 + boff + 16*h + 8)), l0, l1, l2, l3);
                adec[(nw*2 + h)*32 + lane] = make_uint4(hi ? a0.y : a0.x, hi ? c0.y : c0.x, hi ? a1.y : a1.x, hi ? c1.y : c1.x);
            }
        }
        nvfp4pf_bar64(mw + 1);  // the pair's decoded slab and scales are in shared memory

        if (slab_ok && ncols > 0) {
            const float * sx = (const float *) (st + BN*128);
#pragma unroll
            for (int sb = 0; sb < SUBS; ++sb) {
                const float dw0 = wsc[sb*16 + g], dw1 = wsc[sb*16 + g + 8];
                uint4 av[JPS];
#pragma unroll
                for (int jj = 0; jj < JPS; ++jj) {
                    av[jj] = adec[(sb*JPS + jj)*32 + lane];
                }
#pragma unroll
                for (int nb = 0; nb < NTI; nb += 2) {
                    if (nb*8 >= ncols) {
                        continue;
                    }
                    int pacc[2][4] = {};
#pragma unroll
                    for (int jj = 0; jj < JPS; ++jj) {
                        const int j = sb*JPS + jj;
                        const uint32_t a[4] = { av[jj].x, av[jj].y, av[jj].z, av[jj].w };
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            const int c = nw*WN + (nb + u)*8 + g;
                            const uint32_t x0 = *(const uint32_t *) (st + c*128 + ((2*j)     ^ (c & 7))*16 + t*4);
                            const uint32_t x1 = *(const uint32_t *) (st + c*128 + ((2*j + 1) ^ (c & 7))*16 + t*4);
                            nvfp4pf_mma(pacc[u], a, x0, x1);
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const int c0 = nw*WN + (nb + u)*8 + 2*t;
                        const float dx0 = sx[c0*SUBS + sb], dx1 = sx[(c0 + 1)*SUBS + sb];
                        acc[nb + u][0] = fmaf(nvfp4pf_i2f(pacc[u][0]), dw0*dx0, acc[nb + u][0]);
                        acc[nb + u][1] = fmaf(nvfp4pf_i2f(pacc[u][1]), dw0*dx1, acc[nb + u][1]);
                        acc[nb + u][2] = fmaf(nvfp4pf_i2f(pacc[u][2]), dw1*dx0, acc[nb + u][2]);
                        acc[nb + u][3] = fmaf(nvfp4pf_i2f(pacc[u][3]), dw1*dx1, acc[nb + u][3]);
                    }
                }
            }
        }
    }
    nvfp4pf_wait<0>();

    if (!slab_ok || ncols <= 0) {
        return;
    }
    const bool scaled = s_exp || s_all != 1.0f;
#pragma unroll
    for (int i = 0; i < NTI; ++i) {
        const int c = nw*WN + i*8 + 2*t;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            if (c + q < nt) {
                float * yc = y + (int64_t) scol[c + q]*ldy;
                yc[r0] = scaled ? acc[i][q]*oscale     : acc[i][q];
                yc[r1] = scaled ? acc[i][q + 2]*oscale : acc[i][q + 2];
            }
        }
    }
}
