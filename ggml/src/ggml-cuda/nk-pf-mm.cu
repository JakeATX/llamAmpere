// NEONKERN NK_PF_MM: Ampere (sm_80 / sm_86) NVFP4 MoE prefill int8 tensor-core GEMM.
// BM=128, BK=64, BN=64|128, 8 warps (4x2, 2 M-slabs/pair, interleaved nw columns),
// 3-stage cp.async x ring with bank-conflict-free ldmatrix.x4 swizzle,
// 4x leaner cooperative warp-pair dequant with 4-shuffle 4x4 transpose,
// overlapped raw-W prefetch, biased pacc (0x4B400000), 100 KB SMEM carveout (2 CTAs/SM).

#include "nk-pf-mm.cuh"

#include <atomic>
#include <cstdlib>
#include <cstdint>

static std::atomic<int64_t> g_nk_pf_mm_launches{0};

bool ggml_cuda_nk_pf_mm_enabled() {
    static const bool en = []() {
        const char * e = getenv("GGML_CUDA_NK_PF_MM");
        return e && atoi(e) != 0;
    }();
    return en;
}

int64_t ggml_cuda_nk_pf_mm_count() {
    return g_nk_pf_mm_launches.load(std::memory_order_relaxed);
}

static bool nk_pf_mm_trace() {
    static const bool tr = []() {
        const char * e = getenv("GGML_CUDA_NK_TRACE");
        return e && atoi(e) != 0;
    }();
    return tr;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static constexpr float NK_PF_MIN_SCALE = 1e-30f;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define NK_PF_NO_SM80 1
#endif

static __device__ __forceinline__ void nk_pf_cp16(void * s, const void * g, const bool pred) {
#ifdef NK_PF_NO_SM80
    *(int4 *) s = pred ? *(const int4 *) g : make_int4(0, 0, 0, 0);
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sa), "l"(g), "r"(pred ? 16 : 0) : "memory");
#endif
}

static __device__ __forceinline__ void nk_pf_cp4(void * s, const void * g, const bool pred) {
#ifdef NK_PF_NO_SM80
    *(float *) s = pred ? *(const float *) g : 0.0f;
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" :: "r"(sa), "l"(g), "r"(pred ? 4 : 0) : "memory");
#endif
}

static __device__ __forceinline__ void nk_pf_commit() {
#ifndef NK_PF_NO_SM80
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

template <int N>
static __device__ __forceinline__ void nk_pf_wait() {
#ifndef NK_PF_NO_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
#endif
}

static __device__ __forceinline__ void nk_pf_bar64(const int id) {
#ifdef NK_PF_NO_SM80
    (void) id;
    __syncthreads();
#else
    asm volatile("bar.sync %0, 64;\n" :: "r"(id) : "memory");
#endif
}

static __device__ __forceinline__ void nk_pf_ldmatrix_x4(
        uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * s) {
#ifdef NK_PF_NO_SM80
    (void) s;
    r0 = r1 = r2 = r3 = 0;
    __trap();
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(sa));
#endif
}

static __device__ __forceinline__ void nk_pf_mma_init(
        int (&d)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1, const int (&c)[4]) {
#ifdef NK_PF_NO_SM80
    (void) d; (void) a; (void) b0; (void) b1; (void) c;
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
#endif
}

static __device__ __forceinline__ void nk_pf_mma(
        int (&d)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#ifdef NK_PF_NO_SM80
    (void) d; (void) a; (void) b0; (void) b1;
    __trap();
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

static __device__ __forceinline__ float nk_pf_ue4m3(const uint32_t u) {
    const uint32_t x  = u & 0x7Fu;
    const bool     sn = x < 8u;
    const float    v  = __fsub_rn(__uint_as_float((sn ? 0x3C000000u : 0x3B800000u) + (x << 20)), sn ? 0.0078125f : 0.0f);
    return u == 0x7Fu ? 0.0f : v;
}

static __device__ __forceinline__ float nk_pf_i2f_biased(const int x_biased) {
    return __fsub_rn(__int_as_float(x_biased), 12582912.0f);
}

static __device__ __forceinline__ int nk_pf_f2i(const float x) {
    return __float_as_int(__fadd_rn(x, 12582912.0f)) - 0x4B400000;
}

static __device__ __forceinline__ float nk_pf_rcp(const float a) {
    float y = __uint_as_float(0x7EF311C2u - __float_as_uint(a));
    y = y * fmaf(-a, y, 2.0f);
    y = y * fmaf(-a, y, 2.0f);
    return a >= NK_PF_MIN_SCALE ? y : 0.0f;
}

static __device__ __forceinline__ void nk_pf_lut(
        const float sub, const float inv, uint32_t & l0, uint32_t & l1, uint32_t & l2, uint32_t & l3) {
    const uint32_t R  = (uint32_t) min(max(nk_pf_f2i(sub * inv), 0), 43349);
    const uint32_t R3 = 2 * R + R;
    const uint32_t R6 = R3 << 1;
    const uint32_t v1 = (R + 2048u) >> 12;
    const uint32_t v2 = (2 * R + 2048u) >> 12;
    const uint32_t v3 = (R3 + 2048u) >> 12;
    const uint32_t v4 = (4 * R + 2048u) >> 12;
    const uint32_t v6 = (R6 + 2048u) >> 12;
    const uint32_t v8 = (8 * R + 2048u) >> 12;
    const uint32_t v12 = ((R6 << 1) + 2048u) >> 12;
    l0 = (v1 << 8) | (v2 << 16) | (v3 << 24);
    l1 = v4 | (v6 << 8) | (v8 << 16) | (v12 << 24);
    l2 = (0x80808080u - l0) ^ 0x80808080u;
    l3 = (0x80808080u - l1) ^ 0x80808080u;
}

static __device__ __forceinline__ int2 nk_pf_dec8(
        const uint32_t q4, const uint32_t l0, const uint32_t l1, const uint32_t l2, const uint32_t l3) {
    const uint32_t idx = q4 & 0x77777777u;
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    const uint32_t m0  = __byte_perm(__byte_perm(l0, l1, idx), __byte_perm(l2, l3, idx), sel);
    const uint32_t m1  = __byte_perm(__byte_perm(l0, l1, idx >> 16), __byte_perm(l2, l3, idx >> 16), sel >> 16);
    return make_int2((int) __byte_perm(m0, m1, 0x6420u), (int) __byte_perm(m0, m1, 0x7531u));
}

// Cooperative warp-pair decode for one 16-row slab of one 64-weight NVFP4 block.
// nw=0 decodes j=0 (sub-blocks 0, 1); nw=1 decodes j=1 (sub-blocks 2, 3).
// Each lane (g, t) owns row (t & 1 ? r1 : r0) and sub-block 2*nw + (t >> 1), builds 1 LUT,
// decodes both 32-bit nibble words without dropping halves, then runs a 4-shuffle 4x4 transpose.
template <int KF>
static __device__ __forceinline__ void nk_pf_decode_slab(
        const uint32_t d_mine, const uint32_t q0_mine, const uint32_t q1_mine,
        const int nw, const int g, const int t, const int lane,
        uint4 * __restrict__ adec_slab, float * __restrict__ wsc_slab) {
    constexpr float QW = 4096.0f * (127.0f / 12.0f);
    const int s_mine_idx = 2 * nw + (t >> 1);
    const float s_mine = nk_pf_ue4m3((d_mine >> (8 * s_mine_idx)) & 0xFFu);

    float m_span;
    if constexpr (KF == 64) {
        const int s_oth_idx = 2 * (nw ^ 1) + (t >> 1);
        const float s_oth = nk_pf_ue4m3((d_mine >> (8 * s_oth_idx)) & 0xFFu);
        const float s_mine_p = __shfl_xor_sync(0xffffffffu, s_mine, 2);
        const float s_oth_p  = __shfl_xor_sync(0xffffffffu, s_oth, 2);
        m_span = fmaxf(fmaxf(s_mine, s_mine_p), fmaxf(s_oth, s_oth_p));
        if (nw == 0 && t < 2) {
            wsc_slab[g + t * 8] = m_span >= NK_PF_MIN_SCALE ? m_span * (12.0f / 127.0f) : 0.0f;
        }
    } else {
        const float s_mine_p = __shfl_xor_sync(0xffffffffu, s_mine, 2);
        m_span = fmaxf(s_mine, s_mine_p);
        if (t < 2) {
            wsc_slab[nw * 16 + g + t * 8] = m_span >= NK_PF_MIN_SCALE ? m_span * (12.0f / 127.0f) : 0.0f;
        }
    }

    const float inv = QW * nk_pf_rcp(m_span);
    uint32_t l0, l1, l2, l3;
    nk_pf_lut(s_mine, inv, l0, l1, l2, l3);
    const int2 w0 = nk_pf_dec8(q0_mine, l0, l1, l2, l3);
    const int2 w1 = nk_pf_dec8(q1_mine, l0, l1, l2, l3);

    const uint32_t v0 = (uint32_t) w0.x;
    const uint32_t v1 = (uint32_t) w1.x;
    const uint32_t v2 = (uint32_t) w0.y;
    const uint32_t v3 = (uint32_t) w1.y;

    const bool odd = (t & 1) != 0;
    const uint32_t p0 = __shfl_xor_sync(0xffffffffu, odd ? v0 : v1, 1);
    const uint32_t p2 = __shfl_xor_sync(0xffffffffu, odd ? v2 : v3, 1);
    const uint32_t u0 = odd ? p0 : v0;
    const uint32_t u1 = odd ? v1 : p0;
    const uint32_t u2 = odd ? p2 : v2;
    const uint32_t u3 = odd ? v3 : p2;

    const bool hi2 = (t & 2) != 0;
    const uint32_t q0 = __shfl_xor_sync(0xffffffffu, hi2 ? u0 : u2, 2);
    const uint32_t q1 = __shfl_xor_sync(0xffffffffu, hi2 ? u1 : u3, 2);
    const uint32_t r0 = hi2 ? q0 : u0;
    const uint32_t r1 = hi2 ? q1 : u1;
    const uint32_t r2 = hi2 ? u2 : q0;
    const uint32_t r3 = hi2 ? u3 : q1;

    adec_slab[nw * 32 + lane] = make_uint4(r0, r1, r2, r3);
}

template <int KF, int BN>
static constexpr __host__ __device__ int nk_pf_stage_bytes() {
    return BN * 64 + BN * (64 / KF) * 4;
}

template <int KF, int BN>
static constexpr __host__ __device__ int nk_pf_smem_bytes() {
    // 3 stages of (xq + xs) + adec (4 warp-pairs * 2 M-slabs * 2 j * 32 lanes * 16 B)
    // + wsc (4 warp-pairs * 2 M-slabs * (64/KF) * 16 rows * 4 B)
    return 3 * nk_pf_stage_bytes<KF, BN>() + 4 * 2 * 2 * 32 * 16 + 4 * 2 * (64 / KF) * 16 * 4;
}

template <int KF, int BN>
static __global__ void __launch_bounds__(256, 2) k_nk_pf_mm(
        const char * __restrict__ w_all, const int64_t stride_blk,
        const int8_t * __restrict__ xq, const float * __restrict__ xs,
        float * __restrict__ y, const int64_t ldy,
        const int M, const int K, const nk_pf_map map,
        const float * __restrict__ s_exp, const float s_all) {
    static_assert(KF == 32 || KF == 64, "NK_PF_MM: KF must be 32 or 64");
    static_assert(BN == 64 || BN == 128, "NK_PF_MM: BN must be 64 or 128");
    constexpr int WN = BN / 2;
    constexpr int NTI = WN / 8;          // 4 (BN=64) or 8 (BN=128)
    constexpr int NPAIRS = NTI / 2;      // 2 (BN=64) or 4 (BN=128)
    constexpr int SUBS = 64 / KF;        // 1 (KF=64) or 2 (KF=32)
    constexpr int JPS = KF / 32;         // 2 (KF=64) or 1 (KF=32)
    constexpr int STAGE = nk_pf_stage_bytes<KF, BN>();

    extern __shared__ __align__(16) uint8_t smem_nk[];
    __shared__ int srow[BN];
    __shared__ int scol[BN];

    if ((int) blockIdx.y >= map.hdr[1]) {
        return;
    }

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int mw   = warp >> 1;
    const int nw   = warp & 1;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int nk   = K >> 6;   // K % 128 == 0 guarantees nk >= 2
    const int nbx  = K / KF;

    const int act  = map.tiles[2 * blockIdx.y + 0];
    const int toff = map.tiles[2 * blockIdx.y + 1];
    const int off  = map.off[act];
    const int nt   = min(BN, map.off[act + 1] - off - toff);
    const int e    = map.exps[act];
    const char * W = w_all + (int64_t) e * map.nb02;
    const float oscale = s_exp ? s_exp[e] * s_all : s_all;

    for (int i = tid; i < BN; i += 256) {
        srow[i] = i < nt ? map.rows[off + toff + i] : 0;
        scol[i] = i < nt ? map.cols[off + toff + i] : 0;
    }
    __syncthreads();

    uint4 * adec0 = (uint4 *) (smem_nk + 3 * STAGE) + mw * 128;
    uint4 * adec1 = adec0 + 64;
    float * wsc0  = (float *) (smem_nk + 3 * STAGE + 4 * 2 * 2 * 32 * 16) + mw * (2 * SUBS * 16);
    float * wsc1  = wsc0 + SUBS * 16;

    auto load_x = [&](const int kb, const int slot) {
        uint8_t * st = smem_nk + slot * STAGE;
#pragma unroll
        for (int it = 0; it < BN * 4 / 256; ++it) {
            const int i = tid + it * 256;
            const int c = i >> 2;
            const int p = i & 3;
            const bool ok = c < nt;
            nk_pf_cp16(
                st + c * 64 + (p ^ ((c >> 1) & 3)) * 16,
                xq + (ok ? (int64_t) srow[c] * K + kb * 64 + p * 16 : 0),
                ok);
        }
        float * sx = (float *) (st + BN * 64);
        if (tid < BN * SUBS) {
            const int c  = SUBS == 1 ? tid : (tid >> 1);
            const int sb = SUBS == 1 ? 0   : (tid & 1);
            const bool ok = c < nt;
            nk_pf_cp4(sx + tid, xs + (ok ? (int64_t) srow[c] * nbx + kb * SUBS + sb : 0), ok);
        }
    };

    float acc[2][NTI][4];
#pragma unroll
    for (int ms = 0; ms < 2; ++ms) {
#pragma unroll
        for (int i = 0; i < NTI; ++i) {
            acc[ms][i][0] = 0.0f;
            acc[ms][i][1] = 0.0f;
            acc[ms][i][2] = 0.0f;
            acc[ms][i][3] = 0.0f;
        }
    }

    // Prologue: enqueue kb = 0 (slot 0) and kb = 1 (slot 1)
    load_x(0, 0);
    nk_pf_commit();
    load_x(1, 1);
    nk_pf_commit();

    const int n_slabs = M >> 4;
    const int slab0   = (int) blockIdx.x * 8 + mw * 2;
    const int slab1   = slab0 + 1;
    const bool slab_ok0 = slab0 < n_slabs;
    const bool slab_ok1 = slab1 < n_slabs;

    const int r0_0 = slab0 * 16 + g;
    const int r1_0 = r0_0 + 8;
    const int r0_1 = slab1 * 16 + g;
    const int r1_1 = r0_1 + 8;

    const int r_mine0 = (t & 1) ? r1_0 : r0_0;
    const int r_mine1 = (t & 1) ? r1_1 : r0_1;
    const int q_off   = 4 + 8 * (2 * nw + (t >> 1));

    const char * b0 = slab_ok0 ? (W + (int64_t) r_mine0 * stride_blk * 36) : nullptr;
    const char * b1 = slab_ok1 ? (W + (int64_t) r_mine1 * stride_blk * 36) : nullptr;

    uint32_t d0_cur = 0, q00_cur = 0, q01_cur = 0;
    uint32_t d1_cur = 0, q10_cur = 0, q11_cur = 0;
    if (slab_ok0) {
        d0_cur  = __ldg((const uint32_t *) b0);
        q00_cur = __ldg((const uint32_t *) (b0 + q_off));
        q01_cur = __ldg((const uint32_t *) (b0 + q_off + 4));
    }
    if (slab_ok1) {
        d1_cur  = __ldg((const uint32_t *) b1);
        q10_cur = __ldg((const uint32_t *) (b1 + q_off));
        q11_cur = __ldg((const uint32_t *) (b1 + q_off + 4));
    } else if (slab_ok0) {
        adec1[nw * 32 + lane] = make_uint4(0, 0, 0, 0);
    }

    const int u_ld  = lane >> 4;
    const int l7    = lane & 7;
    const int p_bit = (lane >> 3) & 1;
    const int bias[4] = { 0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000 };

    int cur_slot = 0;
    int nxt_slot = 2;

    for (int kb = 0; kb < nk; ++kb) {
        if (kb + 1 < nk) {
            nk_pf_wait<1>();
        } else {
            nk_pf_wait<0>();
        }
        __syncthreads();

        if (kb + 2 < nk) {
            load_x(kb + 2, nxt_slot);
            nk_pf_commit();
        }

        const uint8_t * st = smem_nk + cur_slot * STAGE;

        if (slab_ok0) {
            nk_pf_decode_slab<KF>(d0_cur, q00_cur, q01_cur, nw, g, t, lane, adec0, wsc0);
        }
        if (slab_ok1) {
            nk_pf_decode_slab<KF>(d1_cur, q10_cur, q11_cur, nw, g, t, lane, adec1, wsc1);
        }
        nk_pf_bar64(mw + 1);

        // Prefetch raw W for kb + 1 so global/L2 load latency overlaps the MMA loop
        if (kb + 1 < nk) {
            if (slab_ok0) {
                b0 += 36;
                d0_cur  = __ldg((const uint32_t *) b0);
                q00_cur = __ldg((const uint32_t *) (b0 + q_off));
                q01_cur = __ldg((const uint32_t *) (b0 + q_off + 4));
            }
            if (slab_ok1) {
                b1 += 36;
                d1_cur  = __ldg((const uint32_t *) b1);
                q10_cur = __ldg((const uint32_t *) (b1 + q_off));
                q11_cur = __ldg((const uint32_t *) (b1 + q_off + 4));
            }
        }

        if (slab_ok0 && nw * 16 < nt) {
            const float * sx = (const float *) (st + BN * 64);
#pragma unroll
            for (int sb = 0; sb < SUBS; ++sb) {
                const float dw00 = wsc0[sb * 16 + g];
                const float dw01 = wsc0[sb * 16 + g + 8];
                const float dw10 = slab_ok1 ? wsc1[sb * 16 + g] : 0.0f;
                const float dw11 = slab_ok1 ? wsc1[sb * 16 + g + 8] : 0.0f;

#pragma unroll
                for (int pair_idx = 0; pair_idx < NPAIRS; ++pair_idx) {
                    const int c_base = (pair_idx * 2 + nw) * 16;
                    if (c_base >= nt) {
                        continue;
                    }
                    int pacc0[2][4];
                    int pacc1[2][4];
                    const int c_ld = c_base + u_ld * 8 + l7;
                    const int c_sw = (c_ld >> 1) & 3;
                    const uint8_t * st_row = st + c_ld * 64;

#pragma unroll
                    for (int jj = 0; jj < JPS; ++jj) {
                        const int j = sb * JPS + jj;
                        const int p_ld = 2 * j + p_bit;
                        uint32_t b00, b01, b10, b11;
                        nk_pf_ldmatrix_x4(b00, b01, b10, b11, st_row + (p_ld ^ c_sw) * 16);

                        const uint4 av0 = adec0[j * 32 + lane];
                        const uint32_t a0[4] = { av0.x, av0.y, av0.z, av0.w };
                        if (jj == 0) {
                            nk_pf_mma_init(pacc0[0], a0, b00, b01, bias);
                            nk_pf_mma_init(pacc0[1], a0, b10, b11, bias);
                        } else {
                            nk_pf_mma(pacc0[0], a0, b00, b01);
                            nk_pf_mma(pacc0[1], a0, b10, b11);
                        }
                        const uint4 av1 = adec1[j * 32 + lane];
                        const uint32_t a1[4] = { av1.x, av1.y, av1.z, av1.w };
                        if (jj == 0) {
                            nk_pf_mma_init(pacc1[0], a1, b00, b01, bias);
                            nk_pf_mma_init(pacc1[1], a1, b10, b11, bias);
                        } else {
                            nk_pf_mma(pacc1[0], a1, b00, b01);
                            nk_pf_mma(pacc1[1], a1, b10, b11);
                        }
                    }

#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const int c0 = c_base + u * 8 + 2 * t;
                        const float dx0 = sx[c0 * SUBS + sb];
                        const float dx1 = sx[(c0 + 1) * SUBS + sb];
                        const int ni = pair_idx * 2 + u;
                        acc[0][ni][0] = fmaf(nk_pf_i2f_biased(pacc0[u][0]), dw00 * dx0, acc[0][ni][0]);
                        acc[0][ni][1] = fmaf(nk_pf_i2f_biased(pacc0[u][1]), dw00 * dx1, acc[0][ni][1]);
                        acc[0][ni][2] = fmaf(nk_pf_i2f_biased(pacc0[u][2]), dw01 * dx0, acc[0][ni][2]);
                        acc[0][ni][3] = fmaf(nk_pf_i2f_biased(pacc0[u][3]), dw01 * dx1, acc[0][ni][3]);
                        acc[1][ni][0] = fmaf(nk_pf_i2f_biased(pacc1[u][0]), dw10 * dx0, acc[1][ni][0]);
                        acc[1][ni][1] = fmaf(nk_pf_i2f_biased(pacc1[u][1]), dw10 * dx1, acc[1][ni][1]);
                        acc[1][ni][2] = fmaf(nk_pf_i2f_biased(pacc1[u][2]), dw11 * dx0, acc[1][ni][2]);
                        acc[1][ni][3] = fmaf(nk_pf_i2f_biased(pacc1[u][3]), dw11 * dx1, acc[1][ni][3]);
                    }
                }
            }
        }

        cur_slot = cur_slot == 2 ? 0 : cur_slot + 1;
        nxt_slot = nxt_slot == 2 ? 0 : nxt_slot + 1;
    }

    if (!slab_ok0 || nw * 16 >= nt) {
        return;
    }

    const bool scaled = s_exp != nullptr || s_all != 1.0f;
#pragma unroll
    for (int pair_idx = 0; pair_idx < NPAIRS; ++pair_idx) {
        const int c_base = (pair_idx * 2 + nw) * 16;
        if (c_base >= nt) {
            continue;
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int c = c_base + u * 8 + 2 * t;
            const int ni = pair_idx * 2 + u;
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                if (c + q < nt) {
                    float * yc = y + (int64_t) scol[c + q] * ldy;
                    yc[r0_0] = scaled ? acc[0][ni][q] * oscale     : acc[0][ni][q];
                    yc[r1_0] = scaled ? acc[0][ni][q + 2] * oscale : acc[0][ni][q + 2];
                    if (slab_ok1) {
                        yc[r0_1] = scaled ? acc[1][ni][q] * oscale     : acc[1][ni][q];
                        yc[r1_1] = scaled ? acc[1][ni][q + 2] * oscale : acc[1][ni][q + 2];
                    }
                }
            }
        }
    }
}

template <int KF, int BN>
static void nk_pf_launch_impl(
        const char * w, const int64_t stride_blk,
        const int8_t * xq, const float * xs,
        float * y, const int M, const int K,
        const nk_pf_map & map, const int max_tiles,
        const float * s_exp, const int device, cudaStream_t stream) {
    constexpr int smem = nk_pf_smem_bytes<KF, BN>();
    static_assert(smem + 2 * BN * (int) sizeof(int) <= 50 * 1024,
        "NK_PF_MM: per-CTA shared memory exceeds 2 CTAs/SM budget on 100 KB Ampere SMEM");

    static std::atomic<uint64_t> carveout_set{0};
    const uint64_t dev_bit = (device >= 0 && device < 64) ? (uint64_t(1) << device) : 0;
    if (dev_bit && (carveout_set.load(std::memory_order_relaxed) & dev_bit) == 0) {
        const cudaError_t err = cudaFuncSetAttribute(
            k_nk_pf_mm<KF, BN>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        if (err == cudaSuccess) {
            carveout_set.fetch_or(dev_bit, std::memory_order_relaxed);
        } else {
            (void) cudaGetLastError();
        }
    }

    const dim3 grid((unsigned) ((M + 127) / 128), (unsigned) max_tiles);
    k_nk_pf_mm<KF, BN><<<grid, 256, smem, stream>>>(
        w, stride_blk, xq, xs, y, M, M, K, map, s_exp, 1.0f);
}

void ggml_cuda_nk_pf_mm_launch(
        const int KF, const int bn,
        const char * w, const int64_t stride_blk,
        const int8_t * xq, const float * xs,
        float * y, const int M, const int K,
        const nk_pf_map & map, const int max_tiles,
        const float * s_exp, const int device, cudaStream_t stream) {
    const int64_t cnt = g_nk_pf_mm_launches.fetch_add(1, std::memory_order_relaxed) + 1;
    static std::atomic<int> logged{0};
    if (logged.exchange(1) == 0) {
        GGML_LOG_WARN("[NK_PF_MM] active: KF=%d BN=%d BM=128 BK=64 M=%d K=%d max_tiles=%d\n",
            KF, bn, M, K, max_tiles);
    } else if (nk_pf_mm_trace() && (cnt <= 4 || (cnt & 255) == 0)) {
        GGML_LOG_WARN("[NK_PF_MM] trace #%lld: KF=%d BN=%d M=%d K=%d max_tiles=%d\n",
            (long long) cnt, KF, bn, M, K, max_tiles);
    }

    if (KF == 64) {
        if (bn == 64) {
            nk_pf_launch_impl<64, 64>(w, stride_blk, xq, xs, y, M, K, map, max_tiles, s_exp, device, stream);
        } else {
            nk_pf_launch_impl<64, 128>(w, stride_blk, xq, xs, y, M, K, map, max_tiles, s_exp, device, stream);
        }
    } else {
        if (bn == 64) {
            nk_pf_launch_impl<32, 64>(w, stride_blk, xq, xs, y, M, K, map, max_tiles, s_exp, device, stream);
        } else {
            nk_pf_launch_impl<32, 128>(w, stride_blk, xq, xs, y, M, K, map, max_tiles, s_exp, device, stream);
        }
    }
}

#else

void ggml_cuda_nk_pf_mm_launch(
        int, int, const char *, int64_t, const int8_t *, const float *,
        float *, int, int, const nk_pf_map &, int, const float *, int, cudaStream_t) {
}

#endif
