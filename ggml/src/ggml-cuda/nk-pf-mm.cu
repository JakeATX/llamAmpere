// NEONKERN NK_PF_MM v3: Ampere (sm_80 / sm_86) NVFP4 MoE prefill int8 tensor-core GEMM.
// BM=128, BK=64, BN=64|128, 8 warps (4x2, 2 M-slabs/pair, interleaved nw columns),
// 3-stage cp.async x ring + 2-stage double-buffered adec/wsc (zero bar64 in mainloop),
// exact 577-entry UE4M3->E2M1 LUT + 128-entry wsc LUT in SMEM (zero FP32 in dequant),
// zero-SHFL 0-way-bank-conflict XOR-swizzled adec write/read, 100 KB SMEM carveout (2 CTAs/SM).

#include "nk-pf-mm.cuh"

#include <atomic>
#include <cstdlib>
#include <cstdint>

static std::atomic<int64_t> g_nk_pf_mm_launches{0};

bool ggml_cuda_nk_pf_mm_enabled() {
    static const bool en = []() {
        // default ON (HOME ABBA 100K: prefill +10.2%, byte-exact); GGML_CUDA_NK_PF_MM=0 disables
        const char * e = getenv("GGML_CUDA_NK_PF_MM");
        return !(e && *e && atoi(e) == 0);
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

#include "nk-pf-lut.cuh"

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

static __device__ __forceinline__ float nk_pf_i2f_biased(const int x_biased) {
    return __fsub_rn(__int_as_float(x_biased), 12582912.0f);
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
// Pure integer/LUT decode (zero FP32, zero SHFL, zero subnormal branches): SWAR 0x7F zeroing +
// integer max, s_em/s_lut/s_wsc SMEM lookup, 2x dec8, and 0-bank-conflict XOR-swizzled STS.32.
template <int KF, bool USE_SMEM_LUT>
static __device__ __forceinline__ void nk_pf_decode_slab(
        const uint32_t d_mine, const uint32_t q0_mine, const uint32_t q1_mine,
        const int nw, const int g, const int t, const int s_mine_shift, const int adec_wr_u32,
        uint32_t * __restrict__ adec_root_u32, float * __restrict__ wsc_slab,
        const uint2 * __restrict__ s_lut, const float * __restrict__ s_wsc, const uint8_t * __restrict__ s_em) {
    // Zero any byte == 0x7F (NaN->0) without cross-byte borrow, and strip bit 7 on all 4 bytes
    const uint32_t x7f  = d_mine ^ 0x7F7F7F7Fu;
    const uint32_t is7f = (~(((x7f & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | x7f)) & 0x80808080u;
    const uint32_t m4   = (d_mine & 0x7F7F7F7Fu) & ~(is7f - (is7f >> 7));
    const uint32_t s_u  = (m4 >> s_mine_shift) & 0xFFu;

    uint32_t m_u;
    if constexpr (KF == 64) {
        const uint32_t b0 = m4 & 0xFFu;
        const uint32_t b1 = (m4 >> 8) & 0xFFu;
        const uint32_t b2 = (m4 >> 16) & 0xFFu;
        const uint32_t b3 = m4 >> 24;
        m_u = max(max(b0, b1), max(b2, b3));
    } else {
        const uint32_t mh = nw ? (m4 >> 16) : m4;
        m_u = max(mh & 0xFFu, (mh >> 8) & 0xFFu);
    }

    if constexpr (KF == 64) {
        if (nw == 0 && t < 2) {
            wsc_slab[g + t * 8] = USE_SMEM_LUT ? s_wsc[m_u] : __uint_as_float(__ldg(&g_nk_pf_blob.wsc[m_u]));
        }
    } else {
        if (t < 2) {
            wsc_slab[nw * 16 + g + t * 8] = USE_SMEM_LUT ? s_wsc[m_u] : __uint_as_float(__ldg(&g_nk_pf_blob.wsc[m_u]));
        }
    }

    const uint32_t em_m  = USE_SMEM_LUT ? (uint32_t) s_em[m_u] : (uint32_t) __ldg(&g_nk_pf_blob.em[m_u]);
    const uint32_t em_s  = USE_SMEM_LUT ? (uint32_t) s_em[s_u] : (uint32_t) __ldg(&g_nk_pf_blob.em[s_u]);
    const uint32_t d_exp = (em_m >> 3) - (em_s >> 3);
    const uint32_t idx   = (s_u == 0u || d_exp > 8u) ? 576u : ((d_exp << 6) | ((em_s & 7u) << 3) | (em_m & 7u));
    const uint2 l01      = USE_SMEM_LUT ? s_lut[idx] : __ldg(&g_nk_pf_blob.lut[idx]);
    const uint32_t l0    = l01.x;
    const uint32_t l1    = l01.y;
    const uint32_t l2    = (0x80808080u - l0) ^ 0x80808080u;
    const uint32_t l3    = (0x80808080u - l1) ^ 0x80808080u;

    const int2 w0 = nk_pf_dec8(q0_mine, l0, l1, l2, l3);
    const int2 w1 = nk_pf_dec8(q1_mine, l0, l1, l2, l3);

    // Zero-SHFL, 0-way bank-conflict XOR-swizzled direct write relative to 256B-aligned adec_root_u32
    adec_root_u32[adec_wr_u32]      = (uint32_t) w0.x;
    adec_root_u32[adec_wr_u32 ^ 4]  = (uint32_t) w1.x;
    adec_root_u32[adec_wr_u32 ^ 8]  = (uint32_t) w0.y;
    adec_root_u32[adec_wr_u32 ^ 12] = (uint32_t) w1.y;
}

template <int KF, int BN>
static constexpr __host__ __device__ int nk_pf_stage_bytes() {
    return BN * 64 + BN * (64 / KF) * 4;
}

template <int KF, int BN>
static constexpr __host__ __device__ bool nk_pf_use_smem_lut() {
    return KF == 64 || BN == 64;
}

template <int KF, int BN>
static constexpr __host__ __device__ int nk_pf_smem_bytes() {
    // 3 stages of (xq + xs) + 2 stages of adec + 2 stages of wsc + optional 5136 B LUT+wsc blob
    return 3 * nk_pf_stage_bytes<KF, BN>()
        + 2 * (4 * 2 * 2 * 32 * 16)
        + 2 * (4 * 2 * (64 / KF) * 16 * 4)
        + (nk_pf_use_smem_lut<KF, BN>() ? (int) sizeof(nk_pf_lut_blob) : 0);
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
    constexpr int ADEC_SLOT_U4 = 4 * 128;              // 512 uint4s = 8192 B per wslot
    constexpr int WSC_SLOT_F32 = 4 * 2 * SUBS * 16;    // 128 (KF=64) or 256 (KF=32) floats per wslot
    constexpr bool USE_SMEM_LUT = nk_pf_use_smem_lut<KF, BN>();

    extern __shared__ __align__(16) uint8_t smem_nk[];
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
        scol[i] = i < nt ? map.cols[off + toff + i] : 0;
    }

    uint32_t * adec_root_u32 = (uint32_t *) (smem_nk + 3 * STAGE);
    uint4 * adec_base = (uint4 *) (smem_nk + 3 * STAGE) + mw * 128;
    float * wsc_base  = (float *) (smem_nk + 3 * STAGE + 2 * ADEC_SLOT_U4 * 16) + mw * (2 * SUBS * 16);

    uint2 *   s_lut = nullptr;
    float *   s_wsc = nullptr;
    uint8_t * s_em  = nullptr;
    if constexpr (USE_SMEM_LUT) {
        uint8_t * tbl_smem = smem_nk + 3 * STAGE + 2 * ADEC_SLOT_U4 * 16 + 2 * WSC_SLOT_F32 * 4;
        const uint4 * tbl_src = (const uint4 *) &g_nk_pf_blob;
        uint4 * tbl_dst = (uint4 *) tbl_smem;
        tbl_dst[tid] = __ldg(&tbl_src[tid]);
        if (tid < (int) (sizeof(nk_pf_lut_blob) / 16) - 256) {
            tbl_dst[tid + 256] = __ldg(&tbl_src[tid + 256]);
        }
        s_lut = (uint2 *) tbl_smem;
        s_wsc = (float *) (tbl_smem + 578 * sizeof(uint2));
        s_em  = tbl_smem + 578 * sizeof(uint2) + 128 * sizeof(uint32_t);
    }
    __syncthreads();

    // Precompute loop-invariant load_x swizzle offset and row bases directly from map.rows (saves 512 B SMEM)
    const int c0_x   = tid >> 2;
    const int p0_x   = tid & 3;
    const int xq_off = c0_x * 64 + (p0_x ^ ((c0_x >> 1) & 3)) * 16;
    const bool ok_x0 = c0_x < nt;
    const int8_t * xq_src0 = ok_x0 ? (xq + (int64_t) map.rows[off + toff + c0_x] * K + p0_x * 16) : xq;
    const bool ok_x1 = (BN == 128) && ((c0_x + 64) < nt);
    const int8_t * xq_src1 = xq;
    if constexpr (BN == 128) {
        if (ok_x1) {
            xq_src1 = xq + (int64_t) map.rows[off + toff + c0_x + 64] * K + p0_x * 16;
        }
    }

    const bool has_xs = tid < BN * SUBS;
    const int c_xs    = SUBS == 1 ? tid : (tid >> 1);
    const int sb_xs   = SUBS == 1 ? 0   : (tid & 1);
    const bool ok_xs  = has_xs && (c_xs < nt);
    const float * xs_src = ok_xs ? (xs + (int64_t) map.rows[off + toff + c_xs] * nbx + sb_xs) : xs;

    auto load_x = [&](const int kb, const int slot) {
        uint8_t * st = smem_nk + slot * STAGE;
        nk_pf_cp16(st + xq_off, xq_src0 + kb * 64, ok_x0);
        if constexpr (BN == 128) {
            nk_pf_cp16(st + xq_off + 4096, xq_src1 + kb * 64, ok_x1);
        }
        if (has_xs) {
            float * sx = (float *) (st + BN * 64);
            nk_pf_cp4(sx + tid, xs_src + kb * SUBS, ok_xs);
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

    // Prologue: enqueue x for kb = 0 (slot 0) and kb = 1 (slot 1)
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

    const int r_mine0     = (t & 1) ? r1_0 : r0_0;
    const int r_mine1     = (t & 1) ? r1_1 : r0_1;
    int s_mine_shift      = 8 * (2 * nw + (t >> 1));
    int adec_wr0_base     = (mw * 128 + nw * 32) * 4 + (g * 4 + (g >> 1)) * 4 + t;
    const int lane_sw     = g * 4 + (t ^ (g >> 1));
#ifndef NK_PF_NO_SM80
    if constexpr (KF == 64 || BN == 64) {
        asm("" : "+r"(s_mine_shift), "+r"(adec_wr0_base));
    }
#endif
    const int q_off = 4 + s_mine_shift;

    const char * b0 = slab_ok0 ? (W + (int64_t) r_mine0 * stride_blk * 36) : nullptr;
    const char * b1 = slab_ok1 ? (W + (int64_t) r_mine1 * stride_blk * 36) : nullptr;

    // Prologue weight decode for kb = 0 -> wslot 0 (overlaps cp.async of x[0], x[1])
    uint32_t d0_cur = 0, q00_cur = 0, q01_cur = 0;
    uint32_t d1_cur = 0, q10_cur = 0, q11_cur = 0;
    if (slab_ok0) {
        const uint32_t d0 = __ldg((const uint32_t *) b0);
        const uint32_t q0 = __ldg((const uint32_t *) (b0 + q_off));
        const uint32_t q1 = __ldg((const uint32_t *) (b0 + q_off + 4));
        nk_pf_decode_slab<KF, USE_SMEM_LUT>(d0, q0, q1, nw, g, t, s_mine_shift, adec_wr0_base,
            adec_root_u32, wsc_base, s_lut, s_wsc, s_em);
        // Prefetch raw W for kb = 1 (nk >= 2 guaranteed)
        b0 += 36;
        d0_cur  = __ldg((const uint32_t *) b0);
        q00_cur = __ldg((const uint32_t *) (b0 + q_off));
        q01_cur = __ldg((const uint32_t *) (b0 + q_off + 4));
    }
    if (slab_ok1) {
        const uint32_t d1 = __ldg((const uint32_t *) b1);
        const uint32_t q0 = __ldg((const uint32_t *) (b1 + q_off));
        const uint32_t q1 = __ldg((const uint32_t *) (b1 + q_off + 4));
        nk_pf_decode_slab<KF, USE_SMEM_LUT>(d1, q0, q1, nw, g, t, s_mine_shift, adec_wr0_base + 256,
            adec_root_u32, wsc_base + SUBS * 16, s_lut, s_wsc, s_em);
        b1 += 36;
        d1_cur  = __ldg((const uint32_t *) b1);
        q10_cur = __ldg((const uint32_t *) (b1 + q_off));
        q11_cur = __ldg((const uint32_t *) (b1 + q_off + 4));
    } else if (slab_ok0) {
        (adec_base + 64)[nw * 32 + lane_sw] = make_uint4(0, 0, 0, 0);
        (adec_base + ADEC_SLOT_U4 + 64)[nw * 32 + lane_sw] = make_uint4(0, 0, 0, 0);
    }

    // Hoisted ldmatrix.x4 and sx base offsets (pair_idx steps by +32 cols = +2048 B, swizzle invariant)
    const int u_ld     = lane >> 4;
    const int l7       = lane & 7;
    const int p_bit    = (lane >> 3) & 1;
    const int c_ld0    = nw * 16 + u_ld * 8 + l7;
    const int c_sw0    = (c_ld0 >> 1) & 3;
    int ldm_off0       = c_ld0 * 64 + (p_bit ^ c_sw0) * 16;
    int ldm_off1       = c_ld0 * 64 + ((2 + p_bit) ^ c_sw0) * 16;
    const int sx_off0  = (nw * 16 + 2 * t) * SUBS;
#ifndef NK_PF_NO_SM80
    asm("" : "+r"(ldm_off0), "+r"(ldm_off1));
#endif
    const int bias[4]  = { 0x4B400000, 0x4B400000, 0x4B400000, 0x4B400000 };

    int cur_slot = 0;
    int nxt_slot = 2;
    int wcur     = 0;

    for (int kb = 0; kb < nk; ++kb) {
        if (kb + 1 < nk) {
            nk_pf_wait<1>();
        } else {
            nk_pf_wait<0>();
        }
        // Single CTA barrier synchronizes both x[kb] and adec/wsc[wcur] (zero bar64 needed)
        __syncthreads();

        if (kb + 2 < nk) {
            load_x(kb + 2, nxt_slot);
            nk_pf_commit();
        }

        const int wnxt = wcur ^ 1;
        if (slab_ok0) {
            const uint8_t * st = smem_nk + cur_slot * STAGE;
            const float * sx   = (const float *) (st + BN * 64);
            const uint4 * adec0_cur = adec_base + wcur * ADEC_SLOT_U4;
            const uint4 * adec1_cur = adec0_cur + 64;
            const float * wsc0_cur  = wsc_base  + wcur * WSC_SLOT_F32;
            const float * wsc1_cur  = wsc0_cur  + SUBS * 16;
            const int adec_wr0_nxt  = adec_wr0_base + wnxt * (ADEC_SLOT_U4 * 4);
            float * wsc0_nxt        = wsc_base  + wnxt * WSC_SLOT_F32;
            float * wsc1_nxt        = wsc0_nxt  + SUBS * 16;

#pragma unroll
            for (int sb = 0; sb < SUBS; ++sb) {
                const float dw00 = wsc0_cur[sb * 16 + g];
                const float dw01 = wsc0_cur[sb * 16 + g + 8];
                const float dw10 = slab_ok1 ? wsc1_cur[sb * 16 + g] : 0.0f;
                const float dw11 = slab_ok1 ? wsc1_cur[sb * 16 + g + 8] : 0.0f;

#pragma unroll
                for (int pair_idx = 0; pair_idx < NPAIRS; ++pair_idx) {
                    const int c_base = (pair_idx * 2 + nw) * 16;
                    if (c_base < nt) {
                        int pacc0[2][4];
                        int pacc1[2][4];
                        const uint8_t * st_pair = st + pair_idx * 2048;

#pragma unroll
                        for (int jj = 0; jj < JPS; ++jj) {
                            const int j = sb * JPS + jj;
                            uint32_t b00, b01, b10, b11;
                            nk_pf_ldmatrix_x4(b00, b01, b10, b11, st_pair + (j == 0 ? ldm_off0 : ldm_off1));

                            const uint4 av0 = adec0_cur[j * 32 + lane_sw];
                            const uint32_t a0[4] = { av0.x, av0.y, av0.z, av0.w };
                            if (jj == 0) {
                                nk_pf_mma_init(pacc0[0], a0, b00, b01, bias);
                                nk_pf_mma_init(pacc0[1], a0, b10, b11, bias);
                            } else {
                                nk_pf_mma(pacc0[0], a0, b00, b01);
                                nk_pf_mma(pacc0[1], a0, b10, b11);
                            }
                            const uint4 av1 = adec1_cur[j * 32 + lane_sw];
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
                            const int sx_i  = sx_off0 + (pair_idx * 32 + u * 8) * SUBS + sb;
                            const float dx0 = sx[sx_i];
                            const float dx1 = sx[sx_i + SUBS];
                            const int ni    = pair_idx * 2 + u;
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

                    // Interleave next-block LUT decode + kb+2 raw-W prefetch with MMA pairs
                    if (sb == 0 && pair_idx == 0 && kb + 1 < nk) {
                        nk_pf_decode_slab<KF, USE_SMEM_LUT>(d0_cur, q00_cur, q01_cur, nw, g, t,
                            s_mine_shift, adec_wr0_nxt, adec_root_u32, wsc0_nxt, s_lut, s_wsc, s_em);
                        if (kb + 2 < nk) {
                            b0 += 36;
                            d0_cur  = __ldg((const uint32_t *) b0);
                            q00_cur = __ldg((const uint32_t *) (b0 + q_off));
                            q01_cur = __ldg((const uint32_t *) (b0 + q_off + 4));
                        }
                    }
                    if (sb == 0 && pair_idx == 1 && kb + 1 < nk && slab_ok1) {
                        nk_pf_decode_slab<KF, USE_SMEM_LUT>(d1_cur, q10_cur, q11_cur, nw, g, t,
                            s_mine_shift, adec_wr0_nxt + 256, adec_root_u32, wsc1_nxt, s_lut, s_wsc, s_em);
                        if (kb + 2 < nk) {
                            b1 += 36;
                            d1_cur  = __ldg((const uint32_t *) b1);
                            q10_cur = __ldg((const uint32_t *) (b1 + q_off));
                            q11_cur = __ldg((const uint32_t *) (b1 + q_off + 4));
                        }
                    }
                }
            }
        }

        cur_slot = cur_slot == 2 ? 0 : cur_slot + 1;
        nxt_slot = nxt_slot == 2 ? 0 : nxt_slot + 1;
        wcur     = wnxt;
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
            const int c  = c_base + u * 8 + 2 * t;
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
    static_assert(smem + BN * (int) sizeof(int) <= 50 * 1024,
        "NK_PF_MM: per-CTA shared memory exceeds 2 CTAs/SM budget on 100 KB Ampere SMEM");

    static std::atomic<uint64_t> smem_cfg_set{0};
    const uint64_t dev_bit = (device >= 0 && device < 64) ? (uint64_t(1) << device) : 0;
    if (dev_bit && (smem_cfg_set.load(std::memory_order_relaxed) & dev_bit) == 0) {
        const cudaError_t err1 = cudaFuncSetAttribute(
            k_nk_pf_mm<KF, BN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem);
        const cudaError_t err2 = cudaFuncSetAttribute(
            k_nk_pf_mm<KF, BN>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared);
        if (err1 == cudaSuccess && err2 == cudaSuccess) {
            smem_cfg_set.fetch_or(dev_bit, std::memory_order_relaxed);
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
        GGML_LOG_WARN("[NK_PF_MM] active: KF=%d BN=%d BM=128 BK=64 DB=2 LUT=1 M=%d K=%d max_tiles=%d\n",
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
