#include "mmvq-kq-tc.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

// [KQTC] one CTA owns 16 output rows; its warps split the 256-wide K blocks (warp w takes blocks w, w+nwarps, ...)
// and a shared-memory sum joins them. Within a warp the m16n8 tile holds rows g / g+8 (g = lane/4) and the
// 8 columns; the B operand of column g is the q8_1 activation, C lands at columns 2t / 2t+1 (t = lane%4).
static constexpr int KQTC_ROWS = 16;

static __device__ __forceinline__ void kqtc_mma_k32(int c[4], const int a0, const int a1, const int a2, const int a3,
                                                    const int b0, const int b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ void kqtc_mma_k16(int c[4], const int a0, const int a1, const int b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE && !defined(GGML_USE_HIP)
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
#else
    GGML_UNUSED_VARS(c, a0, a1, b0);
    NO_DEVICE_CODE;
#endif
}

// activations of one 256-wide K block: yg = B column g, y0 / y1 = the C columns 2t / 2t+1 (for their scales)
struct kqtc_y {
    const block_q8_1 * yg;
    const block_q8_1 * y0;
    const block_q8_1 * y1;
    bool hg, h0, h1;
};

static __device__ __forceinline__ int kqtc_b(const kqtc_y & y, const int j, const int i) {
    return y.hg ? ((const int *) y.yg[j].qs)[i] : 0;
}

static __device__ __forceinline__ float2 kqtc_ds(const block_q8_1 * yc, const bool h, const int j) {
    return h ? __half22float2(yc[j].ds) : make_float2(0.0f, 0.0f);
}

static __device__ __forceinline__ int kqtc_byte(const int v, const int k) {
    return (v >> (8*k)) & 0xFF;
}

// Q4_K / Q5_K: x = d*sc*q - dmin*m per 32 values; q8_1 ds.y = d8*sum(q8), as in MMVQ
template <bool q5>
static __device__ __forceinline__ void kqtc_sb_q45(const char * xr0, const char * xr8, const kqtc_y & y, const int t,
                                                   float acc[4]) {
    using blk = typename std::conditional<q5, block_q5_K, block_q4_K>::type;
    const blk * b0 = (const blk *) xr0;
    const blk * b8 = (const blk *) xr8;

    const int * qs0 = (const int *) b0->qs;
    const int * qs8 = (const int *) b8->qs;
    int q0[8], q8[8];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        q0[2*c]     = qs0[8*c + t];
        q0[2*c + 1] = qs0[8*c + 4 + t];
        q8[2*c]     = qs8[8*c + t];
        q8[2*c + 1] = qs8[8*c + 4 + t];
    }
    int h0a = 0, h0b = 0, h8a = 0, h8b = 0;
    if constexpr (q5) {
        h0a = ((const int *) b0->qh)[t];
        h0b = ((const int *) b0->qh)[4 + t];
        h8a = ((const int *) b8->qh)[t];
        h8b = ((const int *) b8->qh)[4 + t];
    }
    const int * s0 = (const int *) b0->scales;
    const int * s8 = (const int *) b8->scales;
    const int u00 = s0[0], u01 = s0[1], u02 = s0[2];
    const int u80 = s8[0], u81 = s8[1], u82 = s8[2];
    const int sc0lo = u00 & 0x3F3F3F3F;
    const int mn0lo = u01 & 0x3F3F3F3F;
    const int sc0hi = (u02 & 0x0F0F0F0F) | ((u00 >> 2) & 0x30303030);
    const int mn0hi = ((u02 >> 4) & 0x0F0F0F0F) | ((u01 >> 2) & 0x30303030);
    const int sc8lo = u80 & 0x3F3F3F3F;
    const int mn8lo = u81 & 0x3F3F3F3F;
    const int sc8hi = (u82 & 0x0F0F0F0F) | ((u80 >> 2) & 0x30303030);
    const int mn8hi = ((u82 >> 4) & 0x0F0F0F0F) | ((u81 >> 2) & 0x30303030);

    const float2 dm0 = __half22float2(b0->dm);
    const float2 dm8 = __half22float2(b8->dm);

    float sd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sm[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c  = j/2;
        const int sh = 4*(j % 2);
        int a0 = (q0[2*c]     >> sh) & 0x0F0F0F0F;
        int a2 = (q0[2*c + 1] >> sh) & 0x0F0F0F0F;
        int a1 = (q8[2*c]     >> sh) & 0x0F0F0F0F;
        int a3 = (q8[2*c + 1] >> sh) & 0x0F0F0F0F;
        if constexpr (q5) {
            a0 |= ((h0a >> j) & 0x01010101) << 4;
            a2 |= ((h0b >> j) & 0x01010101) << 4;
            a1 |= ((h8a >> j) & 0x01010101) << 4;
            a3 |= ((h8b >> j) & 0x01010101) << 4;
        }
        int cc[4] = {0, 0, 0, 0};
        kqtc_mma_k32(cc, a0, a1, a2, a3, kqtc_b(y, j, t), kqtc_b(y, j, 4 + t));

        const int sc0 = kqtc_byte(j < 4 ? sc0lo : sc0hi, j % 4);
        const int mn0 = kqtc_byte(j < 4 ? mn0lo : mn0hi, j % 4);
        const int sc8 = kqtc_byte(j < 4 ? sc8lo : sc8hi, j % 4);
        const int mn8 = kqtc_byte(j < 4 ? mn8lo : mn8hi, j % 4);
        const float2 d0 = kqtc_ds(y.y0, y.h0, j);
        const float2 d1 = kqtc_ds(y.y1, y.h1, j);
        sd[0] += (float) (cc[0]*sc0) * d0.x;
        sd[1] += (float) (cc[1]*sc0) * d1.x;
        sd[2] += (float) (cc[2]*sc8) * d0.x;
        sd[3] += (float) (cc[3]*sc8) * d1.x;
        sm[0] += (float) mn0 * d0.y;
        sm[1] += (float) mn0 * d1.y;
        sm[2] += (float) mn8 * d0.y;
        sm[3] += (float) mn8 * d1.y;
    }
    acc[0] += dm0.x*sd[0] - dm0.y*sm[0];
    acc[1] += dm0.x*sd[1] - dm0.y*sm[1];
    acc[2] += dm8.x*sd[2] - dm8.y*sm[2];
    acc[3] += dm8.x*sd[3] - dm8.y*sm[3];
}

// Q6_K: x = d*sc*(q - 32) with one int8 scale per 16 values, so one k16 mma per 16 values.
// Value p of the block: n = p/128, qd = (p%128)/32, l = p%32: low bits in ql[64n + 32*(qd%2) + l] (nibble qd/2),
// high bits in qh[32n + l] (bits 2*qd), scale sc[8n + 2*qd + l/16]. The 210-byte block is only 2-byte aligned.
static __device__ __forceinline__ void kqtc_sb_q6(const char * xr0, const char * xr8, const kqtc_y & y, const int t,
                                                  float acc[4]) {
    const block_q6_K * b0 = (const block_q6_K *) xr0;
    const block_q6_K * b8 = (const block_q6_K *) xr8;

    int sc0[4], sc8[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sc0[i] = get_int_b2(b0->scales, i);
        sc8[i] = get_int_b2(b8->scales, i);
    }

    float sf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < 2; ++n) {
        int l0[4], l8[4], h0[2], h8[2];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            l0[m] = get_int_b2(b0->ql, 16*n + 4*m + t);
            l8[m] = get_int_b2(b8->ql, 16*n + 4*m + t);
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            h0[m] = get_int_b2(b0->qh, 8*n + 4*m + t);
            h8[m] = get_int_b2(b8->qh, 8*n + 4*m + t);
        }
#pragma unroll
        for (int qd = 0; qd < 4; ++qd) {
            const int j = 4*n + qd; // q8_1 block
            const float d0 = kqtc_ds(y.y0, y.h0, j).x;
            const float d1 = kqtc_ds(y.y1, y.h1, j).x;
#pragma unroll
            for (int lh = 0; lh < 2; ++lh) {
                const int m = 2*(qd % 2) + lh;
                const int a0 = __vsub4(((l0[m] >> (4*(qd/2))) & 0x0F0F0F0F) | (((h0[lh] >> (2*qd)) & 0x03030303) << 4), 0x20202020);
                const int a1 = __vsub4(((l8[m] >> (4*(qd/2))) & 0x0F0F0F0F) | (((h8[lh] >> (2*qd)) & 0x03030303) << 4), 0x20202020);
                int cc[4] = {0, 0, 0, 0};
                kqtc_mma_k16(cc, a0, a1, kqtc_b(y, j, 4*lh + t));

                const int is = 8*n + 2*qd + lh;
                const int s_0 = (int) (int8_t) kqtc_byte(sc0[is/4], is % 4);
                const int s_8 = (int) (int8_t) kqtc_byte(sc8[is/4], is % 4);
                sf[0] += (float) (cc[0]*s_0) * d0;
                sf[1] += (float) (cc[1]*s_0) * d1;
                sf[2] += (float) (cc[2]*s_8) * d0;
                sf[3] += (float) (cc[3]*s_8) * d1;
            }
        }
    }
    const float dd0 = __half2float(b0->d);
    const float dd8 = __half2float(b8->d);
    acc[0] += dd0*sf[0];
    acc[1] += dd0*sf[1];
    acc[2] += dd8*sf[2];
    acc[3] += dd8*sf[3];
}

// IQ3_S: 8 groups of 32; 4 values per grid entry (9-bit index: qs byte + one qh bit), one sign bit per value,
// x = d*(1 + 2*scale)*grid*sign. Lane t takes grid entries t (values 4t..4t+3) and 4+t (16+4t..16+4t+3).
static __device__ __forceinline__ int kqtc_iq3s_vals(const uint32_t * grid, const int idx, const int s4) {
    const int bits = (s4 & 1) | ((s4 & 2) << 7) | ((s4 & 4) << 14) | ((s4 & 8) << 21);
    const int mask = __vcmpne4(bits, 0);
    return __vsub4((int) grid[idx] ^ mask, mask);
}

static __device__ __forceinline__ void kqtc_sb_iq3s(const char * xr0, const char * xr8, const kqtc_y & y, const int t,
                                                    float acc[4], const uint32_t * grid) {
    const block_iq3_s * b0 = (const block_iq3_s *) xr0;
    const block_iq3_s * b8 = (const block_iq3_s *) xr8;

    const int qh0[2] = { get_int_b2(b0->qh, 0), get_int_b2(b0->qh, 1) };
    const int qh8[2] = { get_int_b2(b8->qh, 0), get_int_b2(b8->qh, 1) };
    const int ls0 = get_int_b2(b0->scales, 0);
    const int ls8 = get_int_b2(b8->scales, 0);

    float sf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const int qa0 = get_int_b2(b0->qs, 2*ib), qb0 = get_int_b2(b0->qs, 2*ib + 1);
        const int qa8 = get_int_b2(b8->qs, 2*ib), qb8 = get_int_b2(b8->qs, 2*ib + 1);
        const int h0  = kqtc_byte(qh0[ib/4], ib % 4);
        const int h8  = kqtc_byte(qh8[ib/4], ib % 4);
        const int sg0 = get_int_b2(b0->signs, ib);
        const int sg8 = get_int_b2(b8->signs, ib);

        const int a0 = kqtc_iq3s_vals(grid, kqtc_byte(qa0, t) | ((h0 << (8 - t)) & 0x100), (sg0 >> (4*t))      & 0xF);
        const int a2 = kqtc_iq3s_vals(grid, kqtc_byte(qb0, t) | ((h0 << (4 - t)) & 0x100), (sg0 >> (16 + 4*t)) & 0xF);
        const int a1 = kqtc_iq3s_vals(grid, kqtc_byte(qa8, t) | ((h8 << (8 - t)) & 0x100), (sg8 >> (4*t))      & 0xF);
        const int a3 = kqtc_iq3s_vals(grid, kqtc_byte(qb8, t) | ((h8 << (4 - t)) & 0x100), (sg8 >> (16 + 4*t)) & 0xF);

        int cc[4] = {0, 0, 0, 0};
        kqtc_mma_k32(cc, a0, a1, a2, a3, kqtc_b(y, ib, t), kqtc_b(y, ib, 4 + t));

        const int s_0 = 1 + 2*((ls0 >> (4*ib)) & 0xF);
        const int s_8 = 1 + 2*((ls8 >> (4*ib)) & 0xF);
        const float d0 = kqtc_ds(y.y0, y.h0, ib).x;
        const float d1 = kqtc_ds(y.y1, y.h1, ib).x;
        sf[0] += (float) (cc[0]*s_0) * d0;
        sf[1] += (float) (cc[1]*s_0) * d1;
        sf[2] += (float) (cc[2]*s_8) * d0;
        sf[3] += (float) (cc[3]*s_8) * d1;
    }
    const float dd0 = __half2float(b0->d);
    const float dd8 = __half2float(b8->d);
    acc[0] += dd0*sf[0];
    acc[1] += dd0*sf[1];
    acc[2] += dd8*sf[2];
    acc[3] += dd8*sf[3];
}

template <ggml_type type> struct kqtc_traits;
template <> struct kqtc_traits<GGML_TYPE_Q4_K>  { static constexpr int bytes = sizeof(block_q4_K);  };
template <> struct kqtc_traits<GGML_TYPE_Q5_K>  { static constexpr int bytes = sizeof(block_q5_K);  };
template <> struct kqtc_traits<GGML_TYPE_Q6_K>  { static constexpr int bytes = sizeof(block_q6_K);  };
template <> struct kqtc_traits<GGML_TYPE_IQ3_S> { static constexpr int bytes = sizeof(block_iq3_s); };

template <ggml_type type, int nwarps>
__launch_bounds__(nwarps*32, 1)
static __global__ void mul_mat_kq_tc(
        const char * __restrict__ vx, const block_q8_1 * __restrict__ vy, float * __restrict__ dst,
        const int nsb, const int ncols_dst, const size_t stride_row_x, const int stride_col_y, const int stride_col_dst) {
    ggml_cuda_pdl_lc();

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int g    = lane / 4;
    const int t    = lane % 4;

    constexpr int bytes = kqtc_traits<type>::bytes;

    __shared__ float    red[nwarps][KQTC_ROWS][8];
    __shared__ uint32_t sgrid[type == GGML_TYPE_IQ3_S ? 512 : 1];
    if constexpr (type == GGML_TYPE_IQ3_S) {
        for (int i = threadIdx.x; i < 512; i += nwarps*32) {
            sgrid[i] = iq3s_grid[i];
        }
        __syncthreads();
    }

    ggml_cuda_pdl_sync();

    const int row0 = blockIdx.x * KQTC_ROWS;
    const char * x0 = vx + (size_t) (row0 + g)     * stride_row_x;
    const char * x8 = vx + (size_t) (row0 + g + 8) * stride_row_x;

    kqtc_y y;
    y.hg = g < ncols_dst;
    y.h0 = 2*t     < ncols_dst;
    y.h1 = 2*t + 1 < ncols_dst;
    const block_q8_1 * yg = vy + (size_t) (y.hg ? g       : 0) * stride_col_y;
    const block_q8_1 * y0 = vy + (size_t) (y.h0 ? 2*t     : 0) * stride_col_y;
    const block_q8_1 * y1 = vy + (size_t) (y.h1 ? 2*t + 1 : 0) * stride_col_y;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int kb = warp; kb < nsb; kb += nwarps) {
        y.yg = yg + 8*kb;
        y.y0 = y0 + 8*kb;
        y.y1 = y1 + 8*kb;
        const char * xb0 = x0 + (size_t) kb*bytes;
        const char * xb8 = x8 + (size_t) kb*bytes;
        if constexpr (type == GGML_TYPE_Q4_K) {
            kqtc_sb_q45<false>(xb0, xb8, y, t, acc);
        } else if constexpr (type == GGML_TYPE_Q5_K) {
            kqtc_sb_q45<true>(xb0, xb8, y, t, acc);
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            kqtc_sb_q6(xb0, xb8, y, t, acc);
        } else {
            kqtc_sb_iq3s(xb0, xb8, y, t, acc, sgrid);
        }
    }

    red[warp][g][2*t]         = acc[0];
    red[warp][g][2*t + 1]     = acc[1];
    red[warp][g + 8][2*t]     = acc[2];
    red[warp][g + 8][2*t + 1] = acc[3];
    __syncthreads();

    for (int i = threadIdx.x; i < KQTC_ROWS * 8; i += nwarps * 32) {
        const int r = i / 8;
        const int c = i % 8;
        if (c < ncols_dst) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < nwarps; ++w) {
                s += red[w][r][c];
            }
            dst[(size_t) c * stride_col_dst + row0 + r] = s;
        }
    }
}

struct kqtc_config {
    bool types[GGML_TYPE_COUNT] = {};
    bool any     = false;
    int  min_n   = 2;
    int  warps   = 0; // 0 = auto
    bool q8_cache = true;
    int64_t max_m = 0; // 0 = no limit
};

static const kqtc_config & kqtc_get_config() {
    static const kqtc_config cfg = [] {
        kqtc_config c;
        const char * e = getenv("GGML_CUDA_KQ_TC");
        if (e == nullptr || *e == '\0' || strcmp(e, "0") == 0 || strcmp(e, "off") == 0) {
            return c;
        }
        const ggml_type all[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ3_S };
        if (strcmp(e, "1") == 0 || strcmp(e, "all") == 0) {
            for (ggml_type ty : all) {
                c.types[ty] = true;
            }
        } else {
            std::string s(e);
            size_t pos = 0;
            while (pos <= s.size()) {
                const size_t end = s.find(',', pos);
                const std::string name = s.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
                for (ggml_type ty : all) {
                    if (name == ggml_type_name(ty)) {
                        c.types[ty] = true;
                    }
                }
                if (end == std::string::npos) {
                    break;
                }
                pos = end + 1;
            }
        }
        for (ggml_type ty : all) {
            c.any = c.any || c.types[ty];
        }
        if (const char * m = getenv("GGML_CUDA_KQ_TC_MIN_N")) {
            const int v = atoi(m);
            if (v >= 1 && v <= 8) {
                c.min_n = v;
            }
        }
        if (const char * w = getenv("GGML_CUDA_KQ_TC_WARPS")) {
            const int v = atoi(w);
            c.warps = (v == 4 || v == 8) ? v : 0;
        }
        if (const char * m = getenv("GGML_CUDA_KQ_TC_MAX_M")) {
            c.max_m = atoll(m) > 0 ? atoll(m) : 0;
        }
        if (const char * q = getenv("GGML_CUDA_KQ_TC_Q8CACHE")) {
            c.q8_cache = atoi(q) != 0;
        }
        GGML_LOG_WARN("[kqtc] on: q4_K %d q5_K %d q6_K %d iq3_s %d, min_n %d, warps %s, q8 cache %d, max rows %lld\n",
                      c.types[GGML_TYPE_Q4_K], c.types[GGML_TYPE_Q5_K], c.types[GGML_TYPE_Q6_K], c.types[GGML_TYPE_IQ3_S],
                      c.min_n, c.warps == 0 ? "auto" : (c.warps == 4 ? "4" : "8"), c.q8_cache, (long long) c.max_m);
        return c;
    }();
    return cfg;
}

template <ggml_type type>
static void kqtc_launch(const int nwarps, const char * vx, const block_q8_1 * vy, float * dst, const int nsb, const int nrows,
                        const int ncols_dst, const size_t stride_row_x, const int stride_col_y, const int stride_col_dst,
                        cudaStream_t stream) {
    if (nwarps == 8) {
        const ggml_cuda_kernel_launch_params p(dim3(nrows / KQTC_ROWS), dim3(8*32), 0, stream);
        ggml_cuda_kernel_launch(mul_mat_kq_tc<type, 8>, p, vx, vy, dst, nsb, ncols_dst, stride_row_x, stride_col_y, stride_col_dst);
    } else {
        const ggml_cuda_kernel_launch_params p(dim3(nrows / KQTC_ROWS), dim3(4*32), 0, stream);
        ggml_cuda_kernel_launch(mul_mat_kq_tc<type, 4>, p, vx, vy, dst, nsb, ncols_dst, stride_row_x, stride_col_y, stride_col_dst);
    }
}

bool ggml_cuda_mul_mat_kq_tc_try(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const kqtc_config & cfg = kqtc_get_config();
    if (!cfg.any || src0->type >= GGML_TYPE_COUNT || !cfg.types[src0->type] ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne11 = src1->ne[1];
    const size_t  ts   = ggml_type_size(src0->type);
    // Q4_K / Q5_K blocks are read as int (144 / 176 bytes), Q6_K / IQ3_S as two halves (210 / 110 bytes)
    const size_t  align = (src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K) ? 4 : 2;
    if (ne11 < cfg.min_n || ne11 > 8 || (cfg.max_m > 0 && src0->ne[1] > cfg.max_m) ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[2] != 1 || dst->ne[3] != 1 ||
            ne00 <= 0 || ne00 % QK_K != 0 || ne00 > std::numeric_limits<int>::max() || src1->ne[0] != ne00 ||
            ne01 <= 0 || ne01 % KQTC_ROWS != 0 || ne01 > std::numeric_limits<int>::max() ||
            src0->nb[0] != ts || src0->nb[1] % align != 0 || (uintptr_t) src0->data % align != 0 ||
            src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 ||
            dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 ||
            dst->nb[1] / sizeof(float) > (size_t) std::numeric_limits<int>::max()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t q8_bytes = (size_t) ne11 * ne10_padded * sizeof(block_q8_1) / QK8_1;
    // Same shared-quantize cache as MMVQ (q8_cache in common.cuh): the q8_1 layout here is MMVQ's plain layout (no
    // IQ4_XS swizzle), so an activation already quantized for an MMVQ / fused-norm consumer of the same src1 in this
    // graph eval is reused, and ours is left for the next one. Same quantizer either way, so the bytes are identical.
    ggml_cuda_pool_alloc<char> q8_local(ctx.pool());
    char * q8 = nullptr;
    const bool cacheable = cfg.q8_cache && ggml_cuda_q8_cacheable(ctx, q8_bytes);
    if (cacheable) {
        for (auto & e : ctx.q8_cache.entries) {
            if (e.epoch == ctx.graph_epoch && e.src1 == src1 && e.data == src1->data && e.size == q8_bytes &&
                    e.ne10_padded == ne10_padded && !e.swizzle_iq4 && e.dev == ctx.device) {
                e.last_use = ++ctx.q8_cache.tick;
                ctx.fusion_stats.q8_cache_hits++;
                q8 = e.ptr;
                break;
            }
        }
    }
    if (q8 == nullptr) {
        q8 = cacheable ? ggml_cuda_q8_cache_claim(ctx, src1, src0->type, q8_bytes, ne10_padded) : q8_local.alloc(q8_bytes);
        const int64_t s11 = src1->nb[1] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8, src0->type, ne00, s11, s11*ne11, s11*ne11,
                               ne10_padded, ne11, 1, 1, stream);
    }

    const int nsb    = (int) (ne00 / QK_K);
    const int nwarps = cfg.warps != 0 ? cfg.warps : (nsb >= 16 ? 8 : 4);
    const char * vx = (const char *) src0->data;
    const block_q8_1 * vy = (const block_q8_1 *) q8;
    float * d = (float *) dst->data;
    const int stride_col_y   = (int) (ne10_padded / QK8_1);
    const int stride_col_dst = (int) (dst->nb[1] / sizeof(float));
    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            kqtc_launch<GGML_TYPE_Q4_K>(nwarps, vx, vy, d, nsb, (int) ne01, (int) ne11, src0->nb[1], stride_col_y, stride_col_dst, stream);
            break;
        case GGML_TYPE_Q5_K:
            kqtc_launch<GGML_TYPE_Q5_K>(nwarps, vx, vy, d, nsb, (int) ne01, (int) ne11, src0->nb[1], stride_col_y, stride_col_dst, stream);
            break;
        case GGML_TYPE_Q6_K:
            kqtc_launch<GGML_TYPE_Q6_K>(nwarps, vx, vy, d, nsb, (int) ne01, (int) ne11, src0->nb[1], stride_col_y, stride_col_dst, stream);
            break;
        case GGML_TYPE_IQ3_S:
            kqtc_launch<GGML_TYPE_IQ3_S>(nwarps, vx, vy, d, nsb, (int) ne01, (int) ne11, src0->nb[1], stride_col_y, stride_col_dst, stream);
            break;
        default:
            GGML_ABORT("kqtc: unexpected type");
    }
    return true;
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
#endif
}
