// Single-pass compact SJ-KVaRN 4/4 verify attention for widths up to 16 (sm_80+).
// Packs all gqa*n_q (query, head) pairs into ceil(gqa*n_q/8) warps (up to 12 warps = 96 pairs,
// covering 16 rows at GQA 6) in one CTA so each 16-token KV strip is read once from VRAM.

#include "fattn-sjkvarn-v16.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-sjkvarn-rot.cuh"
#include "qsa-resident-clock.cuh"
#include "mma.cuh"
#include "ggml-inksj.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <string>

namespace fattn_sj_kvarn_v16 {

using namespace ggml_cuda_mma;

constexpr int D               = 256;
constexpr int NTC             = D / 16;
constexpr int NCOLS2          = 8;
constexpr int UNIT            = 128;
constexpr int MAX_NQ          = 16;
constexpr int MAX_WARPS       = 12;

constexpr int STRIP_BYTES     = NTC * WARP_SIZE * 4;
constexpr int META_BYTES      = 96;
constexpr int META_KTOK       = 2 * STRIP_BYTES;
constexpr int META_VS         = META_KTOK + 32;
constexpr int META_VZ         = META_VS + 32;
constexpr int STAGE_BYTES     = 2 * STRIP_BYTES + META_BYTES;
constexpr int STAGE_CHUNKS    = STAGE_BYTES / 16;
constexpr int EXACT_ROW_BYTES = D * 2;
constexpr int EXACT_BYTES     = 16 * EXACT_ROW_BYTES;
constexpr int RMETA_KZ        = 2 * D;
constexpr int RMETA_VCH       = 4 * D;
constexpr int RMETA_BYTES     = 6 * D;
constexpr int RMETA_CHUNKS    = RMETA_BYTES / 16;

typedef tile<16, 8, half2> T_A;
typedef tile< 8, 8, half2> T_B;
typedef tile<16, 8, float> T_C;

static __host__ __device__ __forceinline__ size_t v16_shared_bytes(const int nstage) {
    return (size_t) nstage * STAGE_BYTES + 2 * RMETA_BYTES;
}

static __device__ __forceinline__ uint32_t lop3_and_or(const uint32_t x, const uint32_t mask, const uint32_t magic) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return (x & mask) | magic;
#else
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(mask), "r"(magic));
    return r;
#endif
}

static __device__ __forceinline__ void raw_pairs(const uint32_t x, const uint32_t magic, half2 * const __restrict__ r) {
    const uint32_t y = x >> 8;
    r[0] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(x, 0x000F000FU, magic));
    r[1] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(x, 0x00F000F0U, magic));
    r[2] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(y, 0x000F000FU, magic));
    r[3] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(y, 0x00F000F0U, magic));
}

static __device__ __forceinline__ void decode_pairs(const uint32_t x, const uint32_t magic, half2 * const __restrict__ q) {
    const half2 m1024 = make_half2(1024.0f, 1024.0f);
    const half2 c16   = make_half2(0.0625f, 0.0625f);
    const half2 m64   = make_half2(-64.0f, -64.0f);
    half2 r[4];
    raw_pairs(x, magic, r);
    q[0] = __hsub2(r[0], m1024);
    q[1] = __hfma2(r[1], c16, m64);
    q[2] = __hsub2(r[2], m1024);
    q[3] = __hfma2(r[3], c16, m64);
}

static __device__ __forceinline__ void cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    NO_DEVICE_CODE;
#endif
}

template <int n>
static __device__ __forceinline__ void cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" :: "n"(n) : "memory");
#else
    NO_DEVICE_CODE;
#endif
}

struct cursor {
    int p0;
    int g;
    int sl;
};

template <int max_warps, int NSTAGE, bool rot>
__launch_bounds__(WARP_SIZE * max_warps, max_warps <= 6 ? 2 : 1)
static __global__ void flash_attn_ext_sj_kvarn_v16(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        const int  * __restrict__ KV_rows,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const char    * __restrict__ sj_kvarn_body,
        const int32_t * __restrict__ sj_kvarn_desc) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    GGML_UNUSED_VARS(sinks, KV_max, KV_rows, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb03, ne10, ne13,
                     nb13, nb23, ne31, ne32, ne33, nb32, nb33);

    extern __shared__ __align__(16) char smem[];

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;
    const int n_q    = int(ne01.z);

    const int z_KV  = blockIdx.z;
    const int gqa   = ne02 / ne12;
    const int head0 = z_KV * gqa;
    const int hcol  = lane >> 2;
    const int npair = gqa * n_q;

    // Even gqa guarantees columns 2*(lane&3) and +1 share one query row (and both pad pairs map to row 0).
    auto col_query = [&](const int col) -> int {
        const int p = 8 * warp + col;
        return p < npair ? p / gqa : 0;
    };
    auto col_head = [&](const int col) -> int {
        return (8 * warp + col) % gqa;
    };
    auto col_valid = [&](const int col) -> bool {
        return 8 * warp + col < npair;
    };
    const bool hvalid = col_valid(hcol);

    const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(sj_kvarn_body, sj_kvarn_desc, z_KV, D, 4, 4, nb12, nb22);
    const int n_kv_pad = ne11;
    const int G16 = kv.G / 16;

    const int nsplit = gridDim.y;
    const int split  = blockIdx.y;
    const int sb0    = kv.S / 16;
    const int sb1    = kv.B / 16;
    const int nbody  = sb1 - sb0;
    const int s_lo   = sb0 + (int) (((int64_t) nbody * split) / nsplit);
    const int s_hi   = sb0 + (int) (((int64_t) nbody * (split + 1)) / nsplit);
    const int n_ex_sink = kv.S / 16;
    const int far_end   = sj_kvarn_desc[GGML_INKSJ_DESC_FLAGS] >> GGML_INKSJ_FAR_END_SHIFT;
    const int ex_end    = far_end > 0 ? min(n_kv_pad, max(kv.B, 16 * ((far_end + 15) / 16))) : n_kv_pad;
    const int n_ex      = n_ex_sink + (ex_end - kv.B) / 16;

    const float * Qf = (const float *) (Q + (size_t) col_query(hcol) * nb01 + (size_t) (head0 + (hvalid ? col_head(hcol) : 0)) * nb02);
    const half  * mrow = (const half *) (mask + (size_t) col_query(2 * (lane & 3)) * nb31);
    const char  * Kh = K + (size_t) z_KV * nb12;
    const char  * Vh = V + (size_t) z_KV * nb22;

    static_assert(NSTAGE * STAGE_BYTES >= EXACT_BYTES, "ring must hold one exact strip");
    static_assert(NSTAGE <= 6, "rmeta double buffer needs G/16 > NSTAGE - 1");
    char * ring = smem;
    char * rmeta = ring + NSTAGE * STAGE_BYTES;
    const uint32_t ring_s  = ggml_cuda_cvta_generic_to_shared(ring);
    const uint32_t rmeta_s = ring_s + NSTAGE * STAGE_BYTES;
    const uint32_t magic   = 0x64006400U;

    T_B   qp[NTC];
    float c_r[2] = {0.0f, 0.0f};
    T_C   acc[NTC];
    float kq_max[2] = {-FLT_MAX / 2, -FLT_MAX / 2};
    float kq_sum[2] = {0.0f, 0.0f};
    float ps[2]     = {0.0f, 0.0f};
    float pz[2]     = {0.0f, 0.0f};

    auto load_q = [&](const int c, float2 & q0, float2 & q1) {
        if (hvalid) {
            q0 = *(const float2 *) (Qf + 16 * c + 2 * (lane & 3));
            q1 = *(const float2 *) (Qf + 16 * c + 2 * (lane & 3) + 8);
            q0.x *= scale; q0.y *= scale; q1.x *= scale; q1.y *= scale;
        } else {
            q0 = make_float2(0.0f, 0.0f);
            q1 = make_float2(0.0f, 0.0f);
        }
    };

    auto setup_body = [&](const char * rm) {
        const uint2 * ks2 = (const uint2 *) (rm            + 2 * (lane & 3) * (D / 4));
        const uint2 * kz2 = (const uint2 *) (rm + RMETA_KZ + 2 * (lane & 3) * (D / 4));
        float cr = 0.0f;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 ks = ks2[c];
            const uint2 kz = kz2[c];
            float2 q0, q1;
            load_q(c, q0, q1);
            const half2 ks0 = fattn_sj_kvarn_u32_as_half2(ks.x);
            const half2 ks1 = fattn_sj_kvarn_u32_as_half2(ks.y);
            const float2 kz0 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.x));
            const float2 kz1 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.y));
            qp[c].x[0] = __hmul2(__floats2half2_rn(q0.x, q0.y), ks0);
            qp[c].x[1] = __hmul2(__floats2half2_rn(q1.x, q1.y), ks1);
            cr += q0.x * kz0.x + q0.y * kz0.y + q1.x * kz1.x + q1.y * kz1.y;
        }
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 1, WARP_SIZE);
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 2, WARP_SIZE);
        c_r[0] = __shfl_sync(0xFFFFFFFF, cr, 8 * (lane & 3),     WARP_SIZE);
        c_r[1] = __shfl_sync(0xFFFFFFFF, cr, 8 * (lane & 3) + 4, WARP_SIZE);
    };

    auto setup_exact = [&]() {
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            float2 q0, q1;
            load_q(c, q0, q1);
            qp[c].x[0] = __floats2half2_rn(q0.x, q0.y);
            qp[c].x[1] = __floats2half2_rn(q1.x, q1.y);
        }
        c_r[0] = 0.0f;
        c_r[1] = 0.0f;
    };

    auto record_change = [&](const char * rec_old, const char * rec_new) {
        if (rec_old != nullptr) {
            float p0s = ps[0] + __shfl_xor_sync(0xFFFFFFFF, ps[0], 4, WARP_SIZE);
            float p1s = ps[1] + __shfl_xor_sync(0xFFFFFFFF, ps[1], 4, WARP_SIZE);
            float z0s = pz[0] + __shfl_xor_sync(0xFFFFFFFF, pz[0], 4, WARP_SIZE);
            float z1s = pz[1] + __shfl_xor_sync(0xFFFFFFFF, pz[1], 4, WARP_SIZE);
#pragma unroll
            for (int off = 8; off < WARP_SIZE; off <<= 1) {
                p0s += __shfl_xor_sync(0xFFFFFFFF, p0s, off, WARP_SIZE);
                p1s += __shfl_xor_sync(0xFFFFFFFF, p1s, off, WARP_SIZE);
                z0s += __shfl_xor_sync(0xFFFFFFFF, z0s, off, WARP_SIZE);
                z1s += __shfl_xor_sync(0xFFFFFFFF, z1s, off, WARP_SIZE);
            }
            const float lo0 = z0s - 1024.0f * p0s, lo1 = z1s - 1024.0f * p1s;
            const float hi0 = 16.0f * z0s - 1024.0f * p0s, hi1 = 16.0f * z1s - 1024.0f * p1s;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += lo0; acc[c].x[1] += lo1; acc[c].x[2] += hi0; acc[c].x[3] += hi1;
            }
            ps[0] = 0.0f; ps[1] = 0.0f;
            pz[0] = 0.0f; pz[1] = 0.0f;
        }
        const uint32_t * vo = rec_old != nullptr ? (const uint32_t *) (rec_old + RMETA_VCH + 2 * ((lane >> 2) * (D / 8))) : nullptr;
        const uint32_t * vn = rec_new != nullptr ? (const uint32_t *) (rec_new + RMETA_VCH + 2 * ((lane >> 2) * (D / 8))) : nullptr;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            float r_lo = 1.0f, r_hi = 1.0f;
            if (vo != nullptr) {
                const float2 v = __half22float2(fattn_sj_kvarn_u32_as_half2(vo[c]));
                r_lo = v.x; r_hi = 0.0625f * v.y;
            }
            if (vn != nullptr) {
                const float2 v = __half22float2(fattn_sj_kvarn_u32_as_half2(vn[c]));
                r_lo = __fdividef(r_lo, v.x); r_hi = __fdividef(r_hi, 0.0625f * v.y);
            }
            acc[c].x[0] *= r_lo; acc[c].x[1] *= r_lo; acc[c].x[2] *= r_hi; acc[c].x[3] *= r_hi;
        }
    };

    auto softmax = [&](T_C & kq) -> T_B {
        float mx0 = fmaxf(kq.x[0], kq.x[2]);
        float mx1 = fmaxf(kq.x[1], kq.x[3]);
#pragma unroll
        for (int off = 4; off < WARP_SIZE; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, off, WARP_SIZE));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, off, WARP_SIZE));
        }
        const float nm0 = fmaxf(kq_max[0], mx0);
        const float nm1 = fmaxf(kq_max[1], mx1);
        const bool changed = nm0 != kq_max[0] || nm1 != kq_max[1];
        if (__any_sync(0xFFFFFFFF, changed)) {
            const float sc0 = expf(kq_max[0] - nm0);
            const float sc1 = expf(kq_max[1] - nm1);
            kq_max[0] = nm0;
            kq_max[1] = nm1;
            kq_sum[0] *= sc0;
            kq_sum[1] *= sc1;
            ps[0] *= sc0; ps[1] *= sc1;
            pz[0] *= sc0; pz[1] *= sc1;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] *= sc0; acc[c].x[1] *= sc1; acc[c].x[2] *= sc0; acc[c].x[3] *= sc1;
            }
        }
        kq.x[0] = expf(kq.x[0] - nm0);
        kq.x[1] = expf(kq.x[1] - nm1);
        kq.x[2] = expf(kq.x[2] - nm0);
        kq.x[3] = expf(kq.x[3] - nm1);
        kq_sum[0] += kq.x[0] + kq.x[2];
        kq_sum[1] += kq.x[1] + kq.x[3];
        return get_transposed(get_half2(kq));
    };

    auto cursor_init = [&](const int s) {
        cursor cu;
        const int p0 = 16 * s;
        cu.p0 = p0;
        cu.g  = 0;
        cu.sl = 0;
        if (p0 >= kv.S && p0 < kv.B) {
            cu.g  = (p0 - kv.S) / kv.G;
            cu.sl = ((p0 - kv.S) - cu.g * kv.G) / 16;
        }
        return cu;
    };

    auto cursor_next = [&](cursor & cu) {
        cu.p0 += 16;
        if (cu.p0 == kv.S) {
            cu.g = 0; cu.sl = 0;
        } else if (cu.p0 > kv.S && cu.p0 < kv.B) {
            if (++cu.sl == G16) {
                cu.sl = 0;
                ++cu.g;
            }
        }
    };

    auto issue = [&](const cursor & cu, const int stage, const bool with_meta) {
        const char * rec = fattn_sj_kvarn_rec(kv, cu.g);
        const char * ks  = rec + cu.sl * STRIP_BYTES;
        const uint32_t dst_s = ring_s + stage * STAGE_BYTES;
        if (with_meta) {
            const uint32_t dst_m = rmeta_s + (cu.g & 1) * RMETA_BYTES;
            for (int q = warp * WARP_SIZE + lane; q < RMETA_CHUNKS; q += nwarps * WARP_SIZE) {
                const char * src = q < RMETA_VCH / 16 ? rec + kv.k_scale + 16 * q : rec + kv.v_ch + 16 * (q - RMETA_VCH / 16);
                cp_async_cg_16<0>(dst_m + 16 * q, src);
            }
        }
        for (int q = warp * WARP_SIZE + lane; q < STAGE_CHUNKS; q += nwarps * WARP_SIZE) {
            const char * src;
            if (q < STRIP_BYTES / 16) {
                src = ks + 16 * q;
            } else if (q < 2 * STRIP_BYTES / 16) {
                src = ks + kv.v_payload + 16 * (q - STRIP_BYTES / 16);
            } else {
                const int m = q - 2 * STRIP_BYTES / 16;
                const int half_off = 16 * (m & 1);
                src = m < 2 ? rec + kv.k_tok   + cu.sl * 32 + half_off
                    : m < 4 ? rec + kv.v_scale + cu.sl * 32 + half_off
                    :         rec + kv.v_zero  + cu.sl * 32 + half_off;
            }
            cp_async_cg_16<0>(dst_s + 16 * q, src);
        }
        cp_async_commit();
    };

    static_assert((EXACT_BYTES / 16) % WARP_SIZE == 0, "exact strip staging must step by whole warps");
    auto issue_exact = [&](const char * rows, const size_t nb, const int type) {
        const float tq6_mag = type == GGML_TYPE_TQ6_0 ? TQ6_CENTROIDS[32 + lane] : 0.0f;
        for (int q = warp * WARP_SIZE + lane; q < EXACT_BYTES / 16; q += nwarps * WARP_SIZE) {
            const int r = q / (EXACT_ROW_BYTES / 16);
            const int j = q % (EXACT_ROW_BYTES / 16);
            if (type != GGML_TYPE_F16) {
                half2 * out = (half2 *) (ring + r * EXACT_ROW_BYTES + 16 * (j ^ (r & 7)));
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    out[i] = fattn_sj_kvarn_ring_pair_warp(rows + (size_t) r * nb, 4 * j + i, type, tq6_mag);
                }
            } else {
                cp_async_cg_16<0>(ring_s + r * EXACT_ROW_BYTES + 16 * (j ^ (r & 7)), rows + (size_t) r * nb + 16 * j);
            }
        }
        cp_async_commit();
    };
    auto exact_word = [&](const int r, const int cp) -> uint32_t {
        return *(const uint32_t *) (ring + r * EXACT_ROW_BYTES + 16 * ((cp >> 2) ^ (r & 7)) + 4 * (cp & 3));
    };

    cursor ci = cursor_init(s_lo);
    cursor cc = ci;
    int stage_i = 0;
    int stage_c = 0;
#pragma unroll
    for (int i = 0; i < NSTAGE - 1; ++i) {
        if (s_lo + i < s_hi) {
            issue(ci, stage_i, i == 0 || ci.sl == 0);
            cursor_next(ci);
            stage_i = stage_i + 1 == NSTAGE ? 0 : stage_i + 1;
        } else {
            cp_async_commit();
        }
    }

    int cur_id = INT_MIN;
    const char * cur_rm = nullptr;

    for (int s = s_lo; s < s_hi; ++s) {
        if (s + NSTAGE - 1 < s_hi) {
            issue(ci, stage_i, ci.sl == 0);
            cursor_next(ci);
            stage_i = stage_i + 1 == NSTAGE ? 0 : stage_i + 1;
        } else {
            cp_async_commit();
        }
        cp_async_wait_group<NSTAGE - 1>();
        __syncthreads();

        const int id = cc.g;
        const char * rm = rmeta + (cc.g & 1) * RMETA_BYTES;
        if (id != cur_id) {
            if (cur_id != INT_MIN) {
                record_change(cur_rm, rm);
            }
            setup_body(rm);
            cur_id = id;
            cur_rm = rm;
        }

        T_C kq;
        const char * st = ring + stage_c * STAGE_BYTES;
        const uint32_t * kwp = (const uint32_t *) st + lane;
        const uint32_t * vwp = (const uint32_t *) (st + STRIP_BYTES) + lane;

#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            T_A a;
            decode_pairs(kwp[c * WARP_SIZE], magic, a.x);
            mma(kq, a, qp[c]);
        }
        const half * ktok = (const half *) (st + META_KTOK);
        const half * vsh  = (const half *) (st + META_VS);
        const half * vzh  = (const half *) (st + META_VZ);
        const float tok0 = __half2float(ktok[lane >> 2]);
        const float tok1 = __half2float(ktok[(lane >> 2) + 8]);
        const float mk0  = __half2float(mrow[cc.p0 + (lane >> 2)]);
        const float mk1  = __half2float(mrow[cc.p0 + (lane >> 2) + 8]);
        kq.x[0] = tok0 * (kq.x[0] + c_r[0]) + mk0;
        kq.x[1] = tok0 * (kq.x[1] + c_r[1]) + mk0;
        kq.x[2] = tok1 * (kq.x[2] + c_r[0]) + mk1;
        kq.x[3] = tok1 * (kq.x[3] + c_r[1]) + mk1;

        T_B P = softmax(kq);

        const half2 vs_b0 = *(const half2 *) (vsh + 2 * (lane & 3));
        const half2 vs_b1 = *(const half2 *) (vsh + 2 * (lane & 3) + 8);
        P.x[0] = __hmul2(P.x[0], vs_b0);
        P.x[1] = __hmul2(P.x[1], vs_b1);
        const half2 vs_c  = __halves2half2(vsh[lane >> 2], vsh[(lane >> 2) + 8]);
        const float2 vz_c = make_float2(__half2float(vzh[lane >> 2]), __half2float(vzh[(lane >> 2) + 8]));
        const float2 pe0  = __half22float2(__hmul2(make_half2(kq.x[0], kq.x[2]), vs_c));
        const float2 pe1  = __half22float2(__hmul2(make_half2(kq.x[1], kq.x[3]), vs_c));
        ps[0] += pe0.x + pe0.y;
        ps[1] += pe1.x + pe1.y;
        pz[0] += kq.x[0] * vz_c.x + kq.x[2] * vz_c.y;
        pz[1] += kq.x[1] * vz_c.x + kq.x[3] * vz_c.y;

#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            T_A a;
            raw_pairs(vwp[c * WARP_SIZE], magic, a.x);
            mma(acc[c], a, P);
        }

        __syncthreads();
        cursor_next(cc);
        stage_c = stage_c + 1 == NSTAGE ? 0 : stage_c + 1;
    }

    if (cur_id != INT_MIN) {
        record_change(cur_rm, nullptr);
    }
    cp_async_wait_all();
    __syncthreads();

    if (split < n_ex) {
        setup_exact();
    }

    auto exact_p0   = [&](const int e) { return e < n_ex_sink ? 16 * e : kv.B + 16 * (e - n_ex_sink); };
    auto exact_row0 = [&](const int p0) { return p0 < kv.S ? p0 : kv.S + (p0 - kv.S) % kv.cap; };

    for (int e = split; e < n_ex; e += nsplit) {
        const int p0   = exact_p0(e);
        const int row0 = exact_row0(p0);
        const bool f16_sink = p0 < kv.S && kv.sink_type == GGML_TYPE_F16;
        const char * Kr = f16_sink ? fattn_sj_kvarn_sink_row(Kh, kv, nb11, row0, false) : Kh + (size_t) row0 * nb11;
        const char * Vr = f16_sink ? fattn_sj_kvarn_sink_row(Vh, kv, nb21, row0, true)  : Vh + (size_t) row0 * nb21;

        issue_exact(Kr, f16_sink ? kv.sink_stride : nb11, f16_sink ? GGML_TYPE_F16 : kv.type_k);
        cp_async_wait_all();
        __syncthreads();

        T_C kq;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            T_A a;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int row = (lane >> 2) + 8 * (l & 1);
                const int cp  = 8 * c + 4 * (l >> 1) + (lane & 3);
                a.x[l] = fattn_sj_kvarn_u32_as_half2(exact_word(row, cp));
            }
            mma(kq, a, qp[c]);
        }
        const float mk0 = __half2float(mrow[p0 + (lane >> 2)]);
        const float mk1 = __half2float(mrow[p0 + (lane >> 2) + 8]);
        kq.x[0] += mk0; kq.x[1] += mk0;
        kq.x[2] += mk1; kq.x[3] += mk1;
        const T_B P = softmax(kq);

        __syncthreads();
        issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v);
        cp_async_wait_all();
        __syncthreads();

#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            T_A a;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int row = 8 * (l >> 1) + (lane >> 2);
                const int cp  = 8 * c + 4 * (l & 1) + (lane & 3);
                a.x[l] = ggml_cuda_movmatrix(fattn_sj_kvarn_u32_as_half2(exact_word(row, cp)));
            }
            mma(acc[c], a, P);
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 4; off < WARP_SIZE; off <<= 1) {
        kq_sum[0] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[0], off, WARP_SIZE);
        kq_sum[1] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[1], off, WARP_SIZE);
    }

    const bool single = gridDim.y == 1;
    if constexpr (rot) {
        if (single) {
            const float inv0 = 1.0f / kq_sum[0], inv1 = 1.0f / kq_sum[1];
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[c].x[l] *= (l & 1) ? inv1 : inv0;
                }
            }
            sj_kvarn_rot256_frag<T_C, NTC>(acc, lane);
            kq_sum[0] = 1.0f;
            kq_sum[1] = 1.0f;
        }
    }

#pragma unroll
    for (int l = 0; l < 4; l += 2) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2 * (lane & 3) + e;
            if (!col_valid(col)) {
                continue;
            }
            const int head = head0 + col_head(col);
            const float inv = single ? 1.0f / kq_sum[e] : 1.0f;
            float * out = dst + ((size_t) (col_query(col) * ne02 + head) * gridDim.y + blockIdx.y) * D;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const int d = 16 * c + (lane >> 2) + 8 * (l >> 1);
                out[d] = acc[c].x[l + e] * inv;
            }
        }
    }

    if (!single && lane < 4) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2 * lane + e;
            if (col_valid(col)) {
                dst_meta[(size_t) (col_query(col) * ne02 + head0 + col_head(col)) * gridDim.y + blockIdx.y] = make_float2(kq_max[e], kq_sum[e]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, KV_rows, dst, dst_meta, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                     ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33, sj_kvarn_body, sj_kvarn_desc);
    NO_DEVICE_CODE;
#endif
}

// GGML_SJKVARN_V16: 0/unset = off (default), 1 = n_q in 9..16, 2 = n_q in 4..16.
static int v16_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_SJKVARN_V16");
        if (e == nullptr || *e == '\0') {
            return 0;
        }
        const int v = atoi(e);
        return v >= 0 && v <= 2 ? v : 0;
    }();
    return mode;
}

static int v16_nstage_env() {
    static const int v = [] {
        const char * e = getenv("GGML_SJKVARN_V16_NSTAGE");
        if (e == nullptr || *e == '\0') {
            return 0;
        }
        const int n = atoi(e);
        return (n == 2 || n == 4 || n == 6) ? n : 0;
    }();
    return v;
}

static int v16_waves_env() {
    static const int v = [] {
        const char * e = getenv("GGML_SJKVARN_V16_WAVES");
        if (e == nullptr || *e == '\0') {
            return 0;
        }
        const int n = atoi(e);
        return n >= 1 && n <= 16 ? n : 0;
    }();
    return v;
}

static int v16_splits_env() {
    static const int v = [] {
        const char * e = getenv("GGML_SJKVARN_V16_SPLITS");
        if (e == nullptr || *e == '\0') {
            return 0;
        }
        const int n = atoi(e);
        return n >= 1 ? n : 0;
    }();
    return v;
}

template <int max_warps, int NSTAGE, bool rot>
static fattn_kernel_t v16_kernel() {
    fattn_kernel_t kernel = flash_attn_ext_sj_kvarn_v16<max_warps, NSTAGE, rot>;
    const int id = ggml_cuda_get_device();
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {false};
    if (!initialized[id]) {
        const size_t limit = ggml_cuda_info().devices[id].smpbo;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int) limit));
        initialized[id] = true;
    }
    return kernel;
}

template <int max_warps, int NSTAGE, bool rot>
static int v16_ctas_per_sm(const int W) {
    const int id = ggml_cuda_get_device();
    static std::atomic<int> cached[GGML_CUDA_MAX_DEVICES][MAX_WARPS + 1];
    if (id >= 0 && id < GGML_CUDA_MAX_DEVICES && W >= 1 && W <= MAX_WARPS) {
        const int v = cached[id][W].load(std::memory_order_relaxed);
        if (v != 0) {
            return v - 1;
        }
    }
    const size_t shared = v16_shared_bytes(NSTAGE);
    int blocks = 0;
    if (shared <= ggml_cuda_info().devices[id].smpbo) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, v16_kernel<max_warps, NSTAGE, rot>(), WARP_SIZE * W, shared));
    }
    if (id >= 0 && id < GGML_CUDA_MAX_DEVICES && W >= 1 && W <= MAX_WARPS) {
        cached[id][W].store(blocks + 1, std::memory_order_relaxed);
    }
    return blocks;
}

static int v16_choose_parallel_blocks(const ggml_tensor * dst, const int ctas_per_sm) {
    const int nsm        = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int n_kv       = ggml_get_op_params_i32(dst, 6);
    const int ntiles_KV  = std::max(1, (n_kv + UNIT - 1) / UNIT);
    const int ntiles_dst = std::max(1, (int) (dst->src[1]->ne[2] * dst->src[0]->ne[3]));

    const int splits_ov = v16_splits_env();
    if (splits_ov > 0) {
        return std::min(splits_ov, ntiles_KV);
    }

    const int blocks_per_wave = std::max(1, nsm * std::max(1, ctas_per_sm));
    int waves = v16_waves_env();
    if (waves == 0) {
        // Multi-wave split at deep context hides exact-strip tail imbalance without spilling into a partial wave.
        waves = (ctas_per_sm == 1 && ntiles_KV >= 256) ? 2 : 1;
    }
    // Floor keeps total CTAs <= waves*blocks_per_wave, avoiding a stray partial tail wave.
    int pb = std::max(1, (waves * blocks_per_wave) / ntiles_dst);
    return std::min(pb, ntiles_KV);
}

template <int max_warps, int NSTAGE, bool rot>
static void v16_launch_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int W) {
    fattn_kernel_t kernel = v16_kernel<max_warps, NSTAGE, rot>();
    const size_t shared   = v16_shared_bytes(NSTAGE);
    const int ctas        = v16_ctas_per_sm<max_warps, NSTAGE, rot>(W);
    GGML_ASSERT(ctas > 0);
    const int pb = v16_choose_parallel_blocks(dst, ctas);

    static std::atomic<uint32_t> logged{0};
    const uint32_t bit = 1u << (W & 31);
    if (!(logged.fetch_or(bit) & bit)) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void *>(kernel)));
        GGML_LOG_INFO("SJKVARN V16: W=%d (max %d, n_q=%d, n_kv=%d) at %d CTA/SM (regs %d, local %zu B, nstage %d, rot %d, pb %d)\n",
            W, max_warps, (int) dst->src[0]->ne[1], ggml_get_op_params_i32(dst, 6), ctas, attr.numRegs, attr.localSizeBytes, NSTAGE, (int) rot, pb);
    }

    launch_fattn<D, MAX_NQ, NCOLS2>(ctx, dst, kernel, W, shared, UNIT,
        /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false, /*use_sparse=*/false, WARP_SIZE,
        nullptr, nullptr, pb);
}

template <int max_warps, bool rot>
static void v16_select_nstage(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int W) {
    int nst = v16_nstage_env();
    if (nst == 0) {
        // 1 CTA/SM benefits from 3 strips in flight (19.8 KB smem); 2 CTA/SM keeps 2 stages to preserve L1.
        nst = max_warps <= 6 ? 2 : 4;
    }
    if (nst == 6 && v16_ctas_per_sm<max_warps, 6, rot>(W) > 0) {
        v16_launch_impl<max_warps, 6, rot>(ctx, dst, W);
    } else if (nst >= 4 && v16_ctas_per_sm<max_warps, 4, rot>(W) > 0) {
        v16_launch_impl<max_warps, 4, rot>(ctx, dst, W);
    } else {
        v16_launch_impl<max_warps, 2, rot>(ctx, dst, W);
    }
}

template <bool rot>
static void v16_dispatch_warps(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int n_q = (int) dst->src[0]->ne[1];
    const int gqa = (int) (dst->src[0]->ne[2] / dst->src[1]->ne[2]);
    const int W   = (gqa * n_q + 7) / 8;
    GGML_ASSERT(W >= 1 && W <= MAX_WARPS);
    if (W <= 6) {
        v16_select_nstage<6, rot>(ctx, dst, W);
    } else if (W <= 8) {
        v16_select_nstage<8, rot>(ctx, dst, W);
    } else {
        v16_select_nstage<12, rot>(ctx, dst, W);
    }
}

} // namespace fattn_sj_kvarn_v16

bool ggml_cuda_flash_attn_ext_sj_kvarn_v16_supported(const ggml_tensor * dst) {
    using namespace fattn_sj_kvarn_v16;
    const int mode = v16_mode();
    if (mode == 0 || dst->src[6] == nullptr) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 5) != ((4 << 8) | 4) || ggml_get_op_params_i32(dst, 7) != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_flash_attn_ext_get_lsa_block(dst) > 0 || ggml_get_op_params_i32(dst, 4) > 0 ||
            dst->src[4] != nullptr || dst->src[7] != nullptr || dst->src[8] != nullptr || dst->src[9] != nullptr ||
            ggml_cuda_fattn_sj_live_records != nullptr) {
        return false;
    }
    if (dst->src[5] && ((dst->src[5]->buffer &&
            ggml_cuda_qsa_resident_is_buft(ggml_backend_buffer_get_type(dst->src[5]->buffer))) ||
            ggml_cuda_qsa_strip_body(dst->src[5]))) {
        return false;
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (Q == nullptr || K == nullptr || V == nullptr || mask == nullptr) {
        return false;
    }
    const int64_t n_q = Q->ne[1];
    const int64_t min_nq = mode == 2 ? 4 : 9;
    if (n_q < min_nq || n_q > MAX_NQ || dst->ne[2] != n_q) {
        return false;
    }
    if (Q->ne[0] != D || K->ne[0] != D || V->ne[0] != D ||
            Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (mask->type != GGML_TYPE_F16 || mask->ne[1] < n_q || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    const auto ring_type_ok = [](const ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_Q8_0 || t == GGML_TYPE_TQ6_0;
    };
    if (!ring_type_ok(K->type) || !ring_type_ok(V->type)) {
        return false;
    }
    const int32_t n_kv_pad = ggml_get_op_params_i32(dst, 6);
    if (n_kv_pad <= 0 || n_kv_pad % FATTN_KQ_STRIDE != 0 || mask->ne[0] < n_kv_pad) {
        return false;
    }
    if (((uintptr_t) mask->data) % 16 != 0 || mask->nb[1] % 16 != 0 ||
            ((uintptr_t) Q->data) % 8 != 0 || Q->nb[1] % 16 != 0 || Q->nb[2] % 8 != 0) {
        return false;
    }
    if (K->ne[2] <= 0 || Q->ne[2] % K->ne[2] != 0) {
        return false;
    }
    const int gqa = (int) (Q->ne[2] / K->ne[2]);
    if (gqa < 2 || gqa > NCOLS2 || (gqa & 1) != 0) {
        return false;
    }
    const int W = (gqa * (int) n_q + 7) / 8;
    if (W < 1 || W > MAX_WARPS) {
        return false;
    }
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return cp_async_available(cc) && turing_mma_available(cc);
}

void ggml_cuda_flash_attn_ext_sj_kvarn_v16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fattn_sj_kvarn_v16;
    if (ggml_cuda_fattn_sj_kvarn_rot(dst)) {
        ggml_cuda_pool_alloc<float> q_rot(ctx.pool(), ggml_nelements(dst->src[0]));
        ggml_tensor q2;
        ggml_cuda_sj_kvarn_rot256_q_pass(ctx, dst->src[0], q_rot.ptr, &q2);
        ggml_tensor d2 = *dst;
        d2.src[0] = &q2;
        v16_dispatch_warps<true>(ctx, &d2);
        return;
    }
    v16_dispatch_warps<false>(ctx, dst);
}
