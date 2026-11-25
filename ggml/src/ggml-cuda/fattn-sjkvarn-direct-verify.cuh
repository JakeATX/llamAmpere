// Verify-tuned INT8 IMMA SJ-KVaRN 4/4 decode/verify kernel (T = 1..64+, covering T = 4..8 and T = 16..32 MTP4/DSpine).
//
// Opt-in via GGML_SJKVARN_DIRECT_VERIFY=1 (default 0: zero behavior change when unset).
// Fixes baseline sjkvarn_direct's exclusions & bottlenecks (255 regs, 16-24 B spills, ~90 KB SMEM, 1 CTA/SM,
// unbalanced W-strip windowing, stream_max=8 cutoff, no GQA=6 compact tile, no Q-tiling, no rot/sinks/far_end/head_mask):
//   1. K x Q via IMMA.16832.U8.S8 (285 TFLOPS on SM 8.6) with per-head-column INT8 Q quantization + 7.5*sum(eps) DC
//      correction + KBIAS 0x4BC00000 exact bit extraction.
//   2. P x V via paired 2-strip IMMA.16832.U8.U8 (285 TFLOPS on SM 8.6) within each record (GGML_SJKVARN_DIRECT_VERIFY_PV32=1
//      default), with automatic 1-strip IMMA.16816.U8.U8 (142 TFLOPS) fallback on odd/boundary strips, using per-column
//      P*Vscale peak quantization into [0, 255] clamped at 8388863.0f to prevent fast-math div-approx wraparound.
//   3. Balanced cooperative strip ring (NSTAGE = 4 -> ~19.4 KB SMEM, 0 Q-SMEM, 0 warp barrier skew) + Q-tiling
//      across blockIdx.x (ncols1 = 8 compact GQA=6, or ncols1 = 4/8 general rows) + parallel_blocks across all 82 SMs.
//   4. Non-overlapping lifetimes: Q read on record change via L1/LDG; qp[16] scoped strictly after body loop;
//      full fattn_sj_kvarn_ctx constructed lazily after body loop -> <= 168 regs with 0 spills and 2-3 CTAs/SM!
//   5. Full feature support: GQA 1..8, compact GQA=6 (T=4..32+), fused H256 rotation, attention sinks,
//      per-head F16 masks (ne[2] > 1), INKKV far_end, and LSE parallel-block combine.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "mma.cuh"
#include "fattn-sjkvarn-rot.cuh"
#include "fattn-sjkvarn-stream.cuh"
#include "ggml-inksj.h"

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace fattn_sj_kvarn_direct_verify {

using namespace ggml_cuda_mma;
using fattn_sj_kvarn_stream::D;
using fattn_sj_kvarn_stream::NTC;
using fattn_sj_kvarn_stream::NCOLS2;
using fattn_sj_kvarn_stream::UNIT;
using fattn_sj_kvarn_stream::STRIP_BYTES;
using fattn_sj_kvarn_stream::META_BYTES;
using fattn_sj_kvarn_stream::META_KTOK;
using fattn_sj_kvarn_stream::META_VS;
using fattn_sj_kvarn_stream::META_VZ;
using fattn_sj_kvarn_stream::STAGE_BYTES;
using fattn_sj_kvarn_stream::STAGE_CHUNKS;
using fattn_sj_kvarn_stream::EXACT_ROW_BYTES;
using fattn_sj_kvarn_stream::EXACT_BYTES;
using fattn_sj_kvarn_stream::RMETA_KZ;
using fattn_sj_kvarn_stream::RMETA_VCH;
using fattn_sj_kvarn_stream::RMETA_BYTES;
using fattn_sj_kvarn_stream::RMETA_CHUNKS;
using fattn_sj_kvarn_stream::COL_STRIDE;
using fattn_sj_kvarn_stream::SLOT_FLOATS;
using fattn_sj_kvarn_stream::SLOT_BYTES;
using fattn_sj_kvarn_stream::T_A;
using fattn_sj_kvarn_stream::T_B;
using fattn_sj_kvarn_stream::T_C;
using fattn_sj_kvarn_stream::split_bytes;
using fattn_sj_kvarn_stream::ring_bytes;
using fattn_sj_kvarn_stream::split_sync;
using fattn_sj_kvarn_stream::slot_acc;
using fattn_sj_kvarn_stream::slot_meta;
using fattn_sj_kvarn_stream::cp_async_commit;
using fattn_sj_kvarn_stream::cp_async_wait_group;

constexpr int32_t KBIAS_I = 0x4BC00000;   // 1.5 * 2^24, binade [2^24, 2^25): ulp = 2
constexpr float   KBIAS_F = 25165824.0f;
constexpr int32_t VBIAS_I = 0x4B000000;   // 2^23, binade [2^23, 2^24): ulp = 1
constexpr float   VBIAS_F = 8388608.0f;
constexpr float   VBIAS_CLAMP_F = 8388863.0f; // 2^23 + 255: guards against div.approx u8 overflow -> 256

struct body_cursor {
    int p0;
    int g;
    int sl;
};

static __device__ __forceinline__ uint32_t dv_half2_as_u32(const half2 h) {
    uint32_t u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

// C += A(16 tokens x 32 channels, u8) . B(32 channels x 8 cols, s8), s32 accumulate (285T on SM 8.6)
static __device__ __forceinline__ void imma_u8s8_32(int32_t & c0, int32_t & c1, int32_t & c2, int32_t & c3,
        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// C = A(16 channels x 16 tokens, u8) . B(16 tokens x 8 cols, u8) + bias, s32 (142T on SM 8.6)
static __device__ __forceinline__ void imma_u8u8_16(int32_t & c0, int32_t & c1, int32_t & c2, int32_t & c3,
        const uint32_t a0, const uint32_t a1, const uint32_t b0, const int32_t bias) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %7, %7, %7};"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(bias));
#else
    GGML_UNUSED_VARS(c0, c1, c2, c3, a0, a1, b0, bias);
    NO_DEVICE_CODE;
#endif
}

// C = A(16 channels x 32 tokens, u8) . B(32 tokens x 8 cols, u8) + bias, s32 (285T on SM 8.6)
// Max possible sum = 32 * 240 * 255 = 1,958,400 < 2^23 (8,388,608), so C + VBIAS_I fits [2^23, 2^24) bit-exactly!
static __device__ __forceinline__ void imma_u8u8_32(int32_t & c0, int32_t & c1, int32_t & c2, int32_t & c3,
        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1, const int32_t bias) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(bias));
#else
    GGML_UNUSED_VARS(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, bias);
    NO_DEVICE_CODE;
#endif
}

template <int max_warps, int NSTAGE, int ncols1, bool rot = false, sj_kvarn_tile tile = sj_kvarn_tile::rows, bool pair_32 = true>
__launch_bounds__(WARP_SIZE*max_warps, (max_warps == 3 || max_warps == 4) ? 3 : (max_warps == 5 || max_warps == 6) ? 2 : 1)
static __global__ void flash_attn_ext_sj_kvarn_direct_verify(
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(AMPERE_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    GGML_UNUSED_VARS(KV_max, KV_rows, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb03, ne10, ne13,
                     nb13, nb23, ne31, ne33, nb33);

    extern __shared__ __align__(16) char smem[];

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;
    const int n_q    = int(ne01.z);
    constexpr bool compact = tile == sj_kvarn_tile::compact;
    static_assert(!compact || (max_warps >= 3 && max_warps <= 6), "compact tile uses 3..6 warps");
    const int q_base  = (int) blockIdx.x * ncols1;
    const int nq_tile = min(ncols1, n_q - q_base);

    // For compact GQA=6, all max_warps cooperate on one KV ring (KW = 1).
    // For rows with ncols1=4 multi-Q tiling, n_q > 8 uses NT=4, KW=1.
    // For rows with ncols1=8 single-Q-tile (n_q <= 8), NT = nq_tile, KW = nwarps / NT.
    const bool compact_or_fixed = compact || max_warps == 5 || max_warps == 6 || (ncols1 == 4);
    const int NT = compact_or_fixed ? max_warps : nq_tile;
    const int KW = compact_or_fixed ? 1 : nwarps / NT;
    const int nt = compact_or_fixed ? warp : warp % NT;
    const int kw = compact_or_fixed ? 0 : warp / NT;

    const int z_KV   = blockIdx.z;
    const int gqa    = ne02 / ne12;
    const int head0  = z_KV * gqa;
    const int hcol   = lane >> 2;
    const int npair  = 6 * nq_tile;
    auto col_query = [&](const int col) -> int  { return compact ? (8*nt + col < npair ? (8*nt + col)/6 : 0) : (nt < nq_tile ? nt : 0); };
    auto col_head  = [&](const int col) -> int  { return compact ? (8*nt + col) % 6 : col; };
    auto col_valid = [&](const int col) -> bool { return compact ? (8*nt + col < npair) : (nt < nq_tile && col < gqa); };
    const bool hvalid = col_valid(hcol);

    const int kv_G          = sj_kvarn_desc[GGML_SJKVARN_DESC_G];
    const int kv_S          = sj_kvarn_desc[GGML_SJKVARN_DESC_S];
    const int kv_B          = sj_kvarn_desc[GGML_SJKVARN_DESC_B];
    const int kv_rec_bytes  = sj_kvarn_desc[GGML_SJKVARN_DESC_RECBYTES];
    const int kv_rec_stride = sj_kvarn_desc[GGML_SJKVARN_DESC_HKV] * kv_rec_bytes;
    const int32_t * kv_table = sj_kvarn_desc[GGML_SJKVARN_DESC_TABLE] ? sj_kvarn_desc + GGML_SJKVARN_DESC_N_ENTRIES : nullptr;
    const char * kv_body_base = sj_kvarn_body + (size_t) z_KV * kv_rec_bytes;
    const int G16 = kv_G / 16;

    const int nsplit = gridDim.y * KW;
    const int split  = blockIdx.y * KW + kw;
    const int sb0    = kv_S / 16;
    const int sb1    = kv_B / 16;
    const int nbody  = sb1 - sb0;
    const int s_lo   = sb0 + (int) (((int64_t) nbody *  split     ) / nsplit);
    const int s_hi   = sb0 + (int) (((int64_t) nbody * (split + 1)) / nsplit);

    const float * Qf = (const float *) (Q + (size_t) (q_base + col_query(hcol))*nb01 + (size_t) (head0 + (hvalid ? col_head(hcol) : 0))*nb02);

    const int c0 = 2 * (lane & 3);
    // padding columns (c0 past the last head / query) read the first valid row instead of past the mask's end
    const bool c0_valid = col_valid(c0);
    const half * mrow0 = (const half *) (mask + (size_t) (q_base + (c0_valid ? col_query(c0) : 0))*nb31 + (ne32 > 1 ? (size_t) (head0 + (c0_valid ? col_head(c0) : 0))*nb32 : 0));
    const half * mrow1 = (ne32 > 1 && col_valid(c0 + 1)) ? (const half *) ((const char *) mrow0 + nb32) : mrow0;

    static_assert(NSTAGE * STAGE_BYTES >= EXACT_BYTES, "ring must hold one exact K or V strip");
    static_assert(NSTAGE <= 6, "rmeta buffering requires NSTAGE <= 6");
    static_assert(!pair_32 || NSTAGE >= 4, "paired 2-strip PV requires NSTAGE >= 4");
    char * ring  = smem + (size_t) kw * split_bytes(NSTAGE, false);
    char * rmeta = ring + NSTAGE * STAGE_BYTES;
    const uint32_t ring_s  = ggml_cuda_cvta_generic_to_shared(ring);
    const uint32_t rmeta_s = ring_s + NSTAGE * STAGE_BYTES;

    auto sync_split = [&]() {
        if (NT == 1) {
            __syncwarp();
        } else {
            split_sync(1 + kw, NT * WARP_SIZE);
        }
    };

    T_C   acc[NTC];      // FP32 output accumulator, unnormalized, in units of current record's Vch (exact: 1)
    float kq_max[2] = {-FLT_MAX/2, -FLT_MAX/2};
    float kq_sum[2] = {0.0f, 0.0f};
    float pz[2]     = {0.0f, 0.0f};

    auto load_q = [&](const int c, float2 & q0, float2 & q1) {
        if (hvalid) {
            q0 = *(const float2 *) (Qf + 16*c + 2*(lane & 3));
            q1 = *(const float2 *) (Qf + 16*c + 2*(lane & 3) + 8);
            q0.x *= scale; q0.y *= scale; q1.x *= scale; q1.y *= scale;
        } else {
            q0 = make_float2(0.0f, 0.0f);
            q1 = make_float2(0.0f, 0.0f);
        }
    };

    auto record_change = [&](const char * rec_old, const char * rec_new) {
        if (rec_old != nullptr) {
            float z0s = pz[0], z1s = pz[1];
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                z0s += __shfl_xor_sync(0xFFFFFFFF, z0s, off, WARP_SIZE);
                z1s += __shfl_xor_sync(0xFFFFFFFF, z1s, off, WARP_SIZE);
            }
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += z0s; acc[c].x[1] += z1s;
                acc[c].x[2] += z0s; acc[c].x[3] += z1s;
            }
            pz[0] = 0.0f;
            pz[1] = 0.0f;
        }
        const uint32_t * vo = rec_old != nullptr ? (const uint32_t *) (rec_old + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;
        const uint32_t * vn = rec_new != nullptr ? (const uint32_t *) (rec_new + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            float r_lo = 1.0f, r_hi = 1.0f;
            if (vo != nullptr) {
                const float2 v = __half22float2(fattn_sj_kvarn_u32_as_half2(vo[c]));
                r_lo = v.x; r_hi = v.y;
            }
            if (vn != nullptr) {
                const float2 v = __half22float2(fattn_sj_kvarn_u32_as_half2(vn[c]));
                r_lo = __fdividef(r_lo, v.x); r_hi = __fdividef(r_hi, v.y);
            }
            acc[c].x[0] *= r_lo; acc[c].x[1] *= r_lo;
            acc[c].x[2] *= r_hi; acc[c].x[3] *= r_hi;
        }
    };

    auto softmax_update_rescale = [&](const float mx0, const float mx1, float & w0, float & w1) {
        const float nm0 = fmaxf(kq_max[0], mx0);
        const float nm1 = fmaxf(kq_max[1], mx1);
        if (__any_sync(0xFFFFFFFF, nm0 != kq_max[0] || nm1 != kq_max[1])) {
            const float sc0 = __expf(kq_max[0] - nm0);
            const float sc1 = __expf(kq_max[1] - nm1);
            kq_max[0] = nm0;
            kq_max[1] = nm1;
            kq_sum[0] *= sc0;
            kq_sum[1] *= sc1;
            pz[0] *= sc0;
            pz[1] *= sc1;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] *= sc0; acc[c].x[1] *= sc1;
                acc[c].x[2] *= sc0; acc[c].x[3] *= sc1;
            }
        }
        w0 = __expf(mx0 - nm0);
        w1 = __expf(mx1 - nm1);
    };

    auto rec_of = [&](const int g) -> const char * {
        const int record = kv_table ? kv_table[g] : g;
        return kv_body_base + (size_t) record * kv_rec_stride;
    };
    auto cursor_init = [&](const int s) -> body_cursor {
        body_cursor cu;
        const int p0 = 16 * s;
        cu.p0 = p0;
        cu.g  = 0;
        cu.sl = 0;
        if (p0 >= kv_S && p0 < kv_B) {
            cu.g  = (p0 - kv_S) / kv_G;
            cu.sl = ((p0 - kv_S) - cu.g * kv_G) / 16;
        }
        return cu;
    };
    auto cursor_next = [&](body_cursor & cu) {
        cu.p0 += 16;
        if (++cu.sl == G16) {
            cu.sl = 0;
            ++cu.g;
        }
    };

    auto issue = [&](const body_cursor & cu, const int stage, const bool with_meta) {
        const char * rec = rec_of(cu.g);
        const char * ks  = rec + cu.sl * STRIP_BYTES;
        const int v_payload = (D/2) * kv_G;
        const int k_scale   = v_payload + (D/2) * kv_G;
        const int k_tok     = k_scale + 4 * D;
        const int v_ch      = k_tok   + 2 * kv_G;
        const int v_scale   = v_ch    + 2 * D;
        const int v_zero    = v_scale + 2 * kv_G;
        const uint32_t dst_s = ring_s + stage * STAGE_BYTES;
        if (with_meta) {
            const uint32_t dst_m = rmeta_s + (cu.g & 1) * RMETA_BYTES;
            for (int q = nt*WARP_SIZE + lane; q < RMETA_CHUNKS; q += NT*WARP_SIZE) {
                const char * src = q < RMETA_VCH/16 ? rec + k_scale + 16*q : rec + v_ch + 16*(q - RMETA_VCH/16);
                cp_async_cg_16<0>(dst_m + 16*q, src);
            }
        }
        for (int q = nt*WARP_SIZE + lane; q < STAGE_CHUNKS; q += NT*WARP_SIZE) {
            const char * src;
            if (q < STRIP_BYTES/16) {
                src = ks + 16*q;
            } else if (q < 2*STRIP_BYTES/16) {
                src = ks + v_payload + 16*(q - STRIP_BYTES/16);
            } else {
                const int m = q - 2*STRIP_BYTES/16;
                const int half_off = 16 * (m & 1);
                src = m < 2 ? rec + k_tok   + cu.sl*32 + half_off
                    : m < 4 ? rec + v_scale + cu.sl*32 + half_off
                    :         rec + v_zero  + cu.sl*32 + half_off;
            }
            cp_async_cg_16<0>(dst_s + 16*q, src);
        }
        cp_async_commit();
    };

    // Scoped body loop: qi[NTC], sq_lo/sq_hi/c_r live strictly inside the body block so exact qp[NTC] reuses them.
    {
        uint32_t qi[NTC];
        float c_r[2]   = {0.0f, 0.0f};
        float sq_lo[2] = {0.0f, 0.0f};
        float sq_hi[2] = {0.0f, 0.0f};

        auto setup_body = [&](const char * rm) {
            const uint2 * ks2 = (const uint2 *) (rm            + 2*(lane & 3)*(D/4));
            const uint2 * kz2 = (const uint2 *) (rm + RMETA_KZ + 2*(lane & 3)*(D/4));
            half2 am = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                float2 q0, q1;
                load_q(c, q0, q1);
                const uint2 ks = ks2[c];
                const half2 p0 = __hmul2(__floats2half2_rn(q0.x, q0.y), fattn_sj_kvarn_u32_as_half2(ks.x));
                const half2 p1 = __hmul2(__floats2half2_rn(q1.x, q1.y), fattn_sj_kvarn_u32_as_half2(ks.y));
                am = __hmax2(am, __hmax2(__habs2(p0), __habs2(p1)));
            }
            float amax = fmaxf(__low2float(am), __high2float(am));
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, WARP_SIZE));
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, WARP_SIZE));
            const float sq   = amax > 0.0f ? amax * (1.0f / 126.5f) : 0.0f;
            const float isq  = amax > 0.0f ? 126.5f / amax : 0.0f;
            const half2 isq2 = __float2half2_rn(isq);
            const half2 nsq2 = __float2half2_rn(-sq);
            const half2 mag  = fattn_sj_kvarn_u32_as_half2(0x64806480U); // 1024 + 128
            float cr   = 0.0f;
            half2 eps2 = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                float2 q0, q1;
                load_q(c, q0, q1);
                const uint2 ks = ks2[c];
                const uint2 kz = kz2[c];
                const half2 p0 = __hmul2(__floats2half2_rn(q0.x, q0.y), fattn_sj_kvarn_u32_as_half2(ks.x));
                const half2 p1 = __hmul2(__floats2half2_rn(q1.x, q1.y), fattn_sj_kvarn_u32_as_half2(ks.y));
                const half2 h0 = __hfma2(p0, isq2, mag);
                const half2 h1 = __hfma2(p1, isq2, mag);
                qi[c] = __byte_perm(dv_half2_as_u32(h0), dv_half2_as_u32(h1), 0x6240) ^ 0x80808080U;
                eps2 = __hadd2(eps2, __hfma2(__hsub2(h0, mag), nsq2, p0));
                eps2 = __hadd2(eps2, __hfma2(__hsub2(h1, mag), nsq2, p1));
                const float2 kz0 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.x));
                const float2 kz1 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.y));
                cr += q0.x*kz0.x + q0.y*kz0.y + q1.x*kz1.x + q1.y*kz1.y;
            }
            cr += 7.5f * (__low2float(eps2) + __high2float(eps2));
            cr += __shfl_xor_sync(0xFFFFFFFF, cr, 1, WARP_SIZE);
            cr += __shfl_xor_sync(0xFFFFFFFF, cr, 2, WARP_SIZE);
            c_r[0] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3),     WARP_SIZE);
            c_r[1] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3) + 4, WARP_SIZE);
            const float sqc0 = __shfl_sync(0xFFFFFFFF, sq, 8*(lane & 3),     WARP_SIZE);
            const float sqc1 = __shfl_sync(0xFFFFFFFF, sq, 8*(lane & 3) + 4, WARP_SIZE);
            sq_lo[0] = 0.5f * sqc0;         sq_lo[1] = 0.5f * sqc1;
            sq_hi[0] = (1.0f/32.0f) * sqc0; sq_hi[1] = (1.0f/32.0f) * sqc1;
        };

        auto kq_strip = [&](const char * st, const int p0, T_C & kq) {
            const uint32_t * kwp = (const uint32_t *) st + lane;
            int32_t k0 = KBIAS_I, k1 = KBIAS_I, k2 = KBIAS_I, k3 = KBIAS_I;
#pragma unroll
            for (int c = 0; c < NTC; c += 2) {
                const uint32_t x0 = kwp[c*WARP_SIZE];
                const uint32_t x1 = kwp[(c + 1)*WARP_SIZE];
                imma_u8s8_32(k0, k1, k2, k3, x0 & 0x0F0F0F0FU, x0 & 0xF0F0F0F0U, x1 & 0x0F0F0F0FU, x1 & 0xF0F0F0F0U, qi[c], qi[c + 1]);
            }
            const half * ktok = (const half *) (st + META_KTOK);
            const float tok0  = __half2float(ktok[lane >> 2]);
            const float tok1  = __half2float(ktok[(lane >> 2) + 8]);
            const float mk0_0 = __half2float(mrow0[p0 + (lane >> 2)]);
            const float mk0_1 = mrow1 != mrow0 ? __half2float(mrow1[p0 + (lane >> 2)]) : mk0_0;
            const float mk1_0 = __half2float(mrow0[p0 + (lane >> 2) + 8]);
            const float mk1_1 = mrow1 != mrow0 ? __half2float(mrow1[p0 + (lane >> 2) + 8]) : mk1_0;
            kq.x[0] = tok0 * (sq_lo[0] * (__int_as_float(k0) - KBIAS_F) + c_r[0]) + mk0_0;
            kq.x[1] = tok0 * (sq_lo[1] * (__int_as_float(k1) - KBIAS_F) + c_r[1]) + mk0_1;
            kq.x[2] = tok1 * (sq_hi[0] * (__int_as_float(k2) - KBIAS_F) + c_r[0]) + mk1_0;
            kq.x[3] = tok1 * (sq_hi[1] * (__int_as_float(k3) - KBIAS_F) + c_r[1]) + mk1_1;
        };

        auto pack_pu = [&](const float a0, const float a1, const float a2, const float a3,
                           const float rp0, const float rp1, const float fs0, const float fs1,
                           float & eps0, float & eps1) -> uint32_t {
            const float f0 = fminf(fmaf(a0, rp0, VBIAS_F), VBIAS_CLAMP_F);
            const float f1 = fminf(fmaf(a1, rp1, VBIAS_F), VBIAS_CLAMP_F);
            const float f2 = fminf(fmaf(a2, rp0, VBIAS_F), VBIAS_CLAMP_F);
            const float f3 = fminf(fmaf(a3, rp1, VBIAS_F), VBIAS_CLAMP_F);
            eps0 += (a0 + a2) - fs0 * ((f0 - VBIAS_F) + (f2 - VBIAS_F));
            eps1 += (a1 + a3) - fs1 * ((f1 - VBIAS_F) + (f3 - VBIAS_F));
            const uint32_t m0 = __float_as_uint(f0);
            const uint32_t m1 = __float_as_uint(f1);
            const uint32_t m2 = __float_as_uint(f2);
            const uint32_t m3 = __float_as_uint(f3);
            const uint32_t t0 = (uint32_t) ggml_cuda_movmatrix((int) __byte_perm(m0, m1, 0x5410));
            const uint32_t t1 = (uint32_t) ggml_cuda_movmatrix((int) __byte_perm(m2, m3, 0x5410));
            return __byte_perm(t0, t1, 0x6240);
        };

        constexpr int LEAD = pair_32 ? 2 : (NSTAGE - 1);
        body_cursor ci = cursor_init(s_lo);
        body_cursor cc = ci;
        int stage_i = 0;
        int stage_c = 0;
#pragma unroll
        for (int i = 0; i < LEAD; ++i) {
            if (s_lo + i < s_hi) {
                issue(ci, stage_i, i == 0 || ci.sl == 0);
                cursor_next(ci);
                stage_i = (stage_i + 1) % NSTAGE;
            } else {
                cp_async_commit();
            }
        }

        int          cur_id = INT_MIN;
        const char * cur_rm = nullptr;

        for (int s = s_lo; s < s_hi; ) {
            const bool do_pair = pair_32 && (s + 1 < s_hi) && (cc.sl + 1 < G16);
            if constexpr (pair_32) {
                if (do_pair) {
                    if (s + 2 < s_hi) {
                        issue(ci, stage_i, ci.sl == 0);
                        cursor_next(ci);
                        stage_i = (stage_i + 1) % NSTAGE;
                    } else {
                        cp_async_commit();
                    }
                    if (s + 3 < s_hi) {
                        issue(ci, stage_i, ci.sl == 0);
                        cursor_next(ci);
                        stage_i = (stage_i + 1) % NSTAGE;
                    } else {
                        cp_async_commit();
                    }
                    cp_async_wait_group<2>();
                    sync_split();

                    const int    id = cc.g;
                    const char * rm = rmeta + (id & 1) * RMETA_BYTES;
                    if (id != cur_id) {
                        if (cur_id != INT_MIN) {
                            record_change(cur_rm, rm);
                        }
                        setup_body(rm);
                        cur_id = id;
                        cur_rm = rm;
                    }

                    const int stage_c1 = (stage_c + 1) % NSTAGE;
                    const char * st0 = ring + stage_c  * STAGE_BYTES;
                    const char * st1 = ring + stage_c1 * STAGE_BYTES;
                    T_C kq0, kq1;
                    kq_strip(st0, cc.p0,      kq0);
                    kq_strip(st1, cc.p0 + 16, kq1);

                    float mx0 = fmaxf(fmaxf(kq0.x[0], kq0.x[2]), fmaxf(kq1.x[0], kq1.x[2]));
                    float mx1 = fmaxf(fmaxf(kq0.x[1], kq0.x[3]), fmaxf(kq1.x[1], kq1.x[3]));
#pragma unroll
                    for (int off = 4; off < WARP_SIZE; off <<= 1) {
                        mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, off, WARP_SIZE));
                        mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, off, WARP_SIZE));
                    }
                    mx0 = fmaxf(mx0, -FLT_MAX/2);
                    mx1 = fmaxf(mx1, -FLT_MAX/2);
                    float w0, w1;
                    softmax_update_rescale(mx0, mx1, w0, w1);

                    kq0.x[0] = __expf(kq0.x[0] - mx0); kq0.x[1] = __expf(kq0.x[1] - mx1);
                    kq0.x[2] = __expf(kq0.x[2] - mx0); kq0.x[3] = __expf(kq0.x[3] - mx1);
                    kq1.x[0] = __expf(kq1.x[0] - mx0); kq1.x[1] = __expf(kq1.x[1] - mx1);
                    kq1.x[2] = __expf(kq1.x[2] - mx0); kq1.x[3] = __expf(kq1.x[3] - mx1);
                    kq_sum[0] += w0 * ((kq0.x[0] + kq0.x[2]) + (kq1.x[0] + kq1.x[2]));
                    kq_sum[1] += w1 * ((kq0.x[1] + kq0.x[3]) + (kq1.x[1] + kq1.x[3]));

                    const half * vsh0 = (const half *) (st0 + META_VS);
                    const half * vzh0 = (const half *) (st0 + META_VZ);
                    const half * vsh1 = (const half *) (st1 + META_VS);
                    const half * vzh1 = (const half *) (st1 + META_VZ);
                    const float vs0_a = __half2float(vsh0[lane >> 2]);
                    const float vs0_b = __half2float(vsh0[(lane >> 2) + 8]);
                    const float vs1_a = __half2float(vsh1[lane >> 2]);
                    const float vs1_b = __half2float(vsh1[(lane >> 2) + 8]);
                    const float vz0_a = __half2float(vzh0[lane >> 2]);
                    const float vz0_b = __half2float(vzh0[(lane >> 2) + 8]);
                    const float vz1_a = __half2float(vzh1[lane >> 2]);
                    const float vz1_b = __half2float(vzh1[(lane >> 2) + 8]);

                    const float pv0_0 = kq0.x[0]*vs0_a, pv0_1 = kq0.x[1]*vs0_a, pv0_2 = kq0.x[2]*vs0_b, pv0_3 = kq0.x[3]*vs0_b;
                    const float pv1_0 = kq1.x[0]*vs1_a, pv1_1 = kq1.x[1]*vs1_a, pv1_2 = kq1.x[2]*vs1_b, pv1_3 = kq1.x[3]*vs1_b;
                    float pm0 = fmaxf(fmaxf(pv0_0, pv0_2), fmaxf(pv1_0, pv1_2));
                    float pm1 = fmaxf(fmaxf(pv0_1, pv0_3), fmaxf(pv1_1, pv1_3));
#pragma unroll
                    for (int off = 4; off < WARP_SIZE; off <<= 1) {
                        pm0 = fmaxf(pm0, __shfl_xor_sync(0xFFFFFFFF, pm0, off, WARP_SIZE));
                        pm1 = fmaxf(pm1, __shfl_xor_sync(0xFFFFFFFF, pm1, off, WARP_SIZE));
                    }
                    const float rp0 = pm0 > 0.0f ? __fdividef(255.0f, pm0) : 0.0f;
                    const float rp1 = pm1 > 0.0f ? __fdividef(255.0f, pm1) : 0.0f;
                    const float fs0 = pm0 * (1.0f / 255.0f);
                    const float fs1 = pm1 * (1.0f / 255.0f);
                    float eps0 = 0.0f, eps1 = 0.0f;
                    const uint32_t pu0 = pack_pu(pv0_0, pv0_1, pv0_2, pv0_3, rp0, rp1, fs0, fs1, eps0, eps1);
                    const uint32_t pu1 = pack_pu(pv1_0, pv1_1, pv1_2, pv1_3, rp0, rp1, fs0, fs1, eps0, eps1);
                    pz[0] += w0 * ((kq0.x[0]*vz0_a + kq0.x[2]*vz0_b) + (kq1.x[0]*vz1_a + kq1.x[2]*vz1_b) + 7.5f * eps0);
                    pz[1] += w1 * ((kq0.x[1]*vz0_a + kq0.x[3]*vz0_b) + (kq1.x[1]*vz1_a + kq1.x[3]*vz1_b) + 7.5f * eps1);
                    const float f0 = w0 * fs0;
                    const float f1 = w1 * fs1;
                    const float f2 = f0 * (1.0f / 16.0f), f3 = f1 * (1.0f / 16.0f);

                    const uint32_t * vwp0 = (const uint32_t *) (st0 + STRIP_BYTES) + lane;
                    const uint32_t * vwp1 = (const uint32_t *) (st1 + STRIP_BYTES) + lane;
#pragma unroll
                    for (int c = 0; c < NTC; ++c) {
                        const uint32_t x0 = vwp0[c*WARP_SIZE];
                        const uint32_t x1 = vwp1[c*WARP_SIZE];
                        int32_t r0, r1, r2, r3;
                        imma_u8u8_32(r0, r1, r2, r3,
                                     x0 & 0x0F0F0F0FU, x0 & 0xF0F0F0F0U,
                                     x1 & 0x0F0F0F0FU, x1 & 0xF0F0F0F0U,
                                     pu0, pu1, VBIAS_I);
                        acc[c].x[0] += f0 * (__int_as_float(r0) - VBIAS_F);
                        acc[c].x[1] += f1 * (__int_as_float(r1) - VBIAS_F);
                        acc[c].x[2] += f2 * (__int_as_float(r2) - VBIAS_F);
                        acc[c].x[3] += f3 * (__int_as_float(r3) - VBIAS_F);
                    }

                    sync_split();
                    cursor_next(cc);
                    cursor_next(cc);
                    stage_c = (stage_c + 2) % NSTAGE;
                    s += 2;
                    continue;
                }
            }

            // Single-strip IMMA.16816.U8.U8 path (trailing strip or !pair_32).
            if (s + LEAD < s_hi) {
                issue(ci, stage_i, ci.sl == 0);
                cursor_next(ci);
                stage_i = (stage_i + 1) % NSTAGE;
            } else {
                cp_async_commit();
            }
            cp_async_wait_group<LEAD>();
            sync_split();

            const int    id = cc.g;
            const char * rm = rmeta + (id & 1) * RMETA_BYTES;
            if (id != cur_id) {
                if (cur_id != INT_MIN) {
                    record_change(cur_rm, rm);
                }
                setup_body(rm);
                cur_id = id;
                cur_rm = rm;
            }

            const char * st = ring + stage_c * STAGE_BYTES;
            T_C kq;
            kq_strip(st, cc.p0, kq);

            float mx0 = fmaxf(kq.x[0], kq.x[2]);
            float mx1 = fmaxf(kq.x[1], kq.x[3]);
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, off, WARP_SIZE));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, off, WARP_SIZE));
            }
            mx0 = fmaxf(mx0, -FLT_MAX/2);
            mx1 = fmaxf(mx1, -FLT_MAX/2);
            float w0, w1;
            softmax_update_rescale(mx0, mx1, w0, w1);

            kq.x[0] = __expf(kq.x[0] - mx0);
            kq.x[1] = __expf(kq.x[1] - mx1);
            kq.x[2] = __expf(kq.x[2] - mx0);
            kq.x[3] = __expf(kq.x[3] - mx1);
            kq_sum[0] += w0 * (kq.x[0] + kq.x[2]);
            kq_sum[1] += w1 * (kq.x[1] + kq.x[3]);

            const half * vsh = (const half *) (st + META_VS);
            const half * vzh = (const half *) (st + META_VZ);
            const float vs_a = __half2float(vsh[lane >> 2]);
            const float vs_b = __half2float(vsh[(lane >> 2) + 8]);
            const float vz_a = __half2float(vzh[lane >> 2]);
            const float vz_b = __half2float(vzh[(lane >> 2) + 8]);

            const float pv0 = kq.x[0]*vs_a, pv1 = kq.x[1]*vs_a, pv2 = kq.x[2]*vs_b, pv3 = kq.x[3]*vs_b;
            float pm0 = fmaxf(pv0, pv2);
            float pm1 = fmaxf(pv1, pv3);
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                pm0 = fmaxf(pm0, __shfl_xor_sync(0xFFFFFFFF, pm0, off, WARP_SIZE));
                pm1 = fmaxf(pm1, __shfl_xor_sync(0xFFFFFFFF, pm1, off, WARP_SIZE));
            }
            const float rp0 = pm0 > 0.0f ? __fdividef(255.0f, pm0) : 0.0f;
            const float rp1 = pm1 > 0.0f ? __fdividef(255.0f, pm1) : 0.0f;
            const float fs0 = pm0 * (1.0f / 255.0f);
            const float fs1 = pm1 * (1.0f / 255.0f);
            float eps0 = 0.0f, eps1 = 0.0f;
            const uint32_t pu = pack_pu(pv0, pv1, pv2, pv3, rp0, rp1, fs0, fs1, eps0, eps1);
            pz[0] += w0 * (kq.x[0]*vz_a + kq.x[2]*vz_b + 7.5f * eps0);
            pz[1] += w1 * (kq.x[1]*vz_a + kq.x[3]*vz_b + 7.5f * eps1);
            const float f0 = w0 * fs0;
            const float f1 = w1 * fs1;
            const float f2 = f0 * (1.0f / 16.0f), f3 = f1 * (1.0f / 16.0f);

            const uint32_t * vwp = (const uint32_t *) (st + STRIP_BYTES) + lane;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const uint32_t x = vwp[c*WARP_SIZE];
                int32_t r0, r1, r2, r3;
                imma_u8u8_16(r0, r1, r2, r3, x & 0x0F0F0F0FU, x & 0xF0F0F0F0U, pu, VBIAS_I);
                acc[c].x[0] += f0 * (__int_as_float(r0) - VBIAS_F);
                acc[c].x[1] += f1 * (__int_as_float(r1) - VBIAS_F);
                acc[c].x[2] += f2 * (__int_as_float(r2) - VBIAS_F);
                acc[c].x[3] += f3 * (__int_as_float(r3) - VBIAS_F);
            }

            sync_split();
            cursor_next(cc);
            stage_c = (stage_c + 1) % NSTAGE;
            s += 1;
        }

        if (cur_id != INT_MIN) {
            record_change(cur_rm, nullptr);
        }
    }
    cp_async_wait_all();
    sync_split();

    // Exact rows (sink, ring): qp[NTC] and full fattn_sj_kvarn_ctx scoped here so qi[NTC] registers are reused.
    {
        const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(sj_kvarn_body, sj_kvarn_desc, z_KV, D, 4, 4, nb12, nb22);
        const int n_kv_pad  = ne11;
        const int n_ex_sink = kv.S / 16;
        const int far_end   = sj_kvarn_desc[GGML_INKSJ_DESC_FLAGS] >> GGML_INKSJ_FAR_END_SHIFT;
        const int ex_end    = far_end > 0 ? min(n_kv_pad, max(kv.B, 16*((far_end + 15)/16))) : n_kv_pad;
        const int n_ex      = n_ex_sink + (ex_end - kv.B)/16;
        const char * Kh = K + (size_t) z_KV * nb12;
        const char * Vh = V + (size_t) z_KV * nb22;

        static_assert((EXACT_BYTES/16) % WARP_SIZE == 0, "exact strip staging assumes whole warps");
        auto issue_exact = [&](const char * rows, const size_t nb, const int type) {
            const float tq6_mag = type == GGML_TYPE_TQ6_0 ? TQ6_CENTROIDS[32 + lane] : 0.0f;
            for (int q = nt*WARP_SIZE + lane; q < EXACT_BYTES/16; q += NT*WARP_SIZE) {
                const int r = q / (EXACT_ROW_BYTES/16);
                const int j = q % (EXACT_ROW_BYTES/16);
                if (type != GGML_TYPE_F16) {
                    half2 * out = (half2 *) (ring + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)));
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        out[i] = fattn_sj_kvarn_ring_pair_warp(rows + (size_t) r*nb, 4*j + i, type, tq6_mag);
                    }
                } else {
                    cp_async_cg_16<0>(ring_s + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)), rows + (size_t) r*nb + 16*j);
                }
            }
            cp_async_commit();
        };
        auto exact_word = [&](const int r, const int cp) -> uint32_t {
            return *(const uint32_t *) (ring + r*EXACT_ROW_BYTES + 16*((cp >> 2) ^ (r & 7)) + 4*(cp & 3));
        };

        T_B qp[NTC];
        if (split < n_ex) {
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                float2 q0, q1;
                load_q(c, q0, q1);
                qp[c].x[0] = __floats2half2_rn(q0.x, q0.y);
                qp[c].x[1] = __floats2half2_rn(q1.x, q1.y);
            }
        }

        for (int e = split; e < n_ex; e += nsplit) {
            const int p0   = e < n_ex_sink ? 16*e : kv.B + 16*(e - n_ex_sink);
            const int row0 = p0 < kv.S ? p0 : kv.S + (p0 - kv.S) % kv.cap;
            const bool f16_sink = p0 < kv.S && kv.sink_type == GGML_TYPE_F16;
            const char * Kr = f16_sink ? fattn_sj_kvarn_sink_row(Kh, kv, nb11, row0, false) : Kh + (size_t) row0*nb11;
            const char * Vr = f16_sink ? fattn_sj_kvarn_sink_row(Vh, kv, nb21, row0, true)  : Vh + (size_t) row0*nb21;

            issue_exact(Kr, f16_sink ? kv.sink_stride : nb11, f16_sink ? GGML_TYPE_F16 : kv.type_k);
            cp_async_wait_all();
            sync_split();

            T_C kq;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = (lane >> 2) + 8*(l & 1);
                    const int cp  = 8*c + 4*(l >> 1) + (lane & 3);
                    a.x[l] = fattn_sj_kvarn_u32_as_half2(exact_word(row, cp));
                }
                mma(kq, a, qp[c]);
            }
            const float mk0_0 = __half2float(mrow0[p0 + (lane >> 2)]);
            const float mk0_1 = mrow1 != mrow0 ? __half2float(mrow1[p0 + (lane >> 2)]) : mk0_0;
            const float mk1_0 = __half2float(mrow0[p0 + (lane >> 2) + 8]);
            const float mk1_1 = mrow1 != mrow0 ? __half2float(mrow1[p0 + (lane >> 2) + 8]) : mk1_0;
            kq.x[0] += mk0_0; kq.x[1] += mk0_1;
            kq.x[2] += mk1_0; kq.x[3] += mk1_1;

            float mx0 = fmaxf(kq.x[0], kq.x[2]);
            float mx1 = fmaxf(kq.x[1], kq.x[3]);
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, off, WARP_SIZE));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, off, WARP_SIZE));
            }
            mx0 = fmaxf(mx0, -FLT_MAX/2);
            mx1 = fmaxf(mx1, -FLT_MAX/2);
            float w0, w1;
            softmax_update_rescale(mx0, mx1, w0, w1);
            GGML_UNUSED_VARS(w0, w1);
            kq.x[0] = __expf(kq.x[0] - kq_max[0]);
            kq.x[1] = __expf(kq.x[1] - kq_max[1]);
            kq.x[2] = __expf(kq.x[2] - kq_max[0]);
            kq.x[3] = __expf(kq.x[3] - kq_max[1]);
            kq_sum[0] += kq.x[0] + kq.x[2];
            kq_sum[1] += kq.x[1] + kq.x[3];
            const T_B P = get_transposed(get_half2(kq));

            sync_split();
            issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v);
            cp_async_wait_all();
            sync_split();

#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = 8*(l >> 1) + (lane >> 2);
                    const int cp  = 8*c + 4*(l & 1) + (lane & 3);
                    a.x[l] = ggml_cuda_movmatrix(fattn_sj_kvarn_u32_as_half2(exact_word(row, cp)));
                }
                mma(acc[c], a, P);
            }
            sync_split();
        }
    }

#pragma unroll
    for (int off = 4; off < WARP_SIZE; off <<= 1) {
        kq_sum[0] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[0], off, WARP_SIZE);
        kq_sum[1] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[1], off, WARP_SIZE);
    }

    if (sinks != nullptr && blockIdx.y == 0 && kw == 0) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*(lane & 3) + e;
            if (!col_valid(col)) {
                continue;
            }
            const float sink = ((const float *) sinks)[head0 + col_head(col)];
            const float nm   = fmaxf(kq_max[e], sink);
            const float ms   = expf(kq_max[e] - nm);
            kq_max[e] = nm;
            kq_sum[e] = ms * kq_sum[e] + expf(sink - nm);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0 + e] *= ms;
                acc[c].x[2 + e] *= ms;
            }
        }
    }

    float * smem_combine = (float *) smem;
    if (KW > 1) {
        __syncthreads();
        for (int hf = KW/2; hf >= 1; hf >>= 1) {
            if (kw >= hf && kw < 2*hf) {
                const int slot = nt*hf + (kw - hf);
                float * a = slot_acc (smem_combine, slot);
                float * m = slot_meta(smem_combine, slot);
#pragma unroll
                for (int c = 0; c < NTC; ++c) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int col = 2*(lane & 3) + (l & 1);
                        const int d   = 16*c + (lane >> 2) + 8*(l >> 1);
                        a[col*COL_STRIDE + d] = acc[c].x[l];
                    }
                }
                if (lane < 4) {
                    m[2*(2*lane + 0) + 0] = kq_max[0]; m[2*(2*lane + 0) + 1] = kq_sum[0];
                    m[2*(2*lane + 1) + 0] = kq_max[1]; m[2*(2*lane + 1) + 1] = kq_sum[1];
                }
            }
            __syncthreads();
            if (kw < hf) {
                const int slot = nt*hf + kw;
                const float * a = slot_acc (smem_combine, slot);
                const float * m = slot_meta(smem_combine, slot);
                const float om0 = m[2*(2*(lane & 3) + 0) + 0], os0 = m[2*(2*(lane & 3) + 0) + 1];
                const float om1 = m[2*(2*(lane & 3) + 1) + 0], os1 = m[2*(2*(lane & 3) + 1) + 1];
                const float nm0 = fmaxf(kq_max[0], om0), nm1 = fmaxf(kq_max[1], om1);
                const float sa0 = expf(kq_max[0] - nm0), sb0 = expf(om0 - nm0);
                const float sa1 = expf(kq_max[1] - nm1), sb1 = expf(om1 - nm1);
                kq_max[0] = nm0; kq_max[1] = nm1;
                kq_sum[0] = sa0*kq_sum[0] + sb0*os0;
                kq_sum[1] = sa1*kq_sum[1] + sb1*os1;
#pragma unroll
                for (int c = 0; c < NTC; ++c) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const int col = 2*(lane & 3) + (l & 1);
                        const int d   = 16*c + (lane >> 2) + 8*(l >> 1);
                        const float sa = (l & 1) ? sa1 : sa0;
                        const float sb = (l & 1) ? sb1 : sb0;
                        acc[c].x[l] = sa*acc[c].x[l] + sb*a[col*COL_STRIDE + d];
                    }
                }
            }
            __syncthreads();
        }
        if (kw != 0) {
            return;
        }
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
            const int col = 2*(lane & 3) + e;
            if (!col_valid(col)) {
                continue;
            }
            const int head = head0 + col_head(col);
            const float inv = single ? 1.0f / kq_sum[e] : 1.0f;
            float * out = dst + ((size_t) ((q_base + col_query(col))*ne02 + head)*gridDim.y + blockIdx.y)*D;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const int d = 16*c + (lane >> 2) + 8*(l >> 1);
                out[d] = acc[c].x[l + e] * inv;
            }
        }
    }
    if (!single && lane < 4) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*lane + e;
            if (col_valid(col)) {
                dst_meta[(size_t) ((q_base + col_query(col))*ne02 + head0 + col_head(col))*gridDim.y + blockIdx.y] = make_float2(kq_max[e], kq_sum[e]);
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

template <int max_warps, int NSTAGE, int ncols1, bool rot, sj_kvarn_tile tile, bool pair_32>
static fattn_kernel_t get_direct_verify_kernel() {
    fattn_kernel_t kernel = flash_attn_ext_sj_kvarn_direct_verify<max_warps, NSTAGE, ncols1, rot, tile, pair_32>;
    const int id = ggml_cuda_get_device();
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {false};
    if (!initialized[id]) {
        const size_t limit = ggml_cuda_info().devices[id].smpbo;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int) limit));
        initialized[id] = true;
    }
    return kernel;
}

static inline size_t direct_verify_shared_bytes(const int NT, const int KW, const int nstage) {
    return std::max(ring_bytes(KW, nstage, false), (size_t) NT * (KW / 2) * SLOT_BYTES);
}

template <int max_warps, int ncols1, bool rot, sj_kvarn_tile tile>
static void direct_verify_launch_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int NT, const int KW) {
    static const bool pv32_env = [] {
        const char * e = getenv("GGML_SJKVARN_DIRECT_VERIFY_PV32");
        return e == nullptr || *e == '\0' || atoi(e) != 0;
    }();
    static const int nstage_env = getenv("GGML_SJKVARN_DIRECT_NSTAGE") != nullptr ? atoi(getenv("GGML_SJKVARN_DIRECT_NSTAGE")) : 0;

    // NSTAGE=4 (~19.8 KB) fits >=3 CTAs/SM and supports paired 2-strip IMMA.16832.U8.U8 (285T).
    // NSTAGE=2 (~11.4 KB) is used if SMEM limit rejects NSTAGE=4 or user forces NSTAGE=2 / PV32=0.
    const bool want_pv32 = pv32_env && (nstage_env == 0 || nstage_env >= 4);
    if (want_pv32 && direct_verify_shared_bytes(NT, KW, 4) <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        fattn_kernel_t kernel = get_direct_verify_kernel<max_warps, 4, ncols1, rot, tile, true>();
        launch_fattn<D, ncols1, NCOLS2>(
            ctx, dst, kernel, NT * KW, direct_verify_shared_bytes(NT, KW, 4), UNIT,
            false, false, false, false, WARP_SIZE, nullptr, nullptr, 0, false, KW);
        return;
    }
    if (direct_verify_shared_bytes(NT, KW, 4) <= ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo && nstage_env != 2) {
        fattn_kernel_t kernel = get_direct_verify_kernel<max_warps, 4, ncols1, rot, tile, false>();
        launch_fattn<D, ncols1, NCOLS2>(
            ctx, dst, kernel, NT * KW, direct_verify_shared_bytes(NT, KW, 4), UNIT,
            false, false, false, false, WARP_SIZE, nullptr, nullptr, 0, false, KW);
        return;
    }
    fattn_kernel_t kernel = get_direct_verify_kernel<max_warps, 2, ncols1, rot, tile, false>();
    launch_fattn<D, ncols1, NCOLS2>(
        ctx, dst, kernel, NT * KW, direct_verify_shared_bytes(NT, KW, 2), UNIT,
        false, false, false, false, WARP_SIZE, nullptr, nullptr, 0, false, KW);
}

} // namespace fattn_sj_kvarn_direct_verify
