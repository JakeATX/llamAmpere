// Spill-free PV16 (HMMA.16816.F16, 142 TFLOPS) SJ-KVaRN 4/4 stream decode/verify kernel (n_q <= 8).
//
// Opt-in via GGML_SJKVARN_STREAM_PV16=1 (default 0: zero behavior change when unset).
// Eliminates the 56-112 B register spills in baseline sjkvarn_stream at NT = 5, 6 and doubles P x V Tensor Core throughput
// from HMMA.16816.F32 (71T) to HMMA.16816.F16 (142T):
//   1. FP16 P x V accumulators: T_H acc16[NTC] (tile<16, 4, half2>, 32 regs vs 64 regs for T_C acc[NTC]).
//   2. Centered V nibble decode: q_c = q - 8 in [-8, +7] at uniform s=1 scale across low and high channels
//      (__hsub2(r0, 1032) and __hfma2(r1, 0.0625, -72)), folding +8*Vscale into Vzero (eliminates ps[2] and
//      1024*ps bias cancellation completely, keeping acc16 centered).
//   3. Dynamic FP32 acc_scale[2] + acc_bound[2] per column pair: softmax running-max shifts rescale acc_scale in FP32
//      for free; acc16 only rescales if |acc16| + strip_inc/acc_scale > 8192 (with P*Vscale/acc_scale computed in FP32
//      before FP16 narrowing). Zero overflow/underflow even under Outl100x K/V channels.
//   4. Minimal body-loop ctx (loads S, B, G, rec_stride, table directly and constructs full fattn_sj_kvarn_ctx lazily
//      after the body loop), keeping live body-loop registers well under SM 8.6's 168-reg cap for 2 CTAs/SM at NT = 5, 6.
//   5. Supports GQA=6 compact tile (T = 4..8) and general rows tile (T = 1..8, GQA 1..8), fused H256 rotation, sinks,
//      per-head F16 masks (ne[2] > 1), INKKV far_end, and LSE parallel-block combine.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "mma.cuh"
#include "fattn-sjkvarn-rot.cuh"
#include "fattn-sjkvarn-stream.cuh"
#include "ggml-inksj.h"

#include <atomic>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace fattn_sj_kvarn_stream_pv16 {

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
using fattn_sj_kvarn_stream::raw_pairs;
using fattn_sj_kvarn_stream::decode_pairs;
using fattn_sj_kvarn_stream::cp_async_commit;
using fattn_sj_kvarn_stream::cp_async_wait_group;

typedef tile<16, 4, half2> T_H; // P x V FP16 accumulator (2 regs per 16x8 tile -> 32 regs for D=256)

struct body_cursor {
    int p0;
    int g;
    int sl;
};

// Decode 4-bit V codes centered as (q - 8) in [-8, +7], uniform s=1 scale across both nibbles.
// Compensated by +(8 * Vscale[t]) added to Vzero[t] in FP32 pz[2].
static __device__ __forceinline__ void v_pairs_centered(const uint32_t x, const uint32_t magic, half2 * const __restrict__ v) {
    const half2 m1032 = make_half2(1032.0f, 1032.0f);
    const half2 c16   = make_half2(0.0625f, 0.0625f);
    const half2 m72   = make_half2(-72.0f, -72.0f);
    half2 r[4];
    raw_pairs(x, magic, r);
    v[0] = __hsub2(r[0], m1032);
    v[1] = __hfma2(r[1], c16, m72);
    v[2] = __hsub2(r[2], m1032);
    v[3] = __hfma2(r[3], c16, m72);
}

static __device__ __forceinline__ uint32_t pv16_half2_as_u32(const half2 h) {
    uint32_t u;
    memcpy(&u, &h, sizeof(u));
    return u;
}

static __device__ __forceinline__ void warp_absmax_half2(const half2 h, float & out_abs) {
    const half2 a = __habs2(h);
    out_abs = fmaxf(out_abs, __half2float(__hmax(__low2half(a), __high2half(a))));
}

template <int max_warps, int NSTAGE, bool rot = false, sj_kvarn_tile tile = sj_kvarn_tile::rows>
__launch_bounds__(WARP_SIZE*max_warps, (max_warps == 4 || max_warps == 5 || max_warps == 6 || (max_warps == 3 && tile == sj_kvarn_tile::compact)) ? 2 : 1)
static __global__ void flash_attn_ext_sj_kvarn_stream_pv16(
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
    static_assert(!compact || (max_warps >= 3 && max_warps <= 6), "the compact tile runs 3..6 warps");
    constexpr bool fixed_nt = max_warps == 5 || max_warps == 6 || compact;
    const int NT     = compact || max_warps == 5 || max_warps == 6 ? max_warps : n_q;
    const int KW     = fixed_nt ? 1 : nwarps / NT;
    const int nt     = fixed_nt ? warp : warp % NT;
    const int kw     = fixed_nt ? 0 : warp / NT;

    const int z_KV   = blockIdx.z;
    const int gqa    = ne02 / ne12;
    const int head0  = z_KV * gqa;
    const int hcol   = lane >> 2;
    const int npair  = 6 * n_q;
    auto col_query = [&](const int col) -> int  { return compact ? (8*nt + col < npair ? (8*nt + col)/6 : 0) : nt; };
    auto col_head  = [&](const int col) -> int  { return compact ? (8*nt + col) % 6 : col; };
    auto col_valid = [&](const int col) -> bool { return compact ? 8*nt + col < npair : col < gqa; };
    const bool hvalid = col_valid(hcol);

    // Lightweight body-only parameters (exact ctx constructed lazily after body loop to spare registers).
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

    const float * Qf = (const float *) (Q + (size_t) col_query(hcol)*nb01 + (size_t) (head0 + (hvalid ? col_head(hcol) : 0))*nb02);

    const int c0 = 2 * (lane & 3);
    // padding columns (c0 past the last head / query) read the first valid row instead of past the mask's end
    const bool c0_valid = col_valid(c0);
    const half * mrow0 = (const half *) (mask + (size_t) (c0_valid ? col_query(c0) : 0)*nb31 + (ne32 > 1 ? (size_t) (head0 + (c0_valid ? col_head(c0) : 0))*nb32 : 0));
    const half * mrow1 = (ne32 > 1 && col_valid(c0 + 1)) ? (const half *) ((const char *) mrow0 + nb32) : mrow0;

    static_assert(NSTAGE * STAGE_BYTES >= EXACT_BYTES, "ring must hold one exact K or V strip");
    static_assert(NSTAGE <= 6, "rmeta buffering requires NSTAGE <= 6");
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
    const uint32_t magic = 0x64006400U;

    T_B   qp[NTC];
    T_H   acc16[NTC]; // 32 regs total for D=256 (tile<16, 4, half2>)
    float c_r[2]       = {0.0f, 0.0f};
    float kq_max[2]    = {-FLT_MAX/2, -FLT_MAX/2};
    float kq_sum[2]    = {0.0f, 0.0f};
    float pz[2]        = {0.0f, 0.0f};
    float acc_scale[2] = {1.0f, 1.0f};
    float acc_bound[2] = {0.0f, 0.0f};

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

    auto setup_body = [&](const char * rm) {
        const uint2 * ks2 = (const uint2 *) (rm            + 2*(lane & 3)*(D/4));
        const uint2 * kz2 = (const uint2 *) (rm + RMETA_KZ + 2*(lane & 3)*(D/4));
        float cr = 0.0f;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 ks = ks2[c];
            const uint2 kz = kz2[c];
            float2 q0, q1;
            load_q(c, q0, q1);
            const half2 ks0 = fattn_sj_kvarn_u32_as_half2(ks.x), ks1 = fattn_sj_kvarn_u32_as_half2(ks.y);
            const float2 kz0 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.x));
            const float2 kz1 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.y));
            qp[c].x[0] = __hmul2(__floats2half2_rn(q0.x, q0.y), ks0);
            qp[c].x[1] = __hmul2(__floats2half2_rn(q1.x, q1.y), ks1);
            cr += q0.x*kz0.x + q0.y*kz0.y + q1.x*kz1.x + q1.y*kz1.y;
        }
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 1, WARP_SIZE);
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 2, WARP_SIZE);
        c_r[0] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3),     WARP_SIZE);
        c_r[1] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3) + 4, WARP_SIZE);
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

    // Ensure acc16[c] stays strictly <= 8192 after accumulating a strip bounded by inc0/inc1 (true / rec_Vch units).
    auto ensure_pv16_scale = [&](const float inc0, const float inc1) {
        if (acc_bound[0] == 0.0f && inc0 > 0.0f) {
            acc_scale[0] = fmaxf(inc0 * 0.0625f, 1e-30f);
        }
        if (acc_bound[1] == 0.0f && inc1 > 0.0f) {
            acc_scale[1] = fmaxf(inc1 * 0.0625f, 1e-30f);
        }
        const float nb0 = acc_bound[0] + (inc0 > 0.0f ? inc0 / acc_scale[0] : 0.0f);
        const float nb1 = acc_bound[1] + (inc1 > 0.0f ? inc1 / acc_scale[1] : 0.0f);
        const bool need0 = nb0 > 8192.0f;
        const bool need1 = nb1 > 8192.0f;
        if (__any_sync(0xFFFFFFFF, need0 || need1)) {
            float r0 = 1.0f, r1 = 1.0f;
            if (need0) {
                const float peak0 = acc_scale[0]*acc_bound[0] + inc0;
                const float ns0   = fmaxf(peak0 * 0.015625f, 1e-30f);
                r0 = acc_scale[0] / ns0;
                acc_scale[0] = ns0;
                acc_bound[0] *= r0;
            }
            if (need1) {
                const float peak1 = acc_scale[1]*acc_bound[1] + inc1;
                const float ns1   = fmaxf(peak1 * 0.015625f, 1e-30f);
                r1 = acc_scale[1] / ns1;
                acc_scale[1] = ns1;
                acc_bound[1] *= r1;
            }
            const half2 r2 = make_half2(r0, r1);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc16[c].x[0] = __hmul2(acc16[c].x[0], r2);
                acc16[c].x[1] = __hmul2(acc16[c].x[1], r2);
            }
        }
        acc_bound[0] += inc0 > 0.0f ? inc0 / acc_scale[0] : 0.0f;
        acc_bound[1] += inc1 > 0.0f ? inc1 / acc_scale[1] : 0.0f;
    };

    // Record change old -> new (nullptr = exact rows): centered V has s=1 uniformly across all channels!
    // Fold pz (FP32) and convert units acc_old * Vch_old / Vch_new, then re-center FP16 dynamic range cleanly.
    auto record_change = [&](const char * rec_old, const char * rec_new) {
        float z0s = 0.0f, z1s = 0.0f;
        if (rec_old != nullptr) {
            z0s = pz[0];
            z1s = pz[1];
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                z0s += __shfl_xor_sync(0xFFFFFFFF, z0s, off, WARP_SIZE);
                z1s += __shfl_xor_sync(0xFFFFFFFF, z1s, off, WARP_SIZE);
            }
            pz[0] = 0.0f;
            pz[1] = 0.0f;
        }
        const uint32_t * vo = rec_old != nullptr ? (const uint32_t *) (rec_old + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;
        const uint32_t * vn = rec_new != nullptr ? (const uint32_t *) (rec_new + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;

        float pmax0 = 0.0f, pmax1 = 0.0f;
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
            const float2 f0 = __half22float2(acc16[c].x[0]);
            const float2 f1 = __half22float2(acc16[c].x[1]);
            pmax0 = fmaxf(pmax0, fmaxf(fabsf((acc_scale[0]*f0.x + z0s)*r_lo), fabsf((acc_scale[0]*f1.x + z0s)*r_hi)));
            pmax1 = fmaxf(pmax1, fmaxf(fabsf((acc_scale[1]*f0.y + z1s)*r_lo), fabsf((acc_scale[1]*f1.y + z1s)*r_hi)));
        }
#pragma unroll
        for (int off = 4; off < WARP_SIZE; off <<= 1) {
            pmax0 = fmaxf(pmax0, __shfl_xor_sync(0xFFFFFFFF, pmax0, off, WARP_SIZE));
            pmax1 = fmaxf(pmax1, __shfl_xor_sync(0xFFFFFFFF, pmax1, off, WARP_SIZE));
        }
        const float ns0 = pmax0 > 0.0f ? fmaxf(pmax0 * 0.015625f, 1e-30f) : 1.0f;
        const float ns1 = pmax1 > 0.0f ? fmaxf(pmax1 * 0.015625f, 1e-30f) : 1.0f;
        const float inv0 = pmax0 > 0.0f ? __fdividef(1.0f, ns0) : 0.0f;
        const float inv1 = pmax1 > 0.0f ? __fdividef(1.0f, ns1) : 0.0f;

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
            const float2 f0 = __half22float2(acc16[c].x[0]);
            const float2 f1 = __half22float2(acc16[c].x[1]);
            acc16[c].x[0] = __floats2half2_rn(((acc_scale[0]*f0.x + z0s)*r_lo)*inv0, ((acc_scale[1]*f0.y + z1s)*r_lo)*inv1);
            acc16[c].x[1] = __floats2half2_rn(((acc_scale[0]*f1.x + z0s)*r_hi)*inv0, ((acc_scale[1]*f1.y + z1s)*r_hi)*inv1);
        }
        acc_scale[0] = ns0;
        acc_scale[1] = ns1;
        acc_bound[0] = pmax0 > 0.0f ? pmax0 * inv0 : 0.0f;
        acc_bound[1] = pmax1 > 0.0f ? pmax1 * inv1 : 0.0f;
    };

    // Update online softmax max/sum/acc_scale from raw KQ tile (already masked), leave exp(kq - nm) in kq.
    auto softmax_update = [&](T_C & kq) {
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
            float sc0 = expf(kq_max[0] - nm0);
            float sc1 = expf(kq_max[1] - nm1);
            if (sc0 < 1e-20f) sc0 = 0.0f;
            if (sc1 < 1e-20f) sc1 = 0.0f;
            kq_max[0] = nm0;
            kq_max[1] = nm1;
            kq_sum[0] *= sc0;
            kq_sum[1] *= sc1;
            pz[0] *= sc0;
            pz[1] *= sc1;
            acc_scale[0] *= sc0;
            acc_scale[1] *= sc1;
            const bool z0 = sc0 == 0.0f && acc_bound[0] > 0.0f;
            const bool z1 = sc1 == 0.0f && acc_bound[1] > 0.0f;
            if (__any_sync(0xFFFFFFFF, z0 || z1)) {
                const uint32_t keep = (z0 ? 0xFFFF0000U : 0xFFFFFFFFU) & (z1 ? 0x0000FFFFU : 0xFFFFFFFFU);
#pragma unroll
                for (int c = 0; c < NTC; ++c) {
#pragma unroll
                    for (int l = 0; l < 2; ++l) {
                        const uint32_t u = pv16_half2_as_u32(acc16[c].x[l]) & keep;
                        acc16[c].x[l] = fattn_sj_kvarn_u32_as_half2(u);
                    }
                }
            }
            if (sc0 == 0.0f) { acc_bound[0] = 0.0f; acc_scale[0] = 1.0f; }
            if (sc1 == 0.0f) { acc_bound[1] = 0.0f; acc_scale[1] = 1.0f; }
        }
        kq.x[0] = expf(kq.x[0] - nm0);
        kq.x[1] = expf(kq.x[1] - nm1);
        kq.x[2] = expf(kq.x[2] - nm0);
        kq.x[3] = expf(kq.x[3] - nm1);
        kq_sum[0] += kq.x[0] + kq.x[2];
        kq_sum[1] += kq.x[1] + kq.x[3];
    };

    auto make_scaled_p = [&](const float p0, const float p1, const float p2, const float p3) -> T_B {
        const float inv0 = __fdividef(1.0f, acc_scale[0]);
        const float inv1 = __fdividef(1.0f, acc_scale[1]);
        T_H pc;
        pc.x[0] = __floats2half2_rn(p0 * inv0, p1 * inv1);
        pc.x[1] = __floats2half2_rn(p2 * inv0, p3 * inv1);
        return get_transposed(pc);
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

    body_cursor ci = cursor_init(s_lo);
    body_cursor cc = ci;
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

    int          cur_id = INT_MIN;
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

        {
            T_C kq;
            const char * st = ring + stage_c * STAGE_BYTES;
            const uint32_t * kwp = (const uint32_t *) st + lane;
            const uint32_t * vwp = (const uint32_t *) (st + STRIP_BYTES) + lane;

#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                decode_pairs(kwp[c*WARP_SIZE], magic, a.x);
                mma(kq, a, qp[c]);
            }
            const half * ktok = (const half *) (st + META_KTOK);
            const half * vsh  = (const half *) (st + META_VS);
            const half * vzh  = (const half *) (st + META_VZ);
            const float tok0  = __half2float(ktok[lane >> 2]);
            const float tok1  = __half2float(ktok[(lane >> 2) + 8]);
            const float mk0_0 = __half2float(mrow0[cc.p0 + (lane >> 2)]);
            const float mk0_1 = mrow1 != mrow0 ? __half2float(mrow1[cc.p0 + (lane >> 2)]) : mk0_0;
            const float mk1_0 = __half2float(mrow0[cc.p0 + (lane >> 2) + 8]);
            const float mk1_1 = mrow1 != mrow0 ? __half2float(mrow1[cc.p0 + (lane >> 2) + 8]) : mk1_0;
            kq.x[0] = tok0 * (kq.x[0] + c_r[0]) + mk0_0;
            kq.x[1] = tok0 * (kq.x[1] + c_r[1]) + mk0_1;
            kq.x[2] = tok1 * (kq.x[2] + c_r[0]) + mk1_0;
            kq.x[3] = tok1 * (kq.x[3] + c_r[1]) + mk1_1;

            softmax_update(kq);

            const float vs0 = __half2float(vsh[lane >> 2]);
            const float vs1 = __half2float(vsh[(lane >> 2) + 8]);
            const float vz0 = __half2float(vzh[lane >> 2]) + 8.0f * vs0;
            const float vz1 = __half2float(vzh[(lane >> 2) + 8]) + 8.0f * vs1;
            pz[0] += kq.x[0]*vz0 + kq.x[2]*vz1;
            pz[1] += kq.x[1]*vz0 + kq.x[3]*vz1;

            const float pe0 = kq.x[0]*vs0, pe1 = kq.x[1]*vs0, pe2 = kq.x[2]*vs1, pe3 = kq.x[3]*vs1;
            float inc0 = 8.0f * (fabsf(pe0) + fabsf(pe2));
            float inc1 = 8.0f * (fabsf(pe1) + fabsf(pe3));
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                inc0 += __shfl_xor_sync(0xFFFFFFFF, inc0, off, WARP_SIZE);
                inc1 += __shfl_xor_sync(0xFFFFFFFF, inc1, off, WARP_SIZE);
            }
            ensure_pv16_scale(inc0, inc1);
            const T_B P = make_scaled_p(pe0, pe1, pe2, pe3);

#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                v_pairs_centered(vwp[c*WARP_SIZE], magic, a.x);
                mma(acc16[c], a, P);
            }
        }
        sync_split();
        cursor_next(cc);
        stage_c = stage_c + 1 == NSTAGE ? 0 : stage_c + 1;
    }

    if (cur_id != INT_MIN) {
        record_change(cur_rm, nullptr);
    }
    cp_async_wait_all();
    sync_split();

    // Exact rows (sink, ring) constructed with lazy full ctx after body loop.
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

        if (split < n_ex) {
            setup_exact();
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
            softmax_update(kq);

            sync_split();
            issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v);
            cp_async_wait_all();
            sync_split();

            // Compute exact V absmax across staged 16x256 V strip (8192 B / 16 = 512 uint4s -> 16/lane).
            float v_abs = 0.0f;
#pragma unroll
            for (int q = lane; q < EXACT_BYTES / 16; q += WARP_SIZE) {
                const uint4 u = *(const uint4 *) (ring + 16*q);
                warp_absmax_half2(fattn_sj_kvarn_u32_as_half2(u.x), v_abs);
                warp_absmax_half2(fattn_sj_kvarn_u32_as_half2(u.y), v_abs);
                warp_absmax_half2(fattn_sj_kvarn_u32_as_half2(u.z), v_abs);
                warp_absmax_half2(fattn_sj_kvarn_u32_as_half2(u.w), v_abs);
            }
#pragma unroll
            for (int off = 1; off < WARP_SIZE; off <<= 1) {
                v_abs = fmaxf(v_abs, __shfl_xor_sync(0xFFFFFFFF, v_abs, off, WARP_SIZE));
            }
            float psum0 = kq.x[0] + kq.x[2];
            float psum1 = kq.x[1] + kq.x[3];
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                psum0 += __shfl_xor_sync(0xFFFFFFFF, psum0, off, WARP_SIZE);
                psum1 += __shfl_xor_sync(0xFFFFFFFF, psum1, off, WARP_SIZE);
            }
            ensure_pv16_scale(v_abs * psum0, v_abs * psum1);
            const T_B P = make_scaled_p(kq.x[0], kq.x[1], kq.x[2], kq.x[3]);

#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = 8*(l >> 1) + (lane >> 2);
                    const int cp  = 8*c + 4*(l & 1) + (lane & 3);
                    a.x[l] = ggml_cuda_movmatrix(fattn_sj_kvarn_u32_as_half2(exact_word(row, cp)));
                }
                mma(acc16[c], a, P);
            }
            sync_split();
        }
    }

    // Finish lane-partial sums.
#pragma unroll
    for (int off = 4; off < WARP_SIZE; off <<= 1) {
        kq_sum[0] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[0], off, WARP_SIZE);
        kq_sum[1] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[1], off, WARP_SIZE);
    }

    // Optional attention sinks: only global split 0 adds the sink contribution to (kq_max, kq_sum, acc_scale).
    if (sinks != nullptr && blockIdx.y == 0 && kw == 0) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*(lane & 3) + e;
            if (!col_valid(col)) {
                continue;
            }
            const float sink = ((const float *) sinks)[head0 + col_head(col)];
            const float nm   = fmaxf(kq_max[e], sink);
            float ms = expf(kq_max[e] - nm);
            if (ms < 1e-20f) ms = 0.0f;
            kq_max[e]     = nm;
            kq_sum[e]     = ms * kq_sum[e] + expf(sink - nm);
            acc_scale[e] *= ms;
        }
    }

    // Promote scaled FP16 accumulator to FP32 post-loop (no register overlap with qp or body ctx).
    T_C out_acc[NTC];
#pragma unroll
    for (int c = 0; c < NTC; ++c) {
        const float2 f0 = __half22float2(acc16[c].x[0]);
        const float2 f1 = __half22float2(acc16[c].x[1]);
        out_acc[c].x[0] = acc_scale[0] * f0.x;
        out_acc[c].x[1] = acc_scale[1] * f0.y;
        out_acc[c].x[2] = acc_scale[0] * f1.x;
        out_acc[c].x[3] = acc_scale[1] * f1.y;
    }

    // Combine KV splits inside block if KW > 1.
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
                        a[col*COL_STRIDE + d] = out_acc[c].x[l];
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
                        out_acc[c].x[l] = sa*out_acc[c].x[l] + sb*a[col*COL_STRIDE + d];
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
                    out_acc[c].x[l] *= (l & 1) ? inv1 : inv0;
                }
            }
            sj_kvarn_rot256_frag<T_C, NTC>(out_acc, lane);
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
            const float inv = single ? 1.0f/kq_sum[e] : 1.0f;
            float * out = dst + ((size_t) (col_query(col)*ne02 + head)*gridDim.y + blockIdx.y)*D;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const int d = 16*c + (lane >> 2) + 8*(l >> 1);
                out[d] = out_acc[c].x[l + e] * inv;
            }
        }
    }
    if (!single && lane < 4) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*lane + e;
            if (col_valid(col)) {
                dst_meta[(size_t) (col_query(col)*ne02 + head0 + col_head(col))*gridDim.y + blockIdx.y] = make_float2(kq_max[e], kq_sum[e]);
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

template <int max_warps, int NSTAGE, bool rot, sj_kvarn_tile tile>
static fattn_kernel_t get_pv16_kernel() {
    fattn_kernel_t kernel = flash_attn_ext_sj_kvarn_stream_pv16<max_warps, NSTAGE, rot, tile>;
    const int id = ggml_cuda_get_device();
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {false};
    if (!initialized[id]) {
        const size_t limit = ggml_cuda_info().devices[id].smpbo;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int) limit));
        initialized[id] = true;
    }
    return kernel;
}

static inline size_t pv16_shared_bytes(const int NT, const int KW, const int nstage) {
    return std::max(ring_bytes(KW, nstage, false), (size_t) NT * (KW / 2) * SLOT_BYTES);
}

template <int max_warps, int NSTAGE, bool rot, sj_kvarn_tile tile>
static int pv16_occupancy(const int NT, const int KW) {
    const size_t shared = pv16_shared_bytes(NT, KW, NSTAGE);
    if (shared > ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        return 0;
    }
    int blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
        get_pv16_kernel<max_warps, NSTAGE, rot, tile>(), WARP_SIZE * NT * KW, shared));
    return blocks;
}

template <int max_warps, bool rot, sj_kvarn_tile tile>
static void pv16_launch_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int NT, const int KW) {
    static const int nstage_env = getenv("GGML_SJKVARN_DIRECT_NSTAGE") != nullptr ? atoi(getenv("GGML_SJKVARN_DIRECT_NSTAGE")) : 0;
    int nstage = nstage_env;
    if (nstage == 0) {
        const int b4 = pv16_occupancy<max_warps, 4, rot, tile>(NT, KW);
        const int b2 = pv16_occupancy<max_warps, 2, rot, tile>(NT, KW);
        nstage = (b4 > 0 && b4 >= b2) ? 4 : 2;
    } else {
        nstage = nstage >= 4 ? 4 : 2;
    }
    fattn_kernel_t kernel = (nstage >= 4) ? get_pv16_kernel<max_warps, 4, rot, tile>()
                                          : get_pv16_kernel<max_warps, 2, rot, tile>();
    const size_t nbytes_shared = pv16_shared_bytes(NT, KW, nstage);
    launch_fattn<D, NCOLS2, NCOLS2>(
        ctx, dst, kernel, NT * KW, nbytes_shared, UNIT,
        false, false, false, false, WARP_SIZE, nullptr, nullptr, 0, false, KW);
}

} // namespace fattn_sj_kvarn_stream_pv16
