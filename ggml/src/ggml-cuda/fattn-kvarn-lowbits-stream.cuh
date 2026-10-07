#ifndef GGML_TQ6_PACKED_PREFETCH
#define GGML_TQ6_PACKED_PREFETCH 0
#endif
#include <type_traits>
// KVarN and native Turbo4 body attention, half-MMA streaming variant (n_q <= 8).
// Turbo4 stages native packed row blocks and decodes directly into MMA operands with a warp-register centroid table.
// Both codecs share the per-query softmax, exact ring, sink and split reduction; KVarN-specific metadata is described below.
//
// The record layout matches the windowed kernel; this path keeps query and attention operands in fp16.
// Work split: Here the block is (query rows) x (KV splits) and each split owns its own shared-memory ring and its
// own cp.async pipeline, published by a named barrier per strip, so the splits never lock-step with each
// other. Query rows share the packed bytes and metadata without sharing their masks.
//
// The sealed record layout (ggml_kvarn::code_bit) stores each (16-token strip, 16-channel tile) of a record as 32
// words, one per lane, where each word is exactly the lane's m16n8k16 A fragment: the K word holds the strip's
// tokens lane/4 (+8) x channels 2(lane%4)+{0,1} (+8), the V word holds V^T, channels lane/4 (+8) x tokens
// 2(lane%4)+{0,1} (+8). A strip of a record is therefore two contiguous 2 KB runs (K words, V words).
//
// Loading: every KV split owns a shared-memory ring of NSTAGE strips (K words, V words and the strip's Ktok, Vscale,
// Vzero halves) shared by the split's query-row warps: the 16-byte cp.async copies of a strip are spread over
// all lanes of the split, issued NSTAGE-1 strips ahead of the compute, and one named barrier per strip publishes the
// landed strip and frees the consumed stage. So a strip crosses L2 once per block whatever the decode width, and the
// words are read back with 4-byte, bank-conflict free shared loads right before their tensor-core MMA.
//
// K: the decode (q*Kscale[d] + Kzero[d])*Ktok[t] is folded into the query, Q'[j][d] = Q[j][d]*scale*Kscale[d] and
// c_r[j] = sum_d Kzero[d]*Q[j][d]*scale, so a strip costs one MMA per channel tile on the decoded codes and
// KQ[t][j] = Ktok[t]*(q.Q' + c_r[j]) + mask.
//
// V: (q*Vscale[t] + Vzero[t])*Vch[d] is never materialized. The fp16 magic-number extraction gives the raw code words
// as m[t][d] = 1024 + s[d]*q[t][d] (s = 1 for the low nibble of a byte, 16 for the high one, i.e. per channel half)
// and the MMA accumulates sum_t P'[t][j]*m[t][d] straight into acc with P' = P*Vscale[t] (two fp16 products per
// lane). acc is kept in units of u[d] = Vch[d]*s[d] of the current record (1 for exact rows): at every record
// change acc[d][j] += s[d]*pz[j] - 1024*ps[j] removes the code bias and adds the zero point, with
// ps[j] = sum_t P'[t][j] over the record (the fp16 values fed to the MMA, so the bias cancels exactly) and
// pz[j] = sum_t P[t][j]*Vzero[t], then acc *= u_old[d]/u_new[d]. The sealer clamps Vch to [exp(-0.3), exp(10)], so
// the unit ratios stay far inside the fp32 range. No decode arithmetic and no second accumulator on the V side.
//
// Work split: the body strips (records) form one contiguous range per KV split, walked with an online softmax; the
// exact f16 strips (sink and ring rows) go round-robin over all splits afterwards, each staged through the ring with
// cp.async (K then V, rows swizzled by 16-byte chunk) and read back in fragment order (V transposed with movmatrix)
// into acc directly, in true units. One warp serves one query row
// (all heads of the GQA group, padded to 8 columns); the warps of a block are (query rows) x (KV splits), the
// splits combine through shared memory (aliasing the ring after the loop) and the blocks through the
// parallel_blocks protocol of launch_fattn (flash_attn_combine_results).

#pragma once

#include "common.cuh"
#include "kvarn-lowbits.cuh"
#include "fattn-kvarn-lowbits-common.cuh"
#include "fattn-kvarn-lowbits-mma-f16.cuh" // fattn_kvarn_lowbits_u32_as_half2
#include "mma.cuh"
#include "fattn-kvarn-rot.cuh"

#include <cfloat>
#include <climits>
#include <cstdlib>

namespace fattn_kvarn_lowbits_stream {

using namespace ggml_cuda_mma;

constexpr int D       = 256;
constexpr int NTC     = D/16;    // channel tiles per row
constexpr int NCOLS2  = 8;       // head columns per query row (GQA padded to 8)
constexpr int UNIT    = 128;     // positions per work unit (one record)

constexpr int STRIP_BYTES = NTC*WARP_SIZE*4;           // one K or V strip: 512 words
constexpr int TURBO4_STRIP_BYTES = 16*(D/QK_TURBO4)*sizeof(block_turbo4_0);
constexpr int TURBO4_STAGE_BYTES = 2*TURBO4_STRIP_BYTES;
constexpr int META_BYTES  = 96;                        // ktok, vscale, vzero: 16 halves each
constexpr int META_KTOK   = 2*STRIP_BYTES;
constexpr int META_VS     = META_KTOK + 32;
constexpr int META_VZ     = META_VS   + 32;
constexpr int STAGE_BYTES = 2*STRIP_BYTES + META_BYTES; // 4192
constexpr int STAGE_CHUNKS = STAGE_BYTES/16;            // 16-byte cp.async chunks per strip (262)
constexpr int EXACT_ROW_BYTES = D*2;                    // one f16 row of an exact strip (512)
constexpr int EXACT_BYTES = 16*EXACT_ROW_BYTES;         // K or V of an exact strip (8192), staged in the ring
constexpr int RMETA_KZ    = 2*D;                        // per-record vectors staged in smem: Kscale, Kzero, Vch
constexpr int RMETA_VCH   = 4*D;
constexpr int RMETA_BYTES = 6*D;                        // 1536, double-buffered per split
constexpr int RMETA_CHUNKS = RMETA_BYTES/16;

constexpr int COL_STRIDE = D + 4; // fp32 stride of one head column in the combine buffer (bank spread)
constexpr int SLOT_FLOATS = NCOLS2*COL_STRIDE + 2*NCOLS2; // acc + (max, sum) per column
constexpr size_t SLOT_BYTES = SLOT_FLOATS*sizeof(float);

typedef tile<16, 8, half2> T_A;  // K strip (tokens x channels) / V^T strip (channels x tokens)
typedef tile< 8, 8, half2> T_B;  // Q'^T (heads x channels) / P (heads x tokens)
typedef tile<16, 8, float> T_C;  // KQ (tokens x heads) / O (channels x heads), fp32

static __host__ __device__ __forceinline__ size_t split_bytes(const int nstage, const bool turbo4_body = false) {
    return turbo4_body ? (size_t) nstage*TURBO4_STAGE_BYTES : (size_t) nstage*STAGE_BYTES + 2*RMETA_BYTES;
}
static __host__ __device__ __forceinline__ size_t ring_bytes(const int nsplit, const int nstage, const bool turbo4_body = false) {
    return (size_t) nsplit * split_bytes(nstage, turbo4_body);
}

// barrier among the NT row-warps of one KV split (named barriers 1..KW; 0 is __syncthreads)
static __device__ __forceinline__ void split_sync(const int id, const int nthreads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory");
}

// Combine buffer slot for one warp: acc[col][d] + meta.
static __device__ __forceinline__ float * slot_acc (float * base, const int slot) { return base + slot*SLOT_FLOATS; }
static __device__ __forceinline__ float * slot_meta(float * base, const int slot) { return base + slot*SLOT_FLOATS + NCOLS2*COL_STRIDE; }

// (x & mask) | magic in one LOP3 (the magic register is shared by all four extractions of a word)
static __device__ __forceinline__ uint32_t lop3_and_or(const uint32_t x, const uint32_t mask, const uint32_t magic) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return (x & mask) | magic;
#else
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(mask), "r"(magic));
    return r;
#endif
}

// raw fp16 code words: r[0] = 1024 + v(nibbles 0,4), r[1] = 1024 + 16 v(nibbles 1,5), r[2], r[3] likewise on x >> 8
static __device__ __forceinline__ void raw_pairs(const uint32_t x, const uint32_t magic, half2 * const __restrict__ r) {
    const uint32_t y = x >> 8;
    r[0] = fattn_kvarn_lowbits_u32_as_half2(lop3_and_or(x, 0x000F000FU, magic));
    r[1] = fattn_kvarn_lowbits_u32_as_half2(lop3_and_or(x, 0x00F000F0U, magic));
    r[2] = fattn_kvarn_lowbits_u32_as_half2(lop3_and_or(y, 0x000F000FU, magic));
    r[3] = fattn_kvarn_lowbits_u32_as_half2(lop3_and_or(y, 0x00F000F0U, magic));
}

// Decode native Turbo4 row pairs directly into MMA operands; the centroid table lives in warp registers.
static __device__ __forceinline__ half2 turbo4_pair(const char * packed, const int row, const int cp, const float centroid) {
    const block_turbo4_0 * b = (const block_turbo4_0 *) packed + row*(D/QK_TURBO4) + cp/(QK_TURBO4/2);
    const unsigned q = b->qs[cp % (QK_TURBO4/2)];
    const float norm = __half2float(b->norm);
    const float c0 = __shfl_sync(0xFFFFFFFF, centroid, q & 15, WARP_SIZE);
    const float c1 = __shfl_sync(0xFFFFFFFF, centroid, q >> 4, WARP_SIZE);
    return __floats2half2_rn(c0*norm, c1*norm);
}

// exact codes as fp16 (K side)
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

// strip cursor over the warp's position range: p0 = first position, (g, sl) = record and strip inside the body
struct cursor {
    int p0;
    int g;
    int sl;
};

#ifdef KVARN_STREAM_DBG_TIME
static __device__ unsigned int fattn_kvarn_lowbits_stream_dbg_run = 0;
#endif

template <int max_warps, int NSTAGE, int bits_k, int bits_v, bool turbo4_body = false, bool rot = false>
__launch_bounds__(WARP_SIZE*max_warps, max_warps == 4 || max_warps == 5 ? 2 : 1)
static __global__ void flash_attn_ext_kvarn_lowbits_stream(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
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
        const char    * __restrict__ kvarn_body,
        const int32_t * __restrict__ kvarn_desc) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    GGML_UNUSED_VARS(sinks, KV_max, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb03, ne10, ne13,
                     nb13, nb23, ne31, ne32, ne33, nb32, nb33);

    extern __shared__ __align__(16) char smem[];
#ifdef KVARN_STREAM_DBG_TIME
    const long long dbg_t0 = clock64();
    unsigned int dbg_smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(dbg_smid));
    const unsigned int dbg_run = atomicAdd(&fattn_kvarn_lowbits_stream_dbg_run, 0u);
    int dbg_nexact = 0, dbg_nrec = 0;
    long long dbg_wait = 0, dbg_body = 0, dbg_exact = 0, dbg_rec = 0, dbg_ts = 0;
#endif

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;
    const int n_q    = int(ne01.z);
    const int NT     = max_warps == 5 ? 5 : n_q; // specialize MTP-4 verification
    const int KW     = max_warps == 5 ? 1 : nwarps / NT;  // KV splits inside the block (host: nwarps == NT*KW, KW a power of two)
    const int nt     = max_warps == 5 ? warp : warp % NT;    // this warp's query row
    const int kw     = max_warps == 5 ? 0 : warp / NT;    // this warp's KV split

    const int z_KV  = blockIdx.z;    // KV head (ne03 == 1)
    const int gqa   = ne02 / ne12;
    const int head0 = z_KV*gqa;
    const int hcol  = lane >> 2;     // the lane's head column (B/C fragment row)
    const bool hvalid = hcol < gqa;

    const fattn_kvarn_lowbits_ctx kv = fattn_kvarn_lowbits_make_ctx(kvarn_body, kvarn_desc, z_KV, D, bits_k, bits_v, nb12, nb22);
    const int n_kv_pad = ne11;
    const int G16 = kv.G/16;

    // work split: the body strips (records) form one contiguous range per (block, split); the exact strips (sink and
    // ring rows, ~1% of the positions but ~2x the cost of a record strip) go round-robin over all splits so that no
    // warp becomes the tail of the kernel
    const int nsplit = gridDim.y * KW;
    const int split  = blockIdx.y * KW + kw;
    const int sb0    = kv.S/16, sb1 = kv.B/16;
    const int nbody  = sb1 - sb0;
    const int s_lo   = sb0 + (int) (((int64_t) nbody *  split     ) / nsplit);
    const int s_hi   = sb0 + (int) (((int64_t) nbody * (split + 1)) / nsplit);
    const int n_ex_sink = kv.S/16;
    const int n_ex      = n_ex_sink + (n_kv_pad - kv.B)/16; // padded positions are masked

    const float * Qf = (const float *) (Q + (size_t) nt*nb01 + (size_t) (head0 + (hvalid ? hcol : 0))*nb02);
    // [#139] rot build: Q arrives rotated (ggml_cuda_kvarn_rot256_q_pass in the launcher); only the output is rotated here
    static_assert(!rot || !turbo4_body, "turbo4 bodies use the grouped signed basis");
    const char  * mrow = mask + (size_t) nt*nb31;
    const char  * Kh = K + (size_t) z_KV*nb12;
    const char  * Vh = V + (size_t) z_KV*nb22;

    constexpr int stage_bytes = turbo4_body ? TURBO4_STAGE_BYTES : STAGE_BYTES;
    static_assert(NSTAGE*stage_bytes >= EXACT_BYTES, "the ring must hold one exact K or V strip");
    static_assert(NSTAGE <= 6, "a record (G >= 96 tokens) must span more strips than the issue lead, see rmeta");
    char * ring = smem + (size_t) kw*split_bytes(NSTAGE, turbo4_body); // shared by the NT row-warps of this split
    char * rmeta = ring + NSTAGE*STAGE_BYTES;              // [2][RMETA_BYTES]: record g's vectors in buffer g&1
    const uint32_t ring_s = ggml_cuda_cvta_generic_to_shared(ring);
    const uint32_t rmeta_s = ring_s + NSTAGE*STAGE_BYTES;
    auto sync_split = [&]() {
        if (NT == 1) {
            __syncwarp();
        } else {
            split_sync(1 + kw, NT*WARP_SIZE);
        }
    };
    const uint32_t magic = 0x64006400U;
    const float centroid = turbo4_body ? TURBO_CENTROIDS_4BIT[lane & 15] : 0.0f;
    // The other 32 TQ6 centroids are the exact sign-reversed mirror of these values.
    const float tq6_centroid = TQ6_CENTROIDS[lane];

    T_B   qp[NTC];   // Q' B fragments for the current record
    float c_r[2] = {0.0f, 0.0f};
    T_C   acc[NTC];  // O, fp32, unnormalized, in units of the current record's Vch*s (zero-initialized by the tile ctor)
    float kq_max[2] = {-FLT_MAX/2, -FLT_MAX/2};
    float kq_sum[2] = {0.0f, 0.0f};
    float ps[2] = {0.0f, 0.0f}; // sum of the fp16 P' values of the record (per head column, lane partial)
    float pz[2] = {0.0f, 0.0f}; // sum of P*Vzero of the record

    // Q for the lane's column, scaled: channel pairs 16c + 2(lane%4) (+8) as fp32
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

    // body record: Q' = Qs*Kscale (fp16), c_r = Kzero . Qs (fp32), both in the lane's fragment order
    auto setup_body = [&](const char * rm) { // rm = the record's staged vectors
        const uint2 * ks2 = (const uint2 *) (rm            + 2*(lane & 3)*(D/4));
        const uint2 * kz2 = (const uint2 *) (rm + RMETA_KZ + 2*(lane & 3)*(D/4));
        float cr = 0.0f;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 ks = ks2[c];
            const uint2 kz = kz2[c];
            float2 q0, q1;
            load_q(c, q0, q1);
            const half2 ks0 = fattn_kvarn_lowbits_u32_as_half2(ks.x), ks1 = fattn_kvarn_lowbits_u32_as_half2(ks.y);
            const float2 kz0 = __half22float2(fattn_kvarn_lowbits_u32_as_half2(kz.x));
            const float2 kz1 = __half22float2(fattn_kvarn_lowbits_u32_as_half2(kz.y));
            qp[c].x[0] = __hmul2(__floats2half2_rn(q0.x, q0.y), ks0);
            qp[c].x[1] = __hmul2(__floats2half2_rn(q1.x, q1.y), ks1);
            cr += q0.x*kz0.x + q0.y*kz0.y + q1.x*kz1.x + q1.y*kz1.y;
        }
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 1, WARP_SIZE);
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 2, WARP_SIZE);
        // C-fragment columns 2(lane%4) and +1 live in lanes 8(lane%4) and 8(lane%4)+4
        c_r[0] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3),     WARP_SIZE);
        c_r[1] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3) + 4, WARP_SIZE);
    };

    // exact rows: Q' = Qs, c_r = 0
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

    // record change old -> new (nullptr = exact rows): remove the code bias of the old record, add its zero point,
    // and convert acc = s*O/Vch from the old record's Vch to the new one's (s = 1 lo channel, 16 hi channel)
    auto record_change = [&](const char * rec_old, const char * rec_new) { // staged vectors (nullptr = exact rows)
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
#ifdef KVARN_DBG
            if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && lane == 0) {
                printf("[dbg] bias p0s=%g z0s=%g acc0=%g acc2=%g\n", p0s, z0s, acc[0].x[0], acc[0].x[2]);
            }
#endif
            const float lo0 = z0s - 1024.0f*p0s, lo1 = z1s - 1024.0f*p1s;
            const float hi0 = 16.0f*z0s - 1024.0f*p0s, hi1 = 16.0f*z1s - 1024.0f*p1s;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += lo0; acc[c].x[1] += lo1; acc[c].x[2] += hi0; acc[c].x[3] += hi1;
            }
            ps[0] = 0.0f; ps[1] = 0.0f;
            pz[0] = 0.0f; pz[1] = 0.0f;
        }
        const uint32_t * vo = rec_old != nullptr ? (const uint32_t *) (rec_old + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;
        const uint32_t * vn = rec_new != nullptr ? (const uint32_t *) (rec_new + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            float r_lo = 1.0f, r_hi = 1.0f;
            if (vo != nullptr) {
                const float2 v = __half22float2(fattn_kvarn_lowbits_u32_as_half2(vo[c])); // Vch[lo channel], Vch[hi channel]
                r_lo = v.x; r_hi = 0.0625f*v.y;
            }
            if (vn != nullptr) {
                const float2 v = __half22float2(fattn_kvarn_lowbits_u32_as_half2(vn[c]));
                r_lo = __fdividef(r_lo, v.x); r_hi = __fdividef(r_hi, 0.0625f*v.y);
            }
#ifdef KVARN_DBG
            if (c == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && lane == 0) {
                printf("[dbg] recchg old=%p new=%p r_lo=%g r_hi=%g acc0=%g acc2=%g ps=%g pz=%g kqmax=%g kqsum=%g\n",
                       (const void *) rec_old, (const void *) rec_new, r_lo, r_hi, acc[0].x[0], acc[0].x[2], ps[0], pz[0], kq_max[0], kq_sum[0]);
            }
#endif
            acc[c].x[0] *= r_lo; acc[c].x[1] *= r_lo; acc[c].x[2] *= r_hi; acc[c].x[3] *= r_hi;
        }
    };

    // online softmax on the strip's KQ tile (fp32, mask applied); leaves exp(kq - max) in kq and returns the fp16 P
    // as the B fragment for the V MMA
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

    auto is_body = [&](const int p0) { return p0 >= kv.S && p0 < kv.B; };
    auto rec_of  = [&](const cursor & cu) { return kv.body + (size_t) cu.g*kv.rec_stride; };
    auto cursor_init = [&](const int p0) {
        cursor cu;
        cu.p0 = p0;
        cu.g  = 0;
        cu.sl = 0;
        if (is_body(p0)) {
            cu.g  = (p0 - kv.S) / kv.G;
            cu.sl = ((p0 - kv.S) - cu.g*kv.G) / 16;
        }
        return cu;
    };
    auto cursor_next = [&](cursor & cu) {
        cu.p0 += 16;
        if (cu.p0 == kv.S) {
            cu.g = 0; cu.sl = 0;
        } else if (is_body(cu.p0)) {
            if (++cu.sl == G16) {
                cu.sl = 0;
                ++cu.g;
            }
        }
    };

    // Issue packed bytes and shared metadata across the split; masks remain private to each query.
    // with_meta: also stage the record's Kscale/Kzero/Vch into rmeta buffer g&1 (first strip of a record, or the
    // split's first strip); the buffer of record g-1 stays intact until record g+1 is issued, i.e. while record g
    // has more strips than the issue lead (G/16 > NSTAGE-1)
    auto issue = [&](const cursor & cu, const int stage, const bool with_meta) {
        const char * rec  = rec_of(cu);
        if constexpr (turbo4_body) {
            const char * ks = rec + cu.sl*TURBO4_STRIP_BYTES;
            const size_t v_offset = (size_t) kv.G*(D/QK_TURBO4)*sizeof(block_turbo4_0);
            for (int q = nt*WARP_SIZE + lane; q < TURBO4_STAGE_BYTES/16; q += NT*WARP_SIZE) {
                const char * src = q < TURBO4_STRIP_BYTES/16 ? ks + 16*q : ks + v_offset + 16*q - TURBO4_STRIP_BYTES;
                cp_async_cg_16<0>(ring_s + stage*stage_bytes + 16*q, src);
            }
            cp_async_commit();
            return;
        }
        if (with_meta) {
            half * metadata = reinterpret_cast<half *>(ring + NSTAGE*STAGE_BYTES + (cu.g & 1)*RMETA_BYTES);
            for (int i = nt*WARP_SIZE + lane; i < 3*D; i += NT*WARP_SIZE) {
                const bool is_v = i >= 2*D;
                const int slot = i % D;
                const int channel = is_v ? (bits_v == 4 ? slot : kvarn_lowbits::v_channel(slot))
                                         : (bits_k == 4 ? slot : kvarn_lowbits::k_channel(slot));
                const int offset = is_v ? kv.v_ch : i < D ? kv.k_scale : kv.k_zero;
                metadata[i] = reinterpret_cast<const half *>(rec + offset)[channel];
            }
        }
        uint32_t * codes = reinterpret_cast<uint32_t *>(ring + stage*STAGE_BYTES);
        for (int word = nt*WARP_SIZE + lane; word < STRIP_BYTES/4; word += NT*WARP_SIZE) {
            const int source_word = cu.sl*(STRIP_BYTES/4) + word;
            codes[word] = kvarn_lowbits::fragment_word<bits_k, false>(reinterpret_cast<const uint8_t *>(rec), source_word);
            codes[STRIP_BYTES/4 + word] = kvarn_lowbits::fragment_word<bits_v, true>(reinterpret_cast<const uint8_t *>(rec + kv.v_payload), source_word);
        }
        const uint32_t dst_s = ring_s + stage*STAGE_BYTES;
        for (int m = nt*WARP_SIZE + lane; m < META_BYTES/16; m += NT*WARP_SIZE) {
            const int half_off = 16*(m & 1);
            const char * src = m < 2 ? rec + kv.k_tok + cu.sl*32 + half_off
                : m < 4 ? rec + kv.v_scale + cu.sl*32 + half_off
                : rec + kv.v_zero + cu.sl*32 + half_off;
            cp_async_cg_16<0>(dst_s + 2*STRIP_BYTES + 16*m, src);
        }
        cp_async_commit();
    };

    constexpr int packed_row_bytes = D/QK_TQ6*sizeof(block_tq6_0);
    constexpr int packed_stride = (packed_row_bytes + 12 + 15)/16*16;
    constexpr int packed_strip_bytes = 16*packed_stride;
    constexpr int packed_offset = 2*EXACT_BYTES;
    constexpr bool packed_fits = NSTAGE*stage_bytes >= packed_offset + 2*packed_strip_bytes;
    const int packed_k_shift = (uintptr_t) Kh % 16;
    const int packed_v_shift = (uintptr_t) Vh % 16;
    const bool prefetch_tq6 = GGML_TQ6_PACKED_PREFETCH && packed_fits &&
        kv.type_k == GGML_TYPE_TQ6_0 && kv.type_v == GGML_TYPE_TQ6_0 &&
        nb12 == packed_row_bytes && nb22 == packed_row_bytes &&
        packed_k_shift <= 12 && packed_v_shift <= 12 &&
        nb11 % 16 == 0 && nb21 % 16 == 0 &&
        (size_t) z_KV*nb12 + packed_stride - packed_k_shift <= nb11 &&
        (size_t) z_KV*nb22 + packed_stride - packed_v_shift <= nb21 &&
        kv.cap > 0 && kv.cap % 16 == 0 && kv.S % 16 == 0 && kv.B % 16 == 0 &&
        ((uintptr_t) K % 16) == 0 && ((uintptr_t) V % 16) == 0;
    auto issue_packed = [&](const int e) {
        const int p = kv.B + 16*(e - n_ex_sink);
        const int row = kv.S + (p - kv.S) % kv.cap;
        const char * kr = Kh + (size_t) row*nb11 - packed_k_shift;
        const char * vr = Vh + (size_t) row*nb21 - packed_v_shift;
        for (int q = nt*WARP_SIZE + lane; q < 2*packed_strip_bytes/16; q += NT*WARP_SIZE) {
            const int byte = 16*q;
            const bool is_k = byte < packed_strip_bytes;
            const int local = is_k ? byte : byte - packed_strip_bytes;
            const int r = local / packed_stride;
            const int channel = local % packed_stride;
            const char * src = (is_k ? kr : vr) + (size_t) r*(is_k ? nb11 : nb21) + channel;
            cp_async_cg_16<0>(ring_s + packed_offset + byte, src);
        }
        cp_async_commit();
    };

    // stage one exact f16 strip (16 rows of D halves, K or V) into the ring, rows swizzled by 16-byte chunk
    auto issue_exact = [&](const char * rows, const size_t nb, const int type, const int offset, auto packed_tag) {
        constexpr bool packed = decltype(packed_tag)::value;
        for (int q = nt*WARP_SIZE + lane; q < EXACT_BYTES/16; q += NT*WARP_SIZE) {
            const int r = q / (EXACT_ROW_BYTES/16);
            const int j = q % (EXACT_ROW_BYTES/16);
            if (type != GGML_TYPE_F16) {
                half2 * out = (half2 *) (ring + offset + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)));
                if (type == GGML_TYPE_TQ6_0) {
                    const char * source = rows;
                    if constexpr (packed) {
                        source = ring + packed_offset + (offset == 0 ? packed_k_shift : packed_strip_bytes + packed_v_shift);
                    }
                    const block_tq6_0 * b = (const block_tq6_0 *) (source + (size_t) r*nb) + j/(QK_TQ6/8);
                    const int chunk = j % (QK_TQ6/8);
                    const float norm = __half2float(b->norm);
                    const uint32_t lo = ((const uint16_t *) b->qs)[2*chunk] |
                        (uint32_t) ((const uint16_t *) b->qs)[2*chunk+1] << 16;
                    const uint32_t hi = ((const uint16_t *) b->qh)[chunk];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int q0 = ((lo >> (8*i)) & 15) | (((hi >> (4*i)) & 3) << 4);
                        const int q1 = ((lo >> (8*i+4)) & 15) | (((hi >> (4*i+2)) & 3) << 4);
                        const float c0 = __shfl_sync(0xFFFFFFFF, tq6_centroid, q0 < 32 ? q0 : 63-q0, WARP_SIZE);
                        const float c1 = __shfl_sync(0xFFFFFFFF, tq6_centroid, q1 < 32 ? q1 : 63-q1, WARP_SIZE);
                        out[i] = __floats2half2_rn((q0 < 32 ? c0 : -c0)*norm, (q1 < 32 ? c1 : -c1)*norm);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        out[i] = fattn_kvarn_lowbits_ring_pair(rows + (size_t) r*nb, 4*j+i, type);
                    }
                }
            } else {
                cp_async_cg_16<0>(ring_s + offset + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)), rows + (size_t) r*nb + 16*j);
            }
        }
        cp_async_commit();
    };
    auto exact_word = [&](const int r, const int cp, const int offset) -> uint32_t { // row r, channel pair cp of the staged strip
        return *(const uint32_t *) (ring + offset + r*EXACT_ROW_BYTES + 16*((cp >> 2) ^ (r & 7)) + 4*(cp & 3));
    };

    if constexpr (turbo4_body) {
        setup_exact();
    }

    cursor ci = cursor_init(16*s_lo);   // issue cursor, NSTAGE-1 strips ahead
    cursor cc = ci;                      // compute cursor
    int stage_i = 0;                     // ring stage of the issue cursor
    int stage_c = 0;                     // ring stage of the compute cursor
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

    int          cur_id  = INT_MIN; // record index
    const char * cur_rm  = nullptr; // its staged vectors

    for (int s = s_lo; s < s_hi; ++s) {
        // refill the stage consumed by strip s-1 (freed by the barrier that ended it)
        if (s + NSTAGE - 1 < s_hi) {
            issue(ci, stage_i, ci.sl == 0);
            cursor_next(ci);
            stage_i = stage_i + 1 == NSTAGE ? 0 : stage_i + 1;
        } else {
            cp_async_commit();
        }
#ifdef KVARN_STREAM_DBG_TIME
        dbg_ts = clock64();
#endif
        cp_async_wait_group<NSTAGE - 1>(); // strip s has landed (this warp's chunks) ...
        sync_split();                      // ... and the other warps'
#ifdef KVARN_STREAM_DBG_TIME
        { const long long t = clock64(); dbg_wait += t - dbg_ts; dbg_ts = t; }
#endif

        const int    id = cc.g;
        const char * rm = rmeta + (id & 1)*RMETA_BYTES;
#ifdef KVARN_STREAM_DBG_TIME
        dbg_nrec += id != cur_id ? 1 : 0;
#endif
        if (!turbo4_body && id != cur_id) {
            if (cur_id != INT_MIN) {
                record_change(cur_rm, rm);
            }
            setup_body(rm);
            cur_id = id;
            cur_rm = rm;
        }
#ifdef KVARN_STREAM_DBG_TIME
        { const long long t = clock64(); dbg_rec += t - dbg_ts; dbg_ts = t; }
#endif
        if constexpr (turbo4_body) {
            T_C kq;
            const char * st = ring + stage_c*stage_bytes;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = (lane >> 2) + 8*(l & 1);
                    const int cp = 8*c + 4*(l >> 1) + (lane & 3);
                    a.x[l] = turbo4_pair(st, row, cp, centroid);
                }
                mma(kq, a, qp[c]);
            }
            const half * mk = (const half *) mrow + cc.p0;
            const float mk0 = __half2float(mk[lane >> 2]);
            const float mk1 = __half2float(mk[(lane >> 2) + 8]);
            kq.x[0] += mk0; kq.x[1] += mk0;
            kq.x[2] += mk1; kq.x[3] += mk1;
            const T_B P = softmax(kq);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = (lane >> 2) + 8*(l >> 1);
                    const int cp = 8*c + 4*(l & 1) + (lane & 3);
                    a.x[l] = ggml_cuda_movmatrix(turbo4_pair(st + TURBO4_STRIP_BYTES, row, cp, centroid));
                }
                mma(acc[c], a, P);
            }
        } else {
            T_C kq;
            const char * st = ring + stage_c*STAGE_BYTES;
            const uint32_t * kwp = (const uint32_t *) st + lane;
            const uint32_t * vwp = (const uint32_t *) (st + STRIP_BYTES) + lane;

            // KQ = Ktok[t] * (q . Q' + c_r) + mask
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                decode_pairs(kwp[c*WARP_SIZE], magic, a.x);
                mma(kq, a, qp[c]);
            }
            const half * ktok = (const half *) (st + META_KTOK);
            const half * mk   = (const half *) mrow + cc.p0;
            const half * vsh  = (const half *) (st + META_VS);
            const half * vzh  = (const half *) (st + META_VZ);
            const float tok0 = __half2float(ktok[lane >> 2]);
            const float tok1 = __half2float(ktok[(lane >> 2) + 8]);
            const float mk0  = __half2float(mk[lane >> 2]);
            const float mk1  = __half2float(mk[(lane >> 2) + 8]);
            kq.x[0] = tok0*(kq.x[0] + c_r[0]) + mk0;
            kq.x[1] = tok0*(kq.x[1] + c_r[1]) + mk0;
            kq.x[2] = tok1*(kq.x[2] + c_r[0]) + mk1;
            kq.x[3] = tok1*(kq.x[3] + c_r[1]) + mk1;

            T_B P = softmax(kq);

            // P' = P*Vscale[t] (fp16, B layout: tokens 2(lane%4)+{0,1} and +8), ps from the same fp16 values in the
            // C layout (tokens lane/4 and +8), pz = P*Vzero in fp32
            const half2 vs_b0 = *(const half2 *) (vsh + 2*(lane & 3));
            const half2 vs_b1 = *(const half2 *) (vsh + 2*(lane & 3) + 8);
            P.x[0] = __hmul2(P.x[0], vs_b0);
            P.x[1] = __hmul2(P.x[1], vs_b1);
            const half2 vs_c = __halves2half2(vsh[lane >> 2], vsh[(lane >> 2) + 8]);
            const float2 vz_c = make_float2(__half2float(vzh[lane >> 2]), __half2float(vzh[(lane >> 2) + 8]));
            const float2 pe0 = __half22float2(__hmul2(make_half2(kq.x[0], kq.x[2]), vs_c));
            const float2 pe1 = __half22float2(__hmul2(make_half2(kq.x[1], kq.x[3]), vs_c));
            ps[0] += pe0.x + pe0.y;
            ps[1] += pe1.x + pe1.y;
            pz[0] += kq.x[0]*vz_c.x + kq.x[2]*vz_c.y;
            pz[1] += kq.x[1]*vz_c.x + kq.x[3]*vz_c.y;

            // acc += raw V codes . P' (units Vch*s)
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                raw_pairs(vwp[c*WARP_SIZE], magic, a.x);
                mma(acc[c], a, P);
            }
        }
#ifdef KVARN_STREAM_DBG_TIME
        { const long long t = clock64(); dbg_body += t - dbg_ts; }
#endif
        sync_split(); // everyone is done with the stage of strip s
        cursor_next(cc);
        stage_c = stage_c + 1 == NSTAGE ? 0 : stage_c + 1;
    }
    if (!turbo4_body && cur_id != INT_MIN) {
        record_change(cur_rm, nullptr); // back to true units
    }
    cp_async_wait_all();
    sync_split(); // the ring is free (this also covers empty ranges)

    // exact f16 rows (sink, ring): strips split, split + nsplit, ... in true units
    if (split < n_ex) {
        setup_exact();
    }
    if (prefetch_tq6 && split >= n_ex_sink && split < n_ex) {
        issue_packed(split);
        cp_async_wait_all();
        sync_split();
    }
    for (int e = split; e < n_ex; e += nsplit) {
        const int p0 = e < n_ex_sink ? 16*e : kv.B + 16*(e - n_ex_sink);
#ifdef KVARN_STREAM_DBG_TIME
        dbg_nexact += 1;
        dbg_ts = clock64();
#endif
        {
            T_C kq;
            // exact f16 rows, staged K then V through the ring, read in fragment order: K straight, V via an 8x8
            // transpose
            const int row0 = p0 < kv.S ? p0 : kv.S + (p0 - kv.S) % kv.cap;
            const bool f16_sink = p0 < kv.S && kv.sink_type == GGML_TYPE_F16;
            const char * Kr = f16_sink ? fattn_kvarn_lowbits_sink_row(Kh, kv, nb11, row0, false) : Kh + (size_t) row0*nb11;
            const char * Vr = f16_sink ? fattn_kvarn_lowbits_sink_row(Vh, kv, nb21, row0, true) : Vh + (size_t) row0*nb21;
            const half * mrow_h = (const half *) mrow;
            const float mk0 = __half2float(mrow_h[p0 + (lane >> 2)]);
            const float mk1 = __half2float(mrow_h[p0 + (lane >> 2) + 8]);
            // Reuse the body ring for both exact tiles when it already has enough room.
            constexpr bool paired_exact = NSTAGE*stage_bytes >= 2*EXACT_BYTES;
            constexpr int v_offset = paired_exact ? EXACT_BYTES : 0;
            if (prefetch_tq6 && e >= n_ex_sink) {
                issue_exact(nullptr, packed_stride, GGML_TYPE_TQ6_0, 0, std::true_type{});
                issue_exact(nullptr, packed_stride, GGML_TYPE_TQ6_0, v_offset, std::true_type{});
            } else {
                issue_exact(Kr, f16_sink ? kv.sink_stride : nb11, f16_sink ? GGML_TYPE_F16 : kv.type_k, 0, std::false_type{});
                if constexpr (paired_exact) {
                    issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v, v_offset, std::false_type{});
                }
            }
            cp_async_wait_all();
            sync_split();
            // Packed readers are done; copy the next strip during current MMA work.
            if (prefetch_tq6 && e + nsplit >= n_ex_sink && e + nsplit < n_ex) {
                issue_packed(e + nsplit);
            }
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = (lane >> 2) + 8*(l & 1);
                    const int cp  = 8*c + 4*(l >> 1) + (lane & 3);
                    a.x[l] = fattn_kvarn_lowbits_u32_as_half2(exact_word(row, cp, 0));
                }
                mma(kq, a, qp[c]);
            }
            kq.x[0] += mk0;
            kq.x[1] += mk0;
            kq.x[2] += mk1;
            kq.x[3] += mk1;

            const T_B P = softmax(kq);

            if constexpr (!paired_exact) {
                sync_split(); // everyone has read K
                issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v, 0, std::false_type{});
                cp_async_wait_all();
                sync_split();
            }
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = 8*(l >> 1) + (lane >> 2);      // token row of the 8x8 block
                    const int cp  = 8*c + 4*(l & 1) + (lane & 3);  // channel pair
                    a.x[l] = ggml_cuda_movmatrix(fattn_kvarn_lowbits_u32_as_half2(exact_word(row, cp, v_offset)));
                }
                mma(acc[c], a, P);
            }
            if (prefetch_tq6) {
                cp_async_wait_all();
            }
            sync_split(); // everyone has read V before the next strip (or the combine) reuses the ring
        }
#ifdef KVARN_STREAM_DBG_TIME
        { const long long t = clock64(); dbg_exact += t - dbg_ts; }
#endif
    }
#ifdef KVARN_STREAM_DBG_TIME
    {
        const long long dbg_t1 = clock64();
        if (lane == 0 && (dbg_run == 5 || dbg_run == 300)) {
            printf("[tm] run=%u sm=%u by=%d bz=%d w=%d t0=%lld t1=%lld strips=%d exact=%d rec=%d wait=%lld body=%lld exactc=%lld recc=%lld\n",
                   dbg_run, dbg_smid, (int) blockIdx.y, (int) blockIdx.z, warp, dbg_t0, dbg_t1, s_hi - s_lo, dbg_nexact, dbg_nrec,
                   dbg_wait, dbg_body, dbg_exact, dbg_rec);
        }
    }
#endif

    // finish the lane-partial sums
#pragma unroll
    for (int off = 4; off < WARP_SIZE; off <<= 1) {
        kq_sum[0] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[0], off, WARP_SIZE);
        kq_sum[1] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[1], off, WARP_SIZE);
    }

    // combine the KV splits of the block (pairwise tree, upper half stores, lower half merges); the combine slots
    // alias the ring, so all warps must be past their KV loop first
    float * smem_combine = (float *) smem;
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

    // output: heads head0 + col for col < gqa; unnormalized parts + meta when several blocks share the KV range
    const bool single = gridDim.y == 1;
    if constexpr (rot) {
        // [#139] fused rotation: the final rows are normalized first, then rotated (combine_results rotates parts)
        if (single) {
            const float inv0 = 1.0f/kq_sum[0], inv1 = 1.0f/kq_sum[1];
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[c].x[l] *= (l & 1) ? inv1 : inv0;
                }
            }
            kvarn_rot256_frag<T_C, NTC>(acc, lane);
            kq_sum[0] = 1.0f;
            kq_sum[1] = 1.0f;
        }
    }
#pragma unroll
    for (int l = 0; l < 4; l += 2) {
        // l and l+1 share the row d, differ in column
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*(lane & 3) + e;
            if (col >= gqa) {
                continue;
            }
            const int head = head0 + col;
            const float inv = single ? 1.0f/kq_sum[e] : 1.0f;
            float * out = dst + ((size_t) (nt*ne02 + head)*gridDim.y + blockIdx.y)*D;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const int d = 16*c + (lane >> 2) + 8*(l >> 1);
                out[d] = acc[c].x[l + e]*inv;
            }
        }
    }
#ifdef KVARN_DBG
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && lane == 0) {
        printf("[dbg] out single=%d kqmax=%g kqsum=%g acc0=%g acc2=%g out0=%g out8=%g\n", (int) single, kq_max[0], kq_sum[0],
               acc[0].x[0], acc[0].x[2], acc[0].x[0]/kq_sum[0], acc[0].x[2]/kq_sum[0]);
    }
#endif
#ifdef KVARN_STREAM_DBG_TIME
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && lane == 0) {
        __threadfence();
        atomicAdd(&fattn_kvarn_lowbits_stream_dbg_run, 1u);
    }
#endif
    if (!single && lane < 4) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*lane + e;
            if (col < gqa) {
                dst_meta[(size_t) (nt*ne02 + head0 + col)*gridDim.y + blockIdx.y] = make_float2(kq_max[e], kq_sum[e]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                     ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33, kvarn_body, kvarn_desc);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

} // namespace fattn_kvarn_lowbits_stream


// Dedicated lower-bit instantiations leave the selected 4/4 kernels unchanged.
template<int max_warps, int bits_k, int bits_v, bool rot = false>
static void ggml_cuda_kvarn_lowbits_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int nt, int kw) {
    using namespace fattn_kvarn_lowbits_stream;
    constexpr int stages = 6;
    fattn_kernel_t kernel = flash_attn_ext_kvarn_lowbits_stream<max_warps, stages, bits_k, bits_v, false, rot>;
    const int id = ggml_cuda_get_device();
    // #139: the rot build plans its grid from the plain build's occupancy (register counts differ), so both launch alike
    fattn_kernel_t plain = flash_attn_ext_kvarn_lowbits_stream<max_warps, stages, bits_k, bits_v, false, false>;
    const size_t shared = std::max(ring_bytes(kw, stages), (size_t) nt*(kw/2)*SLOT_BYTES);
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};
    if (!initialized[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int) ggml_cuda_info().devices[id].smpbo));
        if (rot) {
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(plain), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           (int) ggml_cuda_info().devices[id].smpbo));
        }
        initialized[id] = true;
    }
    launch_fattn<D, NCOLS2, NCOLS2>(ctx, dst, kernel, nt*kw, shared, UNIT,
        false, false, false, false, WARP_SIZE, 0, rot ? plain : nullptr);
}

template<int bits_k, int bits_v, bool rot = false>
static void ggml_cuda_kvarn_lowbits_width(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int nt = dst->src[0]->ne[1];
    GGML_ASSERT(nt >= 1 && nt <= 8);
    // [#139] the rot build rotates the output in-kernel; the plain build must not see the flag
    GGML_ASSERT(rot == ggml_cuda_fattn_kvarn_rot(dst));
    // [#139] rot: Q pre-pass here (pool buffer, read-only loads in the kernel), the node copy carries the rotated Q
    ggml_cuda_pool_alloc<float> q_rot;
    ggml_tensor q2, d2;
    if constexpr (rot) {
        q_rot.alloc(ctx.pool(), ggml_nelements(dst->src[0]));
        ggml_cuda_kvarn_rot256_q_pass(ctx, dst->src[0], q_rot.ptr, &q2);
        d2 = *dst;
        d2.src[0] = &q2;
        dst = &d2;
    }
    if (nt <= 4) {
        ggml_cuda_kvarn_lowbits_launch<4,bits_k,bits_v,rot>(ctx,dst,nt,nt == 1 ? 2 : 4/nt);
    } else if (nt == 5) {
        ggml_cuda_kvarn_lowbits_launch<5,bits_k,bits_v,rot>(ctx,dst,nt,1);
    } else {
        ggml_cuda_kvarn_lowbits_launch<8,bits_k,bits_v,rot>(ctx,dst,nt,1);
    }
}

static bool ggml_cuda_kvarn_lowbits_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (ggml_get_op_params_i32(dst,5)) {
        case (4 << 8) | 3: ggml_cuda_kvarn_lowbits_width<4,3>(ctx,dst); return true;
        case (3 << 8) | 3: ggml_cuda_kvarn_lowbits_width<3,3>(ctx,dst); return true;
        case (4 << 8) | 2: ggml_cuda_kvarn_lowbits_width<4,2>(ctx,dst); return true;
        case (2 << 8) | 4: ggml_cuda_kvarn_lowbits_width<2,4>(ctx,dst); return true;
        case (3 << 8) | 2: ggml_cuda_kvarn_lowbits_width<3,2>(ctx,dst); return true;
        case (2 << 8) | 2: ggml_cuda_kvarn_lowbits_width<2,2>(ctx,dst); return true;
        default: return false;
    }
}
