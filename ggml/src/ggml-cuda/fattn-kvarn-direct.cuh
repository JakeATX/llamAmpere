// KVarN fragment-direct decode attention (D = 256, 4-bit K and V records, decode widths 1..8).
//
// The sealed record layout (ggml_kvarn::code_bit) stores each (16-token strip, 16-channel tile) of a record as 32
// words, one per lane, where each word is exactly the lane's m16n8k16 A fragment: the K word holds the strip's
// tokens lane/4 (+8) x channels 2(lane%4)+{0,1} (+8), the V word holds V^T, channels lane/4 (+8) x tokens
// 2(lane%4)+{0,1} (+8). A strip of a record is therefore two contiguous 2 KB runs (K words, V words).
//
// Loading: the block owns one shared-memory ring of NWIN windows of W strips (K words, V words and the strip's
// Ktok, Vscale, Vzero halves) plus the per-record vectors (Kscale, Kzero, Vch, triple-buffered). All lanes of the
// block issue the 16-byte cp.async copies of a window, NWIN-1 windows ahead of the compute, and two __syncthreads per
// window publish the landed window and free the consumed one. So a strip crosses L2 once per block whatever the
// decode width, the record vectors are read from shared memory at a record change, and the words are read back with
// 4-byte, bank-conflict free shared loads right before their tensor-core MMA.
//
// K: the decode (q*Kscale[d] + Kzero[d])*Ktok[t] is folded into the query. Per record, Q'[j][d] = Q[j][d]*scale*Kscale[d]
// is quantized to int8 per head column (scale sq) and c_r[j] = sum_d Kzero[d]*Q[j][d]*scale + 7.5*sum_d eps[d] (the DC
// part of the quantization error), so a strip costs 8 u8.s8 IMMAs straight on the code words and
// KQ[t][j] = Ktok[t]*(sq*(codes . Qi) + c_r[j]) + mask. The K MMAs of a strip are issued one strip ahead of its softmax.
//
// V: (q*Vscale[t] + Vzero[t])*Vch[d] is never materialized. Per strip, e[t] = exp(kq - strip max) (<= 1) and
// u[t] = round(255*e[t]*Vscale[t]/max_t Vscale) go to the tensor core as u8 against the raw V codes (u8.u8 IMMA, exact
// int32 sums), and acc[d] += w*(vsmax/255)*sum_t u[t]*code[t][d] with w = exp(strip max - running max). acc is kept in
// units of Vch[d] of the current record (1 for exact rows): at a record change acc[d] += pz (pz = sum_t P[t]*Vzero[t]
// over the record) and acc *= Vch_old[d]/Vch_new[d]. The sealer clamps Vch to [exp(-0.3), exp(10)], so the unit ratios
// stay far inside the fp32 range.
//
// Work split: the body strips form one contiguous range per block, cut into windows of W strips. The (row, residue)
// items of a window (row-major, W per query row) are cut into 4 equal ranges, one per SM sub-partition; the two warps
// of a sub-partition (w, w + 4) split their range at the row boundary it crosses, or in halves when it lies in one row.
// So every sub-partition carries the same load at every decode width and one warp serves one query row (all heads of
// the GQA group, padded to 8 columns). The exact f16 strips (sink and ring rows) go round-robin over the blocks
// afterwards, each staged through the ring with cp.async (K then V, rows swizzled by 16-byte chunk) and read back in
// fragment order (V transposed with movmatrix) into the row's primary warp, in true units. The warps of a row combine
// through shared memory (aliasing the ring after the loop) and the blocks through the parallel_blocks protocol of
// launch_fattn (flash_attn_combine_results).

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh" // fattn_kvarn_u32_as_half2
#include "mma.cuh"
#include "fattn-kvarn-stream.cuh"

#include <cfloat>
#include <climits>
#include <cstdlib>

namespace fattn_kvarn_direct {

using namespace ggml_cuda_mma;

constexpr int D       = 256;
constexpr int NTC     = D/16;    // channel tiles per row
constexpr int NCOLS2  = 8;       // head columns per query row (GQA padded to 8)
constexpr int UNIT    = 128;     // positions per work unit (one record)
constexpr int NWARPS  = 8;       // warps per block (two per SM sub-partition)

constexpr int STRIP_BYTES = NTC*WARP_SIZE*4;           // one K or V strip: 512 words
constexpr int META_BYTES  = 128;                       // ktok, vscale, vzero, (unused): 16 halves each
constexpr int META_KTOK   = 2*STRIP_BYTES;
constexpr int META_VS     = META_KTOK + 32;
constexpr int META_VZ     = META_VS   + 32;
constexpr int STAGE_BYTES = 2*STRIP_BYTES + META_BYTES; // 4224
constexpr int STAGE_CHUNKS = STAGE_BYTES/16;            // 16-byte cp.async chunks per strip (264)
constexpr int EXACT_ROW_BYTES = D*2;                    // one f16 row of an exact strip (512)
constexpr int EXACT_BYTES = 16*EXACT_ROW_BYTES;         // K or V of an exact strip (8192), staged in the ring
constexpr int RMETA_KZ    = 2*D;                        // per-record vectors staged in smem: Kscale, Kzero, Vch
constexpr int RMETA_VCH   = 4*D;
constexpr int RMETA_BYTES = 6*D;                        // 1536, triple-buffered (record g in buffer g % 3)
constexpr int RMETA_CHUNKS = RMETA_BYTES/16;
constexpr int RMETA_NBUF  = 3; // buffers for an issue lead of one or two windows (see rmeta_nbuf)
constexpr int Q_ROW_BYTES = NTC*WARP_SIZE*8;            // one row's scaled fp16 Q in fragment order (tile, lane) x uint2
constexpr int NQ_MAX      = 8;
// integer MMA results read as fp32 bit patterns (no I2F): C is initialized to a bias whose binade holds the whole
// result range, float(bits) - bias is then exact (the K side lands in [2^24, 2^25): ulp 2, folded into the scales)
constexpr int32_t KBIAS_I = 0x4BC00000;   // 1.5*2^24, K results |r| < 2^23
constexpr float   KBIAS_F = 25165824.0f;
constexpr int32_t VBIAS_I = 0x4B000000;   // 2^23, V results 0 <= r < 2^23
constexpr float   VBIAS_F = 8388608.0f;

static __device__ __forceinline__ uint32_t fattn_kvarn_half2_as_u32(const half2 h) {
    return *((const uint32_t *) &h);
}

// swizzle of the 16-byte chunks of a staged record vector (Kscale/Kzero: 64 chunks) against the fragment-order reads
static __host__ __device__ __forceinline__ int rmeta_swz(const int q) {
    return q < 64 ? q ^ (((q >> 3) & 3) << 1) : q;
}

constexpr int COL_STRIDE = D + 4; // fp32 stride of one head column in the combine buffer (bank spread)
constexpr int SLOT_FLOATS = NCOLS2*COL_STRIDE + 2*NCOLS2; // acc + (max, sum) per column
constexpr size_t SLOT_BYTES = SLOT_FLOATS*sizeof(float);
constexpr int NSLOTS = NWARPS - 1; // warp 0 is always a primary

typedef tile<16, 8, half2> T_A;  // K strip (tokens x channels) / V^T strip (channels x tokens)
typedef tile< 8, 8, half2> T_B;  // Q'^T (heads x channels) / P (heads x tokens)
typedef tile<16, 8, float> T_C;  // KQ (tokens x heads) / O (channels x heads), fp32

// record vectors in flight: the windows staged at once span (nwin*W + 7)/8 records, plus the one being consumed
static __host__ __device__ __forceinline__ constexpr int rmeta_nbuf(const int W, const int nwin) {
    return (nwin*W + 7)/8 + 1 > RMETA_NBUF ? (nwin*W + 7)/8 + 1 : RMETA_NBUF;
}
static __host__ __device__ __forceinline__ constexpr size_t ring_bytes(const int W, const int nwin) {
    return (size_t) nwin * W * STAGE_BYTES + rmeta_nbuf(W, nwin)*RMETA_BYTES;
}

// warp w of a block with n_q rows and windows of W strips: the query row and window residues [lo, hi) it computes.
// The W*n_q (row, residue) items are cut into 4 ranges of T = W*n_q/4 items (7 rows are cut like 8, the last row
// empty); sub-partition q = w & 3 owns range q, warp q takes its part in the range's first row, warp q + 4 the rest.
static __host__ __device__ __forceinline__ void kvarn_direct_assign(const int w, const int n_q, const int W, int & row, int & lo, int & hi) {
    const int nq = n_q == 7 ? 8 : n_q;
    const int T  = (W*nq)/4;
    const int q  = w & 3;
    const int i0 = q*T, i1 = i0 + T;
    const int r0 = i0/W;
    const int b  = W*(r0 + 1);
    if (i1 <= b) {           // the range lies in one row: halves
        const int h = T/2;
        row = r0;
        lo  = (w < 4 ? i0 : i0 + h) - W*r0;
        hi  = lo + (w < 4 ? h : T - h);
    } else if (w < 4) {      // the range crosses a row boundary: first row ...
        row = r0;
        lo  = i0 - W*r0;
        hi  = W;
    } else {                 // ... and second row
        row = r0 + 1;
        lo  = 0;
        hi  = i1 - b;
    }
    if (hi <= lo || row >= n_q) {
        lo = hi = 0;
    }
}

// Combine buffer slot for one warp: acc[col][d] + meta.
static __device__ __forceinline__ float * slot_acc (float * base, const int slot) { return base + slot*SLOT_FLOATS; }
static __device__ __forceinline__ float * slot_meta(float * base, const int slot) { return base + slot*SLOT_FLOATS + NCOLS2*COL_STRIDE; }

// C += A(16 tokens x 32 channels, u8) . B(32 channels x 8 heads, s8), s32 accumulate
static __device__ __forceinline__ void imma_u8s8(int32_t & c0, int32_t & c1, int32_t & c2, int32_t & c3,
        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3, const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    NO_DEVICE_CODE;
#endif
}

// C = A(16 channels x 16 tokens, u8) . B(16 tokens x 8 heads, u8), s32
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

#ifdef KVARN_DBG_TIME
static __device__ unsigned int fattn_kvarn_dbg_run = 0;
#endif

template <int W, int NWIN>
__launch_bounds__(WARP_SIZE*NWARPS, 1)
static __global__ void flash_attn_ext_kvarn_direct(
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
#ifdef KVARN_DBG_TIME
    const long long dbg_t0 = clock64();
    unsigned int dbg_smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(dbg_smid));
    const unsigned int dbg_run = atomicAdd(&fattn_kvarn_dbg_run, 0u);
    int dbg_nexact = 0, dbg_nrec = 0, dbg_nstrip = 0;
    long long dbg_wait = 0, dbg_body = 0, dbg_exact = 0, dbg_rec = 0, dbg_ts = 0;
#endif

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int n_q    = int(ne01.z);
    constexpr int NSTAGE   = W*NWIN;
    constexpr int NTHREADS = WARP_SIZE*NWARPS;
    const int tid = warp*WARP_SIZE + lane;
    int row, res_lo, res_hi;         // this warp's query row and window residues
    kvarn_direct_assign(warp, n_q, W, row, res_lo, res_hi);
    const bool active  = res_hi > res_lo;
    const bool primary = active && res_lo == 0; // the row's warp that also takes the exact strips, combines, writes

    const int z_KV  = blockIdx.z;    // KV head (ne03 == 1)
    const int gqa   = ne02 / ne12;
    const int head0 = z_KV*gqa;
    const int hcol  = lane >> 2;     // the lane's head column (B/C fragment row)
    const bool hvalid = hcol < gqa;

    const fattn_kvarn_ctx kv = fattn_kvarn_make_ctx(kvarn_body, kvarn_desc, z_KV, D, 4, 4, nb12, nb22);
    const int n_kv_pad = ne11;

    // work split: body strips [s_lo, s_hi) in windows of W strips; exact strips (sink, ring) round-robin over blocks
    const int sb0    = kv.S/16, sb1 = kv.B/16;
    const int nbody  = sb1 - sb0;
    const int s_lo   = sb0 + (int) (((int64_t) nbody *  blockIdx.y     ) / gridDim.y);
    const int s_hi   = sb0 + (int) (((int64_t) nbody * (blockIdx.y + 1)) / gridDim.y);
    const int nwin   = (s_hi - s_lo + W - 1)/W;
    const int n_ex_sink = kv.S/16;
    const int n_ex      = n_ex_sink + (n_kv_pad - kv.B)/16; // padded positions are masked

    const char  * mrow = mask + (size_t) row*nb31;
    const half  * mrow_h = (const half *) mrow;
    const char  * Kh = K + (size_t) z_KV*nb12;
    const char  * Vh = V + (size_t) z_KV*nb22;

    static_assert(NSTAGE*STAGE_BYTES >= EXACT_BYTES, "the ring must hold one exact K or V strip");
    static_assert(ring_bytes(W, NWIN) + Q_ROW_BYTES >= NSLOTS*SLOT_BYTES, "the shared memory must hold the combine slots");
    static_assert(NWIN >= 2 && NWIN <= 4, "rmeta buffering assumes an issue lead of at most three windows");
    constexpr int RNBUF = rmeta_nbuf(W, NWIN);
    char * ring  = smem;                           // [NSTAGE][STAGE_BYTES], window k in stages W*(k % NWIN) ..
    char * rmeta = ring + NSTAGE*STAGE_BYTES;      // [RNBUF][RMETA_BYTES]: record g's vectors in buffer g % RNBUF
    char * qsm   = rmeta + RNBUF*RMETA_BYTES; // [n_q][Q_ROW_BYTES]: scaled fp16 Q, (tile, lane) x uint2
    const uint32_t ring_s  = ggml_cuda_cvta_generic_to_shared(ring);
    const uint32_t rmeta_s = ring_s + NSTAGE*STAGE_BYTES;

    // stage Q: warp w writes row w (published by the first window's barrier); lane -> channels 16c + 2j (+1), +8 (+9)
    if (warp < n_q) {
        const float * Qw = (const float *) (Q + (size_t) warp*nb01 + (size_t) (head0 + (hvalid ? hcol : 0))*nb02);
        uint2 * qdst = (uint2 *) (qsm + warp*Q_ROW_BYTES) + lane;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            float2 q0 = make_float2(0.0f, 0.0f), q1 = make_float2(0.0f, 0.0f);
            if (hvalid) {
                q0 = *(const float2 *) (Qw + 16*c + 2*(lane & 3));
                q1 = *(const float2 *) (Qw + 16*c + 2*(lane & 3) + 8);
                q0.x *= scale; q0.y *= scale; q1.x *= scale; q1.y *= scale;
            }
            qdst[c*WARP_SIZE] = make_uint2(fattn_kvarn_half2_as_u32(__floats2half2_rn(q0.x, q0.y)),
                                           fattn_kvarn_half2_as_u32(__floats2half2_rn(q1.x, q1.y)));
        }
    }
    const uint2 * qrow = (const uint2 *) (qsm + row*Q_ROW_BYTES) + lane;

    T_B      qp[NTC];   // Q' B fragments (fp16) for the exact rows
    uint32_t qi[NTC];   // Q' B fragments (int8, channels 2j, 2j+8, 2j+1, 2j+9 of the tile) for the current record
    float c_r[2] = {0.0f, 0.0f};
    float sq_lo[2] = {0.0f, 0.0f}; // int8 scale of the C-fragment columns / 2 (low-nibble rows, bias ulp 2)
    float sq_hi[2] = {0.0f, 0.0f}; // / 32 (high-nibble rows: codes*16)
    T_C   acc[NTC];  // O, fp32, unnormalized, in units of the current record's Vch (zero-initialized by the tile ctor)
    float kq_max[2] = {-FLT_MAX/2, -FLT_MAX/2};
    float kq_sum[2] = {0.0f, 0.0f};
    float pz[2] = {0.0f, 0.0f}; // sum of P*Vzero of the record (per head column, lane partial)

    // the lane's uint2 of a record vector (Kscale at rm, Kzero at rm + RMETA_KZ): halves {2j, 2j+1 | 2j+8, 2j+9} of tile c
    auto rvec = [&](const char * base, const int c) -> uint2 {
        return *(const uint2 *) (base + 16*rmeta_swz(8*(lane & 3) + (c >> 1)) + 8*(c & 1));
    };

    // body record, K side: Q' = Qs*Kscale (half2) quantized to int8 per head column, sq = absmax/126.5 (so |i| <= 127
    // after the fp16 roundings); c_r = Kzero . Qs + 7.5*sum(eps) (fp32; the DC part of the quantization error over the
    // 0..15 codes), in the lane's fragment order
    auto setup_body = [&](const char * rm) { // rm = the record's staged vectors
        half2 am = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 q  = qrow[c*WARP_SIZE];
            const uint2 ks = rvec(rm, c);
            const half2 p0 = __hmul2(fattn_kvarn_u32_as_half2(q.x), fattn_kvarn_u32_as_half2(ks.x));
            const half2 p1 = __hmul2(fattn_kvarn_u32_as_half2(q.y), fattn_kvarn_u32_as_half2(ks.y));
            am = __hmax2(am, __hmax2(__habs2(p0), __habs2(p1)));
        }
        float amax = fmaxf(__low2float(am), __high2float(am));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1, WARP_SIZE));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2, WARP_SIZE));
        const float sq  = amax > 0.0f ? amax*(1.0f/126.5f) : 0.0f;
        const float isq = amax > 0.0f ? 126.5f/amax : 0.0f;
        const half2 isq2 = __float2half2_rn(isq);
        const half2 nsq2 = __float2half2_rn(-sq);
        const half2 mag  = fattn_kvarn_u32_as_half2(0x64806480U); // 1152 = 1024 + 128
        float cr   = 0.0f;
        half2 eps2 = make_half2(0.0f, 0.0f);
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 q  = qrow[c*WARP_SIZE];
            const uint2 ks = rvec(rm, c);
            const uint2 kz = rvec(rm + RMETA_KZ, c);
            const half2 q0 = fattn_kvarn_u32_as_half2(q.x), q1 = fattn_kvarn_u32_as_half2(q.y);
            const half2 p0 = __hmul2(q0, fattn_kvarn_u32_as_half2(ks.x));
            const half2 p1 = __hmul2(q1, fattn_kvarn_u32_as_half2(ks.y));
            const half2 h0 = __hfma2(p0, isq2, mag); // 1024 + 128 + i, integer: low byte = i + 128
            const half2 h1 = __hfma2(p1, isq2, mag);
            // bytes [ch 2j, 2j+8, 2j+1, 2j+9] = [h0.lo, h1.lo, h0.hi, h1.hi], then two's complement
            qi[c] = __byte_perm(fattn_kvarn_half2_as_u32(h0), fattn_kvarn_half2_as_u32(h1), 0x6240) ^ 0x80808080U;
            eps2 = __hadd2(eps2, __hfma2(__hsub2(h0, mag), nsq2, p0));
            eps2 = __hadd2(eps2, __hfma2(__hsub2(h1, mag), nsq2, p1));
            const float2 z0 = __half22float2(__hmul2(q0, fattn_kvarn_u32_as_half2(kz.x)));
            const float2 z1 = __half22float2(__hmul2(q1, fattn_kvarn_u32_as_half2(kz.y)));
            cr += (z0.x + z0.y) + (z1.x + z1.y);
        }
        cr += 7.5f*(__low2float(eps2) + __high2float(eps2));
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 1, WARP_SIZE);
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 2, WARP_SIZE);
        // C-fragment columns 2(lane%4) and +1 live in lanes 8(lane%4) and 8(lane%4)+4
        c_r[0]  = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3),     WARP_SIZE);
        c_r[1]  = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3) + 4, WARP_SIZE);
        const float sqc0 = __shfl_sync(0xFFFFFFFF, sq, 8*(lane & 3),     WARP_SIZE);
        const float sqc1 = __shfl_sync(0xFFFFFFFF, sq, 8*(lane & 3) + 4, WARP_SIZE);
        sq_lo[0] = 0.5f*sqc0;         sq_lo[1] = 0.5f*sqc1;
        sq_hi[0] = (1.0f/32.0f)*sqc0; sq_hi[1] = (1.0f/32.0f)*sqc1;
    };

    // exact rows: Q' = Qs (fp16 B fragments), c_r = 0
    auto setup_exact = [&]() {
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 q = qrow[c*WARP_SIZE];
            qp[c].x[0] = fattn_kvarn_u32_as_half2(q.x);
            qp[c].x[1] = fattn_kvarn_u32_as_half2(q.y);
        }
        c_r[0] = 0.0f;
        c_r[1] = 0.0f;
    };

    // record change old -> new (nullptr = exact rows), acc side: add the old record's zero point and convert
    // acc = O/Vch from the old record's Vch to the new one's
    auto record_change = [&](const char * rec_old, const char * rec_new) { // staged vectors (nullptr = exact rows)
        if (rec_old != nullptr) {
            float z0s = pz[0], z1s = pz[1];
#pragma unroll
            for (int off = 4; off < WARP_SIZE; off <<= 1) {
                z0s += __shfl_xor_sync(0xFFFFFFFF, z0s, off, WARP_SIZE);
                z1s += __shfl_xor_sync(0xFFFFFFFF, z1s, off, WARP_SIZE);
            }
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += z0s; acc[c].x[1] += z1s; acc[c].x[2] += z0s; acc[c].x[3] += z1s;
            }
            pz[0] = 0.0f; pz[1] = 0.0f;
        }
        const uint32_t * vo = rec_old != nullptr ? (const uint32_t *) (rec_old + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;
        const uint32_t * vn = rec_new != nullptr ? (const uint32_t *) (rec_new + RMETA_VCH + 2*((lane >> 2)*(D/8))) : nullptr;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            float r_lo = 1.0f, r_hi = 1.0f;
            if (vo != nullptr) {
                const float2 v = __half22float2(fattn_kvarn_u32_as_half2(vo[c])); // Vch[lo channel], Vch[hi channel]
                r_lo = v.x; r_hi = v.y;
            }
            if (vn != nullptr) {
                const float2 v = __half22float2(fattn_kvarn_u32_as_half2(vn[c]));
                r_lo = __fdividef(r_lo, v.x); r_hi = __fdividef(r_hi, v.y);
            }
            acc[c].x[0] *= r_lo; acc[c].x[1] *= r_lo; acc[c].x[2] *= r_hi; acc[c].x[3] *= r_hi;
        }
    };

    // online softmax on a strip's KQ tile (fp32, mask applied): strip max per column, running max update (rescales
    // acc, sums), returns the strip weights w = exp(strip max - running max) and leaves the strip max in mx
    auto softmax_max = [&](const T_C & kq, float & mx0, float & mx1, float & w0, float & w1) {
        mx0 = fmaxf(kq.x[0], kq.x[2]);
        mx1 = fmaxf(kq.x[1], kq.x[3]);
#pragma unroll
        for (int off = 4; off < WARP_SIZE; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, off, WARP_SIZE));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, off, WARP_SIZE));
        }
        mx0 = fmaxf(mx0, -FLT_MAX/2); // fully masked strips
        mx1 = fmaxf(mx1, -FLT_MAX/2);
        const float nm0 = fmaxf(kq_max[0], mx0);
        const float nm1 = fmaxf(kq_max[1], mx1);
        const bool changed = nm0 != kq_max[0] || nm1 != kq_max[1];
        if (__any_sync(0xFFFFFFFF, changed)) {
            const float sc0 = __expf(kq_max[0] - nm0);
            const float sc1 = __expf(kq_max[1] - nm1);
            kq_max[0] = nm0;
            kq_max[1] = nm1;
            kq_sum[0] *= sc0;
            kq_sum[1] *= sc1;
            pz[0] *= sc0; pz[1] *= sc1;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] *= sc0; acc[c].x[1] *= sc1; acc[c].x[2] *= sc0; acc[c].x[3] *= sc1;
            }
        }
        w0 = __expf(mx0 - nm0);
        w1 = __expf(mx1 - nm1);
    };

    // exact rows: fp16 P = exp(kq - running max) as the B fragment of the fp16 V MMA
    auto softmax_exact = [&](T_C & kq) -> T_B {
        float mx0, mx1, w0, w1;
        softmax_max(kq, mx0, mx1, w0, w1);
        kq.x[0] = __expf(kq.x[0] - kq_max[0]);
        kq.x[1] = __expf(kq.x[1] - kq_max[1]);
        kq.x[2] = __expf(kq.x[2] - kq_max[0]);
        kq.x[3] = __expf(kq.x[3] - kq_max[1]);
        kq_sum[0] += kq.x[0] + kq.x[2];
        kq_sum[1] += kq.x[1] + kq.x[3];
        return get_transposed(get_half2(kq));
    };

    // window issue: strips s_lo + W*k + i (i < W) into stages W*(k % NWIN) + i, the chunks spread over all lanes of
    // the block; the per-record vectors ride with a record's first strip (or the block's first strip) into rmeta
    // buffer g % 3. One commit group per window.
    auto issue_window = [&](const int k) {
        const int base = W*(k % NWIN);
#pragma unroll
        for (int i = 0; i < W; ++i) {
            const int s = s_lo + W*k + i;
            if (s < s_hi) {
                const int pb = 16*s - kv.S;
                const int g  = pb / kv.G;
                const int sl = (pb - g*kv.G) / 16;
                const char * rec = kv.body + (size_t) g*kv.rec_stride;
                const char * ks  = rec + sl*STRIP_BYTES;
                const uint32_t dst_s = ring_s + (base + i)*STAGE_BYTES;
#pragma unroll
                for (int q = tid; q < STAGE_CHUNKS; q += NTHREADS) {
                    const char * src;
                    if (q < STRIP_BYTES/16) {
                        src = ks + 16*q;
                    } else if (q < 2*STRIP_BYTES/16) {
                        src = ks + kv.v_payload + 16*(q - STRIP_BYTES/16);
                    } else {
                        const int m = q - 2*STRIP_BYTES/16; // 0..5: ktok, vscale, vzero (two chunks each)
                        const int half_off = 16*(m & 1);
                        src = m < 2 ? rec + kv.k_tok   + sl*32 + half_off
                            : m < 4 ? rec + kv.v_scale + sl*32 + half_off
                            :         rec + kv.v_zero  + sl*32 + half_off;
                    }
                    cp_async_cg_16<0>(dst_s + 16*q, src);
                }
                if (sl == 0 || s == s_lo) {
                    const uint32_t dst_m = rmeta_s + (g % RNBUF)*RMETA_BYTES;
#pragma unroll
                    for (int q = tid; q < RMETA_CHUNKS; q += NTHREADS) {
                        const char * src = q < RMETA_VCH/16 ? rec + kv.k_scale + 16*q : rec + kv.v_ch + 16*(q - RMETA_VCH/16);
                        cp_async_cg_16<0>(dst_m + 16*rmeta_swz(q), src);
                    }
                }
            }
        }
        cp_async_commit();
    };

    // stage one exact f16 strip (16 rows of D halves, K or V) into the ring, rows swizzled by 16-byte chunk
    static_assert((EXACT_BYTES/16) % NTHREADS == 0, "issue_exact: every thread runs the same trip count");
    static_assert((EXACT_BYTES/16) % WARP_SIZE == 0, "fattn_kvarn_ring_pair_warp needs whole warps per exact strip");
    auto issue_exact = [&](const char * rows, const size_t nb, const int type) {
        // per-strip load keeps the tq6 lane register out of the kernel's live range (#127)
        const float tq6_mag = type == GGML_TYPE_TQ6_0 ? TQ6_CENTROIDS[32 + lane] : 0.0f;
#pragma unroll
        for (int q = tid; q < EXACT_BYTES/16; q += NTHREADS) {
            const int r = q / (EXACT_ROW_BYTES/16);
            const int j = q % (EXACT_ROW_BYTES/16);
            if (type != GGML_TYPE_F16) {
                half2 * out = (half2 *) (ring + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)));
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    out[i] = fattn_kvarn_ring_pair_warp(rows + (size_t) r*nb, 4*j+i, type, tq6_mag); // EXACT_BYTES/16 % NTHREADS == 0: all lanes
                }
            } else {
                cp_async_cg_16<0>(ring_s + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)), rows + (size_t) r*nb + 16*j);
            }
        }
        cp_async_commit();
    };
    auto exact_word = [&](const int r, const int cp) -> uint32_t { // row r, channel pair cp of the staged strip
        return *(const uint32_t *) (ring + r*EXACT_ROW_BYTES + 16*((cp >> 2) ^ (r & 7)) + 4*(cp & 3));
    };

    // K MMAs of a staged strip: 8 u8.s8 IMMAs on the words (low nibbles = token r, high nibbles = token r+8,
    // channels 2j, 2j+8, 2j+1, 2j+9 of the tile) against the current record's Qi
    auto k_mma = [&](const char * st, int32_t & k0, int32_t & k1, int32_t & k2, int32_t & k3) {
        const uint32_t * kwp = (const uint32_t *) st + lane;
        k0 = KBIAS_I; k1 = KBIAS_I; k2 = KBIAS_I; k3 = KBIAS_I;
#pragma unroll
        for (int c = 0; c < NTC; c += 2) { // high nibbles stay in place: rows 8..15 (token r+8) come out 16x
            const uint32_t x0 = kwp[c*WARP_SIZE];
            const uint32_t x1 = kwp[(c + 1)*WARP_SIZE];
            imma_u8s8(k0, k1, k2, k3, x0 & 0x0F0F0F0FU, x0 & 0xF0F0F0F0U, x1 & 0x0F0F0F0FU, x1 & 0xF0F0F0F0U, qi[c], qi[c + 1]);
        }
    };

#pragma unroll
    for (int k = 0; k < NWIN - 1; ++k) {
        if (k < nwin) {
            issue_window(k);
        } else {
            cp_async_commit();
        }
    }

    int          cur_id = INT_MIN; // record index of the K side (Qi, c_r, sq)
    const char * cur_rm = nullptr; // its staged vectors
    const char * acc_rm = nullptr; // record of the acc side (differs from cur_rm while a change is pending)

    // this warp's mask values (token rows lane/4 and +8) for the strips of a window, prefetched one window ahead
    float mkc[W][2], mkn[W][2];
    auto load_masks = [&](const int k, float (&m)[W][2]) {
#pragma unroll
        for (int i = 0; i < W; ++i) {
            const int s = s_lo + W*k + i;
            const bool use = i >= res_lo && i < res_hi && s < s_hi;
            m[i][0] = use ? __half2float(mrow_h[16*s + (lane >> 2)])     : 0.0f;
            m[i][1] = use ? __half2float(mrow_h[16*s + (lane >> 2) + 8]) : 0.0f;
        }
    };
    auto selw = [&](const float (&m)[W][2], const int j, const int e) -> float {
        float r = m[0][e];
#pragma unroll
        for (int i = 1; i < W; ++i) {
            r = j == i ? m[i][e] : r;
        }
        return r;
    };
    load_masks(0, mkc);

    for (int k = 0; k < nwin; ++k) {
        // refill the window consumed at k-1 (freed by the barrier that ended it)
        if (k + NWIN - 1 < nwin) {
            issue_window(k + NWIN - 1);
        } else {
            cp_async_commit();
        }
#ifdef KVARN_DBG_TIME
        dbg_ts = clock64();
#endif
        cp_async_wait_group<NWIN - 1>(); // window k has landed (this thread's chunks) ...
        __syncthreads();                 // ... and everyone's
#ifdef KVARN_DBG_TIME
        { const long long t = clock64(); dbg_wait += t - dbg_ts; }
#endif
        load_masks(k + 1, mkn);
        const char * win = ring + W*(k % NWIN)*STAGE_BYTES;

        // prime the pipeline: K MMAs of the warp's first strip of the window (record change: both sides at once)
        int32_t kn0 = 0, kn1 = 0, kn2 = 0, kn3 = 0;
        if (active && s_lo + W*k + res_lo < s_hi) {
            const int s  = s_lo + W*k + res_lo;
            const int id = (16*s - kv.S)/kv.G;
            if (id != cur_id) {
#ifdef KVARN_DBG_TIME
                dbg_nrec += 1;
                dbg_ts = clock64();
#endif
                const char * rm = rmeta + (id % RNBUF)*RMETA_BYTES;
                if (cur_id != INT_MIN) {
                    record_change(acc_rm, rm);
                }
                setup_body(rm);
                cur_id = id;
                cur_rm = rm;
                acc_rm = rm;
#ifdef KVARN_DBG_TIME
                { const long long t = clock64(); dbg_rec += t - dbg_ts; }
#endif
            }
            k_mma(win + res_lo*STAGE_BYTES, kn0, kn1, kn2, kn3);
        }
        for (int j = res_lo; j < res_hi; ++j) {
            const int s = s_lo + W*k + j;
            if (s >= s_hi) {
                break;
            }
#ifdef KVARN_DBG_TIME
            dbg_nstrip += 1;
            dbg_ts = clock64();
#endif
            const char * st = win + j*STAGE_BYTES;
            const half * ktok = (const half *) (st + META_KTOK);
            const half * vsh  = (const half *) (st + META_VS);
            const half * vzh  = (const half *) (st + META_VZ);

            // KQ = Ktok[t] * (sq*(codes . Qi) + c_r) + mask, with the K-side constants of this strip's record
            T_C kq;
            {
                const float tok0 = __half2float(ktok[lane >> 2]);
                const float tok1 = __half2float(ktok[(lane >> 2) + 8]);
                const float mk0 = selw(mkc, j, 0);
                const float mk1 = selw(mkc, j, 1);
                kq.x[0] = tok0*(sq_lo[0]*(__int_as_float(kn0) - KBIAS_F) + c_r[0]) + mk0;
                kq.x[1] = tok0*(sq_lo[1]*(__int_as_float(kn1) - KBIAS_F) + c_r[1]) + mk0;
                kq.x[2] = tok1*(sq_hi[0]*(__int_as_float(kn2) - KBIAS_F) + c_r[0]) + mk1;
                kq.x[3] = tok1*(sq_hi[1]*(__int_as_float(kn3) - KBIAS_F) + c_r[1]) + mk1;
            }
            // a record change pending from the previous strip: the acc side, after that strip's V MMAs
            if (acc_rm != cur_rm) {
                record_change(acc_rm, cur_rm);
                acc_rm = cur_rm;
            }
            // K MMAs of the next strip of the window (independent of the softmax below); its record's K side first
            if (j + 1 < res_hi && s + 1 < s_hi) {
                const int id = (16*(s + 1) - kv.S)/kv.G;
                if (id != cur_id) {
#ifdef KVARN_DBG_TIME
                    dbg_nrec += 1;
                    const long long t0 = clock64();
#endif
                    const char * rm = rmeta + (id % RNBUF)*RMETA_BYTES;
                    setup_body(rm);
                    cur_id = id;
                    cur_rm = rm;
#ifdef KVARN_DBG_TIME
                    { const long long t = clock64(); dbg_rec += t - t0; }
#endif
                }
                k_mma(st + STAGE_BYTES, kn0, kn1, kn2, kn3);
            }

            // softmax: e = exp(kq - strip max) <= 1, strip weights w; sums and pz in true P = w*e
            float mx0, mx1, w0, w1;
            softmax_max(kq, mx0, mx1, w0, w1);
            kq.x[0] = __expf(kq.x[0] - mx0);
            kq.x[1] = __expf(kq.x[1] - mx1);
            kq.x[2] = __expf(kq.x[2] - mx0);
            kq.x[3] = __expf(kq.x[3] - mx1);
            kq_sum[0] += w0*(kq.x[0] + kq.x[2]);
            kq_sum[1] += w1*(kq.x[1] + kq.x[3]);
            const float2 vz_c = make_float2(__half2float(vzh[lane >> 2]), __half2float(vzh[(lane >> 2) + 8]));
            pz[0] += w0*(kq.x[0]*vz_c.x + kq.x[2]*vz_c.y);
            pz[1] += w1*(kq.x[1]*vz_c.x + kq.x[3]*vz_c.y);

            // u = round(255*e*Vscale[t]/vsmax) as u8 in the B layout (tokens 2j, 2j+8, 2j+1, 2j+9 of head g)
            half2 vsm;
            {
                const uint4 va = *(const uint4 *) vsh;
                const uint4 vb = *(const uint4 *) (vsh + 8);
                vsm = __hmax2(__hmax2(fattn_kvarn_u32_as_half2(va.x), fattn_kvarn_u32_as_half2(va.y)),
                              __hmax2(fattn_kvarn_u32_as_half2(va.z), fattn_kvarn_u32_as_half2(va.w)));
                vsm = __hmax2(vsm, __hmax2(__hmax2(fattn_kvarn_u32_as_half2(vb.x), fattn_kvarn_u32_as_half2(vb.y)),
                                           __hmax2(fattn_kvarn_u32_as_half2(vb.z), fattn_kvarn_u32_as_half2(vb.w))));
            }
            const float vsmax = __half2float(__hmax(__low2half(vsm), __high2half(vsm)));
            const float rvs   = vsmax > 0.0f ? __fdividef(255.0f, vsmax) : 0.0f;
            const float vs0 = __half2float(vsh[lane >> 2])*rvs;
            const float vs1 = __half2float(vsh[(lane >> 2) + 8])*rvs;
            // round(u) is the low byte of the fp32 magic sum u + 2^23; the C-layout bytes (token g | g+8 rows, heads
            // 2j, 2j+1) go through the 8x8 b16 transpose as 16-bit elements, then the token pairs interleave
            const uint32_t m0 = __float_as_uint(fmaf(kq.x[0], vs0, VBIAS_F));
            const uint32_t m1 = __float_as_uint(fmaf(kq.x[1], vs0, VBIAS_F));
            const uint32_t m2 = __float_as_uint(fmaf(kq.x[2], vs1, VBIAS_F));
            const uint32_t m3 = __float_as_uint(fmaf(kq.x[3], vs1, VBIAS_F));
            const uint32_t t0 = (uint32_t) ggml_cuda_movmatrix((int) __byte_perm(m0, m1, 0x5410)); // tokens 2j, 2j+1 of head g
            const uint32_t t1 = (uint32_t) ggml_cuda_movmatrix((int) __byte_perm(m2, m3, 0x5410)); // tokens 2j+8, 2j+9
            const uint32_t pu = __byte_perm(t0, t1, 0x6240);
            const float f0 = w0*vsmax*(1.0f/255.0f);
            const float f1 = w1*vsmax*(1.0f/255.0f);
            const float f2 = f0*(1.0f/16.0f), f3 = f1*(1.0f/16.0f); // high-nibble rows (channel g+8): codes*16

            // acc += f * (raw V codes . u): u8.u8 IMMA per tile, exact int32 sums
            const uint32_t * vwp = (const uint32_t *) (st + STRIP_BYTES) + lane;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const uint32_t x = vwp[c*WARP_SIZE];
                int32_t r0, r1, r2, r3;
                imma_u8u8_16(r0, r1, r2, r3, x & 0x0F0F0F0FU, x & 0xF0F0F0F0U, pu, VBIAS_I);
                acc[c].x[0] += f0*(__int_as_float(r0) - VBIAS_F);
                acc[c].x[1] += f1*(__int_as_float(r1) - VBIAS_F);
                acc[c].x[2] += f2*(__int_as_float(r2) - VBIAS_F);
                acc[c].x[3] += f3*(__int_as_float(r3) - VBIAS_F);
            }
#ifdef KVARN_DBG_TIME
            { const long long t = clock64(); dbg_body += t - dbg_ts; }
#endif
        }
        __syncthreads(); // everyone is done with window k
#pragma unroll
        for (int i = 0; i < W; ++i) {
            mkc[i][0] = mkn[i][0];
            mkc[i][1] = mkn[i][1];
        }
    }
    if (cur_id != INT_MIN) {
        record_change(acc_rm, nullptr); // back to true units
    }
    cp_async_wait_all();
    __syncthreads(); // the ring is free

    // exact f16 rows (sink, ring): strips blockIdx.y, + gridDim.y, ... into the rows' primary warps, in true units
    if (primary) {
        setup_exact();
    }
    for (int e = blockIdx.y; e < n_ex; e += (int) gridDim.y) {
        const int p0 = e < n_ex_sink ? 16*e : kv.B + 16*(e - n_ex_sink);
#ifdef KVARN_DBG_TIME
        dbg_nexact += 1;
        dbg_ts = clock64();
#endif
        // exact f16 rows, staged K then V through the ring, read in fragment order: K straight, V via an 8x8 transpose
        const int row0 = p0 < kv.S ? p0 : kv.S + (p0 - kv.S) % kv.cap;
        const bool f16_sink = p0 < kv.S && kv.sink_type == GGML_TYPE_F16;
        const char * Kr = f16_sink ? fattn_kvarn_sink_row(Kh, kv, nb11, row0, false) : Kh + (size_t) row0*nb11;
        const char * Vr = f16_sink ? fattn_kvarn_sink_row(Vh, kv, nb21, row0, true) : Vh + (size_t) row0*nb21;
        T_B P;
        issue_exact(Kr, f16_sink ? kv.sink_stride : nb11, f16_sink ? GGML_TYPE_F16 : kv.type_k);
        cp_async_wait_all();
        __syncthreads();
        if (primary) {
            T_C kq;
            const float mk0 = __half2float(mrow_h[p0 + (lane >> 2)]);
            const float mk1 = __half2float(mrow_h[p0 + (lane >> 2) + 8]);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = (lane >> 2) + 8*(l & 1);
                    const int cp  = 8*c + 4*(l >> 1) + (lane & 3);
                    a.x[l] = fattn_kvarn_u32_as_half2(exact_word(row, cp));
                }
                mma(kq, a, qp[c]);
            }
            kq.x[0] += mk0;
            kq.x[1] += mk0;
            kq.x[2] += mk1;
            kq.x[3] += mk1;

            P = softmax_exact(kq);
        }
        __syncthreads(); // everyone has read K
        issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v);
        cp_async_wait_all();
        __syncthreads();
        if (primary) {
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = 8*(l >> 1) + (lane >> 2);      // token row of the 8x8 block
                    const int cp  = 8*c + 4*(l & 1) + (lane & 3);  // channel pair
                    a.x[l] = ggml_cuda_movmatrix(fattn_kvarn_u32_as_half2(exact_word(row, cp)));
                }
                mma(acc[c], a, P);
            }
        }
        __syncthreads(); // everyone has read V before the next strip (or the combine) reuses the ring
#ifdef KVARN_DBG_TIME
        { const long long t = clock64(); dbg_exact += t - dbg_ts; }
#endif
    }
#ifdef KVARN_DBG_TIME
    {
        const long long dbg_t1 = clock64();
        if (lane == 0 && (dbg_run == 5 || dbg_run == 300)) {
            printf("[tm] run=%u sm=%u by=%d bz=%d w=%d t0=%lld t1=%lld strips=%d exact=%d rec=%d wait=%lld body=%lld exactc=%lld recc=%lld\n",
                   dbg_run, dbg_smid, (int) blockIdx.y, (int) blockIdx.z, warp, dbg_t0, dbg_t1, dbg_nstrip, dbg_nexact, dbg_nrec,
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

    // combine the warps of a row: the secondary warps (residues starting above 0; never warp 0) store their partial
    // (acc, max, sum) in slot warp - 1, the row's primary merges them; the slots alias the ring (and the rest)
    float * smem_combine = (float *) smem;
    if (active && !primary) {
        const int slot = warp - 1;
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
    if (primary) {
        for (int p = 1; p < NWARPS; ++p) {
            int prow, plo, phi;
            kvarn_direct_assign(p, n_q, W, prow, plo, phi);
            if (!(phi > plo && plo > 0 && prow == row)) {
                continue;
            }
            const float * a = slot_acc (smem_combine, p - 1);
            const float * m = slot_meta(smem_combine, p - 1);
            const float om0 = m[2*(2*(lane & 3) + 0) + 0], os0 = m[2*(2*(lane & 3) + 0) + 1];
            const float om1 = m[2*(2*(lane & 3) + 1) + 0], os1 = m[2*(2*(lane & 3) + 1) + 1];
            const float nm0 = fmaxf(kq_max[0], om0), nm1 = fmaxf(kq_max[1], om1);
            const float sa0 = __expf(kq_max[0] - nm0), sb0 = __expf(om0 - nm0);
            const float sa1 = __expf(kq_max[1] - nm1), sb1 = __expf(om1 - nm1);
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
    }

    if (!primary) {
        return;
    }

    // output: heads head0 + col for col < gqa; unnormalized parts + meta when several blocks share the KV range
    const bool single = gridDim.y == 1;
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
            float * out = dst + ((size_t) (row*ne02 + head)*gridDim.y + blockIdx.y)*D;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const int d = 16*c + (lane >> 2) + 8*(l >> 1);
                out[d] = acc[c].x[l + e]*inv;
            }
        }
    }
#ifdef KVARN_DBG_TIME
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && lane == 0) {
        __threadfence();
        atomicAdd(&fattn_kvarn_dbg_run, 1u);
    }
#endif
    if (!single && lane < 4) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*lane + e;
            if (col < gqa) {
                dst_meta[(size_t) (row*ne02 + head0 + col)*gridDim.y + blockIdx.y] = make_float2(kq_max[e], kq_sum[e]);
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

} // namespace fattn_kvarn_direct

// Host side. Blocks of 8 warps (one per SM). Widths 1..7 run windows of 8 strips with a ring 2 windows deep (one
// window in flight while one is computed); width 8 (and env GGML_KVARN_DIRECT_W4=1 for all widths) runs windows of 4
// strips 3 deep, whose smaller ring leaves room for the 8 staged Q rows.
inline bool ggml_cuda_flash_attn_ext_kvarn_direct_supported_impl(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (sinks != nullptr || mask == nullptr || Q->ne[1] < 1 || Q->ne[1] > 8 || Q->ne[3] != 1) {
        return false;
    }
    if (ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16) { // trellis body: tile-loader path only for now
        return false;
    }
    if (((uintptr_t) mask->data) % 16 != 0 || mask->nb[1] % 16 != 0) {
        return false;
    }
    const int gqa = Q->ne[2] / K->ne[2];
    if (gqa < 1 || gqa > fattn_kvarn_direct::NCOLS2) {
        return false;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!cp_async_available(cc)) {
        return false;
    }
    static const bool disabled = getenv("GGML_KVARN_NO_DIRECT") != nullptr;
    return !disabled;
}

template <int W, int NWIN, int NQ_MAX_T>
static bool ggml_cuda_flash_attn_ext_kvarn_direct_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using namespace fattn_kvarn_direct;
    fattn_kernel_t fattn_kernel = flash_attn_ext_kvarn_direct<W, NWIN>;
    GGML_ASSERT(!ggml_cuda_fattn_kvarn_rot(dst)); // [#139] no in-kernel rotation: fattn.cu runs it in separate passes
    const int n_q = dst->src[0]->ne[1];
    GGML_ASSERT(n_q <= NQ_MAX_T);
    const size_t nbytes_shared = ring_bytes(W, NWIN) + (size_t) n_q*Q_ROW_BYTES; // the combine slots alias the ring
    const int id = ggml_cuda_get_device();
    // This kernel has no static shared memory. Check the current width before opting in.
    const size_t limit = ggml_cuda_info().devices[id].smpbo;
    if (nbytes_shared > limit) {
        return false;
    }
    static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_memory_limit_raised[id]) {
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        (int) std::min(limit, ring_bytes(W, NWIN) + (size_t) NQ_MAX_T*Q_ROW_BYTES)));
        shared_memory_limit_raised[id] = true;
    }
    launch_fattn<D, NCOLS2, NCOLS2>
        (ctx, dst, fattn_kernel, NWARPS, nbytes_shared, UNIT,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false, /*use_sparse=*/false, WARP_SIZE);
    return true;
}

// Use the same half-MMA arithmetic across supported decode widths. The existing override retains the integer path for comparison.
inline void ggml_cuda_flash_attn_ext_kvarn_direct_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 7) == GGML_TYPE_TURBO4_0) {
        ggml_cuda_flash_attn_ext_kvarn_stream_impl<true>(ctx, dst);
        return;
    }
    static const bool w8_env = getenv("GGML_KVARN_DIRECT_W8") != nullptr && atoi(getenv("GGML_KVARN_DIRECT_W8")) != 0;
    static const int stream_max = getenv("GGML_KVARN_DIRECT_STREAM_MAX") != nullptr
                                ? atoi(getenv("GGML_KVARN_DIRECT_STREAM_MAX")) : 8;
    const int n_q = dst->src[0]->ne[1];
    if (n_q <= stream_max) {
        ggml_cuda_flash_attn_ext_kvarn_stream_impl(ctx, dst);
        return;
    }
    static const bool w4d_env = getenv("GGML_KVARN_DIRECT_NWIN4") != nullptr && atoi(getenv("GGML_KVARN_DIRECT_NWIN4")) != 0;
    if (w4d_env) {
        if (ggml_cuda_flash_attn_ext_kvarn_direct_launch<4, 4, 8>(ctx, dst)) {
            return;
        }
    } else if (w8_env && n_q <= 7) {
        if (ggml_cuda_flash_attn_ext_kvarn_direct_launch<8, 2, 7>(ctx, dst)) {
            return;
        }
    }
    // Reduce the ring first to keep the same integer arithmetic when an override does not fit.
    if (!ggml_cuda_flash_attn_ext_kvarn_direct_launch<4, 3, 8>(ctx, dst)) {
        ggml_cuda_flash_attn_ext_kvarn_stream_impl(ctx, dst);
    }
}

// exported entry points, defined once in template-instances/fattn-mma-kvarn-direct-instance.cu
bool ggml_cuda_flash_attn_ext_kvarn_direct_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_kvarn_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
