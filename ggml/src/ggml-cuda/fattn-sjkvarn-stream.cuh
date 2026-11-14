// SJ-KVaRN and native Turbo4 body attention, half-MMA streaming variant (n_q <= 8).
// Turbo4 stages native packed row blocks and decodes directly into MMA operands with a warp-register centroid table.
// Both codecs share the per-query softmax, exact ring, sink and split reduction; SJ-KVaRN-specific metadata is described below.
//
// The record layout matches the windowed kernel; this path keeps query and attention operands in fp16.
// Work split: Here the block is (query rows) x (KV splits) and each split owns its own shared-memory ring and its
// own cp.async pipeline, published by a named barrier per strip, so the splits never lock-step with each
// other. Query rows share the packed bytes and metadata without sharing their masks.
//
// The sealed record layout (ggml_sj_kvarn::code_bit) stores each (16-token strip, 16-channel tile) of a record as 32
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
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh" // fattn_sj_kvarn_u32_as_half2
#include "mma.cuh"
#include "fattn-sjkvarn-rot.cuh"

#include <atomic>
#include <cfloat>
#include <cinttypes>
#include <climits>
#include <cstdlib>
#include <cstring>

// [#195] decode tile of the stream kernel. An enum class, not a bool, so that no positional `true`/`false` meant for
// another trailing template flag can select the compact tile by accident.
enum class sj_kvarn_tile : int { rows = 0, q5g6 = 1 };

namespace fattn_sj_kvarn_stream {

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
static_assert(STAGE_BYTES == GGML_LSA_STRIP_HEAD_BYTES && RMETA_BYTES == GGML_QSA_META_HEAD_BYTES && RMETA_VCH == 1024,
              "LLAMA_LSA_RESIDENT strip slots and card meta hold the stage and RMETA byte images");

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
    r[0] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(x, 0x000F000FU, magic));
    r[1] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(x, 0x00F000F0U, magic));
    r[2] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(y, 0x000F000FU, magic));
    r[3] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(y, 0x00F000F0U, magic));
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
    int o;  // ordinal of the record in the walk (== g on the dense walk); picks the rmeta buffer
    int sm; // live strips of record g (bit sl), all ones unless GGML_SJKVARN_LIVE_STRIPS=1
};

// Opt-in record skip (GGML_SJKVARN_LIVE_RECORDS=1): the body walk visits only the records that some query row can see
// (any finite mask entry in the record's G positions). Flags: one warp per record. List: [0] = count, then the live
// record ids in order (one block, ordered by a ballot scan per 1024 records). Exact strips (sink, ring) are unchanged.
// Per-head mask (ne[2] == n_head, the q heads of a KV group share one mask): one flag row and one list per KV head
// (blockIdx.y / blockIdx.x), read from the group's first q head; head_pitch = halfs between KV heads' masks.
// Strip skip (strips != 0, GGML_SJKVARN_LIVE_STRIPS=1): the flag is the record's live-strip mask (bit sl = some row sees
// a finite entry in strip sl), and the walk skips the dead strips of a live record (no load, no MMA). A dead strip
// adds exactly zero (exp(-inf) = 0 and the running max is unchanged), so the output is the dense masked result.
// Without it the flag is all ones for a live record. List entries: g | flag << 24.
// LSA block list (ggml_flash_attn_ext_set_lsa_blocks): mask entry GGML_LSA_MASK_SEL at position p = visible iff bit
// p >> sel_sh of selb is set (blocks >= nbs hidden); any other entry keeps its value
static __device__ __forceinline__ float sj_lsa_mask_value(const half h, const int p, const uint32_t * selb, const int sel_sh,
        const int nbs) {
    const float m = __half2float(h);
    if (selb != nullptr && m == GGML_LSA_MASK_SEL) {
        const int b = p >> sel_sh;
        return b < nbs && ((selb[b >> 5] >> (b & 31)) & 1u) ? 0.0f : -INFINITY;
    }
    return m;
}

// LSA virtual keys (ggml_flash_attn_ext_set_lsa_vk), stored by sj_live_record_list at list - SJ_LIST_HDR: block mean
// K/V rows of the KV group (head z at + z*D), row pitch in halves; nd row t at nd + t*nd_pitch (< 0.5 picked, else 1 +
// the dropped count); n = blocks (0: off); alloc = mean rows readable from k/v (>= 16)
struct sj_lsa_vk {
    const half  * k;
    const half  * v;
    const float * nd;
    int   k_pitch;
    int   v_pitch;
    int   nd_pitch;
    int   n;
    int   alloc;
    float bias;
};
constexpr int SJ_LIST_HDR = 16; // ints before the lists: sj_lsa_vk, then list[-4 .. -1]
static_assert(sizeof(sj_lsa_vk) <= (SJ_LIST_HDR - 4)*sizeof(int32_t), "list header");

// sel F32 [nbs] -> bit words (vk: row 0 of the virtual-key plane, picked = entry < 0.5)
static __global__ void sj_lsa_sel_bits(const float * sel, const int nbs, uint32_t * bits, const int vk = 0) {
    const int w = blockIdx.x*blockDim.x + threadIdx.x;
    if (w*32 >= nbs) {
        return;
    }
    uint32_t x = 0;
    for (int i = 0; i < 32 && w*32 + i < nbs; ++i) {
        const float f = sel[w*32 + i];
        x |= ((vk ? f < 0.5f : f > 0.5f) ? 1u : 0u) << i;
    }
    bits[w] = x;
}

static __global__ void sj_live_record_flags(const half * mask, const int pitch, const int n_q, const int32_t * desc,
        int32_t * flags, const int nrec_max, const int64_t head_pitch, const int strips,
        const uint32_t * selb = nullptr, const int sel_sh = 0, const int sel_nbs = 0) {
    const int g = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    mask  += blockIdx.y*head_pitch;
    flags += (size_t) blockIdx.y*nrec_max;
    const int lane = threadIdx.x % WARP_SIZE;
    const int S = desc[GGML_SJKVARN_DESC_S], B = desc[GGML_SJKVARN_DESC_B], G = desc[GGML_SJKVARN_DESC_G];
    const int nrec = min(nrec_max, (B - S)/G);
    if (g >= nrec) {
        return;
    }
    const uint32_t full = (1u << (G/16)) - 1u;
    uint32_t sm = 0;
    for (int q = 0; q < n_q && sm != full && (strips || sm == 0); ++q) {
        for (int t0 = 0; t0 < G; t0 += WARP_SIZE) { // 32 positions = strips t0/16 and t0/16 + 1
            const int t = t0 + lane;
            const bool f = t < G && sj_lsa_mask_value(mask[(size_t) q*pitch + S + g*G + t], S + g*G + t, selb, sel_sh,
                                                        sel_nbs) != -INFINITY;
            const uint32_t bal = __ballot_sync(0xFFFFFFFF, f);
            sm |= ((bal & 0xFFFFu) ? 1u : 0u) << (t0/16);
            sm |= ((bal >> 16)     ? 1u : 0u) << (t0/16 + 1);
        }
        sm &= full;
    }
    if (lane == 0) {
        flags[g] = sm == 0 ? 0 : (strips ? (int32_t) sm : (int32_t) full);
    }
}

// header read by the decode kernel: list[-1] = stride between KV heads' lists; list[-2] = offset (ints, from list) of
// the LSA selection bits or 0; list[-3] = log2(LSA block); list[-4] = LSA block count
static __global__ void sj_live_record_list(const int32_t * desc, const int32_t * flags, int32_t * list, const int nrec_max,
        const int head_stride, const int sel_off = 0, const int sel_sh = 0, const int sel_nbs = 0, const sj_lsa_vk vk = {}) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *(sj_lsa_vk *) (list - SJ_LIST_HDR) = vk;
        list[-1] = head_stride; // read by the decode kernel: list of KV head z at list + z*head_stride
        list[-2] = sel_off;
        list[-3] = sel_sh;
        list[-4] = sel_nbs;
    }
    flags += (size_t) blockIdx.x*nrec_max;
    list  += (size_t) blockIdx.x*(nrec_max + 1);
    __shared__ int warp_counts[32];
    __shared__ int base;
    const int tid = threadIdx.x, lane = tid % WARP_SIZE, warp = tid / WARP_SIZE, nwarps = blockDim.x / WARP_SIZE;
    const int nrec = min(nrec_max, (desc[GGML_SJKVARN_DESC_B] - desc[GGML_SJKVARN_DESC_S])/desc[GGML_SJKVARN_DESC_G]);
    if (tid == 0) {
        base = 0;
    }
    __syncthreads();
    for (int g0 = 0; g0 < nrec; g0 += blockDim.x) {
        const int g = g0 + tid;
        const bool live = g < nrec && flags[g] != 0;
        const uint32_t bal = __ballot_sync(0xFFFFFFFF, live);
        if (lane == 0) {
            warp_counts[warp] = __popc(bal);
        }
        __syncthreads();
        int before = base;
        for (int w = 0; w < warp; ++w) {
            before += warp_counts[w];
        }
        if (live) {
            list[1 + before + __popc(bal & ((1u << lane) - 1u))] = (int32_t) ((uint32_t) g | ((uint32_t) flags[g] << 24));
        }
        __syncthreads();
        if (tid == 0) {
            for (int w = 0; w < nwarps; ++w) {
                base += warp_counts[w];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        list[0] = base;
    }
}

#ifdef SJKVARN_STREAM_DBG_TIME
static __device__ unsigned int fattn_sj_kvarn_stream_dbg_run = 0;
#endif

// lsa: opt-in LSA build (live-record walk, strip skip, per-head mask). The default instance (lsa = false) compiles that
// code out, so the default decode kernel is the pre-LSA code (registers/stack checked with cuobjdump).
// max_warps 6 (GGML_SJKVARN_NT6_OCC=1): the 6-row twin of the max_warps 5 body, held to 2 CTAs/SM (<= 168 regs)
// q5g6 [#195]: compact width-5 / GQA-6 tile. The 30 (query, head) pairs of a KV head fill 4 warps x 8 columns (pairs
// 30 and 31 pad) instead of 5 warps x 8 columns with 6 heads each. Every column keeps its own Q, mask row, online
// softmax and output; the KV split partition is the host's choice (forced to the five-warp one), so the result is
// bit-identical to the five-warp kernel while the block issues 4 instead of 5 MMA rows per strip.
template <int max_warps, int NSTAGE, bool turbo4_body = false, bool rot = false, bool lsa = false, sj_kvarn_tile tile = sj_kvarn_tile::rows>
__launch_bounds__(WARP_SIZE*max_warps, max_warps == 4 || max_warps == 5 || max_warps == 6 ? 2 : 1)
static __global__ void flash_attn_ext_sj_kvarn_stream(
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
#ifdef SJKVARN_STREAM_DBG_TIME
    const long long dbg_t0 = clock64();
    unsigned int dbg_smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(dbg_smid));
    const unsigned int dbg_run = atomicAdd(&fattn_sj_kvarn_stream_dbg_run, 0u);
    int dbg_nexact = 0, dbg_nrec = 0;
    long long dbg_wait = 0, dbg_body = 0, dbg_exact = 0, dbg_rec = 0, dbg_ts = 0;
#endif

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;
    const int n_q    = int(ne01.z);
    constexpr bool q5g6 = tile == sj_kvarn_tile::q5g6;
    static_assert(!q5g6 || max_warps == 4, "the compact Q5/G6 tile runs 4 warps");
    constexpr bool fixed_nt = max_warps == 5 || max_warps == 6 || q5g6; // MTP-4 verification (5), 6 rows (NT6_OCC), q5g6
    const int NT     = q5g6 ? 4 : (max_warps == 5 || max_warps == 6) ? max_warps : n_q;
    const int KW     = fixed_nt ? 1 : nwarps / NT;  // KV splits inside the block (host: nwarps == NT*KW, KW a power of two)
    const int nt     = fixed_nt ? warp : warp % NT;    // this warp's query row (q5g6: its column tile)
    const int kw     = fixed_nt ? 0 : warp / NT;    // this warp's KV split

    const int z_KV  = blockIdx.z;    // KV head (ne03 == 1)
    const int gqa   = ne02 / ne12;
    const int head0 = z_KV*gqa;
    const int hcol  = lane >> 2;     // the lane's head column (B/C fragment row)
    // column col of this warp -> query row, head in the group, and whether it is a real pair (q5g6: pair 8*nt + col of
    // the 5x6 grid, pad pairs read query row 0's mask and a zero Q and are never stored)
    auto col_query = [&](const int col) -> int { return q5g6 ? (8*nt + col < 30 ? (8*nt + col)/6 : 0) : nt; };
    auto col_head  = [&](const int col) -> int { return q5g6 ? (8*nt + col) % 6 : col; };
    auto col_valid = [&](const int col) -> bool { return q5g6 ? 8*nt + col < 30 : col < gqa; };
    const bool hvalid = col_valid(hcol);

    const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(sj_kvarn_body, sj_kvarn_desc, z_KV, D, 4, 4, nb12, nb22);
    const int n_kv_pad = ne11;
    const int G16 = kv.G/16;

    // work split: the body strips (records) form one contiguous range per (block, split); the exact strips (sink and
    // ring rows, ~1% of the positions but ~2x the cost of a record strip) go round-robin over all splits so that no
    // warp becomes the tail of the kernel
    const int nsplit = gridDim.y * KW;
    const int split  = blockIdx.y * KW + kw;
    // record skip: KV_max carries the live-record list; the walk is then over virtual strips 0 .. n_live*G/16
    // live[-1] = stride between the KV heads' lists (0: one list shared by all heads)
    const int * live = lsa && KV_max ? KV_max + z_KV*KV_max[-1] : nullptr;
    const int n_live = live ? live[0] : 0;
    // LSA block list: selection bits after the lists (header list[-2..-4], sj_live_record_list)
    const uint32_t * selb = nullptr;
    int sel_sh = 0, sel_nbs = 0;
    if constexpr (lsa) {
        if (live && KV_max[-2] > 0) {
            selb    = (const uint32_t *) (KV_max + KV_max[-2]);
            sel_sh  = KV_max[-3];
            sel_nbs = KV_max[-4];
        }
    }
    auto mask_at = [&](const half * m, const int p) -> float {
        if constexpr (lsa) {
            return sj_lsa_mask_value(m[p], p, selb, sel_sh, sel_nbs);
        } else {
            return __half2float(m[p]);
        }
    };
    const int sb0    = live ? 0 : kv.S/16, sb1 = live ? n_live*G16 : kv.B/16;
    const int nbody  = sb1 - sb0;
    const int s_lo   = sb0 + (int) (((int64_t) nbody *  split     ) / nsplit);
    const int s_hi   = sb0 + (int) (((int64_t) nbody * (split + 1)) / nsplit);
    const int n_ex_sink = kv.S/16;
    const int n_ex      = n_ex_sink + (n_kv_pad - kv.B)/16; // padded positions are masked

    const float * Qf = (const float *) (Q + (size_t) col_query(hcol)*nb01 + (size_t) (head0 + (hvalid ? col_head(hcol) : 0))*nb02);
    // [#139] rot build: Q arrives rotated (ggml_cuda_sj_kvarn_rot256_q_pass in the launcher); only the output is rotated here
    static_assert(!rot || !turbo4_body, "turbo4 bodies use the grouped signed basis");
    // mask row of the lane's two C columns 2(lane%4) and +1. q5g6: pair n = 8nt + 2(lane%4) is even, so n and n + 1 have
    // the same query row n/6 (n + 1 is odd, never a multiple of 6; pads 30 and 31 both read row 0): one row per lane.
    static_assert(30 % 6 == 0 && 6 % 2 == 0, "q5g6 column pairs must share a query row");
    // per-head mask (ne32 == n_head, LSA build, rows tile only): the KV group's first q head (the graph gives a group's
    // q heads one mask)
    static_assert(!lsa || tile == sj_kvarn_tile::rows, "the LSA build has no compact Q5/G6 tile");
    const char  * mrow = mask + (size_t) col_query(2*(lane & 3))*nb31 + (lsa && ne32 > 1 ? (size_t) head0*nb32 : 0);
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
            const half2 ks0 = fattn_sj_kvarn_u32_as_half2(ks.x), ks1 = fattn_sj_kvarn_u32_as_half2(ks.y);
            const float2 kz0 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.x));
            const float2 kz1 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.y));
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
#ifdef SJKVARN_DBG
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
                const float2 v = __half22float2(fattn_sj_kvarn_u32_as_half2(vo[c])); // Vch[lo channel], Vch[hi channel]
                r_lo = v.x; r_hi = 0.0625f*v.y;
            }
            if (vn != nullptr) {
                const float2 v = __half22float2(fattn_sj_kvarn_u32_as_half2(vn[c]));
                r_lo = __fdividef(r_lo, v.x); r_hi = __fdividef(r_hi, 0.0625f*v.y);
            }
#ifdef SJKVARN_DBG
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
    auto rec_of  = [&](const cursor & cu) { return fattn_sj_kvarn_rec(kv, cu.g); };
    auto cursor_init = [&](const int s) { // s = strip index of the walk
        cursor cu;
        if (live) {
            cu.o  = s / G16;
            cu.sl = s - cu.o*G16;
            const uint32_t e = cu.o < n_live ? (uint32_t) live[1 + cu.o] : 0u;
            cu.g  = (int) (e & 0xFFFFFFu);
            cu.sm = (int) (e >> 24);
            cu.p0 = kv.S + cu.g*kv.G + 16*cu.sl;
            return cu;
        }
        const int p0 = 16*s;
        cu.p0 = p0;
        cu.g  = 0;
        cu.sl = 0;
        cu.sm = 0xFF;
        if (is_body(p0)) {
            cu.g  = (p0 - kv.S) / kv.G;
            cu.sl = ((p0 - kv.S) - cu.g*kv.G) / 16;
        }
        cu.o = cu.g;
        return cu;
    };
    auto cursor_next = [&](cursor & cu) {
        if (live) {
            if (++cu.sl == G16) {
                cu.sl = 0;
                ++cu.o;
                if (cu.o < n_live) {
                    const uint32_t e = (uint32_t) live[1 + cu.o];
                    cu.g  = (int) (e & 0xFFFFFFu);
                    cu.sm = (int) (e >> 24);
                }
            }
            cu.p0 = kv.S + cu.g*kv.G + 16*cu.sl;
            return;
        }
        cu.p0 += 16;
        if (cu.p0 == kv.S) {
            cu.g = 0; cu.sl = 0; cu.o = 0;
        } else if (is_body(cu.p0)) {
            if (++cu.sl == G16) {
                cu.sl = 0;
                ++cu.g;
                ++cu.o;
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
        const char * ks   = rec + cu.sl*STRIP_BYTES;
        const uint32_t dst_s = ring_s + stage*STAGE_BYTES;
        // LLAMA_LSA_RESIDENT: record vectors from the card meta (same 1,536-byte image as RMETA), a resident strip from
        // its slot (same 4,192-byte image as the stage); a strip without a slot reads the pinned master
        const char * meta_card = nullptr;
        const char * slot_card = nullptr;
        if (kv.strip_map != nullptr) {
            const ggml_cuda_qsa_clock_map & smap = *kv.strip_map;
            const int record = kv.table ? kv.table[cu.g] : cu.g;
            meta_card = smap.meta + ((size_t) record*smap.heads + kv.resident_head)*GGML_QSA_META_HEAD_BYTES;
            const int slot = smap.table[record*(128/GGML_LSA_STRIP_CELLS) + cu.sl];
            if (slot >= 0) {
                slot_card = smap.card + ((size_t) slot*smap.heads + kv.resident_head)*GGML_LSA_STRIP_HEAD_BYTES;
            }
        }
        if (with_meta) {
            const uint32_t dst_m = rmeta_s + ((lsa ? cu.o : cu.g) & 1)*RMETA_BYTES;
            for (int q = nt*WARP_SIZE + lane; q < RMETA_CHUNKS; q += NT*WARP_SIZE) {
                const char * src = meta_card ? meta_card + 16*q :
                    q < RMETA_VCH/16 ? rec + kv.k_scale + 16*q : rec + kv.v_ch + 16*(q - RMETA_VCH/16);
                cp_async_cg_16<0>(dst_m + 16*q, src);
            }
        }
        if (lsa && cu.sl < 8 && !((cu.sm >> cu.sl) & 1)) { // dead strip (strip skip): record vectors only, empty group keeps the count
            cp_async_commit();
            return;
        }
        for (int q = nt*WARP_SIZE + lane; q < STAGE_CHUNKS; q += NT*WARP_SIZE) {
            const char * src;
            if (slot_card != nullptr) {
                src = slot_card + 16*q;
            } else if (q < STRIP_BYTES/16) {
                src = ks + 16*q;
            } else if (q < 2*STRIP_BYTES/16) {
                src = ks + kv.v_payload + 16*(q - STRIP_BYTES/16);
            } else {
                const int m = q - 2*STRIP_BYTES/16; // ktok, vscale, vzero (two chunks each)
                const int half_off = 16*(m & 1);
                src = m < 2 ? rec + kv.k_tok   + cu.sl*32 + half_off
                    : m < 4 ? rec + kv.v_scale + cu.sl*32 + half_off
                    :         rec + kv.v_zero  + cu.sl*32 + half_off;
            }
            cp_async_cg_16<0>(dst_s + 16*q, src);
        }
        cp_async_commit();
    };

    // stage one exact f16 strip (16 rows of D halves, K or V) into the ring, rows swizzled by 16-byte chunk
    static_assert((EXACT_BYTES/16) % WARP_SIZE == 0, "fattn_sj_kvarn_ring_pair_warp needs whole warps per exact strip");
    auto issue_exact = [&](const char * rows, const size_t nb, const int type) {
        // per-strip load keeps the tq6 lane register out of the kernel's live range (#127)
        const float tq6_mag = type == GGML_TYPE_TQ6_0 ? TQ6_CENTROIDS[32 + lane] : 0.0f;
        for (int q = nt*WARP_SIZE + lane; q < EXACT_BYTES/16; q += NT*WARP_SIZE) {
            const int r = q / (EXACT_ROW_BYTES/16);
            const int j = q % (EXACT_ROW_BYTES/16);
            if (type != GGML_TYPE_F16) {
                half2 * out = (half2 *) (ring + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)));
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    out[i] = fattn_sj_kvarn_ring_pair_warp(rows + (size_t) r*nb, 4*j+i, type, tq6_mag); // warp-uniform: q steps by whole warps
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

    if constexpr (turbo4_body) {
        setup_exact();
    }

    cursor ci = cursor_init(s_lo);      // issue cursor, NSTAGE-1 strips ahead
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
#ifdef SJKVARN_STREAM_DBG_TIME
        dbg_ts = clock64();
#endif
        cp_async_wait_group<NSTAGE - 1>(); // strip s has landed (this warp's chunks) ...
        sync_split();                      // ... and the other warps'
#ifdef SJKVARN_STREAM_DBG_TIME
        { const long long t = clock64(); dbg_wait += t - dbg_ts; dbg_ts = t; }
#endif

        const int    id = cc.g;
        const char * rm = rmeta + ((lsa ? cc.o : cc.g) & 1)*RMETA_BYTES;
#ifdef SJKVARN_STREAM_DBG_TIME
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
#ifdef SJKVARN_STREAM_DBG_TIME
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
            const float mk0 = mask_at((const half *) mrow, cc.p0 + (lane >> 2));
            const float mk1 = mask_at((const half *) mrow, cc.p0 + (lane >> 2) + 8);
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
        } else if (!lsa || cc.sl >= 8 || ((cc.sm >> cc.sl) & 1)) { // dead strips (strip skip) add nothing: skipped; G > 128: strips 8+ always walked
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
            const half * vsh  = (const half *) (st + META_VS);
            const half * vzh  = (const half *) (st + META_VZ);
            const float tok0 = __half2float(ktok[lane >> 2]);
            const float tok1 = __half2float(ktok[(lane >> 2) + 8]);
            const float mk0  = mask_at((const half *) mrow, cc.p0 + (lane >> 2));
            const float mk1  = mask_at((const half *) mrow, cc.p0 + (lane >> 2) + 8);
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
#ifdef SJKVARN_STREAM_DBG_TIME
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
    const half * mrow_h = (const half *) mrow;

    // KQ of the exact strip staged in the ring (fragment order, K straight) + mask, online softmax -> P
    // mk0/mk1: additive term of token rows lane/4 and lane/4 + 8
    auto kq_add = [&](const float mk0, const float mk1) -> T_B {
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
        kq.x[0] += mk0;
        kq.x[1] += mk0;
        kq.x[2] += mk1;
        kq.x[3] += mk1;
        return softmax(kq);
    };
    auto exact_kq = [&](const int p0) -> T_B {
        return kq_add(mask_at(mrow_h, p0 + (lane >> 2)), mask_at(mrow_h, p0 + (lane >> 2) + 8));
    };
    // acc += V^T P from the V strip staged in the ring (8x8 transposes)
    auto exact_pv = [&](const T_B & P) {
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            T_A a;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int row = 8*(l >> 1) + (lane >> 2);      // token row of the 8x8 block
                const int cp  = 8*c + 4*(l & 1) + (lane & 3);  // channel pair
                a.x[l] = ggml_cuda_movmatrix(fattn_sj_kvarn_u32_as_half2(exact_word(row, cp)));
            }
            mma(acc[c], a, P);
        }
    };
    auto exact_p0 = [&](const int e) { return e < n_ex_sink ? 16*e : kv.B + 16*(e - n_ex_sink); };
    auto exact_row0 = [&](const int p0) { return p0 < kv.S ? p0 : kv.S + (p0 - kv.S) % kv.cap; };

    // Pipelined exact strips (the packed ring, and the sink when it shares the ring type): the raw rows of the
    // split's strips go through a queue of items K(e), V(e), K(e + nsplit), ... staged by cp.async into nraw raw
    // slots behind the f16 strip buffer, nraw items ahead of the decode. Each item is decoded from shared memory into
    // the same swizzled f16 strip as before and consumed in the same order, so the result is bit-identical.
    // The separate f16 sink (8 KB per item) keeps the direct cp.async path below.
    auto raw_row_bytes = [](const int type) -> int {
        return type == GGML_TYPE_TQ6_0 ? (D/QK_TQ6)*(int) sizeof(block_tq6_0)
             : type == GGML_TYPE_Q8_0  ? (D/QK8_0)*(int) sizeof(block_q8_0)
             : type == GGML_TYPE_F16   ? D*(int) sizeof(half) : 0;
    };
    constexpr int raw_avail = (turbo4_body ? NSTAGE*TURBO4_STAGE_BYTES : NSTAGE*STAGE_BYTES + 2*RMETA_BYTES) - EXACT_BYTES; // split_bytes()
    constexpr int RAW_MAX = 6;
    const int rb_k = raw_row_bytes(kv.type_k);
    const int rb_v = raw_row_bytes(kv.type_v);
    const int raw_slot = 16*max(rb_k, rb_v);
    const bool raw_ok = rb_k > 0 && rb_v > 0 && raw_slot <= raw_avail &&
        (((uintptr_t) Kh | (uintptr_t) Vh | (uintptr_t) nb11 | (uintptr_t) nb21 | (uintptr_t) rb_k | (uintptr_t) rb_v) & 3) == 0;
    // the MTP-4 verification kernel (max_warps 5) sits at its register cap; the queue's state costs its body loop
    // more than the pipeline saves (measured +30-35 us/call at 102K), so it keeps the direct path
    constexpr bool raw_pipe = max_warps != 5 && max_warps != 6; // q5g6 (4 warps) keeps the pipeline, as before
    const int nraw = raw_pipe && raw_ok ? min(RAW_MAX, raw_avail/raw_slot) : 0;
    int e_pipe = n_ex; // first strip of the pipelined range (n_ex: none)
    if (nraw > 0) {
        e_pipe = split;
        if (kv.sink_type == GGML_TYPE_F16 && split < n_ex_sink) {
            e_pipe = split + ((n_ex_sink - split + nsplit - 1)/nsplit)*nsplit;
        }
    }

    // direct path: the f16 sink, and every exact strip when the raw queue does not fit
    for (int e = split; e < min(e_pipe, n_ex); e += nsplit) {
        const int p0 = exact_p0(e);
#ifdef SJKVARN_STREAM_DBG_TIME
        dbg_nexact += 1;
        dbg_ts = clock64();
#endif
        {
            // exact f16 rows, staged K then V through the ring, read in fragment order: K straight, V via an 8x8
            // transpose
            const int row0 = exact_row0(p0);
            const bool f16_sink = p0 < kv.S && kv.sink_type == GGML_TYPE_F16;
            const char * Kr = f16_sink ? fattn_sj_kvarn_sink_row(Kh, kv, nb11, row0, false) : Kh + (size_t) row0*nb11;
            const char * Vr = f16_sink ? fattn_sj_kvarn_sink_row(Vh, kv, nb21, row0, true) : Vh + (size_t) row0*nb21;
            issue_exact(Kr, f16_sink ? kv.sink_stride : nb11, f16_sink ? GGML_TYPE_F16 : kv.type_k);
            cp_async_wait_all();
            sync_split();
            const T_B P = exact_kq(p0);

            sync_split(); // everyone has read K
            issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v);
            cp_async_wait_all();
            sync_split();
            exact_pv(P);
            sync_split(); // everyone has read V before the next strip (or the combine) reuses the ring
        }
#ifdef SJKVARN_STREAM_DBG_TIME
        { const long long t = clock64(); dbg_exact += t - dbg_ts; }
#endif
    }

    if (e_pipe < n_ex) {
        const uint32_t raw_s = ring_s + EXACT_BYTES;
        const char *   raw   = ring + EXACT_BYTES;
        // stage item it (raw rows, 4-byte cp.async: tq6 rows are only 4-byte aligned) into slot it % nraw
        auto issue_raw = [&](const int it) {
            const int e = e_pipe + (it >> 1)*nsplit;
            if (e < n_ex) {
                const bool   is_v = it & 1;
                const size_t nb   = is_v ? nb21 : nb11;
                const char * src  = (is_v ? Vh : Kh) + (size_t) exact_row0(exact_p0(e))*nb;
                const int    wpr  = (is_v ? rb_v : rb_k)/4;
                const uint32_t dst = raw_s + (it % nraw)*raw_slot;
                const int step = NT*WARP_SIZE;
                int r = (nt*WARP_SIZE + lane) / wpr;
                int c = (nt*WARP_SIZE + lane) - r*wpr;
                while (r < 16) {
                    cp_async_ca_4(dst + 4*(r*wpr + c), src + (size_t) r*nb + 4*c);
                    c += step;
                    while (c >= wpr) {
                        c -= wpr;
                        ++r;
                    }
                }
            }
            cp_async_commit();
        };
        // wait for item it (nraw - 1 younger groups may stay in flight), then decode it into the f16 strip
        auto land_raw = [&]() {
            switch (nraw) {
                case 1:  cp_async_wait_group<0>(); break;
                case 2:  cp_async_wait_group<1>(); break;
                case 3:  cp_async_wait_group<2>(); break;
                case 4:  cp_async_wait_group<3>(); break;
                case 5:  cp_async_wait_group<4>(); break;
                default: cp_async_wait_group<5>(); break;
            }
        };
        auto decode_raw = [&](const int it, const int type, const int rb) {
            const char * rows = raw + (it % nraw)*raw_slot;
            const float tq6_mag = type == GGML_TYPE_TQ6_0 ? TQ6_CENTROIDS[32 + lane] : 0.0f;
            for (int q = nt*WARP_SIZE + lane; q < EXACT_BYTES/16; q += NT*WARP_SIZE) {
                const int r = q / (EXACT_ROW_BYTES/16);
                const int j = q % (EXACT_ROW_BYTES/16);
                half2 * out = (half2 *) (ring + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)));
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    out[i] = fattn_sj_kvarn_ring_pair_warp(rows + r*rb, 4*j+i, type, tq6_mag); // warp-uniform
                }
            }
        };

        for (int it = 0; it < nraw; ++it) {
            issue_raw(it);
        }
        int it = 0;
        for (int e = e_pipe; e < n_ex; e += nsplit, it += 2) {
            const int p0 = exact_p0(e);
#ifdef SJKVARN_STREAM_DBG_TIME
            dbg_nexact += 1;
            dbg_ts = clock64();
#endif
            land_raw();
            sync_split();                 // K raw landed; everyone is done with the previous V strip
            decode_raw(it, kv.type_k, rb_k);
            sync_split();                 // f16 K strip complete; its raw slot is free
            issue_raw(it + nraw);
            const T_B P = exact_kq(p0);

            land_raw();
            sync_split();                 // V raw landed; everyone has read K
            decode_raw(it + 1, kv.type_v, rb_v);
            sync_split();
            issue_raw(it + 1 + nraw);
            exact_pv(P);
#ifdef SJKVARN_STREAM_DBG_TIME
            { const long long t = clock64(); dbg_exact += t - dbg_ts; }
#endif
        }
        cp_async_wait_all();
        sync_split(); // everyone has read V before the combine reuses the ring
    }

    // LSA virtual keys (ggml_flash_attn_ext_set_lsa_vk): the dropped blocks' mean K/V rows, 16 blocks per strip, go
    // through the exact-strip path (f16 rows, same staging, KQ, softmax and PV) with log(n dropped) + bias as the
    // additive term instead of the mask (-inf: picked block, nothing dropped, or past the block count). Strips continue
    // the exact round-robin after strip n_ex - 1, so the extra work lands on the splits with the fewest exact strips.
    if constexpr (lsa) {
        const sj_lsa_vk * vkp = live && KV_max[-2] > 0 ? (const sj_lsa_vk *) (KV_max - SJ_LIST_HDR) : nullptr;
        const int vk_n = vkp ? vkp->n : 0;
        if (vk_n > 0) {
            const int nvs = (vk_n + 15)/16;
            int v = (split - n_ex) % nsplit;
            v += v < 0 ? nsplit : 0;
            if (v < nvs && split >= n_ex) {
                setup_exact();
            }
            const float * ndr = vkp->nd + (size_t) (nt < n_q ? nt : 0)*vkp->nd_pitch;
            for (; v < nvs; v += nsplit) {
                const int b0 = min(16*v, vkp->alloc - 16); // a short last strip re-reads earlier rows (masked below)
                issue_exact((const char *) (vkp->k + (size_t) b0*vkp->k_pitch + (size_t) z_KV*D), 2*(size_t) vkp->k_pitch,
                            GGML_TYPE_F16);
                cp_async_wait_all();
                sync_split();
                auto vk_term = [&](const int b) -> float {
                    const float n = b >= 16*v && b < vk_n && nt < n_q ? ndr[b] - 1.0f : 0.0f; // dropped cells
                    return n >= 0.5f ? logf(n) + vkp->bias : -INFINITY;
                };
                const T_B P = kq_add(vk_term(b0 + (lane >> 2)), vk_term(b0 + (lane >> 2) + 8));
                sync_split(); // everyone has read K
                issue_exact((const char *) (vkp->v + (size_t) b0*vkp->v_pitch + (size_t) z_KV*D), 2*(size_t) vkp->v_pitch,
                            GGML_TYPE_F16);
                cp_async_wait_all();
                sync_split();
                exact_pv(P);
                sync_split(); // everyone has read V
            }
        }
    }
#ifdef SJKVARN_STREAM_DBG_TIME
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
            sj_kvarn_rot256_frag<T_C, NTC>(acc, lane);
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
            if (!col_valid(col)) {
                continue;
            }
            const int head = head0 + col_head(col);
            const float inv = single ? 1.0f/kq_sum[e] : 1.0f;
            float * out = dst + ((size_t) (col_query(col)*ne02 + head)*gridDim.y + blockIdx.y)*D;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const int d = 16*c + (lane >> 2) + 8*(l >> 1);
                out[d] = acc[c].x[l + e]*inv;
            }
        }
    }
#ifdef SJKVARN_DBG
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && lane == 0) {
        printf("[dbg] out single=%d kqmax=%g kqsum=%g acc0=%g acc2=%g out0=%g out8=%g\n", (int) single, kq_max[0], kq_sum[0],
               acc[0].x[0], acc[0].x[2], acc[0].x[0]/kq_sum[0], acc[0].x[2]/kq_sum[0]);
    }
#endif
#ifdef SJKVARN_STREAM_DBG_TIME
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && warp == 0 && lane == 0) {
        __threadfence();
        atomicAdd(&fattn_sj_kvarn_stream_dbg_run, 1u);
    }
#endif
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
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
}

} // namespace fattn_sj_kvarn_stream

// Each query warp shares a packed-data ring with the other query warps in its KV split.
template <int max_warps, int NSTAGE, bool turbo4_body = false, bool rot = false, bool lsa = false, sj_kvarn_tile tile = sj_kvarn_tile::rows>
static fattn_kernel_t ggml_cuda_fattn_sj_kvarn_stream_kernel() {
    using namespace fattn_sj_kvarn_stream;
    fattn_kernel_t kernel = flash_attn_ext_sj_kvarn_stream<max_warps, NSTAGE, turbo4_body, rot, lsa, tile>;
    const int id = ggml_cuda_get_device();
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {false};
    if (!initialized[id]) {
        const size_t limit = ggml_cuda_info().devices[id].smpbo;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, (int) limit));
        initialized[id] = true;
    }
    return kernel;
}

static size_t ggml_cuda_fattn_sj_kvarn_stream_shared(const int NT, const int KW, const int nstage, const bool turbo4_body = false) {
    using namespace fattn_sj_kvarn_stream;
    return std::max(ring_bytes(KW, nstage, turbo4_body), (size_t) NT*(KW/2)*SLOT_BYTES);
}

template <int max_warps, int NSTAGE, bool turbo4_body = false, bool rot = false, sj_kvarn_tile tile = sj_kvarn_tile::rows>
static int ggml_cuda_fattn_sj_kvarn_stream_occupancy(const int NT, const int KW) {
    const size_t shared = ggml_cuda_fattn_sj_kvarn_stream_shared(NT, KW, NSTAGE, turbo4_body);
    if (shared > ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo) {
        return 0;
    }
    int blocks = 0;
    // #139: always the plain build, so the rot build picks the same stage count and KV split
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
        ggml_cuda_fattn_sj_kvarn_stream_kernel<max_warps, NSTAGE, turbo4_body, false, false, tile>(), WARP_SIZE*NT*KW, shared));
    return blocks;
}

static bool ggml_cuda_fattn_sj_nt6_logged = false; // GGML_SJKVARN_NT6_OCC log, one line per process

template <int max_warps, int NSTAGE, bool turbo4_body = false, bool rot = false, sj_kvarn_tile tile = sj_kvarn_tile::rows>
static void ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int NT, const int KW, const int parallel_blocks) {
    using namespace fattn_sj_kvarn_stream;
    // the LSA instance only when the opt-in live-record list is set (never for turbo4 bodies)
    fattn_kernel_t kernel = ggml_cuda_fattn_sj_kvarn_stream_kernel<max_warps, NSTAGE, turbo4_body, rot, false, tile>();
    if constexpr (!turbo4_body) {
        if (ggml_cuda_fattn_sj_live_records != nullptr || (dst->src[3] && dst->src[3]->ne[2] > 1)) {
            // the compact Q5/G6 tile has no LSA build: ggml_cuda_fattn_sj_kvarn_q5g6_mode_for refuses it before the launch
            GGML_ASSERT(tile == sj_kvarn_tile::rows);
            kernel = ggml_cuda_fattn_sj_kvarn_stream_kernel<max_warps, NSTAGE, turbo4_body, rot, true>();
        }
    }
    if constexpr (max_warps == 6) {
        // GGML_SJKVARN_NT6_OCC: one line on the first launch (any instance), so a run can confirm it is engaged
        if (!ggml_cuda_fattn_sj_nt6_logged) {
            ggml_cuda_fattn_sj_nt6_logged = true;
            cudaFuncAttributes attr;
            CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void *>(kernel)));
            int ctas = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kernel, WARP_SIZE*NT*KW,
                ggml_cuda_fattn_sj_kvarn_stream_shared(NT, KW, NSTAGE, turbo4_body)));
            GGML_LOG_INFO("SJKVARN NT6_OCC: 6-row instance at %d CTA/SM (regs %d, local %zu B, nstage %d)\n",
                ctas, attr.numRegs, attr.localSizeBytes, NSTAGE);
        }
    }
    launch_fattn<D, NCOLS2, NCOLS2>
        (ctx, dst, kernel, NT*KW,
         ggml_cuda_fattn_sj_kvarn_stream_shared(NT, KW, NSTAGE, turbo4_body), UNIT,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false, /*use_sparse=*/false, WARP_SIZE, nullptr, nullptr, parallel_blocks,
         /*async_kv_preload=*/false, /*sj_fill_kv_splits=*/turbo4_body ? 0 : KW);
}

// Ring depth for a launch: a requested depth if it can launch, else the occupancy choice cached per device and width.
template <int max_warps, bool turbo4_body = false, bool rot = false, sj_kvarn_tile tile = sj_kvarn_tile::rows>
static int ggml_cuda_fattn_sj_kvarn_stream_nstage(const int NT, const int KW, int nstage) {
    if (nstage != 0) {
        const int blocks = nstage >= 6 ? ggml_cuda_fattn_sj_kvarn_stream_occupancy<max_warps, 6, turbo4_body, rot, tile>(NT, KW)
                         : nstage >= 4 ? ggml_cuda_fattn_sj_kvarn_stream_occupancy<max_warps, 4, turbo4_body, rot, tile>(NT, KW)
                                       : ggml_cuda_fattn_sj_kvarn_stream_occupancy<max_warps, 2, turbo4_body, rot, tile>(NT, KW);
        if (blocks == 0) {
            nstage = 0;
        }
    }
    if (nstage == 0) {
        const int id = ggml_cuda_get_device();
        static int stages[GGML_CUDA_MAX_DEVICES][9] = {};
        int & cached = stages[id][NT];
        if (cached == 0) {
            const int blocks2 = ggml_cuda_fattn_sj_kvarn_stream_occupancy<max_warps, 2, turbo4_body, rot, tile>(NT, KW);
            const int blocks4 = ggml_cuda_fattn_sj_kvarn_stream_occupancy<max_warps, 4, turbo4_body, rot, tile>(NT, KW);
            const int blocks6 = ggml_cuda_fattn_sj_kvarn_stream_occupancy<max_warps, 6, turbo4_body, rot, tile>(NT, KW);
            GGML_ASSERT(std::max(blocks2, std::max(blocks4, blocks6)) > 0);
            // Preserve resident warps first, then allow more strips in flight.
            cached = blocks6 >= blocks4 && blocks6 >= blocks2 ? 6 : blocks4 >= blocks2 ? 4 : 2;
        }
        nstage = cached;
    }
    return nstage >= 6 ? 6 : nstage >= 4 ? 4 : 2;
}

template <int max_warps, bool turbo4_body = false, bool rot = false, sj_kvarn_tile tile = sj_kvarn_tile::rows>
static void ggml_cuda_fattn_sj_kvarn_stream_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int NT, const int KW, int nstage, const int parallel_blocks) {
    nstage = ggml_cuda_fattn_sj_kvarn_stream_nstage<max_warps, turbo4_body, rot, tile>(NT, KW, nstage);
    if (nstage >= 6) {
        ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch<max_warps, 6, turbo4_body, rot, tile>(ctx, dst, NT, KW, parallel_blocks);
    } else if (nstage >= 4) {
        ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch<max_warps, 4, turbo4_body, rot, tile>(ctx, dst, NT, KW, parallel_blocks);
    } else {
        ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch<max_warps, 2, turbo4_body, rot, tile>(ctx, dst, NT, KW, parallel_blocks);
    }
}

// [#195] The KV split count launch_fattn derives for a kernel with max_blocks_per_sm resident blocks (its non-stream-k
// branch: ntiles_x = 1 and one GQA tile per KV head at n_q <= 8, GQA <= 8). The compact Q5/G6 launch passes the value of
// the five-warp kernel so both walk the same KV ranges and combine the same parts.
static int ggml_cuda_fattn_sj_kvarn_stream_parallel_blocks(const ggml_tensor * dst, const int max_blocks_per_sm) {
    const int nsm        = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int n_kv       = ggml_get_op_params_i32(dst, 6);
    const int ntiles_KV  = (n_kv + fattn_sj_kvarn_stream::UNIT - 1) / fattn_sj_kvarn_stream::UNIT;
    const int ntiles_dst = (int) (dst->src[1]->ne[2] * dst->src[0]->ne[3]);
    int parallel_blocks = std::min(max_blocks_per_sm, ntiles_KV);
    const int blocks_per_wave = nsm * max_blocks_per_sm;
    int nwaves_best = 0;
    int efficiency_percent_best = 0;
    for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KV; ++parallel_blocks_test) {
        const int nblocks_total = ntiles_dst * parallel_blocks_test;
        const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);
        if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
            break;
        }
        if (efficiency_percent > efficiency_percent_best) {
            nwaves_best = nwaves;
            efficiency_percent_best = efficiency_percent;
            parallel_blocks = parallel_blocks_test;
        }
    }
    return parallel_blocks;
}

// [#195] GGML_SJKVARN_DIRECT_Q5G6: unset = 1 (default on, Jake 2026-11-14); 0 = off (kill switch); 1 = compact Q5/G6 tile
// on the five-query path with the five-warp kernel's KV split (bit-identical output); 2 = compact tile with its own
// occupancy split (diagnostic, not bit-identical). Unset with GGML_CUDA_SJ_FILL=1 = off (mode 1 is not bit-identical there).
// The refusal warnings below are printed only when the variable is set explicitly; the default stays quiet.
// Covered: the 4/4 stream decode path at width 5 and GQA 6 (scalar and turbo4 bodies, plain and fused rotation). Every
// other cache type or shape runs its usual kernel; the refusals are logged once (fattn.cu logs the non-stream routes).
static int ggml_cuda_fattn_sj_kvarn_q5g6_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_SJKVARN_DIRECT_Q5G6");
        if (e == nullptr) {
            const char * fill = getenv("GGML_CUDA_SJ_FILL");
            return fill != nullptr && strcmp(fill, "1") == 0 ? 0 : 1;
        }
        char * end = nullptr;
        const long m = strtol(e, &end, 10);
        if (*e == '\0' || *end != '\0' || (m != 0 && m != 1 && m != 2)) {
            GGML_LOG_WARN("ggml_cuda_fattn_sj_kvarn_q5g6_mode: GGML_SJKVARN_DIRECT_Q5G6=\"%s\" is not 0, 1 or 2: compact Q5/G6 tile OFF\n", e);
            return 0;
        }
        // GGML_CUDA_SJ_FILL=1 changes the five-warp KV split inside launch_fattn; mode 1 copies only the default split
        const char * fill = getenv("GGML_CUDA_SJ_FILL");
        if (m == 1 && fill != nullptr && strcmp(fill, "1") == 0) {
            GGML_LOG_WARN("ggml_cuda_fattn_sj_kvarn_q5g6_mode: GGML_SJKVARN_DIRECT_Q5G6=1 is not bit-identical under GGML_CUDA_SJ_FILL=1: compact Q5/G6 tile OFF\n");
            return 0;
        }
        return (int) m;
    }();
    return mode;
}

// true when GGML_SJKVARN_DIRECT_Q5G6 is set explicitly to a non-zero value (refusal warnings are for that case only)
static bool ggml_cuda_fattn_sj_kvarn_q5g6_explicit() {
    static const bool v = [] { const char * e = getenv("GGML_SJKVARN_DIRECT_Q5G6"); return e != nullptr && *e != '\0' && strcmp(e, "0") != 0; }();
    return v;
}

// one warning per process when the switch is set but a width-5 node takes a route without the compact tile (shared by
// the shape reasons); a caller may pass its own flag so its reason is logged even after another one (LSA refusal)
static void ggml_cuda_fattn_sj_kvarn_q5g6_off(const char * why, std::atomic_flag * own = nullptr) {
    static std::atomic_flag warned = ATOMIC_FLAG_INIT;
    if (ggml_cuda_fattn_sj_kvarn_q5g6_explicit() && ggml_cuda_fattn_sj_kvarn_q5g6_mode() != 0 && !(own ? own : &warned)->test_and_set()) {
        GGML_LOG_WARN("ggml_cuda_fattn_sj_kvarn_q5g6: GGML_SJKVARN_DIRECT_Q5G6 is set but a width-5 node runs %s: compact Q5/G6 tile OFF for it "
                      "(first such node; later ones are not logged)\n", why);
    }
}

// mode for this node: 0 (with one warning per process) when the shape is not the covered width-5 / GQA-6 one
static int ggml_cuda_fattn_sj_kvarn_q5g6_mode_for(const ggml_tensor * dst) {
    const int mode = ggml_cuda_fattn_sj_kvarn_q5g6_mode();
    if (mode == 0) {
        return 0;
    }
    // per-head masks (mask ne[2] > 1) are not mapped per column by the compact tile
    if (dst->src[3] != nullptr && dst->src[3]->ne[2] > 1) {
        ggml_cuda_fattn_sj_kvarn_q5g6_off("with a per-head mask");
        return 0;
    }
    // LSA live-record walk (GGML_SJKVARN_TILE_SKIP / LIVE_RECORDS / LIVE_STRIPS, block list, resident strips): only the
    // rows tile has the LSA build, and the compact launcher would ignore the record list
    if (ggml_cuda_fattn_sj_live_records != nullptr) {
        static std::atomic_flag warned_lsa = ATOMIC_FLAG_INIT;
        ggml_cuda_fattn_sj_kvarn_q5g6_off("with the LSA live-record walk", &warned_lsa);
        return 0;
    }
    const int64_t gqa = dst->src[0]->ne[2] / dst->src[1]->ne[2];
    if (dst->src[0]->ne[2] != 6*dst->src[1]->ne[2]) {
        static std::atomic_flag warned = ATOMIC_FLAG_INIT;
        if (ggml_cuda_fattn_sj_kvarn_q5g6_explicit() && !warned.test_and_set()) {
            GGML_LOG_WARN("%s: GGML_SJKVARN_DIRECT_Q5G6=%d covers GQA 6 only, this model has GQA %" PRId64 ": compact Q5/G6 tile OFF\n",
                "ggml_cuda_fattn_sj_kvarn_q5g6_mode_for", mode, gqa);
        }
        return 0;
    }
    return mode;
}

template <bool turbo4_body, bool rot>
static void ggml_cuda_fattn_sj_kvarn_stream_q5g6(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int nstage_env, const int mode) {
    GGML_ASSERT(dst->src[0]->ne[1] == 5 && dst->src[0]->ne[2] == 6*dst->src[1]->ne[2]); // the tile hard-codes 5 x 6 pairs
    int parallel_blocks = 0; // mode 2: launch_fattn's own choice for the 4-warp kernel
    if (mode == 1) {
        // the five-warp launch's occupancy (with the ring depth it would pick) fed to launch_fattn's split rule; the
        // compact launch then picks its own ring depth, which does not change the arithmetic
        const int nstage5 = ggml_cuda_fattn_sj_kvarn_stream_nstage<5, turbo4_body, rot>(5, 1, nstage_env);
        const size_t shared5 = ggml_cuda_fattn_sj_kvarn_stream_shared(5, 1, nstage5, turbo4_body);
        const fattn_kernel_t k5 = nstage5 >= 6 ? ggml_cuda_fattn_sj_kvarn_stream_kernel<5, 6, turbo4_body, rot>()
                                : nstage5 >= 4 ? ggml_cuda_fattn_sj_kvarn_stream_kernel<5, 4, turbo4_body, rot>()
                                               : ggml_cuda_fattn_sj_kvarn_stream_kernel<5, 2, turbo4_body, rot>();
        int blocks5 = 1;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks5, k5, WARP_SIZE*5, shared5));
        GGML_ASSERT(blocks5 > 0);
        parallel_blocks = ggml_cuda_fattn_sj_kvarn_stream_parallel_blocks(dst, blocks5);
    }
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        GGML_LOG_INFO("%s: SJ-KVaRN compact Q5/G6 decode tile on (mode %d, parallel_blocks %d on the first node)\n", __func__, mode, parallel_blocks);
    }
    ggml_cuda_fattn_sj_kvarn_stream_select<4, turbo4_body, rot, sj_kvarn_tile::q5g6>(ctx, dst, 4, 1, nstage_env, parallel_blocks);
}

template <bool turbo4_body = false, bool rot = false>
inline void ggml_cuda_flash_attn_ext_sj_kvarn_stream_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // opt-in record skip (LSA on SJ-KVaRN sets it): build the live-record list from the mask, then walk only those
    // GGML_SJKVARN_TILE_SKIP=1 (the low-bit tile kernel's skip of fully masked body tiles, the switch LSA uses on 3/2t
    // and 3/3t) takes the same exact skip here: the live-strip walk (dead records and dead 16-cell strips not read)
    static const bool tile_skip_env = getenv("GGML_SJKVARN_TILE_SKIP") != nullptr && atoi(getenv("GGML_SJKVARN_TILE_SKIP")) != 0;
    static const bool strip_env = tile_skip_env ||
        (getenv("GGML_SJKVARN_LIVE_STRIPS") != nullptr && atoi(getenv("GGML_SJKVARN_LIVE_STRIPS")) != 0);
    static const bool live_env  = strip_env ||
        (getenv("GGML_SJKVARN_LIVE_RECORDS") != nullptr && atoi(getenv("GGML_SJKVARN_LIVE_RECORDS")) != 0);
    // LSA block list (ggml_flash_attn_ext_set_lsa_blocks): the mask's GGML_LSA_MASK_SEL entries resolve through the
    // selection bits, so the node always takes the live-record walk (dead records skipped, mapped mask everywhere)
    const int32_t lsa_blk = ggml_flash_attn_ext_get_lsa_block(dst);
    GGML_ASSERT(lsa_blk == 0 || !turbo4_body);
    // LLAMA_LSA_RESIDENT body (16-cell strip slots): take the live-strip walk, so the strips about to be read are known
    // and resolved into slots before the kernel (a strip without a slot reads the pinned master)
    const bool strip_res = !turbo4_body && ggml_cuda_qsa_strip_body(dst->src[5]);
    if (!turbo4_body && (live_env || lsa_blk > 0 || strip_res) && ggml_cuda_fattn_sj_live_records == nullptr) {
        const ggml_tensor * mask = dst->src[3];
        const int64_t positions = ggml_get_op_params_i32(dst, 6);
        const bool head_mask = mask && mask->ne[2] > 1;
        if (mask && mask->type == GGML_TYPE_F16 && (!head_mask || mask->ne[2] == dst->src[0]->ne[2]) && mask->ne[3] == 1 &&
                dst->src[0]->ne[3] == 1 && (lsa_blk == 0 || !head_mask) &&
                positions > 0 && mask->ne[0] >= positions && mask->ne[1] >= dst->src[0]->ne[1] && dst->src[6] != nullptr) {
            using namespace fattn_sj_kvarn_stream;
            const int nrec_max = int(positions/16) + 1;
            const int n_lists  = head_mask ? int(dst->src[1]->ne[2]) : 1;            // one per KV head
            const int gqa      = int(dst->src[0]->ne[2]/dst->src[1]->ne[2]);
            const int32_t * desc = (const int32_t *) dst->src[6]->data;
            // set_lsa_vk_units: the unit pick sits after the n_vk mean-block counts in row 0 of src[8]
            const int sel_off  = lsa_blk > 0 ? int(ggml_flash_attn_ext_get_lsa_sel_off(dst)) : 0;
            const int sel_nbs  = lsa_blk > 0 ? int(dst->src[8]->ne[0]) - sel_off : 0;
            int sel_sh = 0;
            for (int32_t b = lsa_blk; b > 1; b >>= 1) {
                ++sel_sh;
            }
            const int sel_w    = (sel_nbs + 31)/32;
            const int hdr      = SJ_LIST_HDR;
            const bool vk      = ggml_flash_attn_ext_get_lsa_vk(dst);
            sj_lsa_vk vka = {};
            if (vk) {
                const ggml_tensor * km = dst->src[7];
                const ggml_tensor * vm = ggml_flash_attn_ext_get_lsa_vk_vm(dst);
                const ggml_tensor * nd = dst->src[8];
                // rows readable past the view: the mean caches are views of a larger [d, n_blk] tensor
                auto rows = [](const ggml_tensor * t) -> int64_t {
                    const ggml_tensor * b = t->view_src;
                    return b && b->nb[1] == t->nb[1] && b->ne[0] == t->ne[0] ? (int64_t) (ggml_nbytes(b) - t->view_offs)/t->nb[1] : t->ne[1];
                };
                vka.k        = (const half *) km->data;
                vka.v        = (const half *) vm->data;
                vka.nd       = (const float *) nd->data;
                vka.k_pitch  = int(km->nb[1]/sizeof(half));
                vka.v_pitch  = int(vm->nb[1]/sizeof(half));
                vka.nd_pitch = int(nd->nb[1]/sizeof(float));
                vka.n        = int(ggml_flash_attn_ext_get_lsa_vk_n(dst));
                vka.alloc    = int(std::min(rows(km), rows(vm)));
                vka.bias     = ggml_get_op_params_f32(dst, 11);
                GGML_ASSERT(vka.alloc >= 16 && vka.alloc >= vka.n && "LSA virtual keys need >= 16 readable mean rows");
                GGML_ASSERT(km->nb[1] % 16 == 0 && vm->nb[1] % 16 == 0 && ((uintptr_t) km->data % 16) == 0 &&
                            ((uintptr_t) vm->data % 16) == 0);
            }
            const size_t lists = (size_t) (nrec_max + 1)*n_lists;
            ggml_cuda_pool_alloc<int32_t> flags(ctx.pool(), (size_t) nrec_max*n_lists);
            ggml_cuda_pool_alloc<int32_t> list(ctx.pool(), hdr + lists + sel_w); // header, lists, selection bits
            const uint32_t * selb = nullptr;
            if (lsa_blk > 0) {
                selb = (const uint32_t *) (list.ptr + hdr + lists);
                sj_lsa_sel_bits<<<(sel_w + 127)/128, 128, 0, ctx.stream()>>>((const float *) dst->src[8]->data + sel_off,
                    sel_nbs, (uint32_t *) selb, vk && sel_off == 0 ? 1 : 0);
                CUDA_CHECK(cudaGetLastError());
            }
            sj_live_record_flags<<<dim3((nrec_max + 7)/8, n_lists), 8*WARP_SIZE, 0, ctx.stream()>>>((const half *) mask->data,
                int(mask->nb[1]/sizeof(half)), int(dst->src[0]->ne[1]), desc, flags.ptr, nrec_max,
                head_mask ? int64_t(gqa)*int64_t(mask->nb[2]/sizeof(half)) : 0, strip_env || strip_res ? 1 : 0, selb, sel_sh, sel_nbs);
            CUDA_CHECK(cudaGetLastError());
            sj_live_record_list<<<n_lists, 1024, 0, ctx.stream()>>>(desc, flags.ptr, list.ptr + hdr, nrec_max,
                head_mask ? nrec_max + 1 : 0, lsa_blk > 0 ? int(lists) : 0, sel_sh, sel_nbs, vka);
            CUDA_CHECK(cudaGetLastError());
            if (strip_res && n_lists == 1) {
                ggml_cuda_qsa_strip_prepare(ctx, dst->src[5], desc, list.ptr + hdr);
            }
            ggml_cuda_fattn_sj_live_records = list.ptr + hdr;
            ggml_cuda_flash_attn_ext_sj_kvarn_stream_impl<turbo4_body, rot>(ctx, dst);
            ggml_cuda_fattn_sj_live_records = nullptr;
            return;
        }
        GGML_ASSERT(lsa_blk == 0 && "LSA block list needs a plain F16 [n_kv, n_q] mask on the SJ-KVaRN stream kernel");
    }
    if constexpr (!turbo4_body && !rot) {
        if (ggml_cuda_fattn_sj_kvarn_rot(dst)) { // [#139] fused rotation: Q pre-pass here, output rotation in the kernel
            ggml_cuda_pool_alloc<float> q_rot(ctx.pool(), ggml_nelements(dst->src[0]));
            ggml_tensor q2;
            ggml_cuda_sj_kvarn_rot256_q_pass(ctx, dst->src[0], q_rot.ptr, &q2);
            ggml_tensor d2 = *dst;
            d2.src[0] = &q2;
            ggml_cuda_flash_attn_ext_sj_kvarn_stream_impl<false, true>(ctx, &d2);
            return;
        }
    }
    GGML_ASSERT(turbo4_body || rot || !ggml_cuda_fattn_sj_kvarn_rot(dst));
    static const int nstage_env = getenv("GGML_SJKVARN_DIRECT_NSTAGE") != nullptr ? atoi(getenv("GGML_SJKVARN_DIRECT_NSTAGE")) : 0;
    static const bool kw8 = getenv("GGML_SJKVARN_DIRECT_KW8") != nullptr;
    static const bool stable = getenv("GGML_SJKVARN_DIRECT_STABLE") != nullptr && atoi(getenv("GGML_SJKVARN_DIRECT_STABLE")) != 0;
    // Unset selects by shape; 0 forces specialized half and 1 forces generic half.
    static const int nt5_mode = getenv("GGML_SJKVARN_DIRECT_NT5_GENERIC") != nullptr ? atoi(getenv("GGML_SJKVARN_DIRECT_NT5_GENERIC")) : -1;
    const int NT = dst->src[0]->ne[1];
    GGML_ASSERT(NT >= 1 && NT <= 8);
    if (stable) {
        if (NT == 5) {
            ggml_cuda_fattn_sj_kvarn_q5g6_off("the GGML_SJKVARN_DIRECT_STABLE launch");
        }
        // Fix both KV ranges and reduction order across query widths; useful for verification checks.
        const auto & device = ggml_cuda_info().devices[ggml_cuda_get_device()];
        const int blocks = ggml_cuda_fattn_sj_kvarn_stream_occupancy<8, 6, turbo4_body, rot>(4, 1);
        const int heads = dst->src[1]->ne[2];
        const int parallel_blocks = std::max(1, (device.nsm*blocks + heads - 1)/heads);
        ggml_cuda_flash_attn_ext_sj_kvarn_stream_launch<8, 6, turbo4_body, rot>(ctx, dst, NT, 1, parallel_blocks);
        return;
    }
    const int KW = NT <= 4 ? (kw8 ? 8/NT : 4/NT) : 1;
    // Measured SM86 crossover for 24 query heads / 4 KV heads; retain other shapes.
    const bool short_nt5 = !turbo4_body && ggml_cuda_info().devices[ctx.device].cc == 860 &&
        dst->src[0]->ne[2] == 24 && dst->src[1]->ne[2] == 4 && ggml_get_op_params_i32(dst, 6) <= 5376;
    const bool nt5_generic = nt5_mode < 0 ? short_nt5 : nt5_mode != 0;
    if (NT == 5 && nt5_generic) {
        ggml_cuda_fattn_sj_kvarn_q5g6_off(nt5_mode < 0 ? "the generic half (<= 5,376 positions on SM86, by design)"
                                                       : "the generic half (GGML_SJKVARN_DIRECT_NT5_GENERIC=1)");
    }
    // GGML_SJKVARN_NT6_OCC=1: 6 rows on the max_warps 6 instance (2 CTAs/SM) instead of max_warps 8 (1 CTA/SM). Same
    // math per element; launch_fattn sizes parallel_blocks from the instance's occupancy, so the combine order differs.
    static const bool nt6_occ = getenv("GGML_SJKVARN_NT6_OCC") != nullptr && atoi(getenv("GGML_SJKVARN_NT6_OCC")) != 0;
    if (NT == 5 && !nt5_generic) {
        // [#195] the compact tile replaces the five-warp specialization only (GQA 6: 30 pairs in 4 x 8 columns)
        const int q5g6_mode = ggml_cuda_fattn_sj_kvarn_q5g6_mode_for(dst);
        if (q5g6_mode != 0) {
            ggml_cuda_fattn_sj_kvarn_stream_q5g6<turbo4_body, rot>(ctx, dst, nstage_env, q5g6_mode);
            return;
        }
        ggml_cuda_fattn_sj_kvarn_stream_select<5, turbo4_body, rot>(ctx, dst, NT, KW, nstage_env, 0);
    } else if (NT == 6 && nt6_occ) {
        ggml_cuda_fattn_sj_kvarn_stream_select<6, turbo4_body, rot>(ctx, dst, NT, KW, nstage_env, 0);
    } else if (NT*KW <= 4) {
        ggml_cuda_fattn_sj_kvarn_stream_select<4, turbo4_body, rot>(ctx, dst, NT, KW, nstage_env, 0);
    } else {
        ggml_cuda_fattn_sj_kvarn_stream_select<8, turbo4_body, rot>(ctx, dst, NT, KW, nstage_env, 0);
    }
}
