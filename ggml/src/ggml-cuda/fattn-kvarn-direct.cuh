// KVarN fragment-direct decode attention (D = 256, 4-bit K and V records, decode widths 1..8).
//
// The sealed record layout (ggml_kvarn::code_bit) stores each (16-token strip, 16-channel tile) of a record as 32
// words, one per lane, where each word is exactly the lane's m16n8k16 A fragment: the K word holds the strip's
// tokens lane/4 (+8) x channels 2(lane%4)+{0,1} (+8), the V word holds V^T, channels lane/4 (+8) x tokens
// 2(lane%4)+{0,1} (+8). A warp therefore streams a strip with 32 coalesced 4-byte loads per lane, unpacks the
// nibbles into half2 with three integer ops per pair and feeds the tensor core without touching shared memory.
//
// Per record the K decode (q*Kscale[d] + Kzero[d])*Ktok[t] is folded into the query: Q'[j][d] = Q[j][d]*scale*Kscale[d]
// and c_r[j] = sum_d Kzero[d]*Q[j][d]*scale, so a strip costs one MMA per channel tile on the raw codes and
// KQ[t][j] = Ktok[t]*(q.Q' + c_r[j]). The V decode (q*Vscale[t] + Vzero[t])*Vch[d] is applied to the codes in
// fp16 (two half2 ops per pair) and the P.V product accumulates straight into fp32. The per-lane Kscale/Kzero/Vch
// slices are contiguous in the record because the sealer permutes the channel metadata into fragment order
// (ggml_kvarn::k_ch_idx / v_ch_idx).
//
// Work split: the padded position range is cut into 128-position units; every warp walks a contiguous unit range
// through sink (exact f16 rows), body (records) and ring (exact rows) with an online softmax and no block-level
// synchronization inside the KV loop. Exact rows are loaded straight from the f16 ring tensors (32-bit loads in
// fragment order, V transposed with movmatrix). One warp serves one query row (all heads of the GQA group, padded
// to 8 columns); the warps of a block are (query rows) x (KV splits), the splits combine through shared memory
// and the blocks through the parallel_blocks protocol of launch_fattn (flash_attn_combine_results).

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh" // fattn_kvarn_decode_word, fattn_kvarn_u32_as_half2
#include "mma.cuh"

#include <cfloat>
#include <climits>
#include <cstdlib>

// perf probes (never set in a production build): 1 = body strips load the words but skip decode/MMA/softmax,
// 2 = body strips skip the word loads and decode synthetic words. Both give wrong results.
#ifndef KVARN_DIRECT_PROBE
#define KVARN_DIRECT_PROBE 0
#endif

namespace fattn_kvarn_direct {

using namespace ggml_cuda_mma;

constexpr int D       = 256;
constexpr int NTC     = D/16;    // channel tiles per row
constexpr int NCOLS2  = 8;       // head columns per query row (GQA padded to 8)
constexpr int UNIT    = 128;     // positions per work unit (one record)
constexpr int COL_STRIDE = D + 4; // fp32 stride of one head column in the combine buffer (bank spread)
constexpr int SLOT_FLOATS = NCOLS2*COL_STRIDE + 2*NCOLS2; // acc + (max, sum) per column
constexpr size_t SLOT_BYTES = SLOT_FLOATS*sizeof(float);

typedef tile<16, 8, half2> T_A;  // K strip (tokens x channels) / V^T strip (channels x tokens)
typedef tile< 8, 8, half2> T_B;  // Q'^T (heads x channels) / P (heads x tokens)
typedef tile<16, 8, float> T_C;  // KQ (tokens x heads) / O (channels x heads), fp32

// Combine buffer slot for one warp: acc[col][d] + meta.
static __device__ __forceinline__ float * slot_acc (float * base, const int slot) { return base + slot*SLOT_FLOATS; }
static __device__ __forceinline__ float * slot_meta(float * base, const int slot) { return base + slot*SLOT_FLOATS + NCOLS2*COL_STRIDE; }

template <int max_warps>
__launch_bounds__(WARP_SIZE*max_warps, max_warps == 4 ? 2 : 1)
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
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)
    GGML_UNUSED_VARS(sinks, KV_max, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb03, ne10, ne13,
                     nb13, nb23, ne31, ne32, ne33, nb32, nb33);

    extern __shared__ float smem_combine[];

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;
    const int n_q    = int(ne01.z);
    const int NT     = n_q;          // one n-tile (8 head columns) per query row
    const int KW     = nwarps / NT;  // KV splits inside the block (host: nwarps == NT*KW, KW a power of two)
    const int nt     = warp % NT;    // this warp's query row
    const int kw     = warp / NT;    // this warp's KV split

    const int z_KV  = blockIdx.z;    // KV head (ne03 == 1)
    const int gqa   = ne02 / ne12;
    const int head0 = z_KV*gqa;
    const int hcol  = lane >> 2;     // the lane's head column (B/C fragment row)
    const bool hvalid = hcol < gqa;

    const fattn_kvarn_ctx kv = fattn_kvarn_make_ctx(kvarn_body, kvarn_desc, z_KV, D, 4, 4);
    const int n_kv_pad = ne11;

    // contiguous unit range for this (block, split)
    const int nunits = n_kv_pad / UNIT;
    const int nsplit = gridDim.y * KW;
    const int split  = blockIdx.y * KW + kw;
    const int s_lo   = (int) (((int64_t) nunits *  split     ) / nsplit) * (UNIT/16);
    const int s_hi   = (int) (((int64_t) nunits * (split + 1)) / nsplit) * (UNIT/16);

    const float * Qf = (const float *) (Q + (size_t) nt*nb01 + (size_t) (head0 + (hvalid ? hcol : 0))*nb02);
    const half  * mrow = (const half *) (mask + (size_t) nt*nb31);
    const char  * Kh = K + (size_t) z_KV*nb12;
    const char  * Vh = V + (size_t) z_KV*nb22;

    T_B   qp[NTC];   // Q' B fragments for the current record
    float c_r[2] = {0.0f, 0.0f};
    T_C   acc[NTC];  // unnormalized O, fp32 (zero-initialized by the tile ctor)
    float kq_max[2] = {-FLT_MAX/2, -FLT_MAX/2};
    float kq_sum[2] = {0.0f, 0.0f};

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
    auto setup_body = [&](const char * rec) {
        const uint2 * ks2 = (const uint2 *) (rec + kv.k_scale + 2*(lane & 3)*(D/4));
        const uint2 * kz2 = (const uint2 *) (rec + kv.k_zero  + 2*(lane & 3)*(D/4));
        float cr = 0.0f;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 ks = __ldg(ks2 + c);
            const uint2 kz = __ldg(kz2 + c);
            float2 q0, q1;
            load_q(c, q0, q1);
            const half2 ks0 = fattn_kvarn_u32_as_half2(ks.x), ks1 = fattn_kvarn_u32_as_half2(ks.y);
            const float2 kz0 = __half22float2(fattn_kvarn_u32_as_half2(kz.x));
            const float2 kz1 = __half22float2(fattn_kvarn_u32_as_half2(kz.y));
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

    // online softmax on the strip's KQ tile (fp32, mask applied); returns P as the B fragment for the V MMA
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
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] *= sc0;
                acc[c].x[1] *= sc1;
                acc[c].x[2] *= sc0;
                acc[c].x[3] *= sc1;
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

    uint32_t kw_[NTC], vw_[NTC]; // fragment words of the current body strip
#if KVARN_DIRECT_PROBE == 1
    uint32_t probe_acc = 0;
#endif
    auto load_k_words = [&](const char * rec, const int sl) {
        const uint32_t * w = (const uint32_t *) rec + (sl*NTC)*WARP_SIZE + lane;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
#if KVARN_DIRECT_PROBE == 2
            kw_[c] = (uint32_t) (sl*131 + c*17 + lane) * 0x9E3779B9u; (void) w;
#else
            kw_[c] = __ldg(w + c*WARP_SIZE);
#endif
        }
    };
    auto load_v_words = [&](const char * rec, const int sl) {
        const uint32_t * w = (const uint32_t *) (rec + kv.v_payload) + (sl*NTC)*WARP_SIZE + lane;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
#if KVARN_DIRECT_PROBE == 2
            vw_[c] = (uint32_t) (sl*137 + c*19 + lane) * 0x85EBCA6Bu; (void) w;
#else
            vw_[c] = __ldg(w + c*WARP_SIZE);
#endif
        }
    };

    int          cur_id    = INT_MIN; // record index, -1 sink, -2 ring
    bool         have_words = false;  // kw_/vw_ hold the current strip

    for (int s = s_lo; s < s_hi; ++s) {
        const int p0 = 16*s;

        int          id;
        const char * rec  = nullptr;
        int          sl   = 0;   // strip inside the record
        int          row0 = 0;   // first exact row
        bool         exact;
        if (p0 < kv.S) {
            exact = true; id = -1; row0 = p0;
        } else if (p0 < kv.B) {
            exact = false;
            const int g = (p0 - kv.S) / kv.G;
            id  = g;
            rec = kv.body + (size_t) g*kv.rec_stride;
            sl  = ((p0 - kv.S) - g*kv.G) / 16;
        } else {
            exact = true; id = -2; row0 = kv.S + (p0 - kv.S) % kv.cap;
        }

        if (id != cur_id) {
            if (exact) {
                setup_exact();
            } else {
                setup_body(rec);
            }
            cur_id = id;
        }

        // next strip of this warp (for the word prefetch)
        const int  p0n      = p0 + 16;
        const bool next_body = (s + 1 < s_hi) && p0n >= kv.S && p0n < kv.B;
        const char * recn = nullptr;
        int sln = 0;
        if (next_body) {
            const int gn = (p0n - kv.S) / kv.G;
            recn = kv.body + (size_t) gn*kv.rec_stride;
            sln  = ((p0n - kv.S) - gn*kv.G) / 16;
        }

        const half mk0 = mrow[p0 + (lane >> 2)];
        const half mk1 = mrow[p0 + (lane >> 2) + 8];

        T_C kq;
        if (!exact) {
            if (!have_words) {
                load_k_words(rec, sl);
                load_v_words(rec, sl);
            }
#if KVARN_DIRECT_PROBE == 1
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                probe_acc ^= kw_[c] + vw_[c];
            }
            if (next_body) {
                load_k_words(recn, sln);
                load_v_words(recn, sln);
            }
            have_words = next_body;
            (void) kq;
#else
            // KQ = Ktok[t] * (q . Q' + c_r) + mask
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                fattn_kvarn_decode_word(kw_[c], a.x);
                mma(kq, a, qp[c]);
            }
            if (next_body) {
                load_k_words(recn, sln);
            }
            const half * ktok = (const half *) (rec + kv.k_tok) + sl*16;
            const float tok0 = __half2float(ktok[lane >> 2]);
            const float tok1 = __half2float(ktok[(lane >> 2) + 8]);
            kq.x[0] = tok0*(kq.x[0] + c_r[0]) + __half2float(mk0);
            kq.x[1] = tok0*(kq.x[1] + c_r[1]) + __half2float(mk0);
            kq.x[2] = tok1*(kq.x[2] + c_r[0]) + __half2float(mk1);
            kq.x[3] = tok1*(kq.x[3] + c_r[1]) + __half2float(mk1);

            const T_B P = softmax(kq);

            // V: ((q*Vscale[t] + Vzero[t])*Vch[d]) straight into the fp32 accumulator; the lane's two channels of
            // tile c (lane/4 and lane/4+8) sit in one uint32 of the permuted Vch vector
            const half2 * vsp = (const half2 *) (rec + kv.v_scale + 2*(sl*16 + 2*(lane & 3)));
            const half2 * vzp = (const half2 *) (rec + kv.v_zero  + 2*(sl*16 + 2*(lane & 3)));
            const half2 vs0 = vsp[0], vs1 = vsp[4];
            const half2 vz0 = vzp[0], vz1 = vzp[4];
            const uint32_t * vchp = (const uint32_t *) (rec + kv.v_ch + 2*((lane >> 2)*(D/8)));
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                fattn_kvarn_decode_word(vw_[c], a.x);
                const half2 vch = fattn_kvarn_u32_as_half2(__ldg(vchp + c));
                const half2 vlo = __low2half2(vch);
                const half2 vhi = __high2half2(vch);
                a.x[0] = __hmul2(__hfma2(a.x[0], vs0, vz0), vlo);
                a.x[1] = __hmul2(__hfma2(a.x[1], vs0, vz0), vhi);
                a.x[2] = __hmul2(__hfma2(a.x[2], vs1, vz1), vlo);
                a.x[3] = __hmul2(__hfma2(a.x[3], vs1, vz1), vhi);
                mma(acc[c], a, P);
            }
            if (next_body) {
                load_v_words(recn, sln);
            }
            have_words = next_body;
#endif
        } else {
            // exact f16 rows in fragment order: K straight, V via an 8x8 transpose
            const char * Kr = Kh + (size_t) row0*nb11;
            const char * Vr = Vh + (size_t) row0*nb21;
            // exact rows are a small fraction of the positions (sink + ring); the loads are issued four tiles at a
            // time (compiler barrier) so this path does not set the kernel's register footprint
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                if (c % 4 == 0) {
                    asm volatile("" ::: "memory");
                }
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = (lane >> 2) + 8*(l & 1);
                    const int cp  = 8*c + 4*(l >> 1) + (lane & 3);
                    a.x[l] = fattn_kvarn_u32_as_half2(__ldg((const uint32_t *) (Kr + (size_t) row*nb11 + 4*cp)));
                }
                mma(kq, a, qp[c]);
            }
            if (next_body) {
                load_k_words(recn, sln);
            }
            kq.x[0] += __half2float(mk0);
            kq.x[1] += __half2float(mk0);
            kq.x[2] += __half2float(mk1);
            kq.x[3] += __half2float(mk1);

            const T_B P = softmax(kq);

#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                if (c % 4 == 0) {
                    asm volatile("" ::: "memory");
                }
                T_A a;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int row = 8*(l >> 1) + (lane >> 2);      // token row of the 8x8 block
                    const int cp  = 8*c + 4*(l & 1) + (lane & 3);  // channel pair
                    const uint32_t u = __ldg((const uint32_t *) (Vr + (size_t) row*nb21 + 4*cp));
                    a.x[l] = ggml_cuda_movmatrix(fattn_kvarn_u32_as_half2(u));
                }
                mma(acc[c], a, P);
            }
            if (next_body) {
                load_v_words(recn, sln);
            }
            have_words = next_body;
        }
    }

    // finish the lane-partial sums
#pragma unroll
    for (int off = 4; off < WARP_SIZE; off <<= 1) {
        kq_sum[0] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[0], off, WARP_SIZE);
        kq_sum[1] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[1], off, WARP_SIZE);
    }

    // combine the KV splits of the block (pairwise tree, upper half stores, lower half merges)
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
#if KVARN_DIRECT_PROBE == 1
                out[d] = acc[c].x[l + e]*inv + (float) (probe_acc & 1u);
#else
                out[d] = acc[c].x[l + e]*inv;
#endif
            }
        }
    }
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
#endif // defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)
}

} // namespace fattn_kvarn_direct

// Host side. The block is (query rows) x (KV splits): NT = n_q warps for the rows, KW = 4/NT splits while that
// keeps the block at 4 warps (n_q <= 4), otherwise one split per block (n_q 5..8, 5..8 warps).
inline bool ggml_cuda_flash_attn_ext_kvarn_direct_supported_impl(const ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * sinks = dst->src[4];
    if (sinks != nullptr || Q->ne[1] < 1 || Q->ne[1] > 8 || Q->ne[3] != 1) {
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
    static const bool disabled = getenv("GGML_KVARN_NO_DIRECT") != nullptr;
    return !disabled;
}

template <int max_warps>
static void ggml_cuda_flash_attn_ext_kvarn_direct_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int nwarps) {
    using namespace fattn_kvarn_direct;
    fattn_kernel_t fattn_kernel = flash_attn_ext_kvarn_direct<max_warps>;
    // combine buffer: at most nwarps/2 warps store at once
    const size_t nbytes_shared = (size_t) (nwarps/2) * SLOT_BYTES;
    launch_fattn<D, NCOLS2, NCOLS2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared, UNIT,
         /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false, /*use_sparse=*/false, WARP_SIZE);
}

inline void ggml_cuda_flash_attn_ext_kvarn_direct_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const int NT = Q->ne[1];
    const int KW = NT <= 4 ? 4/NT : 1;
    const int nwarps = NT*KW;
    if (nwarps <= 4) {
        ggml_cuda_flash_attn_ext_kvarn_direct_launch<4>(ctx, dst, nwarps);
    } else {
        ggml_cuda_flash_attn_ext_kvarn_direct_launch<8>(ctx, dst, nwarps);
    }
}

// exported entry points, defined once in template-instances/fattn-mma-kvarn-direct-instance.cu
bool ggml_cuda_flash_attn_ext_kvarn_direct_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_kvarn_direct(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
