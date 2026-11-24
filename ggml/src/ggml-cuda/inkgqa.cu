// INKGQA: GQA-group tensor-core prefill attention (see inkgqa.cuh). All routes default OFF.
//
// Block = 16*RW query rows x the G q heads of one KV head (NW = G*RW warps; warp w serves q head kvh*G + w/RW, rows
// 16*(w%RW) .. +16 of the row block), so a 32-key K/V tile in shared memory serves G heads.
//   Q    : registers, f16 A fragments of m16n8k16 (pre-scaled by the softmax scale), rows gr / gr + 8 of each warp
//   K, V : [2][32 keys][128 + 8] f16 in shared memory, filled by cp.async one tile ahead
//   S    = Q K^T (B fragments by ldmatrix), logits in log2 units: (S + m) * log2(e), m = the mask value (plain), or
//          f16(mask + bias) (INKREL table / factor form)
//   O   += P V (P = the S accumulators as f16 A fragments, V fragments by ldmatrix.trans)
// A pre-pass classifies each (row block, key tile): dead (no row sees a key: not visited), whole (every cell of the
// block is a 0 mask: no mask read) or partial. The live tiles of a row block can be split over gridDim.z blocks; the
// split partials (unnormalised O, row max, row sum) are merged by inkgqa_combine.
// Factor form: per warp and key tile, the 16 x 48 bias block G = R(16 rows x 16) . E[dlo .. dlo + 48)^T is 6 MMAs
// (dlo = the smallest distance the warp's visible cells have, which spans 15 + 31 for contiguous positions); each cell
// then reads G[i][d - dlo] from a per-warp shared skew buffer. Wider spans (ring wrap) fall back to f32 dot products.
// Fragment layout (m16n8k16, lane = 4 gr + tq): A a0 (gr, 2tq..), a1 (gr+8, 2tq..), a2 (gr, 2tq+8..), a3 (gr+8, 2tq+8..);
// B b0 (k 2tq.., n gr), b1 (k 2tq+8.., n gr); C c0,c1 (gr, 2tq..), c2,c3 (gr+8, 2tq..).

#include "inkgqa.cuh"
#include "inkrel-bias.cuh"
#include "ggml-inkrel.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int   GQ_D  = 128;
constexpr int   GQ_BN = 32;          // keys per tile
constexpr int   GQ_LD = GQ_D + 8;    // shared row stride in halves (272 bytes: ldmatrix rows on distinct banks)
constexpr int   GQ_G  = 3;           // q heads per KV head (Inkling 2)
constexpr int   GQ_NREL = 16;        // factor form: rank of the relative bias
constexpr int   GQ_GW = 48;          // factor form: distances per skew block (6 n8 MMAs)
constexpr int   GQ_GLD = GQ_GW + 4;  // skew row stride in floats (rows on distinct banks)
constexpr float GQ_LOG2E = 1.4426950408889634f;
constexpr size_t GQ_SMEM_KV = 4*GQ_BN*GQ_LD*sizeof(half); // K and V, two buffers each (34,816 bytes)

enum { GQ_PLAIN = 0, GQ_TABLE = 1, GQ_FACTOR = 2 };

constexpr uint32_t GQ_WHOLE = 1u << 30;

struct gq_params {
    const char * Q; int64_t nbq1, nbq2;     // f32 [D, T, H]
    const char * K; int64_t nbk1, nbk2;     // f16 [D, n_kv, Hkv]
    const char * V; int64_t nbv1, nbv2;
    const half * M; int64_t m_t;            // f16 broadcast mask, halves per query row
    int m_al4;                              // half2 mask loads
    const uint32_t * mp; int64_t mp_t;      // or the one-bit packed mask (ggml.h): 1 = visible; words per query row
    ink_rel_bias b;                         // table form: table + rel; factor form: rel only
    const float * rf; int extent;           // factor form: E rows [0, extent), r rows extent + t*H + h
    const float * sinks;                    // [H] per head, or [H, T] per row (factor form), or nullptr
    int sinks_row;
    const float * gate; int64_t gate_t;     // [D*H, T] or nullptr (floats per token)
    char * O; int64_t nbo1, nbo2;           // f32 dst [D, H, T]
    float * lse;                            // [T, H] natural-log row log-sum-exp, or nullptr
    float * part_o;                         // split > 1: [split][T][H][D] unnormalised O
    float2 * part_ml;                       //            [split][T][H] (row max in log2 units, row sum)
    const uint32_t * tiles;                 // [n_rb][NT] live key tiles of each row block (| GQ_WHOLE)
    const int32_t * n_tiles;                // [n_rb]
    int NT, T, H, Hkv, n_kv, split;
    float scale;
};

static __device__ __forceinline__ uint32_t gq_h2u(const float x, const float y) {
    const half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<const uint32_t *>(&h);
}

static __device__ __forceinline__ void gq_mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    GGML_UNUSED_VARS(c, a, b0, b1);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void gq_ldsm4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const half * p) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#else
    GGML_UNUSED_VARS(r0, r1, r2, r3, p);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void gq_ldsm4_t(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const half * p) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
#else
    GGML_UNUSED_VARS(r0, r1, r2, r3, p);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// 16 bytes global -> shared, zero-filled when !ok (src must still be a valid address)
static __device__ __forceinline__ void gq_cp16(half * dst, const void * src, const bool ok) {
#ifdef AMPERE_MMA_AVAILABLE
    const unsigned a = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(a), "l"(src), "r"(ok ? 16 : 0) : "memory");
#else
    GGML_UNUSED_VARS(dst, src, ok);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void gq_commit() {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif // AMPERE_MMA_AVAILABLE
}

template <int N>
static __device__ __forceinline__ void gq_wait() {
#ifdef AMPERE_MMA_AVAILABLE
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif // AMPERE_MMA_AVAILABLE
}

// K and V rows [k0, k0 + 32) of one KV head into Ks / Vs (rows past n_kv zero-filled); 512 16-byte chunks per matrix
template <int NT>
static __device__ __forceinline__ void gq_load_tile(half * Ks, half * Vs, const char * Kh, const char * Vh,
        const int64_t nbk1, const int64_t nbv1, const int k0, const int n_kv, const int tid) {
    constexpr int chunks = GQ_BN*(GQ_D/8);
#pragma unroll
    for (int c0 = 0; c0 < chunks; c0 += NT) {
        const int c = c0 + tid;
        if (chunks % NT == 0 || c < chunks) {
            const int row = c / (GQ_D/8);
            const int col = (c % (GQ_D/8))*8;
            const int key = k0 + row;
            const bool ok = key < n_kv;
            const int kr  = ok ? key : 0;
            gq_cp16(Ks + row*GQ_LD + col, Kh + (int64_t) kr*nbk1 + col*sizeof(half), ok);
            gq_cp16(Vs + row*GQ_LD + col, Vh + (int64_t) kr*nbv1 + col*sizeof(half), ok);
        }
    }
}

// flags[rb][tile]: 0 = no row of the BM-row block rb sees a key of the tile, 2 = every cell (rows < T, keys < n_kv,
// the tile inside n_kv) is exactly 0, 1 = otherwise
template <int BM>
static __global__ void inkgqa_tile_flags(const half * __restrict__ M, const int64_t m_t, const uint32_t * __restrict__ MP,
        const int64_t mp_t, const int T, const int n_kv, const int NT, uint8_t * __restrict__ flags) {
    const int tile = blockIdx.x;
    const int rb   = blockIdx.y;
    bool any = false;
    bool all0 = (tile + 1)*GQ_BN <= n_kv;
    for (int e = threadIdx.x; e < BM*GQ_BN; e += blockDim.x) {
        const int row = rb*BM + e / GQ_BN;
        const int key = tile*GQ_BN + e % GQ_BN;
        if (row < T && key < n_kv) {
            const float v = MP ? ((MP[(int64_t) row*mp_t + key/32] >> (key % 32)) & 1u ? 0.0f : -INFINITY) :
                                 __half2float(M[(int64_t) row*m_t + key]);
            any  |= v != -INFINITY;
            all0 &= v == 0.0f;
        }
    }
    any  = __syncthreads_or(any);
    all0 = __syncthreads_and(all0);
    if (threadIdx.x == 0) {
        flags[(int64_t) rb*NT + tile] = any ? (all0 ? 2 : 1) : 0;
    }
}

// tiles[rb][0, n_tiles[rb]) = the live tiles of row block rb in increasing order (| GQ_WHOLE); one warp per row block
static __global__ void inkgqa_tile_lists(const uint8_t * __restrict__ flags, const int NT, uint32_t * __restrict__ tiles,
        int32_t * __restrict__ n_tiles) {
    const int rb   = blockIdx.x;
    const int lane = threadIdx.x;
    int cnt = 0;
    for (int t0 = 0; t0 < NT; t0 += WARP_SIZE) {
        const int t = t0 + lane;
        const int f = t < NT ? flags[(int64_t) rb*NT + t] : 0;
        const unsigned bal = __ballot_sync(0xFFFFFFFF, f != 0);
        if (f) {
            tiles[(int64_t) rb*NT + cnt + __popc(bal & ((1u << lane) - 1))] = (uint32_t) t | (f == 2 ? GQ_WHOLE : 0u);
        }
        cnt += __popc(bal);
    }
    if (lane == 0) {
        n_tiles[rb] = cnt;
    }
}

// final row: O*f (+ gate), LSE; M in log2 units, L the row sum
static __device__ __forceinline__ void gq_row_final(const gq_params & p, const int t, const int h, float M, float L,
        float & f, float & lse_out) {
    float Ls = L, Ms = M, sa = 1.0f;
    if (p.sinks) {
        const float s2 = (p.sinks_row ? p.sinks[(int64_t) t*p.H + h] : p.sinks[h])*GQ_LOG2E;
        const float M2 = fmaxf(Ms, s2);
        sa = Ms == -INFINITY ? 0.0f : exp2f(Ms - M2);
        Ls = Ls*sa + exp2f(s2 - M2);
        Ms = M2;
    }
    f = Ls > 0.0f ? sa / Ls : 0.0f;
    lse_out = Ls > 0.0f ? (Ms + log2f(Ls))/GQ_LOG2E : -INFINITY;
}

template <int RW, int MODE>
static __global__ void __launch_bounds__(GQ_G*RW*WARP_SIZE, 1) inkgqa_fa(const gq_params p) {
#ifdef AMPERE_MMA_AVAILABLE
    constexpr int NW = GQ_G*RW;
    constexpr int NT = NW*WARP_SIZE;
    constexpr int BM = 16*RW;
    constexpr int NJ = GQ_BN/8;     // n8 key blocks per tile
    extern __shared__ __align__(16) half gq_sm[];
    half  * Ks = gq_sm;                        // [2][BN][LD]
    half  * Vs = gq_sm + 2*GQ_BN*GQ_LD;        // [2][BN][LD]

    const int tid  = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int gr   = lane / 4;
    const int tq   = lane % 4;
    const int kvh  = blockIdx.x;
    const int rb   = blockIdx.y;
    const int z    = blockIdx.z;
    const int h    = kvh*GQ_G + warp / RW;

    const int  ia  = rb*BM + (warp % RW)*16 + gr;
    const int  ib  = ia + 8;
    const bool va_ = ia < p.T;
    const bool vb_ = ib < p.T;
    const int  ta  = va_ ? ia : p.T - 1; // clamped rows: loads stay in bounds, results are never written
    const int  tb  = vb_ ? ib : p.T - 1;

    // Q fragments, pre-scaled, f16
    uint32_t qf[GQ_D/16][4];
    {
        const float * qa = (const float *) (p.Q + (int64_t) ta*p.nbq1 + (int64_t) h*p.nbq2);
        const float * qb = (const float *) (p.Q + (int64_t) tb*p.nbq1 + (int64_t) h*p.nbq2);
#pragma unroll
        for (int kk = 0; kk < GQ_D/16; ++kk) {
            const int d = kk*16 + 2*tq;
            const float2 a0 = va_ ? *reinterpret_cast<const float2 *>(qa + d)     : make_float2(0.0f, 0.0f);
            const float2 b0 = vb_ ? *reinterpret_cast<const float2 *>(qb + d)     : make_float2(0.0f, 0.0f);
            const float2 a1 = va_ ? *reinterpret_cast<const float2 *>(qa + d + 8) : make_float2(0.0f, 0.0f);
            const float2 b1 = vb_ ? *reinterpret_cast<const float2 *>(qb + d + 8) : make_float2(0.0f, 0.0f);
            qf[kk][0] = gq_h2u(a0.x*p.scale, a0.y*p.scale);
            qf[kk][1] = gq_h2u(b0.x*p.scale, b0.y*p.scale);
            qf[kk][2] = gq_h2u(a1.x*p.scale, a1.y*p.scale);
            qf[kk][3] = gq_h2u(b1.x*p.scale, b1.y*p.scale);
        }
    }

    // factor form: r of rows ta / tb (A fragment, f16) and this warp's skew buffer
    uint32_t ra[4] = {0u, 0u, 0u, 0u};
    float * Gs = nullptr;
    if constexpr (MODE == GQ_FACTOR) {
        const float * rra = p.rf + ((int64_t) p.extent + (int64_t) ta*p.H + h)*GQ_NREL;
        const float * rrb = p.rf + ((int64_t) p.extent + (int64_t) tb*p.H + h)*GQ_NREL;
        ra[0] = gq_h2u(rra[2*tq],     rra[2*tq + 1]);
        ra[1] = gq_h2u(rrb[2*tq],     rrb[2*tq + 1]);
        ra[2] = gq_h2u(rra[2*tq + 8], rra[2*tq + 9]);
        ra[3] = gq_h2u(rrb[2*tq + 8], rrb[2*tq + 9]);
        Gs = reinterpret_cast<float *>(gq_sm + 4*GQ_BN*GQ_LD) + warp*16*GQ_GLD;
    }

    float o[GQ_D/8][4];
#pragma unroll
    for (int j = 0; j < GQ_D/8; ++j) {
        o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    }
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;

    const half * mra = p.M ? p.M + (int64_t) ta*p.m_t : nullptr;
    const half * mrb = p.M ? p.M + (int64_t) tb*p.m_t : nullptr;
    const char * Kh  = p.K + (int64_t) kvh*p.nbk2;
    const char * Vh  = p.V + (int64_t) kvh*p.nbv2;
    const uint32_t * tl = p.tiles + (int64_t) rb*p.NT;
    const int cnt = p.n_tiles[rb];
    const int it0 = (int) (((int64_t) cnt*z)/p.split);
    const int it1 = (int) (((int64_t) cnt*(z + 1))/p.split);

    if (it0 < it1) {
        gq_load_tile<NT>(Ks, Vs, Kh, Vh, p.nbk1, p.nbv1, (int) (tl[it0] & ~GQ_WHOLE)*GQ_BN, p.n_kv, tid);
    }
    gq_commit();

    for (int it = it0; it < it1; ++it) {
        const int buf = (it - it0) & 1;
        const uint32_t e = tl[it];
        const int  k0    = (int) (e & ~GQ_WHOLE)*GQ_BN;
        const bool whole = (e & GQ_WHOLE) != 0;
        if (it + 1 < it1) {
            gq_load_tile<NT>(Ks + (buf ^ 1)*GQ_BN*GQ_LD, Vs + (buf ^ 1)*GQ_BN*GQ_LD, Kh, Vh, p.nbk1, p.nbv1,
                (int) (tl[it + 1] & ~GQ_WHOLE)*GQ_BN, p.n_kv, tid);
        }
        gq_commit();

        // the value added to each logit of this lane's 2 rows x 8 keys (keys k0 + 8 j + 2 tq + e); -inf = masked
        float ba[NJ][2], bb[NJ][2];
        bool any = false;
#pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int c = k0 + 8*j + 2*tq;
            float2 xa = make_float2(0.0f, 0.0f);
            float2 xb = xa;
            if (!whole) {
                xa = make_float2(-INFINITY, -INFINITY);
                xb = xa;
                if (p.mp) {
                    // c is even and the pair shares a word; keys >= n_kv stay masked
                    const uint32_t wa = p.mp[(int64_t) ta*p.mp_t + c/32] >> (c % 32);
                    const uint32_t wb = p.mp[(int64_t) tb*p.mp_t + c/32] >> (c % 32);
                    if (c < p.n_kv)     { xa.x = wa & 1u ? 0.0f : -INFINITY; xb.x = wb & 1u ? 0.0f : -INFINITY; }
                    if (c + 1 < p.n_kv) { xa.y = wa & 2u ? 0.0f : -INFINITY; xb.y = wb & 2u ? 0.0f : -INFINITY; }
                } else if (p.m_al4 && c + 1 < p.n_kv) {
                    xa = __half22float2(*reinterpret_cast<const half2 *>(mra + c));
                    xb = __half22float2(*reinterpret_cast<const half2 *>(mrb + c));
                } else {
                    if (c < p.n_kv)     { xa.x = __half2float(mra[c]);     xb.x = __half2float(mrb[c]); }
                    if (c + 1 < p.n_kv) { xa.y = __half2float(mra[c + 1]); xb.y = __half2float(mrb[c + 1]); }
                }
            }
            if (!va_) { xa = make_float2(-INFINITY, -INFINITY); }
            if (!vb_) { xb = make_float2(-INFINITY, -INFINITY); }
            if constexpr (MODE == GQ_TABLE) {
                ba[j][0] = xa.x != -INFINITY ? ink_rel_bias_masked(xa.x, ink_rel_bias_load(p.b, h, c,     ta)) : -INFINITY;
                ba[j][1] = xa.y != -INFINITY ? ink_rel_bias_masked(xa.y, ink_rel_bias_load(p.b, h, c + 1, ta)) : -INFINITY;
                bb[j][0] = xb.x != -INFINITY ? ink_rel_bias_masked(xb.x, ink_rel_bias_load(p.b, h, c,     tb)) : -INFINITY;
                bb[j][1] = xb.y != -INFINITY ? ink_rel_bias_masked(xb.y, ink_rel_bias_load(p.b, h, c + 1, tb)) : -INFINITY;
            } else {
                ba[j][0] = xa.x; ba[j][1] = xa.y;
                bb[j][0] = xb.x; bb[j][1] = xb.y;
            }
            any |= ba[j][0] != -INFINITY || ba[j][1] != -INFINITY || bb[j][0] != -INFINITY || bb[j][1] != -INFINITY;
        }
        any = __any_sync(0xFFFFFFFF, any);

        if constexpr (MODE == GQ_FACTOR) {
            if (any) {
                // distances of the visible cells; the far row (d == extent) adds no bias
                int da[NJ][2], db[NJ][2];
                int dlo = INT32_MAX, dhi = -1;
#pragma unroll
                for (int j = 0; j < NJ; ++j) {
#pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const int c = k0 + 8*j + 2*tq + q;
                        da[j][q] = ba[j][q] != -INFINITY ? ink_rel_bias_row(p.b, c, ta) : p.extent;
                        db[j][q] = bb[j][q] != -INFINITY ? ink_rel_bias_row(p.b, c, tb) : p.extent;
                        if (da[j][q] < p.extent) { dlo = min(dlo, da[j][q]); dhi = max(dhi, da[j][q]); }
                        if (db[j][q] < p.extent) { dlo = min(dlo, db[j][q]); dhi = max(dhi, db[j][q]); }
                    }
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    dlo = min(dlo, __shfl_xor_sync(0xFFFFFFFF, dlo, off));
                    dhi = max(dhi, __shfl_xor_sync(0xFFFFFFFF, dhi, off));
                }
                if (dhi >= 0 && dhi - dlo < GQ_GW) {
                    // G[i][n] = r_i . E[dlo + n], n < 48 (rows of E past extent read as 0)
#pragma unroll
                    for (int nb = 0; nb < GQ_GW/8; ++nb) {
                        const int d = dlo + 8*nb + gr;
                        uint32_t b0 = 0u, b1 = 0u;
                        if (d < p.extent) {
                            const float * er = p.rf + (int64_t) d*GQ_NREL;
                            const float2 e0 = *reinterpret_cast<const float2 *>(er + 2*tq);
                            const float2 e1 = *reinterpret_cast<const float2 *>(er + 2*tq + 8);
                            b0 = gq_h2u(e0.x, e0.y);
                            b1 = gq_h2u(e1.x, e1.y);
                        }
                        float c4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                        gq_mma(c4, ra, b0, b1);
                        *reinterpret_cast<float2 *>(Gs + gr*GQ_GLD + 8*nb + 2*tq)       = make_float2(c4[0], c4[1]);
                        *reinterpret_cast<float2 *>(Gs + (gr + 8)*GQ_GLD + 8*nb + 2*tq) = make_float2(c4[2], c4[3]);
                    }
                    __syncwarp();
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            if (ba[j][q] != -INFINITY) {
                                ba[j][q] = ink_rel_bias_masked(ba[j][q], da[j][q] < p.extent ? Gs[gr*GQ_GLD + da[j][q] - dlo] : 0.0f);
                            }
                            if (bb[j][q] != -INFINITY) {
                                bb[j][q] = ink_rel_bias_masked(bb[j][q], db[j][q] < p.extent ? Gs[(gr + 8)*GQ_GLD + db[j][q] - dlo] : 0.0f);
                            }
                        }
                    }
                    __syncwarp(); // the skew buffer is rewritten by the next tile
                } else {
                    // wide span (ring wrap): f32 dot products r . E[d]
                    const float * rra = p.rf + ((int64_t) p.extent + (int64_t) ta*p.H + h)*GQ_NREL;
                    const float * rrb = p.rf + ((int64_t) p.extent + (int64_t) tb*p.H + h)*GQ_NREL;
#pragma unroll
                    for (int j = 0; j < NJ; ++j) {
#pragma unroll
                        for (int q = 0; q < 2; ++q) {
                            if (ba[j][q] != -INFINITY) {
                                float acc = 0.0f;
                                if (da[j][q] < p.extent) {
                                    const float * er = p.rf + (int64_t) da[j][q]*GQ_NREL;
                                    for (int k = 0; k < GQ_NREL; ++k) { acc = fmaf(rra[k], er[k], acc); }
                                }
                                ba[j][q] = ink_rel_bias_masked(ba[j][q], acc);
                            }
                            if (bb[j][q] != -INFINITY) {
                                float acc = 0.0f;
                                if (db[j][q] < p.extent) {
                                    const float * er = p.rf + (int64_t) db[j][q]*GQ_NREL;
                                    for (int k = 0; k < GQ_NREL; ++k) { acc = fmaf(rrb[k], er[k], acc); }
                                }
                                bb[j][q] = ink_rel_bias_masked(bb[j][q], acc);
                            }
                        }
                    }
                }
            }
        }

        gq_wait<1>(); // this tile's K/V (the next tile's copies may still be in flight)
        __syncthreads();

        if (any) {
            const half * K = Ks + buf*GQ_BN*GQ_LD;
            const half * V = Vs + buf*GQ_BN*GQ_LD;

            float s[NJ][4];
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
                const half * kr = K + (8*j + (lane & 7))*GQ_LD + (lane >> 3)*8; // two k-steps per ldmatrix
#pragma unroll
                for (int kk = 0; kk < GQ_D/16; kk += 2) {
                    uint32_t b0, b1, b2, b3;
                    gq_ldsm4(b0, b1, b2, b3, kr + kk*16);
                    gq_mma(s[j], qf[kk],     b0, b1);
                    gq_mma(s[j], qf[kk + 1], b2, b3);
                }
            }

            float mxa = -INFINITY, mxb = -INFINITY;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    s[j][q]     = (s[j][q]     + ba[j][q])*GQ_LOG2E;
                    s[j][2 + q] = (s[j][2 + q] + bb[j][q])*GQ_LOG2E;
                    mxa = fmaxf(mxa, s[j][q]);
                    mxb = fmaxf(mxb, s[j][2 + q]);
                }
            }
            mxa = fmaxf(mxa, __shfl_xor_sync(0xFFFFFFFF, mxa, 1));
            mxa = fmaxf(mxa, __shfl_xor_sync(0xFFFFFFFF, mxa, 2));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xFFFFFFFF, mxb, 1));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xFFFFFFFF, mxb, 2));
            const float mna = fmaxf(ma, mxa);
            const float mnb = fmaxf(mb, mxb);
            const float ca  = mna == -INFINITY ? 1.0f : exp2f(ma - mna);
            const float cb  = mnb == -INFINITY ? 1.0f : exp2f(mb - mnb);
            const float ua  = mna == -INFINITY ? 0.0f : mna;
            const float ub  = mnb == -INFINITY ? 0.0f : mnb;
            ma = mna;
            mb = mnb;
            la *= ca;
            lb *= cb;
#pragma unroll
            for (int j = 0; j < GQ_D/8; ++j) {
                o[j][0] *= ca; o[j][1] *= ca;
                o[j][2] *= cb; o[j][3] *= cb;
            }

            uint32_t pa[GQ_BN/16][4];
#pragma unroll
            for (int kk = 0; kk < GQ_BN/16; ++kk) {
                const float p00 = exp2f(s[2*kk][0] - ua),     p01 = exp2f(s[2*kk][1] - ua);
                const float p02 = exp2f(s[2*kk][2] - ub),     p03 = exp2f(s[2*kk][3] - ub);
                const float p10 = exp2f(s[2*kk + 1][0] - ua), p11 = exp2f(s[2*kk + 1][1] - ua);
                const float p12 = exp2f(s[2*kk + 1][2] - ub), p13 = exp2f(s[2*kk + 1][3] - ub);
                la += p00 + p01 + p10 + p11;
                lb += p02 + p03 + p12 + p13;
                pa[kk][0] = gq_h2u(p00, p01);
                pa[kk][1] = gq_h2u(p02, p03);
                pa[kk][2] = gq_h2u(p10, p11);
                pa[kk][3] = gq_h2u(p12, p13);
            }

#pragma unroll
            for (int kk = 0; kk < GQ_BN/16; ++kk) {
                const half * vr = V + (16*kk + (lane & 7) + ((lane >> 3) & 1)*8)*GQ_LD + (lane >> 4)*8;
#pragma unroll
                for (int j = 0; j < GQ_D/8; j += 2) {
                    uint32_t b0, b1, b2, b3;
                    gq_ldsm4_t(b0, b1, b2, b3, vr + 8*j);
                    gq_mma(o[j],     pa[kk], b0, b1);
                    gq_mma(o[j + 1], pa[kk], b2, b3);
                }
            }
        }
        __syncthreads(); // buffer buf is refilled by the next iteration's prefetch
    }

    la += __shfl_xor_sync(0xFFFFFFFF, la, 1);
    la += __shfl_xor_sync(0xFFFFFFFF, la, 2);
    lb += __shfl_xor_sync(0xFFFFFFFF, lb, 1);
    lb += __shfl_xor_sync(0xFFFFFFFF, lb, 2);

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        if (!(i == 0 ? va_ : vb_)) {
            continue;
        }
        const int t = i == 0 ? ia : ib;
        if (p.split > 1) {
            const int64_t row = ((int64_t) z*p.T + t)*p.H + h;
            float * po = p.part_o + row*GQ_D;
#pragma unroll
            for (int j = 0; j < GQ_D/8; ++j) {
                *reinterpret_cast<float2 *>(po + 8*j + 2*tq) = make_float2(o[j][2*i], o[j][2*i + 1]);
            }
            if (tq == 0) {
                p.part_ml[row] = make_float2(i == 0 ? ma : mb, i == 0 ? la : lb);
            }
            continue;
        }
        float f, lse;
        gq_row_final(p, t, h, i == 0 ? ma : mb, i == 0 ? la : lb, f, lse);
        float * out = (float *) (p.O + (int64_t) t*p.nbo2 + (int64_t) h*p.nbo1);
        const float * gt = p.gate ? p.gate + (int64_t) t*p.gate_t + (int64_t) h*GQ_D : nullptr;
#pragma unroll
        for (int j = 0; j < GQ_D/8; ++j) {
            const int d = 8*j + 2*tq;
            float2 r = make_float2(o[j][2*i]*f, o[j][2*i + 1]*f);
            if (gt) {
                r.x *= 1.0f / (1.0f + expf(-gt[d]));
                r.y *= 1.0f / (1.0f + expf(-gt[d + 1]));
            }
            *reinterpret_cast<float2 *>(out + d) = r;
        }
        if (p.lse && tq == 0) {
            p.lse[(int64_t) t*p.H + h] = lse;
        }
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
}

// merge of the key splits: one warp per (row, head), 4 dims per lane
static __global__ void inkgqa_combine(const gq_params p) {
    const int64_t row = (int64_t) blockIdx.x*blockDim.y + threadIdx.y; // t*H + h
    if (row >= (int64_t) p.T*p.H) {
        return;
    }
    const int lane = threadIdx.x;
    const int t = (int) (row / p.H);
    const int h = (int) (row % p.H);
    const int64_t plane = (int64_t) p.T*p.H;
    float M = -INFINITY;
    for (int z = 0; z < p.split; ++z) {
        M = fmaxf(M, p.part_ml[z*plane + row].x);
    }
    float L = 0.0f;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (M != -INFINITY) {
        for (int z = 0; z < p.split; ++z) {
            const float2 ml = p.part_ml[z*plane + row];
            if (ml.x == -INFINITY) {
                continue;
            }
            const float w = exp2f(ml.x - M);
            L += w*ml.y;
            const float4 v = reinterpret_cast<const float4 *>(p.part_o + (z*plane + row)*GQ_D)[lane];
            acc.x += w*v.x; acc.y += w*v.y; acc.z += w*v.z; acc.w += w*v.w;
        }
    }
    float f, lse;
    gq_row_final(p, t, h, M, L, f, lse);
    float * out = (float *) (p.O + (int64_t) t*p.nbo2 + (int64_t) h*p.nbo1);
    float4 r = make_float4(acc.x*f, acc.y*f, acc.z*f, acc.w*f);
    if (p.gate) {
        const float * gt = p.gate + (int64_t) t*p.gate_t + (int64_t) h*GQ_D + 4*lane;
        r.x *= 1.0f / (1.0f + expf(-gt[0]));
        r.y *= 1.0f / (1.0f + expf(-gt[1]));
        r.z *= 1.0f / (1.0f + expf(-gt[2]));
        r.w *= 1.0f / (1.0f + expf(-gt[3]));
    }
    reinterpret_cast<float4 *>(out)[lane] = r;
    if (p.lse && lane == 0) {
        p.lse[row] = lse;
    }
}

static int gq_env_int(const char * name) {
    const char * e = getenv(name);
    return e && *e ? atoi(e) : 0;
}

static bool gq_env_on(const char * name) {
    return gq_env_int(name) != 0;
}

static size_t gq_smem(const int RW, const int mode) {
    return GQ_SMEM_KV + (mode == GQ_FACTOR ? (size_t) GQ_G*RW*16*GQ_GLD*sizeof(float) : 0);
}

static bool gq_aligned_t(const ggml_tensor * t, const int64_t n) {
    const uintptr_t off = t->data ? reinterpret_cast<uintptr_t>(t->data) : (t->view_src ? (uintptr_t) t->view_offs : 0);
    return off % n == 0 && t->nb[1] % n == 0 && t->nb[2] % n == 0;
}

template <int RW, int MODE>
static void gq_launch_rw(ggml_backend_cuda_context & ctx, gq_params & p) {
    constexpr int NW = GQ_G*RW;
    const size_t smem = gq_smem(RW, MODE);
    static bool smem_set[GGML_CUDA_MAX_DEVICES] = {false};
    if (!smem_set[ctx.device]) {
        CUDA_CHECK(cudaFuncSetAttribute(inkgqa_fa<RW, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem));
        smem_set[ctx.device] = true;
    }
    const int n_rb = (p.T + 16*RW - 1)/(16*RW);
    inkgqa_fa<RW, MODE><<<dim3(p.Hkv, n_rb, p.split), NW*WARP_SIZE, smem, ctx.stream()>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

// tile lists, key split, kernel, combine; p.M / m_t / T / n_kv / H / Hkv set
template <int MODE>
static void gq_run(ggml_backend_cuda_context & ctx, gq_params & p) {
    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    static const int rw_env    = gq_env_int("GGML_CUDA_INKGQA_RW");
    static const int split_env = gq_env_int("GGML_CUDA_INKGQA_SPLIT");
    // 4 row warps (12 warps, 64 rows) when the register budget allows (no bias arrays), 2 for the bias forms
    const int RW = rw_env == 2 || rw_env == 4 ? rw_env : (MODE == GQ_PLAIN ? 4 : 2);
    const int BM = 16*RW;
    const int n_rb = (p.T + BM - 1)/BM;
    p.NT = (p.n_kv + GQ_BN - 1)/GQ_BN;

    ggml_cuda_pool_alloc<uint32_t> tiles(ctx.pool(), (size_t) n_rb*p.NT + n_rb);
    ggml_cuda_pool_alloc<uint8_t>  flags(ctx.pool(), (size_t) n_rb*p.NT);
    p.tiles   = tiles.get();
    p.n_tiles = (const int32_t *) (tiles.get() + (size_t) n_rb*p.NT);
    if (RW == 4) {
        inkgqa_tile_flags<64><<<dim3(p.NT, n_rb), 256, 0, stream>>>(p.M, p.m_t, p.mp, p.mp_t, p.T, p.n_kv, p.NT, flags.get());
    } else {
        inkgqa_tile_flags<32><<<dim3(p.NT, n_rb), 256, 0, stream>>>(p.M, p.m_t, p.mp, p.mp_t, p.T, p.n_kv, p.NT, flags.get());
    }
    CUDA_CHECK(cudaGetLastError());
    inkgqa_tile_lists<<<n_rb, WARP_SIZE, 0, stream>>>(flags.get(), p.NT, (uint32_t *) p.tiles, (int32_t *) p.n_tiles);
    CUDA_CHECK(cudaGetLastError());

    // key split: about 3 waves of 1-block-per-SM blocks, at least 8 tiles per split of a full row block
    int split = split_env > 0 ? split_env : (int) ((3*nsm + (int64_t) p.Hkv*n_rb - 1)/((int64_t) p.Hkv*n_rb));
    split = std::max(1, std::min({split, 16, std::max(1, p.NT/8)}));
    p.split = split;
    ggml_cuda_pool_alloc<float>  part_o(ctx.pool());
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool());
    if (split > 1) {
        p.part_o  = part_o.alloc((size_t) split*p.T*p.H*GQ_D);
        p.part_ml = part_ml.alloc((size_t) split*p.T*p.H);
    } else {
        p.part_o  = nullptr;
        p.part_ml = nullptr;
    }

    if (RW == 4) {
        gq_launch_rw<4, MODE>(ctx, p);
    } else {
        gq_launch_rw<2, MODE>(ctx, p);
    }
    if (split > 1) {
        const int64_t rows = (int64_t) p.T*p.H;
        inkgqa_combine<<<(unsigned) ((rows + 7)/8), dim3(WARP_SIZE, 8), 0, stream>>>(p);
        CUDA_CHECK(cudaGetLastError());
    }
}

static bool gq_device_ok(const int device) {
    const int cc = ggml_cuda_info().devices[device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ampere_mma_available(cc) &&
        ggml_cuda_info().devices[device].smpbo >= gq_smem(4, GQ_FACTOR);
}

// common shape checks of the Q / K / V / mask / dst tensors (no data pointers needed)
static bool gq_check_qkv(const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const bool packed = mask && ggml_flash_attn_ext_mask_packed(mask);
    if (dst->type != GGML_TYPE_F32 || Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 ||
            !mask || (mask->type != GGML_TYPE_F16 && !packed)) {
        return false;
    }
    float max_bias, softcap;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    const int64_t T = Q->ne[1], H = Q->ne[2], Hkv = K->ne[2], n_kv = K->ne[1];
    if (Q->ne[0] != GQ_D || K->ne[0] != GQ_D || V->ne[0] != GQ_D || V->ne[1] != n_kv || V->ne[2] != Hkv || n_kv < 1 ||
            Hkv < 1 || H != GQ_G*Hkv || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || T < 64 || (T + 31)/32 > 65535 ||
            Hkv > 65535 || n_kv > INT32_MAX/2) {
        return false;
    }
    if ((packed ? mask->ne[0]*32 < n_kv || mask->nb[0] != sizeof(uint32_t) || mask->nb[1] % sizeof(uint32_t) != 0 :
                  mask->ne[0] < n_kv || mask->nb[0] != sizeof(half) || mask->nb[1] % sizeof(half) != 0) ||
            mask->ne[1] < T || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    if (Q->nb[0] != sizeof(float) || !gq_aligned_t(Q, 8) || K->nb[0] != sizeof(half) || !gq_aligned_t(K, 16) ||
            V->nb[0] != sizeof(half) || !gq_aligned_t(V, 16) ||
            dst->nb[0] != sizeof(float) || dst->ne[0] != GQ_D || dst->ne[1] != H || dst->ne[2] != T || dst->ne[3] != 1 ||
            !gq_aligned_t(dst, 16)) {
        return false;
    }
    return true;
}

static void gq_fill_qkv(gq_params & p, const ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    memset(&p, 0, sizeof(p));
    p.Q = (const char *) Q->data; p.nbq1 = Q->nb[1]; p.nbq2 = Q->nb[2];
    p.K = (const char *) K->data; p.nbk1 = K->nb[1]; p.nbk2 = K->nb[2];
    p.V = (const char *) V->data; p.nbv1 = V->nb[1]; p.nbv2 = V->nb[2];
    if (ggml_flash_attn_ext_mask_packed(mask)) {
        p.mp = (const uint32_t *) mask->data; p.mp_t = mask->nb[1]/sizeof(uint32_t);
    } else {
        p.M = (const half *) mask->data; p.m_t = mask->nb[1]/sizeof(half);
        p.m_al4 = (reinterpret_cast<uintptr_t>(mask->data) % 4) == 0 && mask->nb[1] % 4 == 0;
    }
    p.O = (char *) dst->data; p.nbo1 = dst->nb[1]; p.nbo2 = dst->nb[2];
    p.T = (int) Q->ne[1]; p.H = (int) Q->ne[2]; p.Hkv = (int) K->ne[2]; p.n_kv = (int) K->ne[1];
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
}

static bool gq_inkrel_check(const int device, const ggml_tensor * dst) {
    if (!ggml_flash_attn_ext_is_inkrel(dst) || !gq_device_ok(device) || !gq_check_qkv(dst) ||
            dst->src[3]->type != GGML_TYPE_F16) {
        return false;
    }
    const bool factor = ggml_flash_attn_ext_is_inkrel_factor(dst);
    if (!factor && !ggml_cuda_inkgqa_swa_enabled()) {
        return false;
    }
    if (dst->src[6] != nullptr || dst->src[9] != nullptr) {
        return false;
    }
    for (int i = 4; i < GGML_MAX_OP_PARAMS / (int) sizeof(int32_t); ++i) {
        if (ggml_get_op_params_i32(dst, i) != 0 && i != GGML_INKREL_PARAM && !(factor && i == GGML_INKREL_F_PARAM)) {
            return false;
        }
    }
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * snk  = dst->src[4];
    const ggml_tensor * tab  = ggml_flash_attn_ext_inkrel_table(dst);
    const ggml_tensor * rp   = ggml_flash_attn_ext_inkrel_pos(dst);
    const ggml_tensor * gate = ggml_flash_attn_ext_inkrel_gate(dst);
    const int64_t T = Q->ne[1], H = Q->ne[2], n_kv = K->ne[1];
    if (rp->type != GGML_TYPE_I32 || rp->nb[0] != sizeof(int32_t) || rp->nb[1] % sizeof(int32_t) != 0 ||
            rp->ne[0] != n_kv || rp->ne[1] != T || rp->ne[2] != 1 || rp->ne[3] != 1 || H > 256) {
        return false;
    }
    if (factor) {
        if (tab->type != GGML_TYPE_F32 || !ggml_is_contiguous(tab) || tab->ne[0] != GQ_NREL || tab->ne[1] <= H*T ||
                tab->ne[2] != 1 || tab->ne[3] != 1 || tab->ne[1] - H*T > INT32_MAX/2 ||
                !snk || snk->type != GGML_TYPE_F32 || !ggml_is_contiguous(snk) || snk->ne[0] != H || snk->ne[1] != T) {
            return false;
        }
    } else {
        if (tab->type != GGML_TYPE_F32 || !ggml_is_contiguous(tab) || tab->ne[0] != H || tab->ne[1] < 1 ||
                tab->ne[1] > INT32_MAX || tab->ne[2] != T || tab->ne[3] != 1 ||
                (snk && (snk->type != GGML_TYPE_F32 || !ggml_is_contiguous(snk) || snk->ne[0] != H || ggml_nelements(snk) != H))) {
            return false;
        }
    }
    if (gate && (gate->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate) || gate->ne[0] != GQ_D*H || gate->ne[1] != T)) {
        return false;
    }
    return true;
}

} // namespace

bool ggml_cuda_inkgqa_far_enabled() {
    static const bool on = gq_env_on("GGML_CUDA_INKGQA_FAR");
    return on;
}

bool ggml_cuda_inkgqa_swa_enabled() {
    static const bool on = gq_env_on("GGML_CUDA_INKGQA_SWA");
    return on;
}

bool ggml_cuda_inkgqa_direct_enabled() {
    static const bool on = gq_env_on("GGML_CUDA_INKGQA_DIRECT");
    return on;
}

bool ggml_cuda_inkgqa_fa_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!ggml_cuda_inkgqa_far_enabled() || dst->op != GGML_OP_FLASH_ATTN_EXT || ggml_flash_attn_ext_is_inkrel(dst) ||
            !gq_device_ok(ctx.device) || !gq_check_qkv(dst)) {
        return false;
    }
    for (int i = 5; i <= 8; ++i) {
        if (dst->src[i] != nullptr) {
            return false;
        }
    }
    for (int i = 4; i < GGML_MAX_OP_PARAMS / (int) sizeof(int32_t); ++i) {
        if (ggml_get_op_params_i32(dst, i) != 0) {
            return false; // n_kv_max, SJ-KVaRN, rotation, LSA and marker slots
        }
    }
    if (ggml_flash_attn_ext_get_kv_rows(dst) != nullptr || ggml_flash_attn_ext_get_lsa_vk(dst) ||
            ggml_flash_attn_ext_get_lsa_block(dst) > 0) {
        return false;
    }
    const ggml_tensor * snk = dst->src[4];
    ggml_tensor * lse = ggml_flash_attn_ext_get_lse(dst);
    if (dst->src[9] != nullptr && lse == nullptr) {
        return false;
    }
    const int64_t T = dst->src[0]->ne[1], H = dst->src[0]->ne[2];
    if ((snk && (snk->type != GGML_TYPE_F32 || !ggml_is_contiguous(snk) || snk->ne[0] != H)) ||
            (lse && (lse->type != GGML_TYPE_F32 || !ggml_is_contiguous(lse) || lse->ne[0] != H || lse->ne[1] != T ||
                     ggml_nelements(lse) != H*T))) {
        return false;
    }
    gq_params p;
    gq_fill_qkv(p, dst);
    p.sinks = snk ? (const float *) snk->data : nullptr;
    p.lse   = lse ? (float *) lse->data : nullptr;
    gq_run<GQ_PLAIN>(ctx, p);
    static bool noted = false;
    if (!noted) {
        noted = true;
        GGML_LOG_WARN("[INKGQA] GQA-3 D128 FA: T %d, n_kv %d, %d KV heads, lse %d\n", p.T, p.n_kv, p.Hkv, lse != nullptr);
    }
    return true;
}

bool ggml_cuda_inkgqa_inkrel_supported(int device, const ggml_tensor * dst) {
    return gq_inkrel_check(device, dst);
}

bool ggml_cuda_inkgqa_inkrel_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!gq_inkrel_check(ctx.device, dst)) {
        return false;
    }
    const bool factor = ggml_flash_attn_ext_is_inkrel_factor(dst);
    const ggml_tensor * snk  = dst->src[4];
    const ggml_tensor * tab  = ggml_flash_attn_ext_inkrel_table(dst);
    const ggml_tensor * rp   = ggml_flash_attn_ext_inkrel_pos(dst);
    const ggml_tensor * gate = ggml_flash_attn_ext_inkrel_gate(dst);
    gq_params p;
    gq_fill_qkv(p, dst);
    p.gate   = gate ? (const float *) gate->data : nullptr;
    p.gate_t = gate ? gate->nb[1]/sizeof(float) : 0;
    p.sinks  = snk ? (const float *) snk->data : nullptr;
    if (factor) {
        p.rf       = (const float *) tab->data;
        p.extent   = (int) (tab->ne[1] - (int64_t) p.H*p.T);
        p.sinks_row = 1;
        p.b.table = nullptr;
        p.b.rel   = (const int32_t *) rp->data;
        p.b.r_t   = rp->nb[1]/sizeof(int32_t);
        p.b.n_e   = p.extent + 1;
        p.b.H     = p.H;
        p.b.n_kv  = rp->ne[0];
        p.b.T     = rp->ne[1];
        gq_run<GQ_FACTOR>(ctx, p);
    } else {
        if (!ink_rel_bias_from_inkrel(dst, p.b)) {
            return false;
        }
        gq_run<GQ_TABLE>(ctx, p);
    }
    static bool noted[2] = {false, false};
    if (!noted[factor]) {
        noted[factor] = true;
        GGML_LOG_WARN("[INKGQA] INKREL %s form: T %d, n_kv %d, %d KV heads\n", factor ? "factor" : "table", p.T, p.n_kv, p.Hkv);
    }
    return true;
}
