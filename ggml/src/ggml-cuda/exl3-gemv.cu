// EXL3 format, "mul1" codebook arithmetic, tile layout and trellis GEMV design: Turboderp, exllamav3
// (https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c) 2025 Turboderp; see
// licenses/LICENSE-exllamav3. EXL3 is Turboderp's streamlined variant of QTIP (Tseng, Sun, Hou, De Sa,
// "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024, arXiv:2406.11235).
// This file is an independent ggml/CUDA reimplementation written with the exllamav3 source as reference;
// no QTIP code was used.
#include "exl3.cuh"

#include "ggml-ledger.h"

#include <algorithm>
#include <atomic>
#include <cstring>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define EXL3_GEMV_HAVE_COOP 1
#else
#define EXL3_GEMV_HAVE_COOP 0
#endif

// EXL3 M2/M3: decode-width GEMV/GEMM (T <= EXL3_GEMV_MAX_T activation columns) straight from the trellis stream.
//
// Data format (exllamav3, MIT; see exl3.cu for the codebook): the weight [K, N] is stored as 16x16 tiles in
// k-tile-major order [K/16][N/16][8*bits words]. Inside a tile the 256 weights follow the tensor-core lane order:
// lane t (0..31), slot j (0..7) -> row k = 2*(t&3) + (j&1) + 8*((j>>1)&1), col n = (t>>2) + 8*(j>>2).
// Weight i of the tile is the 16-bit window ending at bit (i+1)*bits of the tail-biting bit stream (MSB-first
// inside 32-bit words), decoded through the "mul1" codebook. That lane order is exactly the B operand layout of
// mma.m16n8k16 (two n8 halves), which the T >= EXL3_GEMV_MMA_MIN_T path uses with x as the A operand.
//
// Kernel: persistent blocks of 4 warps walk work items (n-tile, k-split). A warp walks k-tiles of one n-tile.
// Loads: at <= 4 bits a tile is <= 32 words, so every lane loads one word (one coalesced 128 B request per warp
// per tile) and the two words covering its eight windows come from lane shuffles; at 5..8 bits each lane loads
// its three words directly. The eight codes are decoded (dp4a codebook) into four half2 row pairs and either
// FMAed against x in half2 (T = 1: 4 HFMA2 per tile) or fed as the B fragments of two mma.m16n8k16 with fp16
// accumulation folded to fp32 every U tiles (T >= 2: cost independent of T up to 8). Warps reduce through shared
// memory; k-splits reduce in the glue_out kernel. Summation order is fixed => deterministic.
//
// M4 (FUSED): the same kernel launched cooperatively does the glue in-kernel -- phase 1 writes the fp16 staging
// buffer xh = scale * H128(suh * x) (grid-strided over 128-chunks), grid barrier, phase 2 = the tile loop into the
// split-K partials, grid barrier, phase 3 reduces the partials, applies H128 and svh and writes y. Two ~1 us grid
// barriers replace two glue kernels and two graph nodes per matmul. Measured 2026-09-17: the main GEMV pays ~10.6 us
// per call for the barriers on the 1008-block persistent grid, more than the glue costs, so the split-glue path stays
// the default; GGML_CUDA_EXL3_FUSED=1 opts in, EXL3_GEMV_BPS caps the cooperative grid at n blocks per SM.

// [#73] weight-major mma (GPT-6 SM86 PTX kit C1; on by default, GGML_CUDA_EXL3_WEIGHT_MAJOR=0 turns it off):
// the decoded weight tile is the A operand ({wv0, wv2, wv1, wv3}: A rows = the 16 output columns, A cols = k) and x
// the B operand (tokens on n8), so one mma.m16n8k16 per 16x16 tile instead of two with x as A, whose rows 8-15 are
// zero at T <= 8. D rows are outputs, D columns tokens. T = 2..8, split glue (not FUSED) only: 3- and 4-bit in
// k_exl3_gemv, every bit width in the #41 k_exl3_gemv_dec (output bits identical to the x-as-A mma, measured on SM86 by
// test-exl3-decode and test-exl3-weight-major; PTX does not specify the mma's internal summation order); the
// split-K, the fold schedule and the reduction are the ones above. -DLLAMAMPERE_EXL3_WEIGHT_MAJOR=0 leaves the
// variant out of the build.
#ifndef LLAMAMPERE_EXL3_WEIGHT_MAJOR
#define LLAMAMPERE_EXL3_WEIGHT_MAJOR 1
#endif
#define EXL3_GEMV_NWARPS 4
#define EXL3_GEMV_MAX_T  16
#ifndef EXL3_GEMV_MMA_MIN_T
#define EXL3_GEMV_MMA_MIN_T 2
#endif

// x is staged in shared memory as fp16, scaled by 1/16. That scale does not by itself keep the fp16 partial sums
// finite. The largest mul1 codebook value is |w| = 3.453125 (exhaustive over the 65,536 codes), and the partials are:
//   - mma path (T >= EXL3_GEMV_MMA_MIN_T): D is fp16 across U = 4 k-tiles before the fp32 fold, 64 products,
//     so it is guaranteed finite for |x/16| <= 65504 / (64 * 3.453125) = 296.4 (|x| <= 4742 after suh and H128);
//   - HFMA2 path (T = 1): 4 products per fp16 partial, finite for |x/16| <= 4742;
//   - the fp16 staging itself holds |x/16| <= 65504.
// Above those bounds a partial can still be finite (signs cancel, most products are far below the maximum), so the
// bounds are sufficient, not necessary. GGML_CUDA_EXL3_ENVELOPE=1 measures the real staged maximum, how many staged
// values exceed the path's bound and how many outputs are inf/nan (see exl3_envelope_* below).
#define EXL3_GEMV_X_SCALE    0.0625f
#define EXL3_CB_ABS_MAX      3.453125f
#define EXL3_GEMV_SAFE_X_MMA (65504.0f / (64.0f * EXL3_CB_ABS_MAX))
#define EXL3_GEMV_SAFE_X_T1  (65504.0f / ( 4.0f * EXL3_CB_ABS_MAX))
template <int T> struct exl3_gemv_cfg {
    static constexpr bool MMA        = T >= EXL3_GEMV_MMA_MIN_T;
    static constexpr int  XROWS      = MMA ? (T <= 8 ? 8 : 16) : T;  // staged x rows (mma: rows T..XROWS-1 stay zero)
    static constexpr int  KTILES_MAX = XROWS <= 2 ? 128 : (XROWS <= 8 ? 64 : 32);   // k-tiles per item (XROWS*KTILES_MAX*32 B)
};

static __device__ __forceinline__ uint32_t exl3_h2u(const half2 v) { return *reinterpret_cast<const uint32_t *>(&v); }
static __device__ __forceinline__ half2    exl3_u2h(const uint32_t v) { return *reinterpret_cast<const half2 *>(&v); }

// d[16x8] += a[16x16] . b[16x8], all fp16 (row-major a, "col" b: the trellis lane order)
static __device__ __forceinline__ void exl3_mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int BITS, int T, bool FUSED, bool WM = false>
static __global__ void __launch_bounds__(32 * EXL3_GEMV_NWARPS)
k_exl3_gemv(const uint32_t * __restrict__ trellis, half2 * __restrict__ x, float * __restrict__ y,
            const int kt, const int nt, const int K, const int N, const int ksplit, const int kpi,
            const float * __restrict__ xf, const float * __restrict__ suh, const float * __restrict__ svh,
            float * __restrict__ yf, const int post) {
    using cfg = exl3_gemv_cfg<T>;
    constexpr bool MMA        = cfg::MMA;
    constexpr bool WEIGHT_MAJOR = WM && MMA && T >= 2 && T <= 8 && (BITS == 3 || BITS == 4) && !FUSED;
    constexpr int  XROWS      = cfg::XROWS;
    constexpr int  KTILES_MAX = cfg::KTILES_MAX;
    constexpr int  NW    = 8 * BITS;       // words per tile
    constexpr int  NBITS = 256 * BITS;     // bits per tile
    // a lane's eight windows span rel0 + 7*BITS + 16 bits with rel0 = ((8*lane+1)*BITS - 16) & 31; max over the
    // lanes: 48 (2b), 64 (3b, 4b), 80 (5b, 6b), 96 (7b, 8b) => 2 words up to 4 bits, 3 words above
    constexpr int  NWORDS = BITS <= 4 ? 2 : 3;
    constexpr bool SHFL   = BITS <= 4;     // one word per lane per tile, windows resolved by shuffles
    constexpr int  U      = 4;             // k-tiles of weight words in flight per warp (= fp16 fold period)
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q    = lane & 3;             // row pair selector
    const int g    = lane >> 2;            // column (and column + 8); mma: A/D row

    // The lane's eight windows start at bit p0 + j*BITS (MSB-first, tail-biting) of the tile stream. Take the
    // NWORDS words covering [ws*32, ...) and shift them left by rel0 = p0 - ws*32 so that window j starts at the
    // compile-time bit offset j*BITS of the shifted stream.
    const int p0   = ((lane * 8 + 1) * BITS - 16 + NBITS) % NBITS;
    const int ws   = p0 >> 5;
    const int rel0 = p0 & 31;
    int woff[NWORDS];                      // word index inside the tile == source lane in the SHFL path
#pragma unroll
    for (int m = 0; m < NWORDS; ++m) {
        woff[m] = (ws + m) % NW;
    }
    const int    own_w  = SHFL ? (lane % NW) : 0;                              // word this lane loads (SHFL)
    const size_t stride = (size_t) nt * NW;                                    // words per k-tile row

    const half2 cb_mul = __halves2half2(__ushort_as_half((unsigned short) 0x1eee), __ushort_as_half((unsigned short) 0x1eee));
    const half2 cb_add = __halves2half2(__ushort_as_half((unsigned short) 0xc931), __ushort_as_half((unsigned short) 0xc931));

    // weight-major writes part[.][2q + {0,1}][g + {0,8}]: a row pitch of 20 floats puts the 32 lanes of a store on
    // banks 8q + g (+ const), conflict-free; the column-sum reader below does not care about the pitch
    __shared__ float part[EXL3_GEMV_NWARPS][T][WEIGHT_MAJOR ? 20 : 16];
    __shared__ __align__(16) half2 xs[XROWS][KTILES_MAX * 8];   // fp16 x of the item's k-range, as row pairs

    if (MMA && T < XROWS) {   // A rows T..7 are zero for every item (the barrier of the first staging orders this)
        half2 * z = &xs[T][0];
        for (int p = threadIdx.x; p < (XROWS - T) * KTILES_MAX * 8; p += 32 * EXL3_GEMV_NWARPS) {
            z[p] = __float2half2_rn(0.0f);
        }
    }

    // eight codes of a tile -> four half2 row pairs: wv[jp] = (slot 2jp, slot 2jp+1) = rows (2q, 2q+1) [jp even]
    // or (2q+8, 2q+9) [jp odd] of column g + 8*(jp>>1)
    auto decode = [&](const uint32_t (&w)[NWORDS], half2 (&wv)[4]) {
        uint32_t a[NWORDS];
#pragma unroll
        for (int m = 0; m < NWORDS; ++m) {
            a[m] = __funnelshift_l(m + 1 < NWORDS ? w[m + 1] : 0u, w[m], rel0);
        }
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
            uint32_t sum[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int o  = (2 * jp + e) * BITS;           // compile-time after unrolling
                const int wi = o >> 5;
                const int sh = o & 31;
                const uint32_t hi = a[wi];
                const uint32_t lo = (wi + 1 < NWORDS) ? a[wi + 1] : 0u;
                uint32_t code;
                if (sh == 0) {
                    code = hi >> 16;
                } else if (sh == 8) {
                    code = __byte_perm(hi, 0u, 0x4421);
                } else if (sh == 16) {
                    code = hi & 0xffffu;
                } else {
                    code = __funnelshift_l(lo, hi, sh) >> 16;
                }
                sum[e] = __dp4a(code * 0x83DCD12Du, 0x01010101u, 0x6400u);
            }
            wv[jp] = __hfma2(__halves2half2(__ushort_as_half((unsigned short) sum[0]),
                                            __ushort_as_half((unsigned short) sum[1])), cb_mul, cb_add);
        }
    };

#if EXL3_GEMV_HAVE_COOP
    const int gw  = blockIdx.x * EXL3_GEMV_NWARPS + warp;   // grid-wide warp index (fused glue phases)
    const int gws = gridDim.x * EXL3_GEMV_NWARPS;
    if (FUSED) {
        // phase 1: xh[T][K] = X_SCALE * H128(suh * x)   (suh null: X_SCALE * x); one warp per 128-chunk
        half * xh = reinterpret_cast<half *>(x);
        const int n_chunks = T * K / 128;
        for (int c = gw; c < n_chunks; c += gws) {
            const size_t base = (size_t) c * 128;
            const int kc = (int) (base % (size_t) K);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[j] = xf[base + j * 32 + lane];
                if (suh != nullptr) {
                    v[j] *= suh[kc + j * 32 + lane];
                }
            }
            float sc = EXL3_GEMV_X_SCALE;
            if (suh != nullptr) {
                exl3_wht128(v);
                sc *= EXL3_WHT128_SCALE;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                xh[base + j * 32 + lane] = __float2half_rn(v[j] * sc);
            }
        }
        cg::this_grid().sync();
    }
#endif

    // persistent: each block walks work items (n-tile, k-split) with a grid stride
    const int n_items = nt * ksplit;
    for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        const int ks = item / nt;                                               // 32-bit: item < 2^31
        const int ni = item - ks * nt;
        const int kb = ks * kpi;
        const int nk = min(kpi, kt - kb);                                       // <= KTILES_MAX

        // stage x[t][kb*16 .. (kb+nk)*16) (fp16 pairs, pre-scaled by EXL3_GEMV_X_SCALE)
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const half2 * xt = x + (size_t) t * (K / 2) + (size_t) kb * 8;
            for (int p = threadIdx.x; p < nk * 8; p += 32 * EXL3_GEMV_NWARPS) {
                xs[t][p] = FUSED ? __ldcg(xt + p) : xt[p];
            }
        }
        __syncthreads();

        const uint32_t * wp = trellis + ((size_t) (kb + warp) * nt + ni) * NW;

        float    acc[2][T];      // T = 1..: fp32 per column slot (HFMA2 path)
        uint32_t d[2][2];        // mma: fp16 D fragments per n8 half
        float2   accf[2];        // mma: fp32 fold of D row g per n8 half
        float2   accg[2];        // mma, XROWS == 16: fp32 fold of D row g + 8 per n8 half
#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[s][t] = 0.0f;
            }
            d[s][0] = 0u; d[s][1] = 0u;
            accf[s] = make_float2(0.0f, 0.0f);
            accg[s] = make_float2(0.0f, 0.0f);
        }

        auto tile_fma = [&](const uint32_t (&w)[NWORDS], const int kl) {
            half2 wv[4];
            decode(w, wv);
            if (MMA) {
                uint32_t a[4];
                a[0] = exl3_h2u(xs[g][kl * 8 + q]);          // A[g][2q, 2q+1]
                a[1] = XROWS > 8 ? exl3_h2u(xs[(XROWS > 8 ? g + 8 : 0)][kl * 8 + q]) : 0u;       // A[g+8][2q, 2q+1]
                a[2] = exl3_h2u(xs[g][kl * 8 + q + 4]);      // A[g][2q+8, 2q+9]
                a[3] = XROWS > 8 ? exl3_h2u(xs[(XROWS > 8 ? g + 8 : 0)][kl * 8 + q + 4]) : 0u;   // A[g+8][2q+8, 2q+9]
                if constexpr (WEIGHT_MAJOR) {
                    const uint32_t wa[4] = {exl3_h2u(wv[0]), exl3_h2u(wv[2]),
                                           exl3_h2u(wv[1]), exl3_h2u(wv[3])};
                    exl3_mma_f16(d[0], wa, a[0], a[2]);
                } else {
                    exl3_mma_f16(d[0], a, exl3_h2u(wv[0]), exl3_h2u(wv[1]));
                    exl3_mma_f16(d[1], a, exl3_h2u(wv[2]), exl3_h2u(wv[3]));
                }
            } else {
                // x row pairs of this tile for this lane: (2q, 2q+1) and (2q+8, 2q+9)
                half2 xa[T], xb[T];
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    xa[t] = xs[t][kl * 8 + q];
                    xb[t] = xs[t][kl * 8 + q + 4];
                }
                half2 acc2[2][T];
#pragma unroll
                for (int s = 0; s < 2; ++s) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        acc2[s][t] = __float2half2_rn(0.0f);
                    }
                }
#pragma unroll
                for (int jp = 0; jp < 4; ++jp) {
                    const int s = jp >> 1;
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        acc2[s][t] = __hfma2(wv[jp], (jp & 1) ? xb[t] : xa[t], acc2[s][t]);
                    }
                }
#pragma unroll
                for (int s = 0; s < 2; ++s) {
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        acc[s][t] += __half2float(__hadd(__low2half(acc2[s][t]), __high2half(acc2[s][t])));
                    }
                }
            }
        };
        auto fold = [&]() {   // mma: fp16 D -> fp32, restart the fp16 accumulation (same schedule in both orientations)
            if constexpr (WEIGHT_MAJOR) {
                const float2 v0 = __half22float2(exl3_u2h(d[0][0]));
                const float2 v1 = __half22float2(exl3_u2h(d[0][1]));
                accf[0].x += v0.x; accf[0].y += v0.y;
                accf[1].x += v1.x; accf[1].y += v1.y;
                d[0][0] = 0u; d[0][1] = 0u;
            } else if (MMA) {
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    const float2 v = __half22float2(exl3_u2h(d[s][0]));
                    accf[s].x += v.x;
                    accf[s].y += v.y;
                    if (XROWS > 8) {
                        const float2 u = __half22float2(exl3_u2h(d[s][1]));
                        accg[s].x += u.x;
                        accg[s].y += u.y;
                    }
                    d[s][0] = 0u; d[s][1] = 0u;
                }
            }
        };
        // weight words of tile u of a group -> the lane's NWORDS window words
        auto gather = [&](const uint32_t (&own)[U], uint32_t (&w)[U][NWORDS]) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) {
                    if (BITS == 4 && m == 1) {
                        w[u][m] = own[u];                    // 4 bits: ws == lane - 1, word 1 is the lane's own
                    } else {
                        w[u][m] = __shfl_sync(0xffffffffu, own[u], woff[m]);
                    }
                }
            }
        };

        const int n_my = (nk - warp + EXL3_GEMV_NWARPS - 1) / EXL3_GEMV_NWARPS;   // tiles this warp owns
        const size_t sw = (size_t) EXL3_GEMV_NWARPS * stride;                    // words between this warp's tiles
        const uint32_t * wl[NWORDS];
#pragma unroll
        for (int m = 0; m < NWORDS; ++m) wl[m] = wp + (SHFL ? own_w : woff[m]);
        int step = 0;
        for (; step + U <= n_my; step += U) {
            uint32_t w[U][NWORDS];
            if (SHFL) {
                uint32_t own[U];
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    own[u] = __ldcs(wl[0] + (size_t) u * sw);
                }
                gather(own, w);
            } else {
#pragma unroll
                for (int u = 0; u < U; ++u) {
#pragma unroll
                    for (int m = 0; m < NWORDS; ++m) {
                        w[u][m] = __ldcs(wl[m] + (size_t) u * sw);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                tile_fma(w[u], (step + u) * EXL3_GEMV_NWARPS + warp);
            }
            fold();
#pragma unroll
            for (int m = 0; m < NWORDS; ++m) wl[m] += (size_t) U * sw;
        }
        for (; step < n_my; ++step) {
            uint32_t w[NWORDS];
            if (SHFL) {
                const uint32_t own = __ldcs(wl[0]);
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) {
                    w[m] = (BITS == 4 && m == 1) ? own : __shfl_sync(0xffffffffu, own, woff[m]);
                }
            } else {
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) {
                    w[m] = __ldcs(wl[m]);
                }
            }
            tile_fma(w, step * EXL3_GEMV_NWARPS + warp);
            fold();
#pragma unroll
            for (int m = 0; m < NWORDS; ++m) wl[m] += sw;
        }

        if constexpr (WEIGHT_MAJOR) {
            // lane holds D[outputs g, g + 8][tokens 2q, 2q+1]
            if (2*q < T) {
                part[warp][2*q][g] = accf[0].x;
                part[warp][2*q][g + 8] = accf[1].x;
            }
            if (2*q + 1 < T) {
                part[warp][2*q + 1][g] = accf[0].y;
                part[warp][2*q + 1][g + 8] = accf[1].y;
            }
        } else if (MMA) {
            // lane holds D[row g][cols 2q, 2q+1] of both n8 halves
            if (g < T) {
                part[warp][g][2 * q]         = accf[0].x;
                part[warp][g][2 * q + 1]     = accf[0].y;
                part[warp][g][8 + 2 * q]     = accf[1].x;
                part[warp][g][8 + 2 * q + 1] = accf[1].y;
            }
            if (XROWS > 8 && g + 8 < T) {   // D rows 8..15 (d[.][1]) hold x rows g + 8
                const int r = (XROWS > 8) ? g + 8 : 0;
                part[warp][r][2 * q]         = accg[0].x;
                part[warp][r][2 * q + 1]     = accg[0].y;
                part[warp][r][8 + 2 * q]     = accg[1].x;
                part[warp][r][8 + 2 * q + 1] = accg[1].y;
            }
        } else {
            // reduce the four lanes sharing a column
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    float v = acc[s][t];
                    v += __shfl_xor_sync(0xffffffffu, v, 1);
                    v += __shfl_xor_sync(0xffffffffu, v, 2);
                    acc[s][t] = v;
                }
            }
            if (q == 0) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    part[warp][t][g]     = acc[0][t];
                    part[warp][t][g + 8] = acc[1][t];
                }
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 16 * T; i += 32 * EXL3_GEMV_NWARPS) {
            const int t = i >> 4;
            const int n = i & 15;
            float v = 0.0f;
#pragma unroll
            for (int wv = 0; wv < EXL3_GEMV_NWARPS; ++wv) {
                v += part[wv][t][n];
            }
            y[((size_t) ks * T + t) * N + (size_t) ni * 16 + n] = v * (1.0f / EXL3_GEMV_X_SCALE);
        }
        __syncthreads();   // part / xs are reused by the next item
    }

#if EXL3_GEMV_HAVE_COOP
    if (FUSED && post) {
        // phase 3: yf[T][N] = svh * H128( sum_ks part[ks][T][N] )   (svh null: the sum); one warp per 128-chunk
        cg::this_grid().sync();
        const int n_chunks = T * N / 128;
        const size_t ks_stride = (size_t) T * N;
        for (int c = gw; c < n_chunks; c += gws) {
            const size_t base = (size_t) c * 128;
            const int nc = (int) (base % (size_t) N);
            float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int ks = 0;
            for (; ks + 4 <= ksplit; ks += 4) {          // four partial rows in flight, fixed summation order
                const float * p = y + (size_t) ks * ks_stride + base + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float a0 = __ldcg(p + j * 32);
                    const float a1 = __ldcg(p + ks_stride + j * 32);
                    const float a2 = __ldcg(p + 2 * ks_stride + j * 32);
                    const float a3 = __ldcg(p + 3 * ks_stride + j * 32);
                    v[j] += (a0 + a1) + (a2 + a3);
                }
            }
            for (; ks < ksplit; ++ks) {
                const float * p = y + (size_t) ks * ks_stride + base + lane;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    v[j] += __ldcg(p + j * 32);
                }
            }
            float sc = 1.0f;
            if (svh != nullptr) {
                exl3_wht128(v);
                sc = EXL3_WHT128_SCALE;
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float m = (svh != nullptr) ? svh[nc + j * 32 + lane] : 1.0f;
                yf[base + j * 32 + lane] = v[j] * sc * m;
            }
        }
    }
#endif
}

// [#41] EXL3DEC: decode-width GEMV with the same arithmetic as k_exl3_gemv (split glue), restructured for bandwidth.
// Every work item (n-tile ni, k-split ks) computes exactly what k_exl3_gemv computes for it -- same k-split, same
// tile-to-warp assignment, same fp16 fold schedule, same warp and split-K reduction order -- so the output is
// bit-identical (tests/test-exl3-decode.cpp compares the two byte for byte). What changes is the schedule:
//   1. contiguous item ranges: block b walks items [b*n/G, (b+1)*n/G) in k-split-major order, so consecutive items
//      share their k-range and x is staged into shared memory once per k-range instead of once per item. The old
//      grid stride restaged x for every item: at T = 8 that read twice as many bytes from L2 as the weights, which is
//      most of why the GEMV slowed down with T although the mma cost does not depend on T;
//   2. register prefetch: the next group of U weight tiles (of this item, or the first group of the block's next
//      item) is issued before the current group is decoded, so a warp keeps up to 2*U tiles in flight and its load
//      stream does not drain at group or item boundaries; the first group is in flight while x is being staged;
//   3. double-buffered warp partials: one __syncthreads per item instead of two;
//   4. weight-major mma (#73) for every bit width at T = 2..8 (was 3/4-bit only), with x staged for the T real
//      tokens only (B fragments of tokens >= T are register zeros instead of zero rows in shared memory): 2*T KiB of
//      shared memory per block instead of 16 KiB, so more blocks fit per SM.
// GGML_CUDA_EXL3_DEC=0 selects k_exl3_gemv (A/B and the identity test); the cooperative FUSED path is unchanged.
//
// [#41 EXL3W] FR (weight-major only): x is staged in the mma B-fragment order, xf[kl][token][q] = {x[t][16kl + 2q, +1],
// x[t][16kl + 2q + 8, +9]} (one uint2 per lane per tile), instead of row-major [token][k]. The row-major layout put every
// token row on the same banks (row pitch 512 words), so the two 32-bit B loads of a tile were T-way bank conflicts:
// 2*T shared-memory wavefronts per tile per warp, a cost that grows with the verify width and at T = 8 nearly fills the
// shared-memory pipe at full DRAM rate. The fragment layout is one conflict-free 64-bit load (T*32 contiguous bytes).
// Same B values, same mma, same everything else => same output bits. GGML_CUDA_EXL3_VW=0 selects the row-major layout.
// [#41 EXL3W] PF2: two load groups in flight per warp instead of one. The prefetch buffer holds the next 8 tiles as
// two halves; each half is refilled right after it is consumed, so 8 tiles of trellis words are outstanding while 4 are
// computed (before: 4 while 4). At T >= 5 the kernel runs fewer resident warps (shared memory per block grows with T),
// so it needs more bytes in flight per warp to cover DRAM latency. Same tiles, same fold-every-4 schedule, same order
// => same output bits. GGML_CUDA_EXL3_PF2=0 keeps one group in flight.
template <int BITS, int T, bool WM, bool FR = false, bool PF2 = false>
static __global__ void __launch_bounds__(32 * EXL3_GEMV_NWARPS)
k_exl3_gemv_dec(const uint32_t * __restrict__ trellis, const half2 * __restrict__ x, float * __restrict__ y,
                const int kt, const int nt, const int K, const int N, const int ksplit, const int kpi, const int bpk) {
    using cfg = exl3_gemv_cfg<T>;
    constexpr bool MMA          = cfg::MMA;
    constexpr bool WEIGHT_MAJOR = WM && MMA && T >= 2 && T <= 8;
    constexpr int  XROWS        = cfg::XROWS;
    constexpr int  XS_ROWS      = WEIGHT_MAJOR ? T : XROWS;   // staged x rows; weight-major zeros tokens >= T in registers
    constexpr bool FRAG         = FR && WEIGHT_MAJOR;          // B-fragment staging layout (same bytes as xs)
    constexpr int  KTILES_MAX   = cfg::KTILES_MAX;
    constexpr int  NW     = 8 * BITS;
    constexpr int  NBITS  = 256 * BITS;
    constexpr int  NWORDS = BITS <= 4 ? 2 : 3;
    constexpr bool SHFL   = BITS <= 4;
    constexpr int  NL     = SHFL ? 1 : NWORDS;   // words a lane loads per tile
    constexpr int  U      = 4;                   // fold period, tiles per load group (must match k_exl3_gemv)
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q    = lane & 3;
    const int g    = lane >> 2;

    const int p0   = ((lane * 8 + 1) * BITS - 16 + NBITS) % NBITS;
    const int ws   = p0 >> 5;
    const int rel0 = p0 & 31;
    int woff[NWORDS];
#pragma unroll
    for (int m = 0; m < NWORDS; ++m) {
        woff[m] = (ws + m) % NW;
    }
    int loff[NL];                                // word offsets this lane loads inside a tile
#pragma unroll
    for (int m = 0; m < NL; ++m) {
        loff[m] = SHFL ? (lane % NW) : woff[m];
    }
    const size_t sw = (size_t) EXL3_GEMV_NWARPS * nt * NW;   // words between a warp's consecutive tiles

    const half2 cb_mul = __halves2half2(__ushort_as_half((unsigned short) 0x1eee), __ushort_as_half((unsigned short) 0x1eee));
    const half2 cb_add = __halves2half2(__ushort_as_half((unsigned short) 0xc931), __ushort_as_half((unsigned short) 0xc931));

    __shared__ float part[2][EXL3_GEMV_NWARPS][T][WEIGHT_MAJOR ? 20 : 16];
    __shared__ __align__(16) half2 xs[XS_ROWS][KTILES_MAX * 8];
    uint2 * xf = reinterpret_cast<uint2 *>(&xs[0][0]);   // FRAG: [KTILES_MAX][T][4] uint2 = the same T*KTILES_MAX*32 B

    if (MMA && T < XS_ROWS) {   // x-as-A mma: A rows T..XROWS-1 are zero (ordered by the first staging barrier)
        half2 * z = &xs[T < XS_ROWS ? T : 0][0];
        for (int p = threadIdx.x; p < (XS_ROWS - T) * KTILES_MAX * 8; p += 32 * EXL3_GEMV_NWARPS) {
            z[p] = __float2half2_rn(0.0f);
        }
    }

    auto decode = [&](const uint32_t (&w)[NWORDS], half2 (&wv)[4]) {
        uint32_t a[NWORDS];
#pragma unroll
        for (int m = 0; m < NWORDS; ++m) {
            a[m] = __funnelshift_l(m + 1 < NWORDS ? w[m + 1] : 0u, w[m], rel0);
        }
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
            uint32_t sum[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int o  = (2 * jp + e) * BITS;
                const int wi = o >> 5;
                const int sh = o & 31;
                const uint32_t hi = a[wi];
                const uint32_t lo = (wi + 1 < NWORDS) ? a[wi + 1] : 0u;
                uint32_t code;
                if (sh == 0) {
                    code = hi >> 16;
                } else if (sh == 8) {
                    code = __byte_perm(hi, 0u, 0x4421);
                } else if (sh == 16) {
                    code = hi & 0xffffu;
                } else {
                    code = __funnelshift_l(lo, hi, sh) >> 16;
                }
                sum[e] = __dp4a(code * 0x83DCD12Du, 0x01010101u, 0x6400u);
            }
            wv[jp] = __hfma2(__halves2half2(__ushort_as_half((unsigned short) sum[0]),
                                            __ushort_as_half((unsigned short) sum[1])), cb_mul, cb_add);
        }
    };

    float    acc[2][T];
    uint32_t d[2][2];
    float2   accf[2] = {};   // zeroed per item below; the initializer only silences #549-D (lambda capture)
    float2   accg[2];

    auto tile_fma = [&](const uint32_t (&w)[NWORDS], const int kl) {
        half2 wv[4];
        decode(w, wv);
        if constexpr (WEIGHT_MAJOR) {
            // B = x^T: lane (g, q) holds tokens g of k rows (2q, 2q+1) and (2q+8, 2q+9); tokens >= T are zero
            const int gr = g < T ? g : 0;
            uint32_t x0, x1;
            if constexpr (FRAG) {
                const uint2 xb = xf[(kl * T + gr) * 4 + q];
                x0 = xb.x;
                x1 = xb.y;
            } else {
                x0 = exl3_h2u(xs[gr][kl * 8 + q]);
                x1 = exl3_h2u(xs[gr][kl * 8 + q + 4]);
            }
            const uint32_t b0 = (T >= 8 || g < T) ? x0 : 0u;
            const uint32_t b1 = (T >= 8 || g < T) ? x1 : 0u;
            const uint32_t wa[4] = {exl3_h2u(wv[0]), exl3_h2u(wv[2]), exl3_h2u(wv[1]), exl3_h2u(wv[3])};
            exl3_mma_f16(d[0], wa, b0, b1);
        } else if constexpr (MMA) {
            uint32_t a[4];
            a[0] = exl3_h2u(xs[g][kl * 8 + q]);
            a[1] = XROWS > 8 ? exl3_h2u(xs[(XROWS > 8 ? g + 8 : 0)][kl * 8 + q]) : 0u;
            a[2] = exl3_h2u(xs[g][kl * 8 + q + 4]);
            a[3] = XROWS > 8 ? exl3_h2u(xs[(XROWS > 8 ? g + 8 : 0)][kl * 8 + q + 4]) : 0u;
            exl3_mma_f16(d[0], a, exl3_h2u(wv[0]), exl3_h2u(wv[1]));
            exl3_mma_f16(d[1], a, exl3_h2u(wv[2]), exl3_h2u(wv[3]));
        } else {
            half2 xa[T], xb[T];
#pragma unroll
            for (int t = 0; t < T; ++t) {
                xa[t] = xs[t][kl * 8 + q];
                xb[t] = xs[t][kl * 8 + q + 4];
            }
            half2 acc2[2][T];
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    acc2[s][t] = __float2half2_rn(0.0f);
                }
            }
#pragma unroll
            for (int jp = 0; jp < 4; ++jp) {
                const int s = jp >> 1;
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    acc2[s][t] = __hfma2(wv[jp], (jp & 1) ? xb[t] : xa[t], acc2[s][t]);
                }
            }
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    acc[s][t] += __half2float(__hadd(__low2half(acc2[s][t]), __high2half(acc2[s][t])));
                }
            }
        }
    };
    auto fold = [&]() {
        if constexpr (WEIGHT_MAJOR) {
            const float2 v0 = __half22float2(exl3_u2h(d[0][0]));
            const float2 v1 = __half22float2(exl3_u2h(d[0][1]));
            accf[0].x += v0.x; accf[0].y += v0.y;
            accf[1].x += v1.x; accf[1].y += v1.y;
            d[0][0] = 0u; d[0][1] = 0u;
        } else if constexpr (MMA) {
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                const float2 v = __half22float2(exl3_u2h(d[s][0]));
                accf[s].x += v.x;
                accf[s].y += v.y;
                if (XROWS > 8) {
                    const float2 u = __half22float2(exl3_u2h(d[s][1]));
                    accg[s].x += u.x;
                    accg[s].y += u.y;
                }
                d[s][0] = 0u; d[s][1] = 0u;
            }
        }
    };

    // prefetched words of the next load group: pf[u][m] = word loff[m] of the warp's tile u of that group
    uint32_t pf[U][NL] = {};
    auto load_group = [&](const uint32_t * base, const int cnt) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (u < cnt) {
#pragma unroll
                for (int m = 0; m < NL; ++m) {
                    pf[u][m] = __ldcs(base + (size_t) u * sw + loff[m]);
                }
            }
        }
    };
    // PF2: pf2[h] = half h (tiles 4h..4h+3) of the warp's next 8-tile group
    uint32_t pf2[2][U][NL] = {};
    auto load_half = [&](const int h, const uint32_t * base, const int cnt) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (u < cnt) {
#pragma unroll
                for (int m = 0; m < NL; ++m) {
                    pf2[h][u][m] = __ldcs(base + (size_t) u * sw + loff[m]);
                }
            }
        }
    };
    auto take = [&](const uint32_t (&src)[U][NL], uint32_t (&w)[U][NWORDS]) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
#pragma unroll
            for (int m = 0; m < NWORDS; ++m) {
                if (SHFL) {
                    w[u][m] = (BITS == 4 && m == 1) ? src[u][0] : __shfl_sync(0xffffffffu, src[u][0], woff[m]);
                } else {
                    w[u][m] = src[u][SHFL ? 0 : m];
                }
            }
        }
    };
    // one original 4-tile load group: full groups fold once, the tail group folds after every tile (as k_exl3_gemv)
    auto run_group = [&](const uint32_t (&w)[U][NWORDS], const int step, const int cnt) {
        if (cnt == U) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                tile_fma(w[u], (step + u) * EXL3_GEMV_NWARPS + warp);
            }
            fold();
        } else {
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (u < cnt) {
                    tile_fma(w[u], (step + u) * EXL3_GEMV_NWARPS + warp);
                    fold();
                }
            }
        }
    };

    // bpk == 0: the block's contiguous item range, k-split major (item = ks * nt + ni).
    // bpk > 0 ([#41 EXL3W] ORDER): bpk blocks per k-split; block b takes k-split b / bpk and the n-tiles
    // b % bpk + j * bpk, j = 0, 1, .. Every block still stays on one k-split (x staged once), and the blocks that run
    // at the same time read neighbouring n-tiles of the same k-tile rows (one contiguous front through the weights)
    // instead of n-tiles ~n_items/grid apart. Same items, same per-item work => same output bits.
    const int n_items = nt * ksplit;
    const int o_ks = bpk > 0 ? (int) blockIdx.x / bpk : 0;
    const int o_nb = bpk > 0 ? (int) blockIdx.x - o_ks * bpk : 0;
    const int i_beg = bpk > 0 ? 0 : (int) (((long long) blockIdx.x * n_items) / gridDim.x);
    const int i_end = bpk > 0 ? ((o_ks < ksplit && o_nb < nt) ? (nt - o_nb + bpk - 1) / bpk : 0)
                              : (int) (((long long) (blockIdx.x + 1) * n_items) / gridDim.x);

    struct item_t { int ks, ni, kb, nk, n_my; const uint32_t * wp; };
    auto setup = [&](const int item) {
        item_t it;
        it.ks   = bpk > 0 ? o_ks : item / nt;
        it.ni   = bpk > 0 ? o_nb + item * bpk : item - it.ks * nt;
        it.kb   = it.ks * kpi;
        it.nk   = min(kpi, kt - it.kb);
        it.n_my = max(0, (it.nk - warp + EXL3_GEMV_NWARPS - 1) / EXL3_GEMV_NWARPS);   // nk < 0 past the last k-split
        it.wp   = trellis + ((size_t) (it.kb + warp) * nt + it.ni) * NW;
        return it;
    };

    item_t cur = {};
    if (i_beg < i_end) {
        cur = setup(i_beg);
        if constexpr (PF2) {
            load_half(0, cur.wp, min(U, cur.n_my));
            load_half(1, cur.wp + (size_t) U * sw, min(U, cur.n_my - U));
        } else {
            load_group(cur.wp, min(U, cur.n_my));
        }
    }
    int staged = -1;
    int pb     = 0;
    for (int item = i_beg; item < i_end; ++item) {
        const bool has_next = item + 1 < i_end;
        const item_t nxt = has_next ? setup(item + 1) : cur;

        if (cur.ks != staged) {
            // every thread is past the previous item's barrier, so its tile loop no longer reads xs
            if constexpr (FRAG) {
                // p = (kl * T + t) * 4 + q  ->  {x[t][kl*8 + q], x[t][kl*8 + q + 4]} (half2 units)
                for (int p = threadIdx.x; p < cur.nk * T * 4; p += 32 * EXL3_GEMV_NWARPS) {
                    const int qq = p & 3;
                    const int r  = p >> 2;
                    const int kl = r / T;
                    const int t  = r - kl * T;
                    const half2 * xt = x + (size_t) t * (K / 2) + (size_t) (cur.kb + kl) * 8 + qq;
                    xf[p] = make_uint2(exl3_h2u(xt[0]), exl3_h2u(xt[4]));
                }
            } else {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const half2 * xt = x + (size_t) t * (K / 2) + (size_t) cur.kb * 8;
                    for (int p = threadIdx.x; p < cur.nk * 8; p += 32 * EXL3_GEMV_NWARPS) {
                        xs[t][p] = xt[p];
                    }
                }
            }
            __syncthreads();
            staged = cur.ks;
        }

#pragma unroll
        for (int s = 0; s < 2; ++s) {
#pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[s][t] = 0.0f;
            }
            d[s][0] = 0u; d[s][1] = 0u;
            accf[s] = make_float2(0.0f, 0.0f);
            accg[s] = make_float2(0.0f, 0.0f);
        }

        if constexpr (PF2) {
            for (int step = 0; step < cur.n_my; step += 2 * U) {
                const bool more = step + 2 * U < cur.n_my;   // the next 8-tile group is in this item
                uint32_t w[U][NWORDS];
                // half 0: tiles step..step+3 (always present)
                take(pf2[0], w);
                if (more) {
                    load_half(0, cur.wp + (size_t) (step + 2 * U) * sw, min(U, cur.n_my - step - 2 * U));
                } else if (has_next) {
                    load_half(0, nxt.wp, min(U, nxt.n_my));
                }
                run_group(w, step, min(U, cur.n_my - step));
                // half 1: tiles step+4..step+7 (absent in an item's short last group)
                const int cnt1 = cur.n_my - step - U;
                if (cnt1 > 0) {
                    take(pf2[1], w);
                }
                if (more) {
                    load_half(1, cur.wp + (size_t) (step + 3 * U) * sw, min(U, cur.n_my - step - 3 * U));
                } else if (has_next) {
                    load_half(1, nxt.wp + (size_t) U * sw, min(U, nxt.n_my - U));
                }
                if (cnt1 > 0) {
                    run_group(w, step + U, min(U, cnt1));
                }
            }
            if (cur.n_my == 0 && has_next) {
                load_half(0, nxt.wp, min(U, nxt.n_my));
                load_half(1, nxt.wp + (size_t) U * sw, min(U, nxt.n_my - U));
            }
        } else {
        for (int step = 0; step < cur.n_my; step += U) {
            const int cnt = min(U, cur.n_my - step);
            uint32_t w[U][NWORDS];
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int m = 0; m < NWORDS; ++m) {
                    if (SHFL) {
                        w[u][m] = (BITS == 4 && m == 1) ? pf[u][0] : __shfl_sync(0xffffffffu, pf[u][0], woff[m]);
                    } else {
                        w[u][m] = pf[u][SHFL ? 0 : m];
                    }
                }
            }
            if (step + U < cur.n_my) {
                load_group(cur.wp + (size_t) (step + U) * sw, min(U, cur.n_my - step - U));
            } else if (has_next) {
                load_group(nxt.wp, min(U, nxt.n_my));
            }
            if (cnt == U) {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    tile_fma(w[u], (step + u) * EXL3_GEMV_NWARPS + warp);
                }
                fold();
            } else {
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    if (u < cnt) {
                        tile_fma(w[u], (step + u) * EXL3_GEMV_NWARPS + warp);
                        fold();
                    }
                }
            }
        }
        if (cur.n_my == 0 && has_next) {
            load_group(nxt.wp, min(U, nxt.n_my));
        }
        }

        float (&pp)[EXL3_GEMV_NWARPS][T][WEIGHT_MAJOR ? 20 : 16] = part[pb];
        if constexpr (WEIGHT_MAJOR) {
            if (2*q < T) {
                pp[warp][2*q][g]     = accf[0].x;
                pp[warp][2*q][g + 8] = accf[1].x;
            }
            if (2*q + 1 < T) {
                pp[warp][2*q + 1][g]     = accf[0].y;
                pp[warp][2*q + 1][g + 8] = accf[1].y;
            }
        } else if constexpr (MMA) {
            if (g < T) {
                pp[warp][g][2 * q]         = accf[0].x;
                pp[warp][g][2 * q + 1]     = accf[0].y;
                pp[warp][g][8 + 2 * q]     = accf[1].x;
                pp[warp][g][8 + 2 * q + 1] = accf[1].y;
            }
            if (XROWS > 8 && g + 8 < T) {
                const int r = (XROWS > 8) ? g + 8 : 0;
                pp[warp][r][2 * q]         = accg[0].x;
                pp[warp][r][2 * q + 1]     = accg[0].y;
                pp[warp][r][8 + 2 * q]     = accg[1].x;
                pp[warp][r][8 + 2 * q + 1] = accg[1].y;
            }
        } else {
#pragma unroll
            for (int s = 0; s < 2; ++s) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    float v = acc[s][t];
                    v += __shfl_xor_sync(0xffffffffu, v, 1);
                    v += __shfl_xor_sync(0xffffffffu, v, 2);
                    acc[s][t] = v;
                }
            }
            if (q == 0) {
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    pp[warp][t][g]     = acc[0][t];
                    pp[warp][t][g + 8] = acc[1][t];
                }
            }
        }
        __syncthreads();   // the next writer of part[pb] is two items away, behind the next item's barrier
        for (int i = threadIdx.x; i < 16 * T; i += 32 * EXL3_GEMV_NWARPS) {
            const int t = i >> 4;
            const int n = i & 15;
            float v = 0.0f;
#pragma unroll
            for (int wv = 0; wv < EXL3_GEMV_NWARPS; ++wv) {
                v += pp[wv][t][n];
            }
            y[((size_t) cur.ks * T + t) * N + (size_t) cur.ni * 16 + n] = v * (1.0f / EXL3_GEMV_X_SCALE);
        }
        pb ^= 1;
        cur = nxt;
    }
}

// [#75] fp16 envelope check, GGML_CUDA_EXL3_ENVELOPE=1 (off by default; the GEMV kernel itself is unchanged). After
// each EXL3 GEMV two small kernels read the staged fp16 x and the fp32 output: the largest staged |x/16| per path, the
// staged values above the path's guaranteed-safe bound, and the inf/nan counts. The kernels are captured into CUDA
// graphs like any other node, so replays keep counting. The totals are read back and printed at process exit (and
// added to the fallback ledger under "cuda.exl3" when GGML_LEDGER=1).
struct exl3_envelope_stats {
    unsigned int       max_x[2];        // float bits of max |x/16|, [0] = T = 1, [1] = mma path
    unsigned long long over[2];         // staged values above EXL3_GEMV_SAFE_X_T1 / EXL3_GEMV_SAFE_X_MMA
    unsigned long long nonfinite_x;     // inf/nan in the staged x
    unsigned long long nonfinite_y[2];  // inf/nan in the output
    unsigned long long calls[2];        // GEMVs checked
    unsigned long long n_x[2];          // staged values checked
};

// one copy per device (module global, zero at load): no allocation or memset that could land inside a graph capture
static __device__ exl3_envelope_stats g_exl3_envelope_dev;

static __global__ void k_exl3_envelope_x(const half * __restrict__ xh, const int64_t n, const int mma) {
    exl3_envelope_stats * st = &g_exl3_envelope_dev;
    const float bound = mma ? EXL3_GEMV_SAFE_X_MMA : EXL3_GEMV_SAFE_X_T1;
    float m = 0.0f;
    unsigned long long over = 0, nonf = 0;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const float v = fabsf(__half2float(xh[i]));
        if (!isfinite(v)) {
            nonf++;
            continue;
        }
        m = fmaxf(m, v);
        over += v > bound;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        m     = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        over += __shfl_xor_sync(0xffffffffu, over, o);
        nonf += __shfl_xor_sync(0xffffffffu, nonf, o);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(&st->max_x[mma], __float_as_uint(m));   // non-negative floats order like their bits
        if (over) atomicAdd(&st->over[mma], over);
        if (nonf) atomicAdd(&st->nonfinite_x, nonf);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&st->n_x[mma], (unsigned long long) n);
    }
}

static __global__ void k_exl3_envelope_y(const float * __restrict__ y, const int64_t n, const int mma) {
    exl3_envelope_stats * st = &g_exl3_envelope_dev;
    unsigned long long nonf = 0;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        nonf += !isfinite(y[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        nonf += __shfl_xor_sync(0xffffffffu, nonf, o);
    }
    if ((threadIdx.x & 31) == 0 && nonf) {
        atomicAdd(&st->nonfinite_y[mma], nonf);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(&st->calls[mma], 1ull);
    }
}

static bool exl3_envelope_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_ENVELOPE");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

static std::atomic<bool> g_exl3_envelope_used[GGML_CUDA_MAX_DEVICES];

static void exl3_envelope_report() {
    for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
        if (!g_exl3_envelope_used[id].load()) {
            continue;
        }
        exl3_envelope_stats h = {};
        if (cudaSetDevice(id) != cudaSuccess ||
            cudaMemcpyFromSymbol(&h, g_exl3_envelope_dev, sizeof(h)) != cudaSuccess) {
            (void) cudaGetLastError();
            fprintf(stderr, "exl3_envelope: device %d: stats unreadable at exit\n", id);
            continue;
        }
        const char * path[2] = {"T=1", "mma"};
        const float  bound[2] = {EXL3_GEMV_SAFE_X_T1, EXL3_GEMV_SAFE_X_MMA};
        for (int p = 0; p < 2; ++p) {
            if (h.calls[p] == 0) {
                continue;
            }
            float mx;
            memcpy(&mx, &h.max_x[p], sizeof(mx));
            fprintf(stderr, "exl3_envelope: device %d %-3s: %llu GEMVs, max |x/16| %.3f (safe bound %.1f), %llu of %llu staged values "
                            "above the bound, %llu non-finite outputs\n",
                    id, path[p], h.calls[p], mx, bound[p], h.over[p], h.n_x[p], h.nonfinite_y[p]);
            ggml_ledger_addf("cuda.exl3", (int64_t) h.calls[p],       "envelope path=%s gemv_calls", path[p]);
            ggml_ledger_addf("cuda.exl3", (int64_t) h.over[p],        "envelope path=%s x_above_safe_bound", path[p]);
            ggml_ledger_addf("cuda.exl3", (int64_t) h.nonfinite_y[p], "envelope path=%s y_nonfinite", path[p]);
            ggml_ledger_addf("cuda.exl3", (int64_t) (mx * 1000.0f),   "envelope path=%s max_abs_x16_milli", path[p]);
        }
        fprintf(stderr, "exl3_envelope: device %d: %llu non-finite staged x values\n", id, h.nonfinite_x);
        ggml_ledger_add("cuda.exl3", "envelope x_nonfinite", (int64_t) h.nonfinite_x);
    }
}

static void exl3_envelope_check(const int id, const half * xh, const float * y, const int64_t T, const int K, const int N, cudaStream_t stream) {
    static std::atomic<bool> registered{false};
    bool expected = false;
    if (registered.compare_exchange_strong(expected, true)) {
        atexit(exl3_envelope_report);   // registered after the CUDA runtime's own handlers, so it runs before them
    }
    g_exl3_envelope_used[id].store(true);
    const int mma = T >= EXL3_GEMV_MMA_MIN_T ? 1 : 0;
    const int64_t nx = T * K;
    const int64_t ny = T * N;
    k_exl3_envelope_x<<<(int) std::min<int64_t>((nx + 255) / 256, 256), 256, 0, stream>>>(xh, nx, mma);
    k_exl3_envelope_y<<<(int) std::min<int64_t>((ny + 255) / 256, 256), 256, 0, stream>>>(y, ny, mma);
}

static int exl3_gemv_bps_cap() {   // EXL3_GEMV_BPS: cap the cooperative grid at this many blocks per SM (0 = none)
    static const int cap = [] {
        const char * env = getenv("EXL3_GEMV_BPS");
        return env ? atoi(env) : 0;
    }();
    return cap;
}

struct exl3_gemv_args {
    const uint32_t * trellis; half2 * x; float * out; int kt, nt, K, N, ksplit;
    const float * xf; const float * suh; const float * svh; float * yf; bool post;
};

template <int BITS, int T, bool FUSED, bool WM = false>
static void exl3_gemv_launch_bt(const exl3_gemv_args & a, cudaStream_t stream) {
    static int grid_max = 0;   // resident blocks on the device (per instantiation), measured once
    if (grid_max == 0) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_exl3_gemv<BITS, T, FUSED, WM>, 32 * EXL3_GEMV_NWARPS, 0));
        const int id = ggml_cuda_get_device();
        nb = std::max(1, nb);
        if (FUSED && exl3_gemv_bps_cap() > 0) {
            nb = std::min(nb, exl3_gemv_bps_cap());
        }
        grid_max = nb * ggml_cuda_info().devices[id].nsm;
    }
    const int kpi = (a.kt + a.ksplit - 1) / a.ksplit;                          // k-tiles per item
    GGML_ASSERT(kpi <= exl3_gemv_cfg<T>::KTILES_MAX);
    const int n_items = a.nt * a.ksplit;
    const dim3 grid(std::min(n_items, grid_max));
    const dim3 block(32 * EXL3_GEMV_NWARPS);
    const uint32_t * trellis = a.trellis; half2 * x = a.x; float * out = a.out;
    int kt = a.kt, nt = a.nt, K = a.K, N = a.N, ksplit = a.ksplit, kpi_ = kpi;
    const float * xf = a.xf; const float * suh = a.suh; const float * svh = a.svh; float * yf = a.yf;
    int post = a.post ? 1 : 0;
#if EXL3_GEMV_HAVE_COOP
    if (FUSED) {
        void * args[] = {&trellis, &x, &out, &kt, &nt, &K, &N, &ksplit, &kpi_, &xf, &suh, &svh, &yf, &post};
        CUDA_CHECK(cudaLaunchCooperativeKernel((const void *) k_exl3_gemv<BITS, T, FUSED>, grid, block, args, 0, stream));
        return;
    }
#endif
    k_exl3_gemv<BITS, T, FUSED, WM><<<grid, block, 0, stream>>>(trellis, x, out, kt, nt, K, N, ksplit, kpi_, xf, suh, svh, yf, post);
}

// [#73] the weight-major mma runs at T = 2..8 (split glue; 3/4-bit in k_exl3_gemv, all bits in k_exl3_gemv_dec) by default (measured 2026-09-25 on 4.0 bpw:
// -2.2% per round, identical output); GGML_CUDA_EXL3_WEIGHT_MAJOR=0 selects the x-as-A mma
static bool exl3_gemv_weight_major_env() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_WEIGHT_MAJOR");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return enabled;
}

template <int T, bool FUSED>
static constexpr bool exl3_gemv_weight_major_ok() {
    return LLAMAMPERE_EXL3_WEIGHT_MAJOR && !FUSED && T >= EXL3_GEMV_MMA_MIN_T && T >= 2 && T <= 8;
}

// [#41] GGML_CUDA_EXL3_DEC=0 selects the pre-#41 split-glue kernel k_exl3_gemv (same output bits)
static bool exl3_gemv_dec_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_DEC");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return enabled;
}

// [#41 EXL3W] GGML_CUDA_EXL3_DEC_TMIN=n: widths below n run k_exl3_gemv (same output bits as the dec kernel)
static int exl3_gemv_dec_tmin() {
    static const int tmin = [] {
        const char * env = getenv("GGML_CUDA_EXL3_DEC_TMIN");
        return env ? atoi(env) : 1;
    }();
    return tmin;
}

// [#41 EXL3W] GGML_CUDA_EXL3_VW=0 selects the row-major x staging in the weight-major dec kernel (same output bits)
static bool exl3_gemv_vw_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_VW");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return enabled;
}

// [#41 EXL3W] GGML_CUDA_EXL3_PF2=0 keeps one load group in flight per warp in the fragment-staged dec kernel (same bits)
static bool exl3_gemv_pf2_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_PF2");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return enabled;
}

// [#41 EXL3W] GGML_CUDA_EXL3_ORDER=0 keeps the contiguous per-block item ranges in the dec kernel (same bits)
static bool exl3_gemv_order_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_ORDER");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return enabled;
}

template <int BITS, int T, bool WM, bool FR = false, bool PF2 = false>
static void exl3_gemv_dec_launch_bt(const exl3_gemv_args & a, cudaStream_t stream) {
    if constexpr (WM && !FR) {
        if (exl3_gemv_vw_enabled()) {
            exl3_gemv_dec_launch_bt<BITS, T, WM, true>(a, stream);
            return;
        }
    }
    if constexpr (WM && FR && !PF2) {
        if (exl3_gemv_pf2_enabled()) {
            exl3_gemv_dec_launch_bt<BITS, T, WM, true, true>(a, stream);
            return;
        }
    }
    static int grid_max = 0;   // resident blocks on the device (per instantiation), measured once
    if (grid_max == 0) {
        int nb = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_exl3_gemv_dec<BITS, T, WM, FR, PF2>, 32 * EXL3_GEMV_NWARPS, 0));
        grid_max = std::max(1, nb) * ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    }
    const int kpi = (a.kt + a.ksplit - 1) / a.ksplit;   // same k-split as k_exl3_gemv => same output bits
    GGML_ASSERT(kpi <= exl3_gemv_cfg<T>::KTILES_MAX);
    const int n_items = a.nt * a.ksplit;
    // test-only: GGML_CUDA_EXL3_DEC_GRID=n caps the grid so blocks walk long item ranges (several k-split changes, the
    // partials buffer reused at distance 2); any grid gives the same output bits
    static const int grid_cap = [] {
        const char * env = getenv("GGML_CUDA_EXL3_DEC_GRID");
        return env ? std::max(0, atoi(env)) : 0;
    }();
    int grid = grid_cap > 0 ? std::min({n_items, grid_max, grid_cap}) : std::min(n_items, grid_max);
    // ORDER: whole groups of blocks per k-split (grid >= ksplit; else the contiguous ranges)
    const int bpk = exl3_gemv_order_enabled() && grid >= a.ksplit ? grid / a.ksplit : 0;
    if (bpk > 0) {
        grid = bpk * a.ksplit;
    }
    // engagement line, once per (bits, T, layout) instantiation (WARN so it reaches server logs at default verbosity);
    // the old kernel (GGML_CUDA_EXL3_DEC=0) never gets here
    static bool announced = false;
    if (!announced) {
        announced = true;
        GGML_LOG_WARN("EXL3 decode GEMV k_exl3_gemv_dec engaged (bits %d, T %d, weight-major %d, fragment staging %d, prefetch depth %d, k-split order %d)\n", BITS, T, (int) WM, (int) FR, PF2 ? 2 : 1, (int) (bpk > 0));
    }
    k_exl3_gemv_dec<BITS, T, WM, FR, PF2><<<grid, 32 * EXL3_GEMV_NWARPS, 0, stream>>>(
        a.trellis, a.x, a.out, a.kt, a.nt, a.K, a.N, a.ksplit, kpi, bpk);
}

// the weight-major mma covers every bit width here on SM86 (k_exl3_gemv: 3/4-bit only; bit-identical to the x-as-A mma on SM86
// by test, see the [#73] note at the top); GGML_CUDA_EXL3_WEIGHT_MAJOR=0 or -DLLAMAMPERE_EXL3_WEIGHT_MAJOR=0 keep the
// x-as-A mma
template <int BITS, int T>
static void exl3_gemv_dec_launch_wm(const exl3_gemv_args & a, cudaStream_t stream) {
    if constexpr (exl3_gemv_weight_major_ok<T, false>()) {
        // 2- and 5..8-bit weight-major is proven bit-identical to x-as-A on SM86 only (PTX does not fix the mma
        // accumulation order): run it on compute capability 8.6 and keep x-as-A everywhere else; 3/4-bit keep their
        // pre-#41 behaviour (weight-major on every arch)
        const bool wm_arch_ok = BITS == 3 || BITS == 4 || ggml_cuda_info().devices[ggml_cuda_get_device()].cc == 860;
        if (exl3_gemv_weight_major_env() && wm_arch_ok) {
            exl3_gemv_dec_launch_bt<BITS, T, true>(a, stream);
            return;
        }
    }
    exl3_gemv_dec_launch_bt<BITS, T, false>(a, stream);
}

template <int T>
static void exl3_gemv_dec_launch(const exl3_gemv_args & a, const int bits, cudaStream_t stream) {
    switch (bits) {
        case 2: exl3_gemv_dec_launch_wm<2, T>(a, stream); break;
        case 3: exl3_gemv_dec_launch_wm<3, T>(a, stream); break;
        case 4: exl3_gemv_dec_launch_wm<4, T>(a, stream); break;
        case 5: exl3_gemv_dec_launch_wm<5, T>(a, stream); break;
        case 6: exl3_gemv_dec_launch_wm<6, T>(a, stream); break;
        case 7: exl3_gemv_dec_launch_wm<7, T>(a, stream); break;
        case 8: exl3_gemv_dec_launch_wm<8, T>(a, stream); break;
        default: GGML_ABORT("exl3 gemv: unsupported bits");
    }
}

template <int T, bool FUSED>
static void exl3_gemv_launch(const exl3_gemv_args & a, const int bits, cudaStream_t stream) {
    if constexpr (!FUSED) {
        if (exl3_gemv_dec_enabled() && T >= exl3_gemv_dec_tmin()) {
            exl3_gemv_dec_launch<T>(a, bits, stream);
            return;
        }
    }
    switch (bits) {
        case 2: exl3_gemv_launch_bt<2, T, FUSED>(a, stream); break;
        case 3:
            if constexpr (exl3_gemv_weight_major_ok<T, FUSED>()) {
                if (exl3_gemv_weight_major_env()) {
                    exl3_gemv_launch_bt<3, T, FUSED, true>(a, stream);
                    break;
                }
            }
            exl3_gemv_launch_bt<3, T, FUSED>(a, stream);
            break;
        case 4:
            if constexpr (exl3_gemv_weight_major_ok<T, FUSED>()) {
                if (exl3_gemv_weight_major_env()) {
                    exl3_gemv_launch_bt<4, T, FUSED, true>(a, stream);
                    break;
                }
            }
            exl3_gemv_launch_bt<4, T, FUSED>(a, stream);
            break;
        case 5: exl3_gemv_launch_bt<5, T, FUSED>(a, stream); break;
        case 6: exl3_gemv_launch_bt<6, T, FUSED>(a, stream); break;
        case 7: exl3_gemv_launch_bt<7, T, FUSED>(a, stream); break;
        case 8: exl3_gemv_launch_bt<8, T, FUSED>(a, stream); break;
        default: GGML_ABORT("exl3 gemv: unsupported bits");
    }
}

template <bool FUSED>
static void exl3_gemv_launch_t(const exl3_gemv_args & a, const int bits, const int T, cudaStream_t stream) {
    switch (T) {
        case 1: exl3_gemv_launch<1, FUSED>(a, bits, stream); break;
        case 2: exl3_gemv_launch<2, FUSED>(a, bits, stream); break;
        case 3: exl3_gemv_launch<3, FUSED>(a, bits, stream); break;
        case 4: exl3_gemv_launch<4, FUSED>(a, bits, stream); break;
        case 5: exl3_gemv_launch<5, FUSED>(a, bits, stream); break;
        case 6: exl3_gemv_launch<6, FUSED>(a, bits, stream); break;
        case 7: exl3_gemv_launch<7, FUSED>(a, bits, stream); break;
        case 8: exl3_gemv_launch<8, FUSED>(a, bits, stream); break;
        case 9: exl3_gemv_launch<9, FUSED>(a, bits, stream); break;
        case 10: exl3_gemv_launch<10, FUSED>(a, bits, stream); break;
        case 11: exl3_gemv_launch<11, FUSED>(a, bits, stream); break;
        case 12: exl3_gemv_launch<12, FUSED>(a, bits, stream); break;
        case 13: exl3_gemv_launch<13, FUSED>(a, bits, stream); break;
        case 14: exl3_gemv_launch<14, FUSED>(a, bits, stream); break;
        case 15: exl3_gemv_launch<15, FUSED>(a, bits, stream); break;
        case 16: exl3_gemv_launch<16, FUSED>(a, bits, stream); break;
        default: GGML_ABORT("exl3 gemv: unsupported T");
    }
}

static bool exl3_gemv_fused_enabled() {
#if EXL3_GEMV_HAVE_COOP
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_FUSED");   // M4 measured slower (barriers on a 1008-block grid): opt-in
        return env != nullptr && strcmp(env, "1") == 0;
    }();
    return enabled;
#else
    return false;
#endif
}

bool ggml_cuda_exl3_gemv_supported(const int64_t T) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_EXL3_GEMV");
        return env == nullptr || strcmp(env, "0") != 0;
    }();
    return enabled && T >= 1 && T <= EXL3_GEMV_MAX_T;
}

static int exl3_gemv_ktiles_max(const int T) {
    static const int tab[EXL3_GEMV_MAX_T] = {
        exl3_gemv_cfg<1>::KTILES_MAX, exl3_gemv_cfg<2>::KTILES_MAX, exl3_gemv_cfg<3>::KTILES_MAX, exl3_gemv_cfg<4>::KTILES_MAX,
        exl3_gemv_cfg<5>::KTILES_MAX, exl3_gemv_cfg<6>::KTILES_MAX, exl3_gemv_cfg<7>::KTILES_MAX, exl3_gemv_cfg<8>::KTILES_MAX,
        exl3_gemv_cfg<9>::KTILES_MAX, exl3_gemv_cfg<10>::KTILES_MAX, exl3_gemv_cfg<11>::KTILES_MAX, exl3_gemv_cfg<12>::KTILES_MAX,
        exl3_gemv_cfg<13>::KTILES_MAX, exl3_gemv_cfg<14>::KTILES_MAX, exl3_gemv_cfg<15>::KTILES_MAX, exl3_gemv_cfg<16>::KTILES_MAX,
    };
    return tab[T - 1];
}

int ggml_cuda_exl3_gemv_ksplit(const int kt, const int nt, const int T) {
    // enough work items to cover the SMs several times over, at most KTILES_MAX k-tiles per item (x is staged per
    // item in shared memory), and >= 2 k-tiles per warp
    const int ktiles_max = exl3_gemv_ktiles_max(T);
    int ksplit = (1536 + nt - 1) / nt;
    const int ksplit_max = kt / (2 * EXL3_GEMV_NWARPS);
    if (ksplit > ksplit_max) ksplit = ksplit_max;
    if (ksplit < 1) ksplit = 1;
    // kpi = ceil(kt/ksplit) must fit the staging buffer
    while ((kt + ksplit - 1) / ksplit > ktiles_max) ksplit++;
    return ksplit;
}

// [#74] raw split-K partials out[ks][T][N] = W . xh for a glued, pre-scaled fp16 x (no svh, no Hadamard)
static void exl3_gemv_raw(const ggml_tensor * src0, const half * xh, float * out, const int T, const int ksplit, cudaStream_t stream) {
    exl3_gemv_args a;
    a.trellis = (const uint32_t *) src0->data; a.x = (half2 *) xh; a.out = out;
    a.kt = (int) src0->ne[0] / 16; a.nt = (int) src0->ne[1] / 16; a.K = (int) src0->ne[0]; a.N = (int) src0->ne[1];
    a.ksplit = ksplit;
    a.xf = nullptr; a.suh = nullptr; a.svh = nullptr; a.yf = nullptr; a.post = true;
    exl3_gemv_launch_t<false>(a, ggml_exl3_bits(src0->type), T, stream);
}

static bool exl3_ffn_mm_ok(const ggml_tensor * mm, const int64_t T) {
    const ggml_tensor * w  = mm->src[0];
    const ggml_tensor * x  = mm->src[1];
    const ggml_tensor * su = mm->src[2];
    const ggml_tensor * sv = mm->src[3];
    return ggml_exl3_bits(w->type) != 0 && w->ne[2] == 1 && w->ne[3] == 1 && w->ne[0] % 128 == 0 && w->ne[1] % 128 == 0 &&
           x->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(mm) &&
           x->ne[1] * x->ne[2] * x->ne[3] == T &&
           su != nullptr && sv != nullptr && su->type == GGML_TYPE_F32 && sv->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(su) && ggml_is_contiguous(sv) && su->ne[0] == w->ne[0] && sv->ne[0] == w->ne[1];
}

bool ggml_cuda_exl3_ffn_bridge(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_gate, const ggml_tensor * mm_up,
                               ggml_tensor * mm_down) {
    const int64_t T = mm_down->src[1]->ne[1] * mm_down->src[1]->ne[2] * mm_down->src[1]->ne[3];
    // the cooperative fused GEMV does its own glue in-kernel; the envelope check wants every GEMV's x and y
    if (!ggml_cuda_exl3_gemv_supported(T) || exl3_gemv_fused_enabled() || exl3_envelope_enabled()) {
        return false;
    }
    if (!exl3_ffn_mm_ok(mm_gate, T) || !exl3_ffn_mm_ok(mm_up, T) || !exl3_ffn_mm_ok(mm_down, T)) {
        return false;
    }
    const ggml_tensor * wg = mm_gate->src[0];
    const ggml_tensor * wu = mm_up->src[0];
    const ggml_tensor * wd = mm_down->src[0];
    const int K  = (int) wg->ne[0];
    const int NF = (int) wg->ne[1];
    const int ND = (int) wd->ne[1];
    if (wu->ne[0] != K || wu->ne[1] != NF || wd->ne[0] != NF) {
        return false;
    }

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();
    const int ks_g = ggml_cuda_exl3_gemv_ksplit(K / 16, NF / 16, (int) T);
    const int ks_u = ggml_cuda_exl3_gemv_ksplit(K / 16, NF / 16, (int) T);
    const int ks_d = ggml_cuda_exl3_gemv_ksplit(NF / 16, ND / 16, (int) T);

    // gate and up: glue_in (shared when both read the same x with the same suh) + raw partials
    const float * suh_g = (const float *) mm_gate->src[2]->data;
    const float * suh_u = (const float *) mm_up->src[2]->data;
    ggml_cuda_pool_alloc<half> xh_g(ctx.pool(id), (size_t) T * K);
    ggml_cuda_exl3_glue_in_f16((const float *) mm_gate->src[1]->data, suh_g, xh_g.get(), K, T, EXL3_GEMV_X_SCALE, stream);
    ggml_cuda_pool_alloc<half> xh_u(ctx.pool(id));
    const half * xu = xh_g.get();
    if (mm_up->src[1]->data != mm_gate->src[1]->data || suh_u != suh_g) {
        xu = xh_u.alloc((size_t) T * K);
        ggml_cuda_exl3_glue_in_f16((const float *) mm_up->src[1]->data, suh_u, xh_u.get(), K, T, EXL3_GEMV_X_SCALE, stream);
    }
    ggml_cuda_pool_alloc<float> part_g(ctx.pool(id), (size_t) ks_g * T * NF);
    ggml_cuda_pool_alloc<float> part_u(ctx.pool(id), (size_t) ks_u * T * NF);
    exl3_gemv_raw(wg, xh_g.get(), part_g.get(), (int) T, ks_g, stream);
    exl3_gemv_raw(wu, xu,         part_u.get(), (int) T, ks_u, stream);

    // bridge: both glue_outs, SwiGLU, the down projection's glue_in
    ggml_cuda_pool_alloc<half> xh_d(ctx.pool(id), (size_t) T * NF);
    ggml_cuda_exl3_ffn_bridge_f16(part_g.get(), ks_g, (const float *) mm_gate->src[3]->data,
                                  part_u.get(), ks_u, (const float *) mm_up->src[3]->data,
                                  (const float *) mm_down->src[2]->data, xh_d.get(), NF, T, EXL3_GEMV_X_SCALE, stream);

    // down: the prepared-fp16 entry, raw partials + glue_out (svh is always present here, so always post)
    ggml_cuda_pool_alloc<float> part_d(ctx.pool(id), (size_t) ks_d * T * ND);
    exl3_gemv_raw(wd, xh_d.get(), part_d.get(), (int) T, ks_d, stream);
    ggml_cuda_exl3_glue_out(part_d.get(), (const float *) mm_down->src[3]->data, (float *) mm_down->data, ND, T, ks_d, 1.0f, stream);
    return true;
}

void ggml_cuda_exl3_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int64_t T,
                         const float * suh, const float * svh) {
    const int bits = ggml_exl3_bits(src0->type);
    GGML_ASSERT(bits != 0 && T >= 1 && T <= EXL3_GEMV_MAX_T);
    const int K  = (int) src0->ne[0];
    const int N  = (int) src0->ne[1];
    const int kt = K / 16;
    const int nt = N / 16;
    const int ksplit = ggml_cuda_exl3_gemv_ksplit(kt, nt, (int) T);

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();

    // x -> fp16 [T][K], pre-scaled (and suh * / Hadamard when fused)
    ggml_cuda_pool_alloc<half> xh(ctx.pool(id), (size_t) T * K);
    // the fused phase 3 walks 128-chunks of [T][N]: without svh and N % 128 != 0 it would drop the T*N % 128 tail,
    // so that shape takes the split glue (its glue_out has a per-element path)
    const bool fused = exl3_gemv_fused_enabled() && (svh != nullptr || N % 128 == 0);
    if (!fused) {
        ggml_cuda_exl3_glue_in_f16((const float *) src1->data, suh, xh.get(), K, T, EXL3_GEMV_X_SCALE, stream);
    }

    const bool post = ksplit > 1 || svh != nullptr;
    ggml_cuda_pool_alloc<float> part(ctx.pool(id));
    float * y   = (float *) dst->data;
    float * out = post ? part.alloc((size_t) ksplit * T * N) : y;
    exl3_gemv_args a;
    a.trellis = (const uint32_t *) src0->data; a.x = (half2 *) xh.get(); a.out = out;
    a.kt = kt; a.nt = nt; a.K = K; a.N = N; a.ksplit = ksplit;
    a.xf = (const float *) src1->data; a.suh = suh; a.svh = svh; a.yf = y; a.post = post;
    if (fused) {
        exl3_gemv_launch_t<true>(a, bits, (int) T, stream);
    } else {
        exl3_gemv_launch_t<false>(a, bits, (int) T, stream);
        if (post) {
            ggml_cuda_exl3_glue_out(out, svh, y, N, T, ksplit, 1.0f, stream);
        }
    }
    if (exl3_envelope_enabled()) {
        exl3_envelope_check(id, xh.get(), y, T, K, N, stream);
    }
}
