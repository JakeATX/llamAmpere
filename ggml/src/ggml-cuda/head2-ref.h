#pragma once
// [HEAD2] host side of the exact two-stage output head (LLAMA_HEAD2, default off). Plain C++, no CUDA: the CUDA
// backend uses it to build the planes at load time and tests/test-head2.cpp uses it as the CPU reference.
//
// Q8_0 plane layout (one row of nb blocks, 34*nb bytes in the file, the same 34*nb bytes after the split):
//   hi part, 26*nb bytes: [nb fp16 scales][nb x 16 B: bits 2..5 of every weight][nb x 8 B: bits 6..7]
//   lo part,  8*nb bytes: [nb x 8 B: bits 0..1]
// Inside a block the 32 weights go in 4 groups of 8 (group g = weights 8g..8g+7, the 8 weights one GPU lane reads):
// group g's nibbles are the uint32 at 16*ib + 4*g (weight k of the group at bits 4k..4k+3), its bit pairs the uint16
// at 8*ib + 2*g (weight k at bits 2k..2k+1), little endian.
//
// The bound. With u = the weight byte as unsigned, q = (int8) u = 4*h + l, h = q >> 2 (arithmetic, -32..31) and
// l = u & 3 (0..3). For one column x and block b with scale d_b:
//   R = sum_b d_b * sum_j q_j x_j = C + E,  C = sum_b d_b * (4 * sum_j h_j x_j + 1.5 * S_b),  S_b = sum_j x_j,
//   |E| = |sum_b d_b sum_j (l_j - 1.5) x_j| <= 1.5 * D,   D = sum_b |d_b| * A_b,  A_b = sum_j |x_j|.
// The reference float result F (any summation order of depth <= nb + 64 roundings, |q| <= 128) is within
// gamma * 128 * D of R, and the screen's own float C~ is within gamma * 129.5 * D of C, gamma = n u / (1 - n u),
// n = nb + 64, u = 2^-24. The computed D~ >= D (1 - gamma). So
//   F in [C~ - rho, C~ + rho],  rho = 258 gamma / (1 - gamma) * D~ + min(1.5 / (1 - gamma) * D~, N_r * X)
// with the products and the +- rounded outward; N_r * X >= |E| by Cauchy-Schwarz (see coef_slack below). Q6_K splits the same way: the 6-bit code u6 = 4*h + l, the weight is
// d * sc * (u6 - 32), so the center uses (4h + 1.5 - 32) and the radius 1.5 * |d * sc| * A per 16-weight group.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace head2 {

static constexpr int QK = 32; // Q8_0 block

inline int64_t hi_row_bytes(int nb) { return 26*(int64_t) nb; }
inline int64_t lo_row_bytes(int nb) { return  8*(int64_t) nb; }

inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31;
    const uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t       man  = h & 0x3ff;
    uint32_t       bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            int e = -1;
            do { ++e; man <<= 1; } while ((man & 0x400) == 0);
            bits = sign | ((uint32_t) (127 - 15 - e) << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// split one Q8_0 row (src: nb blocks of {fp16 d, int8 qs[32]}) into its hi and lo parts
inline void split_row_q8_0(const uint8_t * src, int nb, uint8_t * hi, uint8_t * lo) {
    uint8_t * sc  = hi;
    uint8_t * nib = hi + 2*nb;
    uint8_t * b67 = hi + 18*nb;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t * blk = src + 34*ib;
        memcpy(sc + 2*ib, blk, 2);
        for (int g = 0; g < 4; ++g) {
            uint32_t nw = 0;
            uint16_t hw = 0;
            uint16_t lw = 0;
            for (int k = 0; k < 8; ++k) {
                const uint32_t u = blk[2 + 8*g + k];
                nw |= ((u >> 2) & 15u) << (4*k);
                hw |= (uint16_t) (((u >> 6) & 3u) << (2*k));
                lw |= (uint16_t) ((u & 3u) << (2*k));
            }
            for (int i = 0; i < 4; ++i) { nib[16*ib + 4*g + i] = (uint8_t) (nw >> (8*i)); }
            b67[8*ib + 2*g + 0] = (uint8_t) hw; b67[8*ib + 2*g + 1] = (uint8_t) (hw >> 8);
            lo [8*ib + 2*g + 0] = (uint8_t) lw; lo [8*ib + 2*g + 1] = (uint8_t) (lw >> 8);
        }
    }
}

// weight j (0..31) of block ib, rebuilt from the planes
inline int8_t q_at(const uint8_t * hi, const uint8_t * lo, int nb, int ib, int j) {
    const int g = j / 8, k = j % 8;
    uint32_t nw; memcpy(&nw, hi + 2*nb + 16*ib + 4*g, 4);
    uint16_t hw; memcpy(&hw, hi + 18*nb + 8*ib + 2*g, 2);
    uint16_t lw; memcpy(&lw, lo + 8*ib + 2*g, 2);
    const uint32_t u = (((hw >> (2*k)) & 3u) << 6) | (((nw >> (4*k)) & 15u) << 2) | ((lw >> (2*k)) & 3u);
    return (int8_t) (uint8_t) u;
}

inline int h_at(const uint8_t * hi, int nb, int ib, int j) {
    const int g = j / 8, k = j % 8;
    uint32_t nw; memcpy(&nw, hi + 2*nb + 16*ib + 4*g, 4);
    uint16_t hw; memcpy(&hw, hi + 18*nb + 8*ib + 2*g, 2);
    const int t = (int) ((((hw >> (2*k)) & 3u) << 4) | ((nw >> (4*k)) & 15u));
    return t - ((t & 32) << 1);
}

inline float scale_at(const uint8_t * hi, int ib) {
    uint16_t h; memcpy(&h, hi + 2*ib, 2);
    return fp16_to_f32(h);
}

inline double gamma_n(int nb) {
    const double n = (double) nb + 64.0;
    const double u = std::ldexp(1.0, -24);
    return n*u/(1.0 - n*u);
}

inline float round_up_f(double c) {
    float f = (float) c;
    while ((double) f < c) {
        f = std::nextafter(f, INFINITY);
    }
    return std::nextafter(f, INFINITY);
}

// rho = coef_slack * D~ + min(coef_l1 * D~, N_r * X), every product and sum rounded up:
//   the float error of both sides (258 gamma D) plus the low-bit term, bounded two ways (both always valid):
//   L1: |E| <= 1.5 D <= 1.5 D~ / (1 - gamma);  L2 (Cauchy-Schwarz): |E| <= N_r * X with
//   N_r = || d_b (l_j - 1.5) ||_2 over the row (load time, rounded up) and X = || x ||_2 (per column, rounded up).
// The L2 side is ~sqrt(n) tighter for spread activations, the L1 side wins for a few huge channels.
inline float coef_slack(int nb) { const double g = gamma_n(nb); return round_up_f(258.0*g/(1.0 - g)); }
inline float coef_l1   (int nb) { const double g = gamma_n(nb); return round_up_f(1.5/(1.0 - g)); }

// N_r of one Q8_0 row in file layout (34 B blocks); inf for a non-finite scale (then only the L1 side applies)
inline float row_lo_norm_q8_0(const uint8_t * src, int nb) {
    double n2 = 0.0;
    for (int ib = 0; ib < nb; ++ib) {
        uint16_t dh; memcpy(&dh, src + 34*ib, 2);
        const double d = fp16_to_f32(dh);
        double s = 0.0;
        for (int j = 0; j < 32; ++j) {
            const double e = (double) (src[34*ib + 2 + j] & 3) - 1.5;
            s += e*e;
        }
        n2 += d*d*s;
    }
    if (!std::isfinite(n2)) {
        return INFINITY;
    }
    return round_up_f(std::sqrt(n2)*(1.0 + 1e-12));
}

// X of one column (double sum, rounded up)
inline float col_norm(const float * x, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) { s += (double) x[i]*x[i]; }
    return round_up_f(std::sqrt(s)*(1.0 + 1e-12));
}

// The f32-activation Q8_0 matvec of the CUDA backend's default decode route (Q8A16, T <= 8) for one row and one
// column, in its exact order: lane l takes blocks l/4, l/4 + 8, ... and weights 8*(l%4) .. +7, an 8-term fma chain per
// block, one fma into the lane sum, then the xor butterfly of warp_reduce_sum. It models the GPU bit for bit except
// for subnormals (the GPU flushes them).
template <typename QF, typename DF>
inline float dot_lane_order(QF q /* (ib, j) -> int8 */, DF d /* ib -> float */, const float * x, int nb) {
    float acc[32];
    for (int lane = 0; lane < 32; ++lane) {
        float a = 0.0f;
        const int iq = lane & 3;
        for (int ib = lane >> 2; ib < nb; ib += 8) {
            const float * xp = x + (int64_t) ib*QK + iq*8;
            float f[8];
            for (int k = 0; k < 8; ++k) { f[k] = (float) q(ib, iq*8 + k); }
            float p = f[0]*xp[0];
            for (int k = 1; k < 8; ++k) { p = std::fma(f[k], xp[k], p); }
            a = std::fma(d(ib), p, a);
        }
        acc[lane] = a;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float nxt[32];
        for (int l = 0; l < 32; ++l) { nxt[l] = acc[l] + acc[l ^ off]; }
        memcpy(acc, nxt, sizeof(acc));
    }
    return acc[0];
}

// the screen interval of one row for one column (host model of the CUDA screen; s8/a8 are the per-8-group sums of
// x and |x|). The interval is outward-rounded in double here, which is only wider than the GPU's float one.
inline void screen_row(const uint8_t * hi, int nb, const float * x, const float * s8, const float * a8, float rn, float xn,
        float & lo, float & hi_out) {
    double c = 0.0, a = 0.0; // the host model sums in double (still a valid interval: same bound, fewer roundings)
    for (int ib = 0; ib < nb; ++ib) {
        const float d = scale_at(hi, ib);
        for (int g = 0; g < 4; ++g) {
            double p = 0.0;
            for (int k = 0; k < 8; ++k) { p += (double) h_at(hi, nb, ib, 8*g + k) * x[ib*QK + 8*g + k]; }
            c += (double) d * (4.0*p + 1.5*s8[ib*4 + g]);
            a += std::fabs((double) d) * a8[ib*4 + g];
        }
    }
    const double rho = (double) coef_slack(nb) * a + std::min((double) coef_l1(nb) * a, (double) rn * xn);
    lo     = std::nextafter((float) (c - rho), -INFINITY);
    hi_out = std::nextafter((float) (c + rho),  INFINITY);
    if (!std::isfinite(lo) || !std::isfinite(hi_out)) { lo = -INFINITY; hi_out = INFINITY; }
}

inline void group_sums(const float * x, int64_t ne0, float * s8, float * a8) {
    for (int64_t g = 0; g < ne0/8; ++g) {
        float s = 0.0f, a = 0.0f;
        for (int k = 0; k < 8; ++k) { s += x[8*g + k]; a += std::fabs(x[8*g + k]); }
        s8[g] = s;
        a8[g] = a;
    }
}

// threshold for the K largest of n lower bounds lo[] (selection rule of the CUDA scan: histogram of [M - W, M] in
// nbin bins, the lower edge of the bin that holds the K-th value, one more bin of slack). Any tau <= the K-th largest
// lo is exact; -inf means "rescore everything".
inline float tau_topk(const float * lo, int64_t n, int K, float W, int nbin, int * hist /* nbin */) {
    float M = -INFINITY;
    for (int64_t i = 0; i < n; ++i) { if (lo[i] > M) { M = lo[i]; } }
    if (!std::isfinite(M) || K <= 0 || K >= n) { return -INFINITY; }
    for (int b = 0; b < nbin; ++b) { hist[b] = 0; }
    const float scale = (float) nbin / W;
    for (int64_t i = 0; i < n; ++i) {
        if (lo[i] >= M - W) {
            int b = (int) ((M - lo[i])*scale);
            b = b < 0 ? 0 : b >= nbin ? nbin - 1 : b;
            hist[b]++;
        }
    }
    int64_t cum = 0;
    for (int b = 0; b < nbin; ++b) {
        cum += hist[b];
        if (cum >= K) {
            return std::nextafter(M - (float) (b + 2)*(W/(float) nbin), -INFINITY);
        }
    }
    return -INFINITY;
}

// mass threshold: rows with hi < tau carry < V e^-delta of the softmax mass after the (optional) final softcap
inline float tau_mass(float M, float delta, float softcap) {
    if (!std::isfinite(M)) { return -INFINITY; }
    if (softcap <= 0.0f) {
        return std::nextafter(M - delta - 1e-3f*(1.0f + std::fabs(M)), -INFINITY);
    }
    const double y = (double) softcap*std::tanh((double) M/softcap) - delta;
    if (y <= -0.999999*softcap) { return -INFINITY; }
    const double t = (double) softcap*std::atanh(y/softcap);
    return std::nextafter((float) (t - 1e-3*(1.0 + std::fabs(t))), -INFINITY);
}

// ---- Q6_K reference (bound math only; the CUDA Q6_K variant is not built yet) ----
// block_q6_K: ql[128], qh[64], int8 scales[16], fp16 d; 256 weights; weight i of the block has 6-bit code u6
// and value d * scales[i/16] * (u6 - 32).
inline int q6k_code(const uint8_t * blk, int i) {
    const uint8_t * ql = blk;
    const uint8_t * qh = blk + 128;
    const int n = i / 128, r = i % 128, l = r % 32, part = r / 32; // ggml's dequantize_row_q6_K order
    const int is = l / 16;
    (void) is;
    const uint8_t * qln = ql + 64*n;
    const uint8_t * qhn = qh + 32*n;
    int lo4, hi2;
    switch (part) {
        case 0:  lo4 = qln[l]      & 0xF; hi2 = (qhn[l] >> 0) & 3; break;
        case 1:  lo4 = qln[l + 32] & 0xF; hi2 = (qhn[l] >> 2) & 3; break;
        case 2:  lo4 = qln[l]      >> 4;  hi2 = (qhn[l] >> 4) & 3; break;
        default: lo4 = qln[l + 32] >> 4;  hi2 = (qhn[l] >> 6) & 3; break;
    }
    return lo4 | (hi2 << 4);
}

inline int q6k_scale_index(int i) {
    const int n = i / 128, r = i % 128, l = r % 32, part = r / 32;
    return 8*n + l/16 + 2*part;
}

// interval of one Q6_K row (nbk blocks of 210 bytes) for one column, in double, outward-rounded to float
inline void screen_row_q6_K(const uint8_t * row, int nbk, const float * x, float & lo, float & hi_out) {
    double c = 0.0, a = 0.0;
    for (int b = 0; b < nbk; ++b) {
        const uint8_t * blk = row + 210*b;
        uint16_t dh; memcpy(&dh, blk + 208, 2);
        const double d = fp16_to_f32(dh);
        const int8_t * sc = (const int8_t *) (blk + 192);
        for (int i = 0; i < 256; ++i) {
            const int u6 = q6k_code(blk, i);
            const double s = d*sc[q6k_scale_index(i)];
            const double xi = x[256*b + i];
            c += s*(4.0*(u6 >> 2) + 1.5 - 32.0)*xi;
            a += std::fabs(s)*std::fabs(xi);
        }
    }
    const double n = 256.0*nbk + 64.0, u = std::ldexp(1.0, -24), g = n*u/(1.0 - n*u);
    // |u6 - 32| <= 32 and |4h + 1.5 - 32| <= 30.5, so the float slack is (32 + 30.5) g D; the low part 1.5 D
    const double rho = (1.5 + 63.0*g)/(1.0 - g)*a;
    lo     = std::nextafter((float) (c - rho), -INFINITY);
    hi_out = std::nextafter((float) (c + rho),  INFINITY);
}

} // namespace head2
