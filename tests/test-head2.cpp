// [HEAD2] CPU reference test of the exact two-stage output head (ggml/src/ggml-cuda/head2-ref.h).
//   1. plane split round trip (every weight byte rebuilt exactly)
//   2. bound: the default route's float logit (GPU lane order) and the exact real dot lie inside [lo, hi] from the
//      screen, for random, adversarial and (with a GGUF path) real head rows
//   3. selection: with tau from the histogram rule, every dropped row is below tau and the top K are all kept, for
//      K = 1, 20, 40, 64, and the mass rule drops at most V e^-delta of the softmax mass (with and without softcap)
//   4. the Q6_K bound (math for the 27B / Flash-Next heads; no CUDA kernel yet)
// usage: test-head2 [model.gguf (Q8_0 output.weight)] [rows]
#include "../ggml/src/ggml-cuda/head2-ref.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace head2;

static constexpr int H2_NBIN_TEST = 1024;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { if (g_fail++ < 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } } while (0)

static uint16_t f32_to_fp16(float f) {
    // round to nearest even, enough for test scales (normal range)
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e <= 0) { return (uint16_t) sign; }
    if (e >= 31) { return (uint16_t) (sign | 0x7c00); }
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) { h++; }
    return (uint16_t) h;
}

// the GPU screen in float, lane order (k_h2_screen), with the final directed roundings done in double
static void screen_gpu_model(const uint8_t * hi, int nb, const float * x, const float * s8, const float * a8, float rn, float xn,
        float & lo, float & hi_out) {
    float c[32], a[32];
    for (int lane = 0; lane < 32; ++lane) {
        float cc = 0.0f, aa = 0.0f;
        const int iq = lane & 3;
        for (int ib = lane >> 2; ib < nb; ib += 8) {
            const float d = scale_at(hi, ib);
            float h[8];
            for (int k = 0; k < 8; ++k) { h[k] = (float) h_at(hi, nb, ib, iq*8 + k); }
            const float * xp = x + ib*QK + iq*8;
            float p = h[0]*xp[0];
            for (int k = 1; k < 8; ++k) { p = std::fma(h[k], xp[k], p); }
            cc = std::fma(d, std::fma(4.0f, p, 1.5f*s8[ib*4 + iq]), cc);
            aa = std::fma(std::fabs(d), a8[ib*4 + iq], aa);
        }
        c[lane] = cc; a[lane] = aa;
    }
    for (int off = 16; off > 0; off >>= 1) {
        float nc[32], na[32];
        for (int l = 0; l < 32; ++l) { nc[l] = c[l] + c[l ^ off]; na[l] = a[l] + a[l ^ off]; }
        memcpy(c, nc, sizeof(c)); memcpy(a, na, sizeof(a));
    }
    // __fmul_ru, __fadd_ru(., 1e-30), __fsub_rd, __fadd_ru
    auto ru = [](double v) { float f = (float) v; if ((double) f < v) { f = std::nextafter(f, INFINITY); } return f; };
    auto rd = [](double v) { float f = (float) v; if ((double) f > v) { f = std::nextafter(f, -INFINITY); } return f; };
    const float lowb = std::min(ru((double) a[0]*coef_l1(nb)), ru((double) rn*xn));
    const float rho  = ru((double) ru((double) ru((double) a[0]*coef_slack(nb)) + lowb) + 1e-30);
    lo     = rd((double) c[0] - rho);
    hi_out = ru((double) c[0] + rho);
    if (!std::isfinite(lo) || !std::isfinite(hi_out)) { lo = -INFINITY; hi_out = INFINITY; }
}

struct q8_matrix {
    int nb = 0;
    int64_t nrows = 0;
    std::vector<uint8_t> raw; // 34*nb per row
};

static void make_random(q8_matrix & m, int64_t nrows, int nb, std::mt19937 & rng) {
    m.nb = nb; m.nrows = nrows; m.raw.assign((size_t) nrows*nb*34, 0);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> ui(-128, 127);
    for (int64_t r = 0; r < nrows; ++r) {
        const int kind = (int) (r % 7);
        // a row "temperature": spreads the logits like a real head (a few rows far above the bulk)
        const float rs = std::exp(0.6f*nd(rng));
        for (int ib = 0; ib < nb; ++ib) {
            uint8_t * blk = m.raw.data() + ((size_t) r*nb + ib)*34;
            float d = 2e-3f*rs*(0.5f + std::fabs(nd(rng)));
            if (kind == 5) { d = -d; }                 // negative scales
            if (kind == 6 && ib % 9 == 0) { d = 0.0f; } // zero scales
            const uint16_t dh = f32_to_fp16(d);
            memcpy(blk, &dh, 2);
            for (int j = 0; j < 32; ++j) {
                int q;
                switch (kind) {
                    case 0:  q = ui(rng); break;                                             // uniform, incl. -128
                    case 1:  q = std::max(-127, std::min(127, (int) std::lrint(40*nd(rng)))); break;
                    case 2:  q = (ui(rng) & ~3) | 3; break;                                  // low bits all 3
                    case 3:  q = ui(rng) & ~3; break;                                        // low bits all 0
                    case 4:  q = j % 2 ? 127 : -128; break;                                  // extremes
                    default: q = std::max(-127, std::min(127, (int) std::lrint(20*nd(rng)))); break;
                }
                blk[2 + j] = (uint8_t) (int8_t) q;
            }
        }
    }
}

static bool load_gguf_head(const char * path, int64_t max_rows, q8_matrix & m) {
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    ggml_context * meta = nullptr;
    ip.ctx = &meta;
    gguf_context * g = gguf_init_from_file(path, ip);
    if (!g) { fprintf(stderr, "cannot open %s\n", path); return false; }
    const int64_t id = gguf_find_tensor(g, "output.weight");
    if (id < 0) { fprintf(stderr, "no output.weight\n"); gguf_free(g); return false; }
    ggml_tensor * t = ggml_get_tensor(meta, "output.weight");
    if (t->type != GGML_TYPE_Q8_0) { fprintf(stderr, "output.weight is %s, not Q8_0\n", ggml_type_name(t->type)); gguf_free(g); return false; }
    m.nb = (int) (t->ne[0]/32);
    const int64_t total = t->ne[1];
    m.nrows = std::min(total, max_rows);
    const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
    FILE * f = fopen(path, "rb");
    // spread the sample over the vocabulary: m.nrows rows at a stride
    m.raw.resize((size_t) m.nrows*m.nb*34);
    const int64_t stride = std::max<int64_t>(1, total/m.nrows);
    for (int64_t r = 0; r < m.nrows; ++r) {
        const int64_t src = std::min(total - 1, r*stride);
#ifdef _WIN32
        _fseeki64(f, (long long) (off + (size_t) src*m.nb*34), SEEK_SET);
#else
        fseeko(f, (off_t) (off + (size_t) src*m.nb*34), SEEK_SET);
#endif
        if (fread(m.raw.data() + (size_t) r*m.nb*34, 1, (size_t) m.nb*34, f) != (size_t) m.nb*34) { fclose(f); gguf_free(g); return false; }
    }
    fclose(f);
    fprintf(stderr, "loaded %" PRId64 " of %" PRId64 " rows (stride %" PRId64 ") of %s, nb %d\n", m.nrows, total, stride, path, m.nb);
    ggml_free(meta);
    gguf_free(g);
    return true;
}

static void run_matrix(const char * label, const q8_matrix & m, int ncols, std::mt19937 & rng) {
    const int nb = m.nb;
    const int64_t ne0 = (int64_t) nb*32, nr = m.nrows;
    std::vector<uint8_t> hi((size_t) nr*hi_row_bytes(nb)), lo((size_t) nr*lo_row_bytes(nb));
    for (int64_t r = 0; r < nr; ++r) {
        split_row_q8_0(m.raw.data() + (size_t) r*nb*34, nb, hi.data() + r*hi_row_bytes(nb), lo.data() + r*lo_row_bytes(nb));
    }
    // 1. round trip
    int64_t bad_rt = 0;
    for (int64_t r = 0; r < nr; ++r) {
        const uint8_t * h = hi.data() + r*hi_row_bytes(nb);
        const uint8_t * l = lo.data() + r*lo_row_bytes(nb);
        for (int ib = 0; ib < nb; ++ib) {
            const uint8_t * blk = m.raw.data() + ((size_t) r*nb + ib)*34;
            if (memcmp(h + 2*ib, blk, 2) != 0) { bad_rt++; }
            for (int j = 0; j < 32; ++j) {
                const int8_t q = (int8_t) blk[2 + j];
                if (q_at(h, l, nb, ib, j) != q || h_at(h, nb, ib, j) != (q >> 2)) { bad_rt++; }
            }
        }
    }
    CHECK(bad_rt == 0, "%s: %" PRId64 " round-trip mismatches", label, bad_rt);

    std::vector<float> rn(nr);
    for (int64_t r = 0; r < nr; ++r) { rn[r] = row_lo_norm_q8_0(m.raw.data() + (size_t) r*nb*34, nb); }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(ne0), s8(ne0/8), a8(ne0/8), F(nr), L(nr), H(nr);
    std::vector<int> hist(H2_NBIN_TEST);
    double sum_cand[4] = {0, 0, 0, 0}, sum_width = 0, sum_cand_mass[2] = {0, 0};
    const int Ks[4] = {1, 20, 40, 64};
    for (int c = 0; c < ncols; ++c) {
        // columns: Gaussian, heavy-tailed, a few huge outlier channels (RMS-normed states have them), and one that
        // aligns its signs with the low bits of row 0 (worst case for the low-bit term)
        for (int64_t i = 0; i < ne0; ++i) {
            float v = nd(rng);
            if (c % 4 == 1) { v = v*v*v; }
            if (c % 4 == 2 && i % 97 == 0) { v *= 40.0f; }
            x[i] = v;
        }
        if (c % 4 == 3) {
            for (int64_t i = 0; i < ne0; ++i) {
                const int l = (int) ((uint8_t) m.raw[(size_t) (i/32)*34 + 2 + i%32] & 3);
                x[i] = std::fabs(x[i])*(l >= 2 ? 1.0f : -1.0f);
            }
        }
        group_sums(x.data(), ne0, s8.data(), a8.data());
        const float xn = col_norm(x.data(), ne0);
        int64_t bad_f = 0, bad_r = 0;
        for (int64_t r = 0; r < nr; ++r) {
            const uint8_t * h = hi.data() + r*hi_row_bytes(nb);
            const uint8_t * l = lo.data() + r*lo_row_bytes(nb);
            F[r] = dot_lane_order([&](int ib, int j) { return q_at(h, l, nb, ib, j); }, [&](int ib) { return scale_at(h, ib); }, x.data(), nb);
            double R = 0.0;
            for (int ib = 0; ib < nb; ++ib) {
                double s = 0.0;
                for (int j = 0; j < 32; ++j) { s += (double) q_at(h, l, nb, ib, j)*x[ib*32 + j]; }
                R += (double) scale_at(h, ib)*s;
            }
            screen_gpu_model(h, nb, x.data(), s8.data(), a8.data(), rn[r], xn, L[r], H[r]);
            float hl, hh;
            screen_row(h, nb, x.data(), s8.data(), a8.data(), rn[r], xn, hl, hh);
            if (!(L[r] <= F[r] && F[r] <= H[r]) || !(hl <= F[r] && F[r] <= hh)) { bad_f++; }
            if (!(L[r] <= R && R <= H[r])) { bad_r++; }
            sum_width += H[r] - L[r];
        }
        CHECK(bad_f == 0, "%s col %d: %" PRId64 " float logits outside [lo, hi]", label, c, bad_f);
        CHECK(bad_r == 0, "%s col %d: %" PRId64 " exact dots outside [lo, hi]", label, c, bad_r);

        // 3. selection
        std::vector<float> Fs(F);
        std::sort(Fs.begin(), Fs.end(), std::greater<float>());
        for (int ki = 0; ki < 4; ++ki) {
            const int K = Ks[ki];
            const float tau = tau_topk(L.data(), nr, K, 32.0f, H2_NBIN_TEST, hist.data());
            int64_t ncand = 0, bad = 0;
            for (int64_t r = 0; r < nr; ++r) {
                const bool cand = !(H[r] < tau);
                ncand += cand;
                if (!cand && !(F[r] < tau)) { bad++; }
                if (!cand && K <= nr && F[r] >= Fs[K - 1]) { bad++; }
            }
            CHECK(bad == 0, "%s col %d K %d: %" PRId64 " dropped rows reach the top K", label, c, K, bad);
            sum_cand[ki] += (double) ncand;
        }
        for (int si = 0; si < 2; ++si) {
            const float cap = si ? 30.0f : 0.0f;
            const float delta = (float) (std::log((double) nr) + 53.0*std::log(2.0) + 1.0);
            float M = -INFINITY;
            for (int64_t r = 0; r < nr; ++r) { M = std::max(M, L[r]); }
            const float tau = tau_mass(M, delta, cap);
            auto capf = [&](double v) { return cap > 0 ? cap*std::tanh(v/cap) : v; };
            double fmax = -INFINITY;
            for (int64_t r = 0; r < nr; ++r) { fmax = std::max(fmax, capf(F[r])); }
            double z = 0.0, dropped = 0.0;
            int64_t ncand = 0;
            for (int64_t r = 0; r < nr; ++r) {
                const double p = std::exp(capf(F[r]) - fmax);
                z += p;
                if (H[r] < tau) { dropped += p; } else { ncand++; }
            }
            CHECK(dropped <= z*std::ldexp(1.0, -53), "%s col %d: mass mode drops %.3g of the mass", label, c, dropped/z);
            sum_cand_mass[si] += (double) ncand;
        }
    }
    printf("%-14s rows %7" PRId64 " nb %3d cols %3d: mean interval %.3f; rows kept per column K=1 %.1f K=20 %.1f K=40 %.1f K=64 %.1f; mass %.0f (cap 30: %.0f)\n",
            label, nr, nb, ncols, sum_width/((double) nr*ncols), sum_cand[0]/ncols, sum_cand[1]/ncols, sum_cand[2]/ncols, sum_cand[3]/ncols,
            sum_cand_mass[0]/ncols, sum_cand_mass[1]/ncols);
}

static void run_q6k(std::mt19937 & rng) {
    const int nbk = 20; // 5120 columns
    std::uniform_int_distribution<int> ub(0, 255);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int64_t bad = 0;
    for (int it = 0; it < 400; ++it) {
        std::vector<uint8_t> row((size_t) nbk*210);
        for (auto & b : row) { b = (uint8_t) ub(rng); }
        for (int b = 0; b < nbk; ++b) {
            const uint16_t dh = f32_to_fp16(1e-4f*(0.5f + std::fabs(nd(rng)))*(it % 3 == 0 ? -1.0f : 1.0f));
            memcpy(row.data() + 210*b + 208, &dh, 2);
        }
        std::vector<float> x((size_t) nbk*256);
        for (auto & v : x) { v = nd(rng)*(it % 2 ? 1.0f : 30.0f); }
        double R = 0.0;
        float  F = 0.0f;
        for (int b = 0; b < nbk; ++b) {
            const uint8_t * blk = row.data() + 210*b;
            uint16_t dh; memcpy(&dh, blk + 208, 2);
            const float d = fp16_to_f32(dh);
            const int8_t * sc = (const int8_t *) (blk + 192);
            for (int i = 0; i < 256; ++i) {
                const float w = d*sc[q6k_scale_index(i)]*(q6k_code(blk, i) - 32);
                R += (double) d*sc[q6k_scale_index(i)]*(q6k_code(blk, i) - 32)*x[256*b + i];
                F = std::fma(w, x[256*b + i], F);
            }
        }
        float lo, hi;
        screen_row_q6_K(row.data(), nbk, x.data(), lo, hi);
        if (!(lo <= R && R <= hi && lo <= F && F <= hi)) { bad++; }
    }
    CHECK(bad == 0, "Q6_K: %" PRId64 " dots outside [lo, hi]", bad);
    printf("Q6_K bound: 400 rows x 5120, %s\n", bad == 0 ? "all inside" : "FAIL");
}

int main(int argc, char ** argv) {
    std::mt19937 rng(1234);
    {
        q8_matrix m;
        make_random(m, 4096, 72, rng);
        run_matrix("random nb72", m, 16, rng);
        make_random(m, 1024, 160, rng);
        run_matrix("random nb160", m, 8, rng);
    }
    if (argc > 1) {
        q8_matrix m;
        const int64_t rows = argc > 2 ? atoll(argv[2]) : 20000;
        if (!load_gguf_head(argv[1], rows, m)) {
            return 2;
        }
        run_matrix("gguf head", m, 32, rng);
    }
    run_q6k(rng);
    if (g_fail) {
        printf("test-head2: FAIL (%d)\n", g_fail);
        return 1;
    }
    printf("test-head2: OK\n");
    return 0;
}
