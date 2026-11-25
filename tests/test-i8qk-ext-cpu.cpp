// CPU-only numerical verification for i8qk extensions:
//   1. Exact log2 online softmax + global K-mean restoration (Kcs) + sinks + LSE + per-head masks + arbitrary n_kv tails
//   2. Two-stream LSE/sink combine (SWA + Full merge correctness)
//   3. Full INT8 Q/K + K-mean smoothing + FP16 P/V simulation with Outl100x outlier channels (D=128 & D=256)
//
// Strictly CPU-only: never initializes or touches any CUDA device.

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2   = 0.6931471805599453f;
constexpr int   kBM    = 128;
constexpr int   kBN    = 64;

inline float r_f16(float x) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
}

inline int8_t quant_i8(float x, float inv) {
    int q = static_cast<int>(std::nearbyint(static_cast<double>(x) * static_cast<double>(inv)));
    q = std::max(-127, std::min(127, q));
    return static_cast<int8_t>(q);
}

struct RefOut {
    std::vector<double> out; // [n_q * h_q * D]
    std::vector<double> lse; // [n_q * h_q]
};

static RefOut eval_fp64_reference(
        int D, int n_q, int n_kv, int h_q, int h_kv, int mask_ne2,
        float scale,
        const std::vector<float> & Q,
        const std::vector<float> & K,
        const std::vector<float> & V,
        const std::vector<float> & mask, // empty = unmasked; else [n_q * n_kv * mask_ne2]
        const std::vector<float> & sinks // empty = none; else [h_q]
) {
    const int gqa = h_q / h_kv;
    RefOut res;
    res.out.assign(static_cast<size_t>(n_q) * h_q * D, 0.0);
    res.lse.assign(static_cast<size_t>(n_q) * h_q, -std::numeric_limits<double>::infinity());

    for (int h = 0; h < h_q; ++h) {
        const int kvh = h / gqa;
        const int mh  = mask_ne2 > 1 ? h % mask_ne2 : 0;
        const bool has_sink = !sinks.empty();
        const double sink_v = has_sink ? static_cast<double>(sinks[h]) : -std::numeric_limits<double>::infinity();

        for (int q = 0; q < n_q; ++q) {
            double mx = sink_v;
            std::vector<double> logits(n_kv);
            for (int k = 0; k < n_kv; ++k) {
                double dot = 0.0;
                for (int c = 0; c < D; ++c) {
                    dot += static_cast<double>(Q[(static_cast<size_t>(h) * n_q + q) * D + c]) *
                           static_cast<double>(K[(static_cast<size_t>(kvh) * n_kv + k) * D + c]);
                }
                double m = mask.empty() ? 0.0 : static_cast<double>(mask[(static_cast<size_t>(mh) * n_q + q) * n_kv + k]);
                double s = dot * static_cast<double>(scale) + m;
                logits[k] = s;
                if (s > mx) {
                    mx = s;
                }
            }
            if (!std::isfinite(mx) && mx < 0.0) {
                res.lse[static_cast<size_t>(q) * h_q + h] = -std::numeric_limits<double>::infinity();
                continue;
            }
            double lsum = (has_sink && std::isfinite(sink_v)) ? std::exp(sink_v - mx) : 0.0;
            std::vector<double> acc(D, 0.0);
            for (int k = 0; k < n_kv; ++k) {
                if (std::isfinite(logits[k]) || logits[k] > 0.0) {
                    double p = std::exp(logits[k] - mx);
                    lsum += p;
                    for (int c = 0; c < D; ++c) {
                        acc[c] += p * static_cast<double>(V[(static_cast<size_t>(kvh) * n_kv + k) * D + c]);
                    }
                }
            }
            res.lse[static_cast<size_t>(q) * h_q + h] = lsum > 0.0 ? (mx + std::log(lsum)) : -std::numeric_limits<double>::infinity();
            const double inv = lsum > 0.0 ? 1.0 / lsum : 0.0;
            for (int c = 0; c < D; ++c) {
                res.out[(static_cast<size_t>(q) * h_q + h) * D + c] = acc[c] * inv;
            }
        }
    }
    return res;
}

struct I8SimOut {
    std::vector<float> out;
    std::vector<float> lse;
};

// Simulates exact i8qk kernel control flow:
//   - per-(mh, qt, kt) tile flags (0=all -inf skip, 1=all 0 fast, 2=mixed)
//   - shifted scores S' = Q.(K - mu) * scale_log2 in log2 domain
//   - tail masking k >= n_kv
//   - epilogue Kcs restoration (optional flag to test broken vs fixed), sink fold, LSE output
static I8SimOut eval_i8qk_sim(
        int D, int n_q, int n_kv, int h_q, int h_kv, int mask_ne2,
        float scale,
        const std::vector<float> & Q,
        const std::vector<float> & K,
        const std::vector<float> & V,
        const std::vector<float> & mask,
        const std::vector<float> & sinks,
        bool quantize_qk,
        bool pv16,
        bool smooth_k,
        bool restore_kcs
) {
    const int gqa = h_q / h_kv;
    const float scale_log2 = scale * kLog2E;

    // 1. K-mean per head and INT8 K quantization
    std::vector<float> kmean(static_cast<size_t>(h_kv) * D, 0.0f);
    if (smooth_k) {
        for (int kvh = 0; kvh < h_kv; ++kvh) {
            for (int c = 0; c < D; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < n_kv; ++k) {
                    sum += K[(static_cast<size_t>(kvh) * n_kv + k) * D + c];
                }
                const float m = sum / n_kv;
                kmean[static_cast<size_t>(kvh) * D + c] = std::isfinite(m) ? m : 0.0f;
            }
        }
    }

    const int n_ktiles = (n_kv + kBN - 1) / kBN;
    const int n_kv_pad = n_ktiles * kBN;
    std::vector<int8_t> Kq(static_cast<size_t>(h_kv) * n_kv * D, 0);
    std::vector<float>  Ksc(static_cast<size_t>(h_kv) * n_kv_pad, 0.0f);

    if (quantize_qk) {
        for (int kvh = 0; kvh < h_kv; ++kvh) {
            for (int k = 0; k < n_kv; ++k) {
                float amax = 0.0f;
                std::vector<float> x(D);
                for (int c = 0; c < D; ++c) {
                    x[c] = K[(static_cast<size_t>(kvh) * n_kv + k) * D + c] - kmean[static_cast<size_t>(kvh) * D + c];
                    amax = std::fmax(amax, std::fabs(x[c]));
                }
                const bool fin = std::isfinite(amax);
                const float inv = (fin && amax > 0.0f) ? 127.0f / amax : 0.0f;
                for (int c = 0; c < D; ++c) {
                    Kq[(static_cast<size_t>(kvh) * n_kv + k) * D + c] = quant_i8(x[c], inv);
                }
                Ksc[static_cast<size_t>(kvh) * n_kv_pad + k] = fin ? amax / 127.0f : 0.0f;
            }
        }
    }

    // 2. Tile flags per (mh, qt, kt)
    const int n_qtiles = (n_q + kBM - 1) / kBM;
    std::vector<uint8_t> flags(static_cast<size_t>(mask_ne2) * n_qtiles * n_ktiles, 1);
    if (!mask.empty()) {
        for (int mh = 0; mh < mask_ne2; ++mh) {
            for (int qt = 0; qt < n_qtiles; ++qt) {
                for (int kt = 0; kt < n_ktiles; ++kt) {
                    bool all_ninf = true;
                    bool all_zero = true;
                    for (int r = 0; r < kBM; ++r) {
                        const int q = qt * kBM + r;
                        if (q >= n_q) continue;
                        for (int kc = 0; kc < kBN; ++kc) {
                            const int k = kt * kBN + kc;
                            if (k >= n_kv) continue;
                            const float mv = mask[(static_cast<size_t>(mh) * n_q + q) * n_kv + k];
                            all_ninf &= (std::isinf(mv) && mv < 0.0f);
                            all_zero &= (mv == 0.0f);
                        }
                    }
                    flags[(static_cast<size_t>(mh) * n_qtiles + qt) * n_ktiles + kt] = all_ninf ? 0 : (all_zero ? 1 : 2);
                }
            }
        }
    }

    // 3. Attention per (qt, h)
    I8SimOut res;
    res.out.assign(static_cast<size_t>(n_q) * h_q * D, 0.0f);
    res.lse.assign(static_cast<size_t>(n_q) * h_q, -std::numeric_limits<float>::infinity());

    for (int h = 0; h < h_q; ++h) {
        const int kvh = h / gqa;
        const int mh  = mask_ne2 > 1 ? h % mask_ne2 : 0;
        for (int q = 0; q < n_q; ++q) {
            const int qt = q / kBM;

            // Q row quant + Kcs capture
            std::vector<int8_t> qq(D, 0);
            float sq = 0.0f;
            if (quantize_qk) {
                float amax = 0.0f;
                for (int c = 0; c < D; ++c) {
                    amax = std::fmax(amax, std::fabs(Q[(static_cast<size_t>(h) * n_q + q) * D + c]));
                }
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                for (int c = 0; c < D; ++c) {
                    qq[c] = quant_i8(Q[(static_cast<size_t>(h) * n_q + q) * D + c], inv);
                }
                sq = (amax / 127.0f) * scale_log2;
            }

            float kcs = 0.0f;
            if (smooth_k) {
                float qmu = 0.0f;
                for (int c = 0; c < D; ++c) {
                    qmu += Q[(static_cast<size_t>(h) * n_q + q) * D + c] * kmean[static_cast<size_t>(kvh) * D + c];
                }
                kcs = qmu * scale_log2;
            }

            float m = -std::numeric_limits<float>::infinity();
            float l = 0.0f;
            std::vector<float> O(D, 0.0f);

            for (int kt = 0; kt < n_ktiles; ++kt) {
                const uint8_t flag = flags[(static_cast<size_t>(mh) * n_qtiles + qt) * n_ktiles + kt];
                if (flag == 0) continue;

                const int k_base = kt * kBN;
                const bool partial_k = k_base + kBN > n_kv;
                float Sf[kBN];
                float mx = -std::numeric_limits<float>::infinity();

                for (int kc = 0; kc < kBN; ++kc) {
                    const int k = k_base + kc;
                    if (partial_k && k >= n_kv) {
                        Sf[kc] = -std::numeric_limits<float>::infinity();
                    } else {
                        float score_log2 = 0.0f;
                        if (quantize_qk) {
                            int32_t acc_s32 = 0;
                            for (int c = 0; c < D; ++c) {
                                acc_s32 += static_cast<int32_t>(qq[c]) * static_cast<int32_t>(Kq[(static_cast<size_t>(kvh) * n_kv + k) * D + c]);
                            }
                            score_log2 = static_cast<float>(acc_s32) * (sq * Ksc[static_cast<size_t>(kvh) * n_kv_pad + k]);
                        } else {
                            float dot = 0.0f;
                            for (int c = 0; c < D; ++c) {
                                const float k_s = K[(static_cast<size_t>(kvh) * n_kv + k) * D + c] - kmean[static_cast<size_t>(kvh) * D + c];
                                dot += Q[(static_cast<size_t>(h) * n_q + q) * D + c] * k_s;
                            }
                            score_log2 = dot * scale_log2;
                        }
                        if (flag == 2 && !mask.empty()) {
                            const float mv = mask[(static_cast<size_t>(mh) * n_q + q) * n_kv + k];
                            score_log2 += mv * kLog2E;
                        }
                        Sf[kc] = score_log2;
                    }
                    mx = std::fmax(mx, Sf[kc]);
                }

                const float mn = std::fmax(m, mx);
                const float mu = (mn == -std::numeric_limits<float>::infinity()) ? 0.0f : mn;
                const float alpha = std::exp2(m - mu);
                const float alpha_h = pv16 ? r_f16(alpha) : alpha;
                m = mn;
                l *= alpha;
                for (int c = 0; c < D; ++c) {
                    O[c] = pv16 ? r_f16(O[c] * alpha_h) : (O[c] * alpha);
                }

                float P[kBN];
                for (int kc = 0; kc < kBN; ++kc) {
                    const float p = std::exp2(Sf[kc] - mu);
                    l += p;
                    P[kc] = pv16 ? r_f16(p) : p;
                }
                for (int c = 0; c < D; ++c) {
                    float pv = 0.0f;
                    for (int kc = 0; kc < kBN; ++kc) {
                        const int k = k_base + kc;
                        if (k < n_kv) {
                            const float v = pv16 ? r_f16(V[(static_cast<size_t>(kvh) * n_kv + k) * D + c])
                                                 : V[(static_cast<size_t>(kvh) * n_kv + k) * D + c];
                            pv += P[kc] * v;
                        }
                    }
                    O[c] = pv16 ? r_f16(O[c] + pv) : (O[c] + pv);
                }
            }

            if (restore_kcs && smooth_k && m != -std::numeric_limits<float>::infinity()) {
                m += kcs;
            }

            float il = l > 0.0f ? 1.0f / l : 0.0f;
            if (!sinks.empty()) {
                const float s2   = sinks[h] * kLog2E;
                const float mn   = std::fmax(m, s2);
                const float sa   = (m == -std::numeric_limits<float>::infinity() || mn == -std::numeric_limits<float>::infinity())
                                       ? 0.0f : std::exp2(m - mn);
                const float sadd = (s2 == -std::numeric_limits<float>::infinity() || mn == -std::numeric_limits<float>::infinity())
                                       ? 0.0f : std::exp2(s2 - mn);
                l  = l * sa + sadd;
                m  = mn;
                il = l > 0.0f ? sa / l : 0.0f;
            }
            res.lse[static_cast<size_t>(q) * h_q + h] = l > 0.0f ? (m * kLn2 + std::log(l)) : -std::numeric_limits<float>::infinity();
            for (int c = 0; c < D; ++c) {
                res.out[(static_cast<size_t>(q) * h_q + h) * D + c] = O[c] * il;
            }
        }
    }
    return res;
}

static double max_err_vec(const std::vector<float> & got, const std::vector<double> & ref) {
    double me = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::isinf(got[i]) && std::isinf(ref[i]) && ((got[i] < 0) == (ref[i] < 0))) {
            continue;
        }
        me = std::max(me, std::abs(static_cast<double>(got[i]) - ref[i]));
    }
    return me;
}

static double cosine_sim(const std::vector<float> & got, const std::vector<double> & ref) {
    double dot = 0.0, ng = 0.0, nr = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = static_cast<double>(got[i]);
        const double r = ref[i];
        dot += g * r;
        ng  += g * g;
        nr  += r * r;
    }
    return (ng > 0.0 && nr > 0.0) ? dot / (std::sqrt(ng) * std::sqrt(nr)) : 1.0;
}

} // namespace

int main() {
    std::mt19937 rng(20261123);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // -------------------------------------------------------------------------
    // Test 1: Exact softmax + Kcs mean restoration + sinks + LSE + per-head bias + odd tail n_kv
    // -------------------------------------------------------------------------
    {
        const int D = 128, n_q = 19, n_kv = 131, h_q = 6, h_kv = 2, mask_ne2 = 6;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> Q(static_cast<size_t>(h_q) * n_q * D);
        std::vector<float> K(static_cast<size_t>(h_kv) * n_kv * D);
        std::vector<float> V(static_cast<size_t>(h_kv) * n_kv * D);
        std::vector<float> mask(static_cast<size_t>(mask_ne2) * n_q * n_kv);
        std::vector<float> sinks = {-25.0f, 0.0f, 3.5f, 18.0f, -std::numeric_limits<float>::infinity(), 1.25f};

        for (float & v : Q) v = nd(rng);
        for (int kvh = 0; kvh < h_kv; ++kvh) {
            for (int k = 0; k < n_kv; ++k) {
                for (int c = 0; c < D; ++c) {
                    // Add huge channel mean bias on channels 0..7 so broken un-restored Kcs fails loudly!
                    float bias = (c < 8) ? ((c % 2 == 0 ? 35.0f : -28.0f)) : 0.0f;
                    K[(static_cast<size_t>(kvh) * n_kv + k) * D + c] = r_f16(bias + nd(rng) * 0.6f);
                    V[(static_cast<size_t>(kvh) * n_kv + k) * D + c] = r_f16(nd(rng));
                }
            }
        }
        for (int mh = 0; mh < mask_ne2; ++mh) {
            for (int q = 0; q < n_q; ++q) {
                for (int k = 0; k < n_kv; ++k) {
                    // Make row q == 0 completely masked (tests all -inf + sink!), tile 1 (k in 64..127) skipped or mixed
                    float m = 0.0f;
                    if (q == 0 || (q < 8 && k >= 64 && k < 128 && mh % 2 == 0) || k > q + 115) {
                        m = -std::numeric_limits<float>::infinity();
                    } else if (mh > 0 && k > 0) {
                        // per-head relative bias slope
                        m = r_f16(-0.02f * static_cast<float>(mh) * static_cast<float>(std::abs(q - k) % 17));
                    }
                    mask[(static_cast<size_t>(mh) * n_q + q) * n_kv + k] = m;
                }
            }
        }

        const RefOut ref = eval_fp64_reference(D, n_q, n_kv, h_q, h_kv, mask_ne2, scale, Q, K, V, mask, sinks);
        const I8SimOut fixed = eval_i8qk_sim(D, n_q, n_kv, h_q, h_kv, mask_ne2, scale, Q, K, V, mask, sinks,
                                             false, false, true, true);
        const I8SimOut broken = eval_i8qk_sim(D, n_q, n_kv, h_q, h_kv, mask_ne2, scale, Q, K, V, mask, sinks,
                                              false, false, true, false);

        const double err_out = max_err_vec(fixed.out, ref.out);
        const double err_lse = max_err_vec(fixed.lse, ref.lse);
        const double brk_lse = max_err_vec(broken.lse, ref.lse);

        std::printf("[Test 1 Exact Log2+Kcs+Sinks+LSE+Tail] max_err_out=%.3e max_err_lse=%.3e (without Kcs=%.3e)\n",
                    err_out, err_lse, brk_lse);
        if (!(err_out < 1e-5 && err_lse < 1e-5 && brk_lse > 1.0)) {
            std::fprintf(stderr, "FAIL Test 1: out=%.3e lse=%.3e broken_lse=%.3e\n", err_out, err_lse, brk_lse);
            return 1;
        }
        // Check row q == 0 fully masked with finite sink vs -inf sink
        if (std::fabs(fixed.lse[4] - (-std::numeric_limits<float>::infinity())) > 0.0f &&
                !std::isinf(fixed.lse[4])) {
            std::fprintf(stderr, "FAIL Test 1: all-masked row + -inf sink expected -inf lse\n");
            return 1;
        }
        if (std::fabs(fixed.lse[3] - 18.0f) > 1e-4f) {
            std::fprintf(stderr, "FAIL Test 1: all-masked row + sink=18 expected lse=18, got %g\n", fixed.lse[3]);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: Two-stream inkrel_merge (SWA with per-head mask+sink+LSE + Full with LSE)
    // -------------------------------------------------------------------------
    {
        const int D = 128, n_q = 12, n_kv_a = 131, n_kv_b = 128, n_kv = n_kv_a + n_kv_b;
        const int h_q = 6, h_kv = 2, mask_ne2 = 6;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> Q(static_cast<size_t>(h_q) * n_q * D);
        std::vector<float> K(static_cast<size_t>(h_kv) * n_kv * D);
        std::vector<float> V(static_cast<size_t>(h_kv) * n_kv * D);
        std::vector<float> mask(static_cast<size_t>(mask_ne2) * n_q * n_kv, 0.0f);
        std::vector<float> sinks = {0.5f, 4.2f, -3.0f, 1.1f, -0.7f, 2.8f};

        for (float & v : Q) v = nd(rng) * 0.5f;
        for (float & v : K) v = r_f16(nd(rng) + 6.0f);
        for (float & v : V) v = r_f16(nd(rng));
        for (int mh = 0; mh < mask_ne2; ++mh) {
            for (int q = 0; q < n_q; ++q) {
                for (int k = 0; k < n_kv; ++k) {
                    mask[(static_cast<size_t>(mh) * n_q + q) * n_kv + k] = r_f16(-0.015f * ((mh + q + k) % 13));
                }
            }
        }

        // Split K, V, mask into A [0..n_kv_a) and B [n_kv_a..n_kv)
        auto slice_kv = [&](int k0, int len, const std::vector<float> & src) {
            std::vector<float> dst(static_cast<size_t>(h_kv) * len * D);
            for (int kvh = 0; kvh < h_kv; ++kvh) {
                for (int k = 0; k < len; ++k) {
                    for (int c = 0; c < D; ++c) {
                        dst[(static_cast<size_t>(kvh) * len + k) * D + c] =
                            src[(static_cast<size_t>(kvh) * n_kv + k0 + k) * D + c];
                    }
                }
            }
            return dst;
        };
        auto slice_mask = [&](int k0, int len) {
            std::vector<float> dst(static_cast<size_t>(mask_ne2) * n_q * len);
            for (int mh = 0; mh < mask_ne2; ++mh) {
                for (int q = 0; q < n_q; ++q) {
                    for (int k = 0; k < len; ++k) {
                        dst[(static_cast<size_t>(mh) * n_q + q) * len + k] =
                            mask[(static_cast<size_t>(mh) * n_q + q) * n_kv + k0 + k];
                    }
                }
            }
            return dst;
        };

        const auto Ka = slice_kv(0, n_kv_a, K), Va = slice_kv(0, n_kv_a, V), ma = slice_mask(0, n_kv_a);
        const auto Kb = slice_kv(n_kv_a, n_kv_b, K), Vb = slice_kv(n_kv_a, n_kv_b, V), mb = slice_mask(n_kv_a, n_kv_b);

        const RefOut ref = eval_fp64_reference(D, n_q, n_kv, h_q, h_kv, mask_ne2, scale, Q, K, V, mask, sinks);
        const I8SimOut simA = eval_i8qk_sim(D, n_q, n_kv_a, h_q, h_kv, mask_ne2, scale, Q, Ka, Va, ma, sinks,
                                            false, false, true, true);
        const I8SimOut simB = eval_i8qk_sim(D, n_q, n_kv_b, h_q, h_kv, mask_ne2, scale, Q, Kb, Vb, mb, {},
                                            false, false, true, true);

        // Two-stream LSE combine matching inkrel_merge: w_a = 1/(1 + exp(lse_b - lse_a)) or exp(lse - mx)
        std::vector<float> merged_out(ref.out.size());
        std::vector<float> merged_lse(ref.lse.size());
        for (int q = 0; q < n_q; ++q) {
            for (int h = 0; h < h_q; ++h) {
                const size_t idx = static_cast<size_t>(q) * h_q + h;
                const float la = simA.lse[idx];
                const float lb = simB.lse[idx];
                const float lm = std::fmax(la, lb);
                const float ea = std::exp(la - lm);
                const float eb = std::exp(lb - lm);
                const float inv = 1.0f / (ea + eb);
                merged_lse[idx] = lm + std::log(ea + eb);
                for (int c = 0; c < D; ++c) {
                    merged_out[idx * D + c] = (ea * simA.out[idx * D + c] + eb * simB.out[idx * D + c]) * inv;
                }
            }
        }
        const double err_m_out = max_err_vec(merged_out, ref.out);
        const double err_m_lse = max_err_vec(merged_lse, ref.lse);
        std::printf("[Test 2 Two-stream LSE+Sink merge] max_err_out=%.3e max_err_lse=%.3e\n", err_m_out, err_m_lse);
        if (!(err_m_out < 1e-5 && err_m_lse < 1e-5)) {
            std::fprintf(stderr, "FAIL Test 2: out=%.3e lse=%.3e\n", err_m_out, err_m_lse);
            return 1;
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: INT8 Q/K + K-mean smoothing + FP16 PV16 with Outl100x channels (D=128 & D=256)
    // -------------------------------------------------------------------------
    for (int D : {128, 256}) {
        const int n_q = 32, n_kv = 195, h_q = 6, h_kv = 2, mask_ne2 = 6;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        std::vector<float> Q(static_cast<size_t>(h_q) * n_q * D);
        std::vector<float> K(static_cast<size_t>(h_kv) * n_kv * D);
        std::vector<float> V(static_cast<size_t>(h_kv) * n_kv * D);
        std::vector<float> mask(static_cast<size_t>(mask_ne2) * n_q * n_kv, 0.0f);
        std::vector<float> sinks = {-1.5f, 0.8f, 2.1f, -4.0f, 1.4f, 0.0f};

        for (float & v : Q) v = nd(rng) * 0.8f;
        for (int kvh = 0; kvh < h_kv; ++kvh) {
            for (int k = 0; k < n_kv; ++k) {
                for (int c = 0; c < D; ++c) {
                    // Outl100x channels: 4 channels with massive constant mean +/- 35.0
                    float bias = (c < 4) ? (c % 2 == 0 ? 35.0f : -35.0f) : 0.0f;
                    float amp  = (c < 4) ? 0.5f : 0.9f;
                    K[(static_cast<size_t>(kvh) * n_kv + k) * D + c] = r_f16(bias + nd(rng) * amp);
                    V[(static_cast<size_t>(kvh) * n_kv + k) * D + c] = r_f16(nd(rng) * 0.9f);
                }
            }
        }
        for (int mh = 0; mh < mask_ne2; ++mh) {
            for (int q = 0; q < n_q; ++q) {
                for (int k = 0; k < n_kv; ++k) {
                    mask[(static_cast<size_t>(mh) * n_q + q) * n_kv + k] =
                        (k > q + 150) ? -std::numeric_limits<float>::infinity()
                                      : r_f16(-0.01f * static_cast<float>((mh + 1) * (std::abs(q - k) % 11)));
                }
            }
        }

        const RefOut ref      = eval_fp64_reference(D, n_q, n_kv, h_q, h_kv, mask_ne2, scale, Q, K, V, mask, sinks);
        const I8SimOut smooth = eval_i8qk_sim(D, n_q, n_kv, h_q, h_kv, mask_ne2, scale, Q, K, V, mask, sinks,
                                              true, true, true, true);
        const I8SimOut nosm   = eval_i8qk_sim(D, n_q, n_kv, h_q, h_kv, mask_ne2, scale, Q, K, V, mask, sinks,
                                              true, true, false, false);

        const double cos_sm  = cosine_sim(smooth.out, ref.out);
        const double cos_no  = cosine_sim(nosm.out, ref.out);
        const double lse_sm  = max_err_vec(smooth.lse, ref.lse);
        const double lse_no  = max_err_vec(nosm.lse, ref.lse);

        std::printf("[Test 3 Outl100x INT8+PV16 D=%d] smooth cos=%.6f lse_max_err=%.4f | unsmoothed cos=%.6f lse_max_err=%.4f\n",
                    D, cos_sm, lse_sm, cos_no, lse_no);
        if (!(cos_sm > 0.9985 && lse_sm < 0.08 && cos_sm > cos_no)) {
            std::fprintf(stderr, "FAIL Test 3 D=%d: cos_sm=%.6f lse_sm=%.4f cos_no=%.6f\n", D, cos_sm, lse_sm, cos_no);
            return 1;
        }
    }

    std::printf("ALL i8qk-ext CPU verification tests PASSED\n");
    return 0;
}
