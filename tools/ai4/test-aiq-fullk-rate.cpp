// EXLGAP: GPU test of ai4_cuda_encode_fullk_rate (full-K LDLQ for the AI+ unit-128 layout at rate 2..6) on synthetic data.
//   T1  rate 4 full-K reconstructs exactly like the AI4 1024-layout full-K path (same trellis, different byte layout)
//   T2  the encoder's internal H-weighted error (rotated basis) equals ai4_cuda_werr on the reconstruction (original basis),
//       full mode
//   T3  rate 5 full-K error < rate 4 full-K error
//   T4  rate 5 full-K error < rate 5 with no LDLQ, and full-K error < diagonal-only (mode 2) error on correlated inputs
//   T5  --refit: same encode before the refit, lower error after it, and the rewritten fp16 scales reproduce it
//   T6  optional weighted trellis metric: the internal score equals the full-H score of the stored reconstruction
// usage: llama-ai4-fullk-test [K M]   (K % 1024, M % 16). Exit code 0 = all pass.
#include "ai4-encode.h"
#include "ggml-ai4.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

int main(int argc, char ** argv) {
    const int64_t K = argc > 1 ? atoll(argv[1]) : 1024, M = argc > 2 ? atoll(argv[2]) : 64;
    const float lrms = ai4_level_rms(), smul = AI4_SCALE_MUL;   // the quantizer's values
    std::mt19937 rng(1013);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // correlated inputs: x_k = c_k (z_k + 0.8 z_{k-1} + 0.5 g), c_k log-normal channel scale, g a shared factor per sample
    std::vector<float> cs(K);
    for (auto & c : cs) c = std::exp(0.7f * nd(rng));
    // exact covariance of x (no sampling, O(K^2)): c_i c_j (1.64 [i == j] + 0.8 [|i - j| == 1] + 0.25)
    std::vector<float> H((size_t) K * K);
    for (int64_t i = 0; i < K; ++i) {
        for (int64_t j = 0; j < K; ++j) {
            const double t = (i == j ? 1.64 : 0.0) + (i - j == 1 || j - i == 1 ? 0.8 : 0.0) + 0.25;
            H[(size_t) i * K + j] = (float) (cs[i] * cs[j] * t);
        }
    }
    std::vector<float> W((size_t) M * K);
    for (int64_t r = 0; r < M; ++r) {
        const float rs = std::exp(0.3f * nd(rng)) * 0.02f;
        for (int64_t k = 0; k < K; ++k) W[(size_t) r * K + k] = rs * nd(rng);
    }

    const size_t nq0 = (size_t) (M / 16) * (K / 1024) * 8448, nq4 = (size_t) (M / 16) * (K / 128) * (32 + 256 * 4),
                 nq5 = (size_t) (M / 16) * (K / 128) * (32 + 256 * 5);
    std::vector<uint8_t> q0(nq0), q4(nq4), q5(nq5), q5d(nq5), q5n(nq5);
    std::vector<float> R0((size_t) M * K), R4(R0.size()), R5(R0.size()), R5d(R0.size()), R5n(R0.size());
    float ms = 0.0f;
    double w0 = -1, w4 = -1, w5 = -1, w5d = -1, e0 = -1, e4 = -1, e5 = -1, e5d = -1, e5n = -1;
    if (ai4_cuda_encode_fullk(W.data(), H.data(), K, M, 0, 0.01, lrms, smul, q0.data(), &ms, &w0)) return 1;
    if (ai4_cuda_encode_fullk_rate(W.data(), H.data(), K, M, 0, 0.01, lrms, smul, 4, AI4_MUL, 0, q4.data(), &ms, &w4, nullptr)) return 1;
    if (ai4_cuda_encode_fullk_rate(W.data(), H.data(), K, M, 0, 0.01, lrms, smul, 5, AI4_MUL, 0, q5.data(), &ms, &w5, nullptr)) return 1;
    if (ai4_cuda_encode_fullk_rate(W.data(), H.data(), K, M, 2, 0.01, lrms, smul, 5, AI4_MUL, 0, q5d.data(), &ms, &w5d, nullptr)) return 1;
    if (ai4_cuda_encode_rate(W.data(), K, M, nullptr, nullptr, lrms, smul, 5, AI4_MUL, q5n.data(), &ms)) return 1;
    // refit: rate 5 and the AI4 layout (kbits 0)
    std::vector<uint8_t> q5f(nq5), q0f(nq0);
    std::vector<float> R5f(R0.size()), R0f(R0.size());
    double w5f = -1, w5p = -1, w0f = -1, w0p = -1, e5f = -1, e0f = -1;
    if (ai4_cuda_encode_fullk_rate(W.data(), H.data(), K, M, 0, 0.01, lrms, smul, 5, AI4_MUL, 1, q5f.data(), &ms, &w5f, &w5p)) return 1;
    if (ai4_cuda_encode_fullk_rate(W.data(), H.data(), K, M, 0, 0.01, lrms, smul, 0, AI4_MUL, 1, q0f.data(), &ms, &w0f, &w0p)) return 1;
    if (ai4_cuda_reconstruct_rate(q5f.data(), K, M, 5, AI4_MUL, 0, R5f.data())) return 1;
    if (ai4_cuda_reconstruct(q0f.data(), K, M, R0f.data())) return 1;
    if (ai4_cuda_reconstruct(q0.data(), K, M, R0.data())) return 1;
    if (ai4_cuda_reconstruct_rate(q4.data(), K, M, 4, AI4_MUL, 0, R4.data())) return 1;
    if (ai4_cuda_reconstruct_rate(q5.data(), K, M, 5, AI4_MUL, 0, R5.data())) return 1;
    if (ai4_cuda_reconstruct_rate(q5d.data(), K, M, 5, AI4_MUL, 0, R5d.data())) return 1;
    if (ai4_cuda_reconstruct_rate(q5n.data(), K, M, 5, AI4_MUL, 0, R5n.data())) return 1;
    ai4_cuda_werr(W.data(), R0.data(), H.data(), K, M, &e0);
    ai4_cuda_werr(W.data(), R4.data(), H.data(), K, M, &e4);
    ai4_cuda_werr(W.data(), R5.data(), H.data(), K, M, &e5);
    ai4_cuda_werr(W.data(), R5d.data(), H.data(), K, M, &e5d);
    ai4_cuda_werr(W.data(), R5n.data(), H.data(), K, M, &e5n);
    ai4_cuda_werr(W.data(), R5f.data(), H.data(), K, M, &e5f);
    ai4_cuda_werr(W.data(), R0f.data(), H.data(), K, M, &e0f);
    printf("refit: k5 pre %.4e post %.4e (reconstructed %.4e)  ai4 pre %.4e post %.4e (reconstructed %.4e)\n", w5p, w5f, e5f, w0p, w0f, e0f);
    printf("K %lld M %lld  werr (internal / reconstructed): ai4 %.4e / %.4e  k4 %.4e / %.4e  k5 %.4e / %.4e  k5-diag %.4e / %.4e  "
           "k5-noLDLQ - / %.4e\n", (long long) K, (long long) M, w0, e0, w4, e4, w5, e5, w5d, e5d, e5n);

    double md = 0.0;
    for (size_t i = 0; i < R0.size(); ++i) md = std::fmax(md, std::fabs((double) R0[i] - R4[i]));
    printf("max |R_ai4 - R_k4| = %.3e\n", md);
    check(md == 0.0, "T1 rate-4 full-K == AI4 full-K reconstruction (bit exact)");
    auto rel = [](double a, double b) { return std::fabs(a - b) / std::fmax(std::fabs(b), 1e-30); };
    // Both scores use the full undamped Hessian, including the diagonal-only fit.
    check(rel(w0, e0) < 2e-3 && rel(w4, e4) < 2e-3 && rel(w5, e5) < 2e-3 && rel(w5d, e5d) < 2e-3,
          "T2 internal werr == reconstructed werr in full and diagonal modes (rel < 2e-3)");
    check(e5 < e4, "T3 rate 5 error < rate 4 error");
    check(e5 < e5n && e5 < e5d, "T4 rate 5 full-K error < no-LDLQ and < diagonal-only");
    check(rel(w5p, w5) < 1e-6 && rel(w0p, w0) < 1e-6, "T5a refit pre-error == plain full-K error (same encode)");
    check(w5f <= w5p && w0f <= w0p && e5f <= e5 * 1.0005 && e0f <= e0 * 1.0005, "T5b refit lowers the H-weighted error (internal; reconstructed within fp16 rounding)");
    check(rel(w5f, e5f) < 1e-5 && rel(w0f, e0f) < 1e-5, "T5c refit internal werr == reconstructed werr after fp16 rounding (rel < 1e-5)");
    std::vector<uint8_t> q5w(nq5);
    std::vector<float> R5w(R0.size());
    double w5w = -1.0, e5w = -1.0;
    if (ai4_cuda_encode_fullk_rate(W.data(), H.data(), K, M, 0, 0.01, lrms, smul, 5, AI4_MUL, 0,
                                  q5w.data(), &ms, &w5w, nullptr, 1)) return 1;
    if (ai4_cuda_reconstruct_rate(q5w.data(), K, M, 5, AI4_MUL, 0, R5w.data())) return 1;
    if (ai4_cuda_werr(W.data(), R5w.data(), H.data(), K, M, &e5w)) return 1;
    printf("weighted trellis: k5 full-H %.4e (internal %.4e), normalized %.4e, ratio %.4f\n", e5w, w5w, e5, e5w/e5);
    check(std::isfinite(e5w) && rel(w5w, e5w) < 2e-3, "T6 weighted trellis internal werr == full-H reconstructed werr");
    printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
