// AIBEST (private): synthetic GPU test of the AI+ rates k = 2, 3, 4, 5, 6 (ggml ids AI2_U / AI3_U / AI4_U / AI5_U / AI6_U
// plus the private A/B controls AI6_UO / AI5_UN). Needs no model file.
//   T1  per type: the CUDA reference reconstruct of CPU-encoded bytes is bit-exact against the CPU reference dequant
//       (rotated basis), and matches the unrotated CPU dequant to rotation round-off
//   T2  per type: the GPU encoder and the CPU reference encoder agree on small tiles (byte comparison reported; the gate
//       is GPU error <= 1.03 x CPU error)
//   T3  k = 4 at the AI4 constant: the generalized rate-k encoder reproduces the AI4_U encoder byte for byte
//   T4  weight error ordering k2 > k3 > k4 > k5 > k6 on a random Gaussian tensor
//   T5  AI6_U (0xA1EEE321, mul 1.15) vs AI6_UO (0x83DCD12D, mul 1.17): reported, expected AI6_U lower (not a gate)
//   T6  the span types AI2_S..AI6_S (one fp16 scale per row and 1024 K): a row-scaled AIk_U encode packs losslessly
//       (unpack(pack) == the source bytes), the CPU and CUDA span decoders are bit-identical to the AIk_U decode of the
//       same bytes, and the CPU span encoder + pack reproduces the GPU encoder + pack
//   T7  weighted metric: equal nonzero scales reproduce legacy GPU bytes at every rate; zero scales reconstruct zero;
//       weighted GPU bytes decode exactly on CPU/CUDA. GPU vs float-CPU weighted error is reported (half costs differ).
// usage: llama-ai4-rate-test [K M]   (K % 256, M % 16; default 256 64 for T1/T2, T4 and T6 use 1024 x 64). Exit 0 = pass.
#include "ai4-encode.h"
#include "ggml-ai4.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const char * what) {
    printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++g_fail;
}

static const ggml_type g_types[] = {
    GGML_TYPE_AI2_U, GGML_TYPE_AI3_U, GGML_TYPE_AI4_U, GGML_TYPE_AI5_U, GGML_TYPE_AI6_U,
    GGML_TYPE_AI6_UO, GGML_TYPE_AI5_UN,
};
static const int g_ntype = (int) (sizeof(g_types)/sizeof(g_types[0]));

// relative Frobenius error of Wh vs W
static double rel_err(const std::vector<float> & W, const std::vector<float> & Wh) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < W.size(); ++i) {
        const double d = (double) W[i] - (double) Wh[i];
        num += d*d;
        den += (double) W[i]*(double) W[i];
    }
    return den > 0.0 ? sqrt(num/den) : 0.0;
}

static void cpu_encode(ggml_type ty, const float * W, int64_t K, int64_t M, std::vector<uint8_t> & q) {
    const int      k = ggml_ai4_rate(ty);
    const uint32_t C = ggml_ai4_trellis_mul(ty);
    const float    s = ggml_ai4_scale_mul(ty);
    const size_t  sb = (size_t) ai4x_slab_bytes_k(K, k);
    q.assign(sb*(M/16), 0);
    ai4_work * ws = ai4_work_new();
    for (int64_t sl = 0; sl < M/16; ++sl) {
        aik_quantize_slab_rb_c(ws, k, C, W + (size_t) sl*16*K, K, nullptr, nullptr, nullptr,
                               q.data() + sl*sb, ai4u_rot_size(K), s);
    }
    free(ws);
}

int main(int argc, char ** argv) {
    const int64_t K = argc > 1 ? atoll(argv[1]) : 256, M = argc > 2 ? atoll(argv[2]) : 64;
    if (K % 256 || M % 16) { fprintf(stderr, "K %% 256 and M %% 16 required\n"); return 1; }
    std::mt19937 rng(20261013);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> W((size_t) M*K);
    for (int64_t r = 0; r < M; ++r) {
        const float rs = std::exp(0.3f*nd(rng))*0.02f;
        for (int64_t k = 0; k < K; ++k) W[(size_t) r*K + k] = rs*nd(rng);
    }

    // ---- T1 / T2: per type, CPU reference encode + CPU/GPU decode, then the GPU encoder on the same input ----
    for (int it = 0; it < g_ntype; ++it) {
        const ggml_type ty = g_types[it];
        const int       k  = ggml_ai4_rate(ty);
        const uint32_t  C  = ggml_ai4_trellis_mul(ty);
        const float     sm = ggml_ai4_scale_mul(ty);
        const char *    nm = ggml_type_name(ty);
        const size_t    sb = (size_t) ai4x_slab_bytes_k(K, k);

        std::vector<uint8_t> qc;
        cpu_encode(ty, W.data(), K, M, qc);

        std::vector<float> rot_cpu((size_t) M*K), rot_gpu((size_t) M*K), unr_cpu((size_t) M*K), unr_gpu((size_t) M*K);
        for (int64_t sl = 0; sl < M/16; ++sl) {
            aik_dequantize_slab_rotated_c(k, C, qc.data() + sl*sb, rot_cpu.data() + (size_t) sl*16*K, K);
            aik_dequantize_slab_c        (k, C, qc.data() + sl*sb, unr_cpu.data() + (size_t) sl*16*K, K);
        }
        if (ai4_cuda_reconstruct_rate(qc.data(), K, M, k, C, 1, rot_gpu.data())) return 1;
        if (ai4_cuda_reconstruct_rate(qc.data(), K, M, k, C, 0, unr_gpu.data())) return 1;

        char buf[256];
        snprintf(buf, sizeof(buf), "T1 %-6s k=%d C=%08X: CUDA reference reconstruct bit-exact vs CPU reference (rotated)",
                 nm, k, C);
        check(memcmp(rot_cpu.data(), rot_gpu.data(), rot_cpu.size()*sizeof(float)) == 0, buf);

        double umax = 0.0, rms = 0.0;
        for (size_t i = 0; i < unr_cpu.size(); ++i) {
            umax = std::max(umax, (double) fabsf(unr_cpu[i] - unr_gpu[i]));
            rms += (double) W[i]*(double) W[i];
        }
        rms = sqrt(rms/unr_cpu.size());
        snprintf(buf, sizeof(buf), "T1 %-6s unrotated max|diff|/rms = %.2e <= 1e-5", nm, umax/rms);
        check(umax <= 1e-5*rms, buf);

        // GPU encoder on the same slab set (no LDLQ feedback, so CPU and GPU do the same per-tile Viterbi)
        std::vector<uint8_t> qg(qc.size());
        float ms = 0.0f;
        if (ai4_cuda_encode_rate(W.data(), K, M, nullptr, nullptr, aik_level_rms_c(C), sm, k, C, qg.data(), &ms)) return 1;
        std::vector<float> rg((size_t) M*K);
        if (ai4_cuda_reconstruct_rate(qg.data(), K, M, k, C, 0, rg.data())) return 1;
        const double ec = rel_err(W, unr_cpu), eg = rel_err(W, rg);
        const bool   id = memcmp(qg.data(), qc.data(), qc.size()) == 0;
        snprintf(buf, sizeof(buf), "T2 %-6s GPU/CPU encoder: bytes %s, err gpu %.5e cpu %.5e ratio %.4f <= 1.03",
                 nm, id ? "IDENTICAL" : "differ", eg, ec, eg/ec);
        check(eg <= 1.03*ec, buf);
        printf("      %-6s %.3f bpw, unit %lld B, encode %.1f ms\n", nm,
               (32.0 + 256.0*k)*8.0/2048.0, (long long) AI4U_UNIT_BYTES(k), ms);
    }

    // ---- T3: k = 4 through the generalized kernel == the AI4_U encoder, byte for byte ----
    {
        const size_t sb = (size_t) ai4x_slab_bytes_k(K, 4);
        std::vector<uint8_t> q4a(sb*(M/16)), q4b(sb*(M/16));
        float ms = 0.0f;
        if (ai4_cuda_encode_rate(W.data(), K, M, nullptr, nullptr, ai4_level_rms(), AI4_SCALE_MUL, 4, AI4_MUL, q4a.data(), &ms)) return 1;
        if (ai4_cuda_encode_layout(W.data(), K, M, nullptr, nullptr, ai4_level_rms(), AI4_SCALE_MUL, 1, q4b.data(), &ms)) return 1;
        check(memcmp(q4a.data(), q4b.data(), q4a.size()) == 0, "T3 rate-k kernel at k=4 == AI4_U encoder (byte for byte)");
    }

    // ---- T4 / T5: weight error per rate on a larger Gaussian tensor ----
    {
        const int64_t K2 = 1024, M2 = 64;
        std::vector<float> W2((size_t) M2*K2);
        for (int64_t r = 0; r < M2; ++r) {
            const float rs = std::exp(0.3f*nd(rng))*0.02f;
            for (int64_t k = 0; k < K2; ++k) W2[(size_t) r*K2 + k] = rs*nd(rng);
        }
        double err[g_ntype];
        for (int it = 0; it < g_ntype; ++it) {
            const ggml_type ty = g_types[it];
            const int       k  = ggml_ai4_rate(ty);
            const uint32_t  C  = ggml_ai4_trellis_mul(ty);
            std::vector<uint8_t> q((size_t) ai4x_slab_bytes_k(K2, k)*(M2/16));
            std::vector<float>   R((size_t) M2*K2);
            float ms = 0.0f;
            if (ai4_cuda_encode_rate(W2.data(), K2, M2, nullptr, nullptr, aik_level_rms_c(C), ggml_ai4_scale_mul(ty), k, C,
                                     q.data(), &ms)) return 1;
            if (ai4_cuda_reconstruct_rate(q.data(), K2, M2, k, C, 0, R.data())) return 1;
            err[it] = rel_err(W2, R);
            printf("      %-6s k=%d C=%08X smul %.2f  rel err %.5e\n", ggml_type_name(ty), k, C, ggml_ai4_scale_mul(ty), err[it]);
        }
        // g_types[0..4] = k 2, 3, 4, 5, 6
        bool ord = true;
        for (int i = 0; i + 1 < 5; ++i) ord = ord && err[i] > err[i+1];
        check(ord, "T4 weight error ordering k2 > k3 > k4 > k5 > k6");
        printf("%s  T5 AI6_U %.5e vs AI6_UO %.5e (%.2f%%), AI5_U %.5e vs AI5_UN %.5e (%.2f%%)\n",
               err[4] < err[5] ? "INFO" : "NOTE", err[4], err[5], 100.0*(err[4]/err[5] - 1.0),
               err[3], err[6], 100.0*(err[3]/err[6] - 1.0));
    }

    // ---- T6: the span layout (AIk_S) at every rate ----
    {
        const int64_t KS = (K % 1024 == 0) ? K : 1024, MS = M;
        std::vector<float> WS((size_t) MS*KS);
        for (int64_t r = 0; r < MS; ++r) {
            const float rs = std::exp(0.3f*nd(rng))*0.02f;
            for (int64_t k = 0; k < KS; ++k) WS[(size_t) r*KS + k] = rs*nd(rng);
        }
        // --scale-mode row: one scale per row (the whole-row RMS), which is constant inside every 1024 span
        std::vector<float> sfix((size_t) MS*(KS/128));
        for (int64_t r = 0; r < MS; ++r) {
            double ss = 0.0;
            for (int64_t k = 0; k < KS; ++k) { const double v = WS[(size_t) r*KS + k]; ss += v*v; }
            const float v = (float) sqrt(ss/(double) KS);
            for (int64_t g = 0; g < KS/128; ++g) sfix[(size_t) r*(KS/128) + g] = v;
        }
        const ggml_type sp_types[5] = { GGML_TYPE_AI2_S, GGML_TYPE_AI3_S, GGML_TYPE_AI4_S, GGML_TYPE_AI5_S, GGML_TYPE_AI6_S };
        for (const ggml_type ty : sp_types) {
            const int       k  = ggml_ai4_rate(ty);
            const uint32_t  C  = ggml_ai4_trellis_mul(ty);
            const float     sm = ggml_ai4_scale_mul(ty);
            const char *    nm = ggml_type_name(ty);
            const size_t    sb = (size_t) ai4x_slab_bytes_k(KS, k), sbS = (size_t) ai4s_slab_bytes_k(KS, k);
            const int64_t   ns = MS/16;
            char buf[256];

            // GPU encode in the unit layout with the row scale table, then pack
            std::vector<uint8_t> qu(sb*ns), qs(sbS*ns, 0), qu2(sb*ns, 0xAB), qcs(sbS*ns, 0);
            float ms = 0.0f;
            if (ai4_cuda_encode_rate(WS.data(), KS, MS, nullptr, nullptr, aik_level_rms_c(C), sm, k, C, qu.data(), &ms,
                                     sfix.data())) return 1;
            bool pok = true;
            for (int64_t sl = 0; sl < ns; ++sl) {
                pok &= ai4s_pack_slab(k, qu.data() + sl*sb, qs.data() + sl*sbS, KS) != 0;
                ai4s_unpack_slab(k, qs.data() + sl*sbS, qu2.data() + sl*sb, KS);
            }
            snprintf(buf, sizeof(buf), "T6 %-6s k=%d: row-scaled AIk_U packs and unpack(pack) == source bytes", nm, k);
            check(pok && memcmp(qu2.data(), qu.data(), qu.size()) == 0, buf);

            // CPU: the span decode of the packed bytes == the unit decode of the source bytes, bit for bit
            std::vector<float> du((size_t) MS*KS), ds((size_t) MS*KS), gs((size_t) MS*KS), gu((size_t) MS*KS);
            for (int64_t sl = 0; sl < ns; ++sl) {
                aik_dequantize_slab_c (k, C, qu.data() + sl*sb,  du.data() + (size_t) sl*16*KS, KS);
                ai4s_dequantize_slab_c(k, C, qs.data() + sl*sbS, ds.data() + (size_t) sl*16*KS, KS);
            }
            snprintf(buf, sizeof(buf), "T6 %-6s CPU span dequant bit-exact vs the AIk_U dequant of the same bytes", nm);
            check(memcmp(du.data(), ds.data(), du.size()*sizeof(float)) == 0, buf);

            // CUDA: the span reconstruct (rotated and unrotated) == the rate-k reconstruct of the unit bytes
            if (ai4_cuda_reconstruct_span(qs.data(), KS, MS, k, C, 1, gs.data())) return 1;
            if (ai4_cuda_reconstruct_rate(qu.data(), KS, MS, k, C, 1, gu.data())) return 1;
            snprintf(buf, sizeof(buf), "T6 %-6s CUDA span reconstruct bit-exact vs AIk_U (rotated)", nm);
            check(memcmp(gs.data(), gu.data(), gs.size()*sizeof(float)) == 0, buf);
            if (ai4_cuda_reconstruct_span(qs.data(), KS, MS, k, C, 0, gs.data())) return 1;
            if (ai4_cuda_reconstruct_rate(qu.data(), KS, MS, k, C, 0, gu.data())) return 1;
            snprintf(buf, sizeof(buf), "T6 %-6s CUDA span reconstruct bit-exact vs AIk_U (unrotated)", nm);
            check(memcmp(gs.data(), gu.data(), gs.size()*sizeof(float)) == 0, buf);

            // CPU span encoder (the same scale table) + pack vs the GPU bytes: reported, gated on the error ratio
            std::vector<uint8_t> qc(sb*ns, 0);
            ai4_work * ws = ai4_work_new();
            for (int64_t sl = 0; sl < ns; ++sl) {
                aik_quantize_slab_rb_cf(ws, k, C, WS.data() + (size_t) sl*16*KS, KS, nullptr, nullptr, nullptr,
                                        qc.data() + sl*sb, ai4u_rot_size(KS), sm, sfix.data() + (size_t) sl*16*(KS/128));
                pok &= ai4s_pack_slab(k, qc.data() + sl*sb, qcs.data() + sl*sbS, KS) != 0;
            }
            free(ws);
            std::vector<float> dc((size_t) MS*KS);
            for (int64_t sl = 0; sl < ns; ++sl) {
                ai4s_dequantize_slab_c(k, C, qcs.data() + sl*sbS, dc.data() + (size_t) sl*16*KS, KS);
            }
            const double ec = rel_err(WS, dc), eg = rel_err(WS, ds);
            snprintf(buf, sizeof(buf), "T6 %-6s GPU/CPU span encoder: bytes %s, err gpu %.5e cpu %.5e ratio %.4f <= 1.03",
                     nm, memcmp(qcs.data(), qs.data(), qs.size()) == 0 ? "IDENTICAL" : "differ", eg, ec, eg/ec);
            check(pok && eg <= 1.03*ec, buf);
            printf("      %-6s %.4f bpw (AIk_U %.3f), span %lld B, row %lld B\n", nm,
                   (32.0 + 2048.0*k)*8.0/16384.0, (32.0 + 256.0*k)*8.0/2048.0,
                   (long long) AI4S_SPAN_BYTES(k), (long long) AI4S_ROW_BYTES(k));
        }
    }

    // ---- T7: optional physical row metric, including equal-scale and all-zero scale cases ----
    for (int it = 0; it < g_ntype; ++it) {
        const ggml_type ty = g_types[it];
        const int k = ggml_ai4_rate(ty);
        const uint32_t C = ggml_ai4_trellis_mul(ty);
        const float sm = ggml_ai4_scale_mul(ty), lrms = aik_level_rms_c(C);
        const size_t sb = (size_t) ai4x_slab_bytes_k(K, k);
        std::vector<uint8_t> qn(sb*(M/16)), qw(qn.size()), qc(qn.size());
        std::vector<float> sfix((size_t) M*(K/128), 0.02f);
        std::vector<float> rg((size_t) M*K), rc(rg.size());
        float ms = 0.0f;
        char buf[256];
        if (ai4_cuda_encode_rate(W.data(), K, M, nullptr, nullptr, lrms, sm, k, C, qn.data(), &ms, sfix.data(), 0)) return 1;
        if (ai4_cuda_encode_rate(W.data(), K, M, nullptr, nullptr, lrms, sm, k, C, qw.data(), &ms, sfix.data(), 1)) return 1;
        snprintf(buf, sizeof(buf), "T7 %-6s weighted metric with constant scales == normalized GPU bytes", ggml_type_name(ty));
        check(memcmp(qn.data(), qw.data(), qn.size()) == 0, buf);

        std::fill(sfix.begin(), sfix.end(), 0.0f);
        if (ai4_cuda_encode_rate(W.data(), K, M, nullptr, nullptr, lrms, sm, k, C, qw.data(), &ms, sfix.data(), 1)) return 1;
        if (ai4_cuda_reconstruct_rate(qw.data(), K, M, k, C, 1, rg.data())) return 1;
        bool zero = true;
        for (float v : rg) zero = zero && v == 0.0f;
        snprintf(buf, sizeof(buf), "T7 %-6s weighted metric with zero scales reconstructs zero", ggml_type_name(ty));
        check(zero, buf);

        // Unequal group scales: compare the physical metric using the same FP16 scale values, then check decode identity.
        if (ai4_cuda_encode_rate(W.data(), K, M, nullptr, nullptr, lrms, sm, k, C, qw.data(), &ms, nullptr, 1)) return 1;
        ai4_work * ws = ai4_work_new();
        for (int64_t sl = 0; sl < M/16; ++sl) {
            aik_quantize_slab_rb_cfm(ws, k, C, W.data() + (size_t) sl*16*K, K, nullptr, nullptr, nullptr,
                                     qc.data() + sl*sb, ai4u_rot_size(K), sm, nullptr, 1);
            aik_dequantize_slab_c(k, C, qc.data() + sl*sb, rc.data() + (size_t) sl*16*K, K);
        }
        free(ws);
        if (ai4_cuda_reconstruct_rate(qw.data(), K, M, k, C, 0, rg.data())) return 1;
        const double eg = rel_err(W, rg), ec = rel_err(W, rc);
        printf("INFO  T7 %-6s weighted GPU/CPU bytes %s, err gpu %.5e cpu %.5e ratio %.4f (half vs float costs)\n",
               ggml_type_name(ty), memcmp(qw.data(), qc.data(), qw.size()) == 0 ? "IDENTICAL" : "differ", eg, ec, eg/ec);
        for (int64_t sl = 0; sl < M/16; ++sl) {
            aik_dequantize_slab_rotated_c(k, C, qw.data() + sl*sb, rc.data() + (size_t) sl*16*K, K);
        }
        if (ai4_cuda_reconstruct_rate(qw.data(), K, M, k, C, 1, rg.data())) return 1;
        snprintf(buf, sizeof(buf), "T7 %-6s weighted GPU bytes reconstruct exactly on CPU and CUDA", ggml_type_name(ty));
        check(std::isfinite(eg) && std::isfinite(ec) && memcmp(rg.data(), rc.data(), rg.size()*sizeof(float)) == 0, buf);
    }

    printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
