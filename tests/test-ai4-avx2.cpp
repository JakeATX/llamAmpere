// Opt-in AI+ unit SIMD path of the CPU backend (GGML_CPU_AI4_AVX2=1, ggml-cpu/ai4-avx2.cpp) against the default reference
// path (slab dequantize + inverse rotation + f32 dot), on Flash-Next expert shapes (K = 2560, M = 640):
//  - the AVX2 AIT2P_U pair-table levels are bit-exact to aik_dequantize_slab_rotated_c (C == 0);
//  - MUL_MAT (1, 5, 9, 17 activation rows) and MUL_MAT_ID (4 experts, 3 used: 3 tokens with a broadcast src1 row and one
//    id = -1; 12 tokens with one src1 row per slot, > 8 pairs per expert, ids -1 and n_expert) for AIT2/3/4/5_U, AIT2P_U
//    and AIT2/3/4_U2: every output within 2e-5 of max |reference| (float summation order differs); the 512-K row-scale
//    U2 type (not covered) and the flag unset give the reference bytes exactly. dst is NaN-filled before every run.
//   test-ai4-avx2 [bench reps]: also times MUL_MAT_ID decode (8 experts, 8 used, 1 and 4 tokens) on both paths.
//   test-ai4-avx2 reps threads ink: AIT3_U / AIT4_U on the Inkling 2 Mini expert shapes, 1..16 rows per expert.
#include "ggml.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-ai4.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static std::mt19937 rng(20261124);

static void set_flag(bool on) {
#ifdef _WIN32
    _putenv_s("GGML_CPU_AI4_AVX2", on ? "1" : "0");
#else
    if (on) { setenv("GGML_CPU_AI4_AVX2", "1", 1); } else { unsetenv("GGML_CPU_AI4_AVX2"); }
#endif
}

// random bit streams, valid fp16 row scales at every scale header
static void fill_weights(ggml_tensor * w) {
    uint8_t * p = (uint8_t *) w->data;
    const size_t nb = ggml_nbytes(w);
    for (size_t i = 0; i < nb; ++i) { p[i] = (uint8_t) rng(); }
    const int k = ggml_ai4_rate(w->type);
    const int units = ggml_ai4v2_rate(w->type) ? ggml_ai4v2_chunks(w->type) : 1;
    const size_t block = units == 1 ? AI4U_UNIT_BYTES(k) : AI4V2_BLOCK_BYTES_N(k, units);
    std::uniform_real_distribution<float> sd(0.005f, 0.05f);
    for (size_t off = 0; off + block <= nb; off += block) {
        uint16_t * dh = (uint16_t *) (p + off);
        for (int r = 0; r < AI4_TR; ++r) { dh[r] = ggml_fp32_to_fp16(sd(rng)); }
    }
}

static void fill_f32(ggml_tensor * t) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) { d[i] = u(rng); }
}

static std::vector<float> run(ggml_context * ctx, ggml_tensor * out, bool flag, int n_threads) {
    static std::vector<uint8_t> work;
    set_flag(flag);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(gf, out);
    ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);   // the flag also sizes the work buffer
    if (work.size() < plan.work_size) { work.resize(plan.work_size); }
    plan.work_data = work.data();
    // NaN-fill dst so a row the path under test fails to write cannot pass on the previous run's values
    float * od = (float *) out->data;
    for (int64_t i = 0; i < ggml_nelements(out); ++i) { od[i] = NAN; }
    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        printf("graph compute failed\n");
        exit(1);
    }
    set_flag(false);
    const float * d = (const float *) out->data;
    return std::vector<float>(d, d + ggml_nelements(out));
}

// max |a - ref| and that over max |ref|
static bool compare(const char * what, ggml_type type, const std::vector<float> & ref, const std::vector<float> & got, bool exact) {
    double emax = 0.0, rmax = 0.0;
    size_t bad = 0;   // non-finite outputs (an unwritten NaN-filled row): std::max would skip them
    for (size_t i = 0; i < ref.size(); ++i) {
        bad += !std::isfinite(got[i]) || !std::isfinite(ref[i]);
        emax = std::max(emax, (double) std::fabs(got[i] - ref[i]));
        rmax = std::max(rmax, (double) std::fabs(ref[i]));
    }
    const double rel = rmax > 0.0 ? emax / rmax : emax;
    const bool same = memcmp(ref.data(), got.data(), ref.size()*sizeof(float)) == 0;
    const bool ok = bad == 0 && (exact ? same : (rel <= 2e-5 && std::isfinite(rel)));
    printf("%-12s %-14s max_abs_err=%.3g max|ref|=%.3g rel=%.3g%s%s %s\n", what, ggml_type_name(type), emax, rmax, rel,
           same ? " (bit-identical)" : "", bad ? " NON-FINITE" : "", ok ? "OK" : "FAIL");
    return ok;
}

static bool test_type(ggml_type type, bool covered, int n_threads) {
    const int64_t K = 2560, M = 640;
    bool ok = true;
    ggml_init_params ip = { (size_t) 256 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    for (int n : {1, 5, 9, 17}) {   // 17 = two full 8-row batches + 1
        ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, M);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
        fill_weights(w); fill_f32(x);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        const auto ref = run(ctx, y, false, n_threads);
        const auto again = run(ctx, y, false, n_threads);
        ok &= compare("mul_mat ref", type, ref, again, true);
        char what[32]; snprintf(what, sizeof(what), "mul_mat n=%d", n);
        ok &= compare(what, type, ref, run(ctx, y, true, n_threads), !covered);
    }
    {
        const int n_exp = 4, n_used = 3, n_tok = 3;
        ggml_tensor * w   = ggml_new_tensor_3d(ctx, type, K, M, n_exp);
        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tok);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
        fill_weights(w); fill_f32(x);
        const int32_t sel[n_tok][n_used] = {{0, 2, 3}, {2, -1, 0}, {3, 2, 1}};
        memcpy(ids->data, sel, sizeof(sel));
        ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
        const auto ref = run(ctx, y, false, n_threads);
        ok &= compare("mul_mat_id", type, ref, run(ctx, y, true, n_threads), !covered);
    }
    {
        // down projection layout: one activation row per (slot, token); 12 tokens over 4 experts -> > 8 pairs per expert,
        // out-of-range ids (-1, n_exp) give zero rows
        const int n_exp = 4, n_used = 3, n_tok = 12;
        ggml_tensor * w   = ggml_new_tensor_3d(ctx, type, K, M, n_exp);
        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, n_tok);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
        fill_weights(w); fill_f32(x);
        int32_t * sel = (int32_t *) ids->data;
        for (int t = 0; t < n_tok; ++t) {
            for (int s = 0; s < n_used; ++s) { sel[t*n_used + s] = (t + 2*s) % n_exp; }
        }
        sel[1*n_used + 1] = -1;
        sel[7*n_used + 2] = n_exp;
        ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
        const auto ref = run(ctx, y, false, n_threads);
        ok &= compare("mul_mat_id dn", type, ref, run(ctx, y, true, n_threads), !covered);
    }
    ggml_free(ctx);
    return ok;
}

int ai4_avx2_pair_levels_check(unsigned seed);   // test-ai4-avx2-levels.cpp (built with AVX2 + FMA where possible)

static bool test_pair_levels() {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        printf("AIT2P_U AVX2 pair levels: CPU has no AVX2/FMA, skipped\n");
        return true;
    }
    const int r = ai4_avx2_pair_levels_check(20261124u);
    if (r < 0) {
        printf("AIT2P_U AVX2 pair levels: helper not built with AVX2/FMA, skipped\n");
        return true;
    }
    printf("AIT2P_U AVX2 pair levels vs scalar: %s\n", r ? "bit-identical OK" : "FAIL");
    return r == 1;
#else
    printf("AIT2P_U AVX2 pair levels: not an x86 GCC/Clang build, skipped\n");
    return true;
#endif
}

static void bench(ggml_type type, int reps, int n_threads, int64_t K = 2560, int64_t M = 640, std::vector<int> toks = {1, 4}) {
    const int n_exp = 8, n_used = 8;
    for (int n_tok : toks) {
        ggml_init_params ip = { (size_t) 512 << 20, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w   = ggml_new_tensor_3d(ctx, type, K, M, n_exp);
        ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tok);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
        fill_weights(w); fill_f32(x);
        for (int t = 0; t < n_tok; ++t) {
            for (int s = 0; s < n_used; ++s) { ((int32_t *) ids->data)[t*n_used + s] = (s + t) % n_exp; }
        }
        ggml_tensor * y = ggml_mul_mat_id(ctx, w, x, ids);
        double us[2] = {};
        for (int pass = 0; pass < 4; ++pass) {   // ABBA
            const bool flag = pass == 1 || pass == 2;
            run(ctx, y, flag, n_threads);
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) { run(ctx, y, flag, n_threads); }
            us[flag] += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (2.0*reps);
        }
        printf("BENCH %-10s mul_mat_id K=%lld M=%lld experts=%d used=%d tokens=%d threads=%d: reference %.1f us, avx2 %.1f us (%.2fx); "
               "per expert %.4f / %.4f ms\n", ggml_type_name(type), (long long) K, (long long) M, n_exp, n_used, n_tok, n_threads,
               us[0], us[1], us[0]/us[1], us[0]/1000.0/n_used, us[1]/1000.0/n_used);
        ggml_free(ctx);
    }
}

int main(int argc, char ** argv) {
    const int n_threads = 4;
    bool ok = test_pair_levels();
    const ggml_type covered[] = {
        GGML_TYPE_AIT2_U, GGML_TYPE_AIT3_U, GGML_TYPE_AIT4_U, GGML_TYPE_AIT5_U, GGML_TYPE_AIT2P_U,
        GGML_TYPE_AIT2_U2, GGML_TYPE_AIT3_U2, GGML_TYPE_AIT4_U2,
    };
    for (ggml_type t : covered) { ok &= test_type(t, true, n_threads); }
    ok &= test_type(GGML_TYPE_AIT3_U2_512, false, n_threads);
    if (argc > 1) {
        const int reps = std::max(1, atoi(argv[1]));
        const int bench_threads = argc > 2 ? std::max(1, atoi(argv[2])) : 8;
        if (argc > 3 && strcmp(argv[3], "ink") == 0) {
            // Inkling 2 Mini expert shapes: gate_up K=2304 M=2304, down K=1152 M=2304; every token uses all 8 experts,
            // so tokens = rows per expert (add the two lines for one expert)
            for (ggml_type t : {GGML_TYPE_AIT3_U, GGML_TYPE_AIT4_U}) {
                bench(t, reps, bench_threads, 2304, 2304, {1, 2, 4, 8, 16});
                bench(t, reps, bench_threads, 1152, 2304, {1, 2, 4, 8, 16});
            }
        } else {
            for (ggml_type t : {GGML_TYPE_AIT2_U, GGML_TYPE_AIT3_U, GGML_TYPE_AIT4_U, GGML_TYPE_AIT2P_U, GGML_TYPE_AIT3_U2}) {
                bench(t, reps, bench_threads);
            }
        }
    }
    printf("test-ai4-avx2: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
