// INKMISS CPU expert paths: prints one hash of the MUL_MAT_ID output per case, computed on the CPU.
//
// Run it once with the flags unset and once with GGML_CPU_NVFP4_GROUPED=1 GGML_CPU_AI4_SCHEDULE=1 (#133, FNV7-2): the
// hashes must match (both changes keep the per-row arithmetic). With --check it also compares the NVFP4 grouped dot
// directly against the single-row dot (exit 1 on a mismatch).
//
// --bench: also times Inkling-shaped expert MUL_MAT_ID (gate_up 2304 -> 4608, 128 experts, top 8) at 1 and 4 tokens.
//
//   test-inkmiss-cpu [--threads n] [--check] [--bench]

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" bool ggml_vec_dot_iq_grouped(enum ggml_type type, int n, const void * vx, const void * const * vy, float * s, int nr);

static uint64_t fnv1a(const void * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ ((const uint8_t *) p)[i]) * 1099511628211ull;
    }
    return h;
}

struct mmid_case {
    ggml_type type;
    int64_t K, M, n_exp, n_used, n_tok;
    bool shared; // tokens route to overlapping experts (several rows per expert)
};

static uint64_t run_case(const mmid_case & c, int n_threads, std::mt19937 & rng, int reps = 0) {
    ggml_init_params ip = { (size_t) 1 << 30, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a   = ggml_new_tensor_3d(ctx, c.type, c.K, c.M, c.n_exp);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.K, c.n_used, c.n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, c.n_tok);
    // weights: random bytes (both formats decode any byte pattern; a NaN scale decodes to zero)
    for (size_t i = 0; i < ggml_nbytes(a); ++i) {
        ((uint8_t *) a->data)[i] = (uint8_t) rng();
    }
    if (c.type == GGML_TYPE_NVFP4) {
        // keep the scales finite and moderate: a UE4M3 byte in [0x20, 0x40)
        const size_t bs = ggml_type_size(c.type);
        for (size_t i = 0; i < ggml_nbytes(a); i += bs) {
            for (int k = 0; k < 4; ++k) {
                ((uint8_t *) a->data)[i + k] = (uint8_t) (0x20 + rng() % 0x20);
            }
        }
    }
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(b); ++i) {
        ((float *) b->data)[i] = u(rng);
    }
    for (int64_t t = 0; t < c.n_tok; ++t) {
        std::vector<int32_t> pick;
        while ((int64_t) pick.size() < c.n_used) {
            const int32_t e = (int32_t) (rng() % (c.shared ? std::min<int64_t>(c.n_exp, c.n_used + 2) : c.n_exp));
            bool dup = false;
            for (int32_t q : pick) { dup = dup || q == e; }
            if (!dup) { pick.push_back(e); }
        }
        memcpy((int32_t *) ids->data + t*c.n_used, pick.data(), c.n_used*sizeof(int32_t));
    }
    ggml_tensor * out = ggml_mul_mat_id(ctx, a, b, ids);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (reps > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        printf("bench %s K=%" PRId64 " M=%" PRId64 " E=%" PRId64 " tok=%" PRId64 "%s: %.1f us/run\n",
                ggml_type_name(c.type), c.K, c.M, c.n_exp, c.n_tok, c.shared ? " shared" : "", us);
    }
    const uint64_t h = fnv1a(out->data, ggml_nbytes(out));
    ggml_free(ctx);
    return h;
}

static bool check_nvfp4_grouped(std::mt19937 & rng) {
    const ggml_type_traits_cpu * tr = ggml_get_type_traits_cpu(GGML_TYPE_NVFP4);
    const ggml_type_traits_cpu * tq = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const int n = 2304;
    std::vector<float> xf(n), yf(n);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (float & v : xf) { v = u(rng); }
    std::vector<uint8_t> x(ggml_row_size(GGML_TYPE_NVFP4, n));
    tr->from_float(xf.data(), x.data(), n);
    std::vector<std::vector<uint8_t>> y(8, std::vector<uint8_t>(ggml_row_size(GGML_TYPE_Q8_0, n)));
    const void * yp[8];
    for (int t = 0; t < 8; ++t) {
        for (float & v : yf) { v = u(rng); }
        tq->from_float(yf.data(), y[t].data(), n);
        yp[t] = y[t].data();
    }
    bool ok = true;
    for (int nr = 1; nr <= 8; ++nr) {
        float s[8];
        if (!ggml_vec_dot_iq_grouped(GGML_TYPE_NVFP4, n, x.data(), yp, s, nr)) {
            printf("nvfp4 grouped: not available on this CPU build (nr=%d), skipped\n", nr);
            return true;
        }
        for (int t = 0; t < nr; ++t) {
            float r = 0.0f;
            tr->vec_dot(n, &r, 0, x.data(), 0, yp[t], 0, 1);
            if (memcmp(&r, &s[t], sizeof(float)) != 0) {
                printf("nvfp4 grouped MISMATCH nr=%d t=%d: %.9g vs %.9g\n", nr, t, s[t], r);
                ok = false;
            }
        }
    }
    printf("nvfp4 grouped vs single-row dot: %s\n", ok ? "identical" : "DIFFER");
    return ok;
}

int main(int argc, char ** argv) {
    int n_threads = 4;
    bool check = false;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--check")) {
            check = true;
        } else if (!strcmp(argv[i], "--bench")) {
            bench = true;
        }
    }
    ggml_cpu_init();
    std::mt19937 rng(42);
    bool ok = true;
    if (check) {
        ok = check_nvfp4_grouped(rng);
    }
    const mmid_case cases[] = {
        { GGML_TYPE_NVFP4,  2304,  256, 16, 8, 1, false },
        { GGML_TYPE_NVFP4,  2304,  256, 16, 8, 4, true  },
        { GGML_TYPE_NVFP4,  1152,  512, 32, 8, 8, true  },
        { GGML_TYPE_AIT3_U, 2304,  256, 16, 8, 1, false },
        { GGML_TYPE_AIT3_U, 2304,  256, 16, 8, 4, true  },
        { GGML_TYPE_AIT3_U, 1152,  512, 32, 8, 8, false },
        { GGML_TYPE_AIT3_U, 2304, 4608, 16, 8, 1, false },
    };
    for (const mmid_case & c : cases) {
        const uint64_t h = run_case(c, n_threads, rng);
        printf("case %s K=%" PRId64 " M=%" PRId64 " E=%" PRId64 " used=%" PRId64 " tok=%" PRId64 "%s: %016" PRIx64 "\n",
                ggml_type_name(c.type), c.K, c.M, c.n_exp, c.n_used, c.n_tok, c.shared ? " shared" : "", h);
    }
    if (bench) {
        for (ggml_type t : { GGML_TYPE_NVFP4, GGML_TYPE_AIT3_U }) {
            for (int64_t tok : { 1, 4 }) {
                run_case({ t, 2304, 4608, 128, 8, tok, false }, n_threads, rng, 20);
                run_case({ t, 2304, 4608, 128, 8, tok*2, true }, n_threads, rng, 20); // 10 experts: ~tok*1.6 rows each
            }
        }
    }
    return ok ? 0 : 1;
}
