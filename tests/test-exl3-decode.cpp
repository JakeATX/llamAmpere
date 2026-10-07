// [#41] EXL3DEC: the restructured decode GEMV (k_exl3_gemv_dec, default; GGML_CUDA_EXL3_DEC=0 selects the old
// k_exl3_gemv) must be bit-identical to the kernel it replaces.
//
// k_exl3_gemv_dec keeps the k-split, the tile-to-warp assignment, the fp16 fold schedule and both reductions of
// k_exl3_gemv and changes only the schedule (contiguous k-split-major item ranges with x staged once per k-range,
// register prefetch of the next tile group, double-buffered warp partials), plus the weight-major mma for every bit
// width at T = 2..8 (k_exl3_gemv: 3/4-bit only). Each case is one EXL3 mul_mat, with or without the suh/svh + Hadamard
// glue on src[2]/src[3]. The cases run on a GPU backend in forked children (the env vars are read once per process):
//   old    GGML_CUDA_EXL3_DEC=0                                (the pre-#41 default)
//   dec    GGML_CUDA_EXL3_DEC=1                                (the new default)
//   decxa  GGML_CUDA_EXL3_DEC=1 GGML_CUDA_EXL3_WEIGHT_MAJOR=0  (new schedule, x-as-A mma)
//   oldxa  GGML_CUDA_EXL3_DEC=0 GGML_CUDA_EXL3_WEIGHT_MAJOR=0
//   decg   GGML_CUDA_EXL3_DEC=1 GGML_CUDA_EXL3_DEC_GRID=7         (grid capped at 7 blocks: long item ranges per block)
// dec, decg must equal old, decxa must equal oldxa, and dec must equal oldxa (weight-major vs x-as-A mma at every bit
// width) byte for byte. A CPU child (ggml_compute_forward_mul_mat_exl3) is the
// accuracy reference: every GPU arm must stay within NMSE 5e-4 (the test-backend-ops MUL_MAT tolerance).
// The cases cover every bit width at T = 1..16, the Qwen3.8 27B projection shapes (several items per block, k-range
// changes inside a block's item range), and a k-split whose last item has fewer k-tiles than warps (warps with no
// tile). Trellis words are random: any 16-bit code decodes to a codebook value. Needs a GPU backend; skips without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

struct test_case_def {
    int64_t   K;    // input features  (ne0 of the weight)
    int64_t   N;    // output features (ne1 of the weight)
    int64_t   T;    // activation columns
    ggml_type type;
    bool      glue; // suh/svh on src[2]/src[3]
};

static std::vector<test_case_def> cases() {
    std::vector<test_case_def> out;
    const ggml_type all_types[] = { GGML_TYPE_EXL3_2, GGML_TYPE_EXL3_3, GGML_TYPE_EXL3_4, GGML_TYPE_EXL3_5,
                                    GGML_TYPE_EXL3_6, GGML_TYPE_EXL3_7, GGML_TYPE_EXL3_8 };
    // every bit width at every GEMV width, small (one item per block) and split-K with several items per block
    for (ggml_type type : all_types) {
        for (int64_t T = 1; T <= 16; ++T) {
            out.push_back({ 512, 256, T, type, false });
            out.push_back({ 2048, 4096, T, type, true });
        }
    }
    // last k-split item with 2 k-tiles (kt 112, ksplit 11, kpi 11): warps 2 and 3 own no tile there
    for (ggml_type type : all_types) {
        for (int64_t T : { 1, 2, 5, 8, 9, 16 }) {
            out.push_back({ 1792, 2240, T, type, false });
        }
    }
    // k-splits past the last k-tile (nk <= 0: kt 320 ksplit 39 kpi 9; kt 1088 ksplit 96 kpi 12): empty items
    for (int64_t T : { 1, 2, 5, 8, 16 }) {
        out.push_back({ 5120, 640, T, GGML_TYPE_EXL3_4, true });
        out.push_back({ 17408, 256, T, GGML_TYPE_EXL3_4, false });
        out.push_back({ 5120, 640, T, GGML_TYPE_EXL3_6, false });
    }
    // Qwen3.8 27B EXL3 4.0 bpw shapes: ffn gate/up, ffn down, GDN qkv, attn gate / q, ssm_out / attn_output
    for (int64_t T = 1; T <= 16; ++T) {
        out.push_back({ 5120, 17408, T, GGML_TYPE_EXL3_4, true });
        out.push_back({ 17408, 5120, T, GGML_TYPE_EXL3_4, true });
        out.push_back({ 5120, 10240, T, GGML_TYPE_EXL3_4, true });
        out.push_back({ 5120, 6144, T, GGML_TYPE_EXL3_4, true });
        out.push_back({ 6144, 5120, T, GGML_TYPE_EXL3_4, true });
    }
    // 6-bit output head slice and 3-bit FFN (EXL3 3.0 bpw files)
    for (int64_t T : { 1, 2, 4, 5, 8, 16 }) {
        out.push_back({ 5120, 32768, T, GGML_TYPE_EXL3_6, true });
        out.push_back({ 5120, 17408, T, GGML_TYPE_EXL3_3, true });
        out.push_back({ 17408, 5120, T, GGML_TYPE_EXL3_3, true });
    }
    return out;
}

static void run_case(ggml_backend_t backend, size_t idx, const test_case_def & c, std::vector<uint8_t> & blob) {
    ggml_init_params pw = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_w = ggml_init(pw);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, c.K, c.T);
    ggml_tensor * w   = ggml_new_tensor_2d(ctx_w, c.type, c.K, c.N);
    ggml_tensor * suh = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.K);
    ggml_tensor * svh = ggml_new_tensor_1d(ctx_w, GGML_TYPE_F32, c.N);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // same inputs in every child: the seed depends on the case only
    std::mt19937 rng(7300 + 1009 * (uint32_t) idx);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sd(0.5f, 1.5f);
    {
        std::vector<float> v(c.K * c.T);
        for (float & f : v) {
            f = xd(rng);
        }
        ggml_backend_tensor_set(x, v.data(), 0, ggml_nbytes(x));
    }
    {
        std::vector<uint32_t> words(ggml_nbytes(w) / sizeof(uint32_t));
        for (uint32_t & u : words) {
            u = (uint32_t) rng();
        }
        ggml_backend_tensor_set(w, words.data(), 0, ggml_nbytes(w));
    }
    for (ggml_tensor * s : { suh, svh }) {
        std::vector<float> v(s->ne[0]);
        for (float & f : v) {
            f = (rng() & 1 ? 1.0f : -1.0f) * sd(rng);
        }
        ggml_backend_tensor_set(s, v.data(), 0, ggml_nbytes(s));
    }

    ggml_init_params pg = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    if (c.glue) {
        out->src[2] = suh;
        out->src[3] = svh;
    }
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    const size_t off = blob.size();
    blob.resize(off + ggml_nbytes(out));
    ggml_backend_tensor_get(out, blob.data() + off, 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx_w);
}

enum arm_t { ARM_OLD, ARM_DEC, ARM_DECXA, ARM_OLDXA, ARM_DECG, ARM_CPU };

// runs every case on the arm's backend. Returns false when the arm's device is missing.
static bool run_all(arm_t arm, std::vector<uint8_t> & blob) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(arm == ARM_CPU ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        return false;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    if (arm == ARM_CPU) {
        typedef void (*set_n_threads_t)(ggml_backend_t, int);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        set_n_threads_t fn = reg ? (set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr;
        if (fn) {
            fn(backend, (int) std::max(1u, std::thread::hardware_concurrency()));
        }
    }
    const std::vector<test_case_def> cs = cases();
    for (size_t i = 0; i < cs.size(); ++i) {
        run_case(backend, i, cs[i], blob);
    }
    ggml_backend_free(backend);
    return true;
}

#ifndef _WIN32
// runs run_all() in a child and returns its blob (empty: the arm's device is missing)
static std::vector<uint8_t> run_child(arm_t arm) {
    int fd[2];
    GGML_ASSERT(pipe(fd) == 0);
    const pid_t pid = fork();
    GGML_ASSERT(pid >= 0);
    if (pid == 0) {
        close(fd[0]);
        if (arm != ARM_CPU) {
            setenv("GGML_CUDA_EXL3_DEC", (arm == ARM_DEC || arm == ARM_DECXA || arm == ARM_DECG) ? "1" : "0", 1);
            if (arm == ARM_DECG) {
                setenv("GGML_CUDA_EXL3_DEC_GRID", "7", 1);
            } else {
                unsetenv("GGML_CUDA_EXL3_DEC_GRID");
            }
            setenv("GGML_CUDA_EXL3_WEIGHT_MAJOR", (arm == ARM_DECXA || arm == ARM_OLDXA) ? "0" : "1", 1);
            unsetenv("GGML_CUDA_EXL3_FUSED");    // the cooperative fused GEMV keeps the old kernel
            unsetenv("GGML_CUDA_EXL3_GEMV");     // =0 would send every width to reconstruct + cuBLAS
            unsetenv("GGML_CUDA_EXL3_ENVELOPE");
        }
        std::vector<uint8_t> blob;
        run_all(arm, blob);
        size_t done = 0;
        while (done < blob.size()) {
            const ssize_t n = write(fd[1], blob.data() + done, blob.size() - done);
            if (n <= 0) {
                _exit(2);
            }
            done += (size_t) n;
        }
        close(fd[1]);
        _exit(0);
    }
    close(fd[1]);
    std::vector<uint8_t> blob;
    uint8_t tmp[1 << 16];
    ssize_t n;
    while ((n = read(fd[0], tmp, sizeof(tmp))) > 0) {
        blob.insert(blob.end(), tmp, tmp + n);
    }
    close(fd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    GGML_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    return blob;
}
#endif

// normalized mean squared error of a against the reference r, as test-backend-ops computes it
static double nmse(const float * a, const float * r, size_t n) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) r[i];
        num += d * d;
        den += (double) r[i] * (double) r[i];
    }
    return den > 0.0 ? num / den : (num > 0.0 ? INFINITY : 0.0);
}

int main() {
#ifdef _WIN32
    printf("needs fork(), skipping\n");
    return 0;
#else
    const std::vector<uint8_t> old_  = run_child(ARM_OLD);
    const std::vector<uint8_t> dec   = run_child(ARM_DEC);
    if (old_.empty() && dec.empty()) {
        printf("no GPU backend, skipping\n");
        return 0;
    }
    const std::vector<uint8_t> decxa = run_child(ARM_DECXA);
    const std::vector<uint8_t> oldxa = run_child(ARM_OLDXA);
    const std::vector<uint8_t> decg  = run_child(ARM_DECG);
    const std::vector<uint8_t> ref   = run_child(ARM_CPU);
    GGML_ASSERT(old_.size() == dec.size() && old_.size() == decxa.size() && old_.size() == oldxa.size() && old_.size() == decg.size() &&
                old_.size() == ref.size());

    constexpr double max_nmse = 5e-4;
    const std::vector<test_case_def> cs = cases();
    int    n_fail = 0;
    double worst  = 0.0;
    size_t pos    = 0;
    const struct { const std::vector<uint8_t> * a; const std::vector<uint8_t> * b; const char * name; } pairs[] = {
        { &dec, &old_, "dec vs old" }, { &decxa, &oldxa, "decxa vs oldxa" }, { &decg, &old_, "decg vs old" },
        { &dec, &oldxa, "dec vs oldxa (weight-major vs x-as-A)" },
    };
    const struct { const std::vector<uint8_t> * v; const char * name; } arms[] = {
        { &old_, "old" }, { &dec, "dec" }, { &decxa, "decxa" }, { &oldxa, "oldxa" }, { &decg, "decg" },
    };
    for (const test_case_def & c : cs) {
        const size_t n_vals = (size_t) (c.N * c.T);
        const size_t sz     = n_vals * sizeof(float);
        char label[128];
        snprintf(label, sizeof(label), "%s K %5lld N %5lld T %2lld%s", ggml_type_name(c.type), (long long) c.K,
                 (long long) c.N, (long long) c.T, c.glue ? " glue" : "");

        for (const auto & p : pairs) {
            if (memcmp(p.a->data() + pos, p.b->data() + pos, sz) != 0) {
                size_t n_diff = 0;
                for (size_t j = 0; j < sz; j += sizeof(float)) {
                    n_diff += memcmp(p.a->data() + pos + j, p.b->data() + pos + j, sizeof(float)) != 0;
                }
                printf("%s: %s differs in %zu of %zu values\n", label, p.name, n_diff, n_vals);
                n_fail++;
            }
        }

        std::vector<float> f(n_vals), f_ref(n_vals);
        memcpy(f_ref.data(), ref.data() + pos, sz);
        for (const auto & arm : arms) {
            memcpy(f.data(), arm.v->data() + pos, sz);
            bool finite = true;
            for (float v : f) {
                finite = finite && std::isfinite(v);
            }
            const double e = nmse(f.data(), f_ref.data(), n_vals);
            worst = std::max(worst, e);
            if (!finite || !(e <= max_nmse)) {
                printf("%s: %s vs CPU: NMSE %.3e > %.1e%s\n", label, arm.name, e, max_nmse, finite ? "" : " (non-finite output)");
                n_fail++;
            }
        }
        pos += sz;
    }
    GGML_ASSERT(pos == old_.size());

    printf("%zu cases, worst NMSE vs CPU %.3e\n", cs.size(), worst);
    if (n_fail) {
        printf("FAILED (%d)\n", n_fail);
        return 1;
    }
    printf("OK: EXL3 decode GEMV new vs old bit-identical (weight-major, x-as-A, capped grid; weight-major == x-as-A), all within %.0e of the CPU reference\n", max_nmse);
    return 0;
#endif
}
