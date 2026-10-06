// [#73] EXL3 weight-major mma (GGML_CUDA_EXL3_WEIGHT_MAJOR) must be bit-identical to the x-as-A mma it replaces.
//
// The weight-major variant of the trellis GEMV (ggml-cuda/exl3-gemv.cu, WM = true) runs for 3- and 4-bit EXL3 weights
// at T = 2..8 activation columns with the split glue (not GGML_CUDA_EXL3_FUSED=1). It swaps the mma operands (the
// decoded weight tile is A, x is B) but keeps the fp16 D accumulation, the fold schedule and the reduction, so its
// output must match the x-as-A path byte for byte. Each case is one EXL3 mul_mat, with or without the suh/svh +
// Hadamard glue on src[2]/src[3] (llama build_lora_mm). The cases run on a GPU backend in forked children, once with
// GGML_CUDA_EXL3_WEIGHT_MAJOR=0 and once with =1 (the env var is read once per process), and the outputs are compared
// byte for byte. Under the #41 decode kernel (k_exl3_gemv_dec, the default; GGML_CUDA_EXL3_DEC=0 selects the old
// k_exl3_gemv) the weight-major range is every bit width at T = 2..8, so the 2/5/6/7/8-bit cases compare the two mma
// orientations too; under the old kernel they and T = 1, T > 8 run the same kernel in both children and act as a
// determinism control. A third child computes every case on the CPU backend
// (ggml_compute_forward_mul_mat_exl3, f32 codebook values) and both GPU arms must stay within NMSE 5e-4 of it, the
// test-backend-ops MUL_MAT tolerance.
// The trellis words are random: every 16-bit code decodes to a codebook value, so any bit pattern is a valid EXL3
// tensor. GGML_CUDA_EXL3_FUSED and GGML_CUDA_EXL3_GEMV are cleared in the GPU children so the GEMV split-glue path
// (the only one with a weight-major variant) runs. A build with -DLLAMAMPERE_EXL3_WEIGHT_MAJOR=0 compiles the
// variant out and passes trivially. Needs a GPU backend; skips (exit 0) without one.

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
    // every bit width at the GEMV widths; 3/4-bit with and without the glue
    for (ggml_type type : { GGML_TYPE_EXL3_2, GGML_TYPE_EXL3_3, GGML_TYPE_EXL3_4, GGML_TYPE_EXL3_5,
                            GGML_TYPE_EXL3_6, GGML_TYPE_EXL3_7, GGML_TYPE_EXL3_8 }) {
        const int bits = ggml_exl3_bits(type);
        for (int64_t T : { 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 }) {
            out.push_back({ 512, 256, T, type, false });
            if (bits == 3 || bits == 4) {
                out.push_back({ 512, 256, T, type, true });
            }
        }
    }
    // split-K with full 4-tile fold groups plus a tail: 128 x 256 tiles, ksplit 6
    for (ggml_type type : { GGML_TYPE_EXL3_3, GGML_TYPE_EXL3_4 }) {
        for (int64_t T : { 1, 2, 3, 4, 5, 6, 7, 8 }) {
            out.push_back({ 2048, 4096, T, type, true });
        }
    }
    // Qwen3.8 27B FFN gate/up (5120 -> 17408) and down (17408 -> 5120): 64 k-tiles per item, 16 per warp
    for (ggml_type type : { GGML_TYPE_EXL3_3, GGML_TYPE_EXL3_4 }) {
        for (int64_t T : { 2, 5, 8 }) {
            out.push_back({ 5120, 17408, T, type, true });
            out.push_back({ 17408, 5120, T, type, true });
        }
    }
    return out;
}

// the weight-major mma covers these cases (exl3_gemv_weight_major_ok + the kernel's WEIGHT_MAJOR condition)
static bool weight_major_case(const test_case_def & c) {
    const int bits = ggml_exl3_bits(c.type);
    const char * dec = getenv("GGML_CUDA_EXL3_DEC");
    const bool   all = dec == nullptr || strcmp(dec, "0") != 0;   // #41 k_exl3_gemv_dec: every bit width
    return (all || bits == 3 || bits == 4) && c.T >= 2 && c.T <= 8;
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

enum arm_t { ARM_WM_OFF, ARM_WM_ON, ARM_CPU };

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
            setenv("GGML_CUDA_EXL3_WEIGHT_MAJOR", arm == ARM_WM_ON ? "1" : "0", 1);
            unsetenv("GGML_CUDA_EXL3_FUSED"); // the cooperative fused GEMV has no weight-major variant
            unsetenv("GGML_CUDA_EXL3_GEMV");  // =0 would send every width to reconstruct + cuBLAS
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
    const std::vector<uint8_t> off = run_child(ARM_WM_OFF);
    const std::vector<uint8_t> on  = run_child(ARM_WM_ON);
    if (off.empty() && on.empty()) {
        printf("no GPU backend, skipping\n");
        return 0;
    }
    const std::vector<uint8_t> ref = run_child(ARM_CPU);
    GGML_ASSERT(off.size() == on.size() && off.size() == ref.size());

    constexpr double max_nmse = 5e-4;
    const std::vector<test_case_def> cs = cases();
    int    n_fail = 0;
    int    n_wm   = 0;
    double worst  = 0.0;
    size_t pos    = 0;
    for (const test_case_def & c : cs) {
        const size_t n_vals = (size_t) (c.N * c.T);
        const size_t sz     = n_vals * sizeof(float);
        char label[128];
        snprintf(label, sizeof(label), "%s K %5lld N %5lld T %2lld%s%s", ggml_type_name(c.type), (long long) c.K,
                 (long long) c.N, (long long) c.T, c.glue ? " glue" : "", weight_major_case(c) ? " [wm]" : "");
        n_wm += weight_major_case(c);

        if (memcmp(off.data() + pos, on.data() + pos, sz) != 0) {
            size_t n_diff = 0;
            for (size_t j = 0; j < sz; j += sizeof(float)) {
                n_diff += memcmp(off.data() + pos + j, on.data() + pos + j, sizeof(float)) != 0;
            }
            printf("%s: weight-major on vs off differs in %zu of %zu values\n", label, n_diff, n_vals);
            n_fail++;
        }

        std::vector<float> f_off(n_vals), f_on(n_vals), f_ref(n_vals);
        memcpy(f_off.data(), off.data() + pos, sz);
        memcpy(f_on.data(),  on.data()  + pos, sz);
        memcpy(f_ref.data(), ref.data() + pos, sz);
        for (const std::vector<float> * f : { &f_off, &f_on }) {
            bool finite = true;
            for (float v : *f) {
                finite = finite && std::isfinite(v);
            }
            const double e = nmse(f->data(), f_ref.data(), n_vals);
            worst = std::max(worst, e);
            if (!finite || !(e <= max_nmse)) {
                printf("%s: weight-major %s vs CPU: NMSE %.3e > %.1e%s\n", label, f == &f_on ? "on " : "off", e, max_nmse,
                       finite ? "" : " (non-finite output)");
                n_fail++;
            }
        }
        pos += sz;
    }
    GGML_ASSERT(pos == off.size());

    printf("%zu cases, %d in the weight-major range, worst NMSE vs CPU %.3e\n", cs.size(), n_wm, worst);
    if (n_fail) {
        printf("FAILED (%d)\n", n_fail);
        return 1;
    }
    printf("OK: weight-major on/off bit-identical, both within %.0e of the CPU reference\n", max_nmse);
    return 0;
#endif
}
