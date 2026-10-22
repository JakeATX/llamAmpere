// [ROUNDFIX] GGML_CUDA_TOPK_SELECT_FAST=1 must give the same bytes as the plain radix select (GGML_CUDA_TOPK_SELECT=3
// forces the select path for every row it can take), and both must match a processor reference of the select rule:
// every index whose value is above the k-th largest (in index order), then the lowest-index ties of the k-th value.
// -0.0 counts as +0.0. Data: random, heavy ties, many -inf (ties at -inf), signed zeros, a constant row.
// Each arm runs in a forked child (the variables are read once per process). --perf prints mean ms per compute.
// Needs a GPU backend; skips (exit 0) without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

enum data_kind { D_RANDOM, D_TIES, D_NEGINF, D_ZEROS, D_CONST };
static const char * data_name(data_kind d) {
    switch (d) {
        case D_RANDOM: return "random";
        case D_TIES:   return "ties";
        case D_NEGINF: return "neg-inf";
        case D_ZEROS:  return "zeros";
        case D_CONST:  return "const";
    }
    return "?";
}

struct tk_case { int64_t ncols; int64_t k; int64_t nrows; data_kind data; };

static std::vector<tk_case> cases() {
    std::vector<tk_case> out;
    for (int64_t ncols : {1025, 2048, 25024, 25088, 100352}) {
        for (int64_t k : {64, 512, 2048}) {
            if (k > ncols) {
                continue;
            }
            for (int64_t nrows : {1, 3}) {
                for (data_kind d : {D_RANDOM, D_TIES, D_NEGINF, D_ZEROS, D_CONST}) {
                    out.push_back({ncols, k, nrows, d});
                }
            }
        }
    }
    return out;
}

static std::vector<float> make_data(const tk_case & c) {
    std::mt19937 rng(77 + (uint32_t) c.ncols*3 + (uint32_t) c.k*5 + (uint32_t) c.nrows*7 + (uint32_t) c.data*11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> lv(0, 7);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> x(c.ncols*c.nrows);
    for (auto & v : x) {
        switch (c.data) {
            case D_RANDOM: v = nd(rng); break;
            case D_TIES:   v = (float) lv(rng) * 0.25f; break;
            case D_NEGINF: v = u(rng) < 0.995f ? -INFINITY : nd(rng); break;
            case D_ZEROS:  v = u(rng) < 0.5f ? 0.0f : (u(rng) < 0.5f ? -0.0f : -1.0f); break;
            case D_CONST:  v = 3.0f; break;
        }
    }
    return x;
}

static uint32_t sel_key(float v) {
    uint32_t b;
    memcpy(&b, &v, 4);
    b = b == 0x80000000u ? 0u : b;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

static std::vector<int32_t> reference(const tk_case & c, const std::vector<float> & x) {
    std::vector<int32_t> out(c.k*c.nrows);
    std::vector<uint32_t> keys(c.ncols);
    for (int64_t r = 0; r < c.nrows; ++r) {
        for (int64_t i = 0; i < c.ncols; ++i) {
            keys[i] = sel_key(x[r*c.ncols + i]);
        }
        std::vector<uint32_t> s = keys;
        std::nth_element(s.begin(), s.begin() + (c.k - 1), s.end(), std::greater<uint32_t>());
        const uint32_t kth = s[c.k - 1];
        int64_t n_gt = 0;
        for (uint32_t kk : keys) {
            n_gt += kk > kth;
        }
        int64_t gi = 0, ei = n_gt;
        for (int64_t i = 0; i < c.ncols; ++i) {
            if (keys[i] > kth) {
                out[r*c.k + gi++] = (int32_t) i;
            } else if (keys[i] == kth && ei < c.k) {
                out[r*c.k + ei++] = (int32_t) i;
            }
        }
    }
    return out;
}

static ggml_backend_t gpu_backend() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// returns false on failure; out gets the k x nrows indices
static bool run_case(ggml_backend_t backend, const tk_case & c, int n_perf, std::vector<int32_t> & out, double & ms) {
    ggml_init_params ip = { 8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ncols, c.nrows);
    ggml_tensor * t = ggml_top_k(ctx, a, (int) c.k);
    ggml_set_output(t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = buf != nullptr;
    if (ok) {
        const std::vector<float> x = make_data(c);
        ggml_backend_tensor_set(a, x.data(), 0, ggml_nbytes(a));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out.resize(c.k*c.nrows);
        // the result may be a view of a wider buffer: read row by row
        for (int64_t r = 0; r < c.nrows; ++r) {
            ggml_backend_tensor_get(t, out.data() + r*c.k, r*t->nb[1], c.k*sizeof(int32_t));
        }
        ms = 0.0;
        if (n_perf > 0) {
            ggml_backend_synchronize(backend);
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n_perf; ++i) {
                ggml_backend_graph_compute(backend, gf);
            }
            ggml_backend_synchronize(backend);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n_perf;
        }
    }
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    printf("test-topk-select-fast: needs fork, skipped\n");
    return 0;
#else
    const bool perf = argc > 1 && strcmp(argv[1], "--perf") == 0;
    const std::vector<tk_case> cs = cases();
    const int n = (int) cs.size();
    std::vector<std::vector<int32_t>> res[2];
    std::vector<double> tms[2];
    for (int arm = 0; arm < 2; ++arm) {
        int fds[2];
        if (pipe(fds) != 0) {
            return 1;
        }
        const pid_t pid = fork();
        if (pid == 0) {
            close(fds[0]);
            setenv("GGML_CUDA_TOPK_SELECT", "3", 1);
            setenv("GGML_CUDA_TOPK_SELECT_FAST", arm ? "1" : "0", 1);
            ggml_backend_t backend = gpu_backend();
            int32_t ok = backend != nullptr;
            if (write(fds[1], &ok, 4) != 4) {
                _exit(1);
            }
            for (int i = 0; ok && i < n; ++i) {
                std::vector<int32_t> o;
                double ms = 0.0;
                int32_t good = run_case(backend, cs[i], perf ? 500 : 0, o, ms);
                o.resize(cs[i].k*cs[i].nrows, -2);
                if (write(fds[1], &good, 4) != 4 || write(fds[1], &ms, sizeof(ms)) != sizeof(ms) ||
                    write(fds[1], o.data(), o.size()*4) != (ssize_t) (o.size()*4)) {
                    _exit(1);
                }
            }
            if (backend) {
                ggml_backend_free(backend);
            }
            _exit(0);
        }
        close(fds[1]);
        FILE * f = fdopen(fds[0], "rb");
        int32_t ok = 0;
        if (fread(&ok, 4, 1, f) != 1 || !ok) {
            fclose(f);
            waitpid(pid, nullptr, 0);
            printf("test-topk-select-fast: no GPU backend, skipped\n");
            return 0;
        }
        for (int i = 0; i < n; ++i) {
            int32_t good = 0;
            double ms = 0.0;
            std::vector<int32_t> o(cs[i].k*cs[i].nrows);
            if (fread(&good, 4, 1, f) != 1 || fread(&ms, sizeof(ms), 1, f) != 1 || fread(o.data(), 4, o.size(), f) != o.size() || !good) {
                o.assign(o.size(), -3);
            }
            res[arm].push_back(o);
            tms[arm].push_back(ms);
        }
        fclose(f);
        int status = 0;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "child %d failed\n", arm);
            return 1;
        }
    }
    int fails = 0;
    for (int i = 0; i < n; ++i) {
        const tk_case & c = cs[i];
        const std::vector<int32_t> ref = reference(c, make_data(c));
        const bool same = res[0][i] == res[1][i];
        const bool plain_ref = res[0][i] == ref;
        const bool fast_ref  = res[1][i] == ref;
        const bool pass = same && fast_ref;
        fails += !pass;
        printf("ncols %6lld k %4lld rows %lld %-8s fast==plain %s fast==ref %s plain==ref %s",
               (long long) c.ncols, (long long) c.k, (long long) c.nrows, data_name(c.data),
               same ? "yes" : "NO", fast_ref ? "yes" : "NO", plain_ref ? "yes" : "NO");
        if (perf) {
            printf("  plain %.4f ms  fast %.4f ms", tms[0][i], tms[1][i]);
        }
        printf("\n");
    }
    printf("test-topk-select-fast: %d/%d pass\n", n - fails, n);
    return fails ? 1 : 0;
#endif
}
