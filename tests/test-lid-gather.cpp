// [ROUNDFIX] GGML_CUDA_LID_GATHER=1 (the lightning indexer reads its pooled keys straight from the F16 cache rows
// instead of a GET_ROWS F32 copy) must give the same score bytes as the plain two-node path.
// Graph: LIGHTNING_INDEXER(q, RESHAPE(GET_ROWS(view of F16 cache rows, I32 cells)), w, F16 mask), the shape the
// Qwen4exp helper and verify build (n_embd 128, 4 heads, 1..7 tokens, key|pooled rows of 512 bytes).
// The plain arm also proves the gather ran (its F32 copy holds the gathered keys); the fused arm proves the gather
// was skipped (the copy keeps the sentinel written before the compute). Each arm runs in a forked child.
// --perf prints mean ms per compute. Needs a GPU backend; skips (exit 0) without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

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

struct lg_case { int64_t n_cells; int64_t n_pool; int64_t n_batch; bool dups; };

static std::vector<lg_case> cases() {
    std::vector<lg_case> out;
    for (int64_t n_cells : {300, 4096, 100352}) {
        for (int64_t n_pool : {1, 63, 64, 65, 1000, 25088}) {
            if (n_pool > n_cells) {
                continue;
            }
            for (int64_t n_batch : {1, 3, 4, 7}) {
                out.push_back({n_cells, n_pool, n_batch, (n_pool % 2) == 1});
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

static const float SENTINEL = 12345.5f;

// out: n_pool x n_batch scores followed by the first 128 floats of the gathered copy
static bool run_case(ggml_backend_t backend, const lg_case & c, int n_perf, std::vector<float> & out, double & ms) {
    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    // each cache row is key | pooled, 256 halves = 512 bytes; the pooled half starts at element 128
    ggml_tensor * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, c.n_cells);
    ggml_tensor * cells = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_pool);
    ggml_tensor * q     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 4, c.n_batch);
    ggml_tensor * w     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, c.n_batch);
    ggml_tensor * m     = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_pool, c.n_batch);
    ggml_tensor * pv    = ggml_view_2d(ctx, cache, 128, c.n_cells, cache->nb[1], 128*ggml_element_size(cache));
    ggml_tensor * g     = ggml_get_rows(ctx, pv, cells);
    ggml_tensor * k     = ggml_reshape_3d(ctx, g, 128, 1, c.n_pool);
    ggml_tensor * s     = ggml_lightning_indexer(ctx, q, k, w, m);
    ggml_set_output(s);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, s);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    bool ok = buf != nullptr;
    if (ok) {
        std::mt19937 rng(91 + (uint32_t) c.n_cells*3 + (uint32_t) c.n_pool*5 + (uint32_t) c.n_batch*7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        std::vector<ggml_fp16_t> ch(256*c.n_cells);
        for (auto & v : ch) {
            v = ggml_fp32_to_fp16(nd(rng));
        }
        std::vector<int32_t> cl(c.n_pool);
        std::uniform_int_distribution<int32_t> ci(0, (int32_t) c.n_cells - 1);
        for (int64_t i = 0; i < c.n_pool; ++i) {
            cl[i] = c.dups ? ci(rng) : (int32_t) ((i*7919) % c.n_cells);
        }
        std::vector<float> qh(128*4*c.n_batch), wh(4*c.n_batch);
        for (auto & v : qh) {
            v = nd(rng);
        }
        for (auto & v : wh) {
            v = nd(rng);
        }
        std::vector<ggml_fp16_t> mh(c.n_pool*c.n_batch);
        for (auto & v : mh) {
            v = ggml_fp32_to_fp16(u(rng) < 0.2f ? -INFINITY : 0.0f);
        }
        std::vector<float> sent(ggml_nelements(g), SENTINEL);
        ggml_backend_tensor_set(cache, ch.data(), 0, ggml_nbytes(cache));
        ggml_backend_tensor_set(cells, cl.data(), 0, ggml_nbytes(cells));
        ggml_backend_tensor_set(q, qh.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(w, wh.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(m, mh.data(), 0, ggml_nbytes(m));
        ggml_backend_tensor_set(g, sent.data(), 0, ggml_nbytes(g));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        out.resize(c.n_pool*c.n_batch + 128);
        ggml_backend_tensor_get(s, out.data(), 0, c.n_pool*c.n_batch*sizeof(float));
        ggml_backend_tensor_get(g, out.data() + c.n_pool*c.n_batch, 0, 128*sizeof(float));
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
    printf("test-lid-gather: needs fork, skipped\n");
    return 0;
#else
    const bool perf = argc > 1 && strcmp(argv[1], "--perf") == 0;
    const std::vector<lg_case> cs = cases();
    const int n = (int) cs.size();
    std::vector<std::vector<float>> res[2];
    std::vector<double> tms[2];
    for (int arm = 0; arm < 2; ++arm) {
        int fds[2];
        if (pipe(fds) != 0) {
            return 1;
        }
        const pid_t pid = fork();
        if (pid == 0) {
            close(fds[0]);
            setenv("GGML_CUDA_LID_GATHER", arm ? "1" : "0", 1);
            ggml_backend_t backend = gpu_backend();
            int32_t ok = backend != nullptr;
            if (write(fds[1], &ok, 4) != 4) {
                _exit(1);
            }
            for (int i = 0; ok && i < n; ++i) {
                std::vector<float> o;
                double ms = 0.0;
                int32_t good = run_case(backend, cs[i], perf ? 500 : 0, o, ms);
                o.resize(cs[i].n_pool*cs[i].n_batch + 128, NAN);
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
            printf("test-lid-gather: no GPU backend, skipped\n");
            return 0;
        }
        for (int i = 0; i < n; ++i) {
            int32_t good = 0;
            double ms = 0.0;
            std::vector<float> o(cs[i].n_pool*cs[i].n_batch + 128);
            if (fread(&good, 4, 1, f) != 1 || fread(&ms, sizeof(ms), 1, f) != 1 || fread(o.data(), 4, o.size(), f) != o.size() || !good) {
                o.assign(o.size(), NAN);
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
        const lg_case & c = cs[i];
        const size_t ns = c.n_pool*c.n_batch;
        const bool same = memcmp(res[0][i].data(), res[1][i].data(), ns*sizeof(float)) == 0;
        bool plain_gathered = false, fused_skipped = true;
        for (int j = 0; j < 128; ++j) {
            plain_gathered |= res[0][i][ns + j] != SENTINEL;
            fused_skipped  &= res[1][i][ns + j] == SENTINEL;
        }
        const bool pass = same && plain_gathered && fused_skipped;
        fails += !pass;
        printf("cells %6lld pool %5lld batch %lld dups %d  scores identical %s  plain gathered %s  fused skipped %s",
               (long long) c.n_cells, (long long) c.n_pool, (long long) c.n_batch, c.dups ? 1 : 0,
               same ? "yes" : "NO", plain_gathered ? "yes" : "NO", fused_skipped ? "yes" : "NO");
        if (perf) {
            printf("  plain %.4f ms  fused %.4f ms", tms[0][i], tms[1][i]);
        }
        printf("\n");
    }
    printf("test-lid-gather: %d/%d pass\n", n - fails, n);
    return fails ? 1 : 0;
#endif
}
