// [ROUNDFIX] GGML_CUDA_FA_SKIP_DEAD=1 must be bit-identical to the plain kernel.
//
// The compressed-KV MMA flash-attention kernel (tq5_0/tq6_0/q8_0 K with turbo4 V, the Flash-Next helper's cache)
// may leave out a KV tile whose mask is -inf for every query row of the tile: such a tile adds exactly nothing.
// Each case builds one FLASH_ATTN_EXT with the helper's shapes (D 256, 24 query heads on 2 KV heads) and a sparse
// selection mask like the QSA indexer's (blocks of 4 cells plus a recent window), computes it on the GPU in two
// forked children (the variable is read once per process), hashes the output and compares byte for byte.
// Masks: block selection, fully visible, a few live tiles, every tile dead but one, and an in-place strided mask view.
// With --perf it also prints the mean time per compute for both settings.
// Needs a GPU backend; skips (exit 0) without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

enum mask_kind { MASK_SELECT, MASK_ALL, MASK_FEW, MASK_ONE, MASK_SELECT_STRIDED };

struct fa_case {
    ggml_type type_k;
    ggml_type type_v;
    int64_t   n_q;
    int64_t   n_kv;
    mask_kind mask;
};

static const char * mask_name(mask_kind m) {
    switch (m) {
        case MASK_SELECT:         return "select";
        case MASK_ALL:            return "all";
        case MASK_FEW:            return "few";
        case MASK_ONE:            return "one";
        case MASK_SELECT_STRIDED: return "select-strided";
    }
    return "?";
}

static std::vector<fa_case> cases() {
    std::vector<fa_case> out;
    for (int64_t n_q : {1, 2, 4}) {
        for (int64_t n_kv : {8192, 32768, 100352}) {
            for (mask_kind m : {MASK_SELECT, MASK_ALL, MASK_FEW, MASK_ONE, MASK_SELECT_STRIDED}) {
                out.push_back({GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0, n_q, n_kv, m});
            }
        }
    }
    for (ggml_type tk : {GGML_TYPE_TQ6_0, GGML_TYPE_Q8_0}) {
        out.push_back({tk, GGML_TYPE_TURBO4_0, 1, 32768, MASK_SELECT});
        out.push_back({tk, GGML_TYPE_TURBO4_0, 3, 32768, MASK_SELECT});
    }
    return out;
}

static constexpr int64_t D        = 256;
static constexpr int64_t n_head   = 24;
static constexpr int64_t n_head_kv = 2;

static void fill_quant(std::vector<uint8_t> & dst, ggml_type type, int64_t n_per_row, int64_t nrows, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> row(n_per_row);
    const size_t row_size = ggml_row_size(type, n_per_row);
    dst.resize(row_size*nrows);
    const auto * tt = ggml_get_type_traits(type);
    for (int64_t r = 0; r < nrows; ++r) {
        for (auto & x : row) {
            x = nd(rng);
        }
        tt->from_float_ref(row.data(), dst.data() + r*row_size, n_per_row);
    }
}

// rows x n_kv mask values (0 visible, -inf hidden) for the given pattern
static std::vector<float> make_mask(mask_kind kind, int64_t n_kv, int64_t rows, std::mt19937 & rng) {
    std::vector<float> m(n_kv*rows, -INFINITY);
    for (int64_t r = 0; r < rows; ++r) {
        float * mr = m.data() + r*n_kv;
        switch (kind) {
            case MASK_ALL:
                std::fill(mr, mr + n_kv, 0.0f);
                break;
            case MASK_SELECT:
            case MASK_SELECT_STRIDED: {
                // the indexer keeps 2048 blocks of 4 cells, clustered: a recent window plus scattered runs
                const int64_t n_blocks = n_kv/4;
                int64_t keep = std::min<int64_t>(2048, n_blocks);
                for (int64_t b = n_blocks - 64; b < n_blocks && keep > 0; ++b, --keep) {
                    std::fill(mr + 4*b, mr + 4*b + 4, 0.0f);
                }
                std::uniform_int_distribution<int64_t> start(0, n_blocks - 1);
                std::uniform_int_distribution<int>     run(1, 24);
                while (keep > 0) {
                    const int64_t b0 = start(rng);
                    const int     len = run(rng);
                    for (int64_t b = b0; b < std::min(n_blocks, b0 + len) && keep > 0; ++b, --keep) {
                        std::fill(mr + 4*b, mr + 4*b + 4, 0.0f);
                    }
                }
                break;
            }
            case MASK_FEW:
                for (int64_t k = 0; k < n_kv; k += 4096) {
                    mr[k + (r*7) % 64] = 0.0f;
                }
                mr[n_kv - 1] = 0.0f;
                break;
            case MASK_ONE:
                mr[n_kv - 1 - r] = 0.0f;
                break;
        }
    }
    return m;
}

struct run_out {
    uint64_t hash;
    double   ms;
};

static uint64_t fnv(const void * p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ b[i]) * 1099511628211ull;
    }
    return h;
}

static bool run_case(ggml_backend_t backend, const fa_case & c, int n_perf, run_out & out) {
    std::mt19937 rng(1234 + (uint32_t) c.n_kv + 17*(uint32_t) c.n_q + 101*(uint32_t) c.mask + 7*(uint32_t) c.type_k);

    const int64_t mask_rows = c.n_q;
    const int64_t pad       = c.mask == MASK_SELECT_STRIDED ? 515 : 0;

    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, c.n_q, n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, c.type_k, D, c.n_kv, n_head_kv);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, c.type_v, D, c.n_kv, n_head_kv);
    ggml_tensor * mw = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, c.n_kv + pad, mask_rows);
    ggml_tensor * m  = pad ? ggml_view_2d(ctx, mw, c.n_kv, mask_rows, mw->nb[1], 0) : mw;
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    ggml_set_output(o);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return false;
    }
    if (!ggml_backend_supports_op(backend, o)) {
        fprintf(stderr, "  unsupported on this backend\n");
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> qd(ggml_nelements(q));
        for (auto & x : qd) {
            x = nd(rng);
        }
        ggml_backend_tensor_set(q, qd.data(), 0, ggml_nbytes(q));
        std::vector<uint8_t> kd, vd;
        fill_quant(kd, c.type_k, D, c.n_kv*n_head_kv, rng);
        fill_quant(vd, c.type_v, D, c.n_kv*n_head_kv, rng);
        ggml_backend_tensor_set(k, kd.data(), 0, kd.size());
        ggml_backend_tensor_set(v, vd.data(), 0, vd.size());
        const std::vector<float> mf = make_mask(c.mask, c.n_kv, mask_rows, rng);
        std::vector<ggml_fp16_t> mh((c.n_kv + pad)*mask_rows, ggml_fp32_to_fp16(0.0f));
        for (int64_t r = 0; r < mask_rows; ++r) {
            for (int64_t i = 0; i < c.n_kv; ++i) {
                mh[r*(c.n_kv + pad) + i] = ggml_fp32_to_fp16(mf[r*c.n_kv + i]);
            }
        }
        ggml_backend_tensor_set(mw, mh.data(), 0, ggml_nbytes(mw));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }
    std::vector<uint8_t> res(ggml_nbytes(o));
    ggml_backend_tensor_get(o, res.data(), 0, res.size());
    out.hash = fnv(res.data(), res.size());

    // a second compute must give the same bytes
    ggml_backend_graph_compute(backend, gf);
    std::vector<uint8_t> res2(ggml_nbytes(o));
    ggml_backend_tensor_get(o, res2.data(), 0, res2.size());
    if (memcmp(res.data(), res2.data(), res.size()) != 0) {
        fprintf(stderr, "  second compute differs\n");
        out.hash ^= 1;
    }

    out.ms = 0.0;
    if (n_perf > 0) {
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n_perf; ++i) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);
        out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n_perf;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
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

int main(int argc, char ** argv) {
#ifdef _WIN32
    printf("test-fa-skip-dead: needs fork, skipped\n");
    return 0;
#else
    const bool perf = argc > 1 && strcmp(argv[1], "--perf") == 0;
    const std::vector<fa_case> cs = cases();
    const int n = (int) cs.size();

    std::vector<run_out> res[2];
    bool have_gpu = true;
    for (int arm = 0; arm < 2; ++arm) {
        int fds[2];
        if (pipe(fds) != 0) {
            return 1;
        }
        const pid_t pid = fork();
        if (pid == 0) {
            close(fds[0]);
            setenv("GGML_CUDA_FA_SKIP_DEAD", arm ? "1" : "0", 1);
            ggml_backend_t backend = gpu_backend();
            int32_t ok = backend != nullptr;
            if (write(fds[1], &ok, sizeof(ok)) != sizeof(ok)) {
                _exit(1);
            }
            for (int i = 0; ok && i < n; ++i) {
                run_out ro = {0, 0.0};
                if (!run_case(backend, cs[i], perf ? 200 : 0, ro)) {
                    ro.hash = 0;
                }
                if (write(fds[1], &ro, sizeof(ro)) != sizeof(ro)) {
                    _exit(1);
                }
            }
            if (backend) {
                ggml_backend_free(backend);
            }
            _exit(0);
        }
        close(fds[1]);
        int32_t ok = 0;
        if (read(fds[0], &ok, sizeof(ok)) != sizeof(ok)) {
            ok = 0;
        }
        if (!ok) {
            have_gpu = false;
        }
        for (int i = 0; ok && i < n; ++i) {
            run_out ro = {0, 0.0};
            if (read(fds[0], &ro, sizeof(ro)) != sizeof(ro)) {
                ro.hash = 0;
            }
            res[arm].push_back(ro);
        }
        close(fds[0]);
        int status = 0;
        waitpid(pid, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "child %d failed\n", arm);
            return 1;
        }
    }
    if (!have_gpu) {
        printf("test-fa-skip-dead: no GPU backend, skipped\n");
        return 0;
    }

    int fails = 0;
    for (int i = 0; i < n; ++i) {
        const fa_case & c = cs[i];
        const bool same = res[0][i].hash != 0 && res[0][i].hash == res[1][i].hash;
        fails += !same;
        printf("%-6s/%-7s n_q %d n_kv %6" PRId64 " mask %-14s %s", ggml_type_name(c.type_k), ggml_type_name(c.type_v),
               (int) c.n_q, c.n_kv, mask_name(c.mask), same ? "IDENTICAL" : "DIFFERENT");
        if (perf) {
            printf("  plain %.4f ms  skip %.4f ms", res[0][i].ms, res[1][i].ms);
        }
        printf("\n");
    }
    printf("test-fa-skip-dead: %d/%d identical\n", n - fails, n);
    return fails ? 1 : 0;
#endif
}
