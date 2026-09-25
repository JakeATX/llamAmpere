// [#64] FlashAttention f16-conversion scratch: the reserved region behind dst and the pool fallback must give
// byte-identical output.
//
// ggml_cuda_flash_attn_ext_get_alloc_size reserves room for the f16 K/V copies behind dst, sized by the route that
// will run (GGML_CUDA_FATTN_ALLOC_ROUTE). launch_fattn (fattn-common.cuh) converts into that region when the
// allocation covers it and into the pool otherwise. Each case runs one FLASH_ATTN_EXT twice on the same inputs:
//   arm 1: dst allocated normally (ggml_backend_alloc_ctx_tensors -> the buffer type's alloc size), reserved region;
//   arm 2: dst is a view of a plain f32 tensor (dst->view_src != nullptr), so launch_fattn takes the pool.
// Pass: the two outputs are memcmp-equal, and the fallback ledger's "cuda.fattn" f16_convert counter shows
// scratch=reserved only in arm 1 and scratch=pool only in arm 2 for the converting pair, and no conversion at all for
// the f16/f16 control. K q4_0 / V q8_0 converts at every width here on SM86: widths 4 and 8 route to MMA (always f16
// K/V), width 1 routes to VEC, which has no q4_0-q8_0 instance in the default GGML_CUDA_FA_QUANTS list and falls
// back to f16. A build that compiles that instance fails width 1 with "did not convert".
// test-backend-ops compares a backend against the CPU and cannot read the ledger, hence this file.
// CUDA graphs are off (GGML_CUDA_DISABLE_GRAPHS) so every compute launches on the host and is counted.
// Needs a GPU backend; skips (exit 0) without one.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ledger.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct test_case_def {
    ggml_type type_K;
    ggml_type type_V;
    int64_t   D;
    int64_t   n_kv;
    int64_t   n_q;
    bool      converts; // the case must go through the f16 conversion
};

static constexpr int64_t n_head    = 8;
static constexpr int64_t n_head_kv = 2; // GQA 4, like the models the route decisions are tuned for

static std::vector<test_case_def> cases() {
    std::vector<test_case_def> out;
    for (int pair = 0; pair < 2; ++pair) {
        for (int64_t D : { 128, 256 }) {
            for (int64_t n_kv : { 1024, 4096 }) {
                for (int64_t n_q : { 1, 4, 8 }) {
                    if (pair == 0) {
                        out.push_back({ GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, D, n_kv, n_q, true });
                    } else {
                        out.push_back({ GGML_TYPE_F16, GGML_TYPE_F16, D, n_kv, n_q, false });
                    }
                }
            }
        }
    }
    return out;
}

struct ledger_counts {
    int64_t reserved = 0;
    int64_t pool     = 0;
};

static void ledger_cb(const char * site, const char * key, int64_t count, void * user_data) {
    ledger_counts * c = (ledger_counts *) user_data;
    if (strcmp(site, "cuda.fattn") != 0 || strncmp(key, "f16_convert ", 12) != 0) {
        return;
    }
    if (strstr(key, "scratch=reserved") != nullptr) {
        c->reserved += count;
    } else if (strstr(key, "scratch=pool") != nullptr) {
        c->pool += count;
    }
}

static void set_quantized(ggml_tensor * t, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t n_per_row = t->ne[0];
    const int64_t nrows     = ggml_nelements(t) / n_per_row;
    std::vector<float> f(ggml_nelements(t));
    for (float & x : f) {
        x = nd(rng);
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) q.data(), (int64_t) f.size());
    } else {
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
    }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

// arm 1 (as_view == false): dst gets the buffer type's alloc size; arm 2: dst is a view of `backing`
static void run_arm(ggml_backend_t backend, const test_case_def & c, bool as_view, std::vector<uint8_t> & out_bytes) {
    ggml_init_params pi = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx_in = ggml_init(pi);
    ggml_tensor * q       = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F32, c.D, c.n_q, n_head, 1);
    ggml_tensor * k       = ggml_new_tensor_4d(ctx_in, c.type_K, c.D, c.n_kv, n_head_kv, 1);
    ggml_tensor * v       = ggml_new_tensor_4d(ctx_in, c.type_V, c.D, c.n_kv, n_head_kv, 1);
    ggml_tensor * m       = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F16, c.n_kv, c.n_q, 1, 1);
    ggml_tensor * backing = ggml_new_tensor_4d(ctx_in, GGML_TYPE_F32, c.D, n_head, c.n_q, 1); // dst shape
    ggml_backend_buffer_t buf_in = ggml_backend_alloc_ctx_tensors(ctx_in, backend);
    GGML_ASSERT(buf_in);

    // same inputs in both arms
    std::mt19937 rng(7 + 31*c.D + 1009*c.n_kv + 17*c.n_q + 131*c.type_K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> vq(ggml_nelements(q));
    for (float & x : vq) {
        x = nd(rng);
    }
    ggml_backend_tensor_set(q, vq.data(), 0, ggml_nbytes(q));
    set_quantized(k, rng);
    set_quantized(v, rng);
    // causal mask over the last n_q positions of the cache
    std::vector<ggml_fp16_t> vm(ggml_nelements(m));
    for (int64_t j = 0; j < c.n_q; ++j) {
        for (int64_t i = 0; i < c.n_kv; ++i) {
            vm[j*c.n_kv + i] = ggml_fp32_to_fp16(i <= c.n_kv - c.n_q + j ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(m, vm.data(), 0, ggml_nbytes(m));
    std::vector<float> zero(ggml_nelements(backing), 0.0f);
    ggml_backend_tensor_set(backing, zero.data(), 0, ggml_nbytes(backing));

    ggml_init_params pg = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(pg);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf((float) c.D), 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    GGML_ASSERT(ggml_nbytes(out) == ggml_nbytes(backing));

    ggml_backend_buffer_t buf_out = nullptr;
    if (as_view) {
        out->view_src  = backing;
        out->view_offs = 0;
        GGML_ASSERT(ggml_backend_view_init(out) == GGML_STATUS_SUCCESS);
    } else {
        buf_out = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf_out);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    out_bytes.resize(ggml_nbytes(out));
    ggml_backend_tensor_get(out, out_bytes.data(), 0, out_bytes.size());

    if (buf_out) {
        ggml_backend_buffer_free(buf_out);
    }
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_in);
    ggml_free(ctx_in);
}

int main() {
    // every compute must launch on the host so the ledger sees it (a CUDA graph replay would not count)
#ifdef _WIN32
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
#else
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
#endif
    ggml_ledger_set_enabled(true);

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        printf("test-fattn-alloc-route: no GPU backend, skipping\n");
        return 0;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);

    int n_fail = 0;
    const std::vector<test_case_def> all = cases();
    for (const test_case_def & c : all) {
        std::vector<uint8_t> out1, out2;
        ledger_counts l1, l2;

        ggml_ledger_reset();
        run_arm(backend, c, false, out1);
        ggml_ledger_foreach(ledger_cb, &l1);

        ggml_ledger_reset();
        run_arm(backend, c, true, out2);
        ggml_ledger_foreach(ledger_cb, &l2);

        std::string why;
        const bool equal = out1.size() == out2.size() && memcmp(out1.data(), out2.data(), out1.size()) == 0;
        if (!equal) {
            // tell a route difference from a kernel that is not deterministic run to run
            std::vector<uint8_t> out1b;
            run_arm(backend, c, false, out1b);
            why += out1b == out1 ? " bytes_differ" : " bytes_differ(arm1_not_repeatable)";
        }
        if (c.converts) {
            if (l1.reserved == 0 && l1.pool == 0 && l2.reserved == 0 && l2.pool == 0) {
                why += " did_not_convert";
            } else {
                if (l1.reserved == 0 || l1.pool != 0) {
                    why += " arm1_not_reserved";
                }
                if (l2.pool == 0 || l2.reserved != 0) {
                    why += " arm2_not_pool";
                }
            }
        } else if (l1.reserved != 0 || l1.pool != 0 || l2.reserved != 0 || l2.pool != 0) {
            why += " control_converted";
        }

        printf("K=%-5s V=%-5s D=%3lld n_kv=%4lld n_q=%lld  arm1 reserved=%lld pool=%lld  arm2 reserved=%lld pool=%lld  %s%s\n",
               ggml_type_name(c.type_K), ggml_type_name(c.type_V), (long long) c.D, (long long) c.n_kv, (long long) c.n_q,
               (long long) l1.reserved, (long long) l1.pool, (long long) l2.reserved, (long long) l2.pool,
               why.empty() ? "OK" : "FAIL:", why.c_str());
        n_fail += !why.empty();
    }

    ggml_backend_free(backend);

    printf("test-fattn-alloc-route: %d/%d cases passed\n", (int) all.size() - n_fail, (int) all.size());
    return n_fail == 0 ? 0 : 1;
}
