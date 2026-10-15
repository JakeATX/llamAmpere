// Expert keep policy (#215 MTP draft, #217 lossy trunk map), CPU-only checks.
// llama_decode applies the #217 map only when the call is tagged GENERATION
// (tests/test-decode-phase.cpp). This file sets the scheduler policy directly.
// A node wider than 64 routed rows still skips the mask, on top of that tag.
//
// With no MoE cache provider nothing is resident, so keep(L) = k leaves out every routed row at route rank >= k.
// Checked here:
//   - the --moe-keep-map parser (valid and invalid maps),
//   - policy off (never set, cleared, keep >= top-k, or a node wider than 64 rows) is bit-identical to the base,
//   - keep k: MUL_MAT_ID rows at rank < k are bit-identical to the base, rows at rank >= k are exactly zero,
//   - a full two-layer MoE FFN under keep k (and under a per-layer map) is bit-identical to the base graph run
//     with the router weights of the left-out ranks set to zero (the expert contributes exactly nothing),
// for F32 and Q4_0 expert weights, 1 and 4 CPU threads.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-moe-cache.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, ...) do { \
    if (cond) { g_pass++; } else { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

// ---------------------------------------------------------------------------------------------------------------
// parser

static void test_parser() {
    int32_t def = 0;
    std::vector<int32_t> layers;
    std::string err;

    CHECK(llama_parse_moe_keep_map(nullptr, 8, def, layers, err) && def == -1 && layers.empty(), "null map = off");
    CHECK(llama_parse_moe_keep_map("", 8, def, layers, err) && def == -1 && layers.empty(), "empty map = off");
    CHECK(llama_parse_moe_keep_map("3", 8, def, layers, err) && def == 3 && layers.empty(), "plain N");
    CHECK(llama_parse_moe_keep_map("0", 8, def, layers, err) && def == 0 && layers.empty(), "plain 0");
    CHECK(llama_parse_moe_keep_map("*:4", 8, def, layers, err) && def == 4 && layers.empty(), "*:N");

    bool ok = llama_parse_moe_keep_map("0-1:10,5:2", 8, def, layers, err);
    CHECK(ok && def == -1 && layers.size() == 8, "range + single, size %zu", layers.size());
    if (ok && layers.size() == 8) {
        const int32_t want[8] = {10, 10, -1, -1, -1, 2, -1, -1};
        CHECK(memcmp(layers.data(), want, sizeof(want)) == 0, "range + single values");
    }

    ok = llama_parse_moe_keep_map("6-7:1,*:5", 8, def, layers, err);
    CHECK(ok && def == 5 && layers.size() == 8, "default after list");
    if (ok && layers.size() == 8) {
        const int32_t want[8] = {5, 5, 5, 5, 5, 5, 1, 1};
        CHECK(memcmp(layers.data(), want, sizeof(want)) == 0, "unlisted layers take the * value regardless of order");
    }

    const char * bad[] = {
        "x", "-1", "3,4", "1:", ":3", "1:x", "2-1:3", "8:2", "0-8:2", "1:2,", "a-b:2", "1:-2", "1234567",
    };
    for (const char * b : bad) {
        err.clear();
        CHECK(!llama_parse_moe_keep_map(b, 8, def, layers, err) && !err.empty(), "map '%s' must be rejected", b);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// graph fixture

struct fixture {
    int n_embd   = 64;
    int n_ff     = 32;
    int n_expert = 16;
    int n_used   = 6;
    int n_layer  = 2;
    ggml_type wtype = GGML_TYPE_F32;

    ggml_context * wctx = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;
    std::vector<ggml_tensor *> up, gate, down;
};

static void fill_weight(ggml_tensor * t, std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, 0.2f);
    const int64_t n = ggml_nelements(t);
    std::vector<float> f(n);
    for (auto & v : f) {
        v = dist(rng);
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, f.data(), 0, n * sizeof(float));
        return;
    }
    std::vector<uint8_t> q(ggml_nbytes(t));
    const int64_t n_per_row = t->ne[0];
    ggml_quantize_chunk(t->type, f.data(), q.data(), 0, n / n_per_row, n_per_row, nullptr);
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void fixture_init(fixture & fx, ggml_backend_t cpu, ggml_type wtype) {
    fx.wtype = wtype;
    ggml_init_params ip = { ggml_tensor_overhead() * 16, nullptr, true };
    fx.wctx = ggml_init(ip);
    for (int il = 0; il < fx.n_layer; il++) {
        ggml_tensor * u = ggml_new_tensor_3d(fx.wctx, wtype, fx.n_embd, fx.n_ff, fx.n_expert);
        ggml_tensor * g = ggml_new_tensor_3d(fx.wctx, wtype, fx.n_embd, fx.n_ff, fx.n_expert);
        ggml_tensor * d = ggml_new_tensor_3d(fx.wctx, wtype, fx.n_ff, fx.n_embd, fx.n_expert);
        ggml_format_name(u, "blk.%d.ffn_up_exps.weight", il);
        ggml_format_name(g, "blk.%d.ffn_gate_exps.weight", il);
        ggml_format_name(d, "blk.%d.ffn_down_exps.weight", il);
        fx.up.push_back(u);
        fx.gate.push_back(g);
        fx.down.push_back(d);
    }
    fx.wbuf = ggml_backend_alloc_ctx_tensors(fx.wctx, cpu);
    std::mt19937 rng(1234);
    for (int il = 0; il < fx.n_layer; il++) {
        fill_weight(fx.up[il], rng);
        fill_weight(fx.gate[il], rng);
        fill_weight(fx.down[il], rng);
    }
}

static void fixture_free(fixture & fx) {
    ggml_backend_buffer_free(fx.wbuf);
    ggml_free(fx.wctx);
    fx = fixture();
}

struct run_inputs {
    int n_tokens = 1;
    std::vector<float> x;                     // n_embd * n_tokens
    std::vector<int32_t> ids;                 // n_used * n_tokens, per token distinct, rank order
    std::vector<std::vector<float>> weights;  // per layer, n_used * n_tokens
};

struct run_outputs {
    std::vector<float> out;   // n_embd * n_tokens
    std::vector<float> up0;   // layer-0 up projection, n_ff * n_used * n_tokens
};

static run_inputs make_inputs(const fixture & fx, int n_tokens, uint32_t seed) {
    run_inputs in;
    in.n_tokens = n_tokens;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    in.x.resize((size_t) fx.n_embd * n_tokens);
    for (auto & v : in.x) {
        v = dist(rng);
    }
    in.ids.resize((size_t) fx.n_used * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        std::vector<int32_t> perm(fx.n_expert);
        for (int e = 0; e < fx.n_expert; e++) {
            perm[e] = e;
        }
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int r = 0; r < fx.n_used; r++) {
            in.ids[(size_t) t * fx.n_used + r] = perm[r];
        }
    }
    std::uniform_real_distribution<float> wd(0.05f, 0.5f);
    in.weights.assign(fx.n_layer, std::vector<float>((size_t) fx.n_used * n_tokens));
    for (auto & w : in.weights) {
        for (auto & v : w) {
            v = wd(rng);
        }
    }
    return in;
}

// Two MoE FFN layers with a residual, the same op pattern as build_moe_ffn (views + add over the routed slots).
static bool run_graph(const fixture & fx, ggml_backend_t cpu, const run_inputs & in,
                      int32_t keep_default, const std::vector<int32_t> & keep_layer, bool set_policy,
                      run_outputs & res) {
    const int T = in.n_tokens;
    ggml_init_params ip = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, fx.n_embd, T);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, fx.n_used, T);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);
    std::vector<ggml_tensor *> w(fx.n_layer);
    for (int il = 0; il < fx.n_layer; il++) {
        w[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, fx.n_used, T);
        ggml_format_name(w[il], "w%d", il);
        ggml_set_input(w[il]);
    }

    ggml_tensor * cur = x;
    ggml_tensor * up0 = nullptr;
    for (int il = 0; il < fx.n_layer; il++) {
        ggml_tensor * x3 = ggml_reshape_3d(ctx, cur, fx.n_embd, 1, T);
        ggml_tensor * up = ggml_mul_mat_id(ctx, fx.up[il], x3, ids);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, fx.gate[il], x3, ids);
        if (il == 0) {
            up0 = up;
            ggml_set_output(up0);
        }
        ggml_tensor * glu = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * dn = ggml_mul_mat_id(ctx, fx.down[il], glu, ids);  // [n_embd, n_used, T]
        dn = ggml_mul(ctx, dn, w[il]);
        ggml_tensor * moe = nullptr;
        for (int r = 0; r < fx.n_used; r++) {
            ggml_tensor * v = ggml_view_2d(ctx, dn, fx.n_embd, T, dn->nb[2], r * dn->nb[1]);
            moe = moe ? ggml_add(ctx, moe, v) : v;
        }
        cur = ggml_add(ctx, cur, moe);
    }
    ggml_set_name(cur, "out");
    ggml_set_output(cur);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, up0);
    ggml_build_forward_expand(gf, cur);

    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (set_policy) {
        ggml_backend_sched_set_moe_expert_keep(sched, keep_default,
                keep_layer.empty() ? nullptr : keep_layer.data(), (int32_t) keep_layer.size());
    }
    bool ok = ggml_backend_sched_alloc_graph(sched, gf);
    if (ok) {
        ggml_backend_tensor_set(x, in.x.data(), 0, in.x.size() * sizeof(float));
        ggml_backend_tensor_set(ids, in.ids.data(), 0, in.ids.size() * sizeof(int32_t));
        for (int il = 0; il < fx.n_layer; il++) {
            ggml_backend_tensor_set(w[il], in.weights[il].data(), 0, in.weights[il].size() * sizeof(float));
        }
        ok = ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS;
    }
    if (ok) {
        res.out.resize(ggml_nelements(cur));
        ggml_backend_tensor_get(cur, res.out.data(), 0, res.out.size() * sizeof(float));
        res.up0.resize(ggml_nelements(up0));
        ggml_backend_tensor_get(up0, res.up0.data(), 0, res.up0.size() * sizeof(float));
    }
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}

static bool bits_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static bool all_finite(const std::vector<float> & a) {
    for (float v : a) {
        if (!std::isfinite(v)) {
            return false;
        }
    }
    return true;
}

// reference inputs: router weights of left-out ranks set to zero
static run_inputs zero_ranks(const fixture & fx, const run_inputs & in, const std::vector<int32_t> & keep_per_layer) {
    run_inputs ref = in;
    for (int il = 0; il < fx.n_layer; il++) {
        const int32_t k = keep_per_layer[il];
        if (k < 0) {
            continue;
        }
        for (int t = 0; t < in.n_tokens; t++) {
            for (int r = k; r < fx.n_used; r++) {
                ref.weights[il][(size_t) t * fx.n_used + r] = 0.0f;
            }
        }
    }
    return ref;
}

static void test_graphs(ggml_type wtype, int n_threads) {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, n_threads);
    fixture fx;
    fixture_init(fx, cpu, wtype);
    const char * tname = ggml_type_name(wtype);

    for (int n_tokens : {1, 3, 10}) {  // 6, 18, 60 routed rows: all within the 64-row policy window
        const run_inputs in = make_inputs(fx, n_tokens, 77u + n_tokens);
        run_outputs base;
        CHECK(run_graph(fx, cpu, in, -1, {}, false, base) && all_finite(base.out), "%s T=%d base run", tname, n_tokens);

        // off: cleared policy and keep >= top-k leave the graph bit-identical
        run_outputs off;
        CHECK(run_graph(fx, cpu, in, -1, {}, true, off) && bits_equal(off.out, base.out), "%s T=%d cleared policy", tname, n_tokens);
        CHECK(run_graph(fx, cpu, in, fx.n_used, {}, true, off) && bits_equal(off.out, base.out), "%s T=%d keep=top-k", tname, n_tokens);
        CHECK(run_graph(fx, cpu, in, 99, {}, true, off) && bits_equal(off.out, base.out), "%s T=%d keep>top-k", tname, n_tokens);

        for (int k : {0, 2, 5}) {
            run_outputs got;
            const bool ok = run_graph(fx, cpu, in, k, {}, true, got);
            CHECK(ok, "%s T=%d keep=%d run", tname, n_tokens, k);
            if (!ok) {
                continue;
            }
            // per-row: kept rows identical to base, left-out rows exactly zero
            bool rows_ok = true;
            for (int t = 0; t < n_tokens; t++) {
                for (int r = 0; r < fx.n_used; r++) {
                    const size_t off_row = ((size_t) t * fx.n_used + r) * fx.n_ff;
                    for (int c = 0; c < fx.n_ff; c++) {
                        const float g = got.up0[off_row + c];
                        const float b = base.up0[off_row + c];
                        if (r < k ? memcmp(&g, &b, sizeof(float)) != 0 : g != 0.0f) {
                            rows_ok = false;
                        }
                    }
                }
            }
            CHECK(rows_ok, "%s T=%d keep=%d MUL_MAT_ID rows (kept = base bits, left out = 0)", tname, n_tokens, k);

            // full FFN equals the base graph with the left-out ranks' router weights zeroed
            run_outputs ref;
            const run_inputs ref_in = zero_ranks(fx, in, std::vector<int32_t>(fx.n_layer, k));
            CHECK(run_graph(fx, cpu, ref_in, -1, {}, false, ref) && bits_equal(got.out, ref.out),
                  "%s T=%d keep=%d FFN = zero-weight reference", tname, n_tokens, k);
            if (k < fx.n_used) {
                CHECK(!bits_equal(got.out, base.out), "%s T=%d keep=%d output changed vs base", tname, n_tokens, k);
            }
        }

        // per-layer map: layer 0 keeps everything, layer 1 keeps 2
        {
            const std::vector<int32_t> map = {-1, 2};
            run_outputs got, ref;
            const bool ok = run_graph(fx, cpu, in, -1, map, true, got);
            CHECK(ok && bits_equal(got.up0, base.up0), "%s T=%d map {-1,2}: layer 0 untouched", tname, n_tokens);
            const run_inputs ref_in = zero_ranks(fx, in, map);
            CHECK(ok && run_graph(fx, cpu, ref_in, -1, {}, false, ref) && bits_equal(got.out, ref.out),
                  "%s T=%d map {-1,2}: FFN = zero-weight reference", tname, n_tokens);
        }
        // per-layer map shorter than the model: layers past the table take keep_default
        {
            const std::vector<int32_t> map = {3};
            run_outputs got, ref;
            const bool ok = run_graph(fx, cpu, in, 1, map, true, got);
            const run_inputs ref_in = zero_ranks(fx, in, {3, 1});
            CHECK(ok && run_graph(fx, cpu, ref_in, -1, {}, false, ref) && bits_equal(got.out, ref.out),
                  "%s T=%d map {3} default 1: FFN = zero-weight reference", tname, n_tokens);
        }
    }

    // wider than 64 routed rows (prefill): the policy does not apply
    {
        const run_inputs in = make_inputs(fx, 11, 991u);  // 66 rows
        run_outputs base, got;
        CHECK(run_graph(fx, cpu, in, -1, {}, false, base) &&
              run_graph(fx, cpu, in, 2, {}, true, got) && bits_equal(got.out, base.out),
              "%s T=11 (66 rows) keep=2 ignored, bit-identical", tname);
    }

    // the policy is per scheduler and scoped to its compute: a policy-free sched after a keep sched is unaffected
    {
        const run_inputs in = make_inputs(fx, 2, 5u);
        run_outputs a, b, c;
        CHECK(run_graph(fx, cpu, in, -1, {}, false, a) && run_graph(fx, cpu, in, 0, {}, true, b) &&
              run_graph(fx, cpu, in, -1, {}, false, c) && bits_equal(a.out, c.out) && !bits_equal(a.out, b.out),
              "%s policy does not leak across schedulers", tname);
    }

    fixture_free(fx);
    ggml_backend_free(cpu);
}

int main() {
    ggml_backend_load_all();
    test_parser();
    // outside any scheduler scope nothing is kept out
    CHECK(ggml_moe_cache_expert_keep("blk.3.ffn_up_exps.weight") == -1, "no active policy outside a scheduler");
    for (ggml_type t : {GGML_TYPE_F32, GGML_TYPE_Q4_0}) {
        for (int nth : {1, 4}) {
            test_graphs(t, nth);
        }
    }
    printf("test-moe-expert-keep: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
