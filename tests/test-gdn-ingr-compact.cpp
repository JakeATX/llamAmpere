// [#63] GGML_OP_GATED_DELTA_NET emit_mode 2 (compact ingredients: g and beta stored once per head) against
// emit_mode 1 (k, v, g, beta each S_v wide), on every available backend (CPU, and the GPU when present).
//
// Per case, both modes run on the same inputs and must agree exactly on the attention output, the trailing
// final-state block and (n_tokens > K) the before-the-window block. Every written slot of either layout must
// hold the inputs of its token: k, v, g (scalar or per-channel for KDA) and beta. Slots a call leaves
// untouched (n_tokens < K) are skipped. No model is needed.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct gdn_case {
    int64_t S_v;
    int64_t H;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t K;
    bool    kda;
};

struct gdn_inputs {
    std::vector<float> q, k, v, g, beta, state;
};

static std::vector<float> run_gdn(ggml_backend_t backend, const gdn_case & c, const gdn_inputs & in, int32_t emit_mode) {
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    const int64_t G = c.kda ? c.S_v : 1;
    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * tg = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, G,     c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * tb = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,     c.H, c.n_tokens, c.n_seqs);
    ggml_tensor * ts = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.S_v, c.S_v, c.H, c.n_seqs);

    ggml_tensor * out = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tb, ts, c.K, emit_mode);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    std::vector<float> res;
    if (!ggml_backend_supports_op(backend, out)) {
        ggml_free(ctx);
        return res;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(tq, in.q.data(),     0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tk, in.k.data(),     0, ggml_nbytes(tk));
    ggml_backend_tensor_set(tv, in.v.data(),     0, ggml_nbytes(tv));
    ggml_backend_tensor_set(tg, in.g.data(),     0, ggml_nbytes(tg));
    ggml_backend_tensor_set(tb, in.beta.data(),  0, ggml_nbytes(tb));
    ggml_backend_tensor_set(ts, in.state.data(), 0, ggml_nbytes(ts));

    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    res.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// the slot contents one layout must hold for token t of sequence s, head h
static bool check_slot(const float * slot, const gdn_case & c, const gdn_inputs & in, int32_t emit_mode, int64_t t, int64_t s, int64_t h) {
    const int64_t G    = c.kda ? c.S_v : 1;
    const int64_t tok  = (s * c.n_tokens + t) * c.H + h; // [.., H, n_tokens, n_seqs] flat index of (h, t, s)
    const float * k    = &in.k[tok * c.S_v];
    const float * v    = &in.v[tok * c.S_v];
    const float * g    = &in.g[tok * G];
    const float   beta = in.beta[tok];

    const int64_t off_b = emit_mode == 1 ? 3 * c.S_v : 2 * c.S_v + G;
    const int64_t n_g   = emit_mode == 1 ? c.S_v : G;
    const int64_t n_b   = emit_mode == 1 ? c.S_v : 1;
    for (int64_t i = 0; i < c.S_v; ++i) {
        if (slot[i] != k[i] || slot[c.S_v + i] != v[i]) {
            return false;
        }
    }
    for (int64_t i = 0; i < n_g; ++i) {
        if (slot[2 * c.S_v + i] != (c.kda ? g[i] : g[0])) {
            return false;
        }
    }
    for (int64_t i = 0; i < n_b; ++i) {
        if (slot[off_b + i] != beta) {
            return false;
        }
    }
    return true;
}

static bool run_case(ggml_backend_t backend, const gdn_case & c, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int64_t G = c.kda ? c.S_v : 1;
    const int64_t n_tok = c.H * c.n_tokens * c.n_seqs;

    gdn_inputs in;
    in.q.resize(c.S_v * n_tok);
    in.k.resize(c.S_v * n_tok);
    in.v.resize(c.S_v * n_tok);
    in.g.resize(G * n_tok);
    in.beta.resize(n_tok);
    in.state.resize(c.S_v * c.S_v * c.H * c.n_seqs);
    for (auto & x : in.q)     x = u(rng);
    for (auto & x : in.k)     x = 0.25f * u(rng);
    for (auto & x : in.v)     x = u(rng);
    for (auto & x : in.g)     x = -2.5f + 2.5f * u(rng); // g < 0 like the model's log-decay
    for (auto & x : in.beta)  x = 0.5f + 0.5f * u(rng);
    for (auto & x : in.state) x = 0.1f * u(rng);

    const std::vector<float> o1 = run_gdn(backend, c, in, 1);
    const std::vector<float> o2 = run_gdn(backend, c, in, 2);
    if (o1.empty() || o2.empty()) {
        printf("  %s: GATED_DELTA_NET emit_mode 1/2 unsupported, skipping\n", ggml_backend_name(backend));
        return true;
    }

    const int64_t attn   = c.S_v * c.H * c.n_tokens * c.n_seqs;
    const int64_t state  = c.S_v * c.S_v * c.H * c.n_seqs;
    const int64_t r1     = ggml_gated_delta_net_ingr_region(c.S_v, G, c.H, c.n_seqs, c.K, 1);
    const int64_t r2     = ggml_gated_delta_net_ingr_region(c.S_v, G, c.H, c.n_seqs, c.K, 2);
    const bool    ckpt   = c.n_tokens > c.K;
    const int64_t tail   = state * (ckpt ? 2 : 1);

    // padding to whole rows can eat the saving on tiny shapes (KDA, S_v 16, H 3, K 1), never exceed it
    bool ok = (int64_t) o1.size() == attn + r1 + tail && (int64_t) o2.size() == attn + r2 + tail && r2 <= r1;
    ok = ok && std::memcmp(o1.data(), o2.data(), attn * sizeof(float)) == 0;
    ok = ok && std::memcmp(o1.data() + attn + r1, o2.data() + attn + r2, tail * sizeof(float)) == 0;

    // right-aligned, chronological slots: slot K - n_new + j holds token n_tokens - n_new + j
    const int64_t n_new = c.n_tokens < c.K ? c.n_tokens : c.K;
    for (int32_t mode : { 1, 2 }) {
        const std::vector<float> & o = mode == 1 ? o1 : o2;
        const int64_t w = ggml_gated_delta_net_ingr_width(c.S_v, G, mode);
        for (int64_t j = 0; ok && j < n_new; ++j) {
            const int64_t slot = c.K - n_new + j;
            const int64_t t    = c.n_tokens - n_new + j;
            for (int64_t s = 0; ok && s < c.n_seqs; ++s) {
                for (int64_t h = 0; ok && h < c.H; ++h) {
                    const float * p = o.data() + attn + slot * w * c.H * c.n_seqs + (s * c.H + h) * w;
                    ok = check_slot(p, c, in, mode, t, s, h);
                }
            }
        }
    }

    if (!ok) {
        printf("  FAIL %s S_v=%lld H=%lld n_tokens=%lld n_seqs=%lld K=%lld kda=%d\n", ggml_backend_name(backend),
               (long long) c.S_v, (long long) c.H, (long long) c.n_tokens, (long long) c.n_seqs, (long long) c.K, c.kda);
    }
    return ok;
}

int main() {
    ggml_backend_load_all();

    std::vector<ggml_backend_t> backends;
    backends.push_back(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        backends.push_back(ggml_backend_dev_init(dev, nullptr));
    }

    std::mt19937 rng(63);
    bool ok = true;
    for (ggml_backend_t backend : backends) {
        GGML_ASSERT(backend);
        int n_cases = 0;
        for (int64_t S_v : { 16, 128 }) {
            for (int64_t H : { 3, 4 }) {
                for (int64_t n_seqs : { 1, 2 }) {
                    for (bool kda : { false, true }) {
                        // n_tokens < K (partial window), == K, > K (before-the-window block), and decode
                        static const int64_t shapes[][2] = { { 3, 5 }, { 5, 5 }, { 6, 4 }, { 1, 5 }, { 9, 1 } };
                        for (const auto & sh : shapes) {
                            ok = run_case(backend, { S_v, H, sh[0], n_seqs, sh[1], kda }, rng) && ok;
                            n_cases++;
                        }
                    }
                }
            }
        }
        printf("%s: %d cases\n", ggml_backend_name(backend), n_cases);
        ggml_backend_free(backend);
    }

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
