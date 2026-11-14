// [TAG_GDN_REPLAY_LANES] ggml_gated_delta_net_set_prefix with one mask per seq (w [K, n_seqs]).
// One call with per-lane masks must give, for every lane s, the same attention output, final state and
// checkpoint block as a call where every lane uses lane s's mask as the shared 1-D mask (prefix stride 0).
// The ingredient slots do not depend on the mask and must match too. CPU by default, --cuda for the GPU.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t g_rng = 0x9e3779b97f4a7c15ULL;
static float rnd(float lo, float hi) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    const float u = (float) ((g_rng >> 33) & 0xffffff) / (float) 0x1000000;
    return lo + u * (hi - lo);
}

struct gdn_inputs {
    std::vector<float> q, k, v, g, beta, state, ring;
};

static std::vector<float> run_gdn(ggml_backend_t backend, const gdn_inputs & in, const std::vector<float> & w, bool per_lane,
        int64_t S_v, int64_t H, int64_t n, int64_t n_seqs, int64_t K, int32_t emit) {
    ggml_init_params iparams = { /*.mem_size=*/ 1024 * 1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * tq     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n, n_seqs);
    ggml_tensor * tk     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n, n_seqs);
    ggml_tensor * tv     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n, n_seqs);
    ggml_tensor * tg     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, n_seqs);
    ggml_tensor * tbeta  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n, n_seqs);
    ggml_tensor * tstate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H, n_seqs);
    ggml_tensor * tring  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) in.ring.size() / n_seqs, n_seqs);
    ggml_tensor * tw     = per_lane ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_seqs) : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);

    ggml_tensor * out = ggml_gated_delta_net(ctx, tq, tk, tv, tg, tbeta, tstate, K, emit);
    ggml_gated_delta_net_set_prefix(out, tring, tw, emit);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));

    auto set = [](ggml_tensor * t, const std::vector<float> & d) {
        GGML_ASSERT(d.size() * sizeof(float) == ggml_nbytes(t));
        ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
    };
    set(tq, in.q);
    set(tk, in.k);
    set(tv, in.v);
    set(tg, in.g);
    set(tbeta, in.beta);
    set(tstate, in.state);
    set(tring, in.ring);
    set(tw, w);

    std::vector<float> res;
    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        res.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

// 0 = bit-equal, else the max abs difference over [a0, a0 + len) vs [b0, b0 + len)
static float cmp(const std::vector<float> & a, const std::vector<float> & b, size_t off, size_t len) {
    float m = 0.0f;
    for (size_t i = off; i < off + len; ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            m = std::max(m, std::isfinite(a[i] - b[i]) ? fabsf(a[i] - b[i]) : INFINITY);
            m = std::max(m, 1e-30f);
        }
    }
    return m;
}

int main(int argc, char ** argv) {
    const bool cuda = argc == 2 && strcmp(argv[1], "--cuda") == 0;
    // CPU runs each (seq, head) on its own, so the lanes must match bit for bit; the GPU kernel too
    const float tol = 0.0f;

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(cuda ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        fprintf(stderr, "FAIL: could not init %s backend\n", cuda ? "GPU" : "CPU");
        return 1;
    }

    const int64_t S_v = 128;
    const int64_t H   = 4;
    const int64_t K   = 4;

    int n_fail = 0;
    int n_case = 0;
    for (int32_t emit : { 1, 2 }) {
        for (int64_t n : { K, K + 1 }) {
            for (const std::vector<int64_t> & acc : std::vector<std::vector<int64_t>>{ { 0, 3 }, { 4, 1, 0 }, { 1, 4, 2, 0 }, { 2, 2, 2, 2 } }) {
                const int64_t n_seqs = (int64_t) acc.size();
                const int64_t pw     = ggml_gated_delta_net_ingr_width(S_v, 1, emit);

                gdn_inputs in;
                in.q.resize(S_v * H * n * n_seqs);
                in.k.resize(S_v * H * n * n_seqs);
                in.v.resize(S_v * H * n * n_seqs);
                in.g.resize(H * n * n_seqs);
                in.beta.resize(H * n * n_seqs);
                in.state.resize(S_v * S_v * H * n_seqs);
                in.ring.resize(K * pw * H * n_seqs);
                for (auto & x : in.q)     x = rnd(-1.0f, 1.0f) / sqrtf((float) S_v);
                for (auto & x : in.k)     x = rnd(-1.0f, 1.0f) / sqrtf((float) S_v);
                for (auto & x : in.v)     x = rnd(-0.3f, 5.0f);
                for (auto & x : in.g)     x = rnd(-5.0f, -1e-4f);
                for (auto & x : in.beta)  x = rnd(0.0f, 1.0f);
                for (auto & x : in.state) x = rnd(-0.1f, 0.1f);
                // ring slots: k and v plausible, g < 0, beta in [0, 1] (emit 2: one g, one beta per head)
                for (int64_t s = 0; s < n_seqs; ++s) {
                    for (int64_t j = 0; j < K; ++j) {
                        for (int64_t h = 0; h < H; ++h) {
                            float * slot = in.ring.data() + s * K * pw * H + (j * H + h) * pw;
                            for (int64_t i = 0; i < pw; ++i) {
                                slot[i] = rnd(-1.0f, 1.0f) / sqrtf((float) S_v);
                            }
                            slot[2 * S_v] = rnd(-5.0f, -1e-4f);
                            slot[emit == 1 ? 3 * S_v : 2 * S_v + 1] = rnd(0.0f, 1.0f);
                        }
                    }
                }

                std::vector<float> w_lanes(K * n_seqs, 0.0f);
                for (int64_t s = 0; s < n_seqs; ++s) {
                    std::fill(w_lanes.begin() + s * K, w_lanes.begin() + s * K + acc[s], 1.0f);
                }
                const std::vector<float> out = run_gdn(backend, in, w_lanes, true, S_v, H, n, n_seqs, K, emit);

                const size_t attn   = S_v * H * n;           // per seq
                const size_t region = ggml_gated_delta_net_ingr_region(S_v, 1, H, n_seqs, K, emit);
                const size_t state  = S_v * S_v * H;         // per seq
                const size_t total  = attn * n_seqs + region + 2 * state * n_seqs;
                bool ok = out.size() == total;

                float worst = 0.0f;
                for (int64_t s = 0; ok && s < n_seqs; ++s) {
                    std::vector<float> w1(K, 0.0f);
                    std::fill(w1.begin(), w1.begin() + acc[s], 1.0f);
                    const std::vector<float> ref = run_gdn(backend, in, w1, false, S_v, H, n, n_seqs, K, emit);
                    ok = ref.size() == total;
                    if (!ok) {
                        break;
                    }
                    const size_t fin = attn * n_seqs + region;
                    worst = std::max(worst, cmp(out, ref, s * attn, attn));                              // attention
                    worst = std::max(worst, cmp(out, ref, attn * n_seqs, K * pw * H * n_seqs));           // ingredient slots (not the row padding)
                    worst = std::max(worst, cmp(out, ref, fin + s * state, state));                      // final state
                    worst = std::max(worst, cmp(out, ref, fin + state * n_seqs + s * state, state));     // checkpoint
                }
                ok = ok && worst <= tol;
                n_case++;
                n_fail += !ok;
                printf("  emit=%d n=%lld lanes=%lld accepted=", emit, (long long) n, (long long) n_seqs);
                for (int64_t a : acc) {
                    printf("%lld ", (long long) a);
                }
                printf(": %s (max diff %.3e)\n", ok ? "OK" : "FAIL", worst);
            }
        }
    }

    ggml_backend_free(backend);
    printf("%d/%d cases OK\n", n_case - n_fail, n_case);
    return n_fail == 0 ? 0 : 1;
}
