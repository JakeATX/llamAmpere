// Peer expert tier transport check (GGML_PEER_EXPERTS, default OFF).
//
// The "main" card posts activation rows and routed expert ids into a pinned, mapped host mailbox; the "peer" card
// waits on the doorbell, computes the expert rows with the stock ggml CUDA MUL_MAT_ID kernels on its own copy of
// the expert weights, and posts the rows back; the main card waits and reads them. The rows must be bit-identical to
// MUL_MAT_ID computed directly on the main card. Also checks the bounded wait (timeout) and the abort word, and
// reports round-trip latency.
//
//   test-peer-experts [--main i] [--peer j] [--rounds n] [--type q4_K|q6_K|q8_0|f16] [--no-thread]
// Both cards may be the same device (two backend instances = two streams): no second card is needed.
// Exit 0 = pass (or no CUDA backend: skipped), 1 = fail.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct peer_mbox;
typedef peer_mbox * (*mbox_new_t)(int, size_t, size_t);
typedef void (*mbox_free_t)(peer_mbox *);
typedef bool (*post_t)(ggml_backend_t, peer_mbox *, int, int, const ggml_tensor *, uint32_t);
typedef bool (*wait_t)(ggml_backend_t, peer_mbox *, int, int, ggml_tensor *, uint32_t, uint64_t);
typedef uint32_t (*status_t)(const peer_mbox *, int, int);
typedef void (*abort_t)(peer_mbox *, uint32_t);

static mbox_new_t  p_new;
static mbox_free_t p_free;
static post_t      p_post;
static wait_t      p_wait;
static status_t    p_status;
static abort_t     p_abort;

static constexpr int64_t K        = 2048; // n_embd
static constexpr int64_t N        = 768;  // expert rows (n_ff)
static constexpr int64_t N_EXPERT = 16;
static constexpr int64_t N_USED   = 4;
static constexpr int64_t T_MAX    = 4;    // tokens per step (decode / short verify window)
static constexpr uint64_t WAIT_NS = 2000ull * 1000 * 1000; // 2 s: far above any real round trip

// one card's side: weights + per-token-count graphs, all tensors in one backend buffer
struct side {
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * w = nullptr;
    ggml_tensor * x[T_MAX + 1] = {};
    ggml_tensor * ids[T_MAX + 1] = {};
    ggml_tensor * y[T_MAX + 1] = {};
    ggml_tensor * yback[T_MAX + 1] = {}; // main only: rows read back from the mailbox
    ggml_cgraph * gf[T_MAX + 1] = {};

    bool init(ggml_backend_t be, ggml_type type) {
        backend = be;
        ggml_init_params ip = { 64 * ggml_tensor_overhead() + 8 * ggml_graph_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        w = ggml_new_tensor_3d(ctx, type, K, N, N_EXPERT);
        for (int t = 1; t <= T_MAX; ++t) {
            x[t]     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, t);
            ids[t]   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, t);
            y[t]     = ggml_mul_mat_id(ctx, w, x[t], ids[t]); // [N, N_USED, t]
            yback[t] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N, N_USED, t);
            gf[t]    = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf[t], y[t]);
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        return buf != nullptr;
    }
    void free_all() {
        if (buf) { ggml_backend_buffer_free(buf); }
        if (ctx) { ggml_free(ctx); }
        if (backend) { ggml_backend_free(backend); }
    }
};

int main(int argc, char ** argv) {
    int main_dev = 0, peer_dev = 0, rounds = 64;
    bool use_thread = true;
    ggml_type type = GGML_TYPE_Q4_K;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--main" && i + 1 < argc) { main_dev = atoi(argv[++i]); }
        else if (a == "--peer" && i + 1 < argc) { peer_dev = atoi(argv[++i]); }
        else if (a == "--rounds" && i + 1 < argc) { rounds = atoi(argv[++i]); }
        else if (a == "--no-thread") { use_thread = false; }
        else if (a == "--type" && i + 1 < argc) {
            std::string t = argv[++i];
            type = t == "q6_K" ? GGML_TYPE_Q6_K : t == "q8_0" ? GGML_TYPE_Q8_0 : t == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q4_K;
        } else {
            fprintf(stderr, "usage: %s [--main i] [--peer j] [--rounds n] [--type q4_K|q6_K|q8_0|f16] [--no-thread]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (!reg) {
        printf("SKIP: no CUDA backend\n");
        return 0;
    }
    p_new    = (mbox_new_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_mbox_new");
    p_free   = (mbox_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_mbox_free");
    p_post   = (post_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_post");
    p_wait   = (wait_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_wait");
    p_status = (status_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_status");
    p_abort  = (abort_t)     ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_peer_abort");
    if (!p_new || !p_free || !p_post || !p_wait || !p_status || !p_abort) {
        printf("FAIL: peer transport entry points missing\n");
        return 1;
    }
    const int n_dev = (int) ggml_backend_reg_dev_count(reg);
    if (main_dev >= n_dev || peer_dev >= n_dev) {
        printf("SKIP: %d CUDA devices, asked for %d/%d\n", n_dev, main_dev, peer_dev);
        return 0;
    }

    side m, p;
    if (!m.init(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, main_dev), nullptr), type) ||
            !p.init(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, peer_dev), nullptr), type)) {
        printf("FAIL: backend init / alloc\n");
        return 1;
    }
    printf("main CUDA%d, peer CUDA%d, type %s, K=%" PRId64 " N=%" PRId64 " experts=%" PRId64 " used=%" PRId64 ", %s\n",
            main_dev, peer_dev, ggml_type_name(type), K, N, N_EXPERT, N_USED, use_thread ? "peer thread" : "one thread");

    // same expert weight bytes on both cards
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    {
        std::vector<float> wf(K * N * N_EXPERT);
        for (auto & v : wf) { v = nd(rng); }
        std::vector<uint8_t> wq(ggml_nbytes(m.w));
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, N * N_EXPERT, K, nullptr);
        ggml_backend_tensor_set(m.w, wq.data(), 0, wq.size());
        ggml_backend_tensor_set(p.w, wq.data(), 0, wq.size());
    }

    const size_t bytes_in  = sizeof(float) * K * T_MAX;
    const size_t bytes_out = sizeof(float) * N * N_USED * T_MAX;
    peer_mbox * mb_x   = p_new(2, bytes_in, bytes_out);
    peer_mbox * mb_ids = p_new(2, sizeof(int32_t) * N_USED * T_MAX, 64);
    if (!mb_x || !mb_ids) {
        printf("FAIL: mailbox alloc\n");
        return 1;
    }

    // per round: token count and inputs, fixed up front so the peer thread needs no host data
    std::vector<int> n_tok(rounds);
    std::vector<std::vector<float>> xs(rounds);
    std::vector<std::vector<int32_t>> idss(rounds);
    for (int r = 0; r < rounds; ++r) {
        n_tok[r] = 1 + r % T_MAX;
        xs[r].resize(K * n_tok[r]);
        for (auto & v : xs[r]) { v = nd(rng) * 20.0f; }
        idss[r].resize(N_USED * n_tok[r]);
        for (int t = 0; t < n_tok[r]; ++t) {
            std::vector<int32_t> perm(N_EXPERT);
            for (int e = 0; e < N_EXPERT; ++e) { perm[e] = e; }
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int u = 0; u < N_USED; ++u) { idss[r][t * N_USED + u] = perm[u]; }
        }
    }

    // warm-up: load every kernel on both streams before the timed protocol, so a first-launch stall cannot
    // eat a wait's timeout (a real caller sees that as one fallback step, the test would count it as a fail)
    {
        const uint32_t seq = 0x7fff0000u;
        bool ok = p_post(m.backend, mb_x, 0, 0, m.x[1], seq) && p_post(m.backend, mb_ids, 0, 0, m.ids[1], seq) &&
                  p_wait(p.backend, mb_x, 0, 0, p.x[1], seq, WAIT_NS) && p_wait(p.backend, mb_ids, 0, 0, p.ids[1], seq, WAIT_NS) &&
                  p_post(p.backend, mb_x, 0, 1, p.yback[1], seq) && p_wait(m.backend, mb_x, 0, 1, m.yback[1], seq, WAIT_NS);
        for (int t = 1; t <= T_MAX; ++t) {
            ok = ok && ggml_backend_graph_compute(m.backend, m.gf[t]) == GGML_STATUS_SUCCESS &&
                       ggml_backend_graph_compute(p.backend, p.gf[t]) == GGML_STATUS_SUCCESS;
        }
        ggml_backend_synchronize(m.backend);
        ggml_backend_synchronize(p.backend);
        if (!ok || p_status(mb_x, 0, 1) == seq) {
            printf("FAIL: warm-up round trip\n");
            return 1;
        }
        // doorbells are compared as (bell - seq) >= 0: start the rounds from a clean mailbox
        p_free(mb_x);
        p_free(mb_ids);
        mb_x   = p_new(2, bytes_in, bytes_out);
        mb_ids = p_new(2, sizeof(int32_t) * N_USED * T_MAX, 64);
        if (!mb_x || !mb_ids) {
            printf("FAIL: mailbox alloc\n");
            return 1;
        }
    }

    std::atomic<bool> peer_ok{true};
    std::atomic<int> peer_in_fail{0};
    auto peer_round = [&](int r) {
        const int t = n_tok[r], slot = r & 1;
        const uint32_t seq = (uint32_t) r + 1;
        // wait for the rows and ids, compute with the stock kernels, post the rows back (all stream-ordered)
        bool ok = p_wait(p.backend, mb_x, slot, 0, p.x[t], seq, WAIT_NS) &&
                  p_wait(p.backend, mb_ids, slot, 0, p.ids[t], seq, WAIT_NS);
        ok = ok && ggml_backend_graph_compute_async(p.backend, p.gf[t]) == GGML_STATUS_SUCCESS;
        ok = ok && p_post(p.backend, mb_x, slot, 1, p.y[t], seq);
        if (!ok) { peer_ok = false; }
    };

    int n_fail = 0;
    std::thread peer_thr;
    if (use_thread) {
        peer_thr = std::thread([&] {
            for (int r = 0; r < rounds; ++r) {
                peer_round(r);
                // keep the peer at most one round ahead of its own stream (the data areas alternate by slot)
                ggml_backend_synchronize(p.backend);
                const uint32_t seq = (uint32_t) r + 1;
                if (p_status(mb_x, r & 1, 0) == seq || p_status(mb_ids, r & 1, 0) == seq) {
                    peer_in_fail++;
                }
            }
        });
    }

    double us_sum = 0;
    for (int r = 0; r < rounds; ++r) {
        const int t = n_tok[r], slot = r & 1;
        const uint32_t seq = (uint32_t) r + 1;
        ggml_backend_tensor_set(m.x[t], xs[r].data(), 0, ggml_nbytes(m.x[t]));
        ggml_backend_tensor_set(m.ids[t], idss[r].data(), 0, ggml_nbytes(m.ids[t]));
        ggml_backend_synchronize(m.backend);
        const auto t0 = std::chrono::steady_clock::now();
        bool ok = p_post(m.backend, mb_x, slot, 0, m.x[t], seq) && p_post(m.backend, mb_ids, slot, 0, m.ids[t], seq);
        // the main card's own experts run here while the peer computes the rest (the reference path below)
        ok = ok && ggml_backend_graph_compute_async(m.backend, m.gf[t]) == GGML_STATUS_SUCCESS;
        if (!use_thread) {
            peer_round(r);
        }
        ok = ok && p_wait(m.backend, mb_x, slot, 1, m.yback[t], seq, WAIT_NS);
        ggml_backend_synchronize(m.backend);
        us_sum += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        if (!ok || p_status(mb_x, slot, 1) == seq) {
            printf("FAIL: round %d: transport fallback (ok=%d status=%u)\n", r, ok, p_status(mb_x, slot, 1));
            n_fail++;
            continue;
        }
        std::vector<float> a(ggml_nelements(m.y[t])), b(a.size());
        ggml_backend_tensor_get(m.y[t], a.data(), 0, ggml_nbytes(m.y[t]));
        ggml_backend_tensor_get(m.yback[t], b.data(), 0, ggml_nbytes(m.yback[t]));
        if (memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
            size_t i = 0;
            while (i < a.size() && memcmp(&a[i], &b[i], sizeof(float)) == 0) { ++i; }
            printf("FAIL: round %d (T=%d): peer rows differ at %zu: %.9g vs %.9g\n", r, t, i, a[i], b[i]);
            n_fail++;
        }
    }
    if (peer_thr.joinable()) {
        peer_thr.join();
    }
    if (peer_in_fail > 0) {
        printf("FAIL: %d peer input waits timed out\n", peer_in_fail.load());
        n_fail++;
    }
    if (!peer_ok) {
        printf("FAIL: peer side launch\n");
        n_fail++;
    }
    printf("identity: %d rounds, %d fail; mean main round trip incl. own compute %.1f us\n", rounds, n_fail, us_sum / rounds);

    // bounded wait: nothing posted for this seq, so the wait must time out, write zeros and flag the status
    {
        const uint32_t seq = 1000000;
        std::vector<float> ones(ggml_nelements(m.yback[1]), 1.0f);
        ggml_backend_tensor_set(m.yback[1], ones.data(), 0, ggml_nbytes(m.yback[1]));
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = p_wait(m.backend, mb_x, 0, 1, m.yback[1], seq, 5ull * 1000 * 1000);
        ggml_backend_synchronize(m.backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::vector<float> got(ones.size());
        ggml_backend_tensor_get(m.yback[1], got.data(), 0, ggml_nbytes(m.yback[1]));
        bool zero = true;
        for (float v : got) { zero = zero && v == 0.0f; }
        const bool pass = ok && p_status(mb_x, 0, 1) == seq && zero && ms < 1000.0;
        printf("timeout: %s (%.1f ms, status %u, zeros %d)\n", pass ? "ok" : "FAIL", ms, p_status(mb_x, 0, 1), zero);
        n_fail += pass ? 0 : 1;
    }
    // abort word: a long wait is released by the host
    {
        const uint32_t seq = 2000000;
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = p_wait(m.backend, mb_x, 1, 1, m.yback[1], seq, 30ull * 1000 * 1000 * 1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        p_abort(mb_x, 1);
        ggml_backend_synchronize(m.backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        p_abort(mb_x, 0);
        const bool pass = ok && p_status(mb_x, 1, 1) == seq && ms < 5000.0;
        printf("abort: %s (%.1f ms, status %u)\n", pass ? "ok" : "FAIL", ms, p_status(mb_x, 1, 1));
        n_fail += pass ? 0 : 1;
    }
    // transport-only round trip at T=1 (main -> peer -> main, no compute): the latency a peer hit adds
    {
        const int n = 200;
        double us = 0;
        for (int i = 0; i < n; ++i) {
            const uint32_t seq = 3000000 + i;
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = p_post(m.backend, mb_x, 0, 0, m.x[1], seq);
            ok = ok && p_wait(p.backend, mb_x, 0, 0, p.x[1], seq, WAIT_NS) && p_post(p.backend, mb_x, 0, 1, p.yback[1], seq);
            ok = ok && p_wait(m.backend, mb_x, 0, 1, m.yback[1], seq, WAIT_NS);
            ggml_backend_synchronize(m.backend);
            us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            if (!ok) { n_fail++; break; }
        }
        printf("transport round trip T=1 (%" PRId64 " floats in, %" PRId64 " out): %.1f us mean over %d\n", K, N * N_USED, us / n, n);
    }

    p_free(mb_x);
    p_free(mb_ids);
    m.free_all();
    p.free_all();
    printf("%s\n", n_fail == 0 ? "PASS" : "FAIL");
    return n_fail == 0 ? 0 : 1;
}
