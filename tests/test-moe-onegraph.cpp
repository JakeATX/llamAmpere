// fork (ONEGRAPH, backlog #101): the doorbell handshake of ggml_moe_og_slots (ggml/include/ggml-moe-onegraph.h)
// against a host worker thread, on the first GPU. Not a ctest (needs a GPU); exits 0 with SKIP if there is none.
//   1. all hits, no worker: the kernel publishes a record per call and does not wait
//   2. misses, worker thread: the worker answers with copies (pinned host -> device bank) and slot map changes, which
//      the GPU carries out; many rounds of the same graph (so the backend may replay it as a captured CUDA graph)
//   3. a miss and no worker: the kernel times out, sets ring->error and writes in-range slots
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-moe-onegraph.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

static constexpr int N_EXPERT = 128;
static constexpr int N_ROWS   = 4;
static constexpr int N_SLOTS  = 80;  // >= 64: a lookup never reuses a slot
static constexpr int N_USED   = 8;
static constexpr int N_TOK    = 8;   // 64 ids: the widest record
static constexpr int EMPTY    = N_SLOTS;

static int slot_of(int row, int e) { return (e*7 + row*3) % N_SLOTS; }

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        printf("SKIP: no GPU\n");
        return 0;
    }
    ggml_backend_t main_b = ggml_backend_dev_init(dev, nullptr);
    CHECK(main_b);
    ggml_backend_buffer_type_t buft      = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    CHECK(host_buft);

    ggml_backend_buffer_t buf_ring = ggml_backend_buft_alloc_buffer(host_buft, sizeof(ggml_moe_og_ring));
    CHECK(buf_ring && ggml_backend_buffer_get_type(buf_ring) == host_buft);
    auto * ring = (ggml_moe_og_ring *) ggml_backend_buffer_get_base(buf_ring);
    memset(ring, 0, sizeof(*ring));

    // persistent tensors: map, ctrl, ids
    ggml_init_params pp = { 8*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx_w = ggml_init(pp);
    ggml_tensor * map  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_I32, N_EXPERT, N_ROWS);
    ggml_tensor * ctrl = ggml_new_tensor_1d(ctx_w, GGML_TYPE_I32, GGML_MOE_OG_CTRL_BYTES/4);
    ggml_tensor * ids  = ggml_new_tensor_1d(ctx_w, GGML_TYPE_I32, N_USED*N_TOK);
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, buft);
    CHECK(buf_w);
    ggml_backend_buffer_clear(buf_w, 0);

    auto build = [&](ggml_context ** pctx, int row, int timeout_ms) {
        ggml_init_params gp = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        *pctx = ggml_init(gp);
        ggml_tensor * out = ggml_moe_og_slots(*pctx, ids, map, ctrl, ring, row, 100 + row, N_SLOTS, N_USED, timeout_ms);
        ggml_cgraph * gf = ggml_new_graph(*pctx);
        ggml_build_forward_expand(gf, out);
        return std::make_pair(gf, out);
    };

    std::vector<int32_t> table((size_t) N_EXPERT*N_ROWS), host_ids(N_USED*N_TOK), got(N_USED*N_TOK);
    auto make_ids = [&](int round) {
        for (int t = 0; t < N_TOK; ++t) {
            for (int k = 0; k < N_USED; ++k) {
                host_ids[t*N_USED + k] = (round*13 + t*29 + k*17) % N_EXPERT;
            }
        }
    };

    // 1. all hits, no worker
    {
        printf("1: start\n");
        for (int r = 0; r < N_ROWS; ++r) {
            for (int e = 0; e < N_EXPERT; ++e) {
                table[(size_t) r*N_EXPERT + e] = slot_of(r, e);
            }
        }
        ggml_backend_tensor_set(map, table.data(), 0, ggml_nbytes(map));
        ggml_context * gctx; auto [gf, out] = build(&gctx, 1, 2000);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        CHECK(ggml_gallocr_alloc_graph(ga, gf));
        for (int round = 0; round < 50; ++round) {
            make_ids(round);
            ggml_backend_tensor_set(ids, host_ids.data(), 0, ggml_nbytes(ids));
            CHECK(ggml_backend_graph_compute(main_b, gf) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            for (int i = 0; i < N_USED*N_TOK; ++i) {
                CHECK(got[i] == slot_of(1, host_ids[i]));
            }
            ggml_moe_og_rec rec;
            const uint64_t s = ggml_moe_og_poll(ring, round + 1, &rec);
            CHECK(s == (uint64_t) round + 1 && rec.need == 0 && rec.layer == 101 && rec.n_ids == N_USED*N_TOK && rec.n_used == N_USED);
            CHECK(memcmp(rec.ids, host_ids.data(), sizeof(int32_t)*N_USED*N_TOK) == 0);
        }
        CHECK(ring->error == 0 && ring->last == 50);
        ggml_gallocr_free(ga); ggml_free(gctx);
        printf("ok 1: 50 all-hit lookups, no wait\n");
    }

    // 2. misses with a worker thread: the worker answers with copies and map changes, the GPU carries them out
    {
        const int ROUNDS = 400;
        const size_t EB = 64*1024; // bytes per expert
        ggml_backend_buffer_t buf_host = ggml_backend_buft_alloc_buffer(host_buft, EB*N_EXPERT);
        CHECK(buf_host);
        uint8_t * hx = (uint8_t *) ggml_backend_buffer_get_base(buf_host);
        for (size_t i = 0; i < EB*N_EXPERT; ++i) { hx[i] = (uint8_t) (i/EB*37 + i*11); }
        ggml_init_params bp = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx_b = ggml_init(bp);
        ggml_tensor * bank = ggml_new_tensor_1d(ctx_b, GGML_TYPE_I8, (int64_t) (EB*N_SLOTS));
        ggml_backend_buffer_t buf_b = ggml_backend_alloc_ctx_tensors_from_buft(ctx_b, buft);
        CHECK(buf_b);

        std::atomic<bool> stop{false};
        std::atomic<uint64_t> served{ring->last}, waited{0}; // records before this phase count as served
        std::vector<int32_t> wtable(table);    // the worker's copy of the maps
        std::thread worker([&] {
            uint64_t next = ring->last + 1;
            ggml_moe_og_rec rec;
            while (!stop.load()) {
                const uint64_t s = ggml_moe_og_poll(ring, next, &rec);
                if (s == 0) { std::this_thread::yield(); continue; }
                const int row = rec.layer - 100;
                ggml_moe_og_rec * out = &ring->rec[s % GGML_MOE_OG_RING];
                int nc = 0, nu = 0, next_slot = 0;
                for (int i = 0; i < rec.n_ids; ++i) {
                    const int e = rec.ids[i];
                    int32_t & v = wtable[(size_t) row*N_EXPERT + e];
                    if (v >= N_SLOTS) {
                        v = next_slot++;
                        out->upd[nu][0] = row*N_EXPERT + e; out->upd[nu][1] = v; nu++;
                        out->copy[nc].src = (uint64_t) (uintptr_t) (hx + e*EB);
                        out->copy[nc].dst = (uint64_t) (uintptr_t) ((uint8_t *) bank->data + v*EB);
                        out->copy[nc].bytes = EB; nc++;
                    }
                }
                if (rec.need) {
                    std::this_thread::sleep_for(std::chrono::microseconds(20)); // planning
                    out->n_copy = nc; out->n_upd = nu;
                    ggml_moe_og_answer(out, s);
                    waited++;
                }
                served = s;
                next = s + 1;
            }
        });
        printf("2: start\n");
        ggml_context * gctx; auto [gf, out] = build(&gctx, 2, 2000);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        CHECK(ggml_gallocr_alloc_graph(ga, gf));
        uint64_t n_need = 0;
        std::vector<uint8_t> bank_h(EB*N_SLOTS);
        double ms = 0.0;
        for (int round = 0; round < ROUNDS; ++round) {
            // the test's "cache" forgets row 2 every round, on both copies
            while (served.load() < ring->last) { std::this_thread::yield(); }
            for (int e = 0; e < N_EXPERT; ++e) { wtable[(size_t) 2*N_EXPERT + e] = EMPTY; }
            ggml_backend_tensor_set(map, wtable.data(), 0, ggml_nbytes(map));
            make_ids(round);
            ggml_backend_tensor_set(ids, host_ids.data(), 0, ggml_nbytes(ids));
            const auto t0 = std::chrono::steady_clock::now();
            CHECK(ggml_backend_graph_compute(main_b, gf) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (ring->error != 0) {
                fprintf(stderr, "FAIL round %d: lookup %llu timed out (served %llu, last %llu)\n", round,
                        (unsigned long long) ring->error, (unsigned long long) served.load(), (unsigned long long) ring->last);
                stop = true; worker.join(); return 1;
            }
            if (round < 3) { printf("  round %d ok, last %llu\n", round, (unsigned long long) ring->last); }
            while (served.load() < ring->last) { std::this_thread::yield(); }
            std::vector<int32_t> dmap((size_t) N_EXPERT*N_ROWS);
            ggml_backend_tensor_get(map, dmap.data(), 0, ggml_nbytes(map));
            ggml_backend_tensor_get(bank, bank_h.data(), 0, bank_h.size());
            for (int i = 0; i < N_USED*N_TOK; ++i) {
                const int e = host_ids[i];
                const int want = wtable[(size_t) 2*N_EXPERT + e];
                bool ok = got[i] == want && dmap[(size_t) 2*N_EXPERT + e] == want && want < N_SLOTS;
                ok = ok && memcmp(bank_h.data() + (size_t) want*EB, hx + (size_t) e*EB, EB) == 0;
                if (!ok) {
                    fprintf(stderr, "FAIL round %d id %d: expert %d slot %d want %d (device map %d)\n", round, i, e, got[i], want,
                            dmap[(size_t) 2*N_EXPERT + e]);
                    stop = true; worker.join(); return 1;
                }
            }
            n_need += ring->rec[ring->last % GGML_MOE_OG_RING].need;
        }
        stop = true; worker.join();
        CHECK(ring->error == 0);
        CHECK(n_need == (uint64_t) ROUNDS);
        // keep the device map equal to `table` for phase 3
        ggml_backend_tensor_set(map, table.data(), 0, ggml_nbytes(map));
        ggml_gallocr_free(ga); ggml_free(gctx);
        ggml_backend_buffer_free(buf_b); ggml_free(ctx_b); ggml_backend_buffer_free(buf_host);
        printf("ok 2: %d lookups with misses, %llu answered by the worker, copies and map changes checked, %.3f ms per lookup "
               "(64 KiB per missing expert)\n", ROUNDS, (unsigned long long) waited.load(), ms/ROUNDS);
    }

    // 3. a miss and no worker: timeout path
    {
        for (int e = 0; e < N_EXPERT; ++e) { table[(size_t) 3*N_EXPERT + e] = EMPTY; }
        ggml_backend_tensor_set(map, table.data() + (size_t) 3*N_EXPERT, (size_t) 3*N_EXPERT*sizeof(int32_t), N_EXPERT*sizeof(int32_t));
        ggml_context * gctx; auto [gf, out] = build(&gctx, 3, 50);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        CHECK(ggml_gallocr_alloc_graph(ga, gf));
        make_ids(7);
        ggml_backend_tensor_set(ids, host_ids.data(), 0, ggml_nbytes(ids));
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(ggml_backend_graph_compute(main_b, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK(ring->error == ring->last && ring->error != 0);
        for (int i = 0; i < N_USED*N_TOK; ++i) {
            CHECK(got[i] >= 0 && got[i] < N_SLOTS);
        }
        CHECK(ms >= 45.0 && ms < 2000.0);
        ggml_gallocr_free(ga); ggml_free(gctx);
        printf("ok 3: timeout after %.1f ms, ring->error = %llu, slots in range\n", ms, (unsigned long long) ring->error);
    }

    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_ring);
    ggml_free(ctx_w);
    ggml_backend_free(main_b);
    printf("PASS test-moe-onegraph\n");
    return 0;
}
