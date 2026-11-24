// fork (ONEGRAPH, backlog #101): see llama-moe-onegraph.h
#include "llama-moe-onegraph.h"

#include "llama-impl.h"

#include "ggml-cpp.h"

#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cstring>

static int64_t og_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int llama_moe_og::mode() {
    static const int m = [] { const char * e = getenv("LLAMA_MOE_ONEGRAPH"); return e ? atoi(e) : 0; }();
    return m == 1 || m == 2 ? m : 0;
}

bool llama_moe_og::requested() {
    return mode() != 0;
}

llama_moe_og * llama_moe_og::create(ggml_backend_t main, ggml_backend_buffer_type_t buft, int32_t n_expert, int32_t n_rows,
                                    const std::vector<std::pair<void *, size_t>> & pins, serve_fn serve, snapshot_fn snapshot) {
    ggml_backend_dev_t dev = ggml_backend_get_device(main);
    ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (!host_buft) {
        LLAMA_LOG_WARN("%s: [ONEGRAPH] off: the device has no pinned host buffer type\n", __func__);
        return nullptr;
    }
    std::unique_ptr<llama_moe_og> og(new llama_moe_og());
    og->main_backend = main;
    og->n_expert = n_expert;
    og->n_rows   = n_rows;
    og->serve    = std::move(serve);
    og->snapshot = std::move(snapshot);
    if (const char * t = getenv("LLAMA_MOE_ONEGRAPH_TIMEOUT_MS"); t && atoi(t) > 0) {
        og->timeout_ms = atoi(t);
    }

    // pinned ring: the CUDA host buffer type maps it into the device address space (same pointer under UVA); a
    // fallback to plain host memory (GGML_CUDA_NO_PINNED) is refused
    og->buf_ring = ggml_backend_buft_alloc_buffer(host_buft, sizeof(ggml_moe_og_ring));
    if (!og->buf_ring || ggml_backend_buffer_get_type(og->buf_ring) != host_buft) {
        LLAMA_LOG_WARN("%s: [ONEGRAPH] off: no pinned host memory for the ring\n", __func__);
        return nullptr;
    }
    og->ring = (ggml_moe_og_ring *) ggml_backend_buffer_get_base(og->buf_ring);
    memset(og->ring, 0, sizeof(ggml_moe_og_ring));

    ggml_init_params params = { 2*ggml_tensor_overhead(), nullptr, true };
    og->ctx_dev = ggml_init(params);
    og->map  = ggml_new_tensor_2d(og->ctx_dev, GGML_TYPE_I32, n_expert, n_rows);
    og->ctrl = ggml_new_tensor_1d(og->ctx_dev, GGML_TYPE_I32, GGML_MOE_OG_CTRL_BYTES / 4);
    ggml_set_name(og->map,  "moe_og.map");
    ggml_set_name(og->ctrl, "moe_og.ctrl");
    og->buf_dev = ggml_backend_alloc_ctx_tensors_from_buft(og->ctx_dev, buft);
    if (!og->buf_dev) {
        LLAMA_LOG_WARN("%s: [ONEGRAPH] off: no device memory for the slot table\n", __func__);
        return nullptr;
    }
    ggml_backend_buffer_clear(og->buf_dev, 0);
    ggml_backend_buffer_set_usage(og->buf_dev, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // the GPU copies the missing experts from the host experts: page-lock them read-only and mapped
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto pin = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_og_pin");
    og->unpin = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_moe_og_unpin");
    if (!pin || !og->unpin) {
        LLAMA_LOG_WARN("%s: [ONEGRAPH] off: the backend cannot register host memory\n", __func__);
        return nullptr;
    }
    size_t pinned_bytes = 0;
    for (const auto & p : pins) {
        if (!pin(p.first, p.second)) {
            LLAMA_LOG_WARN("%s: [ONEGRAPH] off: cannot page-lock the host experts (%.1f MiB at %p; a read-only file "
                    "mapping needs a driver with read-only registration, else use --load-mode none)\n", __func__,
                    p.second/1048576.0, p.first);
            return nullptr; // the destructor unpins the ranges pinned so far
        }
        og->pinned.push_back(p.first);
        pinned_bytes += p.second;
    }
    og->table.assign((size_t) n_expert*n_rows, 0);
    og->table_new.assign((size_t) n_expert*n_rows, 0);
    og->worker = std::thread([p = og.get()] { p->run(); });
    if (mode() == 2) {
        LLAMA_LOG_WARN("%s: [ONEGRAPH] pin only (LLAMA_MOE_ONEGRAPH=2, diagnostic): host experts page-locked, regular lookups\n", __func__);
    }
    LLAMA_LOG_WARN("%s: [ONEGRAPH] on: decode lookups (<= 8 rows) on the GPU, misses copied by the GPU on the host worker's answer "
            "(%d rows x %d experts, timeout %d ms, %.0f MiB host experts page-locked)\n", __func__, n_rows, n_expert, og->timeout_ms,
            pinned_bytes/1048576.0);
    return og.release();
}

llama_moe_og::~llama_moe_og() {
    if (worker.joinable()) {
        stop = true;
        cv.notify_all();
        worker.join();
    }
    if (main_backend) {
        ggml_backend_synchronize(main_backend);
    }
    if (n_records.load() > 0) {
        LLAMA_LOG_INFO("llama_moe_og: [ONEGRAPH] %llu lookups served, %llu released a waiting GPU, %llu reader lags\n",
                (unsigned long long) n_records.load(), (unsigned long long) n_waits.load(), (unsigned long long) n_lag.load());
    }
    for (void * p : pinned) {
        unpin(p);
    }
    if (buf_dev)  { ggml_backend_buffer_free(buf_dev); }
    if (buf_ring) { ggml_backend_buffer_free(buf_ring); }
    if (ctx_dev)  { ggml_free(ctx_dev); }
}

ggml_tensor * llama_moe_og::slots(ggml_context * ctx, ggml_tensor * ids, int32_t row, int32_t layer, int32_t n_slots, int32_t n_used) const {
    return ggml_moe_og_slots(ctx, ids, map, ctrl, ring, row, layer, n_slots, n_used, timeout_ms);
}

void llama_moe_og::prepare() {
    const uint64_t err = __atomic_load_n(&ring->error, __ATOMIC_ACQUIRE);
    if (err != 0) {
        GGML_ABORT("[ONEGRAPH] lookup %llu was not served in time (or broke the protocol): results of that step are invalid",
                (unsigned long long) err);
    }
    if (dirty.exchange(false)) {
        // no lookup of this context runs now (the maps changed on the regular path or at a resize/resume, which
        // quiesce first)
        std::lock_guard<std::mutex> lock(mu);
        snapshot(table.data());
        ggml_backend_tensor_set(map, table.data(), 0, ggml_nbytes(map));
    }
    active_until_us = og_now_us() + 20000;
    cv.notify_one();
}

void llama_moe_og::quiesce() {
    ggml_backend_synchronize(main_backend);
    const int64_t t0 = og_now_us();
    while (processed.load(std::memory_order_acquire) < __atomic_load_n(&ring->last, __ATOMIC_ACQUIRE)) {
        active_until_us = og_now_us() + 20000;
        cv.notify_one();
        if (og_now_us() - t0 > (int64_t) timeout_ms*1000) {
            GGML_ABORT("[ONEGRAPH] host worker stalled");
        }
        std::this_thread::yield();
    }
}

void llama_moe_og::run() {
    uint64_t next = 1;
    ggml_moe_og_rec rec;
    while (!stop.load(std::memory_order_relaxed)) {
        const uint64_t s = ggml_moe_og_poll(ring, next, &rec);
        if (s != 0) {
            if (s != next) {
                n_lag++;
            }
            ggml_moe_og_rec * out = &ring->rec[s % GGML_MOE_OG_RING];
            int changed = 0;
            bool bad = false;
            {
                std::lock_guard<std::mutex> lock(mu);
                copies.clear();
                changed = serve(rec, copies);
                upd.clear();
                if (changed) {
                    snapshot(table_new.data());
                    for (size_t i = 0; i < table.size(); ++i) {
                        if (table_new[i] != table[i]) {
                            upd.emplace_back((int32_t) i, table_new[i]);
                        }
                    }
                    table.swap(table_new);
                }
                bad = changed && !rec.need; // the GPU read the old table and went on: the two copies of the maps disagree
                bad = bad || copies.size() > GGML_MOE_OG_MAX_COPY || upd.size() > GGML_MOE_OG_MAX_UPD;
            }
            if (bad) {
                uint64_t zero = 0;
                __atomic_compare_exchange_n(&ring->error, &zero, s, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
            }
            if (rec.need) {
                const size_t nc = std::min(copies.size(), (size_t) GGML_MOE_OG_MAX_COPY);
                const size_t nu = std::min(upd.size(),    (size_t) GGML_MOE_OG_MAX_UPD);
                for (size_t i = 0; i < nc; ++i) {
                    out->copy[i] = copies[i];
                }
                for (size_t i = 0; i < nu; ++i) {
                    out->upd[i][0] = upd[i].first;
                    out->upd[i][1] = upd[i].second;
                }
                out->n_copy = (int32_t) nc;
                out->n_upd  = (int32_t) nu;
                ggml_moe_og_answer(out, s);
                n_waits++;
            }
            if (++n_records % 65536 == 0) {
                LLAMA_LOG_WARN("llama_moe_og: [ONEGRAPH] %llu lookups served, %llu waited for a miss, %llu reader lags\n",
                        (unsigned long long) n_records.load(), (unsigned long long) n_waits.load(), (unsigned long long) n_lag.load());
            }
            processed.store(s, std::memory_order_release);
            next = s + 1;
            active_until_us = og_now_us() + 20000;
            continue;
        }
        if (og_now_us() < active_until_us.load(std::memory_order_relaxed)) {
            std::this_thread::yield();
            continue;
        }
        std::unique_lock<std::mutex> lk(cv_mu);
        cv.wait_for(lk, std::chrono::milliseconds(1));
    }
}
