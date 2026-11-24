#include "llama-moe-prefetch.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>

int32_t llama_moe_prefetch::env_budget() {
    const char * env = getenv("LLAMA_MOE_CACHE_NEXT_PREFETCH");
    if (env == nullptr) {
        return 0;
    }
    return (int32_t) std::clamp(atol(env), 0l, 64l);
}

std::unique_ptr<llama_moe_prefetch> llama_moe_prefetch::from_env(int32_t n_layer) {
    const int32_t k = env_budget();
    return k > 0 ? std::make_unique<llama_moe_prefetch>(n_layer, k) : nullptr;
}

llama_moe_prefetch::llama_moe_prefetch(int32_t n_layer, int32_t budget) :
        n_budget(budget), last(n_layer), marked(n_layer) {
    worker = std::thread([this] { loop(); });
}

llama_moe_prefetch::~llama_moe_prefetch() {
    {
        std::unique_lock<std::mutex> lk(mu);
        stop = true;
    }
    cv_work.notify_all();
    worker.join();
}

void llama_moe_prefetch::record(int32_t il, const int32_t * ids, size_t n_ids) {
    std::vector<int32_t> & v = last[il];
    v.assign(ids, ids + n_ids);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

void llama_moe_prefetch::mark(int32_t il, const std::vector<int32_t> & experts) {
    marked[il] = experts;
}

void llama_moe_prefetch::account(int32_t il, const int32_t * ids, size_t n_ids) {
    std::vector<int32_t> & m = marked[il];
    if (m.empty()) {
        return;
    }
    scratch.assign(ids, ids + n_ids);
    std::sort(scratch.begin(), scratch.end());
    for (int32_t e : m) {
        n_used += std::binary_search(scratch.begin(), scratch.end(), e) ? 1 : 0;
    }
    m.clear();
}

void llama_moe_prefetch::submit(std::vector<job> && jobs) {
    if (jobs.empty()) {
        return;
    }
    {
        std::unique_lock<std::mutex> lk(mu);
        for (const job & j : jobs) {
            n_bytes += j.size;
        }
        queue.insert(queue.end(), jobs.begin(), jobs.end());
        n_rounds++;
    }
    jobs.clear();
    cv_work.notify_one();
}

void llama_moe_prefetch::wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv_done.wait(lk, [this] { return queue.empty() && !busy; });
}

void llama_moe_prefetch::loop() {
    std::vector<job> run;
    std::unique_lock<std::mutex> lk(mu);
    for (;;) {
        cv_work.wait(lk, [this] { return stop || !queue.empty(); });
        if (queue.empty()) {
            return; // stop
        }
        run.swap(queue);
        busy = true;
        lk.unlock();
        for (const job & j : run) {
            ggml_backend_tensor_set(j.dst, j.src, j.offset, j.size);
        }
        run.clear();
        lk.lock();
        busy = false;
        if (queue.empty()) {
            cv_done.notify_all();
        }
    }
}
