#include "ggml-ledger.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

struct ggml_ledger_slot {
    std::string          site;
    std::string          key;
    std::atomic<int64_t> count{0};
};

namespace {

struct ledger_table {
    std::mutex mtx;
    // slots are never freed or moved while the process runs, so handed-out pointers stay valid
    std::map<std::pair<std::string, std::string>, std::unique_ptr<ggml_ledger_slot>> slots;
};

ledger_table & table() {
    static ledger_table * t = new ledger_table(); // leaked on purpose: counters must outlive static destructors
    return *t;
}

int env_state() {
    const char * e = getenv("GGML_LEDGER");
    return e != nullptr && e[0] != '\0' && e[0] != '0' ? 1 : 0;
}

std::atomic<int> g_enabled{-1};
std::atomic<bool> g_atexit_registered{false};

void print_at_exit() {
    if (ggml_ledger_enabled()) {
        ggml_ledger_print();
    }
}

void register_atexit() {
    bool expected = false;
    if (g_atexit_registered.compare_exchange_strong(expected, true)) {
        atexit(print_at_exit);
    }
}

} // namespace

bool ggml_ledger_enabled(void) {
    int s = g_enabled.load(std::memory_order_relaxed);
    if (s < 0) {
        const int e = env_state();
        int expected = -1;
        if (g_enabled.compare_exchange_strong(expected, e)) {
            s = e;
            if (e) {
                register_atexit();
            }
        } else {
            s = expected;
        }
    }
    return s != 0;
}

void ggml_ledger_set_enabled(bool enabled) {
    g_enabled.store(enabled ? 1 : 0);
    if (enabled) {
        register_atexit();
    }
}

ggml_ledger_slot * ggml_ledger_slot_get(const char * site, const char * key) {
    ledger_table & t = table();
    std::lock_guard<std::mutex> lock(t.mtx);
    auto & p = t.slots[{site ? site : "", key ? key : ""}];
    if (!p) {
        p = std::make_unique<ggml_ledger_slot>();
        p->site = site ? site : "";
        p->key  = key  ? key  : "";
    }
    return p.get();
}

void ggml_ledger_slot_add(ggml_ledger_slot * slot, int64_t n) {
    if (slot != nullptr) {
        slot->count.fetch_add(n, std::memory_order_relaxed);
    }
}

void ggml_ledger_add(const char * site, const char * key, int64_t n) {
    if (!ggml_ledger_enabled()) {
        return;
    }
    ggml_ledger_slot_add(ggml_ledger_slot_get(site, key), n);
}

void ggml_ledger_addf(const char * site, int64_t n, const char * key_fmt, ...) {
    if (!ggml_ledger_enabled()) {
        return;
    }
    char key[256];
    va_list args;
    va_start(args, key_fmt);
    vsnprintf(key, sizeof(key), key_fmt, args);
    va_end(args);
    ggml_ledger_slot_add(ggml_ledger_slot_get(site, key), n);
}

void ggml_ledger_foreach(ggml_ledger_cb cb, void * user_data) {
    ledger_table & t = table();
    std::lock_guard<std::mutex> lock(t.mtx);
    for (const auto & kv : t.slots) {
        cb(kv.second->site.c_str(), kv.second->key.c_str(), kv.second->count.load(std::memory_order_relaxed), user_data);
    }
}

void ggml_ledger_print(void) {
    ledger_table & t = table();
    std::lock_guard<std::mutex> lock(t.mtx);
    size_t n_nonzero = 0;
    for (const auto & kv : t.slots) {
        n_nonzero += kv.second->count.load(std::memory_order_relaxed) != 0;
    }
    fprintf(stderr, "ggml_ledger: begin (%zu counters)\n", n_nonzero);
    for (const auto & kv : t.slots) {
        const int64_t c = kv.second->count.load(std::memory_order_relaxed);
        if (c != 0) {
            fprintf(stderr, "ggml_ledger: %-18s %-72s %lld\n", kv.second->site.c_str(), kv.second->key.c_str(), (long long) c);
        }
    }
    fprintf(stderr, "ggml_ledger: end\n");
    fflush(stderr);
}

void ggml_ledger_reset(void) {
    ledger_table & t = table();
    std::lock_guard<std::mutex> lock(t.mtx);
    for (auto & kv : t.slots) {
        kv.second->count.store(0, std::memory_order_relaxed);
    }
}
