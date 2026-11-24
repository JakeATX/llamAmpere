#include "llama-moe-ramlock.h"

#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define LLAMA_RAMLOCK_POSIX 1
#endif

std::unique_ptr<llama_moe_ramlock> llama_moe_ramlock::from_env(int32_t n_layer, int32_t n_expert) {
    const char * g = getenv("LLAMA_MOE_RAM_LOCK_GIB");
    const double gib = g ? atof(g) : 0.0;
    if (!(gib > 0.0) || n_layer <= 0 || n_expert <= 0) {
        return nullptr;
    }
#if !defined(LLAMA_RAMLOCK_POSIX)
    LLAMA_LOG_WARN("%s: LLAMA_MOE_RAM_LOCK_GIB is not supported on this platform, ignored\n", __func__);
    return nullptr;
#else
    std::unique_ptr<llama_moe_ramlock> r(new llama_moe_ramlock());
    r->budget   = (size_t) (gib*1024.0*1024.0*1024.0);
    r->n_layer  = n_layer;
    r->n_expert = n_expert;
    if (const char * e = getenv("LLAMA_MOE_RAM_LOCK_EVERY"); e && atoll(e) > 0) {
        r->every = atoll(e);
    }
    if (const char * p = getenv("LLAMA_MOE_RAM_LOCK_PROFILE"); p && *p) {
        r->profile_path = p;
    }
    if (const char * s = getenv("LLAMA_MOE_RAM_LOCK_SAVE"); s && *s) {
        r->save_path = s;
    }
    r->heat.assign((size_t) n_layer*n_expert, 0.0);
    r->locked.assign((size_t) n_layer*n_expert, 0);
    r->tensors.resize(n_layer);
    return r;
#endif
}

void llama_moe_ramlock::add_layer(int32_t il, const std::vector<ggml_tensor *> & experts) {
    if (il >= 0 && il < n_layer) {
        tensors[il] = experts;
    }
}

void llama_moe_ramlock::start() {
    if (!profile_path.empty()) {
        FILE * f = fopen(profile_path.c_str(), "r");
        if (f == nullptr) {
            LLAMA_LOG_WARN("%s: cannot read %s, starting with no heat\n", __func__, profile_path.c_str());
        } else {
            long long il, e;
            double c;
            size_t n = 0;
            while (fscanf(f, "%lld %lld %lf", &il, &e, &c) == 3) {
                if (il >= 0 && il < n_layer && e >= 0 && e < n_expert && c >= 0.0) {
                    heat[(size_t) il*n_expert + e] = c;
                    n++;
                }
            }
            fclose(f);
            LLAMA_LOG_INFO("%s: %zu expert counts from %s\n", __func__, n, profile_path.c_str());
            rerank();
        }
    }
    LLAMA_LOG_WARN("%s: [RAMLOCK] locking the hottest %.2f GiB of host experts in RAM, re-rank every %lld lookups\n",
            __func__, budget/1024.0/1024.0/1024.0, (long long) every);
}

void llama_moe_ramlock::observe(int32_t il, const int32_t * ids, size_t n_ids) {
    if (failed || il < 0 || il >= n_layer || tensors[il].empty()) {
        return;
    }
    double * h = heat.data() + (size_t) il*n_expert;
    for (size_t i = 0; i < n_ids; ++i) {
        if (ids[i] >= 0 && ids[i] < n_expert) {
            h[ids[i]] += 1.0;
        }
    }
    if (++n_obs % every == 0) {
        rerank();
    }
}

bool llama_moe_ramlock::lock_one(int32_t key, bool lock) {
#if defined(LLAMA_RAMLOCK_POSIX)
    static const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const int32_t il = key / n_expert;
    const int32_t e  = key % n_expert;
    for (const ggml_tensor * t : tensors[il]) {
        // whole pages inside this expert only, so a neighbour's lock state is never changed by ours
        const uintptr_t b = (uintptr_t) t->data + (size_t) e*t->nb[2];
        const uintptr_t lo = (b + page - 1) & ~(uintptr_t) (page - 1);
        const uintptr_t hi = (b + t->nb[2]) & ~(uintptr_t) (page - 1);
        if (hi <= lo) {
            continue;
        }
        n_lock_calls++;
        if ((lock ? mlock((void *) lo, hi - lo) : munlock((void *) lo, hi - lo)) != 0) {
            if (lock) {
                LLAMA_LOG_WARN("%s: [RAMLOCK] mlock failed (%s; check ulimit -l), RAM lock turned off\n", __func__, strerror(errno));
                return false;
            }
        }
    }
    return true;
#else
    (void) key; (void) lock;
    return false;
#endif
}

void llama_moe_ramlock::rerank() {
    if (failed) {
        return;
    }
    const size_t n = heat.size();
    std::vector<int32_t> order;
    order.reserve(n);
    std::vector<size_t> bytes(n, 0);
    for (size_t k = 0; k < n; ++k) {
        const int32_t il = (int32_t) (k / n_expert);
        if (tensors[il].empty()) {
            continue;
        }
        for (const ggml_tensor * t : tensors[il]) {
            bytes[k] += t->nb[2];
        }
        if (heat[k] > 0.0) {
            order.push_back((int32_t) k);
        }
    }
    // hottest per byte first; ties by key so the pick is deterministic
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return heat[a]/bytes[a] > heat[b]/bytes[b];
    });
    std::vector<uint8_t> want(n, 0);
    size_t used = 0;
    for (int32_t k : order) {
        if (used + bytes[k] <= budget) {
            want[k] = 1;
            used += bytes[k];
        }
    }
    // unlock first, so the locked total never exceeds the budget
    for (size_t k = 0; k < n; ++k) {
        if (locked[k] && !want[k]) {
            lock_one((int32_t) k, false);
            locked[k] = 0;
            locked_bytes -= bytes[k];
        }
    }
    for (size_t k = 0; k < n && !failed; ++k) {
        if (want[k] && !locked[k]) {
            if (!lock_one((int32_t) k, true)) {
                failed = true;
                break;
            }
            locked[k] = 1;
            locked_bytes += bytes[k];
        }
    }
    for (double & h : heat) {
        h *= 0.5;
    }
    // the first re-ranks at WARN (the server log hides library INFO; so a smoke run shows the lock working), later ones at DEBUG
    if (++n_rerank <= 4) {
        LLAMA_LOG_WARN("%s: [RAMLOCK] re-rank %zu: %.3f GiB locked, %zu experts ranked\n", __func__, n_rerank,
                locked_bytes/1024.0/1024.0/1024.0, order.size());
    } else {
        LLAMA_LOG_DEBUG("%s: [RAMLOCK] %.3f GiB locked\n", __func__, locked_bytes/1024.0/1024.0/1024.0);
    }
}

llama_moe_ramlock::~llama_moe_ramlock() {
    if (!save_path.empty()) {
        if (FILE * f = fopen(save_path.c_str(), "w")) {
            for (size_t k = 0; k < heat.size(); ++k) {
                if (heat[k] > 0.0) {
                    fprintf(f, "%zu %zu %.6g\n", k / n_expert, k % n_expert, heat[k]);
                }
            }
            fclose(f);
        }
    }
    for (size_t k = 0; k < locked.size(); ++k) {
        if (locked[k]) {
            lock_one((int32_t) k, false);
        }
    }
    LLAMA_LOG_WARN("%s: [RAMLOCK] %.2f GiB were locked at exit, %zu lock calls\n", __func__,
            locked_bytes/1024.0/1024.0/1024.0, n_lock_calls);
}
