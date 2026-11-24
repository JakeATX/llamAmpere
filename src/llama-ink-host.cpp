#include "llama-ink-host.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

int ink_host_env(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

// the same test as the prompt streamer: a plain contiguous host weight in a non-repacked buffer
bool ink_host_weight(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr || t->data == nullptr || t->view_src != nullptr || t->extra != nullptr ||
            ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(t->buffer) || !ggml_is_contiguous(t)) {
        return false;
    }
    const char * bname = ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer));
    return !strstr(bname, "REPACK") && !strstr(bname, "AMX") && !strstr(bname, "KLEIDI");
}

// gate_up, gate, up, down: the streamer's order
std::vector<ggml_tensor *> ink_host_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

struct ink_host_registry {
    std::mutex mtx;
    std::vector<std::pair<uintptr_t, uintptr_t>> pinned; // [begin, end)
    std::unordered_map<const void *, std::pair<size_t, ggml_backend_buffer_type_t>> spans;
};

ink_host_registry & registry() {
    static ink_host_registry r;
    return r;
}

ggml_backend_reg_t ink_host_cuda_reg() {
    return ggml_backend_reg_by_name("CUDA");
}

#ifdef __linux__
// bytes of free 2 MiB (or larger) blocks, from /proc/buddyinfo
size_t ink_host_free_huge_bytes() {
    std::ifstream f("/proc/buddyinfo");
    std::string line;
    size_t blocks = 0;
    while (std::getline(f, line)) {
        const size_t zone = line.find("zone");
        if (zone == std::string::npos) {
            continue;
        }
        const char * c = line.c_str() + zone + 4;
        while (*c == ' ') { ++c; }
        while (*c && *c != ' ') { ++c; }
        for (int order = 0; *c; ++order) {
            char * end = nullptr;
            const unsigned long n = strtoul(c, &end, 10);
            if (end == c) {
                break;
            }
            c = end;
            if (order >= 9) {
                blocks += (size_t) n << (order - 9);
            }
        }
    }
    return blocks << 21;
}
#endif

// the experts' new home and its registrations; freed with the model
struct ink_host_state {
    void * arena = nullptr;
    size_t arena_size = 0;
    std::vector<void *> pins; // ggml_backend_cuda_host_pin(_ro), undone with ggml_backend_cuda_host_unpin

    ~ink_host_state() {
        ggml_backend_reg_t reg = ink_host_cuda_reg();
        if (reg) {
            auto unpin = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_unpin");
            for (void * p : pins) { if (unpin) { unpin(p); } }
        }
        {
            auto & r = registry();
            std::lock_guard<std::mutex> lk(r.mtx);
            const uintptr_t lo = (uintptr_t) arena, hi = lo + arena_size;
            for (auto it = r.spans.begin(); it != r.spans.end();) {
                it = (uintptr_t) it->first >= lo && (uintptr_t) it->first < hi ? r.spans.erase(it) : std::next(it);
            }
            r.pinned.clear(); // one model with host experts per process in practice; stale ranges must not outlive it
        }
#ifdef __linux__
        if (arena) {
            munmap(arena, arena_size);
        }
#endif
    }
};

void ink_host_add_pinned(uintptr_t lo, uintptr_t hi) {
    auto & r = registry();
    std::lock_guard<std::mutex> lk(r.mtx);
    r.pinned.emplace_back(lo, hi);
}

}

bool llama_ink_host_pinned(const void * ptr, size_t size) {
    auto & r = registry();
    std::lock_guard<std::mutex> lk(r.mtx);
    const uintptr_t lo = (uintptr_t) ptr, hi = lo + size;
    for (const auto & p : r.pinned) {
        if (lo >= p.first && hi <= p.second) {
            return true;
        }
    }
    return false;
}

size_t llama_ink_host_layer_span(const void * ptr, ggml_backend_buffer_type_t buft_dev) {
    auto & r = registry();
    std::lock_guard<std::mutex> lk(r.mtx);
    const auto it = r.spans.find(ptr);
    return it != r.spans.end() && it->second.second == buft_dev ? it->second.first : 0;
}

void llama_ink_host_prepare(const llama_model & model, ggml_backend_buffer_type_t buft_dev) {
    const bool want_arena = ink_host_env("LLAMA_INK_HOST_ARENA", 0) != 0;
    const bool want_pin   = ink_host_env("LLAMA_INK_HOST_PIN", 0) != 0;
    if ((!want_arena && !want_pin) || model.ink_host || model.hparams.n_expert == 0) {
        return;
    }

    // the layers whose experts all sit in plain host weights; a looped layer's visits share tensors: once
    std::vector<std::vector<ggml_tensor *>> layers;
    for (const auto & layer : model.layers) {
        auto experts = ink_host_layer_experts(layer);
        if (experts.empty() || !std::all_of(experts.begin(), experts.end(), ink_host_weight) ||
                std::find(layers.begin(), layers.end(), experts) != layers.end()) {
            continue;
        }
        layers.push_back(std::move(experts));
    }
    if (layers.empty()) {
        return;
    }
    auto state = std::make_shared<ink_host_state>();
    model.ink_host = state;

    const size_t page = 4096;
    const size_t align = buft_dev ? std::max<size_t>(256, ggml_backend_buft_get_alignment(buft_dev)) : page;

    if (want_arena) {
#ifdef __linux__
        // layout: each layer at a page boundary, its tensors at the streamer's slot offsets, tails left zero
        struct place { ggml_tensor * t; size_t off; };
        std::vector<place> places;
        std::vector<std::pair<size_t, size_t>> layer_span; // offset of the layer, span
        size_t total = 0;
        for (const auto & experts : layers) {
            total = GGML_PAD(total, page);
            const size_t base = total;
            size_t off = 0;
            for (ggml_tensor * t : experts) {
                off = GGML_PAD(off, align);
                places.push_back({ t, base + off });
                off += buft_dev ? ggml_backend_buft_get_alloc_size(buft_dev, t) + 512 : ggml_nbytes(t);
            }
            layer_span.emplace_back(base, off);
            total = base + off;
        }
        total = GGML_PAD(total, page);

        constexpr size_t huge = 2ull << 20;
        const size_t padded = total + huge;
        void * raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw == MAP_FAILED) {
            LLAMA_LOG_WARN("%s: [INKHOST] cannot map a %.2f GiB expert arena: %s; experts stay in file pages\n", __func__,
                    total/1073741824.0, strerror(errno));
        } else {
            // 2 MiB aligned, so the kernel can back it with huge pages
            const uintptr_t start = (uintptr_t) raw;
            const uintptr_t aligned = (start + huge - 1) & ~(uintptr_t) (huge - 1);
            if (aligned > start) {
                munmap(raw, aligned - start);
            }
            const uintptr_t end = aligned + total, raw_end = start + padded;
            if (raw_end > end) {
                munmap((void *) end, raw_end - end);
            }
            state->arena = (void *) aligned;
            state->arena_size = total;

            const int thp = ink_host_env("LLAMA_INK_HOST_THP", 1);
            if (thp > 0) {
                const size_t free_huge = ink_host_free_huge_bytes();
                if (thp == 1 && free_huge < total/2) {
                    LLAMA_LOG_WARN("%s: [INKHOST] no huge pages: %.1f GiB of free 2 MiB blocks for a %.1f GiB arena "
                            "(LLAMA_INK_HOST_THP=2 asks anyway)\n", __func__, free_huge/1073741824.0, total/1073741824.0);
                } else if (madvise(state->arena, total, MADV_HUGEPAGE) != 0) {
                    LLAMA_LOG_WARN("%s: [INKHOST] madvise(MADV_HUGEPAGE) failed: %s\n", __func__, strerror(errno));
                }
            }

            // copy in parallel (the first touch also faults the arena in), then point the tensors at their copy
            const int64_t t0 = ggml_time_us();
            const int n_thr = std::clamp(ink_host_env("LLAMA_INK_HOST_THREADS", 4), 1, 32);
            std::vector<std::thread> thr;
            for (int i = 0; i < n_thr; ++i) {
                thr.emplace_back([&, i] {
                    for (size_t k = i; k < places.size(); k += n_thr) {
                        memcpy((char *) state->arena + places[k].off, places[k].t->data, ggml_nbytes(places[k].t));
                    }
                });
            }
            for (auto & th : thr) {
                th.join();
            }
            for (const place & p : places) {
                void * old = p.t->data;
                const size_t nb = ggml_nbytes(p.t);
                p.t->data = (char *) state->arena + p.off;
                // the file pages of the old copy are no longer read: drop them from this process (reclaimable cache)
                const uintptr_t lo = ((uintptr_t) old + page - 1) & ~(uintptr_t) (page - 1);
                const uintptr_t hi = ((uintptr_t) old + nb) & ~(uintptr_t) (page - 1);
                if (hi > lo) {
                    madvise((void *) lo, hi - lo, MADV_DONTNEED);
                }
            }
            if (buft_dev) {
                auto & r = registry();
                std::lock_guard<std::mutex> lk(r.mtx);
                for (const auto & ls : layer_span) {
                    r.spans[(char *) state->arena + ls.first] = { ls.second, buft_dev };
                }
            }
            LLAMA_LOG_WARN("%s: [INKHOST] %zu layers of host experts (%.2f GiB) copied to an anonymous arena in %.0f ms "
                    "(LLAMA_INK_HOST_ARENA)\n", __func__, layers.size(), total/1073741824.0, (ggml_time_us() - t0)/1000.0);
        }
#else
        LLAMA_LOG_WARN("%s: [INKHOST] LLAMA_INK_HOST_ARENA needs Linux; experts stay in file pages\n", __func__);
#endif
    }

    if (want_pin) {
        ggml_backend_reg_t reg = ink_host_cuda_reg();
        const int64_t t0 = ggml_time_us();
        if (reg == nullptr) {
            LLAMA_LOG_WARN("%s: [INKHOST] LLAMA_INK_HOST_PIN: no CUDA backend\n", __func__);
        } else if (state->arena) {
            auto pin = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin");
            if (pin && pin(state->arena, state->arena_size)) {
                state->pins.push_back(state->arena);
                ink_host_add_pinned((uintptr_t) state->arena, (uintptr_t) state->arena + state->arena_size);
            }
        } else {
            // in place: the file mapping is read only. Page-aligned ranges, merged so that no page is registered twice.
            auto reg_ro = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_pin_ro");
            std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
            for (const auto & experts : layers) {
                for (const ggml_tensor * t : experts) {
                    const uintptr_t lo = (uintptr_t) t->data & ~(uintptr_t) (page - 1);
                    const uintptr_t hi = ((uintptr_t) t->data + ggml_nbytes(t) + page - 1) & ~(uintptr_t) (page - 1);
                    ranges.emplace_back(lo, hi);
                }
            }
            std::sort(ranges.begin(), ranges.end());
            std::vector<std::pair<uintptr_t, uintptr_t>> merged;
            for (const auto & r : ranges) {
                if (!merged.empty() && r.first <= merged.back().second) {
                    merged.back().second = std::max(merged.back().second, r.second);
                } else {
                    merged.push_back(r);
                }
            }
            for (const auto & r : merged) {
                if (reg_ro && reg_ro((void *) r.first, r.second - r.first)) {
                    state->pins.push_back((void *) r.first);
                    ink_host_add_pinned(r.first, r.second);
                }
            }
        }
        size_t n_pinned = 0;
        {
            auto & r = registry();
            std::lock_guard<std::mutex> lk(r.mtx);
            for (const auto & p : r.pinned) {
                n_pinned += p.second - p.first;
            }
        }
        LLAMA_LOG_WARN("%s: [INKHOST] %.2f GiB of host experts page-locked (%s) in %.0f ms (LLAMA_INK_HOST_PIN)\n", __func__,
                n_pinned/1073741824.0, state->arena ? "arena" : "file pages in place", (ggml_time_us() - t0)/1000.0);
    }
}
