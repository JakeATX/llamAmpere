#include "llama-b1.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>

#ifndef _WIN32
#include <unistd.h>
#endif

static int64_t b1_env_int(const char * name, int64_t def) {
    const char * e = getenv(name);
    return e && *e ? atoll(e) : def;
}

static size_t b1_align_up(size_t x, size_t a) {
    return (x + a - 1) / a * a;
}

static bool b1_read_file(int fd, void * dst, size_t offs, size_t len) {
#ifndef _WIN32
    char * p = (char *) dst;
    while (len > 0) {
        const ssize_t n = pread(fd, p, len, (off_t) offs);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        p    += n;
        offs += (size_t) n;
        len  -= (size_t) n;
    }
    return true;
#else
    GGML_UNUSED_VARS(fd, dst, offs, len);
    return false;
#endif
}

llama_b1::llama_b1(const llama_model & model, ggml_backend_t backend_gpu) : model(model), backend_gpu(backend_gpu) {
    dev = ggml_backend_get_device(backend_gpu);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        LLAMA_LOG_WARN("%s: [B1] the compute backend is not a GPU, B1 is off\n", __func__);
        return;
    }
    buft_dev = ggml_backend_dev_buffer_type(dev);

    const auto & hp = model.hparams;
    n_layer = (int) hp.n_layer();
    n_embd  = hp.n_embd;
    n_hc    = hp.dsv4_hc_mult > 0 ? hp.dsv4_hc_mult : 1;
    layers.resize(n_layer);

    for (const auto & [name, t] : model.tensors_by_name) {
        int il = -1;
        int n  = 0;
        if (sscanf(name.c_str(), "blk.%d.%n", &il, &n) != 1 || n <= 0) {
            continue;
        }
        if (name.find("_exps") == std::string::npos || il < 0 || il >= n_layer) {
            continue; // not an expert tensor, or a layer the prefill does not run (MTP)
        }

        ggml_backend_buffer_t buf = t->buffer;
        const char * tname = ggml_type_name(t->type);
        if (buf == nullptr || t->view_src != nullptr || t->extra != nullptr || t->data == nullptr) {
            LLAMA_LOG_WARN("%s: [B1] %s is not a plain allocated tensor, B1 is off\n", __func__, name.c_str());
            return;
        }
        if (strncmp(tname, "tq", 2) == 0 || strncmp(tname, "turbo", 5) == 0) {
            // these types are converted on upload (set_tensor), so their GPU bytes differ from the host bytes
            LLAMA_LOG_WARN("%s: [B1] %s has type %s, B1 is off\n", __func__, name.c_str(), tname);
            return;
        }

        exps_tensor e;
        e.t         = t;
        e.nbytes    = ggml_nbytes(t);
        e.alloc     = ggml_backend_buft_get_alloc_size(buft_dev, t);
        e.home_buf  = buf;
        e.home_data = t->data;
        e.resident  = !ggml_backend_buffer_is_host(buf) && ggml_backend_buffer_get_type(buf) == buft_dev;

        if (!e.resident) {
            const char * bname = ggml_backend_buft_name(ggml_backend_buffer_get_type(buf));
            if (!ggml_backend_buffer_is_host(buf) || strstr(bname, "REPACK") || strstr(bname, "AMX") || strstr(bname, "KLEIDI")) {
                LLAMA_LOG_WARN("%s: [B1] %s lives in buffer type %s, B1 is off\n", __func__, name.c_str(), bname);
                return;
            }
        }

        const auto it = model.b1_src.find(name);
        if (it != model.b1_src.end()) {
            e.src = it->second;
        }

        const std::string role = name.substr((size_t) n);
        auto rit = std::find(roles.begin(), roles.end(), role);
        if (rit == roles.end()) {
            roles.push_back(role);
            role_max.push_back(0);
            rit = roles.end() - 1;
        }
        e.role = (int) (rit - roles.begin());
        role_max[e.role] = std::max(role_max[e.role], e.alloc);

        layers[il].tensors.push_back((int) exps.size());
        layers[il].bytes += e.alloc;
        exps.push_back(e);
    }

    if (exps.empty()) {
        LLAMA_LOG_WARN("%s: [B1] the model has no expert tensors, B1 is off\n", __func__);
        return;
    }

    int n_res = 0;
    for (auto & L : layers) {
        L.resident = !L.tensors.empty();
        for (int ti : L.tensors) {
            L.resident = L.resident && exps[ti].resident;
        }
        n_res += L.resident;
    }

    backend_copy = ggml_backend_dev_init(dev, nullptr);
    if (backend_copy == nullptr) {
        LLAMA_LOG_WARN("%s: [B1] cannot create the copy backend, B1 is off\n", __func__);
        return;
    }
    // create the copy stream now, on this thread
    ggml_backend_synchronize(backend_copy);

    for (int i = 0; i < 2; ++i) {
        ev_ready[i] = ggml_backend_event_new(dev);
        ev_free[i]  = ggml_backend_event_new(dev);
        if (ev_ready[i] == nullptr || ev_free[i] == nullptr) {
            LLAMA_LOG_WARN("%s: [B1] the device has no events, B1 is off\n", __func__);
            return;
        }
    }

    ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
    if (buft_host == nullptr) {
        LLAMA_LOG_WARN("%s: [B1] the device has no pinned host buffer type, B1 is off\n", __func__);
        return;
    }

    n_workers   = (int) std::clamp<int64_t>(b1_env_int("LLAMA_B1_THREADS", 8), 1, 64);
    piece_bytes = (size_t) std::clamp<int64_t>(b1_env_int("LLAMA_B1_PIECE_MIB", 16), 1, 256) << 20;

    stg.resize(n_workers);
    for (auto & w : stg) {
        w.resize(2);
        for (auto & s : w) {
            s.buf = ggml_backend_buft_alloc_buffer(buft_host, piece_bytes);
            s.ev  = ggml_backend_event_new(dev);
            if (s.buf == nullptr || s.ev == nullptr) {
                LLAMA_LOG_WARN("%s: [B1] cannot allocate the pinned staging buffers, B1 is off\n", __func__);
                return;
            }
            s.data = (char *) ggml_backend_buffer_get_base(s.buf);
        }
    }

    zero_size = 4096;
    for (const auto & e : exps) {
        zero_size = std::max(zero_size, b1_align_up(e.alloc - e.nbytes, 4096));
    }
    zero_buf = ggml_backend_buft_alloc_buffer(buft_host, zero_size);
    if (zero_buf == nullptr) {
        LLAMA_LOG_WARN("%s: [B1] cannot allocate the pinned zero buffer, B1 is off\n", __func__);
        return;
    }
    zero_data = (char *) ggml_backend_buffer_get_base(zero_buf);
    memset(zero_data, 0, zero_size);

    verified.assign(exps.size(), false);
    dirty_flag.assign(exps.size(), false);

    size_t slot_bytes = 0;
    for (size_t r = 0; r < roles.size(); ++r) {
        slot_bytes += role_max[r];
    }
    LLAMA_LOG_WARN("%s: [B1] layer-major prefill ready: %zu expert tensors in %zu roles, %d of %d layers resident, "
            "slot %.1f MiB, %d copy threads x 2 x %zu MiB pinned, %zu resident tensors lendable\n", __func__,
            exps.size(), roles.size(), n_res, n_layer, slot_bytes/1048576.0, n_workers, piece_bytes >> 20,
            (size_t) std::count_if(exps.begin(), exps.end(), [](const exps_tensor & e) { return e.resident && e.src.fd >= 0; }));

    init_ok = true;
}

llama_b1::~llama_b1() {
    stop_threads();
    if (backend_copy) {
        ggml_backend_synchronize(backend_copy);
    }
    if (init_ok && !dirty_tensors.empty()) {
        // give the model its weights back, another context may use it after this one
        refill();
    }
    release_plan();
    for (auto & w : stg) {
        for (auto & s : w) {
            if (s.ev) {
                ggml_backend_event_free(s.ev);
            }
            if (s.buf) {
                ggml_backend_buffer_free(s.buf);
            }
        }
    }
    if (zero_buf) {
        ggml_backend_buffer_free(zero_buf);
    }
    for (int i = 0; i < 2; ++i) {
        if (ev_ready[i]) {
            ggml_backend_event_free(ev_ready[i]);
        }
        if (ev_free[i]) {
            ggml_backend_event_free(ev_free[i]);
        }
    }
    if (backend_copy) {
        ggml_backend_free(backend_copy);
    }
}

void llama_b1::release_plan() {
    if (fresh_buf) {
        ggml_backend_buffer_free(fresh_buf);
        fresh_buf = nullptr;
    }
    if (ctx_meta) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }
    regions.clear();
    slot_place.clear();
    carry_place.clear();
    carries.clear();
    steps.clear();
    lent_layer.assign(n_layer, false);
    queue.clear();
    queue_head      = 0;
    pieces_left     = 0;
    n_ready         = 0;
    n_freed         = 0;
    step_next_begin = 0;
    cur_step        = -1;
    t_first_begin_us = 0;
}

// first-fit decreasing of items into the free tails of regs; returns the items that did not fit
static std::vector<size_t> b1_pack(const std::vector<size_t> & items, size_t align,
        std::vector<std::pair<size_t, size_t>> & regs /* (size, used) */, std::vector<std::pair<int, size_t>> & out) {
    std::vector<size_t> order(items.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return items[a] > items[b]; });

    out.assign(items.size(), { -1, 0 });
    std::vector<size_t> left;
    for (size_t i : order) {
        bool placed = false;
        for (size_t r = 0; r < regs.size(); ++r) {
            const size_t offs = b1_align_up(regs[r].second, align);
            if (offs + items[i] <= regs[r].first) {
                out[i] = { (int) r, offs };
                regs[r].second = offs + items[i];
                placed = true;
                break;
            }
        }
        if (!placed) {
            left.push_back(i);
        }
    }
    return left;
}

bool llama_b1::begin(const std::vector<uint32_t> & ub_tokens, uint32_t group_ub_req, size_t lend_cap) {
    GGML_ASSERT(init_ok);
    stop_threads();
    release_plan();
    st = {};

    ub_n = ub_tokens;
    const size_t nub = ub_n.size();
    if (nub == 0) {
        return false;
    }

    const size_t align = std::max<size_t>(256, ggml_backend_buft_get_alignment(buft_dev));
    const uint32_t max_tok = *std::max_element(ub_n.begin(), ub_n.end());
    const size_t carry_bytes = b1_align_up((size_t) max_tok * n_embd * n_hc * sizeof(float), align);
    const size_t n_roles = roles.size();

    size_t free_mem = 0;
    size_t total_mem = 0;
    ggml_backend_dev_memory(dev, &free_mem, &total_mem);
    const size_t keep = (size_t) std::max<int64_t>(0, b1_env_int("LLAMA_B1_KEEP_FREE_MIB", 1536)) << 20;
    const size_t fresh_avail = free_mem > keep ? free_mem - keep : 0;

    // lendable resident layers, highest first; a layer still holding borrowed data is always lent
    std::vector<int> forced;
    std::vector<int> order;
    for (int il = n_layer - 1; il >= 0; --il) {
        const auto & L = layers[il];
        if (!L.resident) {
            continue;
        }
        bool any_dirty = false;
        bool lendable  = true;
        for (int ti : L.tensors) {
            any_dirty = any_dirty || dirty_flag[ti];
            lendable  = lendable && exps[ti].src.fd >= 0;
        }
        if (any_dirty) {
            forced.push_back(il);
        } else if (lendable) {
            order.push_back(il);
        }
    }

    // group picker: find the largest group that fits (lending allowed up to the cap), then spread the ubatches evenly
    // over the passes that group size needs, and lend as little as that even group size requires. A group larger than
    // the even size takes the same number of passes but carries more ubatches and lends (and later refills) more.
    const size_t g_max = group_ub_req == 0 ? nub : std::min<size_t>(group_ub_req, nub);
    bool   found = false;
    size_t g_sel = 0;
    std::vector<int> lent_sel;
    std::vector<std::pair<int, size_t>> place_sel; // per item: (region, offs); region == n_lent_regions -> fresh
    std::vector<std::pair<char *, std::pair<ggml_backend_buffer_t, size_t>>> regs_sel; // base, (buf, size)
    size_t fresh_sel = 0;

    // the smallest lend (fewest layers, highest first) that fits g ubatches per group; false when none does
    auto try_group = [&](size_t g) -> bool {
        std::vector<size_t> items;
        for (int s = 0; s < 2; ++s) {
            for (size_t r = 0; r < n_roles; ++r) {
                items.push_back(role_max[r]);
            }
        }
        for (size_t j = 0; j < g; ++j) {
            items.push_back(carry_bytes);
        }

        for (size_t k = 0; k <= order.size(); ++k) {
            std::vector<int> lent = forced;
            size_t lent_bytes = 0;
            for (int il : forced) {
                lent_bytes += layers[il].bytes;
            }
            bool over_cap = false;
            for (size_t i = 0; i < k; ++i) {
                lent_bytes += layers[order[i]].bytes;
                over_cap = over_cap || lent_bytes > lend_cap;
                lent.push_back(order[i]);
            }
            if (over_cap && k > 0) {
                break; // layers still holding borrowed data are lent whatever the cap; new ones only within it
            }

            // the lent spans, merged where they touch (the gap is the previous tensor's own alignment padding)
            std::vector<std::pair<char *, std::pair<ggml_backend_buffer_t, size_t>>> spans;
            for (int il : lent) {
                for (int ti : layers[il].tensors) {
                    const auto & e = exps[ti];
                    const size_t balign = ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(e.home_buf));
                    spans.push_back({ (char *) e.home_data, { e.home_buf, b1_align_up(e.alloc, balign) } });
                }
            }
            std::sort(spans.begin(), spans.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
            std::vector<std::pair<char *, std::pair<ggml_backend_buffer_t, size_t>>> merged;
            for (const auto & sp : spans) {
                if (!merged.empty() && merged.back().second.first == sp.second.first &&
                        merged.back().first + merged.back().second.second == sp.first) {
                    merged.back().second.second += sp.second.second;
                } else {
                    merged.push_back(sp);
                }
            }

            std::vector<std::pair<size_t, size_t>> regs;
            for (const auto & m : merged) {
                // a region base must keep the slot alignment
                const size_t skew = (size_t) ((uintptr_t) m.first % align);
                const size_t pad  = skew ? align - skew : 0;
                regs.push_back({ m.second.second > pad ? m.second.second - pad : 0, 0 });
            }
            std::vector<std::pair<int, size_t>> place;
            const std::vector<size_t> left = b1_pack(items, align, regs, place);

            size_t fresh = 0;
            for (size_t i : left) {
                place[i] = { (int) merged.size(), fresh };
                fresh = b1_align_up(fresh + items[i], align);
            }

            if (fresh <= fresh_avail) {
                g_sel     = g;
                lent_sel  = lent;
                place_sel = place;
                regs_sel  = merged;
                fresh_sel = fresh;
                return true;
            }
        }
        return false;
    };

    size_t g_fit = 0;
    for (size_t g = g_max; g >= 1; --g) {
        if (try_group(g)) {
            g_fit = g;
            break;
        }
    }
    size_t passes = 0;
    if (g_fit > 0) {
        found  = true;
        passes = (nub + g_fit - 1)/g_fit;
        const size_t g_even = (nub + passes - 1)/passes;
        if (g_even < g_fit) {
            // keep the g_fit selection if the smaller group unexpectedly does not pack (first-fit is not monotone)
            const auto keep = std::make_tuple(g_sel, lent_sel, place_sel, regs_sel, fresh_sel);
            if (!try_group(g_even)) {
                std::tie(g_sel, lent_sel, place_sel, regs_sel, fresh_sel) = keep;
            }
        }
    }

    if (!found) {
        LLAMA_LOG_WARN("%s: [B1] no plan fits: %.1f MiB free VRAM (keeping %zu MiB), carry %.1f MiB per ubatch, "
                "lend cap %.1f MiB; using the normal prefill\n", __func__, free_mem/1048576.0, keep >> 20,
                carry_bytes/1048576.0, lend_cap/1048576.0);
        return false;
    }

    // check that the borrowed tensors hold their file bytes before overwriting them: the refill copies the file back
    for (int il : lent_sel) {
        for (int ti : layers[il].tensors) {
            auto & e = exps[ti];
            if (verified[ti] || dirty_flag[ti]) {
                continue;
            }
            const size_t n = std::min<size_t>(e.nbytes, 1u << 16);
            std::vector<uint8_t> a(n);
            std::vector<uint8_t> b(n);
            ggml_backend_tensor_get(e.t, a.data(), e.nbytes - n, n);
            if (!b1_read_file(e.src.fd, b.data(), e.src.offs + e.nbytes - n, n) || memcmp(a.data(), b.data(), n) != 0) {
                LLAMA_LOG_WARN("%s: [B1] %s: VRAM bytes differ from the file, it cannot be lent; using the normal prefill\n",
                        __func__, ggml_get_name(e.t));
                return false;
            }
            verified[ti] = true;
        }
    }

    if (fresh_sel > 0) {
        fresh_buf = ggml_backend_buft_alloc_buffer(buft_dev, fresh_sel);
        if (fresh_buf == nullptr) {
            LLAMA_LOG_WARN("%s: [B1] cannot allocate %.1f MiB of VRAM; using the normal prefill\n", __func__, fresh_sel/1048576.0);
            return false;
        }
        ggml_backend_buffer_set_usage(fresh_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    g_ub = g_sel;
    const size_t n_regions = regs_sel.size() + (fresh_buf ? 1 : 0);

    ggml_init_params ip = {
        /*.mem_size   =*/ (n_regions + nub + 4) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_meta = ggml_init(ip);
    GGML_ASSERT(ctx_meta != nullptr);

    regions.resize(n_regions);
    for (size_t r = 0; r < regs_sel.size(); ++r) {
        const size_t skew = (size_t) ((uintptr_t) regs_sel[r].first % align);
        const size_t pad  = skew ? align - skew : 0;
        regions[r].buf  = regs_sel[r].second.first;
        regions[r].base = regs_sel[r].first + pad;
        regions[r].size = regs_sel[r].second.second > pad ? regs_sel[r].second.second - pad : 0;
    }
    if (fresh_buf) {
        regions.back().buf  = fresh_buf;
        regions.back().base = (char *) ggml_backend_buffer_get_base(fresh_buf);
        regions.back().size = fresh_sel;
    }
    for (auto & rg : regions) {
        rg.raw = ggml_new_tensor_1d(ctx_meta, GGML_TYPE_I8, (int64_t) std::max<size_t>(rg.size, 1));
        rg.raw->buffer = rg.buf;
        rg.raw->data   = rg.base;
    }

    slot_place.assign(2*n_roles, {});
    carry_place.assign(g_sel, {});
    for (size_t i = 0; i < place_sel.size(); ++i) {
        const placement p = { place_sel[i].first, place_sel[i].second };
        GGML_ASSERT(p.region >= 0 && (size_t) p.region < regions.size());
        if (i < 2*n_roles) {
            slot_place[i] = p;
        } else {
            carry_place[i - 2*n_roles] = p;
        }
    }

    carries.resize(nub);
    for (size_t u = 0; u < nub; ++u) {
        const placement & p = carry_place[u % g_sel];
        ggml_tensor * t = ggml_new_tensor_3d(ctx_meta, GGML_TYPE_F32, n_embd, n_hc, ub_n[u]);
        t->buffer = regions[p.region].buf;
        t->data   = regions[p.region].base + p.offs;
        ggml_format_name(t, "b1_carry_%zu", u);
        carries[u] = t;
    }

    lent_layer.assign(n_layer, false);
    for (int il : lent_sel) {
        lent_layer[il] = true;
        for (int ti : layers[il].tensors) {
            if (!dirty_flag[ti]) {
                dirty_flag[ti] = true;
                dirty_tensors.push_back(ti);
            }
            st.bytes_lent += exps[ti].alloc;
        }
    }
    st.bytes_fresh = fresh_sel;
    st.bytes_carry = g_sel*carry_bytes;

    // the streamed layers in call order
    const size_t n_grp = (nub + g_sel - 1)/g_sel;
    grp_first.clear();
    for (size_t g = 0; g < n_grp; ++g) {
        grp_first.push_back(g*g_sel);
        for (int il = 0; il < n_layer; ++il) {
            if (!layers[il].tensors.empty() && (!layers[il].resident || lent_layer[il])) {
                steps.push_back({ il, (int) (steps.size() % 2) });
            }
        }
    }

    if (steps.empty() || final_l0() == 0) {
        LLAMA_LOG_WARN("%s: [B1] no expert layer needs streaming; using the normal prefill\n", __func__);
        release_plan();
        return false;
    }

    LLAMA_LOG_WARN("%s: [B1] plan: %zu ubatches (max %u tokens), largest fitting group %zu (asked %zu) -> %zu passes "
            "of g = %zu, %zu layer streams, lent %zu layers (%.1f MiB, cap %.1f MiB), fresh %.1f MiB, carry %.1f MiB, "
            "free before %.1f MiB\n", __func__,
            nub, max_tok, g_fit, g_max, n_grp, g_sel, steps.size(), lent_sel.size(), st.bytes_lent/1048576.0,
            lend_cap == SIZE_MAX ? -1.0 : lend_cap/1048576.0, fresh_sel/1048576.0, st.bytes_carry/1048576.0,
            free_mem/1048576.0);

    stop     = false;
    io_error = false;
    for (int w = 0; w < n_workers; ++w) {
        workers.emplace_back(&llama_b1::worker, this, w);
    }
    coord = std::thread(&llama_b1::coordinator, this);

    return true;
}

bool llama_b1::layer_streamed(int il) const {
    return il >= 0 && il < n_layer && !layers[il].tensors.empty() && (!layers[il].resident || lent_layer[il]);
}

int32_t llama_b1::final_l0() const {
    int32_t l0 = n_layer - 1;
    while (l0 > 0 && !layer_streamed(l0) && !layer_streamed(l0 - 1)) {
        --l0;
    }
    return l0;
}

int32_t llama_b1::range_end(int32_t l0) const {
    if (layer_streamed(l0)) {
        return l0 + 1;
    }
    const int32_t lf = final_l0();
    int32_t l1 = l0 + 1;
    while (l1 < lf && !layer_streamed(l1)) {
        ++l1;
    }
    return l1;
}

ggml_tensor * llama_b1::carry(size_t ub) {
    GGML_ASSERT(ub < carries.size());
    return carries[ub];
}

bool llama_b1::tensor_streamed(int il, const exps_tensor & e) const {
    return !e.resident || lent_layer[il];
}

void llama_b1::push_tensor_pieces(std::vector<piece> & ps, const exps_tensor & e, region * dst, size_t dst_offs, bool from_file) const {
    for (size_t off = 0; off < e.nbytes; off += piece_bytes) {
        piece p;
        p.src_host = from_file ? nullptr : (const char *) e.home_data + off;
        p.fd       = e.src.fd;
        p.src_offs = e.src.offs + off;
        p.dst      = dst;
        p.dst_offs = dst_offs + off;
        p.len      = std::min(piece_bytes, e.nbytes - off);
        ps.push_back(p);
    }
}

bool llama_b1::run_pieces(std::vector<piece> && ps) {
    {
        std::lock_guard<std::mutex> lk(mtx);
        queue       = std::move(ps);
        queue_head  = 0;
        pieces_left = queue.size();
    }
    cv.notify_all();
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [&] { return stop || pieces_left == 0; });
    return !stop && !io_error;
}

void llama_b1::coordinator() {
    const size_t n_roles = roles.size();
    for (size_t k = 0; k < steps.size(); ++k) {
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return stop || k < 2 || n_freed + 1 >= k; });
            if (stop) {
                return;
            }
        }

        const int il   = steps[k].il;
        const int slot = steps[k].slot;
        if (k >= 2) {
            std::lock_guard<std::mutex> ls(mtx_stream);
            ggml_backend_event_wait(backend_copy, ev_free[slot]);
        }

        std::vector<piece> ps;
        for (int ti : layers[il].tensors) {
            const auto & e = exps[ti];
            if (!tensor_streamed(il, e)) {
                continue;
            }
            const placement & p = slot_place[slot*n_roles + e.role];
            push_tensor_pieces(ps, e, &regions[p.region], p.offs, lent_layer[il]);
        }
        if (!run_pieces(std::move(ps))) {
            std::lock_guard<std::mutex> lk(mtx);
            io_error = true;
            cv.notify_all();
            return;
        }

        {
            std::lock_guard<std::mutex> ls(mtx_stream);
            for (int ti : layers[il].tensors) {
                const auto & e = exps[ti];
                if (!tensor_streamed(il, e) || e.alloc <= e.nbytes) {
                    continue;
                }
                // the row padding past the data must read as zeros (see init_tensor of the CUDA buffer)
                const placement & p = slot_place[slot*n_roles + e.role];
                ggml_backend_tensor_set_async(backend_copy, regions[p.region].raw, zero_data, p.offs + e.nbytes, e.alloc - e.nbytes);
            }
            ggml_backend_event_record(ev_ready[slot], backend_copy);
        }

        {
            std::lock_guard<std::mutex> lk(mtx);
            n_ready = k + 1;
        }
        cv.notify_all();
    }
}

void llama_b1::worker(int w) {
    int b = 0;
    while (true) {
        piece p;
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&] { return stop || queue_head < queue.size(); });
            if (stop) {
                return;
            }
            p = queue[queue_head++];
        }

        auto & s = stg[w][b];
        b ^= 1;
        if (s.used) {
            ggml_backend_event_synchronize(s.ev);
        }

        bool ok = true;
        if (p.src_host) {
            memcpy(s.data, p.src_host, p.len);
        } else {
            ok = p.fd >= 0 && b1_read_file(p.fd, s.data, p.src_offs, p.len);
        }

        if (ok) {
            std::lock_guard<std::mutex> ls(mtx_stream);
            ggml_backend_tensor_set_async(backend_copy, p.dst->raw, s.data, p.dst_offs, p.len);
            ggml_backend_event_record(s.ev, backend_copy);
            s.used = true;
        }

        {
            std::lock_guard<std::mutex> lk(mtx);
            if (ok) {
                st.bytes_h2d += p.len;
            } else {
                io_error = true;
            }
            if (--pieces_left == 0 || !ok) {
                cv.notify_all();
            }
        }
    }
}

void llama_b1::stop_threads() {
    {
        std::lock_guard<std::mutex> lk(mtx);
        stop = true;
    }
    cv.notify_all();
    if (coord.joinable()) {
        coord.join();
    }
    for (auto & t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers.clear();
    {
        std::lock_guard<std::mutex> lk(mtx);
        stop = false;
    }
}

bool llama_b1::layer_begin(int il) {
    if (step_next_begin >= steps.size() || steps[step_next_begin].il != il) {
        return true; // resident and not lent: nothing to stream
    }
    const size_t k = step_next_begin;
    const int slot = steps[k].slot;

    const int64_t t0 = ggml_time_us();
    if (t_first_begin_us == 0) {
        t_first_begin_us = t0;
    }
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&] { return io_error || n_ready > k; });
        if (io_error) {
            LLAMA_LOG_ERROR("%s: [B1] reading the experts of layer %d failed\n", __func__, il);
            return false;
        }
    }
    st.t_wait_ms += (ggml_time_us() - t0)/1000.0;

    ggml_backend_event_wait(backend_gpu, ev_ready[slot]);

    const size_t n_roles = roles.size();
    for (int ti : layers[il].tensors) {
        auto & e = exps[ti];
        if (!tensor_streamed(il, e)) {
            continue;
        }
        const placement & p = slot_place[slot*n_roles + e.role];
        e.t->buffer = regions[p.region].buf;
        e.t->data   = regions[p.region].base + p.offs;
    }
    cur_step = (int) k;
    return true;
}

void llama_b1::restore_step(int k) {
    const int il = steps[k].il;
    for (int ti : layers[il].tensors) {
        auto & e = exps[ti];
        e.t->buffer = e.home_buf;
        e.t->data   = e.home_data;
    }
}

void llama_b1::layer_end(int il) {
    if (cur_step < 0 || steps[cur_step].il != il) {
        return;
    }
    const int k = cur_step;
    restore_step(k);
    // the kernels of the layer's graphs captured the slot pointers at launch; the slot is free once they ran
    ggml_backend_event_record(ev_free[steps[k].slot], backend_gpu);
    {
        std::lock_guard<std::mutex> lk(mtx);
        n_freed = (size_t) k + 1;
    }
    cv.notify_all();
    step_next_begin = (size_t) k + 1;
    cur_step = -1;
}

void llama_b1::end(bool failed) {
    if (cur_step >= 0) {
        restore_step(cur_step);
        cur_step = -1;
    }
    stop_threads();
    ggml_backend_synchronize(backend_copy);
    ggml_backend_synchronize(backend_gpu);

    st.t_stream_us = t_first_begin_us ? ggml_time_us() - t_first_begin_us : 0;
    const double s = st.t_stream_us/1e6;
    LLAMA_LOG_WARN("%s: [B1] %s: %zu groups of %zu ubatches, streamed %.2f GiB (%.1f GB/s over %.2f s), waited %.0f ms for "
            "copies, lent %.1f MiB, fresh %.1f MiB, carry %.1f MiB\n", __func__, failed ? "FAILED" : "done",
            grp_first.size(), g_ub, st.bytes_h2d/1073741824.0, s > 0 ? st.bytes_h2d/1e9/s : 0.0, s, st.t_wait_ms,
            st.bytes_lent/1048576.0, st.bytes_fresh/1048576.0, st.bytes_carry/1048576.0);

    release_plan();
}

bool llama_b1::refill() {
    if (dirty_tensors.empty()) {
        return true;
    }
    const int64_t t0 = ggml_time_us();

    stop_threads();
    release_plan();

    ggml_init_params ip = {
        /*.mem_size   =*/ (dirty_tensors.size() + 4) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_meta = ggml_init(ip);
    GGML_ASSERT(ctx_meta != nullptr);

    // each borrowed tensor's home is its own region
    regions.resize(dirty_tensors.size());
    std::vector<piece> ps;
    size_t bytes = 0;
    for (size_t i = 0; i < dirty_tensors.size(); ++i) {
        const auto & e = exps[dirty_tensors[i]];
        auto & rg = regions[i];
        rg.buf  = e.home_buf;
        rg.base = (char *) e.home_data;
        rg.size = e.alloc;
        rg.raw  = ggml_new_tensor_1d(ctx_meta, GGML_TYPE_I8, (int64_t) e.alloc);
        rg.raw->buffer = rg.buf;
        rg.raw->data   = rg.base;
        push_tensor_pieces(ps, e, &rg, 0, true);
        bytes += e.nbytes;
    }

    stop     = false;
    io_error = false;
    for (int w = 0; w < n_workers; ++w) {
        workers.emplace_back(&llama_b1::worker, this, w);
    }
    const uint64_t h2d0 = st.bytes_h2d;
    const bool ok = run_pieces(std::move(ps));
    {
        std::lock_guard<std::mutex> ls(mtx_stream);
        for (size_t i = 0; i < dirty_tensors.size(); ++i) {
            const auto & e = exps[dirty_tensors[i]];
            if (e.alloc > e.nbytes) {
                ggml_backend_tensor_set_async(backend_copy, regions[i].raw, zero_data, e.nbytes, e.alloc - e.nbytes);
            }
        }
    }
    stop_threads();
    ggml_backend_synchronize(backend_copy);
    GGML_UNUSED(h2d0);

    const double ms = (ggml_time_us() - t0)/1000.0;
    if (!ok) {
        LLAMA_LOG_ERROR("%s: [B1] refill of %zu lent tensors FAILED after %.0f ms (I/O error)\n", __func__, dirty_tensors.size(), ms);
        release_plan();
        return false;
    }
    LLAMA_LOG_WARN("%s: [B1] refill: %zu lent tensors, %.2f GiB in %.0f ms (%.1f GB/s)\n", __func__,
            dirty_tensors.size(), bytes/1073741824.0, ms, ms > 0 ? bytes/1e6/ms : 0.0);

    for (int ti : dirty_tensors) {
        dirty_flag[ti] = false;
    }
    dirty_tensors.clear();
    release_plan();
    return true;
}
