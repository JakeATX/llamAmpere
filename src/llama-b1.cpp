#include "llama-b1.h"
#include "llama-graph.h"
#include <cmath>

#include "llama-impl.h"
#include "llama-model.h"
#include "../ggml/src/ggml-backend-moe-cache.h"

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

// Saturate the final loan, not free-minus-keep: a full expert cache may
// leave less free memory than the reserve before any prompt scratch exists.
static size_t b1_borrow_deficit(size_t required, size_t keep, size_t free_mem) {
    const size_t available = free_mem > keep ? free_mem - keep : 0;
    const size_t scratch_shortfall = required > available ? required - available : 0;
    const size_t keep_shortfall = keep > free_mem ? keep - free_mem : 0;
    return scratch_shortfall > SIZE_MAX - keep_shortfall ? SIZE_MAX : scratch_shortfall + keep_shortfall;
}

static bool b1_read_file(int fd, void * dst, size_t offs, size_t len, size_t alignment = 1, void * scratch = nullptr) {
#ifndef _WIN32
    const size_t skip = offs % alignment;
    const size_t needed = skip + len;
    const size_t read_size = b1_align_up(needed, alignment);
    // alignment > 1 means an O_DIRECT descriptor: its reads must never land in pinned or cudaHostRegister'ed pages
    // (UVM in HMM mode can panic the PC), so they go to a plain per-thread malloc bounce buffer and are copied out
    struct bounce_t { void * p = nullptr; size_t n = 0; ~bounce_t() { free(p); } };
    thread_local bounce_t bounce;
    char * p = (char *) (scratch ? scratch : dst);
    if (alignment > 1) {
        if (bounce.n < read_size) {
            free(bounce.p);
            bounce.p = nullptr;
            bounce.n = 0;
            if (posix_memalign(&bounce.p, alignment, read_size) != 0 || !bounce.p) {
                bounce.p = nullptr;
                return false;
            }
            bounce.n = read_size;
        }
        p = (char *) bounce.p;
    }
    offs -= skip;
    size_t got = 0;
    while (got < read_size) {
        const ssize_t n = pread(fd, p + got, read_size - got, (off_t) (offs + got));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            break;
        }
        got += (size_t) n;
        // A final direct read may include padding beyond EOF.
        if (got >= needed) { break; }
    }
    if (got < needed) { return false; }
    if (p + skip != dst) { memcpy(dst, p + skip, len); }
    return true;
#else
    GGML_UNUSED_VARS(fd, dst, offs, len, alignment, scratch);
    return false;
#endif
}

// RAM the page cache can still use, in bytes (MemAvailable; LLAMA_B1_RAM_AVAIL_MIB overrides it). -1 if unknown.
static int64_t b1_ram_available() {
    const int64_t fake = b1_env_int("LLAMA_B1_RAM_AVAIL_MIB", -1);
    if (fake >= 0) {
        return fake << 20;
    }
    int64_t avail = -1;
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char line[256];
        long long kb = 0;
        while (fgets(line, sizeof(line), f)) {
            if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1) {
                avail = (int64_t) kb << 10;
                break;
            }
        }
        fclose(f);
    }
    return avail;
}

// true when the address lies in a file mapping (mmap'd model pages live in the page cache); true when unknown
static bool b1_file_backed(const void * p) {
    bool res = true;
    if (FILE * f = fopen("/proc/self/maps", "r")) {
        char line[512];
        const unsigned long long a = (unsigned long long) (uintptr_t) p;
        while (fgets(line, sizeof(line), f)) {
            unsigned long long lo = 0, hi = 0, off = 0, inode = 0;
            unsigned dmaj = 0, dmin = 0;
            char perms[8];
            if (sscanf(line, "%llx-%llx %7s %llx %x:%x %llu", &lo, &hi, perms, &off, &dmaj, &dmin, &inode) == 7 &&
                    a >= lo && a < hi) {
                res = inode != 0;
                break;
            }
        }
        fclose(f);
    }
    return res;
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
        e.home_nb2  = t->nb[2];
        ggml_tensor compact = *t;
        compact.nb[2] = compact.nb[1] * compact.ne[1];
        compact.nb[3] = compact.nb[2] * compact.ne[2];
        e.nbytes    = ggml_nbytes(&compact);
        e.alloc     = ggml_backend_buft_get_alloc_size(buft_dev, &compact);
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
            L.res_bytes += exps[ti].resident ? exps[ti].alloc : 0;
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
    if (model.moe_arena_size()) {
        // Arena uploads stop at each expert's role boundary. Reserving a full
        // 16 MiB per staging slot wastes pinned RAM when no piece can use it.
        size_t largest_role = 0;
        for (const auto & e : exps) {
            largest_role = std::max(largest_role, e.t->nb[1] * e.t->ne[1]);
        }
        bank_stage   = b1_env_int("LLAMA_B1_BANK_STAGE", 0) != 0;
        if (!bank_stage) {
            piece_bytes = std::min(piece_bytes, largest_role);
        }
        direct       = b1_env_int("LLAMA_B1_DIRECT", 0) != 0;
        direct_piece = (size_t) std::clamp<int64_t>(b1_env_int("LLAMA_B1_DIRECT_PIECE_MIB", 64), 1, 1024) << 20;
    }

    size_t staging_alignment = 1;
    for (const auto & e : exps) { staging_alignment = std::max(staging_alignment, e.src.alignment); }
    stg.resize(n_workers);
    for (auto & w : stg) {
        w.resize(2);
        for (auto & s : w) {
            s.buf = ggml_backend_buft_alloc_buffer(buft_host, piece_bytes + 3 * staging_alignment);
            s.ev  = ggml_backend_event_new(dev);
            if (s.buf == nullptr || s.ev == nullptr) {
                LLAMA_LOG_WARN("%s: [B1] cannot allocate the pinned staging buffers, B1 is off\n", __func__);
                return;
            }
            s.data = (char *) b1_align_up((uintptr_t) ggml_backend_buffer_get_base(s.buf), staging_alignment);
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
            "slot %.1f MiB, %d copy threads x 2 x %.2f MiB pinned%s, %zu resident tensors lendable\n", __func__,
            exps.size(), roles.size(), n_res, n_layer, slot_bytes/1048576.0, n_workers, piece_bytes/1048576.0,
            direct ? " (arena pieces copied direct, LLAMA_B1_DIRECT)" : "",
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

std::vector<ggml_backend_buffer_ptr> llama_b1::spare_views(size_t min_bytes, size_t max_views) const {
    struct span { ggml_backend_buffer_t buf; size_t offs; size_t size; };
    std::vector<span> spans;
    const size_t n_lent = regions.size() - (fresh_buf ? 1 : 0);
    const size_t n_roles = roles.size();
    const size_t align = buft_dev ? ggml_backend_buft_get_alignment(buft_dev) : 0;
    if (!align || slot_place.size() != 2*n_roles || carry_place.size() != g_ub) {
        return {};
    }
    for (size_t r = 0; r < n_lent; ++r) {
        const region & rg = regions[r];
        std::vector<std::pair<size_t, size_t>> used; // [begin, end) within the region
        for (size_t i = 0; i < slot_place.size(); ++i) {
            if (slot_place[i].region == (int) r) {
                used.push_back({ slot_place[i].offs, slot_place[i].offs + role_max[i % n_roles] });
            }
        }
        for (size_t j = 0; j < carry_place.size(); ++j) {
            if (carry_place[j].region != (int) r) {
                continue;
            }
            uint32_t tokens = 0;
            for (size_t u = j; u < ub_n.size(); u += g_ub) {
                tokens = std::max(tokens, ub_n[u]);
            }
            used.push_back({ carry_place[j].offs, carry_place[j].offs + (size_t) tokens * n_embd * n_hc * sizeof(float) });
        }
        std::sort(used.begin(), used.end());
        used.push_back({ rg.size, rg.size });
        const size_t base = rg.base - (char *) ggml_backend_buffer_get_base(rg.buf);
        size_t at = 0;
        for (const auto & u : used) {
            const size_t begin = (base + at + align - 1) / align * align;
            const size_t end   = (base + std::min(u.first, rg.size)) / align * align;
            if (end > begin && end - begin >= min_bytes) {
                spans.push_back({ rg.buf, begin, end - begin });
            }
            at = std::max(at, u.second);
        }
    }
    std::sort(spans.begin(), spans.end(), [](const span & a, const span & b) { return a.size > b.size; });
    if (spans.size() > max_views) {
        spans.resize(max_views);
    }
    std::vector<ggml_backend_buffer_ptr> out;
    for (const auto & sp : spans) {
        ggml_backend_buffer_ptr v(ggml_backend_buffer_view(sp.buf, sp.offs, sp.size));
        if (v) {
            out.push_back(std::move(v));
        }
    }
    return out;
}

void llama_b1::release_plan() {
    GGML_ASSERT(workers.empty() && !coord.joinable());
    // fork: only an explicit keep request may retain clean pageable bank weights.
    const bool release_kept = !b1_env_int("LLAMA_B1_KEEP_BANK_RAM", 0) ||
                              b1_env_int("LLAMA_B1_RELEASE_BANK_RAM", 0) != 0;
    size_t kept_bytes = 0;
    size_t freed_bytes = 0;
    for (size_t ti = 0; ti < exps.size(); ++ti) {
        auto & kept = exps[ti].kept;
        if (kept.empty()) {
            continue;
        }
        GGML_ASSERT(ti < dirty_flag.size());
        if (release_kept && !dirty_flag[ti]) {
            // fork: joined workers no longer read this pageable copy; VRAM holds the weights.
            freed_bytes += kept.size();
            std::vector<char>().swap(kept);
        }
        kept_bytes += kept.size();
    }
    LLAMA_LOG_INFO("%s: [B1] bank RAM: released %zu bytes, retained %zu bytes\n", __func__, freed_bytes, kept_bytes);
    reserve_layer_restore();
    ffn_release();
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
    final_home_ = false;
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

size_t llama_b1::scratch_bytes(const std::vector<uint32_t> & ub_tokens, uint32_t group_ub) const {
    if (ub_tokens.empty()) {
        return 0;
    }
    const size_t align = std::max<size_t>(256, ggml_backend_buft_get_alignment(buft_dev));
    const size_t group = group_ub ? std::min<size_t>(group_ub, ub_tokens.size()) : ub_tokens.size();
    size_t bytes = 0;
    for (size_t lane = 0; lane < group; ++lane) {
        uint32_t tokens = 0;
        for (size_t u = lane; u < ub_tokens.size(); u += group) {
            tokens = std::max(tokens, ub_tokens[u]);
        }
        bytes += b1_align_up((size_t) tokens * n_embd * n_hc * sizeof(float), align);
    }
    for (size_t role : role_max) {
        bytes += 2 * b1_align_up(role, align);
    }
    return bytes;
}

bool llama_b1::begin(const std::vector<uint32_t> & ub_tokens, uint32_t group_ub_req, size_t lend_cap, bool final_home, size_t * borrow_bytes, bool defer_streaming) {
    GGML_ASSERT(init_ok);
    if (borrow_bytes) {
        *borrow_bytes = 0;
    }
    stop_threads();
    release_plan();
    st = {};
    final_home_ = final_home;

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
    size_t phase_scratch = ffn_requested ? (size_t) 4096 << 20 : 0;
    if (ffn_requested && b1_env_int("LLAMA_B1_FFN_MMID", 1) != 0) {
        const size_t rows = (size_t) std::clamp<int64_t>(b1_env_int("LLAMA_B1_FFN_ROUTED_ROWS", 32768), 8192, 65536);
        uint32_t hidden = 0;
        for (int il = 0; il < n_layer; ++il) {
            hidden = std::max(hidden, model.hparams.n_ff_exp(il));
        }
        // Bound gathered input/output, transform copies, gate/up/GLU temporaries,
        // scalar scales and MMQ quantization storage before borrowing the cache.
        phase_scratch = std::max(phase_scratch, rows * (4*n_embd + 8*hidden + 512) * sizeof(float));
    }
    const size_t keep_default = ffn_requested ? phase_scratch >> 20 : 1536;
    const size_t keep = std::max(phase_scratch, (size_t) std::max<int64_t>(0,
            b1_env_int("LLAMA_B1_KEEP_FREE_MIB", keep_default)) << 20);
    const size_t fresh_avail = free_mem > keep + extra_keep ? free_mem - keep - extra_keep : 0;

    // page-cache budget: every pass reads the streamed experts in the same fixed order, the experts kept in RAM from
    // the mmap and the lent layers from the file. When that loop is larger than the RAM free for the page cache,
    // almost every read misses and goes to the disk. So lend only what fits next to the experts kept in RAM (they
    // count only when they are mmap'd file pages). If even those do not fit, the loop misses anyway and fewer passes
    // read less, so there is no cap. Off by default (measured at about 37 GiB MemAvailable: the capped lend split
    // the call into one-ubatch passes and read as much from the disk as the uncapped lend); LLAMA_B1_RAM_MARGIN_MIB
    // >= 0 turns it on with that margin.
    const int64_t ram_margin = b1_env_int("LLAMA_B1_RAM_MARGIN_MIB", -1);
    const int64_t ram_avail  = ram_margin >= 0 ? b1_ram_available() : -1;
    int64_t ram_cap = -1;
    if (ram_avail >= 0) {
        int64_t host_bytes = 0;
        int     paged      = -1; // all host experts come from the same loader: test the first one only
        for (const auto & e : exps) {
            if (!e.resident) {
                paged = paged < 0 ? (int) b1_file_backed(e.home_data) : paged;
                host_bytes += paged ? (int64_t) e.nbytes : 0;
            }
        }
        const int64_t room = ram_avail - (ram_margin << 20) - host_bytes;
        if (room >= 0) {
            ram_cap  = room;
            lend_cap = std::min(lend_cap, (size_t) room);
        }
    }

    // lendable layers, highest first; a layer still holding borrowed data is always lent. A layer lends its
    // VRAM-resident exps tensors only (a mixed layer, such as RAM experts plus an always-on VRAM bank, lends the bank);
    // its RAM tensors stream from RAM as before.
    // Mixed layers lend only with an expert bank loaded or LLAMA_B1_LEND_MIXED=1; otherwise only fully resident
    // layers lend, as before.
    const bool lend_mixed = ggml_moe_bank_any() || b1_env_int("LLAMA_B1_LEND_MIXED", 0) != 0;
    std::vector<int> forced;
    std::vector<int> order;
    for (int il = n_layer - 1; il >= 0; --il) {
        const auto & L = layers[il];
        if (L.res_bytes == 0 || (!L.resident && !lend_mixed)) {
            continue;
        }
        bool any_dirty = false;
        bool lendable  = true;
        for (int ti : L.tensors) {
            if (!exps[ti].resident) {
                continue;
            }
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
    size_t minimum_fresh = SIZE_MAX;

    // the smallest lend (fewest layers, highest first) that fits g ubatches per group; false when none does
    auto try_group = [&](size_t g) -> bool {
        std::vector<size_t> items;
        for (int s = 0; s < 2; ++s) {
            for (size_t r = 0; r < n_roles; ++r) {
                items.push_back(role_max[r]);
            }
        }
        for (size_t j = 0; j < g; ++j) {
            uint32_t tokens = 0;
            for (size_t u = j; u < nub; u += g) {
                tokens = std::max(tokens, ub_n[u]);
            }
            items.push_back(b1_align_up((size_t) tokens * n_embd * n_hc * sizeof(float), align));
        }

        size_t ffn_bytes = 0;
        if (ffn_requested && g == nub) {
            size_t tokens = 0;
            for (uint32_t n : ub_n) {
                tokens += n;
            }
            ffn_bytes = 2 * b1_align_up(tokens * n_embd * sizeof(float), align) +
                b1_align_up(tokens * (n_hc + 2 * model.hparams.n_expert_used()) * sizeof(float), align);
        }

        for (size_t k = 0; k <= order.size(); ++k) {
            std::vector<int> lent = forced;
            size_t lent_bytes = 0;
            for (int il : forced) {
                lent_bytes += layers[il].res_bytes;
            }
            bool over_cap = false;
            for (size_t i = 0; i < k; ++i) {
                lent_bytes += layers[order[i]].res_bytes;
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
                    if (!e.resident) {
                        continue;
                    }
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

            minimum_fresh = std::min(minimum_fresh, fresh + ffn_bytes);
            if (fresh <= fresh_avail && ffn_bytes <= fresh_avail - fresh) {
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
        // fork: ask the cache only for space the full group cannot borrow from resident weights.
        if (borrow_bytes && g == g_max && minimum_fresh != SIZE_MAX) {
            *borrow_bytes = b1_borrow_deficit(minimum_fresh, keep, free_mem);
            if (*borrow_bytes > 0) {
                return false;
            }
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
            if (!e.resident || verified[ti] || dirty_flag[ti]) {
                continue;
            }
            const size_t n = std::min<size_t>(e.nbytes, 1u << 16);
            std::vector<uint8_t> a(n);
            std::vector<uint8_t> b(n);
            ggml_backend_tensor_get(e.t, a.data(), e.nbytes - n, n);
            if (stg[0][0].used) { ggml_backend_event_synchronize(stg[0][0].ev); }
            if (!b1_read_file(e.src.fd, b.data(), e.src.offs + e.nbytes - n, n, e.src.alignment, stg[0][0].data) || memcmp(a.data(), b.data(), n) != 0) {
                LLAMA_LOG_WARN("%s: [B1] %s: VRAM bytes differ from the file, it cannot be lent; using the normal prefill\n",
                        __func__, ggml_get_name(e.t));
                return false;
            }
            verified[ti] = true;
        }
    }

    // fork: opt in to pageable bank copies; the default uses the file and bounded staging.
    if (b1_env_int("LLAMA_B1_KEEP_BANK_RAM", 0)) {
        size_t kept_bytes = 0;
        for (int il : lent_sel) {
            for (int ti : layers[il].tensors) {
                auto & e = exps[ti];
                if (!e.resident || !e.kept.empty()) {
                    continue;
                }
                const int64_t available = b1_ram_available();
                constexpr int64_t reserve = 16LL << 30;
                if (available < reserve || e.nbytes > (uint64_t) (available - reserve)) {
                    LLAMA_LOG_WARN("%s: [B1] RAM reserve prevents keeping %s; using file reads\n", __func__, ggml_get_name(e.t));
                    continue;
                }
                try {
                    e.kept.resize(e.nbytes);
                } catch (const std::bad_alloc &) {
                    LLAMA_LOG_WARN("%s: [B1] cannot keep %s in RAM; using file reads\n", __func__, ggml_get_name(e.t));
                    continue;
                }
                // Dirty homes hold scratch, so recover their immutable bytes from the file.
                if (dirty_flag[ti]) {
                    if (!b1_read_file(e.src.fd, e.kept.data(), e.src.offs, e.nbytes, e.src.alignment)) {
                        std::vector<char>().swap(e.kept);
                        release_plan();
                        return false;
                    }
                } else {
                    ggml_backend_tensor_get(e.t, e.kept.data(), 0, e.nbytes);
                }
                kept_bytes += e.nbytes;
            }
        }
        LLAMA_LOG_INFO("%s: [B1] kept %.2f GiB of lent weights in pageable RAM\n", __func__, kept_bytes/1073741824.0);
    }

    if (fresh_sel > 0) {
        fresh_buf = ggml_backend_buft_alloc_buffer(buft_dev, fresh_sel);
        if (fresh_buf == nullptr) {
            LLAMA_LOG_WARN("%s: [B1] cannot allocate %.1f MiB of VRAM; using the normal prefill\n", __func__, fresh_sel/1048576.0);
            release_plan();
            return false;
        }
        ggml_backend_buffer_set_usage(fresh_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    g_ub = g_sel;
    const size_t n_regions = regs_sel.size() + (fresh_buf ? 1 : 0);

    ggml_init_params ip = {
        /*.mem_size   =*/ (n_regions + 7*nub + 20) * ggml_tensor_overhead(),
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
        } else if (i < 2*n_roles + g_sel) {
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

    if (ffn_requested && g_sel == nub) {
        uint32_t tokens = 0;
        for (uint32_t n : ub_n) {
            tokens += n;
        }
        const uint32_t used = model.hparams.n_expert_used();
        const size_t sizes[] = {
            b1_align_up((size_t) tokens * n_embd * sizeof(float), align),
            b1_align_up((size_t) tokens * n_embd * sizeof(float), align),
            b1_align_up((size_t) tokens * (n_hc + 2*used) * sizeof(float), align),
        };
        for (size_t bytes : sizes) {
            auto * buf = ggml_backend_buft_alloc_buffer(buft_dev, bytes);
            if (!buf) {
                LLAMA_LOG_WARN("%s: [B1] packed FFN allocation failed; use normal prefill\n", __func__);
                release_plan();
                return false;
            }
            ffn_buffers.push_back(buf);
            st.bytes_fresh += bytes;
        }
        ffn_all = std::make_unique<llm_b1_ffn>();
        auto make = [&](ggml_type type, int64_t d0, int64_t d1, size_t item, size_t offset) {
            auto * t = ggml_new_tensor_2d(ctx_meta, type, d0, d1);
            t->buffer = ffn_buffers[item];
            t->data = (char *) ggml_backend_buffer_get_base(t->buffer) + offset;
            return t;
        };
        ffn_all->input = make(GGML_TYPE_F32, n_embd, tokens, 0, 0);
        ffn_all->output = make(GGML_TYPE_F32, n_embd, tokens, 1, 0);
        ffn_all->inject = make(GGML_TYPE_F32, n_hc, tokens, 2, 0);
        ffn_all->ids = make(GGML_TYPE_I32, used, tokens, 2, (size_t) tokens * n_hc * sizeof(float));
        ffn_all->weights = make(GGML_TYPE_F32, used, tokens, 2, (size_t) tokens * (n_hc + used) * sizeof(float));
        ffn_all->max_rows = (uint32_t) std::clamp<int64_t>(b1_env_int("LLAMA_B1_FFN_MAX_ROWS", 8192), 256, 16384);
        ffn_all->grouped_mmid = b1_env_int("LLAMA_B1_FFN_MMID", 1) != 0 && model.hparams.n_expert <= 1024;
        ffn_all->routed_rows = (uint32_t) std::clamp<int64_t>(b1_env_int("LLAMA_B1_FFN_ROUTED_ROWS", 32768), 8192, 65536);
        uint32_t offset = 0;
        for (uint32_t n : ub_n) {
            auto view = std::make_unique<llm_b1_ffn>();
            auto slice = [&](ggml_tensor * base) {
                auto * t = ggml_new_tensor_2d(ctx_meta, base->type, base->ne[0], n);
                t->buffer = base->buffer;
                t->data = (char *) base->data + offset * base->nb[1];
                return t;
            };
            view->input = slice(ffn_all->input);
            view->output = slice(ffn_all->output);
            view->inject = slice(ffn_all->inject);
            view->ids = slice(ffn_all->ids);
            view->weights = slice(ffn_all->weights);
            ffn_views.push_back(std::move(view));
            offset += n;
        }
        LLAMA_LOG_INFO("%s: [B1] whole-prompt expert GEMMs enabled: %u tokens, %.1f MiB routing scratch, max %u rows per expert piece\n",
                __func__, tokens, (double) tokens * (2*n_embd + n_hc + 2*used) * sizeof(float) / 1048576.0, ffn_all->max_rows);
    }

    lent_layer.assign(n_layer, false);
    for (int il : lent_sel) {
        lent_layer[il] = true;
        for (int ti : layers[il].tensors) {
            if (!exps[ti].resident) {
                continue;
            }
            if (!dirty_flag[ti]) {
                dirty_flag[ti] = true;
                dirty_tensors.push_back(ti);
            }
            st.bytes_lent += exps[ti].alloc;
        }
    }
    st.bytes_fresh += fresh_sel;
    for (size_t j = 0; j < g_sel; ++j) {
        uint32_t tokens = 0;
        for (size_t u = j; u < nub; u += g_sel) {
            tokens = std::max(tokens, ub_n[u]);
        }
        st.bytes_carry += b1_align_up((size_t) tokens * n_embd * n_hc * sizeof(float), align);
    }

    // the streamed layers in call order
    const size_t n_grp = (nub + g_sel - 1)/g_sel;
    grp_first.clear();
    for (size_t g = 0; g < n_grp; ++g) {
        grp_first.push_back(g*g_sel);
        for (int il = 0; il < n_layer; ++il) {
            if (layer_streamed(il)) {
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
            "free before %.1f MiB, RAM free %.1f MiB, RAM lend cap %.1f MiB, last layer %s\n", __func__,
            nub, max_tok, g_fit, g_max, n_grp, g_sel, steps.size(), lent_sel.size(), st.bytes_lent/1048576.0,
            lend_cap == SIZE_MAX ? -1.0 : lend_cap/1048576.0, fresh_sel/1048576.0, st.bytes_carry/1048576.0,
            free_mem/1048576.0, ram_avail < 0 ? -1.0 : ram_avail/1048576.0, ram_cap < 0 ? -1.0 : ram_cap/1048576.0,
            layer_streamed(n_layer - 1) ? "streamed" : "not streamed");

    pass_start_us.assign(n_grp, 0);
    pass_wall_us.assign(n_grp, 0);
    pass_compute_us.assign(n_grp, 0);
    pass_copy_ms.assign(n_grp, 0);
    pass_wait_ms.assign(n_grp, 0);
    pass_bytes.assign(n_grp, 0);
    if (!defer_streaming) {
        start_streaming();
    }
    return true;
}

void llama_b1::start_streaming() {
    GGML_ASSERT(reserve_layer < 0 && cur_step < 0 && workers.empty() && !coord.joinable());
    GGML_ASSERT(!steps.empty());
    stop = false;
    io_error = false;
    for (int w = 0; w < n_workers; ++w) {
        workers.emplace_back(&llama_b1::worker, this, w);
        llama_thread_off_caller_cpu(workers.back());
    }
    coord = std::thread(&llama_b1::coordinator, this);
    llama_thread_off_caller_cpu(coord);
}

void llama_b1::reserve_layer_bind(int il) {
    GGML_ASSERT(il >= 0 && il < n_layer && reserve_layer < 0 && cur_step < 0);
    GGML_ASSERT(workers.empty() && !coord.joinable());
    reserve_layer = il;
    for (int ti : layers[il].tensors) {
        auto & e = exps[ti];
        if (tensor_streamed(il, e) && layer_streamed(il)) {
            const placement & p = slot_place[e.role]; // real slot zero, contents are never read
            e.t->buffer = regions[p.region].buf;
            e.t->data = regions[p.region].base + p.offs;
            e.t->nb[2] = e.t->nb[1] * e.t->ne[1];
            e.t->nb[3] = e.t->nb[2] * e.t->ne[2];
        }
    }
}

void llama_b1::reserve_layer_restore() {
    if (reserve_layer >= 0) {
        for (int ti : layers[reserve_layer].tensors) {
            auto & e = exps[ti];
            e.t->buffer = e.home_buf;
            e.t->data = e.home_data;
            e.t->nb[2] = e.home_nb2;
            e.t->nb[3] = e.home_nb2 * e.t->ne[2];
        }
        reserve_layer = -1;
    }
}

bool llama_b1::layer_streamed(int il) const {
    if (il < 0 || il >= n_layer || layers[il].tensors.empty()) {
        return false;
    }
    if (lent_layer[il]) {
        return true;
    }
    // the last layer's FFN runs on the output rows only; with few of them the normal path computes its RAM experts
    // on the CPU, so they stay home and the layer is not streamed
    return !layers[il].resident && !(final_home_ && il == n_layer - 1);
}

bool llama_b1::host_rows_on_gpu(int64_t rows) const {
    for (const auto & e : exps) {
        if (e.resident) {
            continue;
        }
        // the batch size the device's offload test reads from a MUL_MAT_ID node: ne[2] rows
        ggml_tensor op = {};
        op.op     = GGML_OP_MUL_MAT_ID;
        op.type   = GGML_TYPE_F32;
        op.ne[0]  = e.t->ne[1];
        op.ne[1]  = 1;
        op.ne[2]  = rows;
        op.ne[3]  = 1;
        op.src[0] = e.t;
        return ggml_backend_dev_offload_op(dev, &op);
    }
    return true;
}

void llama_b1::ubatch_home(int il, bool home) {
    if (cur_step < 0 || steps[cur_step].il != il) {
        return;
    }
    const int    slot    = steps[cur_step].slot;
    const size_t n_roles = roles.size();
    for (int ti : layers[il].tensors) {
        auto & e = exps[ti];
        // a lent tensor's home VRAM holds scratch now; its slot copy runs the same kernels the home copy would
        if (e.resident) {
            continue;
        }
        if (home) {
            e.t->buffer = e.home_buf;
            e.t->data   = e.home_data;
            e.t->nb[2] = e.home_nb2;
            e.t->nb[3] = e.home_nb2 * e.t->ne[2];
        } else {
            const placement & p = slot_place[slot*n_roles + e.role];
            e.t->buffer = regions[p.region].buf;
            e.t->data   = regions[p.region].base + p.offs;
            e.t->nb[2] = e.t->nb[1] * e.t->ne[1];
            e.t->nb[3] = e.t->nb[2] * e.t->ne[2];
        }
    }
}

int32_t llama_b1::final_l0() const {
    int32_t l0 = n_layer - 1;
    while (l0 > 0 && !layer_streamed(l0) && !layer_streamed(l0 - 1)) {
        --l0;
    }
    return l0;
}

int32_t llama_b1::range_end(int32_t l0) const {
    if (ffn_enabled() || layer_streamed(l0)) {
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
    const size_t expert_bytes = e.t->nb[1] * e.t->ne[1];
    size_t n_holes = 0;
    (void) ggml_moe_bank_holes(&n_holes);
    // Bulk hole uploads already split cold spans from zero-filled bank bytes in resident_copy.
    // Let them span experts to avoid one worker/registry/stream-lock round trip per expert.
    // Permanent GPU spans still need expert boundaries for resident_lookup; strided sources do too.
    const bool bulk_holes = n_holes > 0 && (bank_stage || b1_env_int("GGML_MOE_BANK_HOLES_BULK", 0) == 1);
    const bool permanent = !from_file && (ggml_moe_cache_has_residents() || (n_holes > 0 && !bulk_holes) ||
            b1_env_int("LLAMA_PP_KEEP_EXPERTS", 0) != 0);
    for (size_t off = 0; off < e.nbytes;) {
        piece p;
        const bool strided = !from_file && e.home_nb2 != expert_bytes;
        // per-expert pieces: a resident or dropped bank expert is then one whole piece for the device-side copy
        const size_t source_off = strided ? (off / expert_bytes) * e.home_nb2 + off % expert_bytes : off;
        p.src_host = from_file ? nullptr : (const char *) e.home_data + source_off;
        p.src_kept = from_file && !e.kept.empty() ? e.kept.data() + off : nullptr;
        p.fd       = e.src.fd;
        p.alignment = e.src.alignment;
        p.src_offs = e.src.offs + off;
        p.dst      = dst;
        p.dst_offs = dst_offs + off;
        p.len      = std::min(direct && !from_file && !bulk_holes ? direct_piece : piece_bytes, e.nbytes - off);
        if (strided || permanent) { p.len = std::min(p.len, expert_bytes - off % expert_bytes); }
        ps.push_back(p);
        off += p.len;
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
            push_tensor_pieces(ps, e, &regions[p.region], p.offs, lent_layer[il] && e.resident);
        }
        const size_t pass = k / (steps.size() / grp_first.size());
        for (const auto & piece : ps) {
            pass_bytes[pass] += piece.len;
        }
        const int64_t t_copy0 = ggml_time_us();
        if (!run_pieces(std::move(ps))) {
            std::lock_guard<std::mutex> lk(mtx);
            io_error = true;
            cv.notify_all();
            return;
        }

        const double copy_ms = (ggml_time_us() - t_copy0) / 1000.0;
        st.t_copy_submit_ms += copy_ms;
        pass_copy_ms[pass] += copy_ms;
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

static const ggml_moe_bank_hole * b1_first_hole(uintptr_t begin, const ggml_moe_bank_hole ** end) {
    size_t n = 0;
    const ggml_moe_bank_hole * holes = ggml_moe_bank_holes(&n);
    *end = holes ? holes + n : nullptr;
    return holes ? std::lower_bound(holes, holes + n, begin,
            [](const ggml_moe_bank_hole & h, uintptr_t v) { return h.end <= v; }) : nullptr;
}

static bool b1_touches_holes(const char * src, size_t len) {
    const ggml_moe_bank_hole * end = nullptr;
    const ggml_moe_bank_hole * it = b1_first_hole((uintptr_t) src, &end);
    return it && it != end && it->begin < (uintptr_t) src + len;
}

static void b1_gather_holes(char * dst, const char * src, size_t len) {
    const uintptr_t begin = (uintptr_t) src, last = begin + len;
    const ggml_moe_bank_hole * end = nullptr;
    uintptr_t cursor = begin;
    for (const ggml_moe_bank_hole * it = b1_first_hole(begin, &end); it && it != end && it->begin < last; ++it) {
        const uintptr_t first = std::max(begin, it->begin), stop = std::min(last, it->end);
        if (cursor < first) { memcpy(dst + (cursor - begin), (const void *) cursor, first - cursor); }
        memset(dst + (first - begin), 0, stop - first);
        cursor = stop;
    }
    if (cursor < last) { memcpy(dst + (cursor - begin), (const void *) cursor, last - cursor); }
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

        if (p.src_host && bank_stage && p.len <= piece_bytes && b1_touches_holes(p.src_host, p.len)) {
            auto & s = stg[w][b];
            b ^= 1;
            if (s.used) {
                ggml_backend_event_synchronize(s.ev);
            }
            // read only the kept bytes (dropped ranges were released) and zero the rest, outside the stream lock
            b1_gather_holes(s.data, p.src_host, p.len);
            {
                std::lock_guard<std::mutex> ls(mtx_stream);
                ggml_backend_tensor_set_async(backend_copy, p.dst->raw, s.data, p.dst_offs, p.len);
                ggml_backend_event_record(s.ev, backend_copy);
                s.used = true;
            }
            std::lock_guard<std::mutex> lk(mtx);
            st.bytes_h2d += p.len;
            if (--pieces_left == 0) { cv.notify_all(); }
            continue;
        }

        if (p.src_host) {
            std::lock_guard<std::mutex> ls(mtx_stream);
            if (ggml_moe_cache_resident_copy(backend_copy, p.dst->raw, p.dst_offs, p.src_host, p.len)) {
                std::lock_guard<std::mutex> lk(mtx);
                st.bytes_d2d += p.len;
                if (--pieces_left == 0) { cv.notify_all(); }
                continue;
            }
        }

        if (p.src_host && direct) {
            // the arena is registered with CUDA: the DMA engine reads it in place
            ggml_moe_cache_host_restore(p.src_host, p.len);
            {
                std::lock_guard<std::mutex> ls(mtx_stream);
                ggml_backend_tensor_set_async(backend_copy, p.dst->raw, p.src_host, p.dst_offs, p.len);
            }
            std::lock_guard<std::mutex> lk(mtx);
            st.bytes_h2d += p.len;
            if (--pieces_left == 0) { cv.notify_all(); }
            continue;
        }

        auto & s = stg[w][b];
        b ^= 1;
        if (s.used) {
            ggml_backend_event_synchronize(s.ev);
        }

        bool ok = true;
        if (p.src_kept) {
            memcpy(s.data, p.src_kept, p.len);
        } else if (p.src_host) {
            ggml_moe_cache_host_restore(p.src_host, p.len);
            memcpy(s.data, p.src_host, p.len);
        } else {
            ok = p.fd >= 0 && b1_read_file(p.fd, s.data, p.src_offs, p.len, p.alignment);
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
    const double wait_ms = (ggml_time_us() - t0)/1000.0;
    st.t_wait_ms += wait_ms;
    pass_wait_ms[k / (steps.size() / grp_first.size())] += wait_ms;

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
        e.t->nb[2] = e.t->nb[1] * e.t->ne[1];
        e.t->nb[3] = e.t->nb[2] * e.t->ne[2];
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
        e.t->nb[2] = e.home_nb2;
        e.t->nb[3] = e.home_nb2 * e.t->ne[2];
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

void llama_b1::pass_begin(size_t ub) {
    pass_start_us[ub / g_ub] = ggml_time_us();
}

void llama_b1::pass_compute(size_t ub, int64_t host_us) {
    pass_compute_us[ub / g_ub] += host_us;
}

void llama_b1::pass_end(size_t ub) {
    ggml_backend_synchronize(backend_gpu);
    pass_wall_us[ub / g_ub] = ggml_time_us() - pass_start_us[ub / g_ub];
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

    LLAMA_LOG_INFO("%s: [B1] permanent expert copies: D2D_bytes=%llu H2D_bytes=%llu\n", __func__,
            (unsigned long long)st.bytes_d2d, (unsigned long long)st.bytes_h2d);
    LLAMA_LOG_INFO("%s: [B1] pass timing: pipeline wall %.3f ms, overlapped copy submission %.3f ms, exposed host copy waits %.3f ms; overlapping times are not additive\n",
            __func__, st.t_stream_us / 1000.0, st.t_copy_submit_ms, st.t_wait_ms);
    for (size_t pass = 0; pass < grp_first.size(); ++pass) {
        uint64_t tokens = 0;
        const size_t first = grp_first[pass], last = group_end(first);
        for (size_t u = first; u < last; ++u) {
            tokens += ub_n[u];
        }
        LLAMA_LOG_INFO("%s: [B1] pass %zu/%zu: %llu tokens, %zu ubatches, %.3f GiB streamed; start_us=%lld end_us=%lld, pipeline wall %.3f ms, graph-call host %.3f ms, copy-pipeline host %.3f ms, exposed copy wait %.3f ms; overlap is not additive, graph-call time is not pure GPU compute\n",
                __func__, pass + 1, grp_first.size(), (unsigned long long) tokens, last - first,
                pass_bytes[pass] / 1073741824.0, (long long) pass_start_us[pass], (long long) (pass_start_us[pass] + pass_wall_us[pass]),
                pass_wall_us[pass] / 1000.0, pass_compute_us[pass] / 1000.0,
                pass_copy_ms[pass], pass_wait_ms[pass]);
    }
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

llm_b1_ffn * llama_b1::ffn(size_t ub) {
    return ub < ffn_views.size() ? ffn_views[ub].get() : nullptr;
}

bool llama_b1::ffn_gather() {
    if (!ffn_all) {
        return false;
    }
    ggml_backend_synchronize(backend_gpu);
    const uint32_t tokens = (uint32_t) ffn_all->ids->ne[1];
    const uint32_t used = (uint32_t) ffn_all->ids->ne[0];
    const uint32_t experts = model.hparams.n_expert;
    std::vector<int32_t> ids((size_t) tokens * used);
    std::vector<float> weights(ids.size());
    ggml_backend_tensor_get(ffn_all->ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(ffn_all->weights, weights.data(), 0, weights.size() * sizeof(float));
    const bool hot_bank = std::any_of(model.layers.begin(), model.layers.end(),
            [](const llama_layer & layer) { return layer.ffn_exps_hot_map != nullptr; });
    ffn_all->expert_rows.assign(experts, {});
    ffn_all->expert_weights.assign(experts, {});
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t k = 0; k < used; ++k) {
            const size_t i = (size_t) t * used + k;
            const int32_t e = ids[i];
            if (e < 0 || (uint32_t) e >= experts || !std::isfinite(weights[i])) {
                return false;
            }
            // Each token may select an expert at most once; SET_ROWS relies on unique row indices.
            for (uint32_t j = 0; j < k; ++j) {
                if (ids[(size_t) t * used + j] == e) {
                    return false;
                }
            }
            // a hot-bank expert in the cold bank is routed with weight exactly 0 and adds nothing: skip its GEMM row
            if (hot_bank && weights[i] == 0.0f) {
                continue;
            }
            ffn_all->expert_rows[e].push_back((int32_t) t);
            ffn_all->expert_weights[e].push_back(weights[i]);
        }
    }
    ffn_all->buckets.clear();
    if (ffn_all->grouped_mmid) {
        const size_t cap = ffn_all->routed_rows;
        for (size_t e = 0; e < experts; ++e) {
            const auto & rows = ffn_all->expert_rows[e];
            const auto & weights = ffn_all->expert_weights[e];
            for (size_t offset = 0; offset < rows.size(); ) {
                const size_t count = std::min(cap, rows.size() - offset);
                if (ffn_all->buckets.empty() || ffn_all->buckets.back().rows.size() + count > cap ||
                        ffn_all->buckets.back().spans.size() >= 256) {
                    ffn_all->buckets.emplace_back();
                }
                auto & bucket = ffn_all->buckets.back();
                bucket.spans.push_back({bucket.rows.size(), count});
                bucket.rows.insert(bucket.rows.end(), rows.begin() + offset, rows.begin() + offset + count);
                bucket.weights.insert(bucket.weights.end(), weights.begin() + offset, weights.begin() + offset + count);
                bucket.experts.insert(bucket.experts.end(), count, (int32_t) e);
                offset += count;
            }
        }
    }
    return true;
}

void llama_b1::ffn_release() {
    if (ffn_buffers.empty()) {
        return;
    }
    ggml_backend_synchronize(backend_gpu);
    ffn_views.clear();
    ffn_all.reset();
    for (auto * buffer : ffn_buffers) {
        ggml_backend_buffer_free(buffer);
    }
    ffn_buffers.clear();
}
