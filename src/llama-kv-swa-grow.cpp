// fork (SWA-GROW): a window (SWA) cache sized n_swa + n_ubatch for the prompt ubatches runs decode in n_swa + a few
// rows. LLAMA_INK_SWA_GROW=1 (Inkling small cards, with CACHELOAN) re-allocates the window cache at the decode size
// when the cache loan grows and back at the boot size before the next prompt; the VRAM it frees goes to the MoE cache.
// Off (unset or 0): nothing here runs.

#include "llama-kv-cache.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

size_t llama_kv_cache::swa_cell_bytes() const {
    size_t res = 0;
    for (const auto & layer : layers) {
        res += layer.k ? layer.k->nb[1] : 0;
        res += layer.v ? layer.v->nb[1] : 0;
    }
    return res;
}

const char * llama_kv_cache::swa_resize_why_not() const {
    if (ctxs_bufs.no_alloc) {
        return "no-alloc cache";
    }
    if (swa_type == LLAMA_SWA_TYPE_NONE || n_swa == 0) {
        return "not a window cache";
    }
    if (sj_kvarn.enabled()) {
        return "SJ-KVaRN window cache";
    }
    if (n_stream != 1) {
        return "more than one KV stream";
    }
    if (v_trans) {
        return "V transposed (flash attention off)";
    }
    if (other != nullptr) {
        return "the cache reads another cache's tensors";
    }
    if (kv_stream_runtime.runtime != nullptr) {
        return "block KV streaming";
    }
    if (scratch.user != nullptr) {
        return "the KV tail is lent as scratch";
    }
    if (layers.empty()) {
        return "no layers";
    }
    if (v_cells_impl.use_count() != 1) {
        return "the cell list is shared with another context";
    }
    if (ctxs_bufs.bufs.size() != 1) {
        return "more than one KV buffer";
    }
    const auto & [ctx, buf] = ctxs_bufs.bufs[0];
    if (!buf || ggml_backend_buffer_is_host(buf.get())) {
        return "the KV buffer is not on the GPU";
    }
    for (const auto & layer : layers) {
        if (!layer.k || !layer.v || layer.shared || layer.body || layer.k->buffer != buf.get() || layer.v->buffer != buf.get() ||
                layer.k->ne[2] != 1 || layer.v->ne[2] != 1 || layer.k_stream.size() != 1 || layer.v_stream.size() != 1) {
            return "a layer with shared, paged or multi-stream K/V";
        }
    }
    // the buffer holds the layers' K/V (and their stream views) and nothing else
    size_t n_own = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
        n_own += t->view_src == nullptr ? 1 : 0;
    }
    if (n_own != 2*layers.size()) {
        return "the KV buffer holds more than K/V (turbo rotation or other tensors)";
    }
    return nullptr;
}

bool llama_kv_cache::swa_plan(uint32_t n_new, std::vector<uint32_t> & keep, std::vector<uint32_t> & dst, bool & same) const {
    const auto & cells = v_cells[0];
    const uint32_t n_old = cells.size();
    keep.clear();
    dst.clear();
    if (n_new == 0 || cells.get_has_shift()) {
        return false;
    }
    // live cells, by index; when they do not fit, the cells no query can attend again (SWA-masked for their sequence,
    // past the sinks) go first, lowest position first
    std::vector<std::pair<llama_pos, uint32_t>> masked;
    for (uint32_t i = 0; i < n_old; ++i) {
        if (cells.is_empty(i)) {
            continue;
        }
        keep.push_back(i);
        const llama_pos p = cells.pos_get(i);
        if (cells.seq_count(i) == 1 && p >= (llama_pos) n_swa_sink &&
                llama_hparams::is_masked_swa(n_swa, swa_type, p, cells.seq_pos_max(cells.seq_get(i)) + 1)) {
            masked.emplace_back(p, i);
        }
    }
    if (keep.size() > n_new) {
        const size_t n_drop = keep.size() - n_new;
        if (masked.size() < n_drop) {
            return false;
        }
        std::sort(masked.begin(), masked.end());
        std::vector<uint32_t> drop;
        for (size_t j = 0; j < n_drop; ++j) {
            drop.push_back(masked[j].second);
        }
        std::sort(drop.begin(), drop.end());
        std::vector<uint32_t> rest;
        std::set_difference(keep.begin(), keep.end(), drop.begin(), drop.end(), std::back_inserter(rest));
        keep.swap(rest);
    }
    // the same indices when they fit (a grow, or a shrink of a cache used only below n_new), else packed in position
    // order (a wrapped ring comes out ascending)
    same = keep.empty() || keep.back() < n_new;
    if (!same) {
        std::stable_sort(keep.begin(), keep.end(), [&](uint32_t a, uint32_t b) { return cells.pos_get(a) < cells.pos_get(b); });
    }
    dst.resize(keep.size());
    for (size_t j = 0; j < keep.size(); ++j) {
        dst[j] = same ? keep[j] : (uint32_t) j;
    }
    return true;
}

bool llama_kv_cache::swa_resize_fits(uint32_t n) const {
    if (n == v_cells[0].size()) {
        return true;
    }
    std::vector<uint32_t> keep, dst;
    bool same = true;
    return swa_plan(n, keep, dst, same);
}

bool llama_kv_cache::swa_resize(uint32_t n_new, const std::function<bool(size_t)> & make_room) {
    if (!swa_resize_ok()) {
        return false;
    }
    auto & cells = v_cells[0];
    const uint32_t n_old = cells.size();
    if (n_new == n_old) {
        return true;
    }
    std::vector<uint32_t> keep, dst;
    bool same = true;
    if (!swa_plan(n_new, keep, dst, same)) {
        LLAMA_LOG_WARN("%s: [SWAGROW] the live cells do not fit in %u (or a K shift is pending)\n", __func__, n_new);
        return false;
    }

    auto & buf_old = ctxs_bufs.bufs[0].second;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf_old.get());
    const size_t bytes_old = ggml_backend_buffer_get_size(buf_old.get());
    const size_t cell_bytes = swa_cell_bytes();

    std::vector<std::pair<size_t, size_t>> runs; // [j0, j1) of keep/dst with consecutive source and destination cells
    for (size_t j0 = 0; j0 < keep.size();) {
        size_t j1 = j0 + 1;
        while (j1 < keep.size() && keep[j1] == keep[j1 - 1] + 1 && dst[j1] == dst[j1 - 1] + 1) {
            ++j1;
        }
        runs.emplace_back(j0, j1);
        j0 = j1;
    }

    // a context + buffer with n cells per layer, the stream views inside it (allocated with the tensors)
    struct built { ggml_context_ptr ctx; ggml_backend_buffer_ptr buf; std::vector<ggml_tensor *> k, v, ks, vs; };
    auto build = [&](uint32_t n) -> built {
        built b;
        ggml_init_params params = {
            /*.mem_size   =*/ (4*layers.size() + 4)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        b.ctx.reset(ggml_init(params));
        if (!b.ctx) {
            return b;
        }
        for (const auto & layer : layers) {
            ggml_tensor * k = ggml_new_tensor_3d(b.ctx.get(), layer.k->type, layer.k->ne[0], n, 1);
            ggml_tensor * v = ggml_new_tensor_3d(b.ctx.get(), layer.v->type, layer.v->ne[0], n, 1);
            ggml_set_name(k, layer.k->name);
            ggml_set_name(v, layer.v->name);
            b.k.push_back(k);
            b.v.push_back(v);
            b.ks.push_back(ggml_view_2d(b.ctx.get(), k, k->ne[0], n, k->nb[1], 0));
            b.vs.push_back(ggml_view_2d(b.ctx.get(), v, v->ne[0], n, v->nb[1], 0));
        }
        b.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(b.ctx.get(), buft));
        if (b.buf) {
            ggml_backend_buffer_clear(b.buf.get(), 0); // as at boot: no NaN in cells a masked attention reads
        }
        return b;
    };
    auto install = [&](built & b) {
        for (size_t l = 0; l < layers.size(); ++l) {
            layers[l].k = b.k[l];
            layers[l].v = b.v[l];
            layers[l].k_stream[0] = b.ks[l];
            layers[l].v_stream[0] = b.vs[l];
        }
        ctxs_bufs.bufs[0] = { std::move(b.ctx), std::move(b.buf) };
    };
    auto set_meta = [&]() {
        const llama_kv_cells kept = cells.cp(keep);
        cells.resize(n_new);
        cells.set(dst, kept);
        if (!same) {
            v_heads[0] = (uint32_t) keep.size() % n_new;
        } else if (v_heads[0] >= n_new) {
            v_heads[0] = 0;
        }
    };

    // 1. the new buffer beside the old one, the kept runs copied device to device
    built b = build(n_new);
    if (b.buf) {
        constexpr size_t n_batch = 256; // copies per batch (one context of views, one sync)
        ggml_init_params params = {
            /*.mem_size   =*/ 2*n_batch*ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        std::vector<ggml_tensor *> srcs, dsts;
        ggml_context_ptr vctx;
        auto flush = [&]() {
            if (!srcs.empty()) {
                ggml_backend_tensor_copy_n(srcs.data(), dsts.data(), srcs.size());
            }
            srcs.clear();
            dsts.clear();
            vctx.reset();
        };
        for (const auto & [j0, j1] : runs) {
            for (size_t l = 0; l < layers.size(); ++l) {
                const std::pair<ggml_tensor *, ggml_tensor *> pairs[2] = { { layers[l].k, b.k[l] }, { layers[l].v, b.v[l] } };
                for (const auto & [src, out] : pairs) {
                    if (srcs.size() == n_batch) {
                        flush();
                    }
                    if (!vctx) {
                        vctx.reset(ggml_init(params));
                        GGML_ASSERT(vctx && "SWA-GROW: no host memory for the copy views");
                    }
                    const int64_t nr = (int64_t) (j1 - j0);
                    ggml_tensor * vs = ggml_view_2d(vctx.get(), src, src->ne[0], nr, src->nb[1], (size_t) keep[j0]*src->nb[1]);
                    ggml_tensor * vd = ggml_view_2d(vctx.get(), out, out->ne[0], nr, out->nb[1], (size_t) dst[j0]*out->nb[1]);
                    ggml_backend_view_init(vs);
                    ggml_backend_view_init(vd);
                    srcs.push_back(vs);
                    dsts.push_back(vd);
                }
            }
        }
        flush();
        set_meta();
        LLAMA_LOG_WARN("%s: [SWAGROW] window cache %u -> %u cells (%.1f -> %.1f MiB), %zu live cells %s\n", __func__, n_old, n_new,
                bytes_old/1048576.0, ggml_backend_buffer_get_size(b.buf.get())/1048576.0, keep.size(), same ? "kept in place" : "packed");
        install(b); // the old buffer and tensors go last
        return true;
    }
    b = built{};

    // 2. fallback (no VRAM for both): the kept rows go to host memory (pinned when the device has a host buffer type),
    // the old buffer is freed, then the new size; make_room is asked for VRAM before each retry
    LLAMA_LOG_WARN("%s: [SWAGROW] no VRAM for a %u-cell window cache beside the %u-cell one: staging through host memory\n",
            __func__, n_new, n_old);
    const size_t stage_bytes = std::max<size_t>(1, keep.size()*cell_bytes);
    ggml_backend_buffer_ptr stage_pinned;
    std::unique_ptr<uint8_t[]> stage_plain;
    uint8_t * stage = nullptr;
    {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (host_buft) {
            stage_pinned.reset(ggml_backend_buft_alloc_buffer(host_buft, stage_bytes));
        }
        if (stage_pinned) {
            stage = (uint8_t *) ggml_backend_buffer_get_base(stage_pinned.get());
        } else {
            stage_plain.reset(new uint8_t[stage_bytes]); // not zero-filled: every byte is written below
            stage = stage_plain.get();
        }
    }
    {
        size_t off = 0;
        for (const auto & [j0, j1] : runs) {
            for (const auto & layer : layers) {
                for (ggml_tensor * src : { layer.k, layer.v }) {
                    const size_t n = (j1 - j0)*src->nb[1];
                    ggml_backend_tensor_get(src, stage + off, (size_t) keep[j0]*src->nb[1], n);
                    off += n;
                }
            }
        }
    }
    auto restore = [&](built & o, bool at_dst) {
        size_t off = 0;
        for (const auto & [j0, j1] : runs) {
            for (size_t l = 0; l < layers.size(); ++l) {
                for (ggml_tensor * out : { o.k[l], o.v[l] }) {
                    const size_t n = (j1 - j0)*out->nb[1];
                    ggml_backend_tensor_set(out, stage + off, (size_t) (at_dst ? dst[j0] : keep[j0])*out->nb[1], n);
                    off += n;
                }
            }
        }
    };

    buf_old.reset();
    b = build(n_new);
    if (!b.buf && make_room && make_room((size_t) n_new*cell_bytes)) {
        b = build(n_new);
    }
    if (b.buf) {
        restore(b, true);
        set_meta();
        LLAMA_LOG_WARN("%s: [SWAGROW] window cache %u -> %u cells (%.1f -> %.1f MiB, host staged), %zu live cells %s\n", __func__,
                n_old, n_new, bytes_old/1048576.0, ggml_backend_buffer_get_size(b.buf.get())/1048576.0, keep.size(),
                same ? "kept in place" : "packed");
        install(b);
        return true;
    }

    // back to the old size at the old indices (the cell metadata is unchanged)
    LLAMA_LOG_WARN("%s: [SWAGROW] no VRAM for a %u-cell window cache, staying at %u cells\n", __func__, n_new, n_old);
    built o = build(n_old);
    if (!o.buf && make_room && make_room(bytes_old)) {
        o = build(n_old);
    }
    if (!o.buf) {
        // the old buffer is gone and the layers still point at it: no state to go back to
        GGML_ABORT("[SWAGROW] the window cache could not be allocated again at its old size (%u cells)", n_old);
    }
    restore(o, false);
    install(o);
    // cells dropped by the plan (SWA-masked for good) were not staged: they leave the metadata too
    std::vector<bool> kept(n_old, false);
    for (uint32_t i : keep) {
        kept[i] = true;
    }
    for (uint32_t i = 0; i < n_old; ++i) {
        if (!kept[i] && !cells.is_empty(i)) {
            cells.rm(i);
        }
    }
    return false;
}

// ---- context side ----------------------------------------------------------------------------------------------------

#include "llama-context.h"
#include "llama-ink-stream.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-model.h"
#include "llama-moe-cache.h"

#include <cstdlib>

static llama_kv_cache * llama_ink_swa_cache(llama_memory_i * memory) {
    if (auto * hy = dynamic_cast<llama_memory_hybrid_iswa *>(memory)) {
        return hy->get_mem_attn() ? hy->get_mem_attn()->get_swa() : nullptr;
    }
    if (auto * kv = dynamic_cast<llama_kv_cache_iswa *>(memory)) {
        return kv->get_swa();
    }
    return nullptr;
}

void llama_context::ink_swa_init() {
    const char * e = getenv("LLAMA_INK_SWA_GROW");
    if (!(e && atoi(e) != 0) || !ink_loan || cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || !memory) {
        return;
    }
    llama_kv_cache * kv = llama_ink_swa_cache(memory.get());
    const char * why_not = kv ? kv->swa_resize_why_not() : "no window cache";
    if (why_not) {
        LLAMA_LOG_WARN("%s: [SWAGROW] LLAMA_INK_SWA_GROW=1 but the window cache cannot be resized (%s); off\n", __func__, why_not);
        return;
    }
    const uint32_t boot = kv->get_size();
    // decode and the loan's short-prompt chunks write at most ink_loan_rows cells per ubatch: the window, its sinks and
    // those rows, padded as at boot (llama_kv_cache_iswa)
    const uint32_t dec = GGML_PAD(model.hparams.n_swa + kv->get_swa_sink() + ink_loan_rows, 256);
    if (dec >= boot) {
        LLAMA_LOG_WARN("%s: [SWAGROW] window cache %u cells is already the decode size; off\n", __func__, boot);
        return;
    }
    ink_swa_kv   = kv;
    ink_swa_boot = boot;
    ink_swa_dec  = dec;
    LLAMA_LOG_WARN("%s: [SWAGROW] on: while the cache holds the loan the window cache keeps %u of %u cells "
            "(%.0f MiB more for the MoE cache; LLAMA_INK_SWA_GROW=0 = off)\n", __func__, dec, boot,
            (double) (boot - dec)*kv->swa_cell_bytes()/1048576.0);
}

bool llama_context::ink_swa_set(bool small, const char * why) {
    if (small == ink_swa_small) {
        return true;
    }
    if (!ink_swa_kv) {
        return false;
    }
    if (!prompt_warm_join()) { // a background expert refill never runs beside the realloc
        LLAMA_LOG_WARN("%s: [SWAGROW] the expert refill failed; window cache size unchanged (%s)\n", __func__, why);
        return false;
    }
    if (const char * why_not = ink_swa_kv->swa_resize_why_not()) {
        // only a shared cell list is for good (it leaves the feature off); the rest are refused this time
        const bool shared = strcmp(why_not, "the cell list is shared with another context") == 0;
        LLAMA_LOG_WARN("%s: [SWAGROW] the window cache cannot be resized now (%s)%s\n", __func__, why_not,
                shared ? (ink_swa_small ? "; off, it keeps the decode size" : "; off") : "; size unchanged");
        if (shared) {
            ink_swa_kv = nullptr;
        }
        return false;
    }
    const uint32_t n = small ? ink_swa_dec : ink_swa_boot;
    if (ink_swa_kv->get_has_shift() || !ink_swa_kv->swa_resize_fits(n)) {
        LLAMA_LOG_WARN("%s: [SWAGROW] %s (%s): %s; size unchanged\n", __func__, small ? "decode size" : "boot size", why,
                ink_swa_kv->get_has_shift() ? "a K shift is pending" : "the live cells do not fit");
        return false;
    }
    const int64_t t0 = ggml_time_us();
    synchronize();
    // the CUDA graphs and the reused graphs hold the old K/V tensors
    for (ggml_backend_t backend : backend_ptrs) {
        auto dev = ggml_backend_get_device(backend);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        using reset_fn = bool (*)(ggml_backend_t);
        auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
        if (reset) {
            reset(backend);
        }
    }
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_prev_active = nullptr;
    // the host-staged fallback asks for VRAM: the MoE cache banks go (suspend), and come back below
    bool cache_freed = false;
    auto make_room = [&](size_t bytes) -> bool {
        if (!moe_cache || moe_cache->suspended()) {
            return false;
        }
        LLAMA_LOG_WARN("%s: [SWAGROW] the MoE cache is suspended for %.0f MiB of window cache\n", __func__, bytes/1048576.0);
        moe_cache->suspend();
        cache_freed = true;
        return true;
    };
    const bool ok = ink_swa_kv->swa_resize(n, make_room);
    if (ok) {
        ink_swa_small = small;
    }
    if (cache_freed) {
        // the grown loan was measured beside the decode-size window cache: at the boot size the cache goes back to boot
        if (!ink_swa_small && moe_cache->size() > ink_loan_boot) {
            moe_cache->resize(ink_loan_boot);
        }
        bool back = moe_cache->resume();
        if (!back && moe_cache->size() > ink_loan_boot) {
            moe_cache->suspend();
            moe_cache->resize(ink_loan_boot);
            back = moe_cache->resume();
        }
        ink_loan_grown = back && moe_cache->size() > ink_loan_boot;
        if (!back) {
            LLAMA_LOG_ERROR("%s: [SWAGROW] cannot refill the MoE cache (retried on the next calls)\n", __func__);
        }
    }
    sched_need_reserve = true;
    LLAMA_LOG_WARN("%s: [SWAGROW] %s (%s) in %.1f ms%s\n", __func__, small ? "decode size" : "boot size", why,
            (ggml_time_us() - t0)/1000.0, ok ? "" : ": failed, size unchanged");
    return ok;
}

void llama_context::ink_swa_full(const char * why) {
    if (!ink_swa_small) {
        return;
    }
    prompt_warm_join(); // the refill thread never runs beside the suspend below
    if (ink_loan_grown && !(ink_stream && ink_stream->armed())) {
        ink_loan_shrink(why); // gives the cells back to the window cache before the cache refills at its boot size
    }
    ink_swa_set(false, why);
}
