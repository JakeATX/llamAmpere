// fork (SWA-GROW): a window (SWA) cache sized n_swa + n_ubatch for the prompt ubatches runs decode in n_swa + a few
// rows. LLAMA_INK_SWA_GROW=1 (Inkling small cards, with CACHELOAN) re-allocates the window cache at the decode size
// when the cache loan grows and back at the boot size before the next prompt; the VRAM it frees goes to the MoE cache.
// Off (unset or 0): nothing here runs.

#include "llama-kv-cache.h"

#include "llama-impl.h"

#include <algorithm>
#include <iterator>
#include <vector>

size_t llama_kv_cache::swa_cell_bytes() const {
    size_t res = 0;
    for (const auto & layer : layers) {
        res += layer.k ? layer.k->nb[1] : 0;
        res += layer.v ? layer.v->nb[1] : 0;
    }
    return res;
}

bool llama_kv_cache::swa_resize_ok() const {
    if (ctxs_bufs.no_alloc || ctxs_bufs.bufs.size() != 1 || swa_type == LLAMA_SWA_TYPE_NONE || n_swa == 0 ||
            sj_kvarn.enabled() || n_stream != 1 || other != nullptr || v_trans || kv_stream_runtime.runtime != nullptr ||
            scratch.user != nullptr || layers.empty() || v_cells_impl.use_count() != 1) {
        return false;
    }
    const auto & [ctx, buf] = ctxs_bufs.bufs[0];
    if (!buf || ggml_backend_buffer_is_host(buf.get())) {
        return false;
    }
    // the buffer holds the layers' K/V (and their stream views) and nothing else
    size_t n_own = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
        n_own += t->view_src == nullptr ? 1 : 0;
    }
    for (const auto & layer : layers) {
        if (!layer.k || !layer.v || layer.shared || layer.body || layer.k->buffer != buf.get() || layer.v->buffer != buf.get() ||
                layer.k->ne[2] != 1 || layer.v->ne[2] != 1 || layer.k_stream.size() != 1 || layer.v_stream.size() != 1) {
            return false;
        }
    }
    return n_own == 2*layers.size();
}

bool llama_kv_cache::swa_resize(uint32_t n_new) {
    if (!swa_resize_ok()) {
        return false;
    }
    auto & cells = v_cells[0];
    const uint32_t n_old = cells.size();
    if (n_new == n_old) {
        return true;
    }
    if (n_new == 0 || cells.get_has_shift()) {
        return false;
    }

    // live cells, by index; when they do not fit, the cells no query can attend again (SWA-masked for their sequence,
    // past the sinks) go first, lowest position first
    std::vector<uint32_t> keep;
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
            LLAMA_LOG_WARN("%s: [SWAGROW] %zu live cells do not fit in %u\n", __func__, keep.size() - masked.size(), n_new);
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
    // the same indices when they fit (a grow, or a shrink of a cache used only below n_new), else packed
    std::vector<uint32_t> dst(keep.size());
    const bool same = keep.empty() || keep.back() < n_new;
    for (size_t j = 0; j < keep.size(); ++j) {
        dst[j] = same ? keep[j] : (uint32_t) j;
    }

    auto & buf_old = ctxs_bufs.bufs[0].second;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf_old.get());
    const size_t bytes_old = ggml_backend_buffer_get_size(buf_old.get());

    // the K/V rows of the kept cells go to host memory first, so the old buffer is freed before the new one is
    // allocated (a grow back needs only the boot bytes, never old + new)
    std::vector<std::pair<size_t, size_t>> runs; // [j0, j1) of keep/dst with consecutive source and destination cells
    for (size_t j0 = 0; j0 < keep.size();) {
        size_t j1 = j0 + 1;
        while (j1 < keep.size() && keep[j1] == keep[j1 - 1] + 1 && dst[j1] == dst[j1 - 1] + 1) {
            ++j1;
        }
        runs.emplace_back(j0, j1);
        j0 = j1;
    }
    std::vector<uint8_t> stage((size_t) keep.size()*swa_cell_bytes());
    {
        size_t off = 0;
        for (const auto & [j0, j1] : runs) {
            for (const auto & layer : layers) {
                for (ggml_tensor * src : { layer.k, layer.v }) {
                    const size_t n = (j1 - j0)*src->nb[1];
                    ggml_backend_tensor_get(src, stage.data() + off, keep[j0]*src->nb[1], n);
                    off += n;
                }
            }
        }
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
    auto restore = [&](built & b, bool at_dst) {
        size_t off = 0;
        for (const auto & [j0, j1] : runs) {
            for (size_t l = 0; l < layers.size(); ++l) {
                for (ggml_tensor * out : { b.k[l], b.v[l] }) {
                    const size_t n = (j1 - j0)*out->nb[1];
                    ggml_backend_tensor_set(out, stage.data() + off, (size_t) (at_dst ? dst[j0] : keep[j0])*out->nb[1], n);
                    off += n;
                }
            }
        }
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

    buf_old.reset();
    built b = build(n_new);
    if (!b.buf) {
        // back to the old size at the old indices (the cell metadata is unchanged)
        LLAMA_LOG_WARN("%s: [SWAGROW] no VRAM for a %u-cell window cache, staying at %u cells\n", __func__, n_new, n_old);
        built o = build(n_old);
        if (!o.buf) {
            GGML_ABORT("SWA-GROW: the window cache could not be allocated again at its old size");
        }
        restore(o, false);
        install(o);
        // cells dropped above (SWA-masked for good) were not staged: they leave the metadata too
        for (uint32_t i = 0; i < n_old; ++i) {
            if (!cells.is_empty(i) && !std::binary_search(keep.begin(), keep.end(), i)) {
                cells.rm(i);
            }
        }
        return false;
    }
    restore(b, true);

    // cell metadata at the new indices
    const llama_kv_cells kept = cells.cp(keep);
    cells.resize(n_new);
    cells.set(dst, kept);
    if (!same) {
        v_heads[0] = (uint32_t) keep.size() % n_new;
    } else if (v_heads[0] >= n_new) {
        v_heads[0] = 0;
    }
    LLAMA_LOG_WARN("%s: [SWAGROW] window cache %u -> %u cells (%.1f -> %.1f MiB), %zu live cells %s\n", __func__, n_old, n_new,
            bytes_old/1048576.0, ggml_backend_buffer_get_size(b.buf.get())/1048576.0, keep.size(), same ? "kept in place" : "packed");
    install(b); // the old tensors (context) go last
    return true;
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
    if (!kv || !kv->swa_resize_ok()) {
        LLAMA_LOG_WARN("%s: [SWAGROW] LLAMA_INK_SWA_GROW=1 but the window cache cannot be resized (needs one stream, "
                "flash attention, one GPU buffer); off\n", __func__);
        return;
    }
    const uint32_t boot = kv->get_size();
    // decode and the loan's short-prompt chunks write at most ink_loan_rows cells per ubatch: the window plus those rows,
    // padded as at boot (llama_kv_cache_iswa)
    const uint32_t dec = GGML_PAD(model.hparams.n_swa + ink_loan_rows, 256);
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
    if (!ink_swa_kv || small == ink_swa_small) {
        return true;
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
    memory_update(false); // a pending K shift is applied first (the cells move)
    const bool ok = ink_swa_kv->swa_resize(small ? ink_swa_dec : ink_swa_boot);
    if (ok) {
        ink_swa_small = small;
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
    if (ink_loan_grown && !(ink_stream && ink_stream->armed())) {
        ink_loan_shrink(why); // gives the cells back to the window cache before the cache refills at its boot size
    }
    ink_swa_set(false, why);
}
