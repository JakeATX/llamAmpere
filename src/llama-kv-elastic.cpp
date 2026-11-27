// fork (SURGE-cache, elastic KV): the record pages of a single-sequence SJ-KVaRN cache past the live context are
// unmapped so the MoE expert cache can use that VRAM, and mapped again before the context reaches them.
// Opt-in: LLAMA_ELASTIC_KV=1 (off: the pools stay in the plain KV buffer, nothing here runs).
// LLAMA_ELASTIC_KV_STEP=N: mapping step in positions (default 16384; rounded to whole record groups).
// The record pool of layer l holds group g (positions sink + g*G .. sink + (g+1)*G) of every head at bytes
// [g*H*rec_bytes, (g+1)*H*rec_bytes) (group-major, see the seal kernel), so the groups below a position are a prefix.

#include "llama-kv-cache.h"

#include "llama-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using elastic_set_fn    = bool   (*)(ggml_backend_buffer_t, const size_t *, const size_t *, int);
using elastic_mapped_fn = size_t (*)(ggml_backend_buffer_t);

static void * elastic_proc(ggml_backend_buffer_t buf, const char * name) {
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf));
    return dev ? ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), name) : nullptr;
}

void llama_kv_cache::elastic_init() {
    elastic_buf = nullptr;
    if (ctxs_bufs.no_alloc) {
        return;
    }
    for (const auto & layer : layers) {
        if (layer.body == nullptr || layer.body->buffer == nullptr) {
            continue;
        }
        const char * name = ggml_backend_buffer_name(layer.body->buffer);
        if (name == nullptr || strstr(name, "_Elastic") == nullptr) {
            continue;
        }
        GGML_ASSERT(elastic_buf == nullptr || elastic_buf == layer.body->buffer);
        elastic_buf = layer.body->buffer;
    }
    if (elastic_buf == nullptr) {
        return;
    }
    if (elastic_proc(elastic_buf, "ggml_backend_cuda_elastic_set_ranges") == nullptr) {
        elastic_buf = nullptr;
        return;
    }
    const char * e = getenv("LLAMA_ELASTIC_KV_STEP");
    const long step_pos = e && e[0] ? std::max(1l, strtol(e, nullptr, 10)) : 16384;
    elastic_step     = std::max<uint32_t>(1, (uint32_t) ((step_pos + sj_kvarn.group - 1)/sj_kvarn.group));
    elastic_n_groups = sj_kvarn_n_groups; // all mapped at boot
    LLAMA_LOG_WARN("%s: [ELASTICKV] on: %.0f MiB of SJ-KVaRN record pools in a VMM buffer (%u groups/head, step %u groups); "
            "pages past the live context go to the MoE cache\n", __func__, elastic_bytes(sj_kvarn_n_groups)/1048576.0,
            sj_kvarn_n_groups, elastic_step);
}

uint32_t llama_kv_cache::elastic_groups_for(uint32_t n_pos) const {
    // a position seals only once it is `tail` positions behind the end, so this over-counts by the tail at most
    const uint32_t g = n_pos > sj_kvarn.sink ? (n_pos - sj_kvarn.sink + sj_kvarn.group - 1)/sj_kvarn.group : 0;
    const uint32_t r = elastic_step > 0 ? (g + elastic_step - 1)/elastic_step*elastic_step : g;
    return std::min(sj_kvarn_n_groups, std::max(r, elastic_step));
}

size_t llama_kv_cache::elastic_bytes(uint32_t g) const {
    size_t res = 0;
    for (const auto & layer : layers) {
        if (layer.body != nullptr && layer.body->buffer == elastic_buf) {
            res += (size_t) g*hparams.n_head_kv(layer.il)*layer.sj_kvarn_rec_bytes;
        }
    }
    return res;
}

size_t llama_kv_cache::elastic_mapped() const {
    if (elastic_buf == nullptr) {
        return 0;
    }
    auto fn = (elastic_mapped_fn) elastic_proc(elastic_buf, "ggml_backend_cuda_elastic_mapped");
    return fn ? fn(elastic_buf) : 0;
}

bool llama_kv_cache::elastic_fit(uint32_t n_pos, bool shrink) {
    if (elastic_buf == nullptr) {
        return true;
    }
    const uint32_t g = elastic_groups_for(n_pos);
    if (g == elastic_n_groups || (g < elastic_n_groups && !shrink)) {
        return true;
    }
    std::vector<size_t> off;
    std::vector<size_t> len;
    const uint8_t * base = (const uint8_t *) ggml_backend_buffer_get_base(elastic_buf);
    for (const auto & layer : layers) {
        if (layer.body != nullptr && layer.body->buffer == elastic_buf) {
            off.push_back((size_t) ((const uint8_t *) layer.body->data - base));
            len.push_back((size_t) g*hparams.n_head_kv(layer.il)*layer.sj_kvarn_rec_bytes);
        }
    }
    auto set = (elastic_set_fn) elastic_proc(elastic_buf, "ggml_backend_cuda_elastic_set_ranges");
    if (!set(elastic_buf, off.data(), len.data(), (int) off.size())) {
        return false;
    }
    LLAMA_LOG_DEBUG("%s: [ELASTICKV] %u -> %u groups/head mapped (%.0f MiB)\n", __func__, elastic_n_groups, g, elastic_mapped()/1048576.0);
    elastic_n_groups = g;
    return true;
}

void llama_kv_cache::elastic_require(uint32_t n_pos) {
    if (elastic_buf == nullptr || elastic_groups_for(n_pos) <= elastic_n_groups) {
        return;
    }
    const size_t need = elastic_bytes(elastic_groups_for(n_pos) - elastic_n_groups) + (size_t(64) << 20);
    // the free VRAM the device reports (a capped test card reports less than the driver would still hand out)
    size_t free = 0, total = 0;
    if (ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(elastic_buf))) {
        ggml_backend_dev_memory(dev, &free, &total);
    }
    if (free >= need && elastic_fit(n_pos, false)) {
        return;
    }
    if (elastic_make_room && elastic_make_room(need > free ? need - free : need) && elastic_fit(n_pos, false)) {
        return;
    }
    LLAMA_LOG_ERROR("%s: [ELASTICKV] no VRAM to map the KV records of %u positions (%.0f MiB more)\n", __func__, n_pos, need/1048576.0);
    throw std::runtime_error("elastic KV: out of VRAM for the records of the context");
}

// ---- context side ----------------------------------------------------------------------------------------------------

#include "llama-context.h"
#include "llama-moe-cache.h"
#include "llama-batch.h"

static size_t elastic_gpu_free(const std::vector<ggml_backend_t> & backends, ggml_backend_t skip) {
    for (ggml_backend_t backend : backends) {
        if (backend == skip) {
            continue; // [TSPLITBS] sampler-only card
        }
        auto dev = ggml_backend_get_device(backend);
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            return free;
        }
    }
    return 0;
}

// the prompt compute buffer grown past its boot rows on VRAM the KV lent (dynamic ub ladder) gives it back: halve the rows
// down to the boot reserve until `bytes` are free. False if the rows are at the boot reserve already.
bool llama_context::elastic_kv_shrink_compute(size_t bytes) {
    if (!dynamic_ubatch || elastic_boot_rows == 0 || sched_reserve_tokens <= elastic_boot_rows) {
        return false;
    }
    const int64_t  t0   = ggml_time_us();
    const uint32_t rows0 = sched_reserve_tokens;
    synchronize();
    for (ggml_backend_t backend : backend_ptrs) {
        auto dev = ggml_backend_get_device(backend);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        using reset_fn = bool (*)(ggml_backend_t);
        auto reset = (reset_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_reset");
        if (reset && !reset(backend)) {
            return false;
        }
    }
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_prev_active = nullptr;
    while (sched_reserve_tokens > elastic_boot_rows && elastic_gpu_free(backend_ptrs, backend_tsplit_sampler) < bytes) {
        const uint32_t rows = std::max(elastic_boot_rows, sched_reserve_tokens/2);
        sched_reserve_tokens = rows;
        dynamic_ubatch_rows  = rows;
        sched_need_reserve   = true;
        sched_reserve(false);
    }
    // the ladder grows again only once the KV has stopped needing the room (its next check)
    dynamic_ubatch_pp_check_us = ggml_time_us() + 2000000;
    LLAMA_LOG_WARN("%s: [ELASTICKV] prompt compute buffer %u -> %u rows for %.0f MiB of KV records (%.1f ms)\n", __func__,
            rows0, sched_reserve_tokens, bytes/1048576.0, (ggml_time_us() - t0)/1000.0);
    return true;
}

bool llama_context::elastic_kv_make_room(size_t bytes) {
    if (moe_cache && moe_cache->partial()) {
        // fork (PLOAN): the kept bank pages go first; when they cover it nothing else shrinks (as with the whole cache lent)
        const size_t kept = moe_cache->kept_bytes();
        ink_ploan_give_back("KV records need them");
        if (kept >= bytes) {
            return true;
        }
        bytes -= kept;
    }
    if (!moe_cache || !ink_loan || moe_cache->size() <= ink_loan_boot) {
        // the boot sizes were fitted with every KV page mapped: only the cache loan or the compute rows grown on the
        // lent pages can be given back
        return elastic_kv_shrink_compute(bytes + (size_t(64) << 20));
    }
    const int64_t t0 = ggml_time_us();
    synchronize();
    for (ggml_backend_t backend : backend_ptrs) {
        auto dev = ggml_backend_get_device(backend);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        using trim_fn  = size_t (*)(ggml_backend_t);
        using reset_fn = bool (*)(ggml_backend_t);
        auto reg   = ggml_backend_dev_backend_reg(dev);
        auto trim  = (trim_fn)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_pool_trim");
        auto reset = (reset_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_reset");
        if (reset && reset(backend) && trim) {
            trim(backend);
        }
    }
    for (auto & res : gf_res_prev) {
        res.reset();
    }
    gf_res_prev_active = nullptr;
    const size_t size0  = moe_cache->size();
    const size_t slack  = size_t(32) << 20; // page rounding of the pools
    const size_t target = size0 > bytes + slack + ink_loan_boot ? size0 - bytes - slack : ink_loan_boot;
    const bool was_suspended = moe_cache->suspended();
    if (!was_suspended) {
        moe_cache->suspend();
    }
    moe_cache->resize(target);
    bool ok = true;
    if (!was_suspended) {
        ok = moe_cache->resume();
        if (!ok && moe_cache->size() > ink_loan_boot) {
            moe_cache->suspend();
            moe_cache->resize(ink_loan_boot);
            ok = moe_cache->resume();
        }
    }
    ink_loan_grown = moe_cache->size() > ink_loan_boot;
    LLAMA_LOG_WARN("%s: [ELASTICKV] MoE cache %.0f -> %.0f MiB for %.0f MiB of KV records (%.1f ms)%s\n", __func__,
            size0/1048576.0, moe_cache->size()/1048576.0, bytes/1048576.0, (ggml_time_us() - t0)/1000.0, ok ? "" : ", refill failed");
    return ok;
}

bool llama_context::elastic_kv_prepare(const llama_batch_ext & batch) {
    llama_kv_cache * kv = elastic_kv;
    if (kv == nullptr || !kv->elastic_on()) {
        return true;
    }
    // positions this call can reach: the end of the longest sequence plus every token of the call
    llama_pos p_end = 0;
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq_max(); ++s) {
        p_end = std::max(p_end, memory->seq_pos_max(s) + 1);
    }
    for (const auto & token : batch.tokens) {
        p_end = std::max(p_end, token.pos[0] + 1);
    }
    const uint32_t n_pos = (uint32_t) p_end + (uint32_t) batch.tokens.size();
    const uint32_t want  = kv->elastic_groups_for(n_pos);
    const uint32_t have  = kv->elastic_groups();
    if (want > have) {
        static const size_t margin = [] {
            const char * e = getenv("LLAMA_ELASTIC_KV_MARGIN_MIB");
            return (size_t) std::max(32l, e && e[0] ? strtol(e, nullptr, 10) : 128l) << 20;
        }();
        const size_t need = kv->elastic_bytes(want - have) + (size_t(64) << 20); // + page rounding
        const size_t free = elastic_gpu_free(backend_ptrs, backend_tsplit_sampler);
        if (free < need + margin) {
            elastic_kv_make_room(need + margin - free);
        }
        if (!kv->elastic_fit(n_pos, false) && !(elastic_kv_make_room(need) && kv->elastic_fit(n_pos, false))) {
            LLAMA_LOG_ERROR("%s: [ELASTICKV] no VRAM to map the KV records for %u positions\n", __func__, n_pos);
            return false;
        }
        LLAMA_LOG_INFO("%s: [ELASTICKV] KV records mapped for %u positions: %.0f MiB\n", __func__, n_pos, kv->elastic_mapped()/1048576.0);
    } else if (have > want) {
        // a whole step or more past the need (the context was cut): lend the pages; the cache takes them at its next refill
        const size_t before = kv->elastic_mapped();
        if (kv->elastic_fit(n_pos, true)) {
            LLAMA_LOG_WARN("%s: [ELASTICKV] KV records for %u positions: %.0f -> %.0f MiB mapped (the rest goes to the MoE cache)\n",
                    __func__, n_pos, before/1048576.0, kv->elastic_mapped()/1048576.0);
        }
    }
    return true;
}
