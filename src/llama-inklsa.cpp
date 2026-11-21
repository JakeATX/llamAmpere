#include "llama-inklsa.h"

#include "llama-impl.h"

#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

static int64_t inklsa_env(const char * name, int64_t def) {
    const char * s = getenv(name);
    return s && *s ? atoll(s) : def;
}

const llama_inklsa_config & llama_inklsa_cfg() {
    static const llama_inklsa_config cfg = [] {
        llama_inklsa_config c;
        c.on         = inklsa_env("LLAMA_INKLSA", 0) != 0;
        c.keep       = inklsa_env("LLAMA_INKLSA_KEEP", c.keep);
        c.block      = inklsa_env("LLAMA_INKLSA_BLOCK", c.block);
        c.window     = inklsa_env("LLAMA_INKLSA_WINDOW", c.window);
        c.sinks      = inklsa_env("LLAMA_INKLSA_SINKS", c.sinks);
        c.min_kv     = inklsa_env("LLAMA_INKLSA_MIN_KV", c.min_kv);
        c.max_t      = inklsa_env("LLAMA_INKLSA_MAX_T", c.max_t);
        c.reuse_pass = (int) inklsa_env("LLAMA_INKLSA_REUSE_PASS", 0);
        c.stats      = inklsa_env("LLAMA_INKLSA_STATS", 0) != 0;
        if (c.on) {
            if (c.block <= 0 || 256 % c.block != 0 || c.sinks < 0 || c.sinks % c.block != 0 || c.keep <= 0 ||
                    c.window <= 0 || c.max_t <= 0) {
                LLAMA_LOG_ERROR("%s: invalid INKLSA settings (block %lld must divide 256, sinks %lld a multiple of it)"
                        " - INKLSA off\n", __func__, (long long) c.block, (long long) c.sinks);
                c.on = false;
            } else {
                LLAMA_LOG_WARN("%s: INKLSA on: keep %lld (%lld) block %lld window %lld sinks %lld min_kv %lld max_t %lld"
                        " reuse_pass %d\n", __func__, (long long) c.keep, (long long) ((c.keep + 255)/256*256),
                        (long long) c.block, (long long) c.window, (long long) c.sinks, (long long) c.min_kv,
                        (long long) c.max_t, c.reuse_pass);
            }
        }
        return c;
    }();
    return cfg;
}

int64_t llama_inklsa_n_fix() {
    const auto & c = llama_inklsa_cfg();
    return (c.sinks + c.window + c.block + c.max_t + 255)/256*256;
}

int64_t llama_inklsa_n_keep() {
    return (llama_inklsa_cfg().keep + 255)/256*256;
}

namespace {
struct inklsa_entry {
    ggml_tensor *         t     = nullptr;
    ggml_context *        ctx   = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    uint64_t              gen   = 0;
    uint64_t              built = 0; // valid at creation
};
struct inklsa_state {
    std::mutex mtx;
    std::map<const ggml_tensor *, inklsa_entry> e;
};
inklsa_state & inklsa_st() {
    static auto * s = new inklsa_state;
    return *s;
}
} // namespace

bool llama_inklsa_idx_get(const ggml_tensor * kcache, int64_t d, int64_t n_blk_cap, ggml_backend_buffer_type_t buft,
        llama_inklsa_idx * out) {
    auto & s = inklsa_st();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto & en = s.e[kcache];
    if (en.t && (en.t->ne[0] != d || en.t->ne[1] != n_blk_cap)) {
        return false;
    }
    if (!en.t) {
        ggml_init_params ip = { /*.mem_size =*/ 2*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        en.ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_2d(en.ctx, GGML_TYPE_F16, d, n_blk_cap);
        ggml_format_name(t, "inklsa_idx_%s", kcache->name);
        en.buf = ggml_backend_alloc_ctx_tensors_from_buft(en.ctx, buft);
        if (!en.buf) {
            LLAMA_LOG_ERROR("%s: cannot allocate the INKLSA index for %s\n", __func__, kcache->name);
            ggml_free(en.ctx);
            s.e.erase(kcache);
            return false;
        }
        ggml_backend_buffer_clear(en.buf, 0);
        en.t = t;
        LLAMA_LOG_INFO("%s: INKLSA index for %s: %lld x %lld F16 (%.1f MiB)\n", __func__, kcache->name, (long long) d,
                (long long) n_blk_cap, ggml_nbytes(t)/1048576.0);
    }
    out->t     = en.t;
    out->valid = en.built == en.gen;
    out->epoch = en.gen;
    return true;
}

void llama_inklsa_idx_commit(const ggml_tensor * kcache, uint64_t epoch) {
    auto & s = inklsa_st();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto it = s.e.find(kcache);
    if (it != s.e.end() && it->second.gen == epoch) {
        it->second.built = epoch;
    }
}

void llama_inklsa_invalidate(const ggml_tensor * const * kcaches, size_t n) {
    auto & s = inklsa_st();
    std::lock_guard<std::mutex> lock(s.mtx);
    if (kcaches == nullptr || n == 0) {
        for (auto & it : s.e) {
            it.second.gen++;
        }
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        auto it = s.e.find(kcaches[i]);
        if (it != s.e.end()) {
            it->second.gen++;
        }
    }
}

void llama_inklsa_forget(const ggml_tensor * const * kcaches, size_t n) {
    auto & s = inklsa_st();
    std::lock_guard<std::mutex> lock(s.mtx);
    for (size_t i = 0; kcaches && i < n; ++i) {
        auto it = s.e.find(kcaches[i]);
        if (it != s.e.end()) {
            ggml_backend_buffer_free(it->second.buf);
            ggml_free(it->second.ctx);
            s.e.erase(it);
        }
    }
}
