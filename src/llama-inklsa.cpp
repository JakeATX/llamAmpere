#include "llama-inklsa.h"

#include "llama-hparams.h"
#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

static bool inklsa_env_set(const char * name) {
    const char * s = getenv(name);
    return s && *s;
}

static int64_t inklsa_env(const char * name, int64_t def) {
    const char * s = getenv(name);
    return s && *s ? atoll(s) : def;
}

const llama_inklsa_config & llama_inklsa_cfg() {
    static const llama_inklsa_config cfg = [] {
        llama_inklsa_config c;
        c.on            = inklsa_env("LLAMA_INKLSA", 0) != 0;
        c.keep_env      = inklsa_env_set("LLAMA_INKLSA_KEEP");
        c.keep_max_env  = inklsa_env_set("LLAMA_INKLSA_KEEP_MAX");
        c.block_env     = inklsa_env_set("LLAMA_INKLSA_BLOCK");
        c.window_env    = inklsa_env_set("LLAMA_INKLSA_WINDOW");
        c.sinks_env     = inklsa_env_set("LLAMA_INKLSA_SINKS");
        c.min_kv_env    = inklsa_env_set("LLAMA_INKLSA_MIN_KV");
        c.keep_frac_env = inklsa_env_set("LLAMA_INKLSA_KEEP_FRAC");
        c.keep          = inklsa_env("LLAMA_INKLSA_KEEP", c.keep);
        c.keep_max      = inklsa_env("LLAMA_INKLSA_KEEP_MAX", c.keep_max);
        c.block         = inklsa_env("LLAMA_INKLSA_BLOCK", c.block);
        c.window        = inklsa_env("LLAMA_INKLSA_WINDOW", c.window);
        c.sinks         = inklsa_env("LLAMA_INKLSA_SINKS", c.sinks);
        c.min_kv        = inklsa_env("LLAMA_INKLSA_MIN_KV", c.min_kv);
        c.max_t         = inklsa_env("LLAMA_INKLSA_MAX_T", c.max_t);
        c.stats         = inklsa_env("LLAMA_INKLSA_STATS", 0) != 0;
        c.pick          = (int) inklsa_env("LLAMA_INKLSA_PICK", 0);
        c.comp          = inklsa_env("LLAMA_INKLSA_COMP", 0) != 0;
        if (const char * f = getenv("LLAMA_INKLSA_KEEP_FRAC"); f && *f) {
            c.keep_frac = std::clamp(atof(f), 0.0, 1.0);
        }
        c.relsplit   = inklsa_env("LLAMA_INKREL", 1) != 0;
        c.rel_min_t  = std::max<int64_t>(1, inklsa_env("LLAMA_INKREL_MIN_T", c.rel_min_t));
        if (c.on) {
            if (c.block <= 0 || 256 % c.block != 0 || c.sinks < 0 || c.sinks % c.block != 0 || c.keep <= 0 ||
                    c.window <= 0 || c.max_t <= 0) {
                LLAMA_LOG_ERROR("%s: invalid INKLSA settings (block %lld must divide 256, sinks %lld a multiple of it)"
                        " - INKLSA off\n", __func__, (long long) c.block, (long long) c.sinks);
                c.on = false;
            } else {
                LLAMA_LOG_WARN("%s: INKLSA on: keep %lld (%lld) keep_frac %.3f keep_max %lld pick %d comp %d block %lld window %lld sinks %lld"
                        " min_kv %lld max_t %lld\n", __func__, (long long) c.keep, (long long) ((c.keep + 255)/256*256),
                        c.keep_frac, (long long) c.keep_max, c.pick, (int) c.comp,
                        (long long) c.block, (long long) c.window, (long long) c.sinks, (long long) c.min_kv,
                        (long long) c.max_t);
            }
        }
        return c;
    }();
    return cfg;
}

llama_inklsa_resolved llama_inklsa_resolve(const llama_hparams & hparams, bool sj_mode) {
    const auto & c = llama_inklsa_cfg();
    const bool arch_lsa = hparams.is_lsa_any();
    llama_inklsa_resolved r;
    // priority: explicit env var > GGUF metadata > arch-LSA defaults (2048, 4%, 16384) > legacy env defaults
    r.keep      = c.keep_env      ? c.keep      : (hparams.lsa_keep_min  >  0 ? hparams.lsa_keep_min  : (arch_lsa ? 2048 : c.keep));
    r.keep_frac = c.keep_frac_env ? c.keep_frac : (hparams.lsa_keep_frac >= 0.0f ? (double) hparams.lsa_keep_frac : (arch_lsa ? 0.04 : c.keep_frac));
    r.keep_max  = c.keep_max_env  ? c.keep_max  : (hparams.lsa_keep_max  >= 0 ? hparams.lsa_keep_max  : c.keep_max);
    r.block     = c.block_env     ? c.block     : (hparams.lsa_block     >  0 ? hparams.lsa_block     : c.block);
    r.sinks     = c.sinks_env     ? c.sinks     : (hparams.lsa_sinks     >= 0 ? hparams.lsa_sinks     : c.sinks);
    r.window    = c.window_env    ? c.window    : (hparams.n_swa         >  0 ? (int64_t) hparams.n_swa : c.window);
    r.min_kv    = c.min_kv_env    ? c.min_kv    : (arch_lsa ? 0 : c.min_kv);
    r.max_t     = sj_mode         ? std::min<int64_t>(c.max_t, 8) : c.max_t;

    // sanitize to safe invariants
    if (r.block <= 0 || 256 % r.block != 0 || (r.block & (r.block - 1)) != 0) {
        r.block = 64;
    }
    if (r.sinks < 0 || r.sinks % r.block != 0) {
        r.sinks = (std::max<int64_t>(0, r.sinks) / r.block) * r.block;
    }
    if (r.keep <= 0) {
        r.keep = 2048;
    }
    r.keep_frac = std::clamp(r.keep_frac, 0.0, 1.0);
    if (r.keep_max > 0 && r.keep_max < r.keep) {
        r.keep_max = r.keep;
    }
    return r;
}

bool llama_inkrel_want(int64_t n_kv, int64_t T, int64_t extent) {
    const auto & c = llama_inklsa_cfg();
    // both calls need >= 2 KV tiles (the LSE comes from the parallel-block combine), and the near set fits in the view
    return c.relsplit && extent > 0 && T >= c.rel_min_t && n_kv >= 2*extent + T && n_kv % 256 == 0;
}

int64_t llama_inklsa_n_fix() {
    const auto & c = llama_inklsa_cfg();
    return (c.sinks + c.window + c.block + c.max_t + 255)/256*256;
}

int64_t llama_inklsa_n_fix_for(const llama_inklsa_resolved & rcfg) {
    return (rcfg.sinks + rcfg.window + rcfg.block + rcfg.max_t + 255)/256*256;
}

int64_t llama_inklsa_idx_rows(int64_t kv_size, int64_t B) {
    return (kv_size/B + 255)/256*256*(llama_inklsa_cfg().comp ? 2 : 1);
}

int64_t llama_inklsa_n_keep_for(int64_t n_kv, int64_t keep, double keep_frac, int64_t keep_max) {
    const int64_t raw    = std::max<int64_t>(keep, (int64_t) std::floor(keep_frac*(double) n_kv));
    const int64_t capped = keep_max > 0 ? std::min<int64_t>(raw, keep_max) : raw;
    return (capped + 255)/256*256;
}

int64_t llama_inklsa_n_keep(int64_t n_kv) {
    const auto & c = llama_inklsa_cfg();
    return llama_inklsa_n_keep_for(n_kv, c.keep, c.keep_frac, c.keep_max);
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

void llama_inklsa_reset_valid(const ggml_tensor * const * kcaches, size_t n) {
    auto & s = inklsa_st();
    std::lock_guard<std::mutex> lock(s.mtx);
    if (kcaches == nullptr || n == 0) {
        for (auto & it : s.e) {
            it.second.built = it.second.gen;
        }
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        auto it = s.e.find(kcaches[i]);
        if (it != s.e.end()) {
            it->second.built = it->second.gen;
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
