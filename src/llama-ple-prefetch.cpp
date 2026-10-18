#include "llama-ple-prefetch.h"

#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static int64_t ple_pf_env(const char * name, int64_t def) {
    const char * v = getenv(name);
    return v && *v ? (int64_t) atoll(v) : def;
}

#if defined(__linux__)
// the file and offset behind a mapped address, from /proc/self/maps
static bool ple_pf_resolve(const void * addr, std::string & path, uint64_t & off) {
    FILE * f = fopen("/proc/self/maps", "r");
    if (!f) {
        return false;
    }
    char line[4096];
    bool ok = false;
    while (fgets(line, sizeof(line), f)) {
        unsigned long long a = 0, b = 0, o = 0, ino = 0;
        unsigned int dmaj = 0, dmin = 0;
        char perms[8] = {0};
        int n = 0;
        if (sscanf(line, "%llx-%llx %7s %llx %x:%x %llu %n", &a, &b, perms, &o, &dmaj, &dmin, &ino, &n) < 7) {
            continue;
        }
        const uintptr_t p = (uintptr_t) addr;
        if (p < a || p >= b) {
            continue;
        }
        char * name = line + n;
        name[strcspn(name, "\n")] = 0;
        if (name[0] == '/') {
            path = name;
            off  = o + (p - a);
            ok   = true;
        }
        break;
    }
    fclose(f);
    return ok;
}
#endif

std::unique_ptr<llama_ple_prefetch> llama_ple_prefetch::create(const llama_model & model, uint32_t n_ubatch) {
    const int64_t n_thr = ple_pf_env("LLAMA_PLE_PREFETCH", 0);
    if (n_thr <= 0) {
        return nullptr;
    }
    const ggml_tensor * t = model.per_layer_tok_embd;
    if (model.arch != LLM_ARCH_QWEN4EXP || !t || !model.can_prefetch.count(t) || !t->data || !t->buffer ||
            !ggml_backend_buffer_is_host(t->buffer) || model.hparams.ple_n_heads == 0 || model.hparams.ple_ngram_size < 2) {
        LLAMA_LOG_WARN("%s: [PLEPF] LLAMA_PLE_PREFETCH is set but this model has no lazily mapped qwen4exp PLE table, off\n", __func__);
        return nullptr;
    }

    std::unique_ptr<llama_ple_prefetch> pf(new llama_ple_prefetch());
    pf->model     = &model;
    pf->ple       = t;
    pf->n_threads = (int) std::min<int64_t>(n_thr, 512);
    pf->check     = ple_pf_env("LLAMA_PLE_PF_CHECK", 0) != 0;
    pf->n_min     = (uint32_t) std::max<int64_t>(1, ple_pf_env("LLAMA_PLE_PF_MIN", 256));
    pf->n_ahead   = (uint64_t) std::max<int64_t>(0, ple_pf_env("LLAMA_PLE_PF_AHEAD", 2*(int64_t) n_ubatch));
    pf->n_ubatch  = n_ubatch;
    pf->row_size  = t->nb[1];
    pf->n_rows    = (uint64_t) t->ne[1];

    const char * mode = getenv("LLAMA_PLE_PF_MODE");
    pf->use_pread = !(mode && strcmp(mode, "fault") == 0);
#if defined(__linux__)
    if (pf->use_pread) {
        std::string path;
        uint64_t off = 0;
        if (ple_pf_resolve(t->data, path, off)) {
            pf->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        }
        if (pf->fd < 0) {
            LLAMA_LOG_WARN("%s: [PLEPF] cannot open the file behind the PLE table, reading by page faults\n", __func__);
            pf->use_pread = false;
        } else {
            pf->file_off = off;
            // the descriptor's own readahead state: read exactly the pages asked for
            posix_fadvise(pf->fd, 0, 0, POSIX_FADV_RANDOM);
        }
    }
#else
    pf->use_pread = false;
#endif

    LLAMA_LOG_WARN("%s: [PLEPF] PLE prefetch on: %d threads, %s, calls of %u+ tokens, %" PRIu64 " tokens ahead%s\n", __func__,
            pf->n_threads, pf->use_pread ? "pread" : "page faults", pf->n_min, pf->n_ahead, pf->check ? ", checking the guesses" : "");
    return pf;
}

llama_ple_prefetch::~llama_ple_prefetch() {
    stop();
#if defined(__linux__)
    if (fd >= 0) {
        close(fd);
    }
#endif
}

bool llama_ple_prefetch::start(const llama_batch_ext & batch, llama_pos pos0) {
    stop();

    const size_t n = batch.tokens.size();
    if (n < n_min) {
        return false;
    }
    llama_seq_id seq = -1;
    for (const auto & tok : batch.tokens) {
        if (tok.seq_ids.size() != 1) {
            return false;
        }
        const llama_seq_id s = *tok.seq_ids.begin();
        if (seq < 0) {
            seq = s;
        } else if (s != seq) {
            return false;
        }
    }

    const auto & hp = model->hparams;
    const llama_token img = hp.ple_image_token_id != 0 ? (llama_token) hp.ple_image_token_id : (llama_token) hp.ple_eos_token_id;
    toks.resize(n);
    for (size_t i = 0; i < n; ++i) {
        toks[i] = batch.tokens[i].has_embd ? img : batch.tokens[i].id;
    }
    seq_start = pos0 == 0;

    stopping  = false;
    built     = false;
    cursor    = 0;
    allowed   = 0;
    n_read    = 0;
    consumed  = 0;
    t_wait_us = 0;
    n_ubatch_c = 0;
    n_ubatch_u = 0;
    n_checked  = 0;
    n_mismatch = 0;
    t_build_us = 0;
    t_last_read_us = 0;
    t_start_us = ggml_time_us();
    running    = true;

    coord = std::thread([this]() { build(); });
    return true;
}

size_t llama_ple_prefetch::allowed_for(uint64_t n_tok) const {
    if (n_chunks == 0) {
        return 0;
    }
    const size_t c = (size_t) std::min<uint64_t>(n_chunks, (n_tok + CHUNK - 1)/CHUNK);
    return c == 0 ? 0 : chunk_end[c - 1];
}

void llama_ple_prefetch::build() {
    const auto & hp = model->hparams;

    const int64_t n        = (int64_t) toks.size();
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;

    // the rows, as llm_graph_input_qwen4exp_ple::set_input computes them; a predecessor before the call is unknown,
    // unless the call starts the sequence (then it reads as EOS)
    pred.assign((size_t) (n*n_heads), -1);
    std::vector<int64_t> ctx(n_gram);
    for (int64_t i = 0; i < n; ++i) {
        ctx[0] = toks[i];
        int64_t known = 0;
        bool    cut   = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            if (!cut && i - s < 0) {
                if (!seq_start) {
                    break;
                }
                cut = true;
            }
            const int64_t t = cut ? -1 : toks[i - s];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
            known  = s;
        }
        for (int64_t g = 2; g <= n_gram && g - 1 <= known; ++g) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < g; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (g - 2) * per_gram;
            for (int64_t k = 0; k < per_gram; ++k) {
                const int64_t h = base + k;
                pred[(size_t) (i*n_heads + h)] = (int32_t) (mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h]);
            }
        }
    }

    // the pages, chunk by chunk in token order, each page once
    const uint64_t base  = use_pread ? file_off : (uint64_t) (uintptr_t) ple->data;
    const uint64_t key0  = base / 4096;
    const uint64_t nkeys = (base + n_rows*row_size + 4095)/4096 - key0 + 1;
    std::vector<uint8_t> seen(nkeys, 0);

    n_chunks = (size_t) ((n + CHUNK - 1)/CHUNK);
    chunk_end.assign(n_chunks, 0);
    remaining.reset(new std::atomic<int64_t>[n_chunks]);
    pages.clear();
    page_chunk.clear();
    pages.reserve((size_t) n*n_heads/2);
    page_chunk.reserve((size_t) n*n_heads/2);
    for (size_t c = 0; c < n_chunks; ++c) {
        const size_t p0 = pages.size();
        const int64_t i1 = std::min<int64_t>(n, (int64_t) (c + 1)*CHUNK);
        for (int64_t i = (int64_t) c*CHUNK; i < i1; ++i) {
            for (int64_t h = 0; h < n_heads; ++h) {
                const int32_t r = pred[(size_t) (i*n_heads + h)];
                if (r < 0 || (uint64_t) r >= n_rows) {
                    continue;
                }
                const uint64_t b = base + (uint64_t) r*row_size;
                const uint64_t k0 = b/4096, k1 = (b + row_size - 1)/4096;
                for (uint64_t k = k0; k <= k1; ++k) {
                    if (!seen[k - key0]) {
                        seen[k - key0] = 1;
                        pages.push_back(k);
                    }
                }
            }
        }
        std::sort(pages.begin() + p0, pages.end());
        page_chunk.resize(pages.size(), (uint32_t) c);
        chunk_end[c] = pages.size();
        remaining[c].store((int64_t) (pages.size() - p0));
    }
    t_build_us = ggml_time_us() - t_start_us;

    {
        std::lock_guard<std::mutex> lk(mtx);
        // the first ubatch and the read-ahead
        allowed = std::max(allowed.load(), allowed_for((uint64_t) n_ubatch + n_ahead));
        built = true;
    }
    cv_done.notify_all();
    cv_work.notify_all();

    if (stopping) {
        return;
    }
    std::vector<int64_t> last(n_threads, 0);
    readers.clear();
    readers.reserve(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        readers.emplace_back([this, i, &last]() { last[i] = reader(); });
    }
    for (auto & th : readers) {
        th.join();
    }
    readers.clear();
    int64_t t_last = 0;
    for (int64_t t : last) {
        t_last = std::max(t_last, t);
    }
    t_last_read_us = t_last > 0 ? t_last - t_start_us : 0;
}

int64_t llama_ple_prefetch::reader() {
    char buf[4096];
    int64_t t_last = 0;
    for (;;) {
        if (stopping.load(std::memory_order_relaxed)) {
            break;
        }
        const size_t i = cursor.fetch_add(1);
        if (i >= pages.size()) {
            break;
        }
        if (i >= allowed.load()) {
            std::unique_lock<std::mutex> lk(mtx);
            cv_work.wait(lk, [&]() { return stopping.load() || allowed.load() > i; });
            if (stopping) {
                break;
            }
        }
#if defined(__linux__)
        if (use_pread) {
            if (pread(fd, buf, sizeof(buf), (off_t) (pages[i]*4096)) < 0) {
                // a failed read only costs the gather a fault
            }
        } else
#endif
        {
            const volatile char * p = (const volatile char *) (uintptr_t) (pages[i]*4096);
            buf[0] = *p;
        }
        n_read.fetch_add(1, std::memory_order_relaxed);
        t_last = ggml_time_us();
        if (remaining[page_chunk[i]].fetch_sub(1) == 1) {
            std::lock_guard<std::mutex> lk(mtx);
            cv_done.notify_all();
        }
    }
    return t_last;
}

bool llama_ple_prefetch::consume(uint32_t n_tokens, const int32_t * idx, size_t n_idx) {
    if (!running || n_tokens == 0) {
        return false;
    }
    const int64_t tw = ggml_time_us();
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [&]() { return built.load() || stopping.load(); });
    }
    if (stopping) {
        return false;
    }
    const uint64_t t0 = consumed, t1 = t0 + n_tokens;
    consumed = t1;
    if (t1 > toks.size()) {
        ++n_ubatch_u;
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        const size_t a = allowed_for(t1 + n_ahead);
        if (a > allowed) {
            allowed = a;
        }
    }
    cv_work.notify_all();

    const size_t c0 = (size_t) (t0/CHUNK), c1 = (size_t) ((t1 - 1)/CHUNK);
    {
        std::unique_lock<std::mutex> lk(mtx);
        cv_done.wait(lk, [&]() {
            if (stopping) {
                return true;
            }
            for (size_t c = c0; c <= c1; ++c) {
                if (remaining[c].load() > 0) {
                    return false;
                }
            }
            return true;
        });
    }
    t_wait_us += ggml_time_us() - tw;
    ++n_ubatch_c;

    const int64_t n_heads = model->hparams.ple_n_heads;
    if (check && idx && n_idx == (size_t) n_tokens*n_heads) {
        for (size_t k = 0; k < n_idx; ++k) {
            const int32_t g = pred[(size_t) t0*n_heads + k];
            if (g >= 0) {
                ++n_checked;
                n_mismatch += g != idx[k];
            }
        }
    }
    return true;
}

void llama_ple_prefetch::stop() {
    if (!running) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        stopping = true;
    }
    cv_work.notify_all();
    cv_done.notify_all();
    if (coord.joinable()) {
        coord.join();
    }
    running = false;

    LLAMA_LOG_WARN("%s: [PLEPF] %zu tokens: %zu pages (%.0f MiB) guessed in %.1f ms, %" PRId64 " read, last read at %.1f ms; "
            "%d ubatches waited %.1f ms in all%s", __func__, toks.size(), pages.size(), pages.size()*4096.0/1048576.0,
            t_build_us.load()/1e3, n_read.load(), t_last_read_us.load()/1e3, n_ubatch_c, t_wait_us/1e3,
            n_ubatch_u ? " (some ubatches fell outside the call)" : "");
    if (check) {
        LLAMA_LOG_CONT(", check: %" PRId64 " of %" PRId64 " guessed rows differ\n", n_mismatch, n_checked);
    } else {
        LLAMA_LOG_CONT("\n");
    }

    toks.clear();
    pred.clear();
    pred.shrink_to_fit();
    pages.clear();
    pages.shrink_to_fit();
    page_chunk.clear();
    page_chunk.shrink_to_fit();
    chunk_end.clear();
    remaining.reset();
    n_chunks = 0;
}
