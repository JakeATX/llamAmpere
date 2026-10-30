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
#include <limits>
#include <stdexcept>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static constexpr int64_t PLE_PF_DEFAULT_THREADS = 2;
static constexpr int64_t PLE_PF_DEFAULT_MIN     = 4096;
static constexpr int64_t PLE_PF_DEFAULT_COLD    = 32;  // readers when the table is mostly out of the page cache

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
    // fork: page-touch readers are on for long prompts; LLAMA_PLE_PREFETCH=0 disables them.
    const bool    user  = getenv("LLAMA_PLE_PREFETCH") != nullptr;
    const int64_t n_thr = ple_pf_env("LLAMA_PLE_PREFETCH", ple_pf_env("LLAMA_PLE_PF_THREADS", PLE_PF_DEFAULT_THREADS));
    if (n_thr <= 0) {
        return nullptr;
    }
    const ggml_tensor * t = model.per_layer_tok_embd;
    if (model.arch != LLM_ARCH_QWEN4EXP || !t || !model.can_prefetch.count(t) || !t->data || !t->buffer ||
            !ggml_backend_buffer_is_host(t->buffer) || t->nb[1] == 0 || t->ne[1] <= 0 ||
            model.hparams.ple_n_heads == 0 || model.hparams.ple_ngram_size < 2) {
        if (user) {
            LLAMA_LOG_WARN("%s: [PLEPF] LLAMA_PLE_PREFETCH is set but this model has no lazily mapped qwen4exp PLE table, off\n", __func__);
        }
        return nullptr;
    }

    const uintptr_t buffer_base = (uintptr_t) ggml_backend_buffer_get_base(t->buffer);
    const uintptr_t table_base  = (uintptr_t) t->data;
    const size_t buffer_size = ggml_backend_buffer_get_size(t->buffer);
    // fork: check the full row span before readers form page addresses.
    if (!buffer_base || table_base < buffer_base || table_base - buffer_base > buffer_size ||
            (uint64_t) t->ne[1] > (buffer_size - (table_base - buffer_base))/t->nb[1]) {
        return nullptr;
    }

    const auto & hp = model.hparams;
    if (hp.ple_ngram_size > hp.ple_layer_multipliers.size() || hp.ple_n_heads > hp.ple_head_vocab_sizes.size() ||
            hp.ple_heads_per_ngram == 0 || (uint64_t) (hp.ple_ngram_size - 1)*hp.ple_heads_per_ngram != hp.ple_n_heads) {
        return nullptr;
    }
    for (uint32_t h = 0; h < hp.ple_n_heads; ++h) {
        const uint64_t end = (uint64_t) hp.ple_head_offsets[h] + hp.ple_head_vocab_sizes[h];
        if (hp.ple_head_vocab_sizes[h] == 0 || end > (uint64_t) t->ne[1] || end > (uint64_t) INT32_MAX + 1) {
            return nullptr;
        }
    }

    std::unique_ptr<llama_ple_prefetch> pf(new llama_ple_prefetch());
    pf->model     = &model;
    pf->ple       = t;
    pf->n_threads = (int) std::min<int64_t>(n_thr, 512);
    pf->batch_runs = (size_t) std::min<int64_t>(64, std::max<int64_t>(1, ple_pf_env("LLAMA_PLE_PF_BATCH_RUNS", 1)));
    // fork: a cold table (evicted by another process, first prompt after boot) is bound by disk queue depth, not by
    // the readers' CPU; LLAMA_PLE_PF_COLD_THREADS=0 keeps n_threads in every case
    pf->n_cold    = (int) std::min<int64_t>(std::max<int64_t>(0, ple_pf_env("LLAMA_PLE_PF_COLD_THREADS", PLE_PF_DEFAULT_COLD)), 512);
    pf->n_recheck = (uint32_t) std::min<int64_t>(std::max<int64_t>(0, ple_pf_env("LLAMA_PLE_PF_RECHECK", 0)), UINT32_MAX);
    // by default only long calls (a short check or catch-up batch is cheaper to fault in) and the whole call ahead
    pf->n_min     = (uint32_t) std::min<int64_t>(UINT32_MAX,
            std::max<int64_t>(1, ple_pf_env("LLAMA_PLE_PF_MIN", user ? 256 : PLE_PF_DEFAULT_MIN)));
    pf->n_ahead   = (uint64_t) std::max<int64_t>(0, ple_pf_env("LLAMA_PLE_PF_AHEAD", 1000000));
    // fork: cap read-ahead in ubatches; zero keeps the existing token-based setting.
    const int64_t window = ple_pf_env("LLAMA_PLE_PF_WINDOW_UBATCHES", 0);
    if (window > 0) {
        const uint64_t cap = (uint64_t) std::min<int64_t>(window, 64)*(uint64_t) n_ubatch;
        pf->n_ahead = std::min(pf->n_ahead, cap);
    }
    pf->n_ubatch  = n_ubatch;
    pf->row_size  = t->nb[1];
    pf->n_rows    = (uint64_t) t->ne[1];

    const char * mode = getenv("LLAMA_PLE_PF_MODE");
    pf->coalesce = mode && strcmp(mode, "coalesce") == 0;
    pf->bounded_window = window > 0;
    pf->use_pread = mode && strcmp(mode, "fault") != 0;
    pf->use_advise = mode && strcmp(mode, "advise") == 0;
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
            pf->use_advise = false;
            pf->coalesce = false;
        } else {
            pf->file_off = off;
            // the descriptor's own readahead state: read exactly the pages asked for
            posix_fadvise(pf->fd, 0, 0, POSIX_FADV_RANDOM);
        }
    }
#else
    pf->use_pread = false;
    pf->use_advise = false;
    pf->coalesce = false;
#endif

    LLAMA_LOG_WARN("%s: [PLEPF] PLE prefetch on: %d threads, %s, calls of %u+ tokens, %" PRIu64 " tokens ahead\n", __func__,
            pf->n_threads, pf->coalesce ? "coalesced pread" : pf->use_advise ? "advise" : pf->use_pread ? "pread" : "page faults", pf->n_min, pf->n_ahead);
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

static constexpr int64_t PLE_PF_PENDING = INT64_MAX/4; // remaining[] of a chunk build() has not reached yet

bool llama_ple_prefetch::start(const llama_batch_ext & batch, llama_pos pos0) {
    const size_t n = batch.tokens.size();
    if (n < n_min) {
        stop();
        return false;
    }
    llama_seq_id seq = -1;
    for (const auto & tok : batch.tokens) {
        if (tok.seq_ids.size() != 1) {
            stop();
            return false;
        }
        const llama_seq_id s = *tok.seq_ids.begin();
        if (seq < 0) {
            seq = s;
        } else if (s != seq) {
            stop();
            return false;
        }
    }

    const auto & hp = model->hparams;
    const llama_token img = hp.ple_image_token_id != 0 ? (llama_token) hp.ple_image_token_id : (llama_token) hp.ple_eos_token_id;

    // [PFGAP] the call the caller hinted: keep its reads (the batch may stop early, the reads past it are only spare)
    if (running && hinted && pos0 == pos0_call && n <= toks.size()) {
        bool same = true;
        for (size_t i = 0; i < n && same; ++i) {
            same = (batch.tokens[i].has_embd ? img : batch.tokens[i].id) == toks[i];
        }
        if (same) {
            hinted = false;
            return true;
        }
    }

    stop();
    toks.resize(n);
    for (size_t i = 0; i < n; ++i) {
        toks[i] = batch.tokens[i].has_embd ? img : batch.tokens[i].id;
    }
    begin(pos0);
    return true;
}

void llama_ple_prefetch::hint(const llama_token * tokens, size_t n, llama_pos pos0) {
    if (!tokens || n < n_min) {
        return;
    }
    if (running && pos0 == pos0_call && n == toks.size() && std::equal(tokens, tokens + n, toks.begin())) {
        return;
    }
    stop();
    toks.assign(tokens, tokens + n);
    begin(pos0);
    hinted = true;
}

void llama_ple_prefetch::begin(llama_pos pos0) {
    const size_t n = toks.size();
    pos0_call = pos0;
    seq_start = pos0 == 0;
    hinted    = false;
    chunk_size = coalesce ? std::max<size_t>(CHUNK, n) : CHUNK;
    // fork: small windows must not read a whole 256-token fault chunk ahead.
    if (bounded_window) {
        chunk_size = std::max<size_t>(1, coalesce ? n_ubatch : std::min<uint32_t>(CHUNK, n_ubatch));
    }

    // A row can start within a page and can span more than two pages.
    const uint64_t pages_per_row = 1 + (row_size - 1)/4096 + ((row_size - 1)%4096 != 0);
    const size_t max_runs = std::numeric_limits<size_t>::max()/sizeof(uint64_t) - 1;
    const size_t n_heads = model->hparams.ple_n_heads;
    if (pages_per_row > max_runs || n > max_runs/n_heads || n*n_heads > max_runs/pages_per_row) {
        throw std::length_error("PLE prefetch page count exceeds array size");
    }
    // fork: global page dedup means the run count cannot exceed the table's page span.
    // The extra page covers an unaligned table start in both fault and file-read modes.
    const uint64_t table_bytes = n_rows*row_size;
    const uint64_t table_pages = table_bytes/4096 + (table_bytes % 4096 != 0) + 1;
    const size_t cap = std::min(n*n_heads*(size_t) pages_per_row, (size_t) table_pages) + 1;
    run_key.reset(new uint64_t[cap]);
    run_len.reset(new uint32_t[cap]);
    run_chunk.reset(new uint32_t[cap]);
    n_chunks = n/chunk_size + (n % chunk_size != 0);
    remaining.reset(new std::atomic<int64_t>[n_chunks]);
    for (size_t c = 0; c < n_chunks; ++c) {
        remaining[c].store(PLE_PF_PENDING);
    }

    stopping  = false;
    built     = false;
    cursor    = 0;
    n_built   = 0;
    allowed_chunks = chunks_for((uint64_t) n_ubatch + n_ahead);
    n_read    = 0;
    n_pages_all = 0;
    consumed  = 0;
    t_wait_us = 0;
    n_ubatch_c = 0;
    n_ubatch_u = 0;
    t_build_us = 0;
    t_last_read_us = 0;
    t_start_us = ggml_time_us();
    running    = true;

    coord = std::thread([this]() { build(); });
    llama_thread_off_caller_cpu(coord); // the readers it starts inherit its CPU set
}

size_t llama_ple_prefetch::chunks_for(uint64_t n_tok) const {
    return (size_t) std::min<uint64_t>(n_chunks, n_tok/chunk_size + (n_tok % chunk_size != 0));
}

bool llama_ple_prefetch::run_ready(size_t i) const {
    if (stopping.load()) {
        return true;
    }
    return i < n_built.load() ? run_chunk[i] < allowed_chunks.load() : built.load();
}

void llama_ple_prefetch::build() {
    const auto & hp = model->hparams;

    const int64_t n        = (int64_t) toks.size();
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;

    // [PFGAP] the readers start first and follow the chunks as they are published
    std::vector<int64_t> last(std::max(n_threads, n_cold), 0);
    readers.clear();
    readers.reserve(last.size());
    n_readers = n_threads;
    for (int i = 0; i < n_threads; ++i) {
        readers.emplace_back([this, i, &last]() { last[i] = reader(); });
    }

    const uint64_t base  = use_pread ? file_off : (uint64_t) (uintptr_t) ple->data;
    const uint64_t key0  = base / 4096;
    const uint64_t nkeys = (base + n_rows*row_size + 4095)/4096 - key0 + 1;
    std::vector<uint64_t> seen(nkeys/64 + (nkeys % 64 != 0), 0);
    std::vector<uint64_t> cp;   // the new pages of one chunk
    cp.reserve((size_t) CHUNK*n_heads*2);

    // the rows, as llm_graph_input_qwen4exp_ple::set_input computes them; a predecessor before the call is unknown,
    // unless the call starts the sequence (then it reads as EOS)
    size_t nr = 0;
    for (size_t c = 0; c < n_chunks && !stopping.load(std::memory_order_relaxed); ++c) {
        const int64_t i0 = (int64_t) (c*chunk_size);
        const int64_t i1 = i0 + (int64_t) std::min<size_t>(n - i0, chunk_size);
        cp.clear();
        for (int64_t i = i0; i < i1; ++i) {
            uint64_t mixed = (uint64_t) toks[i] * hp.ple_layer_multipliers[0];
            bool cut = false;
            for (int64_t s = 1; s < n_gram; ++s) {
                if (!cut && i - s < 0) {
                    if (!seq_start) {
                        break;
                    }
                    cut = true;
                }
                const int64_t t = cut ? -1 : toks[i - s];
                cut = cut || t < 0 || t == eos;
                mixed ^= (uint64_t) (cut ? eos : t) * hp.ple_layer_multipliers[s];
                const int64_t hb = (s - 1) * per_gram;
                for (int64_t k = 0; k < per_gram; ++k) {
                    const int64_t h = hb + k;
                    const int32_t r = (int32_t) (mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h]);
                    if (r < 0 || (uint64_t) r >= n_rows) {
                        continue;
                    }
                    const uint64_t b = base + (uint64_t) r*row_size;
                    const uint64_t k0 = b/4096, k1 = (b + row_size - 1)/4096;
                    for (uint64_t page = k0; page <= k1; ++page) {
                        const uint64_t index = page - key0;
                        const uint64_t bit = UINT64_C(1) << (index % 64);
                        uint64_t & word = seen[index/64];
                        if (!(word & bit)) {
                            word |= bit;
                            cp.push_back(page);
                        }
                    }
                }
            }
        }
        std::sort(cp.begin(), cp.end());
        // fork: a hot shared prefix can hide cold pages in the rest of the call.
        const bool sample = c == 0 || (n_recheck > 0 && c % n_recheck == 0);
        if (sample && n_readers == n_threads && n_cold > n_threads && !cp.empty()) {
            const double res = resident_fraction(cp, base);
            if (res < 0.5) {
                for (int i = n_threads; i < n_cold; ++i) {
                    readers.emplace_back([this, i, &last]() { last[i] = reader(); });
                }
                n_readers = n_cold;
                LLAMA_LOG_INFO("%s: [PLEPF] cold table (%.0f%% of sampled pages cached): %d readers at chunk %zu\n", __func__, 100.0*res, n_cold, c);
            }
        }
        const size_t r0 = nr;
        for (size_t j = 0; j < cp.size(); ++j) {
            // fork: bounded windows must not pull unused or already-read pages into the cache.
            const uint64_t gap = coalesce && !bounded_window ? 16 : 0;
            if (nr > r0 && cp[j] <= run_key[nr - 1] + run_len[nr - 1] + gap && cp[j] - run_key[nr - 1] < RUN_MAX) {
                run_len[nr - 1] = (uint32_t) (cp[j] - run_key[nr - 1] + 1);
            } else {
                run_key[nr] = cp[j];
                run_len[nr] = 1;
                run_chunk[nr] = (uint32_t) c;
                ++nr;
            }
        }
        n_pages_all += cp.size();
        {
            std::lock_guard<std::mutex> lk(mtx);
            remaining[c].store((int64_t) (nr - r0));
            n_built.store(nr);
        }
        cv_work.notify_all();
        if (nr == r0) {
            cv_done.notify_all();
        }
    }
    t_build_us = ggml_time_us() - t_start_us;
    {
        std::lock_guard<std::mutex> lk(mtx);
        built = true;
    }
    cv_work.notify_all();
    cv_done.notify_all();

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

// share of up to 256 of the given page keys (spread over the list) that are in the page cache; 1 if unknown
double llama_ple_prefetch::resident_fraction(const std::vector<uint64_t> & keys, uint64_t base) const {
#if defined(__linux__)
    // keys are file pages in pread/advise mode: the mapping keeps the file's offset within a page
    const uintptr_t to_virt = use_pread ? (uintptr_t) ple->data - (uintptr_t) base : 0;
    const size_t step = std::max<size_t>(1, keys.size()/256);
    size_t n = 0, in = 0;
    for (size_t j = 0; j < keys.size(); j += step) {
        unsigned char v = 0;
        if (mincore((void *) (keys[j]*4096 + to_virt), 4096, &v) != 0) {
            return 1.0;
        }
        in += v & 1;
        ++n;
    }
    return n ? (double) in/n : 1.0;
#else
    GGML_UNUSED(keys);
    GGML_UNUSED(base);
    return 1.0;
#endif
}

int64_t llama_ple_prefetch::reader() {
    std::vector<char> buf(use_pread && !use_advise ? (size_t) RUN_MAX*4096 : 0);
    int64_t t_last = 0;
    int64_t pages_read = 0;
    size_t next = 0, end = 0;
    size_t done_chunk = 0;
    int64_t done_runs = 0;
    auto flush_done = [&]() {
        if (done_runs > 0) {
            t_last = ggml_time_us();
        }
        if (done_runs > 0 && remaining[done_chunk].fetch_sub(done_runs) == done_runs) {
            std::lock_guard<std::mutex> lk(mtx);
            cv_done.notify_all();
        }
        done_runs = 0;
    };
    for (;;) {
        if (next == end) {
            next = cursor.fetch_add(batch_runs);
            end = next + batch_runs;
        }
        const size_t i = next++;
        if (!run_ready(i)) {
            // fork: release completed chunks before waiting for the consumer to advance the window.
            flush_done();
            std::unique_lock<std::mutex> lk(mtx);
            cv_work.wait(lk, [&]() { return run_ready(i); });
        }
        if (stopping.load() || i >= n_built.load()) {
            break;
        }
        if (done_runs > 0 && done_chunk != run_chunk[i]) {
            flush_done();
        }
        const uint64_t off = run_key[i]*4096, len = (uint64_t) run_len[i]*4096;
#if defined(__linux__)
        if (use_advise) {
            posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
        } else if (use_pread) {
            // fork: finish short reads, but do not require padding past the table's last byte.
            const size_t need = (size_t) std::min(len, file_off + n_rows*row_size - off);
            size_t done = 0;
            while (done < need && !stopping.load()) {
                const ssize_t got = pread(fd, buf.data() + done, need - done, (off_t) (off + done));
                if (got < 0 && errno == EINTR) {
                    continue;
                }
                if (got <= 0) {
                    break;
                }
                done += (size_t) got;
            }
            if (done != need) {
                // fork: leave an incomplete run uncounted and let the gather use its normal path.
                {
                    std::lock_guard<std::mutex> lk(mtx);
                    stopping = true;
                }
                cv_work.notify_all();
                cv_done.notify_all();
                break;
            }
        } else
#endif
        {
            char sink = 0;
            for (uint64_t o = 0; o < len; o += 4096) {
                sink ^= *(const volatile char *) (uintptr_t) (off + o);
            }
            GGML_UNUSED(sink);
        }
        if (batch_runs == 1) {
            n_read.fetch_add(run_len[i], std::memory_order_relaxed);
        } else {
            pages_read += run_len[i];
        }
        done_chunk = run_chunk[i];
        ++done_runs;
        if (next == end) {
            flush_done();
        }
    }
    flush_done();
    if (batch_runs > 1) {
        n_read.fetch_add(pages_read, std::memory_order_relaxed);
    }
    return t_last;
}

bool llama_ple_prefetch::consume(uint32_t n_tokens, const int32_t * idx, size_t n_idx) {
    if (!running || n_tokens == 0) {
        return false;
    }
    const int64_t tw = ggml_time_us();
    const uint64_t t0 = consumed, t1 = t0 + n_tokens;
    consumed = t1;
    if (t1 > toks.size()) {
        ++n_ubatch_u;
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(mtx);
        const size_t a = chunks_for(t1 + n_ahead);
        if (a > allowed_chunks.load()) {
            allowed_chunks.store(a);
        }
    }
    cv_work.notify_all();

    const size_t c0 = (size_t) (t0/chunk_size), c1 = (size_t) ((t1 - 1)/chunk_size);
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
    if (stopping) {
        return false;
    }
    t_wait_us += ggml_time_us() - tw;
    ++n_ubatch_c;

    GGML_UNUSED(idx);
    GGML_UNUSED(n_idx);
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
    hinted  = false;

    LLAMA_LOG_WARN("%s: [PLEPF] %zu tokens: %zu pages (%.0f MiB) in %zu reads, guessed in %.1f ms, %" PRId64 " read, last read at %.1f ms; "
            "%d ubatches waited %.1f ms in all, %d readers%s", __func__, toks.size(), n_pages_all, n_pages_all*4096.0/1048576.0,
            (size_t) n_built.load(), t_build_us.load()/1e3, n_read.load(), t_last_read_us.load()/1e3, n_ubatch_c, t_wait_us/1e3,
            n_readers, n_ubatch_u ? " (some ubatches fell outside the call)" : "");
    LLAMA_LOG_CONT("\n");

    toks.clear();
    run_key.reset();
    run_len.reset();
    run_chunk.reset();
    remaining.reset();
    n_chunks = 0;
    n_built  = 0;
}
