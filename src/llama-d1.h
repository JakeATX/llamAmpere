#pragma once

// [D1] opt-in exact rescore of the coarse output head (LLAMA_D1_HEAD=<sidecar.gguf>, or @model/<file> beside the model).
// The card runs a coarse output head: a type-128 (AIT2_U2_512) overlay, or the 8 GB file's own type-125 (AIF25_U2) head
// ([EXACTHEAD]). The host keeps an exact copy of the original output rows in a sidecar GGUF (F16 or Q8_0 decoded from
// the base head, or BF16 from the source checkpoint with --exact-from; made offline by scripts/d1_make_sidecar.py). For every output
// row a consumer reads, the coarse top N logits are recomputed as dot(h, exact_row) with h = the post-norm
// hidden row (res->t_embd). Default off; nothing here runs unless LLAMA_D1_HEAD is set.
// Design: LSA/REAL8GB/DESIGN_d1_head_rescore.md (r3).

#include "ggml.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// fingerprint of a tensor's bytes: size plus FNV-1a 64 over its first and last 1 MiB (both ends, in that order)
uint64_t llama_d1_fnv64_ends(const uint8_t * data, size_t nbytes);

// the base file's output.weight, recorded by the model loader before any overlay patch replaces it
struct llama_d1_src {
    bool        valid  = false;
    int         type   = -1;
    size_t      nbytes = 0;
    uint64_t    fnv    = 0;
};

// read [first 1 MiB][last 1 MiB] of a tensor at offs in a file and fingerprint it; false on I/O error
bool llama_d1_src_from_file(const std::string & path, size_t offs, size_t nbytes, int type, llama_d1_src & out);

// exact rows: one mmap'd sidecar, shared by every context of one model
struct llama_d1_head {
    ~llama_d1_head();

    std::string path;
    ggml_type   type    = GGML_TYPE_F16; // GGML_TYPE_F16, GGML_TYPE_BF16 or GGML_TYPE_Q8_0
    int64_t     n_embd  = 0;
    int64_t     n_vocab = 0;
    size_t      row_bytes = 0;
    const uint8_t * rows = nullptr; // n_vocab rows of row_bytes
    uint64_t    src_fnv  = 0;
    uint64_t    self_fnv = 0;

    // [EXACTHEAD] optional certificate stats (tensor output_exact.gl2, F32 [n_embd/32, n_vocab]): the L2 norm of
    // (coarse row - exact row) over each 32-column group. nullptr when the sidecar has none.
    const float *      gl2     = nullptr;
    int64_t            n_group = 0;
    std::vector<float> gl2_rowmax;      // [n_vocab] max over groups of gl2
    float              gl2_max = 0.0f;  // max over rows of gl2_rowmax

    // mapping
    void * map_base = nullptr;
    size_t map_size = 0;
#ifdef _WIN32
    void * h_file = nullptr;
    void * h_map  = nullptr;
#endif

    float dot(const float * h, int64_t id) const;
};

// open + check (a) shape/type/KV rows_type, (b) self fingerprint, (c) source fingerprint == src.
// Returns nullptr and sets err on any failure.
std::shared_ptr<llama_d1_head> llama_d1_head_open(const char * path, const llama_d1_src & src,
                                                  int64_t n_embd, int64_t n_vocab, std::string & err);

// per-model slot (lives in llama_model): the handle is opened once, on the first context that needs it
struct llama_d1_model_slot {
    llama_d1_src                   src;
    std::mutex                     mtx;
    std::shared_ptr<llama_d1_head> head;
    std::string                    err;   // sticky open error
    bool                           tried = false;
};

// scope decision for one context (design 3.1)
enum llama_d1_scope_result {
    LLAMA_D1_SCOPE_UNSET = 0,  // LLAMA_D1_HEAD not set: nothing, no log
    LLAMA_D1_SCOPE_OFF   = 1,  // off for this context, log line, no error
    LLAMA_D1_SCOPE_WARN  = 2,  // off, WARN level (the target's own base head: the type-128 overlay did not load)
    LLAMA_D1_SCOPE_ON    = 3,  // on (handle open and checked)
    LLAMA_D1_SCOPE_ERROR = 4,  // context creation must fail
};

struct llama_d1_scope_in {
    const char * head_path   = nullptr; // LLAMA_D1_HEAD
    bool         ctx_default = true;    // ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT
    bool         embeddings  = false;
    bool         mtp_chain   = false;
    bool         no_alloc    = false;   // model loaded no_alloc (fit probe)
    int          out_type    = -1;      // live output.weight type (-1: none)
    int64_t      out_ne0     = 0;
    int64_t      out_ne1     = 0;
    int64_t      n_embd      = 0;
    int64_t      n_vocab     = 0;
};

// decides and (for type 128) opens the model's handle. msg gets the log text for OFF/WARN/ERROR.
llama_d1_scope_result llama_d1_scope(const llama_d1_scope_in & in, llama_d1_model_slot & slot,
                                     std::shared_ptr<llama_d1_head> & head_out, std::string & msg);

// top-n ids of a logits row (ties to the lower id), any order; n <= n_vocab
void llama_d1_select_top(const float * row, int64_t n_vocab, int n, std::vector<int32_t> & ids);

struct llama_d1_cand { float v; int32_t id; };

// [EXACTHEAD] fused pick, one scan of row[i0, i1): heap = the range's top n (llama_d1_cand, any order); if collect,
// buf (appended, ascending ids) = every id ever kept plus every id with row[i] >= fl(thr_t - D), thr_t the running
// n-th value when i was read. Returns the final fl(thr - D) (-inf if the range has <= n ids): every id of the range
// with row[i] >= that value is in buf.
float llama_d1_scan_chunk(const float * row, int64_t i0, int64_t i1, int n, float D, bool collect,
                          std::vector<llama_d1_cand> & heap, std::vector<int32_t> & buf);

// [EXACTHEAD] certificate: after the top N are exact, any other row whose coarse logit could still reach the k-th
// exact logit (coarse + margin + sigma * s_i >= T_k, s_i = sqrt(sum_g ||h_g||^2 * gl2[i][g]^2 / 32)) is rescored too.
// Statistically exact (Jake 11-21): treating each row's coarse error as Gaussian with std s_i, a row that belongs in
// the exact top k is left coarse with probability ~1e-9 (one-sided 6 sigma). Default on inside the add-on when the
// sidecar has output_exact.gl2; LLAMA_D1_CERT=0 is the kill switch. Study: 0 misses in 16,200 positions.
struct llama_d1_cert {
    bool  on     = false; // the context turns it on by default when the sidecar has gl2
    int   k      = 20;    // the sampler's top-k (all presets use 20)
    float sigma  = 6.0f;
    float margin = 0.1f;  // card activation quantization vs the host dot (measured max 0.071 on the 8 GB head)
};

// per-context state (design 3.3-3.5)
struct llama_d1_ctx {
    enum : uint8_t {
        ROW_NONE    = 0, // no full-vocab logits copied for this row (or not a vocab row): never touched
        ROW_PENDING = 1, // full-vocab logits + h copied, not rescored yet
        ROW_EXACT   = 2, // rescored
        ROW_COARSE  = 3, // full-vocab logits copied on this d1 context but not rescorable (N = 0 or no h)
    };

    llama_d1_ctx(std::shared_ptr<llama_d1_head> head, int n_top, int eager_max, int n_threads,
                 const llama_d1_cert & cert = llama_d1_cert());
    ~llama_d1_ctx();

    std::shared_ptr<llama_d1_head> head;
    int n_top     = 256;
    int eager_max = 64;
    llama_d1_cert cert;

    std::vector<float>   h;      // [rows x n_embd]
    std::vector<uint8_t> state;  // [rows]
    std::vector<uint8_t> seen;   // [rows] read at least once since the rows were produced
    int64_t              n_rows = 0;

    // counters
    std::atomic<uint64_t> n_rescored{0};
    std::atomic<uint64_t> n_read_distinct{0};
    std::atomic<uint64_t> n_read_coarse{0};
    std::atomic<uint64_t> ns_rescore{0}; // wall of rescore calls (caller thread)
    std::atomic<uint64_t> ns_select{0};  // sum of per-row select time (all threads)
    uint64_t              n_h_mismatch = 0;
    // [D0] split of the rescore wall (per-request deltas via llama_d1_log_counters)
    std::atomic<uint64_t> n_calls{0};
    std::atomic<uint64_t> ns_dot{0};     // sum of per-row dot time (all threads)
    std::atomic<uint64_t> ns_wait{0};    // caller wait for the pool after its own share
    std::atomic<uint64_t> n_minflt{0};   // process page faults during rescore calls (getrusage RUSAGE_SELF)
    std::atomic<uint64_t> n_majflt{0};
    std::atomic<uint64_t> n_nvcsw{0};    // process context switches during rescore calls (voluntary / involuntary)
    std::atomic<uint64_t> n_nivcsw{0};
    // [EXACTHEAD] certificate counters
    std::atomic<uint64_t> n_cert_rows{0};    // rows certified
    std::atomic<uint64_t> n_cert_sigma{0};   // non-candidate rows whose s_i was computed
    std::atomic<uint64_t> n_cert_forced{0};  // non-candidate rows rescored by the certificate
    std::atomic<uint64_t> n_cert_maxrow{0};  // most rows forced in one row
    uint64_t              last[14] = {};

    // [EXACTHEAD] LLAMA_D1_AUDIT=n (default 0 = off; slow, for the exactness gate only): every n-th rescored row is
    // also scored against all n_vocab exact rows. Compared on the preset sampler (top-k 20, temp 1, top-p 0.95):
    // the served row vs the full exact row, and the coarse row as it was before the rescore vs the full exact row.
    int audit_every = 0;
    std::atomic<uint64_t> audit_tick{0};
    std::mutex audit_mtx;
    struct audit_t {
        uint64_t rows = 0, top_same = 0, dist_same = 0, coarse_top_same = 0;
        double   tv_sum = 0, tv_max = 0, coarse_tv_sum = 0, coarse_tv_max = 0, dlogit_max = 0;
        double   z_max = 0;           // max |coarse - exact| / s_i over the exact top 20 (needs gl2)
        uint64_t z_gt6 = 0, z_n = 0;
    } audit;
    void audit_row(const float * served, const float * coarse, const float * hr, int64_t n_vocab);
    std::string audit_summary();  // values at the previous summary_delta
    std::string summary_delta();         // counters since the previous call (not thread safe: one reader)

    // a new set of output rows (output_reserve): all rows NONE, seen cleared
    void reset(int64_t n_rows_max);
    // rows [r0, r0 + n) got full-vocab logits; has_h: their h rows were copied
    void mark_copied(int64_t r0, int64_t n, bool has_h);
    float * h_row(int64_t r) { return h.data() + r * head->n_embd; }

    // accessor hooks; logits points at row 0 of the n_out output rows of the current call
    void on_read_row(float * logits, int64_t n_vocab, int64_t j, int64_t n_out);
    void on_read_all(float * logits, int64_t n_vocab, int64_t n_out);

    void swap_rows(int64_t i0, int64_t i1);

    // rescore the PENDING rows in list (threaded when more than one)
    // need >= 0 (fused + pool): may return once row need is exact, the pool finishing the rest (drain() waits)
    void rescore_rows(float * logits, int64_t n_vocab, const std::vector<int64_t> & rows, int64_t need = -1);
    void drain();                // wait for async rows (every other entry and llama_context::synchronize call it)
    bool async_pending = false;
    std::vector<int64_t> arows;  // rows of the current pool job
    void rescore_one(float * logits, int64_t n_vocab, int64_t r, std::vector<int32_t> & ids);
    void certify(float * row, int64_t n_vocab, const float * hr, const std::vector<int32_t> & ids);
    struct cert_prep_t { float T = 0, pre = 0, hn = 0; std::vector<float> hg; std::vector<int32_t> sids; };
    void cert_prepare(const float * row, const float * hr, const std::vector<int32_t> & ids, cert_prep_t & cp) const;
    void cert_range(float * row, const float * hr, const cert_prep_t & cp, int64_t i0, int64_t i1,
                    uint64_t & n_sigma, uint64_t & n_forced) const;
    void cert_count(uint64_t n_sigma, uint64_t n_forced);
    float cert_slack(const float * hr) const; // D: pre = T - D (a hair below margin + sigma * gl2_max * hn)

    // [EXACTHEAD] fused pick (LLAMA_D1_FUSED=1; default off until gated). Each row is split in n_chunk parts over the
    // pool, in three phases: (1) one scan per chunk keeps its top N, the last chunk merges the row's ids; (2) the
    // row's dots are split the same way, the last part prepares the certificate (T, pre); (3) the certificate scan
    // runs per chunk, each part reading and writing only its range; the last part counts and marks the row done.
    // Same rows as the separate path. Jobs run in row order, so the first read can return once its row is done.
    bool fused = false;
    int64_t fused_min_chunk = 16384;           // ids per chunk at least (tests lower it)
    struct fchunk_t { std::vector<llama_d1_cand> heap; std::vector<int32_t> buf; float floor = 0.0f; };
    struct frow_t {
        std::vector<int32_t> ids;
        cert_prep_t cp;
        std::atomic<uint64_t> n_sigma{0}, n_forced{0};
        std::vector<float>   coarse;           // audit only
        bool audit = false;
        std::atomic<int> left{0}, dleft{0}, ready{0}, done{0}, cleft{0}, cready{0};
    };
    std::vector<fchunk_t> fch;                 // [rows x n_chunk]
    std::unique_ptr<frow_t[]> frow;            // [rows]
    size_t                frow_n = 0;
    int                   n_chunk = 1;
    std::atomic<int64_t>  next2{0};
    void fused_scan(int64_t k);   // phase 1 job
    void fused_merge(int64_t ri);
    void fused_dots(int64_t k);   // phase 2 job
    void fused_cert(int64_t k);   // phase 3 job
    void fused_run(int64_t k);    // job k of the row-ordered list (scans, dot parts, certificate parts, per row)

    std::string summary() const;

    // condvar pool (design 3.5): workers sleep between batches, the caller takes a share and blocks until done
    struct pool_t {
        std::vector<std::thread> threads;
        std::mutex               mtx;
        std::condition_variable  cv_work;
        std::condition_variable  cv_done;
        uint64_t                 gen = 0;
        bool                     quit = false;
        int                      n_busy = 0;
        // current job
        float *                  logits = nullptr;
        int64_t                  n_vocab = 0;
        const std::vector<int64_t> * rows = nullptr;
        std::atomic<int64_t>     next{0};
    } pool;
    void worker(int idx);
    void run_job();
    int  n_threads_total() const { return (int) pool.threads.size() + 1; }
};
