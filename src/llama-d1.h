#pragma once

// [D1] opt-in exact rescore of the coarse output head (LLAMA_D1_HEAD=<sidecar.gguf>).
// The card runs a coarse type-128 (AIT2_U2_512) output head. The host keeps an exact copy of the original
// output rows in a sidecar GGUF (F16 or Q8_0, made offline by scripts/d1_make_sidecar.py). For every output
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
    ggml_type   type    = GGML_TYPE_F16; // GGML_TYPE_F16 or GGML_TYPE_Q8_0
    int64_t     n_embd  = 0;
    int64_t     n_vocab = 0;
    size_t      row_bytes = 0;
    const uint8_t * rows = nullptr; // n_vocab rows of row_bytes
    uint64_t    src_fnv  = 0;
    uint64_t    self_fnv = 0;

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

// per-context state (design 3.3-3.5)
struct llama_d1_ctx {
    enum : uint8_t {
        ROW_NONE    = 0, // no full-vocab logits copied for this row (or not a vocab row): never touched
        ROW_PENDING = 1, // full-vocab logits + h copied, not rescored yet
        ROW_EXACT   = 2, // rescored
        ROW_COARSE  = 3, // full-vocab logits copied on this d1 context but not rescorable (N = 0 or no h)
    };

    llama_d1_ctx(std::shared_ptr<llama_d1_head> head, int n_top, int eager_max, int n_threads);
    ~llama_d1_ctx();

    std::shared_ptr<llama_d1_head> head;
    int n_top     = 256;
    int eager_max = 64;

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
    void rescore_rows(float * logits, int64_t n_vocab, const std::vector<int64_t> & rows);
    void rescore_one(float * logits, int64_t n_vocab, int64_t r, std::vector<int32_t> & ids);

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
};
