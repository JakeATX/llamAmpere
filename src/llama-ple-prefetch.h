#pragma once

// [PLEPF] PLE page prefetch (default 4 readers, LLAMA_PLE_PREFETCH=<threads>, 0 = off, qwen4exp only).
//
// The per_layer_token_embd table (27 GiB on Flash-Next) stays memory-mapped and mostly out of the page cache, and every
// prompt token gathers ple_n_heads rows of it at scattered places. The normal input step reads the rows of one ubatch at
// a time with a one-thread madvise loop, and the CPU gather then waits on the disk while the GPU sits idle (about 1 s
// per 12,288-token ubatch on a 980 PRO when the table is cold).
//
// At the start of a prompt call this computes the rows of every token of the call on its own (the same n-gram hash as
// the graph input, with the predecessors taken from the call itself), and many threads read their pages into the page
// cache in ubatch order, a few ubatches ahead of the gather. The gather then finds its pages resident. Reads only: the
// result of the gather cannot change, a wrong guess only costs time.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

struct llama_model;
struct llama_hparams;
struct llama_batch_ext;
struct ggml_tensor;

// prev holds ngram_size - 1 tokens, oldest first; missing tokens use LLAMA_TOKEN_NULL.
void llama_ple_hash_rows(const llama_hparams & hp, llama_token token, const llama_token * prev, int32_t * rows);

class llama_ple_prefetch {
public:
    // nullptr when LLAMA_PLE_PREFETCH=0, when the default meets a low-MemAvailable host, or unless the model has a lazily mapped PLE table this can read
    static std::unique_ptr<llama_ple_prefetch> create(const llama_model & model, uint32_t n_ubatch);

    ~llama_ple_prefetch();

    // a decode call begins: start reading the pages of its tokens (one sequence, at least LLAMA_PLE_PF_MIN tokens).
    // Returns false when this call is not covered.
    // pos0 = position of the first token (0: the predecessors before the call read as EOS)
    bool start(const llama_batch_ext & batch, llama_pos pos0);

    // [PFGAP] the caller knows the tokens of the next prompt call before decode builds its batch: start reading now.
    // A following start() of these tokens (or of a prefix of them) at the same pos0 keeps the reads going.
    void hint(const llama_token * tokens, size_t n, llama_pos pos0);

    // the next ubatch of this call (n_tokens, in call order) is about to gather: waits until its pages are read and lets
    // the readers move on. idx = the rows the graph input computed.
    // Returns true when the ubatch was covered, so the caller skips its own madvise.
    bool consume(uint32_t n_tokens, const int32_t * idx, size_t n_idx);

    // the decode call ends: stop the readers and log the totals
    void stop();

    // fork: LLAMA_PLE_DECODE=1 (opt-in, prompt ubatches of at least LLAMA_PLE_DECODE_MIN tokens): the graph takes the
    // gathered rows as an F32 input instead of get_rows on the mapped table. Host threads decode the next ubatch of the
    // call while the card runs this one and keep the rows until that ubatch takes them, so a page evicted after its
    // read costs no wait on the card's critical path. Same to_float as the CPU get_rows, so the same bits.
    bool     decode_enabled() const { return decode_on; }
    uint32_t decode_min() const { return n_decode_min; }
    // called after consume(): writes the rows idx (n_idx of them) to dst, F32 [row elements, n_idx]
    void gather(uint32_t n_tokens, const int32_t * idx, size_t n_idx, ggml_tensor * dst);

private:
    llama_ple_prefetch() = default;

    void begin(llama_pos pos0); // toks is set: reset the call state and start the coordinator
    size_t fit_pages(size_t requested) const;
    void build();   // runs on the coordinator thread: rows and page runs chunk by chunk, readers follow behind
    int64_t reader();  // one reader thread, returns the time of its last read
    size_t chunks_for(uint64_t n_tok) const;
    bool   run_ready(size_t i) const; // under mtx: run i exists and may be read, or there is nothing more to read
    double resident_fraction(const std::vector<uint64_t> & keys, uint64_t base) const;
    void   verify_rows(const int32_t * idx, size_t n_idx); // fork: re-queue pages evicted after their read
    void   test_drop_rows(const int32_t * idx, size_t n_idx); // fork: test only, page out N% of the ubatch's rows
    void   decode_into(const int32_t * idx, size_t n_idx, float * dst) const; // fork: rows to F32 on n_decode threads
    void   decode_join();

    // fork: LLAMA_PLE_DECODE state. dec_t0 = call offset of the ubatch in dec_rows/dec_data (UINT64_MAX: none)
    bool     decode_on    = false;
    int      n_decode     = 16;
    uint32_t n_decode_min = 1024;
    uint64_t dec_t0       = UINT64_MAX;
    std::vector<int32_t> dec_rows;
    std::vector<float>   dec_data;
    // fork: LLAMA_PLE_DECODE_PINNED (default 1 with decode): rows land in page-locked host memory of the input's
    // device, so the upload is a direct DMA instead of a staged pageable copy. dec_ptr is dec_pin or dec_data.
    bool                  dec_pinned = true;
    ggml_backend_buffer_t dec_pin    = nullptr;
    size_t                dec_cap    = 0; // floats
    float *               dec_ptr    = nullptr;
    float * decode_buf(size_t n, const ggml_tensor * dst); // main thread only, decode thread joined
    std::thread          dec_thread;
    int      n_dec_hit = 0, n_dec_miss = 0;
    int64_t  t_dec_wait_us = 0;

    const llama_model * model = nullptr;
    const ggml_tensor * ple   = nullptr;

    // configuration
    int      n_threads = 0;
    int      n_cold    = 0;      // readers once a sample finds the table mostly uncached (0 = never more)
    int      n_readers = 0;      // readers of the current call
    size_t   batch_runs = 1;     // runs claimed per shared cursor update
    uint32_t n_recheck = 0;      // chunks between later residency samples (0 = first chunk only)
    bool     use_pread = true;
    int      populate = 0;      // 1: touch after reads; 2: bulk after reads; 3: try bulk without scratch reads
    bool     use_advise = false; // [PFGAP] file mode: ask the kernel to queue the reads instead of copying them
    bool     coalesce = false;
    bool     bounded_window = false;
    bool     verify = false;     // fork: LLAMA_PLE_PF_VERIFY=1 checks each consumed ubatch's pages are still cached
    std::vector<unsigned char> verify_map;   // mincore result over one page cluster (one byte per page)
    std::vector<uint64_t>      verify_pages; // pages of the ubatch found evicted
    bool     verify_sync = false; // fork: LLAMA_PLE_PF_VERIFY=2 reads the evicted pages back before the gather
    std::vector<std::pair<uint64_t, size_t>> verify_runs; // (first page, page count) of the evicted pages
    int64_t  n_verify_missing = 0;           // pages of the call found evicted at consume time
    int     n_verify_ubatch  = 0;            // ubatches with at least one evicted page
    int64_t  t_verify_us      = 0;
    int      test_drop        = 0;           // fork: LLAMA_PLE_PF_TEST_DROP percent of rows paged out (test only)
    int64_t  n_test_dropped   = 0;
    uint32_t n_min     = 256;
    uint32_t n_ubatch  = 0;
    uint64_t n_ahead   = 0;      // tokens read ahead of the gather
    static constexpr uint32_t CHUNK = 256; // tokens per completion chunk
    size_t chunk_size = CHUNK;

    // where the table lives
    int      fd       = -1;
    uint64_t file_off = 0;       // file offset of the table's first byte (pread mode)
    uint64_t row_size = 0;       // bytes per row
    uint64_t n_rows   = 0;

    // the current call
    std::vector<int32_t> toks;   // token ids of the call
    // [PFGAP] runs of consecutive pages, one read each, in chunk order (published chunk by chunk as build() goes)
    static constexpr uint32_t RUN_MAX = 256; // pages per run (1 MiB)
    std::unique_ptr<uint64_t[]> run_key;   // first page key (file page for pread/advise, virtual page for fault)
    std::unique_ptr<uint32_t[]> run_len;
    std::unique_ptr<uint32_t[]> run_chunk;
    std::unique_ptr<std::atomic<int64_t>[]> remaining; // unread runs per chunk (PENDING until the chunk is built)
    size_t n_chunks = 0;
    std::atomic<size_t> n_built{0};          // runs published
    std::atomic<size_t> allowed_chunks{0};   // readers may read runs of chunks below this
    size_t   n_pages_all = 0;
    llama_pos pos0_call  = 0;
    bool     hinted      = false;            // the running call was started by hint()

    std::thread              coord;
    std::vector<std::thread> readers;
    std::mutex               mtx;
    std::condition_variable  cv_work;  // readers wait for the window
    std::condition_variable  cv_done;  // the consumer waits for chunks and for build()
    std::atomic<bool>        stopping{false};
    std::atomic<bool>        built{false};
    std::atomic<size_t>      cursor{0};
    std::atomic<int64_t>     n_read{0};
    std::atomic<int64_t>     n_map_runs{0};
    std::atomic<int64_t>     n_direct_runs{0};
    bool                     running = false;
    uint64_t                 consumed = 0; // tokens consumed in this call
    bool                     seq_start = false; // the call starts its sequence

    // totals of the call
    int64_t t_start_us = 0;
    std::atomic<int64_t> t_build_us{0};
    std::atomic<int64_t> t_last_read_us{0};
    int64_t t_wait_us  = 0;
    int     n_ubatch_c = 0;
    int     n_ubatch_u = 0; // consume() calls that fell outside the call (not covered)
};
