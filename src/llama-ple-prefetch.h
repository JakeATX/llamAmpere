#pragma once

// [PLEPF] PLE page prefetch (opt-in, LLAMA_PLE_PREFETCH=<threads>, qwen4exp only).
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
struct llama_batch_ext;
struct ggml_tensor;

class llama_ple_prefetch {
public:
    // nullptr unless LLAMA_PLE_PREFETCH > 0 and the model has a lazily mapped PLE table this can read
    static std::unique_ptr<llama_ple_prefetch> create(const llama_model & model, uint32_t n_ubatch);

    ~llama_ple_prefetch();

    // a decode call begins: start reading the pages of its tokens (one sequence, at least LLAMA_PLE_PF_MIN tokens).
    // Returns false when this call is not covered.
    // pos0 = position of the first token (0: the predecessors before the call read as EOS)
    bool start(const llama_batch_ext & batch, llama_pos pos0);

    // the next ubatch of this call (n_tokens, in call order) is about to gather: waits until its pages are read and lets
    // the readers move on. idx = the rows the graph input computed.
    // Returns true when the ubatch was covered, so the caller skips its own madvise.
    bool consume(uint32_t n_tokens, const int32_t * idx, size_t n_idx);

    // the decode call ends: stop the readers and log the totals
    void stop();

private:
    llama_ple_prefetch() = default;

    void build();   // runs on the coordinator thread: rows, pages, then the readers
    int64_t reader();  // one reader thread, returns the time of its last read
    size_t allowed_for(uint64_t n_tok) const;

    const llama_model * model = nullptr;
    const ggml_tensor * ple   = nullptr;

    // configuration
    int      n_threads = 0;
    bool     use_pread = true;
    uint32_t n_min     = 256;
    uint32_t n_ubatch  = 0;
    uint64_t n_ahead   = 0;      // tokens read ahead of the gather
    static constexpr uint32_t CHUNK = 256; // tokens per completion chunk

    // where the table lives
    int      fd       = -1;
    uint64_t file_off = 0;       // file offset of the table's first byte (pread mode)
    uint64_t row_size = 0;       // bytes per row
    uint64_t n_rows   = 0;

    // the current call
    std::vector<int32_t> toks;   // token ids of the call
    std::vector<int32_t> pred;   // guessed rows, n_heads per token, -1 = unknown predecessor
    std::vector<uint64_t> pages; // page keys in read order (file page for pread, virtual page for fault)
    std::vector<uint32_t> page_chunk;
    std::vector<size_t>   chunk_end;   // pages[0, chunk_end[c]) cover chunks 0..c
    std::unique_ptr<std::atomic<int64_t>[]> remaining; // unread pages per chunk
    size_t n_chunks = 0;

    std::thread              coord;
    std::vector<std::thread> readers;
    std::mutex               mtx;
    std::condition_variable  cv_work;  // readers wait for the window
    std::condition_variable  cv_done;  // the consumer waits for chunks and for build()
    std::atomic<bool>        stopping{false};
    std::atomic<bool>        built{false};
    std::atomic<size_t>      cursor{0};
    std::atomic<size_t>      allowed{0};
    std::atomic<int64_t>     n_read{0};
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
