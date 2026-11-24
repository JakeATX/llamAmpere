#pragma once

// fork (NVMETIER, backlog #161): three-tier expert store for host-expert MoE (VRAM cache slots / RAM tier / NVMe),
// for hosts whose RAM cannot hold every host expert. Off by default; LLAMA_MOE_NVME_TIER=1 turns it on.
//
// The model's mmap'd expert tensors stay mapped as before (any reader that is not routed through the tier still reads
// the right bytes, through the page cache). Each registered expert tensor gets a "shadow": an anonymous, lazily
// populated region with the same byte layout. An expert that a reader needs is read from the model file with O_DIRECT
// (aligned preads on a small thread pool, no page cache) straight into its place in the shadow; the shadow pages of
// the least recently used experts are released (MADV_DONTNEED) to keep the RAM tier under its budget. The bytes are
// the file's, so every result is bit-identical to the mmap path.
//
// Environment (read once):
//   LLAMA_MOE_NVME_TIER=1          on (default off: nothing here is created or called)
//   LLAMA_MOE_NVME_RAM_MIB=N       RAM tier budget in MiB (default 4096)
//   LLAMA_MOE_NVME_THREADS=N       reader threads (default 8, 1..64)
//   LLAMA_MOE_NVME_DROP=0          keep the experts' page cache at load (default: posix_fadvise DONTNEED them)
//   LLAMA_MOE_NVME_STATS=1         print the tier counters at every MoE cache stats print and at exit

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

struct llama_moe_nvme_stats {
    uint64_t hits        = 0; // experts found in the RAM tier
    uint64_t misses      = 0; // experts read into the RAM tier
    uint64_t evictions   = 0;
    uint64_t overflows   = 0; // misses admitted above the budget (every resident expert was protected)
    uint64_t bytes_read  = 0; // bytes read from the file (aligned)
    uint64_t direct_reads = 0; // read() bytes served from the file without filling the tier
    uint64_t fallbacks   = 0; // reads that failed and were served from the mapping instead
    double   read_s      = 0; // wall time spent waiting for file reads
    uint64_t direct_files = 0; // registered tensors read with O_DIRECT (the others: buffered + fadvise DONTNEED)
    size_t   resident    = 0; // bytes charged to the RAM tier now
    size_t   budget      = 0;
};

class llama_moe_nvme_tier {
public:
    // budget: RAM tier bytes; n_threads: reader threads
    llama_moe_nvme_tier(size_t budget, int n_threads);
    ~llama_moe_nvme_tier();

    // register one expert tensor: `host` is the address its readers use (the mapped file bytes of the tensor),
    // `fd` a descriptor of the file (reopened with O_DIRECT), `file_offset` the offset of host[0] in the file and
    // expert e is host[e*stride, (e+1)*stride). false: not registered (the caller keeps the plain path for it).
    bool add(const void * host, int fd, size_t file_offset, size_t stride, int64_t n_expert, const void * owner);
    // drop every tensor registered with this owner
    void remove(const void * owner);
    bool empty() const;

    // the RAM tier copy of host[0, len): every expert it overlaps is resident (read from the file when it was not)
    // and stays so until two more epoch() calls; nullptr when host is not inside a registered tensor
    const void * ensure(const void * host, size_t len);
    // ensure() of several ranges, with their missing experts read in parallel
    void ensure_many(const std::vector<std::pair<const void *, size_t>> & ranges);
    // copy host[0, len) into dst: resident experts from RAM, the others straight from the file (the tier is not
    // filled, so a streamed prompt does not evict the decode working set). false: not registered, nothing copied.
    bool read(const void * host, size_t len, void * dst);

    // a new protection window: experts ensured before the previous epoch() may be evicted
    void epoch();

    llama_moe_nvme_stats stats() const;
    void print_stats(const char * tag) const;

    // test hooks: whether expert e of the tensor registered at host is resident; the tier's shadow of host
    bool resident(const void * host, int64_t e) const;
    const void * shadow(const void * host) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// LLAMA_MOE_NVME_TIER=1 (read once)
bool llama_moe_nvme_enabled();
// the process tier, nullptr when off or nothing is registered
llama_moe_nvme_tier * llama_moe_nvme_global();
// the process tier, created on first use (only when enabled)
llama_moe_nvme_tier * llama_moe_nvme_global_create();
// drop the owner's tensors; the tier is destroyed when it has none left
void llama_moe_nvme_global_release(const void * owner);

// fork (NVMETIER): register the model's mapped host expert tensors with the process tier (called after the tensors
// are loaded, only when enabled). Lives in llama-moe-nvme-setup.cpp so the tier itself builds without the model.
struct llama_model;
struct llama_model_loader;
void llama_moe_nvme_setup(const llama_model & model, llama_model_loader & ml);
// fork (NVMETIER): expert tensor names the tier may take (their file ranges skip the load-time page prefetch)
bool llama_moe_nvme_is_expert_name(const char * name);
