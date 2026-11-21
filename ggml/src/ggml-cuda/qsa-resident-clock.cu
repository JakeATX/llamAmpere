#include "qsa-resident-clock.cuh"
#include "ggml-backend-impl.h"
#include "../ggml-sjkvarn.h"
#include "../ggml-lsa-recall.h"
#include "sjkvarn-lowbits.cuh"
#include "sjkvarn-seal.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// ctl words per map: 0 epoch, 1 hand, 2 queued misses, 3 picked, 4 assigned, 5 prefetched, 6 prefetch resolves,
// 7 not admitted, 8..15 u64 own counters (fetched, picked, resolves, over), 16 epoch of the last own resolve,
// 17 J3 marks already resident, 18 J3 marks with no slot (cap or every slot protected), 19..23 J3 cascade counters,
// 24 records the last own resolve picked (J3: the previous pick is protected only while it leaves slots free)
constexpr int QSA_CLOCK_CTL_WORDS = 25;

struct qsa_clock_type {
    ggml_backend_buffer_type buft;
    size_t stride;
    uint32_t slots;
    uint32_t group;
    uint32_t heads;
    uint32_t cells; // GGML_QSA_PAGE_CELLS or GGML_LSA_STRIP_CELLS
    int device;
    std::string name;
};

// bytes of one page of one KV head in a slot, and of one record's card meta per head (record pages need no meta)
static __host__ __device__ inline size_t qsa_page_head_bytes(const ggml_cuda_qsa_clock_map & m) {
    return m.cells == GGML_LSA_STRIP_CELLS ? GGML_LSA_STRIP_HEAD_BYTES : m.cells == GGML_LSA_PAGE_CELLS ? m.stride/m.heads : GGML_QSA_PAGE_HEAD_BYTES;
}
static __host__ __device__ inline size_t qsa_meta_head_bytes(const ggml_cuda_qsa_clock_map & m) {
    return m.cells == GGML_LSA_PAGE_CELLS ? 0 : GGML_QSA_META_HEAD_BYTES;
}

struct qsa_clock_tensor {
    ggml_cuda_qsa_clock_map map{};
    char * host = nullptr; // host address of the root tensor; map.host is a device mapping
    ggml_cuda_qsa_clock_map * device_map = nullptr;
    int device = -1;
    uint64_t ahead_seq = 0; // GGML_CUDA_LSA_STAGE_AHEAD: prepare number of the layer that queued this layer's ahead copy
    uint64_t prep_seq = 0;  // GGML_CUDA_LSA_STAGE_AHEAD: this layer's last prepare number
    const void * owner = nullptr; // GGML_CUDA_LSA_STAGE_AHEAD: the buffer (an ahead copy never leaves its buffer)
    // HOSTKV08, GGML_CUDA_LSA_STAGE_AHEAD_DMA (copy engine instead of a kernel): pinned mapped host block of
    // 2*(stage_n + 1) ints. mirror = [count, list...] of this layer's last offered pick, written by the device after each
    // decode resolve (mirror_done); applied = [count, records...] the host actually copied to stage indices 0.., read by
    // the target's stage reset (applied_used), so what the reset applies is always exactly what was copied.
    int32_t * ahead_host = nullptr;
    int32_t * ahead_host_dev = nullptr; // device mapping of ahead_host
    cudaEvent_t mirror_done = nullptr;
    cudaEvent_t applied_used = nullptr;
    bool mirror_valid = false;          // a mirror write was queued since the last reset
    // GGML_CUDA_LSA_PREFETCH (opt-in, record pages): the previous LSA layer's pick fills this layer's slots on a side
    // stream; pf_done joins it back before this layer's seal refresh or resolve touches the map.
    cudaEvent_t pf_done = nullptr;
    bool pf_pending = false;
    uint32_t * recall_history = nullptr; // diagnostic only: previous visible end, resolve, then strip bits per page
    size_t recall_pages = 0;
    // LLAMA_LSA_J3 (GGML_OP_LSA_PREFETCH): per source, the prediction and descriptor copied on the main stream (graph
    // memory is reused once the op is done) and read by the side stream's mark; issued = copied, not yet joined
    float * j3_units[GGML_LSA_PREFETCH_SOURCES] = {};
    size_t j3_units_cap[GGML_LSA_PREFETCH_SOURCES] = {};
    int32_t * j3_desc[GGML_LSA_PREFETCH_SOURCES] = {};
    size_t j3_desc_cap[GGML_LSA_PREFETCH_SOURCES] = {};
    bool j3_issued[GGML_LSA_PREFETCH_SOURCES] = {};
};

// GGML_CUDA_LSA_PREFETCH=<records> (0/unset = off): most records one layer's pick may prefetch into the next LSA
// layer's slots per step. Output never depends on residency (a record without a slot is read from the master).
static int lsa_prefetch_cap() {
    static const int cap = [] {
        const char * v = getenv("GGML_CUDA_LSA_PREFETCH");
        return v ? std::max(0, atoi(v)) : 0;
    }();
    return cap;
}
// GGML_CUDA_LSA_RESIDENT_FETCH=N (opt-in): copy at most N missed record pages into slots per resolve; the other misses
// are read straight from the pinned master (same bytes over the link, no slot write and no extra launch work). When the
// slots are smaller than one step's read set, CLOCK otherwise evicts pages the next step reads again (af1: 40% of the
// records on the card = 43.7 vs 14.4 ms/token). 0 or unset = up to the slot count (unchanged).
static uint32_t lsa_fetch_cap(uint32_t slots) {
    static const int cap = [] {
        const char * v = getenv("GGML_CUDA_LSA_RESIDENT_FETCH");
        return v ? std::max(0, atoi(v)) : 0;
    }();
    return cap > 0 ? std::min<uint32_t>(slots, uint32_t(cap)) : slots;
}
// GGML_CUDA_LSA_RESIDENT_ADMIT=1 (opt-in): a missed page gets a slot only if this layer's previous resolve picked it too;
// one-step picks are read from the master. Output never depends on residency, so either knob changes speed only.
static bool lsa_admit_filter() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_RESIDENT_ADMIT");
        return v && atoi(v) > 0;
    }();
    return on;
}
// GGML_CUDA_LSA_RESIDENT_STAGE=<records> (opt-in, record pages): size of the per-buffer on-card staging area for picked
// records that get no CLOCK slot (past the slots, not admitted, or every slot protected by the pick). They are copied in
// by the streaming page copy each resolve instead of being read by the low-bit attention kernel over the bus (kf3: the
// 40%-slot case reads ~18 MiB per layer per step past the slots at ~3.4 GB/s; the page copy moves ~12 GB/s). Same bytes
// either way, so output never depends on it. 0 or unset = no stage, unless STAGE_ALL is on.
// RESIDENT=auto may size it instead (ggml_backend_cuda_lsa_resident_set_stage, called before the KV buffers are made);
// the env value wins, except STAGE_ALL=1 forces room for every record in one layer.
static uint32_t lsa_stage_auto_n = 0;
void ggml_backend_cuda_lsa_resident_set_stage(uint32_t records) {
    lsa_stage_auto_n = std::max(lsa_stage_auto_n, records);
}
static uint32_t lsa_stage_cap() {
    static const int64_t env = [] {
        const char * v = getenv("GGML_CUDA_LSA_RESIDENT_STAGE");
        return v ? int64_t(std::max(0, atoi(v))) : int64_t(-1);
    }();
    return env >= 0 ? uint32_t(env) : lsa_stage_auto_n;
}
static bool lsa_stage_all() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_RESIDENT_STAGE_ALL");
        return v && atoi(v) > 0;
    }();
    return on;
}

static bool lsa_copy_wide() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_RESIDENT_COPY_WIDE");
        return v && atoi(v) > 0;
    }();
    return on;
}
// GGML_CUDA_LSA_J3_COPY_BLOCKS: blocks per KV head of the J3 side-stream record copy (default 16, as the old prefetch)
static uint32_t lsa_j3_copy_blocks() {
    static const uint32_t n = [] {
        const char * v = getenv("GGML_CUDA_LSA_J3_COPY_BLOCKS");
        return v && atoi(v) > 0 ? uint32_t(std::min(atoi(v), 1024)) : 16u;
    }();
    return n;
}
// GGML_CUDA_LSA_J3_GRAPHS=1 (opt-in): LLAMA_LSA_J3 under CUDA graphs. All host setup runs from the graph check before
// capture (ggml_cuda_lsa_prefetch_reserve); captured kernels hold the map and scratch pointers, so every allocation
// that changes one bumps lsa_j3_gen and the graphs re-capture (ggml_cuda_graph_update_required).
static std::atomic<uint64_t> lsa_j3_gen{0};
bool ggml_cuda_lsa_j3_graphs() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_J3_GRAPHS");
        return v && atoi(v) > 0;
    }();
    return on;
}
uint64_t ggml_cuda_lsa_j3_generation() {
    return lsa_j3_gen.load(std::memory_order_relaxed);
}
static bool lsa_stage_strips() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_RESIDENT_STAGE_STRIPS");
        return v && atoi(v) > 0;
    }();
#if !GGML_CUDA_LSA_STAGE_STRIPS_KERNEL
    static const bool warned = [] {
        if (on) {
            GGML_LOG_WARN("%s: GGML_CUDA_LSA_RESIDENT_STAGE_STRIPS ignored: built without GGML_CUDA_LSA_STAGE_STRIPS_KERNEL (whole-record stage)\n", __func__);
        }
        return true;
    }();
    GGML_UNUSED(warned);
    return false;
#else
    return lsa_stage_all() && on;
#endif
}
static bool lsa_overflow_log() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_RESIDENT_OVERFLOW_LOG");
        return v && atoi(v) > 0;
    }();
    return lsa_stage_all() && on;
}
// stage_ctl words: 0 entries this resolve, 1 first entry this resolve copies (the ahead entries sit below it), u64 at 2
// copied total, u64 at 4 stage full, 7 ahead_list entries for the next ahead copy, u64 at 8 ahead entries picked again,
// u64 at 10 ahead entries offered
constexpr int LSA_STAGE_CTL_WORDS = 12;
// GGML_CUDA_LSA_STAGE_AHEAD (opt-in =1): see ggml_cuda_qsa_clock_map::ahead_list. Needs the whole-record stage
// (GGML_CUDA_LSA_RESIDENT_STAGE) and stays off with STAGE_ALL, stage strips and GGML_CUDA_LSA_PREFETCH; a J3 prefetch
// (LLAMA_LSA_J3) turns it off for the process. Costs a second stage of the same size per buffer.
static bool lsa_stage_ahead_env() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_STAGE_AHEAD"); // opt-in (=1) until the HOSTKV08 gate passes
        return v != nullptr && atoi(v) > 0;
    }();
    return on;
}
// GGML_CUDA_LSA_STAGE_AHEAD_BLOCKS: blocks per KV head of the ahead copy (default 8: few SMs, the link is the limit)
static uint32_t lsa_stage_ahead_blocks() {
    static const uint32_t n = [] {
        const char * v = getenv("GGML_CUDA_LSA_STAGE_AHEAD_BLOCKS");
        return v && atoi(v) > 0 ? uint32_t(std::min(atoi(v), 256)) : 8u;
    }();
    return n;
}
// HOSTKV08: GGML_CUDA_LSA_STAGE_AHEAD_DMA (default 1 = the ahead copy runs on a copy engine with cudaMemcpyAsync from
// the pinned master, no SMs; =0 = DEEP8's copy kernel on the side stream)
static bool lsa_stage_ahead_dma() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_STAGE_AHEAD_DMA");
        return v == nullptr || *v == 0 || atoi(v) > 0;
    }();
    return on;
}
static int lsa_prefetch_cap();
static bool lsa_stage_ahead_on(const ggml_cuda_qsa_clock_map & m) {
    return lsa_stage_ahead_env() && m.cells == GGML_LSA_PAGE_CELLS && m.stage_n > 0 && !m.stage_all && !m.stage_stride &&
        !m.strip_master && lsa_prefetch_cap() == 0;
}
static size_t lsa_stage_stride(size_t stride, uint32_t heads) {
    GGML_ASSERT(heads > 0 && stride <= SIZE_MAX - size_t(heads)*4096);
    return lsa_stage_strips() ? stride + size_t(heads)*4096 : stride;
}
static std::mutex lsa_pf_mutex;
static std::vector<qsa_clock_tensor *> lsa_pf_entries; // record-page entries, for the next-layer lookup
static cudaStream_t lsa_pf_stream[GGML_CUDA_MAX_DEVICES] = {};
static cudaEvent_t lsa_pf_fork[GGML_CUDA_MAX_DEVICES] = {};

// GGML_CUDA_LSA_STAGE_AHEAD: one low-priority side stream per device; at most one ahead copy in flight, joined by the
// next record-page prepare on the device whatever layer it is. A prepare counter tells the target layer whether the copy
// was queued by the prepare right before its own (else the copy is dropped and the resolve copies as before).
static cudaStream_t lsa_ahead_stream[GGML_CUDA_MAX_DEVICES] = {};
static cudaEvent_t lsa_ahead_fork[GGML_CUDA_MAX_DEVICES] = {};
static cudaEvent_t lsa_ahead_done[GGML_CUDA_MAX_DEVICES] = {};
static bool lsa_ahead_pending[GGML_CUDA_MAX_DEVICES] = {};
static uint64_t lsa_ahead_seq[GGML_CUDA_MAX_DEVICES] = {};
static std::vector<qsa_clock_tensor *> lsa_ahead_entries; // under lsa_pf_mutex
static std::atomic<bool> lsa_ahead_blocked{false};         // set by the first J3 prefetch
static bool lsa_ahead_logged = false;
static uint64_t lsa_ahead_first_issued = 0; // HOSTKV08: ahead copies queued at graph begin (log)
static uint64_t lsa_ahead_issued = 0;       // HOSTKV08: ahead copies queued (log)
static uint64_t lsa_ahead_skipped = 0;      // HOSTKV08 DMA: mirror or applied list not ready, no copy (log)
static uint64_t lsa_ahead_dma_copies = 0;   // HOSTKV08 DMA: cudaMemcpyAsync calls (log)

static void lsa_ahead_join(cudaStream_t stream, int device) {
    if (device >= 0 && lsa_ahead_pending[device]) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, lsa_ahead_done[device], 0));
        lsa_ahead_pending[device] = false;
    }
}

static void lsa_pf_join(cudaStream_t stream, qsa_clock_tensor * e) {
    if (e && e->pf_pending) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, e->pf_done, 0));
        e->pf_pending = false;
        for (bool & issued : e->j3_issued) issued = false; // the side stream has read every J3 scratch copy
    }
}

struct qsa_clock_buffer {
    void * host = nullptr;
    int device;
    std::vector<std::unique_ptr<qsa_clock_tensor>> tensors;
    // fork VRAMPOOL: one device block shared by every layer's maps; cudaMalloc rounds each call
    // to 2 MiB, so ten calls per layer wasted ~3.5 MiB per layer.
    char * pool = nullptr;
    size_t pool_size = 0, pool_used = 0;
    bool pool_tried = false;
    std::vector<void *> blocks;
    char * stage = nullptr; // GGML_CUDA_LSA_RESIDENT_STAGE: shared by this buffer's record-page layers
    char * stage2 = nullptr; // GGML_CUDA_LSA_STAGE_AHEAD: the second stage (odd record-page layers of the buffer)
    uint32_t page_layers = 0; // record-page layers given a stage so far (parity)
    size_t stage_bytes = 0;
};

// Carve one layer's device maps from base (nullptr: size only). Every piece is 256-byte aligned.
static size_t qsa_clock_layout(ggml_cuda_qsa_clock_map & m, ggml_cuda_qsa_clock_map ** device_map, char * base) {
    size_t off = 0;
    auto take = [&](size_t bytes) -> char * {
        char * p = base ? base + off : nullptr;
        off += (bytes + 255) & ~size_t(255);
        return p;
    };
    char * card = take(size_t(m.slots)*m.heads*qsa_page_head_bytes(m));
    char * meta = take(size_t(m.records)*m.heads*qsa_meta_head_bytes(m));
    char * table = take(size_t(m.records)*(m.group/m.cells)*sizeof(int32_t));
    char * tags = take(m.slots*sizeof(int32_t));
    char * stamp = take(m.slots*sizeof(uint32_t));
    char * ref = take(m.slots*sizeof(int32_t));
    char * miss_page = take(m.slots*sizeof(int32_t));
    char * miss_slot = take(m.slots*sizeof(int32_t));
    char * ctl = take(QSA_CLOCK_CTL_WORDS*sizeof(uint32_t));
    char * seen = take(m.cells == GGML_LSA_PAGE_CELLS && lsa_admit_filter() ? size_t(m.records)*sizeof(uint32_t) : 0);
    char * stage_page = take(size_t(m.stage_n)*sizeof(int32_t));
    char * stage_ctl = take(m.stage_n ? LSA_STAGE_CTL_WORDS*sizeof(uint32_t) : 0);
    char * ahead_list = take(lsa_stage_ahead_on(m) ? size_t(m.stage_n)*sizeof(int32_t) : 0);
    const bool overflow_log = m.cells == GGML_LSA_PAGE_CELLS && m.stage_n && lsa_overflow_log();
    const bool strip_mask = m.stage_stride || overflow_log;
    char * stage_mask = take(strip_mask ? size_t(m.records)*sizeof(uint32_t) : 0);
    char * stage_valid = take(m.stage_stride ? size_t(m.stage_n)*m.heads*sizeof(uint32_t) : 0);
    char * stage_stats = take(m.stage_stride ? 3*sizeof(unsigned long long) : 0);
    char * overflow_hist = take(overflow_log ? 9*sizeof(unsigned long long) : 0);
    char * dmap = take(sizeof(ggml_cuda_qsa_clock_map));
    if (base) {
        m.card = card; m.meta = meta; m.table = (int32_t *) table; m.tags = (int32_t *) tags;
        m.stamp = (uint32_t *) stamp; m.ref = (int32_t *) ref; m.miss_page = (int32_t *) miss_page;
        m.miss_slot = (int32_t *) miss_slot; m.ctl = (uint32_t *) ctl;
        m.seen = m.cells == GGML_LSA_PAGE_CELLS && lsa_admit_filter() ? (uint32_t *) seen : nullptr;
        m.stage_page = m.stage_n ? (int32_t *) stage_page : nullptr;
        m.stage_ctl = m.stage_n ? (uint32_t *) stage_ctl : nullptr;
        m.ahead_list = lsa_stage_ahead_on(m) ? (int32_t *) ahead_list : nullptr;
        m.stage_mask = strip_mask ? (uint32_t *) stage_mask : nullptr;
        m.stage_valid = m.stage_stride ? (uint32_t *) stage_valid : nullptr;
        m.stage_stats = m.stage_stride ? (unsigned long long *) stage_stats : nullptr;
        m.overflow_hist = overflow_log ? (unsigned long long *) overflow_hist : nullptr;
        *device_map = (ggml_cuda_qsa_clock_map *) dmap;
    }
    return off;
}

static const char * qsa_clock_name(ggml_backend_buffer_type_t buft) {
    return static_cast<qsa_clock_type *>(buft->context)->name.c_str();
}

bool ggml_cuda_qsa_clock_is_buft(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == qsa_clock_name;
}

static qsa_clock_tensor * qsa_clock_entry(const ggml_tensor * tensor) {
    while (tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor->buffer && ggml_cuda_qsa_clock_is_buft(tensor->buffer->buft) ?
        static_cast<qsa_clock_tensor *>(tensor->extra) : nullptr;
}

static void qsa_clock_sync(ggml_backend_buffer_t buffer) {
    ggml_cuda_set_device(static_cast<qsa_clock_buffer *>(buffer->context)->device);
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void qsa_clock_reset(ggml_backend_buffer_t buffer) {
    qsa_clock_sync(buffer);
    {
        const int device = static_cast<qsa_clock_buffer *>(buffer->context)->device;
        lsa_ahead_pending[device] = false; // the device is idle: any ahead copy has landed
        ++lsa_ahead_seq[device];           // and none is applied after the reset
        for (const auto & entry : static_cast<qsa_clock_buffer *>(buffer->context)->tensors) {
            entry->mirror_valid = false;   // HOSTKV08 DMA: no list from before the reset is copied
        }
    }
    for (const auto & entry : static_cast<qsa_clock_buffer *>(buffer->context)->tensors) {
        auto & m = entry->map;
        if (entry->recall_history) {
            CUDA_CHECK(cudaMemset(entry->recall_history, 0, (entry->recall_pages + 2)*sizeof(uint32_t)));
        }
        CUDA_CHECK(cudaMemset(m.tags, 0xff, m.slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(m.table, 0xff, size_t(m.records)*(m.group/m.cells)*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(m.stamp, 0, m.slots*sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(m.ref, 0, m.slots*sizeof(int32_t)));
        CUDA_CHECK(cudaMemset(m.ctl, 0, QSA_CLOCK_CTL_WORDS*sizeof(uint32_t)));
        if (m.seen) {
            CUDA_CHECK(cudaMemset(m.seen, 0, size_t(m.records)*sizeof(uint32_t)));
        }
        if (m.prev) {
            CUDA_CHECK(cudaMemset(m.prev, 0, size_t(m.records)*sizeof(uint32_t)));
        }
        if (m.j3_src) {
            CUDA_CHECK(cudaMemset(m.j3_src, 0, size_t(m.records)*sizeof(uint32_t)));
        }
        if (m.stage_ctl) {
            CUDA_CHECK(cudaMemset(m.stage_ctl, 0, LSA_STAGE_CTL_WORDS*sizeof(uint32_t)));
        }
        if (m.stage_mask) {
            CUDA_CHECK(cudaMemset(m.stage_mask, 0, size_t(m.records)*sizeof(uint32_t)));
        }
        if (m.overflow_hist) {
            CUDA_CHECK(cudaMemset(m.overflow_hist, 0, 9*sizeof(unsigned long long)));
        }
        if (m.stage_stride) {
            CUDA_CHECK(cudaMemset(m.stage_valid, 0, size_t(m.stage_n)*m.heads*sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(m.stage_stats, 0, 3*sizeof(unsigned long long)));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void qsa_clock_free(ggml_backend_buffer_t buffer) {
    auto * b = static_cast<qsa_clock_buffer *>(buffer->context);
    qsa_clock_sync(buffer);
    if (lsa_ahead_issued) {
        GGML_LOG_WARN("LSA STAGE AHEAD: issued=%llu (graph begin %llu), mode %s, dma copies=%llu, skipped (list not ready)=%llu\n",
            (unsigned long long) lsa_ahead_issued, (unsigned long long) lsa_ahead_first_issued,
            lsa_stage_ahead_dma() ? "dma" : "kernel", (unsigned long long) lsa_ahead_dma_copies,
            (unsigned long long) lsa_ahead_skipped);
    }
    {
        std::lock_guard<std::mutex> lock(lsa_pf_mutex);
        for (const auto & e : b->tensors) {
            lsa_pf_entries.erase(std::remove(lsa_pf_entries.begin(), lsa_pf_entries.end(), e.get()), lsa_pf_entries.end());
            lsa_ahead_entries.erase(std::remove(lsa_ahead_entries.begin(), lsa_ahead_entries.end(), e.get()), lsa_ahead_entries.end());
            if (e->ahead_host) CUDA_CHECK(cudaFreeHost(e->ahead_host));
            if (e->mirror_done) CUDA_CHECK(cudaEventDestroy(e->mirror_done));
            if (e->applied_used) CUDA_CHECK(cudaEventDestroy(e->applied_used));
            if (e->pf_done) CUDA_CHECK(cudaEventDestroy(e->pf_done));
            if (e->recall_history) CUDA_CHECK(cudaFree(e->recall_history));
            if (e->map.prev) CUDA_CHECK(cudaFree(e->map.prev));
            if (e->map.j3_src) CUDA_CHECK(cudaFree(e->map.j3_src));
            for (int i = 0; i < GGML_LSA_PREFETCH_SOURCES; ++i) {
                if (e->j3_units[i]) CUDA_CHECK(cudaFree(e->j3_units[i]));
                if (e->j3_desc[i]) CUDA_CHECK(cudaFree(e->j3_desc[i]));
            }
        }
    }
    for (const auto & e : b->tensors) {
        uint64_t counters[8]{};
        CUDA_CHECK(cudaMemcpy(counters, e->map.ctl, sizeof(counters), cudaMemcpyDeviceToHost));
        if (e->map.cells == GGML_LSA_STRIP_CELLS) {
            // u64 at ctl 8 fetched strips, 10 strips read (picked), 12 resolves, 14 picked strips past the slots
            const double strip_mib = double(e->map.heads)*GGML_LSA_STRIP_HEAD_BYTES/(1024.0*1024.0);
            const double calls = std::max<double>(1.0, double(counters[6]));
            GGML_LOG_WARN("LSA RESIDENT: layer %d slots=%u strips, resolves=%llu picked=%llu fetched=%llu over_slots=%llu, "
                "per resolve: %.2f MiB fetched + %.2f MiB read from host past the slots\n", e->map.layer, e->map.slots,
                (unsigned long long) counters[6], (unsigned long long) counters[5], (unsigned long long) counters[4],
                (unsigned long long) counters[7], counters[4]*strip_mib/calls, counters[7]*strip_mib/calls);
            continue;
        }
        if (e->map.cells == GGML_LSA_PAGE_CELLS) {
            // same counters, in whole records (128 cells, every head)
            const double rec_mib = double(e->map.stride)/(1024.0*1024.0);
            const double calls = std::max<double>(1.0, double(counters[6]));
            GGML_LOG_WARN("LSA RESIDENT: layer %d slots=%u records, resolves=%llu picked=%llu fetched=%llu over_slots=%llu, "
                "per resolve: %.2f MiB fetched + %.2f MiB read from host past the slots\n", e->map.layer, e->map.slots,
                (unsigned long long) counters[6], (unsigned long long) counters[5], (unsigned long long) counters[4],
                (unsigned long long) counters[7], counters[4]*rec_mib/calls, counters[7]*rec_mib/calls);
            const uint32_t * pf = reinterpret_cast<const uint32_t *>(counters); // u32 ctl 5 prefetched, 6 prefetch resolves
            if (pf[6]) {
                GGML_LOG_WARN("LSA RESIDENT: layer %d prefetch resolves=%u prefetched=%u records (%.2f MiB per prefetch)\n",
                    e->map.layer, pf[6], pf[5], pf[5]*rec_mib/pf[6]);
            }
            if (e->map.seen) { // admission control on: misses refused a slot (u32 ctl 7)
                GGML_LOG_WARN("LSA RESIDENT: layer %d admission not_admitted=%u records (%.2f per resolve)\n",
                    e->map.layer, pf[7], pf[7]/calls);
            }
            if (e->map.stage_ctl) {
                uint64_t st[LSA_STAGE_CTL_WORDS/2]{}; // u64 at stage_ctl 2 staged, 4 stage full (totals lag the last resolve)
                CUDA_CHECK(cudaMemcpy(st, e->map.stage_ctl, sizeof(st), cudaMemcpyDeviceToHost));
                if (e->map.ahead_list) {
                    GGML_LOG_WARN("LSA STAGE AHEAD: layer %d offered=%llu reused=%llu, per resolve: %.2f MiB copied ahead, "
                        "%.2f MiB reused (not copied by the resolve)\n", e->map.layer, (unsigned long long) st[5],
                        (unsigned long long) st[4], st[5]*rec_mib/calls, st[4]*rec_mib/calls);
                }
                GGML_LOG_WARN("LSA RESIDENT: layer %d stage=%u records, staged=%llu stage_full=%llu, per resolve: %.2f MiB "
                    "%s, %.2f MiB still read from host\n", e->map.layer, e->map.stage_n,
                    (unsigned long long) st[1], (unsigned long long) st[2], st[1]*rec_mib/calls,
                    e->map.stage_stride ? "whole-record equivalent (actual bytes in LSA STRIP STAGE)" : "copied to the stage", st[2]*rec_mib/calls);
            }
            if (e->map.stage_stats) {
                unsigned long long st[3]{};
                CUDA_CHECK(cudaMemcpy(st, e->map.stage_stats, sizeof(st), cudaMemcpyDeviceToHost));
                GGML_LOG_WARN("LSA STRIP STAGE: layer %d bytes=%llu strips=%llu records=%llu heads=%u\n",
                    e->map.layer, st[0], st[1], st[2], e->map.heads);
            }
            if (e->map.overflow_hist) {
                unsigned long long hist[9]{}, records = 0, strips = 0;
                CUDA_CHECK(cudaMemcpy(hist, e->map.overflow_hist, sizeof(hist), cudaMemcpyDeviceToHost));
                for (int s = 0; s <= 8; ++s) {
                    records += hist[s];
                    strips += s*hist[s];
                }
                const double mean = records ? double(strips)/records : 0.0;
                GGML_LOG_WARN("LSA OVERFLOW STRIPS: layer=%d records=%llu strips=%llu mean=%.9f heads=%u "
                    "whole_record_bytes=%zu histogram=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu "
                    "scope=staged_only decision=%s\n", e->map.layer, records, strips, mean, e->map.heads,
                    e->map.stride, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8],
                    !records ? "NO_DATA" : hist[0] ? "INVALID_EMPTY_RECORD" : mean > 5.0 ? "STOP_STRIP_MASTER" : "WITHIN_5_STRIPS");
            }
            continue;
        }
        GGML_LOG_INFO("QSA CLOCK: slots=%u calls=%llu page_misses=%llu selected_cells=%llu last_misses=%u overflow=%u\n",
            e->map.slots, (unsigned long long) counters[6], (unsigned long long) counters[4],
            (unsigned long long) counters[5], uint32_t(counters[1]), uint32_t(counters[2]));
    }
    for (void * block : b->blocks) CUDA_CHECK(cudaFree(block));
    if (b->pool) CUDA_CHECK(cudaFree(b->pool));
    if (b->stage) CUDA_CHECK(cudaFree(b->stage));
    if (b->stage2) CUDA_CHECK(cudaFree(b->stage2));
    if (b->host) CUDA_CHECK(cudaFreeHost(b->host));
    delete b;
}

static void * qsa_clock_base(ggml_backend_buffer_t buffer) {
    return static_cast<qsa_clock_buffer *>(buffer->context)->host;
}

static ggml_status qsa_clock_init(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (tensor->view_src) {
        return GGML_STATUS_SUCCESS;
    }
    auto * t = static_cast<qsa_clock_type *>(buffer->buft->context);
    if (tensor->type != GGML_TYPE_I8 || ggml_nbytes(tensor) % t->stride != 0) {
        return GGML_STATUS_FAILED;
    }
    auto entry = std::make_unique<qsa_clock_tensor>();
    auto & m = entry->map;
    void * mapped = nullptr;
    auto * b = static_cast<qsa_clock_buffer *>(buffer->context);
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped, b->host, 0));
    entry->host = static_cast<char *>(tensor->data);
    m.host = static_cast<const char *>(mapped) + (static_cast<const char *>(tensor->data) - static_cast<const char *>(b->host));
    m.stride = t->stride;
    GGML_ASSERT(ggml_nbytes(tensor)/t->stride <= INT32_MAX/(t->group/4));
    m.records = ggml_nbytes(tensor)/t->stride;
    m.group = t->group;
    m.heads = t->heads;
    m.cells = t->cells;
    m.slots = std::min<size_t>(t->slots, m.records)*(m.group/m.cells);
    if (m.cells == GGML_LSA_STRIP_CELLS || m.cells == GGML_LSA_PAGE_CELLS) {
        int il = -1;
        const char * l = strrchr(tensor->name, 'l');
        if (l && sscanf(l, "l%d", &il) == 1) m.layer = il;
        const char * every = getenv("GGML_CUDA_LSA_RESIDENT_LOG");
        m.log_every = every ? uint32_t(std::max(0, atoi(every))) : 0;
        const char * prot = getenv("GGML_CUDA_LSA_RESIDENT_PROTECT");
        m.protect = prot ? uint32_t(std::max(1, atoi(prot))) : 1;
    }
    if (!m.slots) {
        return GGML_STATUS_FAILED;
    }
    ggml_cuda_set_device(t->device);
    const uint32_t stage_n = lsa_stage_all() ? m.records : lsa_stage_cap();
    if (m.cells == GGML_LSA_PAGE_CELLS && stage_n > 0) {
        // one stage per buffer (made here, never during CUDA graph capture); layers in a buffer share the record size
        const size_t stride = lsa_stage_stride(m.stride, m.heads);
        GGML_ASSERT(stage_n <= INT32_MAX - 2 && stride <= SIZE_MAX/stage_n);
        const size_t sb = size_t(stage_n)*stride;
        if (!b->stage) {
            if (cudaMalloc((void **) &b->stage, sb) != cudaSuccess) {
                (void) cudaGetLastError();
                b->stage = nullptr;
                if (lsa_stage_all()) {
                    GGML_LOG_ERROR("LSA RESIDENT: full stage allocation failed (%.1f MiB); refusing host decode reads\n", sb/1048576.0);
                    return GGML_STATUS_ALLOC_FAILED;
                }
                GGML_LOG_WARN("LSA RESIDENT: could not allocate the %u-record stage (%.1f MiB); records past the slots "
                    "are read from the master\n", stage_n, sb/1048576.0);
            } else {
                b->stage_bytes = sb;
                GGML_LOG_WARN("LSA RESIDENT: stage of %u records (%.1f MiB) for records past the slots\n",
                    stage_n, sb/1048576.0);
            }
        }
        if (b->stage && b->stage_bytes >= sb) {
            m.stage = b->stage;
            m.stage_n = stage_n;
            m.stage_all = lsa_stage_all();
            m.stage_stride = lsa_stage_strips() ? stride : 0;
            if (lsa_stage_ahead_on(m) && !lsa_ahead_blocked) {
                // GGML_CUDA_LSA_STAGE_AHEAD: odd record-page layers of the buffer take the second stage, so a layer's
                // ahead copy never writes the stage the layer before it is reading
                if (!b->stage2) {
                    if (cudaMalloc((void **) &b->stage2, b->stage_bytes) != cudaSuccess) {
                        (void) cudaGetLastError();
                        b->stage2 = nullptr;
                        lsa_ahead_blocked = true;
                        GGML_LOG_WARN("LSA STAGE AHEAD: could not allocate the second stage (%.1f MiB); off\n",
                            b->stage_bytes/1048576.0);
                    } else {
                        GGML_LOG_WARN("LSA STAGE AHEAD: on, second stage of %u records (%.1f MiB); "
                            "GGML_CUDA_LSA_STAGE_AHEAD=0 turns it off\n", stage_n, b->stage_bytes/1048576.0);
                    }
                }
                if (b->stage2 && (b->page_layers++ & 1u)) {
                    m.stage = b->stage2;
                }
            }
        } else if (lsa_stage_all()) {
            GGML_LOG_ERROR("LSA RESIDENT: shared full stage is too small for this layer\n");
            return GGML_STATUS_ALLOC_FAILED;
        }
    }
    const size_t need = qsa_clock_layout(m, nullptr, nullptr);
    if (!b->pool_tried) {
        // Layers in one buffer share a shape, so size the pool for all of them up front.
        b->pool_tried = true;
        const size_t layers = buffer->size/ggml_nbytes(tensor);
        if (layers > 1 && cudaMalloc((void **) &b->pool, layers*need) == cudaSuccess) {
            b->pool_size = layers*need;
        } else {
            (void) cudaGetLastError();
            b->pool = nullptr;
        }
    }
    char * base = nullptr;
    if (b->pool && b->pool_size - b->pool_used >= need) {
        base = b->pool + b->pool_used;
        b->pool_used += need;
    } else {
        if (cudaMalloc((void **) &base, need) != cudaSuccess) {
            (void) cudaGetLastError();
            return GGML_STATUS_ALLOC_FAILED;
        }
        b->blocks.push_back(base);
    }
    qsa_clock_layout(m, &entry->device_map, base);
    CUDA_CHECK(cudaMemset(m.tags, 0xff, m.slots*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(m.table, 0xff, size_t(m.records)*(m.group/m.cells)*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(m.stamp, 0, m.slots*sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(m.ref, 0, m.slots*sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(m.ctl, 0, QSA_CLOCK_CTL_WORDS*sizeof(uint32_t)));
    if (m.seen) {
        CUDA_CHECK(cudaMemset(m.seen, 0, size_t(m.records)*sizeof(uint32_t)));
    }
    if (m.stage_ctl) {
        CUDA_CHECK(cudaMemset(m.stage_ctl, 0, LSA_STAGE_CTL_WORDS*sizeof(uint32_t)));
    }
    if (m.stage_mask) {
        CUDA_CHECK(cudaMemset(m.stage_mask, 0, size_t(m.records)*sizeof(uint32_t)));
    }
    if (m.overflow_hist) {
        CUDA_CHECK(cudaMemset(m.overflow_hist, 0, 9*sizeof(unsigned long long)));
    }
    if (m.stage_stride) {
        CUDA_CHECK(cudaMemset(m.stage_valid, 0, size_t(m.stage_n)*m.heads*sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(m.stage_stats, 0, 3*sizeof(unsigned long long)));
    }
    CUDA_CHECK(cudaMemset(m.meta, 0, size_t(m.records)*m.heads*qsa_meta_head_bytes(m)));
    CUDA_CHECK(cudaMemcpy(entry->device_map, &m, sizeof(m), cudaMemcpyHostToDevice));
    entry->device = t->device;
    if (m.cells == GGML_LSA_PAGE_CELLS && lsa_prefetch_cap() > 0) {
        // events and the side stream are made here, never while a CUDA graph is being captured
        std::lock_guard<std::mutex> lock(lsa_pf_mutex);
        CUDA_CHECK(cudaEventCreateWithFlags(&entry->pf_done, cudaEventDisableTiming));
        if (!lsa_pf_stream[t->device]) {
            int lo = 0, hi = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            CUDA_CHECK(cudaStreamCreateWithPriority(&lsa_pf_stream[t->device], cudaStreamNonBlocking, lo));
            CUDA_CHECK(cudaEventCreateWithFlags(&lsa_pf_fork[t->device], cudaEventDisableTiming));
        }
        lsa_pf_entries.push_back(entry.get());
    }
    if (m.ahead_list) {
        // stream and events are made here, never while a CUDA graph is being captured
        std::lock_guard<std::mutex> lock(lsa_pf_mutex);
        if (!lsa_ahead_stream[t->device]) {
            int lo = 0, hi = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            CUDA_CHECK(cudaStreamCreateWithPriority(&lsa_ahead_stream[t->device], cudaStreamNonBlocking, lo));
            CUDA_CHECK(cudaEventCreateWithFlags(&lsa_ahead_fork[t->device], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&lsa_ahead_done[t->device], cudaEventDisableTiming));
        }
        entry->owner = buffer->context;
        if (lsa_stage_ahead_dma()) {
            const size_t bytes = 2*(size_t(m.stage_n) + 1)*sizeof(int32_t);
            CUDA_CHECK(cudaHostAlloc((void **) &entry->ahead_host, bytes, cudaHostAllocMapped));
            memset(entry->ahead_host, 0, bytes);
            CUDA_CHECK(cudaHostGetDevicePointer((void **) &entry->ahead_host_dev, entry->ahead_host, 0));
            CUDA_CHECK(cudaEventCreateWithFlags(&entry->mirror_done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&entry->applied_used, cudaEventDisableTiming));
        }
        lsa_ahead_entries.push_back(entry.get());
    }
    tensor->extra = entry.get();
    static_cast<qsa_clock_buffer *>(buffer->context)->tensors.push_back(std::move(entry));
    return GGML_STATUS_SUCCESS;
}

static void qsa_clock_written(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, size_t offset, size_t size);
static void qsa_clock_written_range(ggml_backend_buffer_t buffer, qsa_clock_tensor * e, size_t offset, size_t size);

static sj_kvarn_lowbits::strip_master_layout qsa_clock_master_layout(ggml_backend_buffer_t buffer, const qsa_clock_tensor * e) {
    const auto & m = e->map;
    sj_kvarn_lowbits::strip_master_layout l{};
    GGML_ASSERT(m.cells == GGML_LSA_PAGE_CELLS && m.strip_master && e->host);
    GGML_ASSERT(sj_kvarn_lowbits::strip_master_clock_layout(m.strip_master_bits >> 8, m.strip_master_bits & 255,
        m.records, m.heads, m.stride, m.stage_stride, m.strip_master_capacity, l));
    const uintptr_t base = reinterpret_cast<uintptr_t>(static_cast<qsa_clock_buffer *>(buffer->context)->host);
    const uintptr_t start = reinterpret_cast<uintptr_t>(e->host);
    GGML_ASSERT(start >= base && start - base <= buffer->size && m.strip_master_capacity <= buffer->size - (start - base));
    return l;
}

static size_t qsa_clock_master_offset(const qsa_clock_tensor * e, const sj_kvarn_lowbits::strip_master_layout & l,
        const ggml_tensor * tensor, size_t offset, size_t size) {
    const uintptr_t root = reinterpret_cast<uintptr_t>(e->host), start = reinterpret_cast<uintptr_t>(tensor->data);
    GGML_ASSERT(start >= root);
    size_t absolute;
    GGML_ASSERT(sj_kvarn_lowbits::strip_master_view_range(l, size_t(e->map.records)*e->map.heads,
        start - root, ggml_nbytes(tensor), offset, size, absolute));
    return absolute;
}

const ggml_cuda_qsa_clock_map * ggml_cuda_qsa_clock_strip_master_map(ggml_backend_cuda_context & ctx, const ggml_tensor * body) {
    if (!body) return nullptr;
    auto * e = qsa_clock_entry(body);
    if (!e || !e->map.strip_master) return nullptr;
    const auto * root = body;
    while (root->view_src) root = root->view_src;
    const auto l = qsa_clock_master_layout(root->buffer, e);
    const auto & m = e->map;
    GGML_ASSERT(e->device == ctx.device && body->data == e->host && body->type == GGML_TYPE_I8 && ggml_is_contiguous(body));
    GGML_ASSERT(ggml_nbytes(body) == size_t(m.records)*m.heads*l.canonical_bytes);
    GGML_ASSERT(m.stage_all && m.stage_n >= m.records);
    return &m;
}

static bool qsa_clock_memset_span(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        uint8_t value, size_t offset, size_t size) {
    auto * b = static_cast<qsa_clock_buffer *>(buffer->context);
    if (std::none_of(b->tensors.begin(), b->tensors.end(), [](const auto & e) { return e->map.strip_master; })) return false;
    const uintptr_t base = reinterpret_cast<uintptr_t>(b->host), start = reinterpret_cast<uintptr_t>(tensor->data);
    GGML_ASSERT(start >= base && start - base <= buffer->size);
    const size_t bytes = ggml_nbytes(tensor);
    GGML_ASSERT(bytes <= buffer->size - (start - base) && offset <= bytes && size <= bytes - offset);
    offset += start - base;
    const size_t end = offset + size;
    std::vector<qsa_clock_tensor *> entries;
    for (const auto & e : b->tensors) entries.push_back(e.get());
    std::sort(entries.begin(), entries.end(), [](const auto * a, const auto * c) {
        return reinterpret_cast<uintptr_t>(a->host) < reinterpret_cast<uintptr_t>(c->host);
    });
    std::vector<sj_kvarn_lowbits::strip_master_arena> arenas;
    size_t previous_end = 0;
    for (const auto * e : entries) {
        const uintptr_t root = reinterpret_cast<uintptr_t>(e->host);
        GGML_ASSERT(root >= base && root - base <= buffer->size);
        GGML_ASSERT(e->map.stride && size_t(e->map.records) <= SIZE_MAX/e->map.stride);
        const size_t canonical = size_t(e->map.records)*e->map.stride;
        const size_t capacity = e->map.strip_master ? e->map.strip_master_capacity : canonical;
        GGML_ASSERT(root - base >= previous_end && capacity <= buffer->size - (root - base));
        previous_end = root - base + capacity;
        if (e->map.strip_master) {
            const auto l = qsa_clock_master_layout(buffer, e);
            arenas.push_back({l.bits_k, l.bits_v, size_t(root - base), size_t(e->map.records)*e->map.heads, capacity});
        }
    }
    GGML_ASSERT(sj_kvarn_lowbits::strip_master_fill_arenas(reinterpret_cast<uint8_t *>(b->host),
        buffer->size, arenas.data(), arenas.size(), value, offset, size));
    for (auto * e : entries) {
        const size_t root = reinterpret_cast<uintptr_t>(e->host) - base;
        const size_t first = std::max(offset, root);
        const size_t last = std::min(end, root + size_t(e->map.records)*e->map.stride);
        if (first < last) qsa_clock_written_range(buffer, e, first - root, last - first);
    }
    return true;
}

static void qsa_clock_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    qsa_clock_sync(buffer);
    auto * e = qsa_clock_entry(tensor);
    if (e && e->map.strip_master) {
        const auto l = qsa_clock_master_layout(buffer, e);
        const size_t absolute = qsa_clock_master_offset(e, l, tensor, offset, size);
        GGML_ASSERT(sj_kvarn_lowbits::strip_master_set_records(l, reinterpret_cast<uint8_t *>(e->host),
            size_t(e->map.records)*e->map.heads, static_cast<const uint8_t *>(data), absolute, size));
    } else {
        memcpy(static_cast<char *>(tensor->data) + offset, data, size);
    }
    qsa_clock_written(buffer, tensor, offset, size);
}

static void qsa_clock_get(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    qsa_clock_sync(buffer);
    auto * e = qsa_clock_entry(tensor);
    if (e && e->map.strip_master) {
        const auto l = qsa_clock_master_layout(buffer, e);
        const size_t absolute = qsa_clock_master_offset(e, l, tensor, offset, size);
        GGML_ASSERT(sj_kvarn_lowbits::strip_master_get_records(l, reinterpret_cast<const uint8_t *>(e->host),
            size_t(e->map.records)*e->map.heads, static_cast<uint8_t *>(data), absolute, size));
    } else {
        memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
    }
}

static void qsa_clock_memset(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    qsa_clock_sync(buffer);
    auto * e = qsa_clock_entry(tensor);
    if (e && e->map.strip_master) {
        const auto l = qsa_clock_master_layout(buffer, e);
        const size_t absolute = qsa_clock_master_offset(e, l, tensor, offset, size);
        GGML_ASSERT(sj_kvarn_lowbits::strip_master_fill_records(l, reinterpret_cast<uint8_t *>(e->host),
            size_t(e->map.records)*e->map.heads, value, absolute, size));
    } else if (e || !qsa_clock_memset_span(buffer, tensor, value, offset, size)) {
        memset(static_cast<char *>(tensor->data) + offset, value, size);
    }
    qsa_clock_written(buffer, tensor, offset, size);
}

static void qsa_clock_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    qsa_clock_reset(buffer);
    if (buffer->size) memset(qsa_clock_base(buffer), value, buffer->size);
    for (const auto & e : static_cast<qsa_clock_buffer *>(buffer->context)->tensors) {
        if (e->map.strip_master) {
            const auto l = qsa_clock_master_layout(buffer, e.get());
            const size_t records = size_t(e->map.records)*e->map.heads;
            GGML_ASSERT(sj_kvarn_lowbits::strip_master_fill_records(l, reinterpret_cast<uint8_t *>(e->host),
                records, value, 0, records*size_t(l.canonical_bytes)));
        }
        // record pages carry no card meta (qsa_meta_head_bytes = 0): the fixed QSA size overran their empty region
        const size_t meta_bytes = size_t(e->map.records)*e->map.heads*qsa_meta_head_bytes(e->map);
        if (meta_bytes) CUDA_CHECK(cudaMemset(e->map.meta, value, meta_bytes));
    }
}

static const ggml_backend_buffer_i qsa_clock_buffer_i = {
    qsa_clock_free, qsa_clock_base, qsa_clock_init,
    qsa_clock_memset, qsa_clock_set, qsa_clock_get,
    nullptr, nullptr, nullptr, qsa_clock_clear, nullptr,
};

static ggml_backend_buffer_t qsa_clock_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * t = static_cast<qsa_clock_type *>(buft->context);
    auto b = std::make_unique<qsa_clock_buffer>();
    b->device = t->device;
    ggml_cuda_set_device(t->device);
    if (size == 0) return ggml_backend_buffer_init(buft, qsa_clock_buffer_i, b.release(), size);
    const cudaError_t error = cudaHostAlloc(&b->host, std::max<size_t>(size, 1), cudaHostAllocMapped);
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: pinned allocation failed: %s\n", __func__, cudaGetErrorString(error));
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, qsa_clock_buffer_i, b.release(), size);
}

static size_t qsa_clock_alignment(ggml_backend_buffer_type_t) { return 256; }

// Report a device buffer so all host access uses the synchronized get/set callbacks.
static bool qsa_clock_host(ggml_backend_buffer_type_t) { return false; }

static ggml_backend_buffer_type_t qsa_clock_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots, uint32_t cells,
        uint32_t page_heads = 0) {
    const uint32_t group = 128;
    // 4/4 scalar record (QSA pages, LSA strips); LSA record pages take any record size with 16-byte head slices
    const uint32_t heads = cells == GGML_LSA_PAGE_CELLS ? page_heads : stride/(128*256 + 2*(3*256 + 3*128));
    const bool shape_ok = cells == GGML_LSA_PAGE_CELLS ? heads && stride % heads == 0 && (stride/heads) % 16 == 0 :
        heads && stride == size_t(heads)*(128*256 + 2*(3*256 + 3*128));
    if (!dev || !stride || stride % 4 || !slots || group != 128 || !shape_ok || stride > std::numeric_limits<size_t>::max()/slots) {
        return nullptr;
    }
    int device = -1;
    for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
        if (ggml_backend_buft_get_device(ggml_backend_cuda_buffer_type(i)) == dev) device = i;
    }
    if (device < 0) return nullptr;
    static std::mutex mutex;
    static std::vector<std::unique_ptr<qsa_clock_type>> types;
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto & t : types) {
        if (t->device == device && t->stride == stride && t->slots == slots && t->cells == cells) return &t->buft;
    }
    auto t = std::make_unique<qsa_clock_type>();
    t->stride = stride;
    t->slots = slots;
    t->group = group;
    t->heads = heads;
    t->cells = cells;
    t->device = device;
    t->name = std::string(cells == GGML_LSA_STRIP_CELLS ? "CUDA_LSA_Strip_" : cells == GGML_LSA_PAGE_CELLS ? "CUDA_LSA_Page_" : "CUDA_QSA_Clock_") + std::to_string(device) +
        "_" + std::to_string(stride) + "_" + std::to_string(slots);
    if (cells == GGML_LSA_PAGE_CELLS) {
        GGML_LOG_WARN("LSA RESIDENT: 128-cell record slots, records=%u heads=%u (%.1f MiB per layer)\n", slots, heads,
            double(slots)*stride/(1024.0*1024.0));
    } else if (cells == GGML_LSA_STRIP_CELLS) {
        GGML_LOG_WARN("LSA RESIDENT: 16-cell strip slots, record_groups=%u strips=%llu heads=%u (%.1f MiB per layer)\n", slots,
            (unsigned long long) slots*(group/cells), heads, double(slots)*(group/cells)*heads*GGML_LSA_STRIP_HEAD_BYTES/(1024.0*1024.0));
    } else {
        GGML_LOG_INFO("QSA CLOCK: selected-page policy enabled, record_groups=%u pages=%llu heads=%u\n", slots,
            (unsigned long long) slots*(group/4), heads);
    }
    t->buft = {{qsa_clock_name, qsa_clock_alloc, nullptr, qsa_clock_alignment, nullptr, nullptr, nullptr, qsa_clock_host}, dev, t.get()};
    auto * result = &t->buft;
    types.push_back(std::move(t));
    return result;
}

ggml_backend_buffer_type_t ggml_backend_cuda_qsa_clock_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots) {
    return qsa_clock_buffer_type(dev, stride, slots, GGML_QSA_PAGE_CELLS);
}

ggml_backend_buffer_type_t ggml_backend_cuda_qsa_strip_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots) {
    return qsa_clock_buffer_type(dev, stride, slots, GGML_LSA_STRIP_CELLS);
}

ggml_backend_buffer_type_t ggml_backend_cuda_qsa_page_buffer_type(ggml_backend_dev_t dev, size_t stride, uint32_t slots, uint32_t heads) {
    return qsa_clock_buffer_type(dev, stride, slots, GGML_LSA_PAGE_CELLS, heads);
}

size_t ggml_backend_cuda_qsa_clock_device_bytes(ggml_backend_buffer_type_t buft, const ggml_tensor * body) {
    const auto * t = static_cast<const qsa_clock_type *>(buft->context);
    const size_t slots = std::min<size_t>(t->slots, ggml_nbytes(body)/t->stride);
    const size_t pages = slots*(t->group/t->cells);
    const size_t records = ggml_nbytes(body)/t->stride;
    ggml_cuda_qsa_clock_map m{};
    m.cells = t->cells; m.stride = t->stride; m.heads = t->heads;
    if (t->cells == GGML_LSA_PAGE_CELLS && lsa_stage_all()) {
        m.slots = pages; m.records = records; m.group = t->group; m.stage_n = records;
        m.stage_stride = lsa_stage_strips() ? lsa_stage_stride(t->stride, t->heads) : 0;
        return qsa_clock_layout(m, nullptr, nullptr);
    }
    return pages*(t->heads*qsa_page_head_bytes(m) + 5*sizeof(int32_t)) +
        records*(t->heads*qsa_meta_head_bytes(m) + (t->group/t->cells)*sizeof(int32_t)) +
        QSA_CLOCK_CTL_WORDS*sizeof(uint32_t) + sizeof(ggml_cuda_qsa_clock_map);
}

size_t ggml_backend_cuda_qsa_clock_shared_bytes(ggml_backend_buffer_type_t buft, const ggml_context * ctx) {
    const auto * t = static_cast<const qsa_clock_type *>(buft->context);
    if (t->cells != GGML_LSA_PAGE_CELLS || !lsa_stage_all()) return 0;
    size_t bytes = 0;
    for (auto * body = ggml_get_first_tensor(ctx); body; body = ggml_get_next_tensor(ctx, body)) {
        if (!body->view_src) {
            const size_t records = ggml_nbytes(body)/t->stride;
            const size_t stride = lsa_stage_stride(t->stride, t->heads);
            GGML_ASSERT(records <= SIZE_MAX/stride);
            bytes = std::max(bytes, records*stride);
        }
    }
    return bytes;
}

bool ggml_cuda_lsa_page_full_stage(const ggml_tensor * body) {
    const auto * e = body ? qsa_clock_entry(body) : nullptr;
    return e && e->map.cells == GGML_LSA_PAGE_CELLS && e->map.stage_all;
}

const char * ggml_cuda_qsa_clock_body(const ggml_tensor * body) {
    auto * entry = qsa_clock_entry(body);
    // Body records are aligned; the low bit identifies a device-side split map.
    // tag 3: QSA four-cell pages; tag 2: LSA strip slots (read only by the SJ-KVaRN stream kernel, others read host) or
    // LSA record pages (low-bit kernels; ggml_cuda_lsa_page_body)
    return entry ? reinterpret_cast<const char *>(reinterpret_cast<uintptr_t>(entry->device_map) |
                                                  (entry->map.cells != GGML_QSA_PAGE_CELLS ? 2 : 3)) :
        static_cast<const char *>(body->data);
}

// Repack nibbles, not reconstructed values. Channel scales retain their original fragment order.
static __device__ void qsa_clock_copy_page(ggml_cuda_qsa_clock_map m, int page, int slot, int head) {
    const int record = page/(m.group/4);
    const int first = (page%(m.group/4))*4;
    const char * src = m.host + size_t(record)*m.stride + size_t(head)*(m.stride/m.heads);
    char * dst = m.card + (size_t(slot)*m.heads + head)*GGML_QSA_PAGE_HEAD_BYTES;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        const bool is_v = i >= 512;
        const int j = i%512, token = first + j/128, d = 2*(j%128);
        const uint32_t b0 = ggml_sj_kvarn::code_bit(token, d, is_v, 256, 128, 4);
        const uint32_t b1 = ggml_sj_kvarn::code_bit(token, d + 1, is_v, 256, 128, 4);
        const auto * payload = reinterpret_cast<const uint32_t *>(src + (is_v ? 16384 : 0));
        dst[i] = ((payload[b0/32] >> (b0%32)) & 15) | (((payload[b1/32] >> (b1%32)) & 15) << 4);
    }
    for (int i = threadIdx.x; i < 12; i += blockDim.x) {
        const int offset = i < 4 ? 33792 : i < 8 ? 34560 : 34816;
        reinterpret_cast<uint16_t *>(dst + 1024)[i] =
            reinterpret_cast<const uint16_t *>(src + offset)[first + i%4];
    }
}

// LSA strip slot: byte copy of strip sl of the record (K strip, V strip, k_tok, v_scale, v_zero; scalar 4/4, D 256, G 128)
static __device__ void qsa_strip_copy(ggml_cuda_qsa_clock_map m, int page, int slot, int head) {
    constexpr int D = 256, G = 128, STRIP = 2048;
    constexpr int v_payload = D*4/8*G, k_tok = v_payload + D*4/8*G + 4*D, v_scale = k_tok + 2*G + 2*D, v_zero = v_scale + 2*G;
    const int record = page/(G/GGML_LSA_STRIP_CELLS), sl = page%(G/GGML_LSA_STRIP_CELLS);
    const char * src = m.host + size_t(record)*m.stride + size_t(head)*(m.stride/m.heads);
    char * dst = m.card + (size_t(slot)*m.heads + head)*GGML_LSA_STRIP_HEAD_BYTES;
    for (int q = threadIdx.x; q < GGML_LSA_STRIP_HEAD_BYTES/16; q += blockDim.x) {
        const char * s;
        if (q < STRIP/16) {
            s = src + sl*STRIP + 16*q;
        } else if (q < 2*STRIP/16) {
            s = src + v_payload + sl*STRIP + 16*(q - STRIP/16);
        } else {
            const int i = q - 2*STRIP/16;
            s = src + (i < 2 ? k_tok : i < 4 ? v_scale : v_zero) + sl*32 + 16*(i & 1);
        }
        reinterpret_cast<uint4 *>(dst)[q] = *reinterpret_cast<const uint4 *>(s);
    }
}

static __device__ sj_kvarn_lowbits::strip_master_layout lsa_master_layout(const ggml_cuda_qsa_clock_map & m) {
    sj_kvarn_lowbits::strip_master_layout l{};
    if (m.cells != GGML_LSA_PAGE_CELLS || m.group != 128 || !m.stage_all ||
        !sj_kvarn_lowbits::strip_master_clock_layout(m.strip_master_bits >> 8, m.strip_master_bits & 255,
            m.records, m.heads, m.stride, m.stage_stride, m.strip_master_capacity, l) ||
        !m.strip_master || reinterpret_cast<uintptr_t>(m.strip_master)%16 || !m.slots || m.slots > m.records) __trap();
    return l;
}

// LSA record page: byte copy of one KV head's slice of a whole record (any SJ-KVaRN record layout, 16-byte multiple)
static __device__ void lsa_record_copy(const ggml_cuda_qsa_clock_map & m, int record, char * base, int index, int head) {
    if (m.strip_master) {
        const auto l = lsa_master_layout(m);
        if (record < 0 || uint32_t(record) >= m.records || head < 0 || uint32_t(head) >= m.heads ||
            !base || reinterpret_cast<uintptr_t>(base)%16 || base != m.card ||
            index < 0 || uint32_t(index) >= m.slots) { __trap(); return; }
        const auto * src = reinterpret_cast<const uint4 *>(m.strip_master + (size_t(record)*m.heads + head)*l.bytes);
        auto * dst = reinterpret_cast<uint8_t *>(base + (size_t(index)*m.heads + head)*l.canonical_bytes);
        for (int q = threadIdx.x; q < l.bytes/16; q += blockDim.x) {
            const uint4 value = src[q];
            const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
            for (int b = 0; b < 16; ++b) {
                const int from = sj_kvarn_lowbits::strip_master_source_byte(l, q*16 + b);
                if (from >= 0 && sj_kvarn_lowbits::strip_master_byte(l, from) == q*16 + b) dst[from] = bytes[b];
            }
        }
        return;
    }
    const size_t hb = m.stride/m.heads;
    const uint4 * src = reinterpret_cast<const uint4 *>(m.host + size_t(record)*m.stride + size_t(head)*hb);
    uint4 * dst = reinterpret_cast<uint4 *>(base + size_t(index)*m.stride + size_t(head)*hb);
    for (size_t q = threadIdx.x; q < hb/16; q += blockDim.x) dst[q] = src[q];
}
static __device__ void lsa_page_copy(const ggml_cuda_qsa_clock_map & m, int record, int slot, int head) {
    lsa_record_copy(m, record, m.card, slot, head);
}
// GGML_CUDA_LSA_STAGE_AHEAD: on the side stream, ahead_list entry i (this layer's last staged pick) to stage index i
static __global__ void lsa_stage_ahead_copy(ggml_cuda_qsa_clock_map m) {
    const uint32_t n = min(m.stage_ctl[7], m.stage_n);
    for (uint32_t i = blockIdx.x; i < n; i += gridDim.x) {
        const int record = m.ahead_list[i];
        if (record < 0 || uint32_t(record) >= m.records) continue;
        lsa_record_copy(m, record, m.stage, int(i), blockIdx.y);
    }
}

// HOSTKV08 DMA ahead: after a decode resolve, its offer list (stage_ctl 7 entries of ahead_list) to the mapped host mirror
static __global__ void lsa_ahead_mirror(ggml_cuda_qsa_clock_map m, int32_t * mirror) {
    const uint32_t n = min(m.stage_ctl[7], m.stage_n);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) mirror[1 + i] = m.ahead_list[i];
    if (threadIdx.x == 0) mirror[0] = int32_t(n);
}

// GGML_CUDA_LSA_RESIDENT_STAGE: a picked record without a CLOCK slot takes the next stage index for this resolve
// (table -3 - k); with no stage, or the stage full, it is read from the master (table -1) exactly as before.
static __device__ void lsa_stage_or_master(const ggml_cuda_qsa_clock_map & m, int record) {
    if (m.stage_n) {
        const uint32_t k = atomicAdd(m.stage_ctl, 1u);
        if (k < m.stage_n) {
            m.stage_page[k] = record;
            m.table[record] = -3 - int(k);
            if (m.ahead_list) { // GGML_CUDA_LSA_STAGE_AHEAD: offered to the next ahead copy of this layer
                const uint32_t a = atomicAdd(m.stage_ctl + 7, 1u);
                if (a < m.stage_n) m.ahead_list[a] = record;
            }
            return;
        }
    }
    m.table[record] = -1;
}

// start of a resolve (and of a prefill call): the previous resolve's stage entries go back to -1, so nothing reads the
// stage after another layer has overwritten it; counters. One block.
// apply (GGML_CUDA_LSA_STAGE_AHEAD, decode only, the ahead copy of this layer has landed): ahead_list entry j now sits
// at stage index j, so its record takes table entry -3 - j and the resolve's new stage entries start after them. The
// mark drops an entry whose record this ubatch seals (and the resolve copies it again).
// applied (HOSTKV08 DMA ahead, nullptr otherwise): mapped [count, records...] the host copied to stage indices 0..; it
// replaces the device's own offer list as what the reset applies.
static __global__ void lsa_stage_reset(ggml_cuda_qsa_clock_map m, int apply, const int32_t * applied) {
    const uint32_t n = m.stage_ctl[0];
    const uint32_t used = min(n, m.stage_n);
    const uint32_t first = min(m.stage_ctl[1], used);
    for (uint32_t i = threadIdx.x; i < used; i += blockDim.x) {
        const int record = m.stage_page[i];
        if (record >= 0 && uint32_t(record) < m.records && m.table[record] == -3 - int(i)) m.table[record] = -1;
    }
    __syncthreads();
    uint32_t keep = 0;
    const int32_t * list = m.ahead_list;
    if (m.ahead_list && apply) {
        if (applied) {
            keep = min(uint32_t(max(applied[0], 0)), m.stage_n);
            list = applied + 1;
        } else {
            keep = min(m.stage_ctl[7], m.stage_n);
        }
    }
    for (uint32_t j = threadIdx.x; j < keep; j += blockDim.x) {
        const int record = list[j];
        const bool ok = record >= 0 && uint32_t(record) < m.records;
        m.stage_page[j] = ok ? record : -1;
        if (ok) atomicCAS(m.table + record, -1, -3 - int(j));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        auto * c = reinterpret_cast<unsigned long long *>(m.stage_ctl + 2);
        c[0] += used - first; // copied by the resolve (ahead entries below first were not)
        c[1] += n - used;
        if (m.ahead_list) {
            reinterpret_cast<unsigned long long *>(m.stage_ctl + 8)[1] += keep;
            m.stage_ctl[7] = 0; // this resolve offers the next list
        }
        m.stage_ctl[0] = keep;
        m.stage_ctl[1] = keep;
    }
}

static __global__ void lsa_stage_copy(ggml_cuda_qsa_clock_map m, int bits_k, int bits_v) {
    if (m.strip_master) {
        lsa_master_layout(m);
        if (!m.stage || reinterpret_cast<uintptr_t>(m.stage)%16 || !m.stage_ctl || !m.stage_page ||
            !m.stage_mask || !m.stage_valid || !m.stage_stats || m.stage_n < m.records ||
            bits_k != int(m.strip_master_bits >> 8) || bits_v != int(m.strip_master_bits & 255)) { __trap(); return; }
    }
    const uint32_t used = min(m.stage_ctl[0], m.stage_n);
    const uint32_t first = min(m.stage_ctl[1], used); // GGML_CUDA_LSA_STAGE_AHEAD: entries below first are in place
    for (uint32_t i = first + blockIdx.x; i < used; i += gridDim.x) {
        if (m.strip_master && (m.stage_page[i] < 0 || uint32_t(m.stage_page[i]) >= m.records)) { __trap(); return; }
        if (m.overflow_hist && blockIdx.y == 0 && threadIdx.x == 0) {
            const uint32_t wanted = m.stage_mask[m.stage_page[i]] & 255u;
            atomicAdd(m.overflow_hist + __popc(wanted), 1ull);
        }
        if (!m.stage_stride) {
            lsa_record_copy(m, m.stage_page[i], m.stage, i, blockIdx.y);
            continue;
        }
        const int record = m.stage_page[i], head = blockIdx.y;
        const uint32_t wanted = m.stage_mask[record];
        if (m.strip_master) {
            const auto l = lsa_master_layout(m);
            if (record < 0 || uint32_t(record) >= m.records || uint32_t(head) >= m.heads ||
                !m.stage_valid || !m.stage_stats || wanted > 255u) { __trap(); return; }
            const auto * src = reinterpret_cast<const uint4 *>(m.strip_master + (size_t(record)*m.heads + head)*l.bytes);
            auto * dst = reinterpret_cast<uint4 *>(m.stage + (size_t(i)*m.heads + head)*l.bytes);
            for (int q = threadIdx.x; q < 1536/16; q += blockDim.x) dst[q] = src[q];
            int rank = 0;
            for (int s = 0; s < 8; ++s) {
                if (!(wanted & (1u << s))) continue;
                const int from = (1536 + s*l.strip_bytes)/16, to = (1536 + rank++*l.strip_bytes)/16;
                for (int q = threadIdx.x; q < l.strip_bytes/16; q += blockDim.x) dst[to + q] = src[from + q];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                m.stage_valid[size_t(i)*m.heads + head] = wanted;
                if (head == 0) {
                    atomicAdd(m.stage_stats, (1536ull + rank*l.strip_bytes)*m.heads);
                    atomicAdd(m.stage_stats + 1, (unsigned long long) rank*m.heads);
                    atomicAdd(m.stage_stats + 2, 1ull);
                }
            }
            __syncthreads();
            continue;
        }
        const char * src = m.host + size_t(record)*m.stride + head*(m.stride/m.heads);
        char * dst = m.stage + size_t(i)*m.stage_stride + head*(m.stage_stride/m.heads);
        const int vp = 4096*bits_k, ks = vp + 4096*bits_v, kz = ks + 512;
        const int kt = kz + 512, vc = kt + 256, vs = vc + 512, vz = vs + 256;
        for (int q = threadIdx.x; q < 1536/16; q += blockDim.x) {
            const int off = q*16;
            const int from = off < 512 ? ks + off : off < 1024 ? kz + off - 512 : vc + off - 1024;
            reinterpret_cast<uint4 *>(dst)[q] = *reinterpret_cast<const uint4 *>(src + from);
        }
        int rank = 0;
        const int strip_bytes = sj_kvarn_lowbits::stage_strip_bytes(bits_k, bits_v);
        for (int s = 0; s < 8; ++s) {
            if (!(wanted & (1u << s))) continue;
            char * out = dst + 1536 + rank++*strip_bytes;
            const int kb = 256*(2*bits_k + 1), vb = 256*(2*bits_v + 1);
            for (int b = threadIdx.x; b < kb + vb; b += blockDim.x) {
                const bool v = b >= kb;
                const int from = sj_kvarn_lowbits::stage_source_byte(v ? bits_v : bits_k, s, v ? b - kb : b);
                out[b] = from < 0 ? 0 : src[(v ? vp : 0) + from];
            }
            for (int q = threadIdx.x; q < 6; q += blockDim.x) {
                const int from = (q < 2 ? kt : q < 4 ? vs : vz) + s*32 + (q & 1)*16;
                reinterpret_cast<uint4 *>(out + kb + vb)[q] = *reinterpret_cast<const uint4 *>(src + from);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            m.stage_valid[size_t(i)*m.heads + head] = wanted;
            if (head == 0) {
                atomicAdd(m.stage_stats, (1536ull + rank*strip_bytes)*m.heads);
                atomicAdd(m.stage_stats + 1, (unsigned long long) rank*m.heads);
                atomicAdd(m.stage_stats + 2, 1ull);
            }
        }
        __syncthreads();
    }
}

static __global__ void qsa_clock_refresh_records(ggml_cuda_qsa_clock_map m, const int32_t * desc,
        int first_record, int end_record) {
    int record = first_record + blockIdx.x;
    if (desc) {
        const int S = desc[GGML_SJKVARN_DESC_S], G = desc[GGML_SJKVARN_DESC_G];
        const int g = (desc[GGML_SJKVARN_DESC_B_OLD] - S)/G + blockIdx.x;
        if (g >= (desc[GGML_SJKVARN_DESC_B] - S)/G) return;
        record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
    } else if (record >= end_record) {
        return;
    }
    if (record < 0 || record >= m.records) return;
    const int head = blockIdx.y;
    if (m.cells == GGML_LSA_PAGE_CELLS) {
        const int slot = m.table[record];
        if (slot >= 0) lsa_page_copy(m, record, slot, head);
        return;
    }
    const char * src = m.host + size_t(record)*m.stride + size_t(head)*(m.stride/m.heads);
    char * meta = m.meta + (size_t(record)*m.heads + head)*GGML_QSA_META_HEAD_BYTES;
    for (int i = threadIdx.x; i < 384; i += blockDim.x) {
        reinterpret_cast<uint32_t *>(meta)[i] =
            reinterpret_cast<const uint32_t *>(src + (i < 256 ? 32768 : 33024))[i];
    }
    for (int p = 0; p < m.group/m.cells; ++p) {
        const int page = record*(m.group/m.cells) + p;
        const int slot = m.table[page];
        if (slot < 0) continue;
        if (m.cells == GGML_LSA_STRIP_CELLS) {
            qsa_strip_copy(m, page, slot, head);
        } else {
            qsa_clock_copy_page(m, page, slot, head);
        }
    }
}

static void qsa_clock_written_range(ggml_backend_buffer_t buffer, qsa_clock_tensor * e, size_t offset, size_t size) {
    if (!size) return;
    if (e->map.cells == GGML_LSA_STRIP_CELLS || e->map.cells == GGML_LSA_PAGE_CELLS) {
        // test only (restore gate power arm): leave LSA slots stale after a host write, so a restore must change the output
        static const bool no_refresh = getenv("GGML_CUDA_LSA_RESIDENT_TEST_NO_REFRESH") != nullptr;
        if (no_refresh) return;
    }
    if (e->pf_done) {
        // unconditional: a stream-side join (lsa_pf_join) clears pf_pending after only queueing its wait, so the host
        // write must wait for the last recorded pass itself (an event never recorded returns at once)
        CUDA_CHECK(cudaEventSynchronize(e->pf_done));
        e->pf_pending = false;
        for (bool & issued : e->j3_issued) issued = false;
    }
    if (e->device >= 0 && lsa_ahead_done[e->device]) {
        // GGML_CUDA_LSA_STAGE_AHEAD: the master changes under any queued ahead copy, so none is applied
        CUDA_CHECK(cudaEventSynchronize(lsa_ahead_done[e->device]));
        ++lsa_ahead_seq[e->device];
    }
    GGML_ASSERT(e->map.stride && size_t(e->map.records) <= SIZE_MAX/e->map.stride);
    const size_t bytes = size_t(e->map.records)*e->map.stride;
    GGML_ASSERT(offset <= bytes && size <= bytes - offset);
    const size_t first = offset/e->map.stride;
    const size_t end = (offset + size - 1)/e->map.stride + 1;
    GGML_ASSERT(end <= e->map.records);
    qsa_clock_refresh_records<<<dim3(end - first, e->map.heads), 128>>>(e->map, nullptr, first, end);
    CUDA_CHECK(cudaGetLastError());
    qsa_clock_sync(buffer);
}

static void qsa_clock_written(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, size_t offset, size_t size) {
    auto * e = qsa_clock_entry(tensor);
    if (!e || !size) return;
    const uintptr_t root = reinterpret_cast<uintptr_t>(e->host), start = reinterpret_cast<uintptr_t>(tensor->data);
    GGML_ASSERT(start >= root && start - root <= SIZE_MAX - offset);
    qsa_clock_written_range(buffer, e, start - root + offset, size);
}

void ggml_cuda_qsa_clock_refresh(ggml_backend_cuda_context & ctx, const ggml_tensor * body) {
    auto * e = qsa_clock_entry(body);
    if (!e) return;
    lsa_pf_join(ctx.stream(), e); // the seal refresh reads this layer's table: an in-flight prefetch must land first
    const auto * desc = body->src[2];
    const int max_records = ggml_get_op_params_i32(body, 5);
    GGML_ASSERT(desc && max_records > 0);
    qsa_clock_refresh_records<<<dim3(max_records, e->map.heads), 128, 0, ctx.stream()>>>(
        e->map, static_cast<const int32_t *>(desc->data), 0, e->map.records);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_qsa_clock_seal_strip_master(ggml_backend_cuda_context & ctx, ggml_tensor * body) {
    auto * e = qsa_clock_entry(body);
    if (!e || !e->map.strip_master) return false;
    const auto * root = body;
    while (root->view_src) root = root->view_src;
    const auto l = qsa_clock_master_layout(root->buffer, e);
    const auto & m = e->map;
    GGML_ASSERT(e->device == ctx.device && body->data == e->host && body->type == GGML_TYPE_I8 && ggml_is_contiguous(body));
    GGML_ASSERT(ggml_nbytes(body) == size_t(m.records)*m.heads*l.canonical_bytes && m.stage_all && m.stage_n >= m.records);
    GGML_ASSERT(m.slots && m.slots <= m.records);
    GGML_ASSERT(int(m.strip_master_bits >> 8) == ggml_get_op_params_i32(body, 2) && int(m.strip_master_bits & 255) == ggml_get_op_params_i32(body, 3));
    GGML_ASSERT(body->src[0] && body->src[0]->ne[0] == int64_t(m.heads)*256);
    // The pack and refresh share the main stream; a side-stream copy must finish before either writes a slot.
    lsa_pf_join(ctx.stream(), e);
    GGML_ASSERT(ggml_cuda_sj_kvarn_seal_strip_master(ctx, body, const_cast<char *>(m.strip_master),
        size_t(m.records)*m.heads, m.strip_master_capacity));
    ggml_cuda_qsa_clock_refresh(ctx, body);
    return true;
}

static __global__ void qsa_clock_resolve(ggml_cuda_qsa_clock_map m, const int32_t * desc,
        const int32_t * ids, int width, int queries, const char * mask, size_t mask_pitch, int mask_n) {
    __shared__ uint32_t epoch, assigned, hand;
    if (threadIdx.x == 0) {
        epoch = m.ctl[0] + 1;
        m.ctl[0] = epoch;
        m.ctl[2] = 0;
        m.ctl[3] = 1;
        assigned = 0;
        hand = m.ctl[1];
    }
    __syncthreads();
    if (epoch == 0) {
        for (int i = threadIdx.x; i < m.slots; i += blockDim.x) m.stamp[i] = ~0u;
        __syncthreads();
    }
    const int S = desc[GGML_SJKVARN_DESC_S], B = desc[GGML_SJKVARN_DESC_B];
    for (int i = threadIdx.x; i < width*queries; i += blockDim.x) {
        const int p = ids[i];
        if (p < S || p >= B || p >= mask_n ||
                __half2float(reinterpret_cast<const half *>(mask + (i/width)*mask_pitch)[p]) == -INFINITY) continue;
        const int g = (p - S)/m.group;
        const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
        const int page = record*(m.group/4) + ((p - S)%m.group)/4;
        if (record < 0 || record >= m.records) { atomicExch(m.ctl + 4, 1u); continue; }
        const int slot = atomicCAS(m.table + page, -1, -2);
        if (slot >= 0) {
            atomicExch(m.stamp + slot, epoch);
            atomicExch(m.ref + slot, 1);
        } else if (slot == -1) {
            const uint32_t miss = atomicAdd(m.ctl + 2, 1u);
            if (miss < m.slots) m.miss_page[miss] = page;
        }
    }
    __syncthreads();
    if (m.ctl[2] > m.slots || m.ctl[4]) {
        if (threadIdx.x == 0) { printf("QSA CLOCK: selection exceeds resident capacity or record bounds\n"); __trap(); }
        return;
    }
    const uint32_t lanes = min(uint32_t(blockDim.x), m.slots);
    uint32_t scanned = 0;
    // Each sweep visits disjoint slots. A second visit consumes the CLOCK reference bit.
    while (assigned < m.ctl[2] && scanned < 3*m.slots + lanes) {
        const uint32_t slot = (hand + scanned + threadIdx.x)%m.slots;
        if (threadIdx.x < lanes && m.stamp[slot] != epoch && atomicExch(m.ref + slot, 0) == 0) {
            const uint32_t item = atomicAdd(&assigned, 1u);
            if (item < m.ctl[2]) {
                const int old = m.tags[slot];
                if (old >= 0) m.table[old] = -1;
                const int page = m.miss_page[item];
                m.tags[slot] = page;
                m.stamp[slot] = epoch;
                m.ref[slot] = 1;
                m.miss_slot[item] = slot;
                m.table[page] = slot;
            }
        }
        scanned += lanes;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (assigned < m.ctl[2]) {
            m.ctl[4] = 1;
            printf("QSA CLOCK: all slots are protected by this selection\n");
            __trap();
        }
        m.ctl[1] = (hand + scanned)%m.slots;
        reinterpret_cast<unsigned long long *>(m.ctl + 8)[0] += m.ctl[2];
        reinterpret_cast<unsigned long long *>(m.ctl + 10)[0] += uint64_t(width)*queries;
        reinterpret_cast<unsigned long long *>(m.ctl + 12)[0] += 1;
    }
}

static __global__ void qsa_clock_copy_misses(ggml_cuda_qsa_clock_map m) {
    for (uint32_t i = blockIdx.x; i < m.ctl[2]; i += gridDim.x) {
        qsa_clock_copy_page(m, m.miss_page[i], m.miss_slot[i], blockIdx.y);
    }
}

void ggml_cuda_qsa_clock_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!dst->src[5] || !dst->src[6]) return;
    auto * e = qsa_clock_entry(dst->src[5]);
    if (!e || e->map.cells != GGML_QSA_PAGE_CELLS) return; // LSA strips/records resolve elsewhere (ggml_cuda_qsa_strip_prepare, ggml_cuda_lsa_page_prepare)
    const auto * mask = dst->src[3];
    const auto * ids = dst->src[7];
    if (dst->src[0]->ne[1] > 8 || !ids) {
        // Prefill and non-indexed attention use the existing host-backed expansion path.
        CUDA_CHECK(cudaMemsetAsync(e->map.ctl + 3, 0, sizeof(uint32_t), ctx.stream()));
        return;
    }
    GGML_ASSERT(mask && mask->type == GGML_TYPE_F16 && mask->ne[2] == 1 && mask->ne[3] == 1);
    GGML_ASSERT(ids && ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids));
    GGML_ASSERT(ids->ne[1] == dst->src[0]->ne[1] && ids->ne[2] == 1 && ids->ne[3] == 1);
    qsa_clock_resolve<<<1, 1024, 0, ctx.stream()>>>(e->map,
        static_cast<const int32_t *>(dst->src[6]->data), static_cast<const int32_t *>(ids->data),
        ids->ne[0], ids->ne[1], static_cast<const char *>(mask->data), mask->nb[1], mask->ne[0]);
    CUDA_CHECK(cudaGetLastError());
    qsa_clock_copy_misses<<<dim3(128, e->map.heads), 128, 0, ctx.stream()>>>(e->map);
    CUDA_CHECK(cudaGetLastError());
}

// CLOCK victim rule of the LSA strip resolve and the record-page assign. protect 1 (default): any slot this resolve did not
// stamp, as before. protect n > 1: an empty slot, or one no resolve of the last n stamped. The map is shared by the streams
// of a layer and every stream's resolve is a new epoch, so n = active streams keeps the strips the other streams picked
// this round. A miss that finds no victim is read from the master, so residency never changes results.
static __device__ __forceinline__ bool lsa_victim_ok(const ggml_cuda_qsa_clock_map & m, uint32_t slot, uint32_t epoch) {
    return m.protect <= 1 ? m.stamp[slot] != epoch : m.tags[slot] < 0 || epoch - m.stamp[slot] >= m.protect;
}

// protect > 1: true (block-uniform) when a full CLOCK lap ended with no slot past the protect window, so the scan
// stops instead of running its 3-lap bound; the misses left read the master. Call after the loop's __syncthreads.
static __device__ __forceinline__ bool lsa_lap_dry(const ggml_cuda_qsa_clock_map & m, uint32_t scanned, uint32_t lanes,
        uint32_t & cand) {
    if (m.protect <= 1 || scanned/m.slots == (scanned - lanes)/m.slots) return false;
    const bool dry = cand == 0;
    __syncthreads();
    if (threadIdx.x == 0) cand = 0;
    __syncthreads();
    return dry;
}

// ---------------------------------------------------------------------------------------------------------------------
// LLAMA_LSA_RESIDENT (opt-in): CLOCK over 16-cell strip slots, driven by the live-record list of the SJ-KVaRN stream
// kernel (the strips it is about to read). Hits are stamped with this resolve's epoch (never evicted by it), misses
// take CLOCK victims and are copied from the pinned master. Unlike the QSA resolve it never traps: picked strips past
// the free slots keep table entry -1 and the kernel reads them from the master over the bus (counted, ctl 14).
static __global__ void qsa_strip_resolve(ggml_cuda_qsa_clock_map m, const int32_t * desc, const int32_t * list) {
    __shared__ uint32_t epoch, assigned, hand, nmiss, picked, cand;
    if (threadIdx.x == 0) {
        cand = 0;
        epoch = m.ctl[0] + 1;
        m.ctl[0] = epoch;
        m.ctl[2] = 0;
        assigned = 0;
        picked = 0;
        hand = m.ctl[1];
    }
    __syncthreads();
    if (epoch == 0) {
        for (int i = threadIdx.x; i < m.slots; i += blockDim.x) m.stamp[i] = ~0u;
        __syncthreads();
    }
    constexpr int P = 128/GGML_LSA_STRIP_CELLS;
    const int n_live = list[0];
    uint32_t mine = 0;
    for (int o = threadIdx.x; o < n_live; o += blockDim.x) {
        const uint32_t e = uint32_t(list[1 + o]);
        const int g = int(e & 0xFFFFFFu), sm = int(e >> 24);
        const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
        if (record < 0 || record >= int(m.records)) continue;
        for (int sl = 0; sl < P; ++sl) {
            if (!((sm >> sl) & 1)) continue;
            ++mine;
            const int page = record*P + sl;
            const int slot = atomicCAS(m.table + page, -1, -2);
            if (slot >= 0) {
                atomicExch(m.stamp + slot, epoch);
                atomicExch(m.ref + slot, 1);
            } else if (slot == -1) {
                const uint32_t miss = atomicAdd(m.ctl + 2, 1u);
                if (miss < m.slots) {
                    m.miss_page[miss] = page;
                } else {
                    m.table[page] = -1; // past the slots: read from the master
                }
            }
        }
    }
    atomicAdd(&picked, mine);
    __syncthreads();
    if (threadIdx.x == 0) nmiss = min(m.ctl[2], m.slots);
    __syncthreads();
    const uint32_t lanes = min(uint32_t(blockDim.x), m.slots);
    uint32_t scanned = 0;
    while (assigned < nmiss && scanned < 3*m.slots + lanes) {
        const uint32_t slot = (hand + scanned + threadIdx.x)%m.slots;
        const bool ok = threadIdx.x < lanes && lsa_victim_ok(m, slot, epoch);
        if (ok && m.protect > 1) cand = 1;
        if (ok && atomicExch(m.ref + slot, 0) == 0) {
            const uint32_t item = atomicAdd(&assigned, 1u);
            if (item < nmiss) {
                const int old = m.tags[slot];
                if (old >= 0) m.table[old] = -1;
                const int page = m.miss_page[item];
                m.tags[slot] = page;
                m.stamp[slot] = epoch;
                m.ref[slot] = 1;
                m.miss_slot[item] = slot;
                m.table[page] = slot;
            }
        }
        scanned += lanes;
        __syncthreads();
        if (lsa_lap_dry(m, scanned, lanes, cand)) break;
    }
    const uint32_t got = min(assigned, nmiss);
    for (uint32_t i = got + threadIdx.x; i < nmiss; i += blockDim.x) {
        m.table[m.miss_page[i]] = -1; // every slot is protected by this pick: read from the master
    }
    if (threadIdx.x == 0) {
        const uint32_t over = m.ctl[2] - got;
        m.ctl[2] = got; // the copy kernel copies the assigned misses only
        m.ctl[1] = (hand + scanned)%m.slots;
        auto * c = reinterpret_cast<unsigned long long *>(m.ctl + 8);
        c[0] += got;
        c[1] += picked;
        c[2] += 1;
        c[3] += over;
        // LLAMA_LSA_J3_LEAD (PROTECT_PREV): the own picks since the last strip prefetch pass of this layer (every stream's
        // resolve of the step) keep their slots in the next pass: ctl 16 = epoch of the first such resolve, 24 = picked
        if (m.ctl[7] == 0) {
            m.ctl[7] = 1;
            m.ctl[16] = epoch;
            m.ctl[24] = 0;
        }
        m.ctl[24] += picked;
        if (m.log_every && c[2] % m.log_every == 0) {
            printf("LSA RESIDENT: layer %d resolves=%llu picked=%llu fetched=%llu over_slots=%llu slots=%u strip_bytes=%u "
                "prefetched=%u pf_passes=%u pf_hits=%u pf_dropped=%u\n", m.layer, c[2], c[1], c[0], c[3], m.slots,
                m.heads*GGML_LSA_STRIP_HEAD_BYTES, m.ctl[5], m.ctl[6], m.ctl[17], m.ctl[18]);
        }
    }
}

// count: ctl word with the copy count (2 after qsa_strip_resolve, 4 after lsa_page_assign)
static __global__ void qsa_strip_copy_misses(ggml_cuda_qsa_clock_map m, int count = 2) {
    for (uint32_t i = blockIdx.x; i < m.ctl[count]; i += gridDim.x) {
        qsa_strip_copy(m, m.miss_page[i], m.miss_slot[i], blockIdx.y);
    }
}

void ggml_cuda_qsa_strip_join(ggml_backend_cuda_context & ctx, const ggml_tensor * body) {
    auto * e = body ? qsa_clock_entry(body) : nullptr;
    if (e && e->map.cells == GGML_LSA_STRIP_CELLS) lsa_pf_join(ctx.stream(), e);
}

bool ggml_cuda_qsa_strip_body(const ggml_tensor * body) {
    if (!body) return false;
    const auto * e = qsa_clock_entry(body);
    return e && e->map.cells == GGML_LSA_STRIP_CELLS;
}

void ggml_cuda_qsa_strip_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * body, const int32_t * desc,
        const int32_t * list) {
    auto * e = body ? qsa_clock_entry(body) : nullptr;
    if (!e || e->map.cells != GGML_LSA_STRIP_CELLS || !desc || !list) return;
    lsa_pf_join(ctx.stream(), e); // LLAMA_LSA_J3_LEAD: the strip prefetch for this layer lands before its resolve
    qsa_strip_resolve<<<1, 1024, 0, ctx.stream()>>>(e->map, desc, list);
    CUDA_CHECK(cudaGetLastError());
    qsa_strip_copy_misses<<<dim3(128, e->map.heads), 128, 0, ctx.stream()>>>(e->map);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// LLAMA_LSA_RESIDENT on low-bit bodies (opt-in): CLOCK over whole 128-cell record slots. Three launches per decode
// attention: mark (one warp per body group: does any query row's f16 mask keep a cell of it; hits are stamped with this
// resolve's epoch, misses queued), assign (one block: CLOCK victims for the misses, counters) and copy (misses from the
// pinned master). A record without a slot is read from the master by the kernel, so the result never depends on
// residency. ctl: 0 epoch, 1 hand, 2 queued misses, 3 picked this resolve, 4 assigned (copy count), u64 at 8.. counters.
// limit: most misses queued (m.slots normally). pf: prefetch for the next LSA layer's map m from this layer's mask:
// picked is not counted and records sealed by this ubatch are skipped (their master bytes are being written now).
static __global__ void lsa_page_mark(ggml_cuda_qsa_clock_map m, const int32_t * desc, const half * mask, int64_t pitch,
        int nrows, uint32_t limit, int pf) {
    const int g = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int S = desc[GGML_SJKVARN_DESC_S], B = desc[GGML_SJKVARN_DESC_B], G = desc[GGML_SJKVARN_DESC_G];
    if (G != GGML_LSA_PAGE_CELLS || g >= (B - S)/G) return;
    if (pf && g >= (desc[GGML_SJKVARN_DESC_B_OLD] - S)/G) return;
    bool live = mask == nullptr; // no usable mask: every sealed record is read
    uint32_t strips = mask == nullptr ? 255 : 0;
    for (int r = 0; r < nrows && (!live || m.stage_mask); ++r) { // strip copy and diagnostics need the full union
        const half * row = mask + r*pitch + S + (int64_t) g*G;
        bool any = false;
        for (int i = lane; i < G; i += WARP_SIZE) any |= __half2float(row[i]) != -INFINITY;
        live = __any_sync(0xffffffff, any);
        if (m.stage_mask) {
            for (int s = 0; s < 8; ++s) {
                if (__any_sync(0xffffffff, lane < 16 && __half2float(row[s*16 + lane]) != -INFINITY)) strips |= 1u << s;
            }
        }
    }
    if (lane != 0) return;
    const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
    if (record < 0 || record >= int(m.records)) return;
    if (m.ahead_list && !pf && g >= (desc[GGML_SJKVARN_DESC_B_OLD] - S)/G && m.table[record] <= -3) {
        m.table[record] = -1; // GGML_CUDA_LSA_STAGE_AHEAD: this ubatch seals the record, the ahead copy may be stale
    }
    if (m.stage_mask) {
        m.stage_mask[record] = strips;
        live = strips != 0;
    }
    if (!live) return;
    const uint32_t epoch = m.ctl[0] + 1; // epochs start at 1 (stamps reset to 0); 2^32 resolves never wrap in practice
    if (!pf) atomicAdd(m.ctl + 3, 1u);
    if (m.prev && !pf) {
        // LLAMA_LSA_J3 previous-pick source: this own resolve's number (the assign below counts it)
        m.prev[record] = uint32_t(reinterpret_cast<const unsigned long long *>(m.ctl + 8)[2]) + 1;
    }
    const uint32_t j3_bits = m.j3_src && !pf && (m.j3_src[record] >> 3) ==
        uint32_t(reinterpret_cast<const unsigned long long *>(m.ctl + 8)[2]) + 1 ? m.j3_src[record] & 7u : 0u;
    if (m.j3_src && !pf) {
        // LLAMA_LSA_J3 cascade: each picked record counted once, by the first source that marked it (ahead-3, then the
        // ahead-1 top-up, then the previous pick), else unpredicted; the four add up to picked
        atomicAdd(m.ctl + (j3_bits & 1u ? 19 : j3_bits & 2u ? 20 : j3_bits & 4u ? 21 : 22), 1u);
    }
    bool admit = true;
    if (m.seen && !pf) {
        // own resolves only (prefetch resolves also advance the epoch): step n admits pages step n-1 picked
        const uint32_t step = uint32_t(reinterpret_cast<const unsigned long long *>(m.ctl + 8)[2]) + 1;
        admit = m.seen[record] + 1 == step;
        m.seen[record] = step;
    }
    const int slot = atomicCAS(m.table + record, -1, -2);
    if (slot == -1 && j3_bits) {
        atomicAdd(m.ctl + 23, 1u); // predicted, but no slot at the own resolve (dropped, evicted or never copied in)
    }
    if (slot >= 0) {
        m.stamp[slot] = epoch;
        m.ref[slot] = 1;
    } else if (slot == -1 && !admit) {
        lsa_stage_or_master(m, record); // first pick in a row: no slot (stage, else the master)
        atomicAdd(m.ctl + 7, 1u);
    } else if (slot == -1) {
        const uint32_t miss = atomicAdd(m.ctl + 2, 1u);
        if (miss < limit) {
            m.miss_page[miss] = record;
        } else if (pf) {
            m.table[record] = -1; // prefetch past its cap: not picked by anyone yet
        } else {
            lsa_stage_or_master(m, record); // past the slots: stage, else the master
        }
    } else if (slot <= -3 && !pf && m.ahead_list) {
        // GGML_CUDA_LSA_STAGE_AHEAD: picked again, read in place from its ahead entry and offered to the next list
        const uint32_t a = atomicAdd(m.stage_ctl + 7, 1u);
        if (a < m.stage_n) m.ahead_list[a] = record;
        atomicAdd(reinterpret_cast<unsigned long long *>(m.stage_ctl + 8), 1ull);
    }
}

// pf: 0 own resolve, 1 GGML_CUDA_LSA_PREFETCH, 2/3 LLAMA_LSA_J3 (2: the previous own pick's slots are protected too).
// A J3 prefetch never evicts a slot stamped after the target's last own resolve (an earlier source of the same step
// put or found it there), so the sources of one step add up instead of evicting each other.
static __global__ void lsa_page_assign(ggml_cuda_qsa_clock_map m, uint32_t limit, int pf, bool stage_all = false) {
    __shared__ uint32_t epoch, assigned, hand, nmiss, keep_from, cand;
    if (threadIdx.x == 0) {
        cand = 0;
        epoch = m.ctl[0] + 1;
        assigned = 0;
        hand = m.ctl[1];
        nmiss = min(m.ctl[2], limit);
        // pf 2 protects the previous own pick only while it is smaller than the slot pool: at a share where one
        // step's pick fills every slot, protecting it would freeze the pool (r390: j35/j20 copied one pool, then 0)
        keep_from = pf == 2 ? (m.ctl[24] < m.slots ? m.ctl[16] : m.ctl[16] + 1) : pf == 3 ? m.ctl[16] + 1 : epoch;
    }
    __syncthreads();
    const uint32_t lanes = min(uint32_t(blockDim.x), m.slots);
    uint32_t scanned = 0;
    while (assigned < nmiss && scanned < 3*m.slots + lanes) {
        const uint32_t slot = (hand + scanned + threadIdx.x)%m.slots;
        const bool free_slot = pf >= 2 ? m.tags[slot] < 0 || m.stamp[slot] < keep_from : lsa_victim_ok(m, slot, epoch);
        if (threadIdx.x < lanes && (free_slot || pf >= 2) && m.protect > 1) cand = 1; // pf >= 2 keeps its own rule: never dry
        if (threadIdx.x < lanes && free_slot && atomicExch(m.ref + slot, 0) == 0) {
            const uint32_t item = atomicAdd(&assigned, 1u);
            if (item < nmiss) {
                const int old = m.tags[slot];
                if (old >= 0) m.table[old] = -1;
                const int record = m.miss_page[item];
                m.tags[slot] = record;
                m.stamp[slot] = epoch;
                m.ref[slot] = 1;
                m.miss_slot[item] = slot;
                m.table[record] = slot;
            }
        }
        scanned += lanes;
        __syncthreads();
        if (lsa_lap_dry(m, scanned, lanes, cand)) break;
    }
    const uint32_t got = min(assigned, nmiss);
    for (uint32_t i = got + threadIdx.x; i < nmiss; i += blockDim.x) {
        if (pf) m.table[m.miss_page[i]] = -1; else lsa_stage_or_master(m, m.miss_page[i]); // every slot protected by this pick
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t clock_over = m.ctl[2] - got;
        const uint32_t over = stage_all ? m.stage_ctl[0] - min(m.stage_ctl[0], m.stage_n) : clock_over;
        if (stage_all) {
            assert(m.stage_n >= m.records && over == 0);
        }
        auto * c = reinterpret_cast<unsigned long long *>(m.ctl + 8);
        if (pf) {
            // prefetched slots keep ref 1 (one CLOCK pass of grace) but not the next resolve's epoch
            m.ctl[5] += got;
            m.ctl[6] += 1;
        } else {
            c[0] += got;
            c[1] += m.ctl[3];
            c[2] += 1;
            c[3] += over;
            m.ctl[16] = epoch; // J3: slots stamped after this belong to the next own resolve's prefetch
            m.ctl[24] = m.ctl[3]; // J3: records this own resolve picked
        }
        m.ctl[0] = epoch;
        m.ctl[1] = (hand + scanned)%m.slots;
        m.ctl[2] = 0;
        m.ctl[3] = 0;
        m.ctl[4] = got;
        if (!pf && m.log_every && c[2] % m.log_every == 0) {
            if (stage_all) {
                printf("LSA STAGE: layer %d resolves=%llu requested=%u capacity=%u clock_over_slots=%u host_records=%u\n",
                    m.layer, c[2], m.stage_ctl[0], m.stage_n, clock_over, over);
            }
            if (m.ctl[6]) {
                printf("LSA RESIDENT: layer %d resolves=%llu picked=%llu fetched=%llu over_slots=%llu slots=%u record_bytes=%llu prefetched=%u pf_resolves=%u pf_hits=%u pf_dropped=%u j3_a3=%u j3_a1_add=%u j3_prev_add=%u j3_unpredicted=%u j3_lost=%u\n",
                    m.layer, c[2], c[1], c[0], c[3], m.slots, (unsigned long long) m.stride, m.ctl[5], m.ctl[6], m.ctl[17], m.ctl[18],
                    m.ctl[19], m.ctl[20], m.ctl[21], m.ctl[22], m.ctl[23]);
            } else if (m.seen) {
                printf("LSA RESIDENT: layer %d resolves=%llu picked=%llu fetched=%llu over_slots=%llu slots=%u record_bytes=%llu not_admitted=%u\n",
                    m.layer, c[2], c[1], c[0], c[3], m.slots, (unsigned long long) m.stride, m.ctl[7]);
            } else {
                printf("LSA RESIDENT: layer %d resolves=%llu picked=%llu fetched=%llu over_slots=%llu slots=%u record_bytes=%llu\n",
                    m.layer, c[2], c[1], c[0], c[3], m.slots, (unsigned long long) m.stride);
            }
        }
    }
}

static __global__ void lsa_page_copy_misses(ggml_cuda_qsa_clock_map m) {
    for (uint32_t i = blockIdx.x; i < m.ctl[4]; i += gridDim.x) {
        lsa_page_copy(m, m.miss_page[i], m.miss_slot[i], blockIdx.y);
    }
}

const char * ggml_cuda_lsa_page_body(const ggml_tensor * body) {
    const auto * e = body ? qsa_clock_entry(body) : nullptr;
    if (!e || e->map.cells != GGML_LSA_PAGE_CELLS) {
        return body ? static_cast<const char *>(body->data) : nullptr;
    }
    return ggml_cuda_qsa_clock_body(body);
}

const int32_t * ggml_cuda_lsa_page_table(const ggml_tensor * body, uint32_t * records) {
    const auto * e = body ? qsa_clock_entry(body) : nullptr;
    if (!e || e->map.cells != GGML_LSA_PAGE_CELLS) {
        return nullptr;
    }
    *records = e->map.records;
    return e->map.table;
}

// GGML_CUDA_LSA_PREFETCH: mark the next LSA layer's misses from this layer's mask on the main stream (the mask buffer
// may be reused once this node is done), then assign + copy them on a low-priority side stream that overlaps the
// layers in between; the next layer joins it (lsa_pf_join) before its seal refresh or its own resolve.
static void lsa_page_prefetch(ggml_backend_cuda_context & ctx, qsa_clock_tensor * e, const int32_t * desc,
        const half * m16, int64_t pitch, int nrows) {
    qsa_clock_tensor * next = nullptr;
    {
        std::lock_guard<std::mutex> lock(lsa_pf_mutex);
        for (auto * c : lsa_pf_entries) {
            if (c != e && c->device == e->device && c->map.cells == GGML_LSA_PAGE_CELLS && c->map.layer > e->map.layer &&
                    c->map.records == e->map.records &&
                    (!next || c->map.layer < next->map.layer)) {
                next = c;
            }
        }
    }
    if (!next || next->pf_pending || !m16) return;
    const uint32_t cap = std::min<uint32_t>(next->map.slots, uint32_t(lsa_prefetch_cap()));
    const int per_block = 8;
    lsa_page_mark<<<(next->map.records + per_block - 1)/per_block, per_block*WARP_SIZE, 0, ctx.stream()>>>(
        next->map, desc, m16, pitch, nrows, cap, 1);
    CUDA_CHECK(cudaGetLastError());
    cudaStream_t side = lsa_pf_stream[e->device];
    CUDA_CHECK(cudaEventRecord(lsa_pf_fork[e->device], ctx.stream()));
    CUDA_CHECK(cudaStreamWaitEvent(side, lsa_pf_fork[e->device], 0));
    lsa_page_assign<<<1, 1024, 0, side>>>(next->map, cap, 1);
    CUDA_CHECK(cudaGetLastError());
    lsa_page_copy_misses<<<dim3(16, next->map.heads), 256, 0, side>>>(next->map);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(next->pf_done, side));
    next->pf_pending = true;
}

// LLAMA_LSA_J3: one thread per sealed body group of the target layer. A record is marked if a unit of the prediction
// that overlaps it is set in any row (units: `unit` cells of the cell axis, the decode mask's columns) or, with prev,
// if the target's last own resolve picked it. Records sealed by this ubatch are skipped (their master bytes are being
// written now). A record's stage entry from the last own resolve (-3 - k) is stale for the next one: it may be claimed
// here (lsa_stage_reset only clears entries that still hold its own value). Resident hits get this pass's epoch, so
// later passes of the step keep them. Runs on the side stream, between two own resolves of the target.
static __global__ void lsa_j3_mark(ggml_cuda_qsa_clock_map m, const int32_t * desc, const float * units, int64_t nunits,
        int nrows, int unit, int use_prev, uint32_t limit, uint32_t unit_bit) {
    const int g = blockIdx.x*blockDim.x + threadIdx.x;
    const int S = desc[GGML_SJKVARN_DESC_S], G = desc[GGML_SJKVARN_DESC_G];
    if (G != GGML_LSA_PAGE_CELLS || g >= (desc[GGML_SJKVARN_DESC_B_OLD] - S)/G) return;
    bool live = false;
    if (units) {
        const int64_t c0 = int64_t(S) + int64_t(g)*G;
        const int64_t u0 = c0/unit, u1 = min(nunits - 1, (c0 + G - 1)/unit);
        for (int r = 0; r < nrows && !live; ++r) {
            for (int64_t u = u0; u <= u1 && !live; ++u) live = units[r*nunits + u] > 0.5f;
        }
    }
    const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
    if (record < 0 || record >= int(m.records)) return;
    const uint32_t own = uint32_t(reinterpret_cast<const unsigned long long *>(m.ctl + 8)[2]);
    const bool prev_live = use_prev && m.prev && own != 0 && m.prev[record] == own;
    if (!live && !prev_live) return;
    if (m.j3_src) {
        // tag the sources for the target's next own resolve (number own + 1); a stale tag is replaced
        const uint32_t bits = (live ? unit_bit : 0u) | (prev_live ? 4u : 0u);
        uint32_t t = m.j3_src[record];
        for (;;) {
            const uint32_t want = ((t >> 3) == own + 1 ? t : (own + 1) << 3) | bits;
            if (want == t) break;
            const uint32_t seen = atomicCAS(m.j3_src + record, t, want);
            if (seen == t) break;
            t = seen;
        }
    }
    const uint32_t epoch = m.ctl[0] + 1;
    int cur = m.table[record];
    for (;;) {
        if (cur >= 0) {
            m.stamp[cur] = epoch;
            m.ref[cur] = 1;
            atomicAdd(m.ctl + 17, 1u);
            return;
        }
        if (cur == -2) return; // queued by this pass already
        const int old = atomicCAS(m.table + record, cur, -2);
        if (old == cur) break;
        cur = old;
    }
    const uint32_t miss = atomicAdd(m.ctl + 2, 1u);
    if (miss < limit) {
        m.miss_page[miss] = record;
    } else {
        m.table[record] = -1;
        atomicAdd(m.ctl + 18, 1u);
    }
}

// LLAMA_LSA_J3_LEAD on 16-cell strip slots: one thread per (sealed body group, strip) of the target layer. A strip is
// marked if a unit of the prediction that overlaps its 16 cells is set in any row. Records sealed by this ubatch are
// skipped (their master bytes are being written now). Hits get the pass's epoch and ref; misses are queued for
// lsa_page_assign (pf 1: table index = strip page) and qsa_strip_copy_misses on the side stream. The previous pick is
// not used: the strips the last own resolve picked already hold slots.
static __global__ void lsa_j3_strip_mark(ggml_cuda_qsa_clock_map m, const int32_t * desc, const float * units,
        int64_t nunits, int nrows, int unit, uint32_t limit) {
    constexpr int P = GGML_LSA_PAGE_CELLS/GGML_LSA_STRIP_CELLS;
    const int i = blockIdx.x*blockDim.x + threadIdx.x, g = i/P, sl = i%P;
    const int S = desc[GGML_SJKVARN_DESC_S], G = desc[GGML_SJKVARN_DESC_G];
    if (G != GGML_LSA_PAGE_CELLS || g >= (desc[GGML_SJKVARN_DESC_B_OLD] - S)/G) return;
    const int64_t c0 = int64_t(S) + int64_t(g)*G + sl*GGML_LSA_STRIP_CELLS;
    const int64_t u0 = c0/unit, u1 = min(nunits - 1, (c0 + GGML_LSA_STRIP_CELLS - 1)/unit);
    bool live = false;
    for (int r = 0; r < nrows && !live; ++r) {
        for (int64_t u = u0; u <= u1 && !live; ++u) live = units[r*nunits + u] > 0.5f;
    }
    const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
    if (!live || record < 0 || record >= int(m.records)) return;
    const int page = record*P + sl;
    const uint32_t epoch = m.ctl[0] + 1;
    int cur = m.table[page];
    for (;;) {
        if (cur >= 0) {
            m.stamp[cur] = epoch;
            m.ref[cur] = 1;
            atomicAdd(m.ctl + 17, 1u);
            return;
        }
        if (cur == -2) return; // queued by this pass already
        const int old = atomicCAS(m.table + page, cur, -2);
        if (old == cur) break;
        cur = old;
    }
    const uint32_t miss = atomicAdd(m.ctl + 2, 1u);
    if (miss < limit) {
        m.miss_page[miss] = page;
    } else {
        m.table[page] = -1;
        atomicAdd(m.ctl + 18, 1u);
    }
}

// grow a J3 scratch buffer; cudaFree waits for the device, so an old copy still read by the side stream is safe
template <typename T>
static void lsa_j3_reserve(T *& p, size_t & cap, size_t n) {
    if (n <= cap) return;
    if (p) CUDA_CHECK(cudaFree(p));
    cap = ((n + 65535)/65536)*65536;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&p), cap*sizeof(T)));
    lsa_j3_gen.fetch_add(1, std::memory_order_relaxed); // captured graphs hold the old pointer
}

// a J3 target: record-page slots, or (LLAMA_LSA_J3_LEAD) 16-cell strip slots; never the unpublished strip-master source
static bool lsa_j3_body(const qsa_clock_tensor * e) {
    return (e->map.cells == GGML_LSA_PAGE_CELLS || e->map.cells == GGML_LSA_STRIP_CELLS) && !e->map.strip_master;
}

// Host setup of one J3 pass that never runs during capture: side stream, events, prev/j3_src, scratch. Returns the
// entry, or nullptr when the op is a no-op for this body (same rule as ggml_cuda_lsa_prefetch).
static qsa_clock_tensor * lsa_j3_setup(const ggml_tensor * dst, int device, cudaStream_t stream) {
    const ggml_tensor * desc = dst->src[0];
    const ggml_tensor * units = dst->src[2];
    auto * e = dst->src[1] ? qsa_clock_entry(dst->src[1]) : nullptr;
    if (!e || !lsa_j3_body(e) || e->device != device) {
        return nullptr;
    }
    const bool strip = e->map.cells == GGML_LSA_STRIP_CELLS;
    {
        std::lock_guard<std::mutex> lock(lsa_pf_mutex);
        if (strip && std::find(lsa_pf_entries.begin(), lsa_pf_entries.end(), e) == lsa_pf_entries.end()) {
            lsa_pf_entries.push_back(e); // joined by ggml_cuda_lsa_j3_graph_end; the next-layer lookup takes pages only
        }
        if (!e->pf_done) {
            CUDA_CHECK(cudaEventCreateWithFlags(&e->pf_done, cudaEventDisableTiming));
        }
        if (!lsa_pf_stream[e->device]) {
            int lo = 0, hi = 0;
            CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
            CUDA_CHECK(cudaStreamCreateWithPriority(&lsa_pf_stream[e->device], cudaStreamNonBlocking, lo));
            CUDA_CHECK(cudaEventCreateWithFlags(&lsa_pf_fork[e->device], cudaEventDisableTiming));
        }
    }
    if (strip && !units) {
        return e; // strips use the units only
    }
    if (!strip && !e->map.prev) {
        // the own mark fills both from the next own resolve on (prev: the previous-pick source; j3_src: the cascade
        // log), so this first call marks no previous pick
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&e->map.prev), size_t(e->map.records)*sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&e->map.j3_src), size_t(e->map.records)*sizeof(uint32_t)));
        if (stream) {
            CUDA_CHECK(cudaMemsetAsync(e->map.prev, 0, size_t(e->map.records)*sizeof(uint32_t), stream));
            CUDA_CHECK(cudaMemsetAsync(e->map.j3_src, 0, size_t(e->map.records)*sizeof(uint32_t), stream));
        } else {
            CUDA_CHECK(cudaMemset(e->map.prev, 0, size_t(e->map.records)*sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset(e->map.j3_src, 0, size_t(e->map.records)*sizeof(uint32_t)));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        lsa_j3_gen.fetch_add(1, std::memory_order_relaxed); // own-resolve graphs captured before hold map.prev == nullptr
    }
    const int src = ggml_get_op_params_i32(dst, 2);
    GGML_ASSERT(src >= 0 && src < GGML_LSA_PREFETCH_SOURCES);
    if (!units && !(ggml_get_op_params_i32(dst, 1) & GGML_LSA_PREFETCH_PREV)) {
        return e;
    }
    if (e->j3_issued[src] && stream) {
        lsa_pf_join(stream, e); // same source twice before the target ran: its scratch may still be read
    }
    lsa_j3_reserve(e->j3_desc[src], e->j3_desc_cap[src], size_t(ggml_nelements(desc)));
    if (units) {
        lsa_j3_reserve(e->j3_units[src], e->j3_units_cap[src], size_t(ggml_nelements(units)));
    }
    return e;
}

void ggml_cuda_lsa_prefetch_reserve(const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_LSA_PREFETCH);
    if (dst->src[1] && dst->src[0]) {
        // cudaFree in a scratch growth waits for the device, so a pass still reading the old copy is safe
        lsa_j3_setup(dst, ggml_cuda_get_device(), nullptr);
    }
}

void ggml_cuda_lsa_j3_graph_end(ggml_backend_cuda_context & ctx) {
    // a stream capture must rejoin every forked stream before it ends: join any pass whose target did not run
    std::lock_guard<std::mutex> lock(lsa_pf_mutex);
    for (auto * e : lsa_pf_entries) {
        if (e->device == ctx.device) {
            lsa_pf_join(ctx.stream(), e);
        }
    }
}

void ggml_cuda_lsa_prefetch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    lsa_ahead_blocked = true; // GGML_CUDA_LSA_STAGE_AHEAD: off for the process (J3 marks this layer's table on its own side stream)
    CUDA_CHECK(cudaMemsetAsync(dst->data, 0, sizeof(float), ctx.stream())); // the result is a placeholder
    const ggml_tensor * desc = dst->src[0];
    const ggml_tensor * units = dst->src[2];
    const int unit = ggml_get_op_params_i32(dst, 0), flags = ggml_get_op_params_i32(dst, 1), src = ggml_get_op_params_i32(dst, 2);
    GGML_ASSERT(src >= 0 && src < GGML_LSA_PREFETCH_SOURCES);
    GGML_ASSERT(desc->type == GGML_TYPE_I32 && ggml_is_contiguous(desc));
    GGML_ASSERT(!units || (units->type == GGML_TYPE_F32 && ggml_is_contiguous(units) && unit >= 1));
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
    const bool capturing = capture != cudaStreamCaptureStatusNone;
    GGML_ASSERT(!capturing || ggml_cuda_lsa_j3_graphs()); // the CUDA graph checks refuse this op otherwise
    const bool use_prev = (flags & GGML_LSA_PREFETCH_PREV) != 0;
    qsa_clock_tensor * e = nullptr;
    if (capturing) {
        // ggml_cuda_lsa_prefetch_reserve ran from the graph check for this graph: no allocation, only the join
        e = dst->src[1] ? qsa_clock_entry(dst->src[1]) : nullptr;
        if (!e || !lsa_j3_body(e) || e->device != ctx.device) {
            return;
        }
        if (e->map.cells == GGML_LSA_STRIP_CELLS && !units) return;
        GGML_ASSERT(e->pf_done && lsa_pf_stream[e->device] && (e->map.cells == GGML_LSA_STRIP_CELLS || (e->map.prev && e->map.j3_src)));
        if (!units && !use_prev) return;
        if (e->j3_issued[src]) {
            lsa_pf_join(ctx.stream(), e);
        }
        GGML_ASSERT(size_t(ggml_nelements(desc)) <= e->j3_desc_cap[src]);
        GGML_ASSERT(!units || size_t(ggml_nelements(units)) <= e->j3_units_cap[src]);
    } else {
        e = lsa_j3_setup(dst, ctx.device, ctx.stream());
        if (!e) return; // no record-page or strip slots (the whole body is on the card), or the unpublished strip-master source
        if (!units && (!use_prev || e->map.cells == GGML_LSA_STRIP_CELLS)) return;
    }
    const size_t ndesc = size_t(ggml_nelements(desc));
    CUDA_CHECK(cudaMemcpyAsync(e->j3_desc[src], desc->data, ndesc*sizeof(int32_t), cudaMemcpyDeviceToDevice, ctx.stream()));
    int64_t nunits = 0;
    int nrows = 0;
    if (units) {
        nunits = units->ne[0];
        nrows = int(units->ne[1]);
        CUDA_CHECK(cudaMemcpyAsync(e->j3_units[src], units->data, ggml_nbytes(units), cudaMemcpyDeviceToDevice, ctx.stream()));
    }
    e->j3_issued[src] = true;
    cudaStream_t side = lsa_pf_stream[e->device];
    CUDA_CHECK(cudaEventRecord(lsa_pf_fork[e->device], ctx.stream()));
    CUDA_CHECK(cudaStreamWaitEvent(side, lsa_pf_fork[e->device], 0));
    const uint32_t cap = e->map.slots;
    if (e->map.cells == GGML_LSA_STRIP_CELLS) {
        // LLAMA_LSA_J3_LEAD: the predicted strips of the target, joined by its ggml_cuda_qsa_strip_prepare
        const int n = int(e->map.records)*(GGML_LSA_PAGE_CELLS/GGML_LSA_STRIP_CELLS);
        // qsa_strip_resolve leaves its copy count in ctl 2: start the miss queue empty
        CUDA_CHECK(cudaMemsetAsync(e->map.ctl + 2, 0, sizeof(uint32_t), side));
        lsa_j3_strip_mark<<<(n + 127)/128, 128, 0, side>>>(e->map, e->j3_desc[src], e->j3_units[src], nunits, nrows, unit, cap);
        CUDA_CHECK(cudaGetLastError());
        // PROTECT_PREV: pf 2 keeps the slots of the own picks since the last pass (ctl 16/24, while they fit the pool)
        const bool protect = (flags & GGML_LSA_PREFETCH_PROTECT_PREV) != 0;
        lsa_page_assign<<<1, 1024, 0, side>>>(e->map, cap, protect ? 2 : 1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(e->map.ctl + 7, 0, sizeof(uint32_t), side)); // the next own resolve opens a new window
        qsa_strip_copy_misses<<<dim3(lsa_j3_copy_blocks(), e->map.heads), 128, 0, side>>>(e->map, 4);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e->pf_done, side));
        e->pf_pending = true;
        return;
    }
    lsa_j3_mark<<<(e->map.records + 127)/128, 128, 0, side>>>(e->map, e->j3_desc[src], units ? e->j3_units[src] : nullptr,
        nunits, nrows, unit, use_prev, cap, src % 2 == 0 ? 1u : 2u); // even src = look-ahead, odd = ahead-1 top-up
    CUDA_CHECK(cudaGetLastError());
    lsa_page_assign<<<1, 1024, 0, side>>>(e->map, cap, (flags & GGML_LSA_PREFETCH_PROTECT_PREV) ? 2 : 3);
    CUDA_CHECK(cudaGetLastError());
    lsa_page_copy_misses<<<dim3(lsa_j3_copy_blocks(), e->map.heads), 256, 0, side>>>(e->map);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e->pf_done, side));
    e->pf_pending = true;
}

// Logging-only: observe the table BEFORE mark/assign makes misses look resident. Single-block reduction keeps
// counters on the device; no readback, prefetch, eviction or slot-map mutation. This diagnostic is not timed.
static __global__ void lsa_page_spec_recall(ggml_cuda_qsa_clock_map m, const int32_t * desc,
        const half * mask, int64_t pitch, int nrows, const float * predicted, int nunits, int unit,
        uint32_t * history) {
    const int S = desc[GGML_SJKVARN_DESC_S], G = desc[GGML_SJKVARN_DESC_G];
    const int B = desc[GGML_SJKVARN_DESC_B], old = desc[GGML_SJKVARN_DESC_B_OLD];
    const int N = desc[GGML_SJKVARN_DESC_N], qpos = desc[GGML_SJKVARN_DESC_QPOS0];
    if (G != 128 || S < 0 || S % G != 0 || B < S || N < B || N > nunits*unit ||
            (unit != 16 && unit != 128)) {
        if (threadIdx.x == 0) {
            printf("LSA SPEC RECALL INVALID: layer=%d S=%d G=%d B=%d N=%d units=%d unit=%d\n",
                    m.layer, S, G, B, N, nunits, unit);
        }
        return;
    }
    const uint32_t resolve = m.ctl[0] + 1;
    // Consecutive, single-sequence append only. Reset, restore, prompt replay and growing allocations start cold.
    const bool warm = ggml_lsa_history_contiguous(history[0], history[1], uint32_t(qpos), uint32_t(N), nrows, resolve);
    ggml_lsa_workset_counts ws{};
    constexpr int planes = GGML_LSA_RECALL_PLANES;
    ggml_lsa_recall_counts count[3*planes]{}; // {missing pages, missing strips, all strips} per plane
    for (int page = threadIdx.x; page < (N + 127)/128; page += blockDim.x) {
        const int first = page*128;
        uint32_t actual_bits = 0, pred_bits[planes]{};
        for (int strip = 0; strip < 8 && first + strip*16 < N; ++strip) {
            bool actual = false;
            for (int r = 0; r < nrows; ++r) {
                for (int i = 0; i < 16 && first + strip*16 + i < N; ++i) {
                    actual |= __half2float(mask[r*pitch + first + strip*16 + i]) != -INFINITY;
                }
                for (int p = 0; p < planes; ++p) {
                    if (predicted[(p*nrows + r)*int64_t(nunits) + (first + strip*16)/unit] > 0.5f) {
                        pred_bits[p] |= 1u << strip;
                    }
                }
            }
            if (actual) { actual_bits |= 1u << strip; }
        }
        ws.observe(actual_bits, warm && first < int(history[0]) ? history[page + 2] : 0);
        history[page + 2] = actual_bits;
        // Whole-pick coverage is separate from missing-page recall; this is what a lossy prediction would drop.
        for (int p = 0; p < planes; ++p) {
            for (int strip = 0; strip < 8 && first + strip*16 < N; ++strip) {
                count[3*p + 2].observe(false, (actual_bits >> strip) & 1, (pred_bits[p] >> strip) & 1, true);
            }
        }
        // Physical-page recall matches exactly what the current prefetch can act on (full sealed records only).
        if (first < S || first + 128 > B) { continue; }
        const int g = (first - S)/G;
        const int record = desc[GGML_SJKVARN_DESC_TABLE] ? desc[GGML_SJKVARN_DESC_N_ENTRIES + g] : g;
        if (record < 0 || record >= int(m.records)) {
            // Descriptor corruption must not silently reduce the denominator and improve recall.
            printf("LSA SPEC RECALL INVALID: layer=%d record=%d records=%u\n", m.layer, record, m.records);
            continue;
        }
        const bool resident = m.table[record] >= 0, was_sealed = first + G <= old;
        for (int p = 0; p < planes; ++p) {
            count[3*p].observe(resident, actual_bits != 0, pred_bits[p] != 0, was_sealed);
            for (int strip = 0; strip < 8; ++strip) {
                count[3*p + 1].observe(resident, (actual_bits >> strip) & 1, (pred_bits[p] >> strip) & 1, was_sealed);
            }
        }
    }
    __shared__ ggml_lsa_recall_counts counts[64][3*planes];
    __shared__ ggml_lsa_workset_counts worksets[64];
    static_assert(sizeof(counts) + sizeof(worksets) <= 48*1024, "recall reduction exceeds shared memory");
    for (int j = 0; j < 3*planes; ++j) { counts[threadIdx.x][j] = count[j]; }
    worksets[threadIdx.x] = ws;
    __syncthreads();
    for (int d = 32; d > 0; d >>= 1) {
        if (threadIdx.x < d) {
            for (int j = 0; j < 3*planes; ++j) { counts[threadIdx.x][j].add(counts[threadIdx.x + d][j]); }
            worksets[threadIdx.x].add(worksets[threadIdx.x + d]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const auto w = worksets[0];
        // Shared-head arm only. head_strips/heads is its per-KV-head mean, with the same denominator as strips.
        printf("LSA WORKSET: layer=%d resolve=%u rows=%d N=%d B=%d partial=%d warm=%d mask_heads=1 heads=%u "
                "total_pages=%d pages=%u prev_pages=%u new_pages=%u total_strips=%d strips=%u prev_strips=%u new_strips=%u "
                "head_strips=%llu new_head_strips=%llu\n", m.layer, resolve, nrows, N, B,
                int(ggml_lsa_workset_partial_page(uint32_t(N))), int(warm), m.heads,
                (N + 127)/128, w.pages, w.prev_pages, w.new_pages, (N + 15)/16, w.strips, w.prev_strips, w.new_strips,
                (unsigned long long) w.strips*m.heads, (unsigned long long) w.new_strips*m.heads);
        for (int p = 0; p < planes; ++p) {
            const int ahead = p%GGML_LSA_RECALL_HORIZONS + 1;
            const double budget = 0.01*ggml_lsa_recall_budget_pct(p/GGML_LSA_RECALL_HORIZONS);
            for (int scale = 0; scale < 2; ++scale) {
                const auto c = counts[0][3*p + scale];
                printf("LSA SPEC RECALL: layer=%d resolve=%u rows=%d ahead=%d budget=%.2f cells=%d unit=%d actual_new=%u predicted_new=%u "
                        "hit=%u false=%u newly_sealed=%u record_bytes=%llu\n", m.layer, resolve, nrows, ahead, budget,
                        scale ? 16 : 128, unit, c.actual_new, c.predicted_new, c.hit, c.predicted_new - c.hit,
                        c.newly_sealed, (unsigned long long) m.stride);
            }
            const auto c = counts[0][3*p + 2];
            printf("LSA SPEC COVERAGE: layer=%d resolve=%u rows=%d ahead=%d budget=%.2f cells=16 unit=%d actual=%u predicted=%u "
                    "hit=%u miss=%u false=%u\n", m.layer, resolve, nrows, ahead, budget, unit, c.actual_new, c.predicted_new,
                    c.hit, c.actual_new - c.hit, c.predicted_new - c.hit);
        }
        history[0] = N;
        history[1] = resolve;
    }
}

void ggml_cuda_lsa_page_prepare(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!dst->src[5] || !dst->src[6]) return;
    auto * e = qsa_clock_entry(dst->src[5]);
    if (!e || e->map.cells != GGML_LSA_PAGE_CELLS) return;
    const int bits = ggml_get_op_params_i32(dst, 5);
    if (e->map.stage_stride) {
        GGML_ASSERT(ggml_get_op_params_i32(dst, 7) == GGML_TYPE_I16 && ggml_sj_kvarn::trellis3::tokens());
        GGML_ASSERT((bits >> 8 == 2 || bits >> 8 == 3) && ((bits & 255) == 2 || (bits & 255) == 3));
        const auto layout = ggml_sj_kvarn::make_layout(256, 128, bits >> 8, bits & 255);
        GGML_ASSERT(e->map.stride/e->map.heads == layout.bytes);
        GGML_ASSERT(layout.k_payload == 0);
        GGML_ASSERT(layout.v_payload == 4096*(bits >> 8));
        GGML_ASSERT(layout.k_scale == layout.v_payload + 4096*(bits & 255));
        GGML_ASSERT(layout.k_zero == layout.k_scale + 512);
        GGML_ASSERT(layout.k_tok == layout.k_zero + 512);
        GGML_ASSERT(layout.v_ch == layout.k_tok + 256);
        GGML_ASSERT(layout.v_scale == layout.v_ch + 512);
        GGML_ASSERT(layout.v_zero == layout.v_scale + 256);
    }
    lsa_pf_join(ctx.stream(), e);
    // GGML_CUDA_LSA_STAGE_AHEAD: every prepare joins the in-flight ahead copy (it writes a stage); the copy is applied
    // only by its target, and only when the target's prepare comes right after the prepare that queued it
    lsa_ahead_join(ctx.stream(), e->device);
    const uint64_t seq = ++lsa_ahead_seq[e->device];
    const bool apply = e->map.ahead_list && e->ahead_seq != 0 && e->ahead_seq + 1 == seq && !lsa_ahead_blocked &&
        dst->src[0]->ne[1] <= 8;
    e->ahead_seq = 0;
    e->prep_seq = seq;
    if (e->map.stage_n) {
        // before anything reads this layer's table again
        lsa_stage_reset<<<1, 256, 0, ctx.stream()>>>(e->map, apply ? 1 : 0,
            apply && e->ahead_host ? e->ahead_host_dev + (e->map.stage_n + 1) : nullptr);
        CUDA_CHECK(cudaGetLastError());
        if (e->applied_used) {
            CUDA_CHECK(cudaEventRecord(e->applied_used, ctx.stream())); // the host may rewrite the applied list after this
        }
    }
    if (dst->src[0]->ne[1] > 8) return; // prefill reads every record: slots as they are, the rest from the master
    const ggml_tensor * mask = dst->src[3];
    const half * m16 = nullptr;
    int64_t pitch = 0;
    int nrows = 0;
    if (mask && mask->type == GGML_TYPE_F16 && mask->ne[3] == 1 && (mask->ne[2] == 1 || mask->nb[2] == mask->ne[1]*mask->nb[1])) {
        // every plane (per-head masks) and query row: a record is live if any of them keeps one of its cells
        m16 = static_cast<const half *>(mask->data);
        pitch = mask->nb[1]/sizeof(half);
        nrows = int(std::min<int64_t>(dst->src[0]->ne[1], mask->ne[1]));
        if (mask->ne[2] > 1) {
            nrows = int(mask->ne[1]*mask->ne[2]); // contiguous planes: walk them as rows
        }
    } else {
        static bool warned = false;
        if (!warned) {
            warned = true;
            GGML_LOG_WARN("LSA RESIDENT: no plain f16 decode mask on a record-page body; every sealed record counts as picked\n");
        }
    }
    if (const ggml_tensor * prediction = ggml_flash_attn_ext_get_lsa_spec_recall(dst)) {
        GGML_ASSERT(ggml_cuda_graphs_disabled_by_env()); // diagnostic allocation/logging only
        GGML_ASSERT(m16 && nrows == dst->src[0]->ne[1] && mask->ne[2] == 1 && lsa_prefetch_cap() == 0);
        GGML_ASSERT(prediction->type == GGML_TYPE_F32 && ggml_is_contiguous(prediction) &&
                mask->ne[0] % prediction->ne[0] == 0 && prediction->ne[1] == nrows &&
                prediction->ne[2] == GGML_LSA_RECALL_HORIZONS && prediction->ne[3] == GGML_LSA_RECALL_BUDGETS);
        const int unit = int(mask->ne[0]/prediction->ne[0]);
        GGML_ASSERT(unit == 16 || unit == 128);
        const size_t pages = (size_t(mask->ne[0]) + 127)/128;
        if (pages > e->recall_pages) {
            // Diagnostic arm runs without CUDA graphs. Capacity grows in 256K-cell chunks, never per token.
            if (e->recall_history) { CUDA_CHECK(cudaFree(e->recall_history)); }
            e->recall_pages = ((pages + 2047)/2048)*2048;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&e->recall_history), (e->recall_pages + 2)*sizeof(uint32_t)));
            CUDA_CHECK(cudaMemsetAsync(e->recall_history, 0, (e->recall_pages + 2)*sizeof(uint32_t), ctx.stream()));
        }
        lsa_page_spec_recall<<<1, 64, 0, ctx.stream()>>>(e->map, static_cast<const int32_t *>(dst->src[6]->data),
                m16, pitch, nrows, static_cast<const float *>(prediction->data), int(prediction->ne[0]), unit,
                e->recall_history);
        CUDA_CHECK(cudaGetLastError());
    }
    if (lsa_prefetch_cap() > 0) {
        lsa_page_prefetch(ctx, e, static_cast<const int32_t *>(dst->src[6]->data), m16, pitch, nrows);
    }
    const int per_block = 8;
    lsa_page_mark<<<(e->map.records + per_block - 1)/per_block, per_block*WARP_SIZE, 0, ctx.stream()>>>(
        e->map, static_cast<const int32_t *>(dst->src[6]->data), m16, pitch, nrows, lsa_fetch_cap(e->map.slots), 0);
    CUDA_CHECK(cudaGetLastError());
    lsa_page_assign<<<1, 1024, 0, ctx.stream()>>>(e->map, lsa_fetch_cap(e->map.slots), 0, lsa_stage_all());
    CUDA_CHECK(cudaGetLastError());
    if (e->ahead_host && e->map.ahead_list && !lsa_ahead_blocked) {
        lsa_ahead_mirror<<<1, 256, 0, ctx.stream()>>>(e->map, e->ahead_host_dev);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(e->mirror_done, ctx.stream()));
        e->mirror_valid = true;
    }
    // More record copies in flight can hide mapped-host load latency.
    const uint32_t copy_blocks = e->map.stage_all && lsa_copy_wide() ? 256 : 64;
    lsa_page_copy_misses<<<dim3(copy_blocks, e->map.heads), 256, 0, ctx.stream()>>>(e->map);
    CUDA_CHECK(cudaGetLastError());
    if (e->map.stage_n) {
        lsa_stage_copy<<<dim3(copy_blocks, e->map.heads), 256, 0, ctx.stream()>>>(e->map, bits >> 8, bits & 255);
        CUDA_CHECK(cudaGetLastError());
    }
}

// GGML_CUDA_LSA_STAGE_AHEAD: queue the ahead copy of `next` (its last staged pick into its stage) on the side stream,
// after everything queued on the compute stream so far; next's prepare joins it and applies it only if that prepare is
// the device's very next one (ahead_seq + 1). from = issuing layer for the log (-1: graph begin).
static void lsa_ahead_issue(ggml_backend_cuda_context & ctx, qsa_clock_tensor * next, int from) {
    const int dev = next->device;
    if (!lsa_ahead_logged) {
        lsa_ahead_logged = true;
        GGML_LOG_WARN("LSA STAGE AHEAD: first ahead copy, layer %d -> layer %d (%u blocks per head)\n", from,
            next->map.layer, lsa_stage_ahead_blocks());
    }
    cudaStream_t side = lsa_ahead_stream[dev];
    if (next->ahead_host) {
        // copy engine: the list is the target's last offered pick, read from its mirror once that write has landed
        // (else no ahead copy this time); records sorted so neighbours go in one cudaMemcpyAsync
        if (!next->mirror_valid) {
            return;
        }
        const cudaError_t q0 = cudaEventQuery(next->mirror_done);
        const cudaError_t q1 = q0 == cudaSuccess ? cudaEventQuery(next->applied_used) : cudaSuccess;
        if (q0 != cudaSuccess || q1 != cudaSuccess) {
            // not ready (or captured): no ahead copy this time; clear only the error these queries reported
            if (cudaPeekAtLastError() == (q0 != cudaSuccess ? q0 : q1)) (void) cudaGetLastError();
            ++lsa_ahead_skipped;
            return;
        }
        const uint32_t sn = next->map.stage_n;
        const int32_t * mirror = next->ahead_host;
        int32_t * applied = next->ahead_host + (sn + 1);
        const uint32_t n = std::min<uint32_t>(uint32_t(std::max(mirror[0], 0)), sn);
        uint32_t k = 0;
        for (uint32_t i = 0; i < n; ++i) {
            const int32_t r = mirror[1 + i];
            if (r >= 0 && uint32_t(r) < next->map.records) applied[1 + k++] = r;
        }
        std::sort(applied + 1, applied + 1 + k);
        k = uint32_t(std::unique(applied + 1, applied + 1 + k) - (applied + 1));
        applied[0] = int32_t(k);
        if (k == 0) {
            return;
        }
        CUDA_CHECK(cudaEventRecord(lsa_ahead_fork[dev], ctx.stream()));
        CUDA_CHECK(cudaStreamWaitEvent(side, lsa_ahead_fork[dev], 0));
        const size_t stride = next->map.stride;
        for (uint32_t j = 0; j < k;) {
            uint32_t run = 1;
            while (j + run < k && applied[1 + j + run] == applied[1 + j] + int32_t(run)) ++run;
            CUDA_CHECK(cudaMemcpyAsync(next->map.stage + size_t(j)*stride, next->host + size_t(applied[1 + j])*stride,
                size_t(run)*stride, cudaMemcpyHostToDevice, side));
            ++lsa_ahead_dma_copies;
            j += run;
        }
    } else {
        CUDA_CHECK(cudaEventRecord(lsa_ahead_fork[dev], ctx.stream()));
        CUDA_CHECK(cudaStreamWaitEvent(side, lsa_ahead_fork[dev], 0));
        lsa_stage_ahead_copy<<<dim3(lsa_stage_ahead_blocks(), next->map.heads), 256, 0, side>>>(next->map);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(lsa_ahead_done[dev], side));
    lsa_ahead_pending[dev] = true;
    next->ahead_seq = lsa_ahead_seq[dev];
    ++lsa_ahead_issued;
    if (from < 0) {
        ++lsa_ahead_first_issued;
    }
}

// GGML_CUDA_LSA_STAGE_AHEAD: called right after a decode attention on a record-page body is queued. The next LSA layer
// (by layer number, same device, the other stage) gets its last staged pick copied into its stage on the side stream,
// after everything queued so far (this attention included: the copy then overlaps the layers in between, not the
// attention kernel, which fills every SM). Its prepare joins the copy before the stage reset applies it.
void ggml_cuda_lsa_page_ahead(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    if (!dst->src[5] || !dst->src[6] || dst->src[0]->ne[1] > 8 || lsa_ahead_blocked) return;
    auto * e = qsa_clock_entry(dst->src[5]);
    if (!e || e->map.cells != GGML_LSA_PAGE_CELLS || !e->map.ahead_list || e->device < 0) return;
    const int dev = e->device;
    if (e->prep_seq != lsa_ahead_seq[dev] || lsa_ahead_pending[dev] || !lsa_ahead_stream[dev]) return;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &cap));
    if (cap != cudaStreamCaptureStatusNone) return; // CUDA graphs: the resolve copies as before
    qsa_clock_tensor * next = nullptr;
    {
        std::lock_guard<std::mutex> lock(lsa_pf_mutex);
        for (auto * c : lsa_ahead_entries) {
            if (c != e && c->owner == e->owner && c->device == dev && c->map.layer > e->map.layer &&
                    c->map.records == e->map.records &&
                    c->map.stage != e->map.stage && (!next || c->map.layer < next->map.layer)) {
                next = c;
            }
        }
    }
    if (!next) return;
    lsa_ahead_issue(ctx, next, e->map.layer);
}

// GGML_CUDA_LSA_STAGE_AHEAD: at the start of every graph compute on the device. An ahead copy never carries across
// graphs (device-side master writes such as K-shift or defrag graphs do not pass through the host write path): the
// stream waits for any copy still in flight, and the prepare counter moves so it is not applied.
// HOSTKV08 (GGML_CUDA_LSA_STAGE_AHEAD_FIRST, default 1 with the ahead copy; =0 off): after that, the first decode
// attention on a record-page body in this graph gets its ahead copy here, so it overlaps the layers before it (the
// first LSA layer of a verify, and the first one of every later split). The copy is queued after all work so far on
// the compute stream (earlier graphs included), so it reads the master as every earlier graph left it; this graph's
// own seal is handled by the mark exactly as for an in-graph ahead copy.
static bool lsa_stage_ahead_first() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_LSA_STAGE_AHEAD_FIRST");
        return v == nullptr || *v == 0 || atoi(v) > 0;
    }();
    return on;
}

void ggml_cuda_lsa_ahead_graph_begin(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    const int dev = ctx.device;
    if (dev < 0 || dev >= GGML_CUDA_MAX_DEVICES || !lsa_ahead_stream[dev]) return;
    lsa_ahead_join(ctx.stream(), dev);
    ++lsa_ahead_seq[dev];
    // eager graphs only: a captured graph's prepare could not wait on a copy queued outside the capture
    if (!cgraph || lsa_ahead_blocked || !lsa_stage_ahead_first() || !ggml_cuda_graphs_disabled_by_env()) return;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_FLASH_ATTN_EXT || !node->src[5] || !node->src[6]) continue;
        auto * e = qsa_clock_entry(node->src[5]);
        if (!e || e->map.cells != GGML_LSA_PAGE_CELLS) continue;
        // the first record-page attention decides: decode with an ahead list on this device, else nothing
        if (node->src[0]->ne[1] > 8 || !e->map.ahead_list || e->device != dev) return;
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &cap));
        if (cap != cudaStreamCaptureStatusNone) return;
        lsa_ahead_issue(ctx, e, -1);
        return;
    }
}
