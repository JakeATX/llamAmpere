#include "moe-cache.cuh"

#if defined(GGML_USE_MUSA)

extern "C" size_t ggml_moe_cache_trim(int device) {
    (void) device;
    return 0;
}

uint64_t ggml_cuda_moe_cache_device_capture_generation(int) { return 0; }
void ggml_cuda_moe_cache_device_graph_record(int, bool, const ggml_cgraph *) {}
bool ggml_cuda_moe_cache_device_supports(const ggml_tensor *, int) { return false; }
bool ggml_cuda_moe_cache_device_compute(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
bool ggml_cuda_moe_cache_device_fused_compute(ggml_backend_cuda_context &, ggml_tensor *, ggml_tensor *, ggml_tensor *) { return false; }

static void moe_cache_register_stub(const void * owner) {
    (void) owner;
}

#else

#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "../ggml-backend-moe-cache.h"
#include "../ggml-moe-prefetch.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#define MOE_CACHE_LOG(...) GGML_LOG_INFO(__VA_ARGS__)

static constexpr int    moe_cache_cc_forced_min               = 700;
static constexpr int    moe_cache_cc_ampere                   = 800;
static constexpr size_t moe_cache_expert_bytes_ampere_min     = 512u << 10;
static constexpr size_t moe_cache_expert_bytes_pre_ampere_min = 1u << 20;
static constexpr int    moe_cache_batch_max                   = 8;
static constexpr int    moe_cache_pool_slots_min              = 64;
static constexpr size_t moe_cache_slab_bytes_auto_min         = 1ull << 30;
static constexpr int    moe_cache_node_rows_max               = 64;
static constexpr size_t moe_cache_overlap_bytes_per_token     = 8u << 20;

// Adaptive cpu-overlap calibration (see moe_cache_calibration_on_enter/leave):
// only runs while overlap_cpu_rows is left automatic. Candidate -1 means
// "use the existing byte-budget formula below" and is tried first, so a
// session that never runs enough single-token decode scopes to finish
// calibrating (e.g. every unit test, or a prefill-only run) behaves exactly
// as before this change. Candidates are tried in order, each over a window
// of real decode scopes; whichever measures the lowest average wall time
// wins and is locked in for the rest of the session.
static constexpr int    moe_cache_calib_candidates[]           = { -1, 0, 1, 2, 4, 8 };
static constexpr int    moe_cache_calib_candidate_count        = 6;
// Absorbs one-time cold-start cost (CUDA kernel JIT, page cache fill, thread/
// frequency governor settling) before any candidate timing counts -- this
// dwarfs the row-count effect being measured if skipped.
static constexpr int    moe_cache_calib_global_warmup_scopes   = 64;
// Round-robin sweep: each candidate gets `rounds` short trials rather than
// one long block, so residual drift after warm-up spreads evenly instead of
// biasing whichever candidate happens to run first.
static constexpr int    moe_cache_calib_rounds                 = 3;
static constexpr int    moe_cache_calib_round_warmup_scopes    = 1;
static constexpr int    moe_cache_calib_round_measured_scopes  = 4;

enum class moe_cache_slot_state : uint8_t {
    free,
    copying,
    valid,
};

struct moe_cache_key {
    const void * tensor = nullptr;
    int32_t expert = -1;

    bool operator==(const moe_cache_key & other) const {
        return tensor == other.tensor && expert == other.expert;
    }
};

struct moe_cache_key_hash {
    size_t operator()(const moe_cache_key & key) const {
        uint64_t value = (uint64_t)(uintptr_t)key.tensor;
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= (uint64_t)(uint32_t)key.expert * 0x9e3779b97f4a7c15ULL;
        value ^= value >> 29;
        return (size_t)value;
    }
};

struct moe_prefetch_copy {
    int physical = -1;
    const void * tensor = nullptr;
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    bool published = false;
    int source_layer = -1;
    int target_layer = -1;
    int expert = -1;
    int rank = -1;
    size_t bytes = 0;
    int64_t issue = 0;
    int64_t start = 0;
    int64_t done = 0;
    int64_t target_start = 0;
    int64_t deadline = 0;
    bool selected = false;
    bool used = false;
    bool closed = false;
    bool logged = false;
};

static void moe_prefetch_log(moe_prefetch_copy & copy) {
    if (!copy.closed || copy.done == 0 || copy.logged) {
        return;
    }
    copy.logged = true;
    MOE_CACHE_LOG("[moe-prefetch] source=%d target=%d expert=%d rank=%d bytes=%zu issue_us=%lld start_us=%lld done_us=%lld target_start_us=%lld selected=%d canceled=%d outcome=%s\n",
        copy.source_layer, copy.target_layer, copy.expert, copy.rank, copy.bytes,
        (long long) copy.issue, (long long) copy.start, (long long) copy.done, (long long) copy.target_start,
        copy.selected, copy.done < 0, copy.used ? "used" : "wasted");
}

struct moe_cache_slot {
    moe_cache_key key;
    std::shared_ptr<moe_prefetch_copy> prediction;
    uint64_t generation = 0;
    int prev = -1;
    int next = -1;
    int readers = 0;
    // Resident hit count since the slot was populated. Used by the heat-aware
    // eviction policy to keep frequently-used experts resident.
    uint32_t uses = 0;
    // Prompt/profile frequency ranks lending survivors, not decode eviction heat.
    uint32_t warm_uses = 0;
    bool staging = false;
    moe_cache_slot_state state = moe_cache_slot_state::free;
};

static uint64_t moe_cache_lending_uses(const moe_cache_slot & slot) {
    return (uint64_t)slot.warm_uses + slot.uses;
}

struct moe_cache_pool {
    size_t expert_size = 0;
    int wtype = -1;
    char * slab = nullptr;
    int n_slots = 0;
    bool covers_all_entries = false;

    std::vector<moe_cache_slot> slots;
    std::vector<int> free_slots;
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    int lru_head = -1;
    int lru_tail = -1;
};

struct moe_cache_shape {
    size_t expert_size = 0;
    int wtype = -1;
    uint64_t n_entries = 0;
    int64_t n_tensors = 0;
    int pool = -1;
    bool finished = false;
};

struct moe_cache_seen_tensor {
    size_t bytes = 0;
    size_t expert_size = 0;
    int wtype = -1;
    int64_t n_expert = 0;
};

struct moe_cache_job {
    int pool = -1;
    int slot = -1;
    uint64_t generation = 0;
    moe_cache_key key;
    const void * source = nullptr;
    size_t bytes = 0;
    std::shared_ptr<moe_prefetch_copy> prediction;
};

struct moe_cache_demand {
    uint16_t count = 0;
    size_t expert_size = 0;
};

struct moe_cache_config {
    bool enabled = true;
    bool automatic = true;
    size_t budget_mb = 0;
    size_t reserve_mb = 3072;
    size_t minimum_slab_bytes = moe_cache_slab_bytes_auto_min;
    size_t min_expert_bytes = moe_cache_expert_bytes_pre_ampere_min;
    bool min_expert_explicit = false;
    int max_batch = moe_cache_batch_max;
    bool max_batch_explicit = false;
    int inserts_per_plan = 8;
    int admit_after = 2;
    bool admit_after_explicit = false;
    int readmit_after = 8;
    int hot_uses = 4;
    int queue_max = 128;
    size_t queue_mb = 512;
    int stats_every = 0;
    int max_devices = INT_MAX;
    int min_compute_capability = moe_cache_cc_forced_min;
    bool serial_fill = true;
    bool serial_fill_explicit = false;
    bool force_dedicated_mmv = false;
    int overlap_cpu_rows = -1;
    bool overlap_cpu_rows_explicit = false;
    std::string fail_stage;
};

struct moe_cache_session;

struct moe_cache_scratch {
    size_t input = 0;
    size_t q8 = 0;
    size_t out = 0;
};

struct moe_cache_device {
    moe_cache_device(int logical, int physical) : logical(logical), physical(physical) {}

    int logical;
    int physical;
    std::atomic<bool> dead{false};
    std::mutex dispatch_mu;

    std::vector<std::unique_ptr<moe_cache_pool>> pools;
    std::vector<moe_cache_shape> shapes;
    std::unordered_map<const void *, moe_cache_seen_tensor> seen_tensors;
    std::unordered_map<const void *, size_t> host_strides;
    std::unordered_map<moe_cache_key, moe_cache_demand, moe_cache_key_hash> demand_count;
    size_t visits_since_new_tensor = 0;
    bool budget_ready = false;
    bool budget_registered = false;
    bool budget_claimed = false;
    size_t budget_reserve_bytes = 0;
    size_t budget_limit = 0;
    size_t coordinator_allocated_bytes = 0;
    size_t allocated_bytes = 0;
    moe_cache_scratch scratch_reserve;
    // Device table arena: mandatory workspace, materialized before expert slabs.
    char * table_arena = nullptr;
    size_t table_arena_cap = 0;

    std::deque<moe_cache_job> queue;
    size_t queued_bytes = 0;
    bool worker_started = false;
    bool inflight = false;
    bool inflight_prefetch = false;
    // Complete-layer and legacy miss streams raise this before staging real misses.
    std::atomic<unsigned> urgent_transfers{0};
    const void * inflight_source = nullptr;
    size_t inflight_bytes = 0;
    std::thread worker;

    double copy_bytes_per_us = 0.0;
    int copy_samples = 0;
    cudaStream_t compute_stream = nullptr;
    char * h_input = nullptr;
    char * d_input = nullptr;
    size_t h_input_cap = 0;
    size_t d_input_cap = 0;
    void * d_act_q8 = nullptr;
    size_t act_q8_cap = 0;
    float * d_out = nullptr;
    size_t d_out_cap = 0;
    float * h_out = nullptr;
    size_t h_out_cap = 0;

    // hits and misses count tensor-expert probes; fusion counters count row pairs.
    long long hits = 0;
    long long misses = 0;
    long long inserts = 0;
    long long fills = 0;
    long long fill_failures = 0;
    long long evictions = 0;
    // Evictions that sacrificed a hot (frequently-used) expert because every
    // eligible slot was hot. A steady zero means hot experts stay resident.
    long long heat_evictions = 0;
    long long insert_skips = 0;
    long long admission_skips = 0;
    long long dispatch_failures = 0;
    long long collect_failures = 0;
    long long activation_dedup = 0;
    long long overlap_rows = 0;
    long long fused_rows = 0;
    long long fused_candidates = 0;
    long long pair_both = 0;
    long long pair_up_only = 0;
    long long pair_gate_only = 0;
    long long pair_neither = 0;
    long long fused_attempts = 0;
    long long fused_nodes = 0;
    long long nodes = 0;
    long long collect_calls = 0;
    // Dispatch mutex contention that turned a potential cache hit into a
    // complete CPU fallback (try_to_lock failed). Measured first so a bounded
    // wait or per-device FIFO can be justified with data.
    long long contention_bypasses = 0;
    std::atomic<int> error_logs{0};
};

struct moe_cache_mapped_range {
    uintptr_t begin = 0;
    uintptr_t end = 0;
    void * device = nullptr;
    void * staging = nullptr;
    bool registered = false;
};

struct moe_cache_device_table {
    ggml_moe_cache_tensor_desc desc{}; // compact expert bytes
    size_t host_stride = 0;
    const void * borrowed = nullptr; // model-owned registered arena; never unregistered here
    const void ** device = nullptr;
    const void ** staging_device = nullptr;
    uint8_t * hot = nullptr;
    std::vector<const void *> addresses;
    std::vector<const void *> fallback;
    // Host addresses are distinct from the CUDA aliases in fallback.
    std::vector<const void *> copy_source;
    size_t count_offset = 0;
};

struct moe_cache_session {
    uint64_t phase_stats_id = 0;
    moe_cache_config config;
    std::vector<std::unique_ptr<moe_cache_device>> devices;
    std::unordered_map<int, int> layer_devices;
    std::unordered_map<const void *, int> tensor_devices;

    std::mutex mu;
    std::mutex fill_mu;
    std::condition_variable cv;
    std::condition_variable idle_cv;
    std::atomic<bool> stopping{false};
    std::atomic<bool> dormant{false};
    std::atomic<bool> config_announced{false};
    std::atomic<bool> enabled_announced{false};
    std::atomic<bool> batch_bypass_announced{false};
    std::atomic<bool> row_bypass_announced{false};
    uint64_t device_capture_generation = 0;
    std::atomic<uint64_t> device_captures{0};
    std::atomic<uint64_t> device_replays{0};
    unsigned long long * device_expert_counts = nullptr;
    std::vector<unsigned long long> device_previous_counts;
    uint64_t device_adapt_scopes = 0;
    uint64_t device_adapt_interval = 64;
    size_t device_adapt_copy_cap = 64u << 20;
    // One stream for idle-boundary expert copies. Payloads finish before the
    // address tables are published and before the next graph is launched.
    cudaStream_t device_adapt_stream = nullptr;
    size_t device_fallback_staging_bytes = 0;
    unsigned long long * device_route_counts = nullptr;
    // Diagnostic only; guarded by mu and unused without DEVICE_STATS.
    uint64_t device_stats_scopes = 0;
    std::atomic<uint64_t> device_dispatches{0};
    bool profile_load_attempted = false;
    std::unordered_map<moe_cache_key, uint64_t, moe_cache_key_hash> profile_counts;
    bool prefetch_enabled = false;
    std::vector<std::shared_ptr<moe_prefetch_copy>> predictions;
    int64_t layer_start[1024] = {};
    double layer_time_us[1024] = {};
    int last_layer = -1;
    // Table allocations and slabs stay fixed; entries change only at idle boundaries.
    bool device_frozen = false;
    bool device_attempted = false;
    std::atomic<bool> device_ready{false};
    std::vector<moe_cache_mapped_range> mapped_ranges;
    // Host-source mappings survive table/slab reranks, until source invalidation.
    std::unordered_map<const void *, ggml_moe_cache_tensor_desc> mapped_sources;
    std::unordered_map<const void *, moe_cache_device_table> device_tables;
    int active_scopes = 0;
    int active_nodes = 0;
    struct active_source {
        size_t bytes = 0;
        int references = 0;
    };
    std::unordered_map<const void *, active_source> active_sources;

    // Adaptive cpu-overlap calibration state, guarded by `mu` (same lock
    // session_enter/session_leave/moe_cache_plan already hold). Unused
    // whenever overlap_cpu_rows is explicitly configured.
    //
    // Two-phase: a long unmeasured global warm-up first (CUDA kernel JIT,
    // page cache fill, thread/frequency governor settling all cost far more
    // than the row-count effect being measured, so they must fully drain
    // before any timing counts), then a round-robin sweep -- one short trial
    // per candidate per round, cycling through all candidates several times
    // and averaging each candidate's trials across rounds -- rather than one
    // long block per candidate, so any *residual* drift after warm-up is
    // spread evenly across every candidate instead of biasing whichever one
    // happened to run first.
    bool     calib_started            = false;
    bool     calib_done               = false;
    bool     calib_global_warmup_done = false;
    int      calib_global_warmup_seen = 0;
    int      calib_round              = 0;
    int      calib_slot               = 0; // rotated by calib_round; see moe_cache_calibration_on_leave
    int      calib_current_rows       = -1; // what moe_cache_overlap_rows() should use right now (-1 = formula)
    int      calib_scopes_seen        = 0;
    double   calib_time_sum_ms        = 0.0;
    int      calib_time_count         = 0;
    double   calib_candidate_sum_ms[moe_cache_calib_candidate_count]  = {};
    int      calib_candidate_samples[moe_cache_calib_candidate_count] = {};
    int      calib_best_rows          = -1;
    double   calib_best_avg_ms        = -1.0;
    std::chrono::steady_clock::time_point calib_scope_start{};
    int64_t  calib_scope_max_tokens = 0;
    int      calib_scope_nodes      = 0;
};

struct moe_cache_pin {
    moe_cache_pool * pool = nullptr;
    int slot = -1;
};

static size_t moe_cache_host_stride(const moe_cache_device & device, const void * base, size_t bytes) {
    const auto it = device.host_strides.find(base);
    return it == device.host_strides.end() ? bytes : it->second;
}

struct moe_cache_node {
    moe_cache_session * session = nullptr;
    moe_cache_device * device = nullptr;
    moe_cache_pool * pool = nullptr;
    int pool_index = -1;
    const void * host_base = nullptr;
    const void * host_base2 = nullptr;
    size_t expert_size = 0;
    int64_t n_in = 0;
    int64_t n_out = 0;
    int64_t n_expert = 0;
    int64_t n_tokens = 0;
    int wtype = -1;
    std::unique_lock<std::mutex> dispatch_lock;
    moe_cache_pin pins[2 * moe_cache_node_rows_max];
    int n_pins = 0;
    bool planned = false;
    bool dispatched = false;
};

static std::mutex g_registry_mu;
static std::unordered_set<moe_cache_session *> g_sessions;
static std::atomic<int> g_session_count{0};
static std::atomic<uint64_t> g_phase_stats_session_id{1};
struct moe_cache_physical_budget {
    int participants = 0;
    int prepared = 0;
    std::multiset<size_t> reserves;
    size_t outstanding_bytes = 0;
};
static std::mutex g_budget_mu;
static std::unordered_map<int, moe_cache_physical_budget> g_physical_budgets;
struct moe_cache_scope_frame {
    moe_cache_session * requested = nullptr;
    moe_cache_session * active = nullptr;
};
static thread_local std::vector<moe_cache_scope_frame> g_session_stack;
static thread_local int g_session_suppressed = 0;

static void moe_prefetch_finish(moe_cache_session & session, moe_prefetch_copy & copy);

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device);
static void moe_cache_device_release(moe_cache_session & session, bool retain_sources = false);
static void moe_cache_device_adapt(moe_cache_session & session);

static void moe_cache_budget_remove_participant(
        moe_cache_physical_budget & state, size_t reserve_bytes) {
    auto reserve = state.reserves.find(reserve_bytes);
    if (reserve != state.reserves.end()) {
        state.reserves.erase(reserve);
    }
    state.participants = std::max(state.participants - 1, 0);
}

static bool moe_cache_budget_register(moe_cache_session & session) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    std::vector<moe_cache_device *> registered;
    try {
        registered.reserve(session.devices.size());
        for (const auto & device_ptr : session.devices) {
            moe_cache_device & device = *device_ptr;
            moe_cache_physical_budget & state = g_physical_budgets[device.physical];
            const size_t reserve_bytes = session.config.reserve_mb << 20;
            state.reserves.insert(reserve_bytes);
            state.participants++;
            device.budget_registered = true;
            device.budget_reserve_bytes = reserve_bytes;
            registered.push_back(&device);
        }
        return true;
    } catch (...) {
        for (moe_cache_device * device : registered) {
            auto found = g_physical_budgets.find(device->physical);
            if (found != g_physical_budgets.end()) {
                moe_cache_budget_remove_participant(
                        found->second, device->budget_reserve_bytes);
                if (found->second.participants == 0) {
                    g_physical_budgets.erase(found);
                }
            }
            device->budget_registered = false;
            device->budget_reserve_bytes = 0;
        }
        for (auto found = g_physical_budgets.begin();
             found != g_physical_budgets.end();) {
            if (found->second.participants == 0) {
                found = g_physical_budgets.erase(found);
            } else {
                ++found;
            }
        }
        return false;
    }
}

static void moe_cache_budget_allocation(
        moe_cache_device & device, size_t bytes, bool allocated) {
    if (!device.budget_registered || !device.budget_claimed || bytes == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }
    moe_cache_physical_budget & state = found->second;
    if (allocated) {
        const size_t remaining = device.budget_limit > device.coordinator_allocated_bytes
            ? device.budget_limit - device.coordinator_allocated_bytes : 0;
        const size_t tracked = std::min(bytes, remaining);
        device.coordinator_allocated_bytes += tracked;
        state.outstanding_bytes = tracked <= state.outstanding_bytes
            ? state.outstanding_bytes - tracked : 0;
    } else {
        const size_t tracked = std::min(bytes, device.coordinator_allocated_bytes);
        device.coordinator_allocated_bytes -= tracked;
        if (tracked <= SIZE_MAX - state.outstanding_bytes) {
            state.outstanding_bytes += tracked;
        } else {
            state.outstanding_bytes = SIZE_MAX;
        }
    }
}

static void moe_cache_budget_reallocation(
        moe_cache_device & device, size_t old_bytes, size_t new_bytes) {
    if (!device.budget_registered || !device.budget_claimed || old_bytes == new_bytes) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return;
    }

    moe_cache_physical_budget & state = found->second;
    const size_t tracked_old = std::min(old_bytes, device.coordinator_allocated_bytes);
    const size_t base = device.coordinator_allocated_bytes - tracked_old;
    const size_t remaining = device.budget_limit > base
        ? device.budget_limit - base : 0;
    const size_t tracked_new = std::min(new_bytes, remaining);
    device.coordinator_allocated_bytes = base + tracked_new;

    if (tracked_new >= tracked_old) {
        const size_t consumed = tracked_new - tracked_old;
        state.outstanding_bytes = consumed <= state.outstanding_bytes
            ? state.outstanding_bytes - consumed : 0;
    } else {
        const size_t released = tracked_old - tracked_new;
        state.outstanding_bytes = released <= SIZE_MAX - state.outstanding_bytes
            ? state.outstanding_bytes + released : SIZE_MAX;
    }
}

static void moe_cache_budget_unregister(moe_cache_device & device) {
    if (!device.budget_registered) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    auto found = g_physical_budgets.find(device.physical);
    if (found != g_physical_budgets.end()) {
        moe_cache_physical_budget & state = found->second;
        if (device.budget_claimed) {
            const size_t unallocated = device.budget_limit > device.coordinator_allocated_bytes
                ? device.budget_limit - device.coordinator_allocated_bytes : 0;
            state.outstanding_bytes = unallocated <= state.outstanding_bytes
                ? state.outstanding_bytes - unallocated : 0;
            state.prepared = std::max(state.prepared - 1, 0);
        }
        moe_cache_budget_remove_participant(
                state, device.budget_reserve_bytes);
        if (state.participants == 0) {
            g_physical_budgets.erase(found);
        }
    }
    device.budget_registered = false;
    device.budget_claimed = false;
    device.budget_ready = false;
    device.budget_reserve_bytes = 0;
    device.budget_limit = 0;
    device.coordinator_allocated_bytes = 0;
}

static size_t moe_cache_budget_claim(
        moe_cache_session & session, moe_cache_device & device,
        size_t free_memory, size_t & reserve_bytes) {
    std::lock_guard<std::mutex> lock(g_budget_mu);
    reserve_bytes = 0;
    auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end() || !device.budget_registered) {
        return 0;
    }
    moe_cache_physical_budget & state = found->second;
    reserve_bytes = state.reserves.empty()
        ? 0 : *state.reserves.rbegin();
    const size_t available_after_reserve = free_memory > reserve_bytes
        ? free_memory - reserve_bytes : 0;
    const size_t unclaimed = available_after_reserve > state.outstanding_bytes
        ? available_after_reserve - state.outstanding_bytes : 0;
    const int remaining_participants = std::max(
            state.participants - state.prepared, 1);
    size_t claim = unclaimed / (size_t)remaining_participants;
    if (session.config.budget_mb > 0) {
        claim = std::min(claim, session.config.budget_mb << 20);
    }
    state.prepared++;
    state.outstanding_bytes += claim;
    device.budget_claimed = true;
    return claim;
}

static bool moe_cache_env_i64(
        const char * name, int64_t min_value, int64_t max_value, int64_t & value) {
    const char * text = getenv(name);
    if (!text || !text[0]) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long long parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        MOE_CACHE_LOG("[moe-cache] ignoring invalid %s=%s\n", name, text);
        return false;
    }

    value = parsed;
    return true;
}

static int moe_cache_min_compute_capability(bool automatic) {
    int result = automatic ? moe_cache_cc_ampere : moe_cache_cc_forced_min;
    int64_t value = 0;
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_CC", 0, 999, value)) {
        result = (int)value;
    }
    return result;
}

static size_t moe_cache_default_min_expert_bytes(int compute_capability) {
    return compute_capability >= moe_cache_cc_ampere
        ? moe_cache_expert_bytes_ampere_min
        : moe_cache_expert_bytes_pre_ampere_min;
}

static void moe_cache_apply_mode_defaults(moe_cache_config & config) {
    config.minimum_slab_bytes = config.automatic
        ? moe_cache_slab_bytes_auto_min : 0;
    if (!config.min_expert_explicit) {
        config.min_expert_bytes = config.automatic
            ? moe_cache_expert_bytes_ampere_min
            : moe_cache_expert_bytes_pre_ampere_min;
    }
    if (!config.max_batch_explicit) {
        config.max_batch = moe_cache_batch_max;
    }
    if (!config.overlap_cpu_rows_explicit) {
        config.overlap_cpu_rows = -1;
    }
}

static moe_cache_config moe_cache_read_config() {
    moe_cache_config config;
    int64_t value = 0;
    bool mode_off = false;
    bool mode_valid = false;

    if (const char * mode = getenv("GGML_CUDA_MOE_CACHE_MODE")) {
        if (strcmp(mode, "auto") == 0) {
            config.automatic = true;
            mode_valid = true;
        } else if (strcmp(mode, "on") == 0) {
            config.automatic = false;
            mode_valid = true;
        } else if (strcmp(mode, "off") == 0) {
            config.enabled = false;
            mode_off = true;
            mode_valid = true;
        } else {
            MOE_CACHE_LOG("[moe-cache] ignoring invalid GGML_CUDA_MOE_CACHE_MODE=%s\n", mode);
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE", 0, 1, value)) {
        config.enabled = value != 0 && !mode_off;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_BUDGET_MB", 1, 1024 * 1024, value)) {
        config.budget_mb = (size_t)value;
        if (!mode_valid) {
            config.automatic = false;
        }
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_RESERVE_MB", 0, 1024 * 1024, value)) {
        config.reserve_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB", 1, 1024 * 1024, value)) {
        config.min_expert_bytes = (size_t)value << 10;
        config.min_expert_explicit = true;
    }
    if (moe_cache_env_i64(
            "GGML_CUDA_MOE_CACHE_MAX_BATCH", 1, moe_cache_batch_max, value)) {
        config.max_batch = (int)value;
        config.max_batch_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_INSERTS", 1, 1024, value)) {
        config.inserts_per_plan = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_ADMIT_AFTER", 1, 255, value)) {
        config.admit_after = (int)value;
        config.admit_after_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_THROTTLE", 1, 1024, value)) {
        config.readmit_after = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_HOT_USES", 0, 1 << 30, value)) {
        config.hot_uses = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE", 1, 65536, value)) {
        config.queue_max = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_QUEUE_MB", 1, 1024 * 1024, value)) {
        config.queue_mb = (size_t)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_STATS", 0, INT_MAX, value)) {
        config.stats_every = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_NDEV", 1, INT_MAX, value)) {
        config.max_devices = (int)value;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_SERIAL_FILL", 0, 1, value)) {
        config.serial_fill = value != 0;
        config.serial_fill_explicit = true;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_DEDICATED_MMV", 0, 1, value)) {
        config.force_dedicated_mmv = value != 0;
    }
    if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_OVERLAP_CPU_ROWS", 0, 8, value)) {
        config.overlap_cpu_rows = (int)value;
        config.overlap_cpu_rows_explicit = true;
    }
    moe_cache_apply_mode_defaults(config);
    config.min_compute_capability = moe_cache_min_compute_capability(config.automatic);
    if (const char * fail = getenv("GGML_CUDA_MOE_CACHE_FAIL")) {
        config.fail_stage = fail;
    }

    return config;
}

static bool moe_cache_fail(const moe_cache_session & session, const char * stage) {
    const std::string & value = session.config.fail_stage;
    if (value.empty()) {
        return false;
    }
    if (value == "all" || value == stage) {
        return true;
    }

    size_t begin = 0;
    while (begin < value.size()) {
        size_t end = value.find(',', begin);
        if (end == std::string::npos) {
            end = value.size();
        }
        if (value.compare(begin, end - begin, stage) == 0) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

static bool moe_cache_ranges_overlap(
        const void * lhs, size_t lhs_size, const void * rhs, size_t rhs_size) {
    if (!lhs || !rhs || lhs_size == 0 || rhs_size == 0) {
        return false;
    }
    const uintptr_t l = (uintptr_t)lhs;
    const uintptr_t r = (uintptr_t)rhs;
    return (l <= r ? r - l < lhs_size : l - r < rhs_size);
}

static uint64_t moe_cache_name_hash(const char * text) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    while (*text) {
        hash ^= (unsigned char)*text++;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static bool moe_cache_layer_number(const char * name, int & layer) {
    const char * marker = strstr(name, "blk.");
    if (!marker) {
        return false;
    }

    const char * first = marker + 4;
    char * end = nullptr;
    errno = 0;
    const long parsed = strtol(first, &end, 10);
    if (errno != 0 || end == first || parsed < 0 || parsed > INT_MAX) {
        return false;
    }
    if (*end != '.' && *end != '\0') {
        return false;
    }
    layer = (int)parsed;
    return true;
}

static bool moe_cache_tensor_name_supported(const char * name) {
    return strstr(name, "_exps") || strstr(name, "_chexps");
}

static bool moe_cache_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static void moe_cache_lru_remove(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.prev >= 0) {
        pool.slots[slot.prev].next = slot.next;
    } else {
        pool.lru_head = slot.next;
    }
    if (slot.next >= 0) {
        pool.slots[slot.next].prev = slot.prev;
    } else {
        pool.lru_tail = slot.prev;
    }
    slot.prev = -1;
    slot.next = -1;
}

static void moe_cache_lru_push_back(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    slot.prev = pool.lru_tail;
    slot.next = -1;
    if (pool.lru_tail >= 0) {
        pool.slots[pool.lru_tail].next = index;
    } else {
        pool.lru_head = index;
    }
    pool.lru_tail = index;
}

static void moe_cache_map_erase(moe_cache_pool & pool, int index) {
    moe_cache_slot & slot = pool.slots[index];
    auto it = pool.map.find(slot.key);
    if (it != pool.map.end() && it->second == index) {
        pool.map.erase(it);
    }
}

static void moe_cache_slot_reset(moe_cache_pool & pool, int index, bool add_to_free) {
    moe_cache_slot & slot = pool.slots[index];
    if (slot.state == moe_cache_slot_state::valid) {
        moe_cache_lru_remove(pool, index);
    }
    moe_cache_map_erase(pool, index);
    slot.key = {};
    slot.prediction.reset();
    slot.generation++;
    slot.readers = 0;
    slot.uses = 0;
    slot.warm_uses = 0;
    slot.state = moe_cache_slot_state::free;
    slot.prev = -1;
    slot.next = -1;
    if (add_to_free && !slot.staging) {
        pool.free_slots.push_back(index);
    }
}

static bool moe_cache_cuda_ok(
        moe_cache_device & device, cudaError_t error, const char * operation, bool fatal) {
    if (error == cudaSuccess) {
        return true;
    }

    (void)cudaGetLastError();
    if (device.error_logs.fetch_add(1) < 8) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d %s failed: %s\n",
                device.physical, operation, cudaGetErrorString(error));
    }
    if (fatal && error != cudaErrorMemoryAllocation) {
        device.dead.store(true);
    }
    return false;
}

// Recheck free memory at allocation time: a cached grant may predate the
// draft context, a different turn's compute graph, or another CUDA user.
struct moe_cache_memory_view {
    size_t free = 0;
    size_t total = 0;
    size_t reserve = 0;
    size_t other_outstanding = 0;
    size_t headroom = 0;
};

static bool moe_cache_memory_query(moe_cache_device & device, moe_cache_memory_view & view) {
    ggml_cuda_set_device(device.logical);
    if (!moe_cache_cuda_ok(device, cudaMemGetInfo(&view.free, &view.total), "live memory query", false)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    const auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end() || !device.budget_registered || !device.budget_claimed) {
        return false;
    }
    const auto & state = found->second;
    view.reserve = state.reserves.empty() ? 0 : *state.reserves.rbegin();
    const size_t own_outstanding = device.budget_limit > device.coordinator_allocated_bytes
        ? device.budget_limit - device.coordinator_allocated_bytes : 0;
    view.other_outstanding = state.outstanding_bytes > own_outstanding
        ? state.outstanding_bytes - own_outstanding : 0;
    const size_t available = view.free > view.reserve ? view.free - view.reserve : 0;
    view.headroom = available > view.other_outstanding ? available - view.other_outstanding : 0;
    return true;
}

static size_t moe_cache_allocation_headroom(moe_cache_device & device, const char * operation, size_t requested) {
    moe_cache_memory_view view;
    if (!moe_cache_memory_query(device, view)) {
        return 0;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d allocation bytes: operation=%s requested=%zu free=%zu total=%zu reserve=%zu other_outstanding=%zu headroom=%zu owned=%zu grant=%zu\n",
            device.physical, operation, requested, view.free, view.total, view.reserve, view.other_outstanding,
            view.headroom, device.coordinator_allocated_bytes, device.budget_limit);
    return view.headroom;
}

static bool moe_cache_refresh_budget(
        moe_cache_session & session, moe_cache_device & device, const char * phase) {
    moe_cache_memory_view view;
    if (!moe_cache_memory_query(device, view)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_budget_mu);
    const auto found = g_physical_budgets.find(device.physical);
    if (found == g_physical_budgets.end()) {
        return false;
    }
    auto & state = found->second;
    const size_t old_limit = device.budget_limit;
    const size_t own_outstanding = old_limit > device.coordinator_allocated_bytes
        ? old_limit - device.coordinator_allocated_bytes : 0;
    const size_t other_outstanding = state.outstanding_bytes > own_outstanding
        ? state.outstanding_bytes - own_outstanding : 0;
    const size_t available = view.free > view.reserve ? view.free - view.reserve : 0;
    size_t headroom = available > other_outstanding ? available - other_outstanding : 0;
    // Sessions that registered but have not prepared a budget still need
    // their initial share, just as in moe_cache_budget_claim().
    const int remaining_participants = std::max(state.participants - state.prepared + 1, 1);
    headroom /= (size_t)remaining_participants;
    if (session.config.budget_mb) {
        const size_t cap = session.config.budget_mb << 20;
        headroom = std::min(headroom, cap > device.coordinator_allocated_bytes
                ? cap - device.coordinator_allocated_bytes : 0);
    }
    headroom = std::min(headroom, SIZE_MAX - device.coordinator_allocated_bytes);
    device.budget_limit = device.coordinator_allocated_bytes + headroom;
    state.outstanding_bytes = other_outstanding + headroom;
    MOE_CACHE_LOG("[moe-cache] CUDA%d budget refresh bytes: phase=%s old=%zu grant=%zu owned=%zu free=%zu total=%zu reserve=%zu other_outstanding=%zu participants=%d headroom=%zu\n",
            device.physical, phase, old_limit, device.budget_limit, device.coordinator_allocated_bytes,
            view.free, view.total, view.reserve, other_outstanding, remaining_participants, headroom);
    return true;
}

static bool moe_cache_grow_device(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation, bool exact_capacity = false) {
    if (capacity >= required) {
        return true;
    }
    if (!exact_capacity && required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = exact_capacity ? required : required * 2 + 256;
    void * fresh = nullptr;
    cudaError_t error = requested <= moe_cache_allocation_headroom(device, operation, requested)
        ? cudaMalloc(&fresh, requested) : cudaErrorMemoryAllocation;
    if (error == cudaErrorMemoryAllocation && *pointer) {
        (void)cudaGetLastError();
        if (!moe_cache_cuda_ok(
                    device, cudaFree(*pointer), "scratch replacement free", true)) {
            return false;
        }
        moe_cache_budget_allocation(device, capacity, false);
        *pointer = nullptr;
        capacity = 0;
        error = requested <= moe_cache_allocation_headroom(device, operation, requested)
            ? cudaMalloc(&fresh, requested) : cudaErrorMemoryAllocation;
    }
    if (!moe_cache_cuda_ok(device, error, operation, false)) {
        return false;
    }
    if (*pointer) {
        const size_t old_capacity = capacity;
        if (!moe_cache_cuda_ok(
                    device, cudaFree(*pointer), "scratch replacement free", true)) {
            (void)cudaFree(fresh);
            return false;
        }
        moe_cache_budget_reallocation(device, old_capacity, requested);
    } else {
        moe_cache_budget_allocation(device, requested, true);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static size_t moe_cache_growth_capacity(size_t capacity, size_t required) {
    if (capacity >= required) {
        return capacity;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return 0;
    }
    return required * 2 + 256;
}

static bool moe_cache_scratch_requirements(
        int64_t n_in, int64_t n_out, moe_cache_scratch & result) {
    constexpr size_t max_rows = moe_cache_node_rows_max;
    if (n_in <= 0 || n_out <= 0 ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return false;
    }
    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    if ((uint64_t)n_in > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)n_out > SIZE_MAX / (max_rows * sizeof(float)) ||
        (uint64_t)(padded_n_in / QK8_1) >
            SIZE_MAX / (max_rows * sizeof(block_q8_1))) {
        return false;
    }

    const size_t ids_bytes = 3 * max_rows * sizeof(int32_t);
    const size_t act_bytes = max_rows * (size_t)n_in * sizeof(float);
    if (ids_bytes > SIZE_MAX - act_bytes) {
        return false;
    }
    const size_t capacities[] = {
        moe_cache_growth_capacity(0, ids_bytes + act_bytes),
        moe_cache_growth_capacity(
                0, max_rows * (size_t)(padded_n_in / QK8_1) * sizeof(block_q8_1)),
        moe_cache_growth_capacity(0, max_rows * (size_t)n_out * sizeof(float)),
    };
    for (size_t capacity : capacities) {
        if (capacity == 0) {
            return false;
        }
    }
    result = {capacities[0], capacities[1], capacities[2]};
    return true;
}

static size_t moe_cache_scratch_total(
        const moe_cache_scratch & current,
        const moe_cache_scratch * additional = nullptr) {
    const size_t capacities[] = {
        additional ? std::max(current.input, additional->input) : current.input,
        additional ? std::max(current.q8, additional->q8) : current.q8,
        additional ? std::max(current.out, additional->out) : current.out,
    };
    size_t total = 0;
    for (size_t capacity : capacities) {
        if (capacity > SIZE_MAX - total) {
            return SIZE_MAX;
        }
        total += capacity;
    }
    return total;
}

static int moe_cache_query_config(
        int automatic, size_t budget_mib, ggml_moe_cache_config * result) {
    if (!result) {
        return 0;
    }

    moe_cache_config config = moe_cache_read_config();
    if (automatic >= 0) {
        config.enabled = true;
        config.automatic = automatic != 0;
        moe_cache_apply_mode_defaults(config);
        config.min_compute_capability =
            moe_cache_min_compute_capability(config.automatic);
    }
    if (budget_mib > 0) {
        config.budget_mb = budget_mib;
    }
    if (!config.enabled || config.budget_mb > (SIZE_MAX >> 20) ||
        config.reserve_mb > (SIZE_MAX >> 20)) {
        return 0;
    }

    result->budget_bytes = config.budget_mb << 20;
    result->reserve_bytes = config.reserve_mb << 20;
    result->minimum_slab_bytes = config.minimum_slab_bytes;
    result->min_expert_bytes = config.min_expert_bytes;
    result->min_expert_explicit = config.min_expert_explicit;
    result->max_batch = config.max_batch;
    result->min_compute_capability = config.min_compute_capability;
    result->min_devices = 1; // automatic mode accepts a single device when the slab floor is met
    result->overlap_cpu_rows = config.overlap_cpu_rows;
    return 1;
}

// Backend registration object this provider registered under. Set by
// moe_cache_register; query_device matches it against the device's backend.
static const void * g_moe_cache_owner = nullptr;

static int moe_cache_query_device(
        void * opaque, const ggml_moe_cache_config * config,
        ggml_moe_cache_device_caps * result) {
    if (!opaque || !config || !result || !g_moe_cache_owner) {
        return 0;
    }

    ggml_backend_dev_t device = (ggml_backend_dev_t)opaque;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if ((const void *)reg != g_moe_cache_owner) {
        return 0;
    }

    int logical = -1;
    const size_t count = ggml_backend_reg_dev_count(reg);
    for (size_t index = 0; index < count; index++) {
        if (ggml_backend_reg_dev_get(reg, index) == device) {
            logical = (int)index;
            break;
        }
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (logical < 0 || logical >= info.device_count) {
        return 0;
    }

    const auto & cuda_device = info.devices[logical];
    if (cuda_device.cc < config->min_compute_capability) {
        return 0;
    }
    result->logical_device = logical;
    result->physical_device = cuda_device.physical_device;
    result->compute_capability = cuda_device.cc;
    result->min_expert_bytes = config->min_expert_explicit
        ? config->min_expert_bytes
        : moe_cache_default_min_expert_bytes(cuda_device.cc);
    return 1;
}

static int moe_cache_query_shape(
        int wtype, int64_t n_in, int64_t n_out, int64_t n_expert,
        size_t expert_size,
        ggml_moe_cache_shape_caps * result) {
    if (!result || n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return 0;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size < (size_t)n_out * row_size ||
        (expert_size != (size_t)n_out * row_size && expert_size % GGML_MEM_ALIGN != 0) ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return 0;
    }

    moe_cache_scratch scratch;
    if (!moe_cache_scratch_requirements(n_in, n_out, scratch)) {
        return 0;
    }
    const size_t scratch_bytes = moe_cache_scratch_total(scratch);
    expert_size = (size_t)n_out * row_size;
    const size_t pool_bytes = expert_size * moe_cache_pool_slots_min;
    if (scratch_bytes == SIZE_MAX || pool_bytes > SIZE_MAX - scratch_bytes) {
        return 0;
    }
    result->scratch_bytes = scratch_bytes;
    result->pool_bytes = pool_bytes;
    result->minimum_bytes = scratch_bytes + pool_bytes;
    return 1;
}

static bool moe_cache_grow_host(
        moe_cache_device & device, void ** pointer, size_t & capacity,
        size_t required, const char * operation) {
    if (capacity >= required) {
        return true;
    }
    if (required > (std::numeric_limits<size_t>::max() - 256) / 2) {
        return false;
    }

    const size_t requested = required * 2 + 256;
    void * fresh = nullptr;
    if (!moe_cache_cuda_ok(device, cudaMallocHost(&fresh, requested), operation, false)) {
        return false;
    }
    if (*pointer) {
        (void)cudaFreeHost(*pointer);
    }
    *pointer = fresh;
    capacity = requested;
    return true;
}

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device, cudaStream_t stream) {
    char * stage = nullptr;
    size_t stage_capacity = 0;

    for (;;) {
        moe_cache_job job;
        {
            std::unique_lock<std::mutex> lock(session->mu);
            session->cv.wait(lock, [&] {
                return session->stopping || device->dead.load() ||
                    !device->queue.empty();
            });
            if ((session->stopping || device->dead.load()) &&
                device->queue.empty()) {
                break;
            }

            job = device->queue.front();
            device->queue.pop_front();
            device->queued_bytes = job.bytes <= device->queued_bytes
                ? device->queued_bytes - job.bytes : 0;
            device->inflight = true;
            device->inflight_prefetch = (bool) job.prediction;
            device->inflight_source = job.source;
            device->inflight_bytes = job.bytes;
        }

        const int64_t copy_start = ggml_time_us();
        bool expired = false;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            expired = job.prediction && (device->urgent_transfers.load(std::memory_order_acquire) != 0 ||
                job.prediction->closed || job.prediction->target_start != 0 ||
                (double) (job.prediction->deadline - copy_start) * device->copy_bytes_per_us < job.bytes || std::any_of(device->queue.begin(), device->queue.end(),
                    [](const moe_cache_job & queued) { return !queued.prediction; }));
        }
        cudaError_t error = cudaSuccess;
        ggml_cuda_set_device(device->logical);

        if (!expired && (device->dead.load() || moe_cache_fail(*session, "insert"))) {
            error = cudaErrorUnknown;
        }
        if (!expired && error == cudaSuccess && stage_capacity < job.bytes) {
            char * fresh = nullptr;
            cudaError_t alloc_error = cudaMallocHost((void **)&fresh, job.bytes);
            if (alloc_error == cudaSuccess) {
                if (stage) {
                    (void)cudaFreeHost(stage);
                }
                stage = fresh;
                stage_capacity = job.bytes;
            } else {
                (void)cudaGetLastError();
            }
        }

        moe_cache_pool * pool = nullptr;
        char * destination = nullptr;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            if (job.pool >= 0 && job.pool < (int)device->pools.size()) {
                pool = device->pools[job.pool].get();
                if (pool->slab && job.slot >= 0 && job.slot < pool->n_slots) {
                    destination = pool->slab + (size_t)job.slot * pool->expert_size;
                }
            }
        }

        if (error == cudaSuccess && !destination) {
            error = cudaErrorInvalidValue;
        }
        {
            std::unique_lock<std::mutex> fill_lock(
                    session->fill_mu, std::defer_lock);
            if (session->config.serial_fill) {
                fill_lock.lock();
            }
            if (job.prediction) {
                std::lock_guard<std::mutex> lock(session->mu);
                job.prediction->start = ggml_time_us();
                expired |= device->urgent_transfers.load(std::memory_order_acquire) != 0 ||
                    !stage || stage_capacity < job.bytes || job.prediction->closed ||
                    job.prediction->target_start != 0 ||
                    (double) (job.prediction->deadline - ggml_time_us()) * device->copy_bytes_per_us < job.bytes ||
                    std::any_of(device->queue.begin(), device->queue.end(),
                        [](const moe_cache_job & queued) { return !queued.prediction; });
            }
            if (!expired && error == cudaSuccess && stage && stage_capacity >= job.bytes) {
                memcpy(stage, job.source, job.bytes);
                error = cudaMemcpyAsync(
                        destination, stage, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (!expired && error == cudaSuccess) {
                error = cudaMemcpy(
                        destination, job.source, job.bytes, cudaMemcpyHostToDevice);
            }
        }

        {
            std::lock_guard<std::mutex> lock(session->mu);
            const int64_t finished = ggml_time_us();
            if (!expired && error == cudaSuccess && finished > copy_start) {
                const double elapsed = (double) (finished - copy_start);
                const double rate = (double) job.bytes / elapsed;
                device->copy_bytes_per_us = device->copy_samples == 0 ? rate :
                    0.875 * device->copy_bytes_per_us + 0.125 * rate;
                device->copy_samples++;
            }
            if (!job.prediction && pool && job.slot >= 0 && job.slot < pool->n_slots &&
                    pool->slots[job.slot].generation == job.generation) {
                job.prediction = pool->slots[job.slot].prediction;
                if (job.prediction && job.prediction->start == 0) {
                    job.prediction->start = copy_start; // A queued prediction promoted to demand.
                }
            }
            if (job.prediction) {
                job.prediction->done = expired || error != cudaSuccess ? -1 : finished;
                moe_prefetch_log(*job.prediction);
            }
            device->inflight = false;
            device->inflight_prefetch = false;
            device->inflight_source = nullptr;
            device->inflight_bytes = 0;

            if (pool && job.slot >= 0 && job.slot < pool->n_slots) {
                moe_cache_slot & slot = pool->slots[job.slot];
                if (slot.state == moe_cache_slot_state::copying &&
                    slot.generation == job.generation && slot.key == job.key) {
                    if (!expired && error == cudaSuccess && (!slot.prediction || !slot.prediction->closed)) {
                        slot.state = moe_cache_slot_state::valid;
                        moe_cache_lru_push_back(*pool, job.slot);
                        device->demand_count.erase(job.key);
                        device->fills++;
                    } else {
                        moe_cache_slot_reset(*pool, job.slot, true);
                        if (!expired && error != cudaSuccess) {
                            device->fill_failures++;
                        }
                    }
                }
            }
            session->idle_cv.notify_all();
        }

        if (error != cudaSuccess) {
            moe_cache_cuda_ok(*device, error, "expert fill", true);
            moe_cache_trim_session(*session, device->physical);
        }
    }

    if (stream) {
        (void)cudaStreamSynchronize(stream);
        (void)cudaStreamDestroy(stream);
    }
    if (stage) {
        (void)cudaFreeHost(stage);
    }
}

static bool moe_cache_start_worker(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.worker_started) {
        return true;
    }
    // Create the fill stream before later pool slabs spend the remaining
    // budget. Its driver bookkeeping is real VRAM, even before the first fill.
    ggml_cuda_set_device(device.logical);
    cudaStream_t stream = nullptr;
    int least_priority = 0, greatest_priority = 0;
    cudaError_t error = cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    if (error == cudaSuccess) {
        error = cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, least_priority);
    } else {
        (void)cudaGetLastError();
        error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    }
    if (!moe_cache_cuda_ok(device, error, "fill stream reservation", false)) {
        return false;
    }
    try {
        device.worker = std::thread(moe_cache_worker, &session, &device, stream);
        device.worker_started = true;
        return true;
    } catch (...) {
        (void)cudaStreamDestroy(stream);
        device.dead.store(true);
        MOE_CACHE_LOG("[moe-cache] CUDA%d failed to start fill worker\n", device.physical);
        return false;
    }
}

static int moe_cache_find_pool(
        const moe_cache_device & device, size_t expert_size, int wtype) {
    for (int index = 0; index < (int)device.pools.size(); index++) {
        const moe_cache_pool & pool = *device.pools[index];
        if (pool.expert_size == expert_size && pool.wtype == wtype) {
            return index;
        }
    }
    return -1;
}

static bool moe_cache_prepare_budget(
        moe_cache_session & session, moe_cache_device & device) {
    if (device.budget_ready) {
        return device.budget_limit > 0;
    }
    device.budget_ready = true;

    ggml_cuda_set_device(device.logical);
    size_t free_memory = 0;
    size_t total_memory = 0;
    cudaError_t error = cudaMemGetInfo(&free_memory, &total_memory);
    if (!moe_cache_cuda_ok(device, error, "memory query", false)) {
        device.dead.store(true);
        return false;
    }

    size_t reserve_bytes = 0;
    const size_t available = moe_cache_budget_claim(
            session, device, free_memory, reserve_bytes);
    device.budget_limit = available;

    if (session.config.budget_mb > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: cap=%zu MiB granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, session.config.budget_mb, available >> 20,
                free_memory >> 20, reserve_bytes >> 20);
    } else {
        MOE_CACHE_LOG("[moe-cache] CUDA%d capacity: policy=free-minus-reserve granted=%zu MiB free=%zu MiB reserve=%zu MiB\n",
                device.physical, available >> 20, free_memory >> 20,
                reserve_bytes >> 20);
    }
    return available > 0;
}

// cudaMalloc may consume more device memory than its requested payload.
// Admission uses the actual post-allocation free bytes and keeps every other
// session's outstanding claim. Retry only the minimum expert slots needed.
static bool moe_cache_slab_preserves_reserve(
        moe_cache_device & device, size_t requested, const char * operation, size_t & deficit) {
    moe_cache_memory_view view;
    if (!moe_cache_memory_query(device, view) || view.other_outstanding > SIZE_MAX - view.reserve) {
        return false;
    }
    const size_t required_free = view.reserve + view.other_outstanding;
    deficit = view.free < required_free ? required_free - view.free : 0;
    MOE_CACHE_LOG("[moe-cache] CUDA%d slab actual bytes: operation=%s requested=%zu free_after=%zu reserve=%zu other_outstanding=%zu deficit=%zu policy=actual-allocator-padding\n",
            device.physical, operation, requested, view.free, view.reserve, view.other_outstanding, deficit);
    return true;
}

// Device table arena begin: this mandatory allocation precedes expert pools.
static bool moe_cache_table_stats_enabled() {
    const char * value = getenv("GGML_MOE_CACHE_DEVICE_STATS");
    return value && strcmp(value, "1") == 0;
}

static bool moe_cache_table_bytes_add(size_t & total, size_t bytes) {
    const size_t alignment = 256;
    if (bytes > SIZE_MAX - (alignment - 1)) {
        return false;
    }
    bytes = (bytes + alignment - 1) & ~(alignment - 1);
    if (bytes > (64u << 20) || total > (64u << 20) - bytes) {
        return false;
    }
    total += bytes;
    return true;
}

static bool moe_cache_reserve_table_arena(moe_cache_session & session, moe_cache_device & device) {
    const char * enabled = getenv("GGML_MOE_CACHE_DEVICE");
    if (!enabled || strcmp(enabled, "1") != 0 || session.devices.size() != 1) {
        return true;
    }
    size_t required = 0, entries = 0;
    const bool stats = moe_cache_table_stats_enabled();
    for (const auto & entry : device.seen_tensors) {
        const size_t count = (size_t)entry.second.n_expert;
        if (count > SIZE_MAX / sizeof(void *) || count > SIZE_MAX - entries ||
                !moe_cache_table_bytes_add(required, count * sizeof(void *)) ||
                (session.prefetch_enabled && !moe_cache_table_bytes_add(required, count * sizeof(void *))) ||
                (stats && !moe_cache_table_bytes_add(required, count))) {
            return false;
        }
        entries += count;
    }
    if (entries > SIZE_MAX / sizeof(unsigned long long) ||
            !moe_cache_table_bytes_add(required, entries * sizeof(unsigned long long)) ||
            (stats && !moe_cache_table_bytes_add(required, 3 * sizeof(unsigned long long)))) {
        return false;
    }
    if ((session.device_ready.load() && required > device.table_arena_cap) ||
            !moe_cache_grow_device(device, (void **)&device.table_arena, device.table_arena_cap,
                required, "reserved device table arena", true)) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d device table arena unavailable: requested=%zu allocated=%zu; resident experts preserved\n",
                device.physical, required, device.table_arena_cap);
        return false;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d device table arena bytes: requested=%zu allocated=%zu tensors=%zu entries=%zu policy=materialize-before-expert-pools\n",
            device.physical, required, device.table_arena_cap, device.seen_tensors.size(), entries);
    return true;
}
// Device table arena end.

static bool moe_cache_allocate_pool(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_shape & shape, size_t budget) {
    if (shape.expert_size == 0 || shape.n_tensors <= 0 || shape.n_entries == 0) {
        return false;
    }
    shape.finished = true;

    budget = std::min(budget, moe_cache_allocation_headroom(device, "new expert pool", budget));
    size_t slots_by_budget = budget / shape.expert_size;
    size_t slot_count = std::min<size_t>(
            slots_by_budget,
            (size_t)std::min<uint64_t>(shape.n_entries, INT_MAX));
    const size_t type_size = ggml_type_size((ggml_type)shape.wtype);
    if (type_size == 0 || shape.expert_size % type_size != 0) {
        return false;
    }
    const size_t stride_blocks = shape.expert_size / type_size;
    if (stride_blocks == 0) {
        return false;
    }
    slot_count = std::min(slot_count, (size_t)INT_MAX / stride_blocks);
    if (slot_count > INT_MAX) {
        slot_count = INT_MAX;
    }
    if (slot_count < moe_cache_pool_slots_min) {
        return false;
    }

    ggml_cuda_set_device(device.logical);
    char * slab = nullptr;
    cudaError_t error = cudaSuccess;
    while (slot_count >= moe_cache_pool_slots_min) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            error = cudaMalloc((void **)&slab, slot_count * shape.expert_size);
        }
        if (error == cudaSuccess) {
            size_t deficit = 0;
            const bool measured = moe_cache_slab_preserves_reserve(
                    device, slot_count * shape.expert_size, "new expert pool", deficit);
            if (measured && !deficit) {
                break;
            }
            (void)cudaFree(slab);
            slab = nullptr;
            if (!measured) {
                return false;
            }
            const size_t remove = deficit / shape.expert_size + (deficit % shape.expert_size != 0);
            slot_count -= std::min(slot_count, remove);
            continue;
        }
        (void)cudaGetLastError();
        slot_count /= 2;
    }
    if (error != cudaSuccess || !slab || slot_count < moe_cache_pool_slots_min) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d skipped %zu KiB expert pool: allocation failed\n",
                device.physical, shape.expert_size >> 10);
        return false;
    }
    const size_t allocated = slot_count * shape.expert_size;
    moe_cache_budget_allocation(device, allocated, true);

    std::unique_ptr<moe_cache_pool> pool(new (std::nothrow) moe_cache_pool());
    if (!pool) {
        (void)cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }
    try {
        pool->expert_size = shape.expert_size;
        pool->wtype = shape.wtype;
        pool->slab = slab;
        pool->n_slots = (int)slot_count;
        pool->covers_all_entries = (uint64_t)slot_count >= shape.n_entries;
        pool->slots.resize(slot_count);
        pool->free_slots.reserve(slot_count);
        pool->map.reserve(slot_count);
        for (int index = (int)slot_count - 1; index >= 0; index--) {
            pool->free_slots.push_back(index);
        }
        device.pools.push_back(std::move(pool));
    } catch (...) {
        (void)cudaFree(slab);
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }

    shape.pool = (int)device.pools.size() - 1;
    device.allocated_bytes += allocated;

    if (!moe_cache_start_worker(session, device)) {
        (void)cudaFree(device.pools.back()->slab);
        moe_cache_budget_allocation(device, allocated, false);
        device.pools.back()->slab = nullptr;
        device.pools.pop_back();
        device.allocated_bytes -= allocated;
        shape.pool = -1;
        return false;
    }

    MOE_CACHE_LOG("[moe-cache] CUDA%d pool[%d]: type=%s expert=%zu KiB slots=%zu entries=%llu coverage=%s total=%zu MiB allocated_bytes=%zu\n",
            device.physical, shape.pool, ggml_type_name((ggml_type)shape.wtype),
            shape.expert_size >> 10, slot_count,
            (unsigned long long)shape.n_entries,
            device.pools.back()->covers_all_entries ? "complete" : "partial",
            allocated >> 20, allocated);
    bool expected = false;
    if (session.enabled_announced.compare_exchange_strong(expected, true)) {
        MOE_CACHE_LOG("[moe-cache] enabled: first pool allocated on CUDA%d\n", device.physical);
    }

    return true;
}

static void moe_cache_build_pending(
        moe_cache_session & session, moe_cache_device & device) {
    if (!moe_cache_prepare_budget(session, device)) {
        for (moe_cache_shape & shape : device.shapes) {
            if (shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    // Never resize workspace that another active node is using. Do not
    // block here: session.mu is held, whereas dispatches take dispatch_mu
    // before reacquiring session.mu.
    std::unique_lock<std::mutex> workspace_lock(device.dispatch_mu, std::try_to_lock);
    if (!workspace_lock.owns_lock()) {
        return;
    }
    // Materialize the bounded, max-row runtime workspace before spending
    // the rest of the grant on experts. CUDA allocation granularity and
    // post-prompt work must not turn a first-token workspace allocation into
    // a policy failure that discards the whole expert cache.
    if (!moe_cache_grow_device(device, (void **)&device.d_input, device.d_input_cap,
                device.scratch_reserve.input, "reserved input workspace", true) ||
            !moe_cache_grow_device(device, &device.d_act_q8, device.act_q8_cap,
                device.scratch_reserve.q8, "reserved q8 workspace", true) ||
            !moe_cache_grow_device(device, (void **)&device.d_out, device.d_out_cap,
                device.scratch_reserve.out, "reserved output workspace", true)) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d pool build deferred: bounded runtime workspace unavailable; resident experts preserved\n",
                device.physical);
        return;
    }
    // Device table arena reserve before optional expert slabs.
    if (!moe_cache_reserve_table_arena(session, device)) {
        return;
    }
    if (!moe_cache_refresh_budget(session, device, "build pools")) {
        return;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d runtime workspace bytes: input=%zu q8=%zu output=%zu; policy=materialize-before-expert-pools\n",
            device.physical, device.d_input_cap, device.act_q8_cap, device.d_out_cap);
    const size_t scratch_reserve =
        moe_cache_scratch_total(device.scratch_reserve) + device.table_arena_cap;
    const size_t slab_limit =
        scratch_reserve < device.budget_limit
            ? device.budget_limit - scratch_reserve : 0;
    size_t remaining = slab_limit > device.allocated_bytes
        ? slab_limit - device.allocated_bytes : 0;
    if (remaining == 0) {
        for (moe_cache_shape & shape : device.shapes) {
            if (!shape.finished && shape.n_tensors > 0) {
                shape.finished = true;
            }
        }
        return;
    }

    std::vector<moe_cache_shape *> pending;
    long double total_weight = 0.0;
    size_t minimum_remaining = 0;
    for (moe_cache_shape & shape : device.shapes) {
        if (!shape.finished && shape.n_tensors > 0) {
            pending.push_back(&shape);
            total_weight +=
                (long double)shape.expert_size * (long double)shape.n_tensors;
            const size_t minimum = shape.expert_size * moe_cache_pool_slots_min;
            minimum_remaining = minimum <= SIZE_MAX - minimum_remaining
                ? minimum_remaining + minimum : SIZE_MAX;
        }
    }
    std::sort(pending.begin(), pending.end(), [](const auto * lhs, const auto * rhs) {
        const long double lhs_weight =
            (long double)lhs->expert_size * (long double)lhs->n_tensors;
        const long double rhs_weight =
            (long double)rhs->expert_size * (long double)rhs->n_tensors;
        return lhs_weight > rhs_weight;
    });

    auto weighted_share = [](size_t available, long double weight, long double total) {
        if (available == 0 || weight <= 0.0 || total <= 0.0) {
            return (size_t)0;
        }
        const long double value = (long double)available * weight / total;
        return value >= (long double)available ? available : (size_t)value;
    };
    const bool complete_pools = minimum_remaining <= remaining;
    for (moe_cache_shape * shape : pending) {
        const long double weight =
            (long double)shape->expert_size * (long double)shape->n_tensors;
        const size_t minimum = shape->expert_size * moe_cache_pool_slots_min;
        size_t share = weighted_share(remaining, weight, total_weight);
        if (complete_pools) {
            const size_t extra = remaining - minimum_remaining;
            const size_t bonus = weighted_share(extra, weight, total_weight);
            share = minimum + bonus;
        } else if (share < minimum && remaining >= minimum) {
            share = minimum;
        }
        const size_t before = device.allocated_bytes;
        moe_cache_allocate_pool(session, device, *shape, share);
        const size_t consumed = device.allocated_bytes - before;
        remaining = consumed <= remaining ? remaining - consumed : 0;
        total_weight -= weight;
        if (complete_pools) {
            minimum_remaining -= minimum;
        }
    }
}

static int moe_cache_discover_pool(
        moe_cache_session & session, moe_cache_device & device,
        const void * host_base, size_t tensor_size, size_t expert_size,
        int wtype, int64_t n_expert) {
    int pool = moe_cache_find_pool(device, expert_size, wtype);
    if (pool >= 0 && !device.pools[pool]->covers_all_entries) {
        return pool;
    }
    if (pool >= 0 && device.seen_tensors.find(host_base) != device.seen_tensors.end()) {
        return pool;
    }

    moe_cache_shape * shape = nullptr;
    for (moe_cache_shape & candidate : device.shapes) {
        if (candidate.expert_size == expert_size && candidate.wtype == wtype) {
            shape = &candidate;
            break;
        }
    }
    if (!shape) {
        device.shapes.push_back({expert_size, wtype, 0, 0, -1, false});
        shape = &device.shapes.back();
        shape->pool = pool;
    }

    const bool first_visit = device.seen_tensors.emplace(
            host_base,
            moe_cache_seen_tensor{tensor_size, expert_size, wtype, n_expert}).second;
    if (first_visit) {
        if (shape->n_tensors == 0 && shape->pool < 0) {
            shape->finished = false;
        }
        const uint64_t entries = (uint64_t)n_expert;
        shape->n_entries = entries <= UINT64_MAX - shape->n_entries
            ? shape->n_entries + entries : UINT64_MAX;
        shape->n_tensors++;
        device.visits_since_new_tensor = 0;
    } else {
        if (device.visits_since_new_tensor < SIZE_MAX) {
            device.visits_since_new_tensor++;
        }
    }

    if (pool >= 0) {
        device.pools[pool]->covers_all_entries =
            (uint64_t)device.pools[pool]->n_slots >= shape->n_entries;
        return pool;
    }

    if (device.visits_since_new_tensor < device.seen_tensors.size()) {
        return -1;
    }

    moe_cache_build_pending(session, device);
    return moe_cache_find_pool(device, expert_size, wtype);
}

static void moe_cache_log_stats(moe_cache_device & device) {
    size_t used = 0;
    size_t slots = 0;
    for (const auto & pool_ptr : device.pools) {
        const moe_cache_pool & pool = *pool_ptr;
        slots += pool.n_slots;
        used += pool.n_slots - pool.free_slots.size();
    }
    const long long total = device.hits + device.misses;
    MOE_CACHE_LOG("[moe-cache] CUDA%d hits=%lld/%lld (%.1f%%) used=%zu/%zu enqueued=%lld filled=%lld fill-fail=%lld evictions=%lld heat=%lld skips=%lld admission=%lld queue=%zu jobs/%zu MiB dispatch-fail=%lld collect-fail=%lld act-dedup=%lld cpu-overlap=%lld fusion=%lld/%lld pairs=%lld/%lld/%lld/%lld fusion-attempts=%lld fusion-nodes=%lld bypass=%lld\n",
            device.physical, device.hits, total,
            total ? 100.0 * (double)device.hits / (double)total : 0.0,
            used, slots, device.inserts, device.fills, device.fill_failures,
            device.evictions, device.heat_evictions, device.insert_skips,
            device.admission_skips, device.queue.size(), device.queued_bytes >> 20,
            device.dispatch_failures, device.collect_failures,
            device.activation_dedup, device.overlap_rows,
            device.fused_rows, device.fused_candidates,
            device.pair_both, device.pair_up_only,
            device.pair_gate_only, device.pair_neither,
            device.fused_attempts,
            device.fused_nodes, device.contention_bypasses);
}

static void moe_cache_log_configuration(moe_cache_session & session) {
    bool expected = false;
    if (!session.config_announced.compare_exchange_strong(expected, true)) {
        return;
    }

    const std::string budget = session.config.budget_mb > 0
        ? std::to_string(session.config.budget_mb) + " MiB cap"
        : "free-minus-reserve";
    const std::string overlap_cpu_rows = session.config.overlap_cpu_rows < 0
        ? "auto" : std::to_string(session.config.overlap_cpu_rows);
    const std::string admission = session.config.admit_after_explicit
        ? std::to_string(session.config.admit_after) + "-fixed/" +
            std::to_string(session.config.readmit_after) + "-replace"
        : "1-complete/" + std::to_string(session.config.admit_after) +
            "-partial/" + std::to_string(session.config.readmit_after) + "-replace";
    MOE_CACHE_LOG("[moe-cache] configured: mode=%s devices=%zu budget=%s reserve=%zu MiB min-slab=%zu MiB min-expert=%zu KiB max-batch=%d admit=%s hot=%d cpu-overlap=%s fills=%s\n",
            session.config.automatic ? "auto" : "on",
            session.devices.size(), budget.c_str(),
            session.config.reserve_mb,
            session.config.minimum_slab_bytes >> 20,
            session.config.min_expert_bytes >> 10,
            session.config.max_batch,
            admission.c_str(),
            session.config.hot_uses,
            overlap_cpu_rows.c_str(),
            session.config.serial_fill ? "serial" : "parallel");
}

static void * moe_cache_session_create(
        void * const * backends, int n_backends,
        const ggml_moe_cache_config * supplied_config) {
    try {
        moe_cache_config config = moe_cache_read_config();
        if (supplied_config) {
            constexpr size_t MiB = 1024 * 1024;
            if (supplied_config->budget_bytes % MiB != 0 ||
                supplied_config->reserve_bytes % MiB != 0 ||
                supplied_config->minimum_slab_bytes % MiB != 0 ||
                supplied_config->min_expert_bytes == 0 ||
                supplied_config->min_expert_explicit < 0 ||
                supplied_config->min_expert_explicit > 1 ||
                supplied_config->max_batch < 1 ||
                supplied_config->max_batch > moe_cache_batch_max ||
                supplied_config->min_devices < 1 ||
                supplied_config->min_compute_capability < 0 ||
                supplied_config->min_compute_capability > 999 ||
                supplied_config->overlap_cpu_rows < -1 ||
                supplied_config->overlap_cpu_rows > 8) {
                return nullptr;
            }
            config.enabled = true;
            // automatic iff the provider advertised the auto slab floor; device count no longer implies the mode
            config.automatic = supplied_config->minimum_slab_bytes > 0;
            config.budget_mb = supplied_config->budget_bytes / MiB;
            config.reserve_mb = supplied_config->reserve_bytes / MiB;
            config.minimum_slab_bytes = supplied_config->minimum_slab_bytes;
            config.min_expert_bytes = supplied_config->min_expert_bytes;
            config.min_expert_explicit = supplied_config->min_expert_explicit;
            config.max_batch = supplied_config->max_batch;
            config.min_compute_capability = supplied_config->min_compute_capability;
            config.overlap_cpu_rows = supplied_config->overlap_cpu_rows;
        }
        if (!config.enabled) {
            MOE_CACHE_LOG("[moe-cache] disabled by environment (GGML_CUDA_MOE_CACHE=0 or GGML_CUDA_MOE_CACHE_MODE=off)\n");
            return nullptr;
        }

        std::unique_ptr<moe_cache_session> session(new (std::nothrow) moe_cache_session());
        if (!session) {
            return nullptr;
        }
        session->config = std::move(config);

        std::unordered_set<int> seen_devices;
        size_t default_min_expert_bytes = moe_cache_expert_bytes_ampere_min;
        int minimum_capability = INT_MAX;
        for (int index = 0; index < n_backends &&
                            (int)session->devices.size() < session->config.max_devices; index++) {
            ggml_backend_t backend = (ggml_backend_t)backends[index];
            if (!backend || !ggml_backend_is_cuda(backend)) {
                continue;
            }
            ggml_backend_cuda_context * context =
                (ggml_backend_cuda_context *)backend->context;
            const int logical = context->device;
            const ggml_cuda_device_info & info = ggml_cuda_info();
            if (logical < 0 || logical >= info.device_count) {
                continue;
            }
            const int physical = info.devices[logical].physical_device;
            if (!seen_devices.insert(physical).second) {
                continue;
            }

            ggml_cuda_set_device(logical);
            const int capability = info.devices[logical].cc;
            if (capability < session->config.min_compute_capability) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d skipped: compute capability %d is below %d\n",
                        physical, capability,
                        session->config.min_compute_capability);
                continue;
            }

            default_min_expert_bytes = std::max(
                    default_min_expert_bytes,
                    moe_cache_default_min_expert_bytes(capability));
            minimum_capability = std::min(minimum_capability, capability);
            session->devices.emplace_back(new moe_cache_device(logical, physical));
        }

        if (session->devices.empty()) {
            MOE_CACHE_LOG("[moe-cache] session creation failed: no devices meet the compute capability floor (%d)\n",
                    session->config.min_compute_capability);
            return nullptr;
        }
        if (!session->config.min_expert_explicit) {
            session->config.min_expert_bytes = default_min_expert_bytes;
        }
        if (!session->config.serial_fill_explicit) {
            session->config.serial_fill =
                session->devices.size() < 2 || minimum_capability < moe_cache_cc_ampere;
        }
        if (!moe_cache_budget_register(*session)) {
            MOE_CACHE_LOG("[moe-cache] session creation failed: budget registration failed\n");
            return nullptr;
        }

        session->phase_stats_id = g_phase_stats_session_id.fetch_add(1, std::memory_order_relaxed);
        moe_cache_session * result = session.get();
        try {
            std::lock_guard<std::mutex> lock(g_registry_mu);
            g_sessions.insert(result);
            g_session_count.fetch_add(1, std::memory_order_release);
        } catch (...) {
            for (const auto & device_ptr : session->devices) {
                moe_cache_budget_unregister(*device_ptr);
            }
            throw;
        }
        session.release();
        return result;
    } catch (...) {
        MOE_CACHE_LOG("[moe-cache] failed to create cache session\n");
        return nullptr;
    }
}

static void moe_cache_cancel_queue_locked(
        moe_cache_device & device, const void * base, size_t size, bool all) {
    for (auto it = device.queue.begin(); it != device.queue.end();) {
        if (all || moe_cache_ranges_overlap(it->source, it->bytes, base, size)) {
            if (!it->prediction && it->pool >= 0 && it->pool < (int) device.pools.size()) {
                const auto & pool = *device.pools[it->pool];
                if (it->slot >= 0 && it->slot < pool.n_slots && pool.slots[it->slot].generation == it->generation) {
                    it->prediction = pool.slots[it->slot].prediction;
                }
            }
            if (it->prediction) {
                it->prediction->done = -1;
                it->prediction->closed = true;
                moe_prefetch_log(*it->prediction);
            }
            device.queued_bytes = it->bytes <= device.queued_bytes
                ? device.queued_bytes - it->bytes : 0;
            if (it->pool >= 0 && it->pool < (int)device.pools.size()) {
                moe_cache_pool & pool = *device.pools[it->pool];
                if (it->slot >= 0 && it->slot < pool.n_slots) {
                    moe_cache_slot & slot = pool.slots[it->slot];
                    if (slot.state == moe_cache_slot_state::copying &&
                        slot.generation == it->generation && slot.key == it->key) {
                        moe_cache_slot_reset(pool, it->slot, true);
                    }
                }
            }
            it = device.queue.erase(it);
        } else {
            ++it;
        }
    }
}

static void moe_cache_free_device(moe_cache_device & device) {
    ggml_cuda_set_device(device.logical);
    if (device.compute_stream) {
        (void)cudaStreamSynchronize(device.compute_stream);
    }
    for (auto & pool_ptr : device.pools) {
        if (pool_ptr->slab) {
            (void)cudaFree(pool_ptr->slab);
            moe_cache_budget_allocation(
                    device, (size_t)pool_ptr->n_slots * pool_ptr->expert_size, false);
            pool_ptr->slab = nullptr;
        }
    }
    if (device.d_input) {
        (void)cudaFree(device.d_input);
        moe_cache_budget_allocation(device, device.d_input_cap, false);
        device.d_input = nullptr;
    }
    if (device.d_act_q8) {
        (void)cudaFree(device.d_act_q8);
        moe_cache_budget_allocation(device, device.act_q8_cap, false);
        device.d_act_q8 = nullptr;
    }
    if (device.d_out) {
        (void)cudaFree(device.d_out);
        moe_cache_budget_allocation(device, device.d_out_cap, false);
        device.d_out = nullptr;
    }
    // Device table arena free after all table users have drained.
    if (device.table_arena) {
        (void)cudaFree(device.table_arena);
        moe_cache_budget_allocation(device, device.table_arena_cap, false);
        device.table_arena = nullptr;
        device.table_arena_cap = 0;
    }
    if (device.h_input) {
        (void)cudaFreeHost(device.h_input);
        device.h_input = nullptr;
    }
    if (device.h_out) {
        (void)cudaFreeHost(device.h_out);
        device.h_out = nullptr;
    }
    if (device.compute_stream) {
        (void)cudaStreamDestroy(device.compute_stream);
        device.compute_stream = nullptr;
    }
    device.pools.clear();
    device.allocated_bytes = 0;
    device.d_input_cap = 0;
    device.act_q8_cap = 0;
    device.d_out_cap = 0;
    device.h_input_cap = 0;
    device.h_out_cap = 0;
}

static void moe_cache_session_destroy(void * opaque) {
    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->stopping = true;
        for (auto & device_ptr : session->devices) {
            moe_cache_cancel_queue_locked(*device_ptr, nullptr, 0, true);
        }
        session->cv.notify_all();
        session->idle_cv.wait(lock, [&] {
            return session->active_scopes == 0 && session->active_nodes == 0;
        });
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->worker_started && device_ptr->worker.joinable()) {
            device_ptr->worker.join();
        }
    }
    {
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        if (g_sessions.erase(session) > 0) {
            g_session_count.fetch_sub(1, std::memory_order_release);
        }
    }

    for (auto & device_ptr : session->devices) {
        if (device_ptr->nodes > 0 || device_ptr->dispatch_failures > 0 ||
            device_ptr->collect_failures > 0) {
            moe_cache_log_stats(*device_ptr);
        }
    }

    moe_cache_device_release(*session);
    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        moe_cache_free_device(*device_ptr);
        moe_cache_budget_unregister(*device_ptr);
    }

    delete session;
}

// Which real candidate index the current (calib_slot, calib_round) trial
// refers to. Rotating by calib_round means candidate 0 doesn't always go
// first within a round, so any residual (post-warm-up) drift across a round
// lands on a different candidate each time instead of always the same one.
static int moe_cache_calibration_candidate_index(const moe_cache_session & session) {
    return (session.calib_slot + session.calib_round) % moe_cache_calib_candidate_count;
}

// Called under session.mu at the start of every scope (one
// ggml_backend_sched_compute_splits call, i.e. one decode step during
// generation) while overlap_cpu_rows is left automatic. Starts calibration
// on the first call and stamps the scope start time; per-node scope stats
// are accumulated by moe_cache_plan().
static void moe_cache_calibration_on_enter(moe_cache_session & session) {
    if (session.config.overlap_cpu_rows >= 0 || session.calib_done) {
        return;
    }
    if (!session.calib_started) {
        session.calib_started            = true;
        session.calib_global_warmup_done = false;
        session.calib_global_warmup_seen = 0;
        session.calib_round              = 0;
        session.calib_slot               = 0;
        session.calib_scopes_seen        = 0;
        session.calib_time_sum_ms        = 0.0;
        session.calib_time_count         = 0;
        for (int i = 0; i < moe_cache_calib_candidate_count; i++) {
            session.calib_candidate_sum_ms[i]  = 0.0;
            session.calib_candidate_samples[i] = 0;
        }
        session.calib_best_rows    = -1;
        session.calib_best_avg_ms  = -1.0;
        session.calib_current_rows = -1; // formula, for the duration of the global warm-up
    }
    session.calib_scope_start      = std::chrono::steady_clock::now();
    session.calib_scope_max_tokens = 0;
    session.calib_scope_nodes      = 0;
}

// Called under session.mu at the end of every scope. Only scopes that
// actually dispatched a single-token decode step through the cache count --
// prefill/batched scopes have very different timing and would corrupt the
// comparison, so they're silently skipped (calibration just takes longer).
static void moe_cache_calibration_on_leave(moe_cache_session & session) {
    if (session.config.overlap_cpu_rows >= 0 || session.calib_done || !session.calib_started) {
        return;
    }
    if (session.calib_scope_nodes <= 0 || session.calib_scope_max_tokens != 1) {
        return;
    }

    const double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - session.calib_scope_start).count();

    if (!session.calib_global_warmup_done) {
        session.calib_global_warmup_seen++;
        if (session.calib_global_warmup_seen < moe_cache_calib_global_warmup_scopes) {
            return;
        }
        session.calib_global_warmup_done = true;
        session.calib_current_rows = moe_cache_calib_candidates[moe_cache_calibration_candidate_index(session)];
        MOE_CACHE_LOG("[moe-cache] cpu-overlap calibration: warm-up complete, starting sweep\n");
        return;
    }

    session.calib_scopes_seen++;
    if (session.calib_scopes_seen <= moe_cache_calib_round_warmup_scopes) {
        return; // per-round settle for this candidate
    }
    session.calib_time_sum_ms += elapsed_ms;
    session.calib_time_count++;
    if (session.calib_scopes_seen < moe_cache_calib_round_warmup_scopes + moe_cache_calib_round_measured_scopes) {
        return; // still measuring this (candidate, round) trial
    }

    const int candidate_index = moe_cache_calibration_candidate_index(session);
    const int candidate_rows  = moe_cache_calib_candidates[candidate_index];
    const double avg_ms = session.calib_time_sum_ms / std::max(1, session.calib_time_count);
    session.calib_candidate_sum_ms[candidate_index]  += avg_ms;
    session.calib_candidate_samples[candidate_index] += 1;
    MOE_CACHE_LOG("[moe-cache] cpu-overlap calibration: round=%d candidate=%s avg=%.3f ms/decode\n",
            session.calib_round,
            candidate_rows < 0 ? "formula" : std::to_string(candidate_rows).c_str(), avg_ms);

    session.calib_scopes_seen = 0;
    session.calib_time_sum_ms = 0.0;
    session.calib_time_count  = 0;
    session.calib_slot++;
    if (session.calib_slot >= moe_cache_calib_candidate_count) {
        session.calib_slot = 0;
        session.calib_round++;
    }

    if (session.calib_round >= moe_cache_calib_rounds) {
        for (int i = 0; i < moe_cache_calib_candidate_count; i++) {
            if (session.calib_candidate_samples[i] <= 0) {
                continue;
            }
            const double avg = session.calib_candidate_sum_ms[i] / session.calib_candidate_samples[i];
            if (session.calib_best_avg_ms < 0.0 || avg < session.calib_best_avg_ms) {
                session.calib_best_avg_ms = avg;
                session.calib_best_rows   = moe_cache_calib_candidates[i];
            }
        }
        session.calib_current_rows = session.calib_best_rows;
        session.calib_done = true;
        MOE_CACHE_LOG("[moe-cache] cpu-overlap calibration complete: rows=%s (%.3f ms/decode avg)\n",
                session.calib_best_rows < 0 ? "formula" : std::to_string(session.calib_best_rows).c_str(),
                session.calib_best_avg_ms);
    } else {
        session.calib_current_rows = moe_cache_calib_candidates[moe_cache_calibration_candidate_index(session)];
    }
}

static void moe_cache_session_enter(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed++;
        return;
    }

    moe_cache_session * session = (moe_cache_session *)opaque;
    if (!session || session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->dormant.load()) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    if (session->stopping) {
        if (g_session_stack.empty()) {
            return;
        }
        try {
            g_session_stack.push_back({session, nullptr});
        } catch (...) {
            g_session_suppressed++;
        }
        return;
    }
    try {
        g_session_stack.push_back({session, session});
    } catch (...) {
        g_session_suppressed++;
        return;
    }
    session->active_scopes++;
    moe_cache_calibration_on_enter(*session);
}

static void moe_cache_session_leave(void * opaque) {
    if (g_session_suppressed > 0) {
        g_session_suppressed--;
        return;
    }
    moe_cache_session * expected = (moe_cache_session *)opaque;
    auto found = std::find_if(
            g_session_stack.rbegin(), g_session_stack.rend(),
            [expected](const moe_cache_scope_frame & frame) {
                return frame.requested == expected;
            });
    if (found == g_session_stack.rend()) {
        return;
    }
    moe_cache_session * active = found->active;
    g_session_stack.erase(std::next(found).base());
    if (active) {
        std::lock_guard<std::mutex> lock(active->mu);
        if (active->prefetch_enabled) {
            if (std::any_of(active->predictions.begin(), active->predictions.end(),
                    [](const std::shared_ptr<moe_prefetch_copy> & copy) { return copy->published; })) {
                ggml_cuda_set_device(active->devices[0]->logical);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (const auto & copy : active->predictions) {
                moe_prefetch_finish(*active, *copy);
                copy->closed = true;
                moe_prefetch_log(*copy);
            }
            active->predictions.clear();
            active->last_layer = -1;
            std::fill(std::begin(active->layer_start), std::end(active->layer_start), 0);
        }
        moe_cache_calibration_on_leave(*active);
        // This is outside CUDA graph capture/replay. The opt-in aggregate
        // diagnostic is separate from the default bounded adaptation below;
        // neither performs a host routing round trip at every layer.
        if (active->device_route_counts && active->device_ready.load()) {
            const uint64_t scopes = ++active->device_stats_scopes;
            if (scopes == 1 || scopes % 64 == 0) {
                ggml_cuda_set_device(active->devices[0]->logical);
                unsigned long long counts[3] = {};
                if (cudaDeviceSynchronize() == cudaSuccess &&
                        cudaMemcpy(counts, active->device_route_counts, sizeof(counts), cudaMemcpyDeviceToHost) == cudaSuccess) {
                    MOE_CACHE_LOG("[moe-cache] device routes: hits=%llu misses=%llu dispatches=%llu; graph snapshot scopes=%llu (diagnostic sync)\n",
                            counts[0], counts[1], counts[2],
                            (unsigned long long)scopes);
                } else {
                    MOE_CACHE_LOG("[moe-cache] device route snapshot unavailable\n");
                    (void) cudaGetLastError();
                }
            }
        }
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        if (active->active_scopes == 0 && active->active_nodes == 0 &&
                active->device_ready.load() && ++active->device_adapt_scopes % active->device_adapt_interval == 0) {
            moe_cache_device_adapt(*active);
        }
        active->idle_cv.notify_all();
    }
}

static void * moe_cache_begin(
        const char * name, const void * host_base, size_t expert_size,
        int64_t n_in, int64_t n_out, int wtype, int64_t n_expert,
        int64_t n_tokens, int64_t n_rows) {
    if (g_session_suppressed > 0 || g_session_stack.empty()) {
        return nullptr;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    moe_cache_log_configuration(*session);
    if (!name || !host_base ||
        !moe_cache_tensor_name_supported(name) || n_tokens < 1 ||
        expert_size < session->config.min_expert_bytes ||
        n_in <= 0 || n_out <= 0 || n_expert <= 0 ||
        !moe_cache_type_supported((ggml_type)wtype)) {
        return nullptr;
    }
    if (n_rows < n_tokens || n_rows % n_tokens != 0) {
        return nullptr;
    }
    if (n_tokens > session->config.max_batch) {
        bool expected = false;
        if (session->batch_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-token node above max-batch=%d; prompt batches normally use the CPU path, while larger decode or speculative batches will not use the cache\n",
                    (long long)n_tokens, session->config.max_batch);
        }
        return nullptr;
    }
    if (n_rows > moe_cache_node_rows_max) {
        bool expected = false;
        if (session->row_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-row node above row limit=%d; the stock CPU path remains active\n",
                    (long long)n_rows, moe_cache_node_rows_max);
        }
        return nullptr;
    }

    const size_t row_size = ggml_row_size((ggml_type)wtype, n_in);
    if (row_size == 0 || (uint64_t)n_out > SIZE_MAX / row_size ||
        expert_size < (size_t)n_out * row_size ||
        (expert_size != (size_t)n_out * row_size && expert_size % GGML_MEM_ALIGN != 0) ||
        (uint64_t)n_expert > SIZE_MAX / expert_size ||
        expert_size > SIZE_MAX / moe_cache_pool_slots_min) {
        return nullptr;
    }
    const size_t host_stride = expert_size;
    expert_size = (size_t)n_out * row_size;
    const size_t tensor_size = (size_t)(n_expert - 1) * host_stride + expert_size;
    moe_cache_scratch scratch_requirements;
    if (!moe_cache_scratch_requirements(
            n_in, n_out, scratch_requirements)) {
        return nullptr;
    }
    int layer = -1;
    const bool has_layer = moe_cache_layer_number(name, layer);

    moe_cache_device * selected = nullptr;
    int pool_index = -1;
    moe_cache_pool * pool = nullptr;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (session->stopping) {
            return nullptr;
        }

        int budget_devices = 0;
        int slab_devices = 0;
        int eligible_devices = 0;
        const size_t minimum_pool = expert_size * moe_cache_pool_slots_min;
        for (const auto & device_ptr : session->devices) {
            moe_cache_device & candidate = *device_ptr;
            if (candidate.dead.load() ||
                !moe_cache_prepare_budget(*session, candidate)) {
                continue;
            }
            budget_devices++;
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (slab_limit >= session->config.minimum_slab_bytes) {
                slab_devices++;
            }
            if (slab_limit >= minimum_pool &&
                slab_limit >= session->config.minimum_slab_bytes) {
                eligible_devices++;
            }
        }
        if (budget_devices == 0 ||
            (session->config.automatic && slab_devices < 1)) {
            if (session->config.automatic && slab_devices < 1) {
                MOE_CACHE_LOG("[moe-cache] session dormant: no device satisfies the %zu MiB automatic slab floor\n",
                        session->config.minimum_slab_bytes >> 20);
            }
            session->dormant.store(true);
            lock.unlock();
            for (const auto & device_ptr : session->devices) {
                moe_cache_trim_session(*session, device_ptr->physical);
            }
            return nullptr;
        }
        if (session->config.automatic && eligible_devices < 1) {
            return nullptr;
        }

        auto route_weight = [&](moe_cache_device & candidate) -> size_t {
            if (candidate.dead.load() || !candidate.budget_ready) {
                return 0;
            }
            const size_t reserved = moe_cache_scratch_total(
                    candidate.scratch_reserve, &scratch_requirements);
            const size_t slab_limit =
                candidate.budget_limit > reserved
                    ? candidate.budget_limit - reserved : 0;
            if (candidate.allocated_bytes > slab_limit ||
                slab_limit < session->config.minimum_slab_bytes) {
                return 0;
            }
            if (moe_cache_find_pool(candidate, expert_size, wtype) >= 0) {
                return slab_limit;
            }
            const size_t available = slab_limit - candidate.allocated_bytes;
            return available >= minimum_pool ? available : 0;
        };

        bool tensor_override = !has_layer;
        auto find_routed_device = [&](int physical) -> moe_cache_device * {
            for (const auto & device_ptr : session->devices) {
                if (device_ptr->physical == physical) {
                    return device_ptr.get();
                }
            }
            return nullptr;
        };

        auto tensor_route = session->tensor_devices.find(host_base);
        if (tensor_route != session->tensor_devices.end()) {
            selected = find_routed_device(tensor_route->second);
            if (!selected || selected->dead.load() ||
                route_weight(*selected) == 0) {
                session->tensor_devices.erase(tensor_route);
                selected = nullptr;
            } else {
                tensor_override = true;
            }
        }

        if (!selected && has_layer) {
            auto layer_route = session->layer_devices.find(layer);
            if (layer_route != session->layer_devices.end()) {
                selected = find_routed_device(layer_route->second);
                if (!selected || selected->dead.load()) {
                    session->layer_devices.erase(layer);
                    selected = nullptr;
                } else if (route_weight(*selected) == 0) {
                    selected = nullptr;
                    tensor_override = true;
                }
            }
        }

        uint64_t total_weight = 0;
        if (!selected) {
            for (const auto & device_ptr : session->devices) {
                const size_t weight = route_weight(*device_ptr);
                if (weight > UINT64_MAX - total_weight) {
                    total_weight = UINT64_MAX;
                } else {
                    total_weight += weight;
                }
            }
        }
        if (!selected && total_weight == 0) {
            return nullptr;
        }

        if (!selected && has_layer && !tensor_override) {
            long double best_score = -1.0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                size_t assigned = 0;
                for (const auto & route : session->layer_devices) {
                    assigned += route.second == candidate.physical;
                }
                const long double score =
                    (long double)weight / (long double)(assigned + 1);
                if (!selected || score > best_score) {
                    selected = &candidate;
                    best_score = score;
                }
            }
            if (selected) {
                try {
                    session->layer_devices.emplace(layer, selected->physical);
                } catch (...) {
                    return nullptr;
                }
            }
        }

        const uint64_t hash = has_layer
            ? ((uint64_t)(uint32_t)layer + 1) * 0x9e3779b97f4a7c15ULL
            : moe_cache_name_hash(name);
        if (!selected) {
            const uint64_t target = hash % total_weight;
            uint64_t cumulative = 0;
            for (const auto & device_ptr : session->devices) {
                moe_cache_device & candidate = *device_ptr;
                const size_t weight = route_weight(candidate);
                if (weight == 0) {
                    continue;
                }
                cumulative = weight > UINT64_MAX - cumulative
                    ? UINT64_MAX : cumulative + weight;
                if (target < cumulative) {
                    selected = &candidate;
                    break;
                }
            }
            if (!selected) {
                return nullptr;
            }
            try {
                if (tensor_override) {
                    session->tensor_devices.emplace(host_base, selected->physical);
                } else {
                    session->layer_devices.emplace(layer, selected->physical);
                }
            } catch (...) {
                return nullptr;
            }
        }

        const size_t selected_scratch = moe_cache_scratch_total(
                selected->scratch_reserve, &scratch_requirements);
        if (selected_scratch >= selected->budget_limit ||
            minimum_pool > selected->budget_limit - selected_scratch) {
            return nullptr;
        }
        selected->scratch_reserve.input = std::max(
                selected->scratch_reserve.input, scratch_requirements.input);
        selected->scratch_reserve.q8 = std::max(
                selected->scratch_reserve.q8, scratch_requirements.q8);
        selected->scratch_reserve.out = std::max(
                selected->scratch_reserve.out, scratch_requirements.out);

        try {
            selected->host_strides[host_base] = host_stride;
            pool_index = moe_cache_discover_pool(
                    *session, *selected, host_base, tensor_size, expert_size, wtype, n_expert);
        } catch (...) {
            MOE_CACHE_LOG("[moe-cache] disabled one node after host allocation failure\n");
            return nullptr;
        }
        if (pool_index < 0 || pool_index >= (int)selected->pools.size() ||
            !selected->pools[pool_index]->slab) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        moe_cache_session::active_source * source = nullptr;
        try {
            source = &session->active_sources[host_base];
        } catch (...) {
            return nullptr;
        }
        source->bytes = std::max(source->bytes, tensor_size);
        source->references++;
        session->active_nodes++;
    }

    moe_cache_device & device = *selected;
    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                device.dispatch_mu, std::try_to_lock);
    } catch (...) {
        device.contention_bypasses++;
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock()) {
        device.contention_bypasses++;
    }
    if (!dispatch_lock.owns_lock() || device.dead.load()) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        std::lock_guard<std::mutex> lock(session->mu);
        auto source = session->active_sources.find(host_base);
        if (source != session->active_sources.end() && --source->second.references == 0) {
            session->active_sources.erase(source);
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
        return nullptr;
    }
    node->session = session;
    node->device = &device;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = host_base;
    node->expert_size = expert_size;
    node->n_in = n_in;
    node->n_out = n_out;
    node->n_expert = n_expert;
    node->n_tokens = n_tokens;
    node->wtype = wtype;
    node->dispatch_lock = std::move(dispatch_lock);
    return node.release();
}

// The original automatic policy: bound CPU overlap work by expert bytes,
// token count, and one quarter of the node. Still used as calibration
// candidate -1 (see moe_cache_calib_candidates) and as the fallback for any
// session that never finishes calibrating.
static int moe_cache_overlap_rows_formula(
        const moe_cache_node & node, int n_ids, int n_tokens, int top_k) {
    const size_t work_budget = moe_cache_overlap_bytes_per_token * (size_t) n_tokens;
    int rows = (int) std::min<size_t>(work_budget / node.expert_size, INT_MAX);
    rows = std::max(rows, 1);
    if (top_k >= 8) {
        rows = std::max(rows, 2);
    }

    rows = std::min(rows, std::max(1, n_ids / 4));
    rows = std::min(rows, std::max(2, n_tokens));
    return std::min(rows, n_ids - 1);
}

static int moe_cache_overlap_rows(const moe_cache_node & node, int n_ids) {
    const int configured = node.session->config.overlap_cpu_rows;
    if (configured >= 0) {
        return std::min(configured, std::max(0, n_ids - 1));
    }
    if (n_ids <= 1 || node.n_tokens <= 0 || n_ids % node.n_tokens != 0) {
        return 0;
    }

    const int n_tokens = (int) node.n_tokens;
    const int top_k = n_ids / n_tokens;

    const int candidate = node.session->calib_current_rows;
    if (candidate >= 0) {
        return std::min(candidate, n_ids - 1);
    }
    return moe_cache_overlap_rows_formula(node, n_ids, n_tokens, top_k);
}

static int moe_cache_lookup_or_queue_locked(
        moe_cache_session & session, moe_cache_device & device,
        moe_cache_pool & pool, int pool_index, const void * host_base,
        int32_t expert, size_t expert_size, int & inserts_left,
        bool & wake_worker) {
    const moe_cache_key key{host_base, expert};
    auto found = pool.map.find(key);
    if (found != pool.map.end()) {
        auto & slot = pool.slots[found->second];
        if (slot.state == moe_cache_slot_state::copying && slot.prediction) {
            const auto queued = std::find_if(device.queue.begin(), device.queue.end(),
                [&](const moe_cache_job & job) { return job.key == key; });
            if (queued != device.queue.end()) {
                // Promote to demand; retain the record on the slot for outcome accounting.
                queued->prediction.reset();
                std::rotate(device.queue.begin(), queued, std::next(queued));
            }
        }
        return pool.slots[found->second].state == moe_cache_slot_state::valid
            ? found->second : -1;
    }

    if (session.device_frozen) {
        return -1;
    }
    moe_cache_demand * demand = nullptr;
    try {
        demand = &device.demand_count[key];
    } catch (...) {
        device.insert_skips++;
        return -1;
    }
    demand->expert_size = expert_size;
    if (demand->count < std::numeric_limits<uint16_t>::max()) {
        demand->count++;
    }
    const int initial_admit_after =
        pool.covers_all_entries && !session.config.admit_after_explicit
            ? 1 : session.config.admit_after;
    const int admit_after = pool.free_slots.empty()
        ? std::max(initial_admit_after, session.config.readmit_after)
        : initial_admit_after;
    if (demand->count < admit_after) {
        device.admission_skips++;
        return -1;
    }
    const size_t queue_limit = session.config.queue_mb << 20;
    if (inserts_left <= 0 || (int)device.queue.size() >= session.config.queue_max ||
        expert_size > queue_limit - std::min(queue_limit, device.queued_bytes)) {
        device.insert_skips++;
        return -1;
    }

    int slot_index = -1;
    if (!pool.free_slots.empty()) {
        slot_index = pool.free_slots.back();
        pool.free_slots.pop_back();
    } else {
        // Prefer the least-recently-used slot whose uses is at or below the hot
        // threshold, so a frequently-used (hot) expert stays resident.
        int candidate = -1;
        int fallback = -1;
        for (int c = pool.lru_head; c >= 0; c = pool.slots[c].next) {
            moe_cache_slot & s = pool.slots[c];
            if (s.readers > 0 || s.staging) {
                continue;
            }
            if (fallback < 0) {
                fallback = c;
            }
            if ((int)s.uses <= session.config.hot_uses) {
                candidate = c;
                break;
            }
        }
        if (candidate < 0) {
            candidate = fallback;
        }
        if (candidate < 0) {
            device.insert_skips++;
            return -1;
        }
        slot_index = candidate;
        const bool sacrificed_hot =
            (int)pool.slots[slot_index].uses > session.config.hot_uses;
        moe_cache_slot_reset(pool, slot_index, false);
        device.evictions++;
        if (sacrificed_hot) {
            device.heat_evictions++;
        }
    }

    moe_cache_slot & slot = pool.slots[slot_index];
    slot.key = key;
    slot.generation++;
    slot.readers = 0;
    slot.state = moe_cache_slot_state::copying;
    try {
        const auto inserted = pool.map.emplace(key, slot_index);
        if (!inserted.second) {
            moe_cache_slot_reset(pool, slot_index, true);
            device.insert_skips++;
            return -1;
        }

        const void * source =
            (const char *)host_base + (size_t)expert * moe_cache_host_stride(device, host_base, expert_size);
        const auto first_prediction = std::find_if(device.queue.begin(), device.queue.end(),
            [](const moe_cache_job & job) { return (bool) job.prediction; });
        device.queue.insert(first_prediction, {
                pool_index, slot_index, slot.generation,
                key, source, expert_size, nullptr});
        device.queued_bytes += expert_size;
    } catch (...) {
        moe_cache_slot_reset(pool, slot_index, true);
        device.insert_skips++;
        return -1;
    }
    device.inserts++;
    inserts_left--;
    wake_worker = true;
    return -1;
}

static int moe_cache_plan(
        void * opaque, const int32_t * ids, int n_ids, int32_t * slot_indices) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    if (!node || !ids || !slot_indices || n_ids < 0 ||
        n_ids > moe_cache_node_rows_max || node->planned) {
        return 0;
    }
    node->planned = true;
    for (int index = 0; index < n_ids; index++) {
        slot_indices[index] = -1;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    int hits = 0;
    int inserts_left = session.config.inserts_per_plan;
    bool wake_worker = false;

    std::unique_lock<std::mutex> lock(session.mu);
    if (session.stopping) {
        return 0;
    }
    if (session.calib_started && !session.calib_done) {
        session.calib_scope_max_tokens = std::max(session.calib_scope_max_tokens, node->n_tokens);
        session.calib_scope_nodes++;
    }
    for (int index = 0; index < n_ids; index++) {
        const int32_t expert = ids[index];
        if (expert < 0 || expert >= node->n_expert || device.dead.load()) {
            continue;
        }

        const int slot_index = moe_cache_lookup_or_queue_locked(
                session, device, pool, node->pool_index,
                node->host_base, expert, node->expert_size,
                inserts_left, wake_worker);
        if (slot_index < 0) {
            device.misses++;
            continue;
        }

        moe_cache_slot & slot = pool.slots[slot_index];
        slot.readers++;
        slot.uses++;
        moe_cache_lru_remove(pool, slot_index);
        moe_cache_lru_push_back(pool, slot_index);
        node->pins[node->n_pins++] = {&pool, slot_index};
        slot_indices[index] = slot_index;
        device.hits++;
        hits++;
    }
    if (hits == n_ids) {
        int rows = moe_cache_overlap_rows(*node, n_ids);
        for (int index = n_ids - 1; index >= 0 && rows > 0; index--) {
            if (slot_indices[index] < 0 || node->n_pins <= 0) {
                continue;
            }
            const moe_cache_pin & pin = node->pins[node->n_pins - 1];
            if (pin.slot != slot_indices[index]) {
                continue;
            }
            moe_cache_slot & slot = pin.pool->slots[pin.slot];
            if (slot.readers > 0) {
                slot.readers--;
            }
            node->n_pins--;
            slot_indices[index] = -1;
            hits--;
            rows--;
            device.overlap_rows++;
        }
    }
    device.nodes++;
    lock.unlock();
    if (wake_worker) {
        session.cv.notify_all();
    }
    return hits;
}

static int moe_cache_dispatch_internal(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows,
        moe_cache_pool * gate_pool, const int32_t * gate_slot_indices,
        int glu_op, float up_min, float up_max,
        float gate_min, float gate_max) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    const bool fused = gate_pool != nullptr;
    if (!node || !node->planned || node->dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        n_hits * (fused ? 2 : 1) != node->n_pins ||
        !slot_indices || !act_rows ||
        (fused && (!gate_slot_indices ||
                   gate_pool->wtype != wtype ||
                   gate_pool->expert_size != node->expert_size ||
                   glu_op != GGML_GLU_OP_SWIGLU ||
                   std::isnan(up_min) || std::isnan(up_max) ||
                   std::isnan(gate_min) || std::isnan(gate_max) ||
                   up_min > up_max || gate_min > gate_max)) ||
        wtype != node->wtype || n_in != node->n_in || n_out != node->n_out ||
        n_in > INT_MAX || n_out > INT_MAX ||
        n_in > INT64_MAX - (MATRIX_ROW_PADDING - 1)) {
        return 0;
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    moe_cache_pool & pool = *node->pool;
    if (device.dead.load() || moe_cache_fail(session, "dispatch")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    ggml_cuda_set_device(device.logical);
    if (!device.compute_stream) {
        if (!moe_cache_cuda_ok(device,
                    cudaStreamCreateWithFlags(&device.compute_stream, cudaStreamNonBlocking),
                    "compute stream creation", true)) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
    }

    const float * unique_act_rows[moe_cache_node_rows_max];
    int32_t activation_indices[moe_cache_node_rows_max];
    int activation_rows = 0;
    bool modulo_activation = true;
    for (int index = 0; index < n_hits; index++) {
        if (slot_indices[index] < 0 || slot_indices[index] >= pool.n_slots ||
            (fused && (gate_slot_indices[index] < 0 ||
                       gate_slot_indices[index] >= gate_pool->n_slots)) ||
            !act_rows[index]) {
            return 0;
        }
        int activation = 0;
        while (activation < activation_rows &&
               unique_act_rows[activation] != act_rows[index]) {
            activation++;
        }
        if (activation == activation_rows) {
            unique_act_rows[activation_rows++] = act_rows[index];
        }
        activation_indices[index] = activation;
        modulo_activation &= activation == index % activation_rows;
    }

    const int64_t padded_n_in =
        ((n_in + MATRIX_ROW_PADDING - 1) / MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
    const size_t type_size = ggml_type_size((ggml_type)wtype);
    if (type_size == 0 || node->expert_size % type_size != 0 ||
        ggml_row_size((ggml_type)wtype, n_in) % type_size != 0 ||
        node->expert_size / type_size > INT_MAX ||
        ggml_row_size((ggml_type)wtype, n_in) / type_size > INT_MAX ||
        padded_n_in / QK8_1 > INT_MAX ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) > INT_MAX ||
        (uint64_t)n_out * n_hits > INT_MAX ||
        (uint64_t)(node->expert_size / type_size) * pool.n_slots > INT_MAX) {
        return 0;
    }

    if (n_in > INT64_MAX / activation_rows ||
        (uint64_t)n_in * activation_rows > SIZE_MAX / sizeof(float) ||
        n_out > INT64_MAX / n_hits ||
        (uint64_t)n_out * n_hits > SIZE_MAX / sizeof(float) ||
        (uint64_t)activation_rows * (padded_n_in / QK8_1) >
            SIZE_MAX / sizeof(block_q8_1)) {
        return 0;
    }
    const bool use_activation_map =
        session.config.force_dedicated_mmv || !modulo_activation;
    const size_t id_arrays = 1 + (fused ? 1 : 0) +
        (use_activation_map ? 1 : 0);
    const size_t ids_bytes = (size_t)n_hits * sizeof(int32_t) * id_arrays;
    const size_t act_bytes = (size_t)activation_rows * n_in * sizeof(float);
    if (ids_bytes > SIZE_MAX - act_bytes) {
        return 0;
    }
    const size_t input_bytes = ids_bytes + act_bytes;
    const size_t q8_bytes =
        (size_t)activation_rows * (padded_n_in / QK8_1) * sizeof(block_q8_1);
    const size_t out_bytes = (size_t)n_hits * n_out * sizeof(float);

    const size_t desired_caps[] = {
        moe_cache_growth_capacity(device.d_input_cap, input_bytes),
        moe_cache_growth_capacity(device.act_q8_cap, q8_bytes),
        moe_cache_growth_capacity(device.d_out_cap, out_bytes),
    };
    size_t scratch_bytes = 0;
    for (size_t capacity : desired_caps) {
        if (capacity == 0 || capacity > SIZE_MAX - scratch_bytes) {
            std::lock_guard<std::mutex> lock(session.mu);
            device.dispatch_failures++;
            return 0;
        }
        scratch_bytes += capacity;
    }
    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (scratch_bytes > device.budget_limit ||
            device.allocated_bytes > device.budget_limit - scratch_bytes) {
            device.dispatch_failures++;
            return 0;
        }
    }

    if (!moe_cache_grow_host(device, (void **)&device.h_input, device.h_input_cap,
                             input_bytes, "input host allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_input, device.d_input_cap,
                               input_bytes, "input device allocation") ||
        !moe_cache_grow_device(device, &device.d_act_q8, device.act_q8_cap,
                               q8_bytes, "q8 activation allocation") ||
        !moe_cache_grow_device(device, (void **)&device.d_out, device.d_out_cap,
                               out_bytes, "output device allocation") ||
        !moe_cache_grow_host(device, (void **)&device.h_out, device.h_out_cap,
                             out_bytes, "output host allocation")) {
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        // Workspace pressure is a per-node fallback, not invalid expert
        // data. Preserve the resident cache for subsequent exact dispatches.
        return 0;
    }

    int32_t * h_ids = (int32_t *)device.h_input;
    float * h_act = (float *)(device.h_input + ids_bytes);
    int32_t * d_ids = (int32_t *)device.d_input;
    float * d_act = (float *)(device.d_input + ids_bytes);
    for (int index = 0; index < n_hits; index++) {
        h_ids[index] = slot_indices[index];
        if (fused) {
            h_ids[n_hits + index] = gate_slot_indices[index];
        }
        if (use_activation_map) {
            h_ids[(fused ? 2 : 1) * n_hits + index] =
                activation_indices[index];
        }
    }
    for (int index = 0; index < activation_rows; index++) {
        memcpy(h_act + (size_t)index * n_in,
               unique_act_rows[index], n_in * sizeof(float));
    }

    (void)cudaGetLastError();
    bool ok =
        moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.d_input, device.h_input, input_bytes,
                cudaMemcpyHostToDevice, device.compute_stream), "input upload", true);
    if (ok) {
        quantize_row_q8_1_cuda(
                d_act, nullptr, device.d_act_q8, (ggml_type)wtype,
                n_in, n_in, (int64_t)activation_rows * n_in,
                (int64_t)activation_rows * n_in, padded_n_in,
                activation_rows, 1, 1, device.compute_stream);
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "activation quantization", true);
    }
    if (ok) {
        if (fused) {
            ggml_cuda_moe_cache_mmv_fused(
                    pool.slab, gate_pool->slab, (ggml_type)wtype,
                    (const char *)device.d_act_q8,
                    d_ids, d_ids + n_hits,
                    use_activation_map ? d_ids + 2 * n_hits : nullptr,
                    device.d_out, n_in, n_out,
                    (int64_t)pool.expert_size, n_hits, activation_rows,
                    up_min, up_max, gate_min, gate_max,
                    device.compute_stream);
        } else {
            ggml_cuda_moe_cache_mmv(
                    pool.slab, (ggml_type)wtype,
                    (const char *)device.d_act_q8, d_ids,
                    use_activation_map ? d_ids + n_hits : nullptr,
                    device.d_out, n_in, n_out, pool.n_slots,
                    (int64_t)pool.expert_size, n_hits, activation_rows,
                    device.compute_stream);
        }
        ok = moe_cache_cuda_ok(
                device, cudaPeekAtLastError(), "expert matvec launch", true);
    }

    if (!ok) {
        (void)cudaStreamSynchronize(device.compute_stream);
        std::lock_guard<std::mutex> lock(session.mu);
        device.dispatch_failures++;
        return 0;
    }

    device.activation_dedup += n_hits - activation_rows;
    node->dispatched = true;
    return 1;
}

static int moe_cache_dispatch(
        void * opaque, int wtype, int64_t n_in, int64_t n_out, int n_hits,
        const int32_t * slot_indices, const float * const * act_rows) {
    return moe_cache_dispatch_internal(
            opaque, wtype, n_in, n_out, n_hits,
            slot_indices, act_rows, nullptr, nullptr, -1,
            0.0f, 0.0f, 0.0f, 0.0f);
}

static int moe_cache_collect(
        void * opaque, int n_hits, float * const * dst_rows, int64_t n_out) {
    moe_cache_node * node = (moe_cache_node *)opaque;
    const int pins_per_hit = node && node->host_base2 ? 2 : 1;
    if (!node || !node->dispatched || n_hits <= 0 ||
        n_hits > moe_cache_node_rows_max ||
        n_hits * pins_per_hit != node->n_pins ||
        !dst_rows || n_out != node->n_out) {
        return 0;
    }
    for (int index = 0; index < n_hits; index++) {
        if (!dst_rows[index]) {
            return 0;
        }
    }

    moe_cache_session & session = *node->session;
    moe_cache_device & device = *node->device;
    ggml_cuda_set_device(device.logical);

    bool ok = !device.dead.load();
    if (moe_cache_fail(session, "collect")) {
        ok = false;
    }
    const size_t bytes = (size_t)n_hits * n_out * sizeof(float);
    if (ok) {
        ok = moe_cache_cuda_ok(device, cudaMemcpyAsync(
                device.h_out, device.d_out, bytes,
                cudaMemcpyDeviceToHost, device.compute_stream), "output download", true);
    }
    if (ok) {
        ok = moe_cache_cuda_ok(
                device, cudaStreamSynchronize(device.compute_stream),
                "output synchronization", true);
    } else {
        (void)cudaStreamSynchronize(device.compute_stream);
    }
    node->dispatched = false;

    if (ok) {
        for (int index = 0; index < n_hits; index++) {
            memcpy(dst_rows[index], device.h_out + (size_t)index * n_out,
                   n_out * sizeof(float));
        }
    }

    {
        std::lock_guard<std::mutex> lock(session.mu);
        if (!ok) {
            device.collect_failures++;
        }
        if (ok) {
            for (int i = 0; i < node->n_pins; ++i) {
                auto & slot = node->pins[i].pool->slots[node->pins[i].slot];
                if (slot.prediction && !slot.prediction->closed) {
                    slot.prediction->used = true;
                }
            }
        }
        device.collect_calls++;
        if (session.config.stats_every > 0 &&
            device.collect_calls % session.config.stats_every == 0) {
            moe_cache_log_stats(device);
        }
    }
    return ok ? 1 : 0;
}

static void moe_cache_end(void * opaque) {
    std::unique_ptr<moe_cache_node> node((moe_cache_node *)opaque);
    if (!node) {
        return;
    }

    if (node->dispatched) {
        ggml_cuda_set_device(node->device->logical);
        moe_cache_cuda_ok(
                *node->device, cudaStreamSynchronize(node->device->compute_stream),
                "end synchronization", true);
        node->dispatched = false;
    }

    moe_cache_session & session = *node->session;
    const bool trim = node->device->dead.load();
    {
        std::lock_guard<std::mutex> lock(session.mu);
        for (int index = 0; index < node->n_pins; index++) {
            const moe_cache_pin & pin = node->pins[index];
            if (pin.pool && pin.slot >= 0 && pin.slot < pin.pool->n_slots) {
                moe_cache_slot & slot = pin.pool->slots[pin.slot];
                if (slot.readers > 0) {
                    slot.readers--;
                }
            }
        }
        for (const void * base : {node->host_base, node->host_base2}) {
            if (!base) {
                continue;
            }
            auto source = session.active_sources.find(base);
            if (source != session.active_sources.end() &&
                --source->second.references == 0) {
                session.active_sources.erase(source);
            }
        }
        session.active_nodes--;
        session.idle_cv.notify_all();
    }
    if (trim && node->dispatch_lock.owns_lock()) {
        node->dispatch_lock.unlock();
        moe_cache_trim_session(session, node->device->physical);
    }
}

static void * moe_cache_fused_begin(
        const ggml_moe_cache_tensor_desc * up,
        const ggml_moe_cache_tensor_desc * gate,
        int glu_op, float up_min, float up_max,
        float gate_min, float gate_max,
        const int32_t * ids, int n_ids, int64_t n_tokens,
        const float * const * act_rows, uint64_t * hit_mask) {
    if (hit_mask) {
        *hit_mask = 0;
    }
    if (g_session_suppressed > 0 || g_session_stack.empty() ||
        !up || !gate || !ids || !act_rows || !hit_mask ||
        !up->name || !gate->name || !up->data || !gate->data ||
        up->data == gate->data ||
        !moe_cache_tensor_name_supported(up->name) ||
        !moe_cache_tensor_name_supported(gate->name) ||
        glu_op != GGML_GLU_OP_SWIGLU ||
        std::isnan(up_min) || std::isnan(up_max) ||
        std::isnan(gate_min) || std::isnan(gate_max) ||
        up_min > up_max || gate_min > gate_max ||
        n_ids < 1 || n_ids > moe_cache_node_rows_max ||
        n_tokens < 1 || n_tokens > n_ids || n_ids % n_tokens != 0 ||
        up->expert_size != gate->expert_size ||
        up->n_in != gate->n_in || up->n_out != gate->n_out ||
        up->n_expert != gate->n_expert || up->type != gate->type ||
        up->n_in <= 0 || up->n_out <= 0 ||
        up->n_expert <= 0 ||
        up->type == GGML_TYPE_NVFP4 ||
        !moe_cache_type_supported((ggml_type)up->type)) {
        return nullptr;
    }
    for (int index = 0; index < n_ids; index++) {
        if (!act_rows[index]) {
            return nullptr;
        }
    }

    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return nullptr;
    }
    moe_cache_log_configuration(*session);
    if (up->expert_size < session->config.min_expert_bytes) {
        return nullptr;
    }
    if (n_tokens > session->config.max_batch) {
        bool expected = false;
        if (session->batch_bypass_announced.compare_exchange_strong(expected, true)) {
            MOE_CACHE_LOG("[moe-cache] bypassing %lld-token fused node above max-batch=%d; increase the limit only when decode or speculative batches benefit on this hardware\n",
                    (long long)n_tokens, session->config.max_batch);
        }
        return nullptr;
    }

    const size_t up_stride = up->expert_size;
    const size_t gate_stride = gate->expert_size;
    const size_t row_size = ggml_row_size((ggml_type)up->type, up->n_in);
    if (row_size == 0 || (uint64_t)up->n_out > SIZE_MAX / row_size) {
        return nullptr;
    }
    const size_t expert_bytes = row_size * (size_t)up->n_out;
    if (up_stride < expert_bytes || gate_stride < expert_bytes ||
        (up_stride != expert_bytes && up_stride % GGML_MEM_ALIGN != 0) ||
        (gate_stride != expert_bytes && gate_stride % GGML_MEM_ALIGN != 0) ||
        (uint64_t)up->n_expert > SIZE_MAX / std::max(up_stride, gate_stride)) {
        return nullptr;
    }
    ggml_moe_cache_tensor_desc up_compact = *up, gate_compact = *gate;
    up_compact.expert_size = gate_compact.expert_size = expert_bytes;
    up = &up_compact;
    gate = &gate_compact;
    const size_t tensor_size = (size_t)(up->n_expert - 1) * std::max(up_stride, gate_stride) + expert_bytes;

    int up_layer = -1;
    int gate_layer = -1;
    if (!moe_cache_layer_number(up->name, up_layer) ||
        !moe_cache_layer_number(gate->name, gate_layer) ||
        up_layer != gate_layer) {
        return nullptr;
    }

    std::unique_ptr<moe_cache_node> node(new (std::nothrow) moe_cache_node());
    if (!node) {
        return nullptr;
    }

    moe_cache_device * selected = nullptr;
    moe_cache_pool * pool = nullptr;
    int pool_index = -1;
    {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->stopping || session->dormant) {
            return nullptr;
        }

        auto route_for = [&](const void * tensor) {
            const auto tensor_route = session->tensor_devices.find(tensor);
            if (tensor_route != session->tensor_devices.end()) {
                return tensor_route->second;
            }
            const auto layer_route = session->layer_devices.find(up_layer);
            return layer_route != session->layer_devices.end()
                ? layer_route->second : -1;
        };
        const int up_device = route_for(up->data);
        const int gate_device = route_for(gate->data);
        if (up_device < 0 || up_device != gate_device) {
            return nullptr;
        }
        for (const auto & device_ptr : session->devices) {
            if (device_ptr->physical == up_device) {
                selected = device_ptr.get();
                break;
            }
        }
        if (!selected || selected->dead.load()) {
            return nullptr;
        }
        selected->host_strides[up->data] = up_stride;
        selected->host_strides[gate->data] = gate_stride;
        pool_index = moe_cache_find_pool(
                *selected, up->expert_size, up->type);
        if (pool_index < 0 || pool_index >= (int)selected->pools.size()) {
            return nullptr;
        }
        pool = selected->pools[pool_index].get();
        if (!pool || !pool->slab) {
            return nullptr;
        }

        auto acquire_source = [&](const void * base) {
            try {
                moe_cache_session::active_source & source =
                    session->active_sources[base];
                source.bytes = std::max(source.bytes, tensor_size);
                source.references++;
                return true;
            } catch (...) {
                return false;
            }
        };
        if (!acquire_source(up->data)) {
            return nullptr;
        }
        if (!acquire_source(gate->data)) {
            auto source = session->active_sources.find(up->data);
            if (source != session->active_sources.end() &&
                --source->second.references == 0) {
                session->active_sources.erase(source);
            }
            return nullptr;
        }
        session->active_nodes++;
    }

    auto release_active = [&] {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const void * base : {up->data, gate->data}) {
            auto source = session->active_sources.find(base);
            if (source != session->active_sources.end() &&
                --source->second.references == 0) {
                session->active_sources.erase(source);
            }
        }
        session->active_nodes--;
        session->idle_cv.notify_all();
    };

    std::unique_lock<std::mutex> dispatch_lock;
    try {
        dispatch_lock = std::unique_lock<std::mutex>(
                selected->dispatch_mu, std::try_to_lock);
    } catch (...) {
        selected->contention_bypasses++;
        release_active();
        return nullptr;
    }
    if (!dispatch_lock.owns_lock()) {
        selected->contention_bypasses++;
    }
    if (!dispatch_lock.owns_lock() || selected->dead.load()) {
        release_active();
        return nullptr;
    }

    node->session = session;
    node->device = selected;
    node->pool = pool;
    node->pool_index = pool_index;
    node->host_base = up->data;
    node->host_base2 = gate->data;
    node->expert_size = up->expert_size;
    node->n_in = up->n_in;
    node->n_out = up->n_out;
    node->n_expert = up->n_expert;
    node->n_tokens = n_tokens;
    node->wtype = up->type;
    node->dispatch_lock = std::move(dispatch_lock);
    node->planned = true;

    int32_t up_slots[moe_cache_node_rows_max];
    int32_t gate_slots[moe_cache_node_rows_max];
    int32_t resident_up[moe_cache_node_rows_max];
    int32_t resident_gate[moe_cache_node_rows_max];
    const float * hit_acts[moe_cache_node_rows_max];
    uint64_t mask = 0;
    int hits = 0;
    int inserts_left = session->config.inserts_per_plan;
    bool wake_worker = false;
    {
        std::unique_lock<std::mutex> lock(session->mu);
        if (!session->stopping && !selected->dead.load()) {
            for (int index = 0; index < n_ids; index++) {
                resident_up[index] = -1;
                resident_gate[index] = -1;
                const int32_t expert = ids[index];
                if (expert < 0 || expert >= up->n_expert) {
                    continue;
                }
                const auto up_found = pool->map.find({up->data, expert});
                const auto gate_found = pool->map.find({gate->data, expert});
                if (up_found != pool->map.end() &&
                    pool->slots[up_found->second].state == moe_cache_slot_state::valid) {
                    resident_up[index] = up_found->second;
                }
                if (gate_found != pool->map.end() &&
                    pool->slots[gate_found->second].state == moe_cache_slot_state::valid) {
                    resident_gate[index] = gate_found->second;
                }
                if (resident_up[index] >= 0 && resident_gate[index] >= 0) {
                    selected->pair_both++;
                } else if (resident_up[index] >= 0) {
                    selected->pair_up_only++;
                } else if (resident_gate[index] >= 0) {
                    selected->pair_gate_only++;
                } else {
                    selected->pair_neither++;
                }
            }

            for (int index = 0; index < n_ids; index++) {
                if (resident_up[index] < 0 || resident_gate[index] < 0) {
                    continue;
                }
                moe_cache_slot & up_slot = pool->slots[resident_up[index]];
                moe_cache_slot & gate_slot = pool->slots[resident_gate[index]];
                up_slot.readers++;
                up_slot.uses++;
                gate_slot.readers++;
                gate_slot.uses++;
                moe_cache_lru_remove(*pool, resident_up[index]);
                moe_cache_lru_push_back(*pool, resident_up[index]);
                moe_cache_lru_remove(*pool, resident_gate[index]);
                moe_cache_lru_push_back(*pool, resident_gate[index]);
                node->pins[node->n_pins++] = {pool, resident_up[index]};
                node->pins[node->n_pins++] = {pool, resident_gate[index]};
                up_slots[hits] = resident_up[index];
                gate_slots[hits] = resident_gate[index];
                hit_acts[hits] = act_rows[index];
                mask |= UINT64_C(1) << index;
                selected->hits += 2;
                hits++;
            }

            for (int index = 0; index < n_ids && hits > 0; index++) {
                if (mask & (UINT64_C(1) << index)) {
                    continue;
                }
                const int32_t expert = ids[index];
                if (expert < 0 || expert >= up->n_expert) {
                    selected->misses += 2;
                    continue;
                }

                const int up_slot = moe_cache_lookup_or_queue_locked(
                        *session, *selected, *pool, pool_index,
                        up->data, expert, up->expert_size,
                        inserts_left, wake_worker);
                if (up_slot >= 0) {
                    moe_cache_slot & slot = pool->slots[up_slot];
                    slot.readers++;
                    slot.uses++;
                    moe_cache_lru_remove(*pool, up_slot);
                    moe_cache_lru_push_back(*pool, up_slot);
                    selected->hits++;
                } else {
                    selected->misses++;
                }

                const int gate_slot = moe_cache_lookup_or_queue_locked(
                        *session, *selected, *pool, pool_index,
                        gate->data, expert, gate->expert_size,
                        inserts_left, wake_worker);
                if (gate_slot >= 0) {
                    moe_cache_slot & slot = pool->slots[gate_slot];
                    slot.readers++;
                    slot.uses++;
                    moe_cache_lru_remove(*pool, gate_slot);
                    moe_cache_lru_push_back(*pool, gate_slot);
                    selected->hits++;
                } else {
                    selected->misses++;
                }

                if (up_slot >= 0) {
                    pool->slots[up_slot].readers--;
                }
                if (gate_slot >= 0) {
                    pool->slots[gate_slot].readers--;
                }
            }
            if (hits > 0) {
                selected->nodes++;
                selected->fused_attempts++;
            }
        }
    }
    if (wake_worker) {
        session->cv.notify_all();
    }

    if (hits == 0 || !moe_cache_dispatch_internal(
            node.get(), up->type, up->n_in, up->n_out, hits,
            up_slots, hit_acts, pool, gate_slots, glu_op,
            up_min, up_max, gate_min, gate_max)) {
        moe_cache_end(node.release());
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(session->mu);
        selected->fused_rows += hits;
        selected->fused_candidates += n_ids;
        selected->fused_nodes++;
    }

    *hit_mask = mask;
    return node.release();
}

static void moe_cache_invalidate_session(
        moe_cache_session & session, const void * base, size_t size) {
    std::unique_lock<std::mutex> lock(session.mu);
    for (const auto & entry : session.mapped_sources) {
        const auto & d = entry.second;
        if (moe_cache_ranges_overlap(d.data, d.expert_size * (size_t)d.n_expert, base, size)) {
            session.idle_cv.wait(lock, [&] { return session.active_scopes == 0; });
            MOE_CACHE_LOG("[moe-cache] retained source mapping invalidated: base=%p bytes=%zu\n", d.data,
                    d.expert_size * (size_t)d.n_expert);
            moe_cache_device_release(session);
            session.device_attempted = false;
            break;
        }
    }
    session.tensor_devices.erase(base);
    for (auto it = session.profile_counts.begin(); it != session.profile_counts.end();) {
        if (moe_cache_ranges_overlap(it->first.tensor, 1, base, size)) {
            it = session.profile_counts.erase(it);
        } else {
            ++it;
        }
    }
    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
    }

    session.idle_cv.wait(lock, [&] {
        for (const auto & active : session.active_sources) {
            if (active.second.references > 0 &&
                moe_cache_ranges_overlap(active.first, active.second.bytes, base, size)) {
                return false;
            }
        }
        for (const auto & device_ptr : session.devices) {
            if (device_ptr->inflight &&
                moe_cache_ranges_overlap(
                    device_ptr->inflight_source, device_ptr->inflight_bytes, base, size)) {
                return false;
            }
        }
        return true;
    });

    for (auto & device_ptr : session.devices) {
        moe_cache_cancel_queue_locked(*device_ptr, base, size, false);
        moe_cache_device & device = *device_ptr;
        for (auto & pool_ptr : device.pools) {
            moe_cache_pool & pool = *pool_ptr;
            for (int index = 0; index < pool.n_slots; index++) {
                moe_cache_slot & slot = pool.slots[index];
                const void * source = slot.key.expert >= 0
                    ? (const char *)slot.key.tensor +
                        (size_t)slot.key.expert * moe_cache_host_stride(device, slot.key.tensor, pool.expert_size)
                    : nullptr;
                if (slot.state != moe_cache_slot_state::free &&
                    moe_cache_ranges_overlap(source, pool.expert_size, base, size)) {
                    moe_cache_slot_reset(pool, index, true);
                }
            }
        }
        for (auto it = device.seen_tensors.begin(); it != device.seen_tensors.end();) {
            if (moe_cache_ranges_overlap(it->first, it->second.bytes, base, size)) {
                session.tensor_devices.erase(it->first);
                for (moe_cache_shape & shape : device.shapes) {
                    if (shape.expert_size == it->second.expert_size &&
                        shape.wtype == it->second.wtype) {
                        const uint64_t entries = (uint64_t)it->second.n_expert;
                        shape.n_entries = entries <= shape.n_entries
                            ? shape.n_entries - entries : 0;
                        shape.n_tensors = std::max<int64_t>(shape.n_tensors - 1, 0);
                        if (shape.pool >= 0 && shape.pool < (int)device.pools.size()) {
                            device.pools[shape.pool]->covers_all_entries =
                                (uint64_t)device.pools[shape.pool]->n_slots >= shape.n_entries;
                        }
                        if (shape.n_tensors == 0 && shape.pool < 0) {
                            shape.finished = false;
                        }
                        break;
                    }
                }
                it = device.seen_tensors.erase(it);
                device.visits_since_new_tensor = 0;
            } else {
                ++it;
            }
        }
        for (auto it = device.demand_count.begin(); it != device.demand_count.end();) {
            const void * source = it->first.expert >= 0
                ? (const char *)it->first.tensor +
                    (size_t)it->first.expert * moe_cache_host_stride(device, it->first.tensor, it->second.expert_size)
                : nullptr;
            if (moe_cache_ranges_overlap(
                    source, it->second.expert_size, base, size)) {
                it = device.demand_count.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = device.host_strides.begin(); it != device.host_strides.end();) {
            if (moe_cache_ranges_overlap(it->first, 1, base, size)) { it = device.host_strides.erase(it); }
            else { ++it; }
        }
    }
}

static void moe_cache_invalidate(const void * base, size_t size) {
    if (!base || size == 0 ||
        g_session_count.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        moe_cache_invalidate_session(*session, base, size);
    }
}

static size_t moe_cache_trim_session(
        moe_cache_session & session, int physical_device) {
    if (session.device_frozen) {
        return 0;
    }
    moe_cache_device * selected = nullptr;
    for (auto & device_ptr : session.devices) {
        if (device_ptr->physical == physical_device) {
            selected = device_ptr.get();
            break;
        }
    }
    if (!selected) {
        return 0;
    }

    std::unique_lock<std::mutex> dispatch_lock(selected->dispatch_mu);
    std::unique_lock<std::mutex> lock(session.mu);
    selected->dead.store(true);
    moe_cache_cancel_queue_locked(*selected, nullptr, 0, true);
    session.cv.notify_all();
    session.idle_cv.wait(lock, [&] {
        return !selected->inflight;
    });

    size_t freed = 0;
    for (const auto & pool_ptr : selected->pools) {
        if (pool_ptr->slab) {
            freed += (size_t)pool_ptr->n_slots * pool_ptr->expert_size;
        }
    }
    freed += selected->d_out_cap + selected->d_input_cap +
             selected->act_q8_cap;
    moe_cache_device_release(session);
    lock.unlock();
    moe_cache_free_device(*selected);
    moe_cache_budget_unregister(*selected);

    if (freed > 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d trimmed %zu MiB after a cache failure or allocator pressure\n",
                physical_device, freed >> 20);
    }
    return freed;
}

extern "C" size_t ggml_moe_cache_trim(int device) {
    if (g_session_count.load(std::memory_order_acquire) == 0) {
        return 0;
    }
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count) {
        return 0;
    }
    const int physical_device = info.devices[device].physical_device;
    size_t freed = 0;
    std::lock_guard<std::mutex> registry_lock(g_registry_mu);
    for (moe_cache_session * session : g_sessions) {
        freed += moe_cache_trim_session(*session, physical_device);
    }
    return freed;
}

// fork: profile files contain bounded usage counts, never expert weights or pointers.
static uint64_t moe_cache_profile_hash(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = (const unsigned char *) data;
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t moe_cache_profile_signature(const ggml_moe_cache_tensor_desc * tensors, int n_tensors) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < n_tensors; ++i) {
        const auto & t = tensors[i];
        hash = moe_cache_profile_hash(hash, t.name, strlen(t.name) + 1);
        const size_t expert_bytes = ggml_row_size((ggml_type)t.type, t.n_in) * (size_t)t.n_out;
        const uint64_t shape[] = {(uint64_t) expert_bytes, (uint64_t) t.n_in,
            (uint64_t) t.n_out, (uint64_t) t.n_expert, (uint64_t) t.type};
        hash = moe_cache_profile_hash(hash, shape, sizeof(shape));
        const size_t bytes = expert_bytes * (size_t) t.n_expert;
        const size_t sample = std::min<size_t>(bytes, 64);
        // Sample across each tensor so equal-shaped models do not share a profile by accident.
        for (size_t j = 0; j < 64; ++j) {
            const size_t offset = ((bytes - sample) / 63) * j;
            size_t remaining = sample, logical = offset;
            while (remaining) {
                const size_t part = std::min(remaining, expert_bytes - logical % expert_bytes);
                const size_t physical = (logical / expert_bytes) * t.expert_size + logical % expert_bytes;
                hash = moe_cache_profile_hash(hash, (const char *)t.data + physical, part);
                logical += part;
                remaining -= part;
            }
        }
    }
    return hash;
}

static bool moe_cache_profile_load(const char * path, uint64_t signature, size_t entries,
        const ggml_moe_cache_tensor_desc * tensors, int n_tensors,
        std::vector<std::vector<uint64_t>> & storage, std::vector<const uint64_t *> & counts) {
    std::unique_ptr<FILE, int (*)(FILE *)> file(fopen(path, "rb"), fclose);
    uint64_t header[5] = {};
    const uint64_t magic = UINT64_C(0x31464f5250454f4d); // MOEPROF1, native uint64_t format
    if (!file || fread(header, sizeof(header), 1, file.get()) != 1 ||
            header[0] != magic || header[1] != (uint64_t) n_tensors ||
            header[2] != entries || header[3] != signature) {
        return false;
    }
    uint64_t checksum = UINT64_C(14695981039346656037);
    storage.resize(n_tensors);
    counts.reserve(n_tensors);
    for (int i = 0; i < n_tensors; ++i) {
        storage[i].resize(tensors[i].n_expert);
        const size_t bytes = storage[i].size() * sizeof(uint64_t);
        if (fread(storage[i].data(), 1, bytes, file.get()) != bytes) {
            return false;
        }
        checksum = moe_cache_profile_hash(checksum, storage[i].data(), bytes);
        counts.push_back(storage[i].data());
    }
    return checksum == header[4] && fgetc(file.get()) == EOF && !ferror(file.get());
}

static bool moe_cache_profile_save(const char * path, uint64_t signature, size_t entries,
        const ggml_moe_cache_tensor_desc * tensors, const uint64_t * const * counts, int n_tensors) {
    uint64_t checksum = UINT64_C(14695981039346656037);
    for (int i = 0; i < n_tensors; ++i) {
        checksum = moe_cache_profile_hash(checksum, counts[i], (size_t) tensors[i].n_expert * sizeof(uint64_t));
    }
    const uint64_t header[] = {UINT64_C(0x31464f5250454f4d), (uint64_t) n_tensors, entries, signature, checksum};
    const std::string temporary = std::string(path) + ".tmp";
    std::unique_ptr<FILE, int (*)(FILE *)> file(fopen(temporary.c_str(), "wb"), fclose);
    if (!file) {
        return false;
    }
    bool ok = fwrite(header, sizeof(header), 1, file.get()) == 1;
    for (int i = 0; ok && i < n_tensors; ++i) {
        const size_t bytes = (size_t) tensors[i].n_expert * sizeof(uint64_t);
        ok = fwrite(counts[i], 1, bytes, file.get()) == bytes;
    }
    ok = fclose(file.release()) == 0 && ok;
    if (ok && rename(temporary.c_str(), path) == 0) {
        return true;
    }
    remove(temporary.c_str());
    return false;
}

// fork: a saved ranking is a startup hint; current prompt observations replace it.
static bool moe_cache_profile_rerank(moe_cache_session & session,
        std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(session.mu);
    const bool idle = session.idle_cv.wait_until(lock, deadline, [&] {
        if (session.stopping) {
            return true;
        }
        if (session.active_scopes || session.active_nodes) {
            return false;
        }
        for (const auto & device : session.devices) {
            if (!device->queue.empty() || device->inflight) {
                return false;
            }
        }
        return true;
    });
    if (!idle || session.stopping) {
        MOE_CACHE_LOG("[moe-cache] profile rerank skipped: cache did not become idle\n");
        return false;
    }
    for (const auto & device : session.devices) {
        for (const auto & pool : device->pools) {
            for (const auto & slot : pool->slots) {
                if (slot.readers) {
                    MOE_CACHE_LOG("[moe-cache] profile rerank skipped: cache slot still pinned\n");
                    return false;
                }
            }
        }
    }
    moe_cache_device_release(session, true);
    session.device_attempted = false;
    return true;
}

// Contiguous slabs cannot shrink in place. Recopy retained exact weights after freeing the old slab.
static bool moe_cache_resize_pool(moe_cache_device & device, moe_cache_pool & pool, int slots) {
    if (slots == pool.n_slots) {
        return true;
    }
    std::vector<moe_cache_slot> retained;
    for (const auto & slot : pool.slots) {
        if (slot.state == moe_cache_slot_state::valid) {
            retained.push_back(slot);
        }
    }
    std::stable_sort(retained.begin(), retained.end(), [](const moe_cache_slot & a, const moe_cache_slot & b) {
        return moe_cache_lending_uses(a) > moe_cache_lending_uses(b);
    });
    if (retained.size() > (size_t)slots) {
        device.evictions += retained.size() - slots;
        retained.resize(slots);
    }
    std::sort(retained.begin(), retained.end(), [](const moe_cache_slot & a, const moe_cache_slot & b) {
        return (uintptr_t)a.key.tensor < (uintptr_t)b.key.tensor ||
            (a.key.tensor == b.key.tensor && a.key.expert < b.key.expert);
    });
    std::vector<moe_cache_slot> replacement(slots);
    std::vector<int> free_slots;
    free_slots.reserve(slots);
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    map.reserve(slots);
    for (int i = 0; i < (int)retained.size(); ++i) {
        replacement[i] = retained[i];
        replacement[i].prev = i ? i - 1 : -1;
        replacement[i].next = i + 1 < (int)retained.size() ? i + 1 : -1;
        replacement[i].readers = 0;
        map.emplace(replacement[i].key, i);
    }
    for (int i = slots - 1; i >= (int)retained.size(); --i) {
        free_slots.push_back(i);
    }
    ggml_cuda_set_device(device.logical);
    const size_t before = (size_t)pool.n_slots * pool.expert_size;
    if (pool.slab && !moe_cache_cuda_ok(device, cudaFree(pool.slab), "lend slab free", true)) {
        return false;
    }
    moe_cache_budget_allocation(device, before, false);
    device.allocated_bytes -= before;
    pool.slab = nullptr;
    pool.n_slots = 0;
    pool.slots.clear();
    pool.free_slots.clear();
    pool.map.clear();
    pool.lru_head = pool.lru_tail = -1;
    pool.covers_all_entries = false;
    if (!slots) {
        return true;
    }
    // Query after freeing the old slab, so a replacement can use all of
    // the returned space without a transient old-plus-new allocation peak.
    const size_t headroom = moe_cache_allocation_headroom(device, "resize expert pool", (size_t)slots * pool.expert_size);
    slots = (int)std::min((size_t)slots, headroom / pool.expert_size);
    if (!slots) {
        device.evictions += retained.size();
        return true;
    }
    size_t bytes = 0;
    for (;;) {
        if (retained.size() > (size_t)slots) {
            std::stable_sort(retained.begin(), retained.end(), [](const moe_cache_slot & a, const moe_cache_slot & b) {
                return moe_cache_lending_uses(a) > moe_cache_lending_uses(b);
            });
            device.evictions += retained.size() - slots;
            retained.resize(slots);
            std::sort(retained.begin(), retained.end(), [](const moe_cache_slot & a, const moe_cache_slot & b) {
                return (uintptr_t)a.key.tensor < (uintptr_t)b.key.tensor ||
                    (a.key.tensor == b.key.tensor && a.key.expert < b.key.expert);
            });
        }
        replacement.resize(slots);
        free_slots.clear();
        map.clear();
        for (int i = 0; i < (int)retained.size(); ++i) {
            replacement[i] = retained[i];
            replacement[i].prev = i ? i - 1 : -1;
            replacement[i].next = i + 1 < (int)retained.size() ? i + 1 : -1;
            replacement[i].readers = 0;
            map.emplace(replacement[i].key, i);
        }
        for (int i = slots - 1; i >= (int)retained.size(); --i) {
            free_slots.push_back(i);
        }
        bytes = (size_t)slots * pool.expert_size;
        if (!moe_cache_cuda_ok(device, cudaMalloc((void **)&pool.slab, bytes), "lend slab resize", false)) {
            return false;
        }
        size_t deficit = 0;
        if (!moe_cache_slab_preserves_reserve(device, bytes, "resize expert pool", deficit)) {
            (void)cudaFree(pool.slab);
            pool.slab = nullptr;
            return false;
        }
        if (!deficit) {
            break;
        }
        (void)cudaFree(pool.slab);
        pool.slab = nullptr;
        const size_t remove = deficit / pool.expert_size + (deficit % pool.expert_size != 0);
        slots -= (int)std::min((size_t)slots, remove);
        if (!slots) {
            device.evictions += retained.size();
            return true;
        }
    }
    moe_cache_budget_allocation(device, bytes, true);
    device.allocated_bytes += bytes;
    pool.n_slots = slots;
    pool.slots.swap(replacement);
    pool.free_slots.swap(free_slots);
    pool.map.swap(map);
    pool.lru_head = retained.empty() ? -1 : 0;
    pool.lru_tail = (int)retained.size() - 1;
    for (size_t i = 0; i < retained.size();) {
        size_t end = i + 1;
        while (moe_cache_host_stride(device, retained[i].key.tensor, pool.expert_size) == pool.expert_size &&
                end < retained.size() && retained[end].key.tensor == retained[i].key.tensor &&
                retained[end].key.expert == retained[i].key.expert + (int32_t)(end - i)) {
            ++end;
        }
        const void * source = (const char *)retained[i].key.tensor +
            (size_t)retained[i].key.expert * moe_cache_host_stride(device, retained[i].key.tensor, pool.expert_size);
        if (!moe_cache_cuda_ok(device, cudaMemcpy(pool.slab + i * pool.expert_size,
                source, (end - i) * pool.expert_size, cudaMemcpyHostToDevice), "lend survivor refill", false)) {
            for (int j = 0; j < (int)retained.size(); ++j) {
                moe_cache_slot_reset(pool, j, true);
            }
            return false;
        }
        i = end;
    }
    for (const auto & shape : device.shapes) {
        if (shape.expert_size == pool.expert_size && shape.wtype == pool.wtype) {
            pool.covers_all_entries = (uint64_t)slots >= shape.n_entries;
            break;
        }
    }
    return true;
}

static size_t moe_cache_lend(void * opaque, size_t bytes) {
    auto * session = (moe_cache_session *)opaque;
    if (!session || !bytes || session->dormant || session->stopping) {
        return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!moe_cache_profile_rerank(*session, start + std::chrono::seconds(30))) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(session->mu);
    // A second compute may have started while the idle wait released the mutex.
    if (session->active_scopes || session->active_nodes || session->stopping) {
        return 0;
    }
    struct candidate { moe_cache_pool * pool; uint64_t uses; };
    std::vector<candidate> ranked;
    std::unordered_map<moe_cache_pool *, int> targets;
    for (const auto & device : session->devices) {
        if (device->dead.load() || !device->queue.empty() || device->inflight) {
            continue;
        }
        for (const auto & pool : device->pools) {
            targets[pool.get()] = pool->n_slots;
            for (int i = 0; i < pool->n_slots; ++i) {
                const auto & slot = pool->slots[i];
                ranked.push_back({pool.get(), slot.state == moe_cache_slot_state::free ? 0 : moe_cache_lending_uses(slot)});
            }
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const candidate & a, const candidate & b) {
        return a.uses < b.uses;
    });
    size_t selected = 0;
    for (const auto & candidate : ranked) {
        if (selected >= bytes) {
            break;
        }
        --targets[candidate.pool];
        selected += candidate.pool->expert_size;
    }
    size_t freed = 0;
    for (const auto & device : session->devices) {
        for (const auto & pool : device->pools) {
            const auto found = targets.find(pool.get());
            if (found == targets.end() || found->second == pool->n_slots) {
                continue;
            }
            const size_t before = (size_t)pool->n_slots * pool->expert_size;
            moe_cache_resize_pool(*device, *pool, found->second);
            freed += before - (size_t)pool->n_slots * pool->expert_size;
        }
    }
    MOE_CACHE_LOG("[moe-cache] lent %zu MiB requested=%zu MiB; least-use eviction and exact survivor refill %.3f ms\n",
            freed >> 20, bytes >> 20, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return freed;
}

static size_t moe_cache_return(void * opaque) {
    auto * session = (moe_cache_session *)opaque;
    if (!session || session->dormant || session->stopping) {
        return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!moe_cache_profile_rerank(*session, start + std::chrono::seconds(30))) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(session->mu);
    if (session->active_scopes || session->active_nodes || session->stopping) {
        return 0;
    }
    size_t restored = 0;
    for (const auto & device : session->devices) {
        if (device->dead.load() || !device->queue.empty() || device->inflight) {
            continue;
        }
        if (!device->budget_ready) {
            moe_cache_prepare_budget(*session, *device);
        }
        if (!device->budget_claimed || !moe_cache_reserve_table_arena(*session, *device) ||
                !moe_cache_refresh_budget(*session, *device, "return prompt scratch")) {
            continue;
        }
        const size_t scratch = moe_cache_scratch_total(device->scratch_reserve) + device->table_arena_cap;
        const size_t slab_limit = device->budget_limit > scratch ? device->budget_limit - scratch : 0;
        size_t remaining = slab_limit > device->allocated_bytes ? slab_limit - device->allocated_bytes : 0;
        long double weights = 0;
        for (const auto & shape : device->shapes) {
            if (shape.pool >= 0 && shape.pool < (int)device->pools.size()) {
                weights += (long double)shape.expert_size * shape.n_tensors;
            }
        }
        for (const auto & shape : device->shapes) {
            if (shape.pool < 0 || shape.pool >= (int)device->pools.size()) {
                continue;
            }
            auto & pool = *device->pools[shape.pool];
            const long double weight = (long double)shape.expert_size * shape.n_tensors;
            const size_t share = weights > 0 ? (size_t)((long double)remaining * weight / weights) : 0;
            const size_t blocks = pool.expert_size / ggml_type_size((ggml_type)pool.wtype);
            const size_t max_slots = std::min<uint64_t>(shape.n_entries, INT_MAX / blocks);
            const size_t target = std::min(max_slots, (size_t)pool.n_slots + share / pool.expert_size);
            const size_t before = (size_t)pool.n_slots * pool.expert_size;
            if (target > (size_t)pool.n_slots) {
                moe_cache_resize_pool(*device, pool, (int)target);
            }
            const size_t after = (size_t)pool.n_slots * pool.expert_size;
            if (after >= before) {
                restored += after - before;
                remaining -= std::min(remaining, after - before);
            }
            weights -= weight;
        }
    }
    MOE_CACHE_LOG("[moe-cache] returned %zu MiB to expert slots; exact survivor refill %.3f ms\n",
            restored >> 20, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return restored;
}

static int moe_cache_warm(void * opaque, const ggml_moe_cache_tensor_desc * tensors,
        const uint64_t * const * counts, int n_tensors) {
    auto * session = (moe_cache_session *)opaque;
    if (!session || !tensors || n_tensors < 1 || n_tensors > 4096 || session->dormant) {
        return 0;
    }
    size_t entries = 0;
    for (int i = 0; i < n_tensors; ++i) {
        ggml_moe_cache_shape_caps shape = {};
        if ((counts && !counts[i]) || !tensors[i].name || !tensors[i].data ||
                !tensors[i].expert_size || tensors[i].n_expert < 1 || tensors[i].n_expert > 65536 ||
                tensors[i].expert_size > SIZE_MAX / (size_t) tensors[i].n_expert ||
                (size_t)tensors[i].n_expert > 1048576 - entries ||
                !moe_cache_type_supported((ggml_type)tensors[i].type) || tensors[i].n_in <= 0 ||
                tensors[i].n_in % ggml_blck_size((ggml_type)tensors[i].type) != 0 ||
                !moe_cache_query_shape(tensors[i].type, tensors[i].n_in, tensors[i].n_out,
                        tensors[i].n_expert, tensors[i].expert_size, &shape)) {
            return 0;
        }
        entries += tensors[i].n_expert;
    }
    const bool loading = !counts;
    bool observed = false;
    if (counts) {
        for (int i = 0; i < n_tensors && !observed; ++i) {
            for (int32_t e = 0; e < tensors[i].n_expert; ++e) {
                if (counts[i][e]) {
                    observed = true;
                    break;
                }
            }
        }
    }
    if (loading) {
        std::lock_guard<std::mutex> lock(session->mu);
        if (session->profile_load_attempted) {
            return 0;
        }
        session->profile_load_attempted = true;
    }
    std::vector<std::vector<uint64_t>> loaded_storage;
    std::vector<const uint64_t *> loaded_counts;
    const char * path = getenv(loading ? "GGML_MOE_CACHE_PROFILE_LOAD" : "GGML_MOE_CACHE_PROFILE_SAVE");
    if (path && path[0] && (loading || observed)) {
        const uint64_t signature = moe_cache_profile_signature(tensors, n_tensors);
        if (loading) {
            if (!moe_cache_profile_load(path, signature, entries, tensors, n_tensors, loaded_storage, loaded_counts)) {
                MOE_CACHE_LOG("[moe-cache] profile rejected: %s (missing, mismatched, or corrupt)\n", path);
                return 0;
            }
            counts = loaded_counts.data();
            MOE_CACHE_LOG("[moe-cache] profile loaded: %s, %zu ranked entries; sampled weight identity verified\n", path, entries);
        } else {
            const bool saved = moe_cache_profile_save(path, signature, entries, tensors, counts, n_tensors);
            MOE_CACHE_LOG("[moe-cache] profile %s: %s, %zu usage counts\n", saved ? "saved" : "save failed", path, entries);
        }
    } else if (loading) {
        return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::seconds(30);
    if (!loading && !moe_cache_profile_rerank(*session, deadline)) {
        return 0;
    }
    moe_cache_session_enter(session);
    struct scope_guard {
        moe_cache_session * session;
        ~scope_guard() { moe_cache_session_leave(session); }
    } guard{session};
    auto begin = [&](int i) -> moe_cache_node * {
        const auto & t = tensors[i];
        return (moe_cache_node *)moe_cache_begin(t.name, t.data, t.expert_size,
                t.n_in, t.n_out, t.type, t.n_expert, 1, 1);
    };
    // Discover every shape before the first admission chooses a pool budget.
    for (int pass = 0; pass < 3; ++pass) {
        for (int i = 0; i < n_tensors; ++i) {
            moe_cache_end(begin(i));
        }
    }
    struct candidate { int tensor; int32_t expert; uint64_t count; bool current; long double startup_score; bool first_layer; };
    std::vector<candidate> ranked;
    ranked.reserve(entries);
    for (int i = 0; i < n_tensors; ++i) {
        long double observed_total = 0;
        for (int32_t e = 0; e < tensors[i].n_expert; ++e) { observed_total += counts[i][e]; }
        for (int32_t e = 0; e < tensors[i].n_expert; ++e) {
            const moe_cache_key key{tensors[i].data, e};
            if (loading && counts[i][e]) {
                session->profile_counts[key] = counts[i][e];
            }
            int layer = 0;
            moe_cache_layer_number(tensors[i].name, layer);
            const size_t bytes = ggml_row_size((ggml_type)tensors[i].type, tensors[i].n_in) * (size_t)tensors[i].n_out;
            // Output pruning can observe fewer tokens in the last layer. Compare
            // conditional use rates, preserving the original counts on disk.
            const long double rate = counts[i][e] / std::max(1.0L, observed_total);
            const long double priority = rate * (1 + 4.0L / (layer + 1)) / bytes;
            if (counts[i][e] || loading) {
                ranked.push_back({i, e, counts[i][e], !loading, priority, layer == 0});
            } else if (!loading) {
                const auto previous = session->profile_counts.find(key);
                if (previous != session->profile_counts.end()) {
                    ranked.push_back({i, e, previous->second, false, priority, layer == 0});
                }
            }
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [loading](const candidate & a, const candidate & b) {
        if (loading && a.first_layer != b.first_layer) { return a.first_layer > b.first_layer; }
        if (loading && a.startup_score != b.startup_score) { return a.startup_score > b.startup_score; }
        if (a.current != b.current) {
            return a.current > b.current;
        }
        return a.count > b.count;
    });
    std::unordered_map<moe_cache_key, size_t, moe_cache_key_hash> ranks;
    ranks.reserve(ranked.size());
    for (size_t i = 0; i < ranked.size(); ++i) {
        ranks.emplace(moe_cache_key{tensors[ranked[i].tensor].data, ranked[i].expert}, i);
    }
    int queued = 0;
    std::unordered_set<moe_cache_pool *> settled;
    for (const auto & c : ranked) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        std::unique_ptr<moe_cache_node, decltype(&moe_cache_end)> node(begin(c.tensor), moe_cache_end);
        if (!node) {
            continue;
        }
        bool wake = false;
        {
            std::unique_lock<std::mutex> lock(session->mu);
            auto & device = *node->device;
            auto & pool = *node->pool;
            const moe_cache_key key{node->host_base, c.expert};
            const auto existing = pool.map.find(key);
            if (existing != pool.map.end()) {
                pool.slots[existing->second].warm_uses = (uint32_t)std::min<uint64_t>(c.count, UINT32_MAX);
            } else if (pool.free_slots.empty() && !session->device_frozen && !settled.count(&pool)) {
                const size_t rank = ranks.at(key);
                int victim = -1;
                size_t worst = rank;
                for (int i = pool.lru_head; i >= 0; i = pool.slots[i].next) {
                    const auto & slot = pool.slots[i];
                    const auto found = ranks.find(slot.key);
                    const size_t resident_rank = found == ranks.end() ? SIZE_MAX : found->second;
                    if (!slot.readers && resident_rank > worst) {
                        victim = i;
                        worst = resident_rank;
                    }
                }
                if (victim >= 0) {
                    moe_cache_slot_reset(pool, victim, true);
                    ++device.evictions;
                } else {
                    settled.insert(&pool);
                }
            }
            if (!session->stopping && !device.dead.load() && !pool.free_slots.empty() &&
                    pool.map.find(key) == pool.map.end()) {
                // Wait for bounded queue space; the worker never uses dispatch_mu.
                session->idle_cv.wait_until(lock, deadline, [&] {
                    return session->stopping || device.dead.load() ||
                            ((int)device.queue.size() < session->config.queue_max &&
                             node->expert_size <= (session->config.queue_mb << 20) -
                                std::min(session->config.queue_mb << 20, device.queued_bytes));
                });
                if (!session->stopping && !device.dead.load() && std::chrono::steady_clock::now() < deadline) {
                    device.demand_count[key].count = std::numeric_limits<uint16_t>::max();
                    int inserts_left = 1;
                    moe_cache_lookup_or_queue_locked(*session, device, pool, node->pool_index,
                            node->host_base, c.expert, node->expert_size, inserts_left, wake);
                    queued += 1 - inserts_left;
                    const auto inserted = pool.map.find(key);
                    if (inserted != pool.map.end()) {
                        pool.slots[inserted->second].warm_uses = (uint32_t)std::min<uint64_t>(c.count, UINT32_MAX);
                    }
                }
            }
        }
        if (wake) {
            session->cv.notify_all();
        }
    }
    {
        std::unique_lock<std::mutex> lock(session->mu);
        session->idle_cv.wait_until(lock, deadline, [&] {
            for (const auto & device : session->devices) {
                if (!device->queue.empty() || device->inflight) {
                    return false;
                }
            }
            return true;
        });
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    MOE_CACHE_LOG("[moe-cache] %s warm: %d queued of %zu touched entries, %.3f ms; decode lookup counters unchanged\n", loading ? "profile" : "prompt", queued, ranked.size(), ms);
    return queued;
}

static moe_cache_session * moe_prefetch_session(void * opaque) {
    if (opaque) {
        return (moe_cache_session *) opaque;
    }
    return g_session_suppressed == 0 && !g_session_stack.empty() ? g_session_stack.back().active : nullptr;
}

static void moe_cache_prefetch_configure(void * opaque, int mode) {
    auto * session = (moe_cache_session *) opaque;
    if (!session) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    session->prefetch_enabled = mode != 0;
}

static void moe_prefetch_finish(moe_cache_session & session, moe_prefetch_copy & copy) {
    const auto selected = std::find_if(session.devices.begin(), session.devices.end(),
            [&](const std::unique_ptr<moe_cache_device> & device) { return device->physical == copy.physical; });
    if (selected == session.devices.end()) {
        return;
    }
    auto & device = **selected;
    const auto found = session.device_tables.find(copy.tensor);
    if (copy.published && found != session.device_tables.end() && found->second.staging_device &&
            copy.expert >= 0 && (size_t) copy.expert < found->second.addresses.size()) {
        auto & table = found->second;
        ggml_cuda_set_device(device.logical);
        CUDA_CHECK(cudaMemcpyAsync(table.staging_device + copy.expert, &table.addresses[copy.expert],
                sizeof(void *), cudaMemcpyHostToDevice, nullptr));
        const uint8_t hot = table.addresses[copy.expert] != table.fallback[copy.expert];
        if (table.hot) {
            CUDA_CHECK(cudaMemcpyAsync(table.hot + copy.expert, &hot, sizeof(hot), cudaMemcpyHostToDevice, nullptr));
        }
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    if (copy.pool >= 0 && copy.pool < (int) device.pools.size()) {
        auto & pool = *device.pools[copy.pool];
        if (copy.slot >= 0 && copy.slot < pool.n_slots) {
            auto & slot = pool.slots[copy.slot];
            if (slot.generation == copy.generation) {
                if (copy.published && slot.readers > 0) {
                    slot.readers--;
                }
                if (slot.staging && slot.readers == 0 && slot.state == moe_cache_slot_state::valid) {
                    moe_cache_slot_reset(pool, copy.slot, false);
                }
            }
        }
    }
    copy.published = false;
}

static void moe_cache_prefetch_observe(void * opaque, int layer, const int32_t * ids, int n_ids) {
    auto * session = moe_prefetch_session(opaque);
    if (!session || layer < 0 || layer >= 1024 || n_ids < 0 || n_ids > 512) {
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (!session->prefetch_enabled || session->active_scopes != 1) {
        return;
    }
    const int64_t now = ggml_time_us();
    if (!ids) {
        if (session->last_layer >= 0 && layer == session->last_layer + 1) {
            const int previous = session->last_layer;
            const double elapsed = (double) (now - session->layer_start[previous]);
            double & estimate = session->layer_time_us[previous];
            estimate = estimate == 0.0 ? elapsed : 0.875 * estimate + 0.125 * elapsed;
        }
        session->layer_start[layer] = now;
        session->last_layer = layer;
    }
    for (const auto & copy : session->predictions) {
        if (!ids && copy->target_layer < layer) {
            moe_prefetch_finish(*session, *copy);
            copy->closed = true;
            moe_prefetch_log(*copy);
        }
        if (copy->target_layer != layer) {
            continue;
        }
        if (!ids) {
            copy->target_start = now;
        } else {
            copy->selected = std::find(ids, ids + n_ids, copy->expert) != ids + n_ids;
            if (!copy->closed && copy->selected && copy->done > 0 && !copy->published && session->device_ready && !session->devices.empty()) {
                auto & device = *session->devices[0];
                if (copy->physical != device.physical) {
                    continue;
                }
                const auto found = session->device_tables.find(copy->tensor);
                if (found == session->device_tables.end() || !found->second.staging_device ||
                        copy->pool < 0 || copy->pool >= (int) device.pools.size() ||
                        copy->expert < 0 || (size_t) copy->expert >= found->second.addresses.size()) {
                    continue;
                }
                auto & pool = *device.pools[copy->pool];
                if (copy->slot < 0 || copy->slot >= pool.n_slots) {
                    continue;
                }
                auto & slot = pool.slots[copy->slot];
                if (slot.generation != copy->generation || slot.state != moe_cache_slot_state::valid) {
                    continue;
                }
                auto & table = found->second;
                const void * address = pool.slab + (size_t) copy->slot * pool.expert_size;
                ggml_cuda_set_device(device.logical);
                CUDA_CHECK(cudaMemcpyAsync(table.staging_device + copy->expert, &address, sizeof(address), cudaMemcpyHostToDevice, nullptr));
                const uint8_t hot = 1;
                if (table.hot) {
                    CUDA_CHECK(cudaMemcpyAsync(table.hot + copy->expert, &hot, sizeof(hot), cudaMemcpyHostToDevice, nullptr));
                }
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
                slot.readers++;
                copy->published = true;
            }
        }
    }
}

static int moe_prefetch_stage_slot(moe_cache_pool & pool, moe_cache_device & device, bool frozen, bool reserve = false) {
    int reserved = 0;
    for (int i = 0; i < pool.n_slots; ++i) {
        auto & slot = pool.slots[i];
        if (!slot.staging) {
            continue;
        }
        reserved++;
        if (!reserve && slot.readers == 0 && (slot.state == moe_cache_slot_state::free ||
                (slot.state == moe_cache_slot_state::valid && slot.prediction && slot.prediction->closed))) {
            moe_cache_slot_reset(pool, i, false);
            return i;
        }
    }
    if (reserved >= 4 || pool.n_slots < 16) {
        return -1;
    }
    if (!pool.free_slots.empty()) {
        const int index = pool.free_slots.back();
        pool.free_slots.pop_back();
        pool.slots[index].staging = true;
        return index;
    }
    if (frozen) {
        return -1;
    }
    int candidate = -1;
    for (int i = pool.lru_head; i >= 0; i = pool.slots[i].next) {
        const auto & slot = pool.slots[i];
        if (slot.readers == 0 && !slot.staging && (candidate < 0 || slot.uses < pool.slots[candidate].uses)) {
            candidate = i;
        }
    }
    if (candidate >= 0) {
        device.evictions++;
        moe_cache_slot_reset(pool, candidate, false);
        pool.slots[candidate].staging = true;
    }
    return candidate;
}

static void moe_cache_prefetch(void * opaque, const ggml_moe_cache_tensor_desc * tensors, int n_tensors,
        const int32_t * ids, int n_ids, int source_layer, int target_layer, int automatic) {
    auto * session = moe_prefetch_session(opaque);
    if (!session || !tensors || !ids || n_tensors < 1 || n_tensors > 3 || n_ids < 1 || n_ids > 32 ||
            source_layer < 0 || target_layer <= source_layer || target_layer - source_layer > 3 || target_layer >= 1024) {
        return;
    }
    // Descriptors retain physical bank stride; pools and link budgets count packed bytes.
    ggml_moe_cache_tensor_desc packed[3];
    size_t source_strides[3];
    for (int t = 0; t < n_tensors; ++t) {
        const auto & desc = tensors[t];
        if (!desc.data || desc.n_in <= 0 || desc.n_out <= 0 || desc.n_expert <= 0 ||
                !moe_cache_type_supported((ggml_type) desc.type) ||
                desc.n_in % ggml_blck_size((ggml_type) desc.type) != 0) {
            return;
        }
        if ((uint64_t) (desc.n_in / ggml_blck_size((ggml_type) desc.type)) >
                SIZE_MAX / ggml_type_size((ggml_type) desc.type)) {
            return;
        }
        const size_t row = ggml_row_size((ggml_type) desc.type, desc.n_in);
        if (row == 0 || (uint64_t) desc.n_out > SIZE_MAX / row) {
            return;
        }
        const size_t bytes = row * (size_t) desc.n_out;
        if (desc.expert_size < bytes ||
                (uint64_t) (desc.n_expert - 1) > (SIZE_MAX - bytes) / desc.expert_size) {
            return;
        }
        packed[t] = desc;
        packed[t].expert_size = bytes;
        source_strides[t] = desc.expert_size;
    }
    tensors = packed;
    std::lock_guard<std::mutex> lock(session->mu);
    if (!session->prefetch_enabled || session->stopping || session->dormant ||
            session->active_scopes != 1 || session->predictions.size() >= 1024) {
        return;
    }
    const int64_t now = ggml_time_us();
    double lead_us = 0.0;
    for (int layer = source_layer; layer < target_layer; ++layer) {
        if (session->layer_time_us[layer] <= 0.0) {
            return;
        }
        lead_us += session->layer_time_us[layer];
    }
    const double elapsed_us = (double) (now - session->layer_start[source_layer]);
    const double available_us = 0.75 * (lead_us - elapsed_us);
    if (available_us <= 0.0) {
        return;
    }
    for (auto & device_ptr : session->devices) {
        auto & device = *device_ptr;
        if (device.dead || device.urgent_transfers.load(std::memory_order_acquire) != 0 ||
                device.copy_samples < 4 || device.copy_bytes_per_us <= 0.0 ||
                (device.inflight && !device.inflight_prefetch) ||
                std::any_of(device.queue.begin(), device.queue.end(),
                    [](const moe_cache_job & job) { return !job.prediction; })) {
            continue;
        }
        // Farther layers may queue behind earlier guesses, but all pending bytes consume lead time.
        const size_t pending = device.queued_bytes + (device.inflight ? device.inflight_bytes : 0);
        size_t budget = ggml_moe_prefetch_budget(lead_us, elapsed_us, device.copy_bytes_per_us, pending);
        // Auto selects the shortest lead that covers one complete expert at measured speed.
        size_t whole_expert = 0;
        for (int t = 0; t < n_tensors; ++t) {
            if (tensors[t].expert_size > (16u << 20) - whole_expert) {
                whole_expert = 16u << 20;
                break;
            }
            whole_expert += tensors[t].expert_size;
        }
        if (automatic && budget < whole_expert) {
            continue;
        }
        if (automatic && target_layer - source_layer > 1) {
            const double shorter = available_us - 0.75 * session->layer_time_us[target_layer - 1];
            if (shorter * device.copy_bytes_per_us >= (double) whole_expert + pending) {
                continue;
            }
        }
        for (int rank = 0; rank < n_ids && budget > 0; ++rank) {
            for (int t = 0; t < n_tensors; ++t) {
                const auto & desc = tensors[t];
                if (ids[rank] < 0 || ids[rank] >= desc.n_expert || !desc.data || desc.expert_size > budget) {
                    continue;
                }
                // Only known host tensors and already allocated pools are eligible. No speculative allocation.
                const auto seen = device.seen_tensors.find(desc.data);
                if (seen == device.seen_tensors.end() || seen->second.expert_size != desc.expert_size || seen->second.wtype != desc.type || seen->second.n_expert != desc.n_expert ||
                        moe_cache_host_stride(device, desc.data, desc.expert_size) != source_strides[t]) {
                    continue;
                }
                const auto route = session->layer_devices.find(target_layer);
                if (route == session->layer_devices.end() || route->second != device.physical) {
                    continue;
                }
                for (int p = 0; p < (int) device.pools.size(); ++p) {
                    auto & pool = *device.pools[p];
                    const moe_cache_key key{desc.data, ids[rank]};
                    // Staging slots remain outside demand eviction and cannot be overwritten while pinned.
                    if (!pool.slab || pool.expert_size != desc.expert_size || pool.wtype != desc.type ||
                            pool.map.find(key) != pool.map.end() ||
                            (int) device.queue.size() >= session->config.queue_max ||
                            desc.expert_size > (session->config.queue_mb << 20) - std::min(session->config.queue_mb << 20, device.queued_bytes)) {
                        continue;
                    }
                    const int index = moe_prefetch_stage_slot(pool, device, session->device_frozen);
                    if (index < 0) {
                        continue;
                    }
                    std::shared_ptr<moe_prefetch_copy> copy;
                    try {
                        copy = std::make_shared<moe_prefetch_copy>();
                    } catch (...) {
                        return;
                    }
                    copy->physical = device.physical;
                    copy->tensor = desc.data;
                    copy->pool = p;
                    copy->slot = index;
                    copy->generation = pool.slots[index].generation + 1;
                    copy->source_layer = source_layer;
                    copy->target_layer = target_layer;
                    copy->expert = ids[rank];
                    copy->rank = rank;
                    copy->bytes = desc.expert_size;
                    copy->issue = now;
                    copy->deadline = now + (int64_t) available_us;
                    auto & slot = pool.slots[index];
                    try {
                        pool.map.emplace(key, index);
                        device.queue.push_back({p, index, slot.generation + 1, key,
                            (const char *) desc.data + (size_t) ids[rank] * source_strides[t], desc.expert_size, copy});
                        session->predictions.push_back(copy);
                    } catch (...) {
                        pool.map.erase(key);
                        if (!device.queue.empty() && device.queue.back().prediction == copy) {
                            device.queue.pop_back();
                        }
                        return;
                    }
                    slot.key = key;
                    slot.generation++;
                    slot.prediction = copy;
                    slot.state = moe_cache_slot_state::copying;
                    device.queued_bytes += desc.expert_size;
                    budget -= desc.expert_size;
                    break;
                }
            }
        }
    }
    session->cv.notify_all();
}

static int moe_cache_session_status(void * opaque, size_t * allocated_bytes) {
    auto & session = *static_cast<moe_cache_session *>(opaque);
    std::lock_guard<std::mutex> lock(session.mu);
    *allocated_bytes = 0;
    if (session.dormant || session.stopping) { return 0; }
    for (const auto & device : session.devices) {
        if (!device->dead.load()) { *allocated_bytes += device->allocated_bytes; }
    }
    return 1;
}

// Explicit phase diagnostics; this function is never used by layer dispatch.
static void moe_cache_print_stats(void * opaque) {
    auto * session = (moe_cache_session *)opaque;
    if (!session) {
        MOE_CACHE_LOG("[moe-cache] phase snapshot unavailable: reason=no-session\n");
        return;
    }
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->active_nodes || session->active_scopes || session->stopping) {
        MOE_CACHE_LOG("[moe-cache] phase snapshot unavailable: session=%llu reason=not-idle\n",
                (unsigned long long)session->phase_stats_id);
        return;
    }
    // Device phase snapshot begin.
    if (session->device_ready.load()) {
        unsigned long long counts[3] = {};
        if (session->devices.size() != 1 || !session->device_route_counts) {
            MOE_CACHE_LOG("[moe-cache] phase snapshot unavailable: session=%llu reason=device-stats-disabled\n",
                    (unsigned long long)session->phase_stats_id);
            return;
        }
        auto & device = *session->devices[0];
        ggml_cuda_set_device(device.logical);
        if (cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(counts, session->device_route_counts, sizeof(counts), cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void)cudaGetLastError();
            MOE_CACHE_LOG("[moe-cache] phase snapshot unavailable: session=%llu reason=device-read\n",
                    (unsigned long long)session->phase_stats_id);
            return;
        }
        MOE_CACHE_LOG("[moe-cache] phase snapshot: session=%llu epoch=%llu device=%d hits=%llu misses=%llu dispatches=%llu kind=device monotonic_us=%llu\n",
                (unsigned long long)session->phase_stats_id, (unsigned long long)session->device_capture_generation,
                device.physical, counts[0], counts[1], counts[2], (unsigned long long)ggml_time_us());
        return;
    }
    // Device phase snapshot end.
    for (const auto & device : session->devices) {
        MOE_CACHE_LOG("[moe-cache] phase snapshot: session=%llu epoch=1 device=%d hits=%llu misses=%llu dispatches=%llu kind=host monotonic_us=%llu\n",
                (unsigned long long)session->phase_stats_id, device->physical,
                (unsigned long long)device->hits, (unsigned long long)device->misses,
                (unsigned long long)device->nodes, (unsigned long long)ggml_time_us());
    }
}

// fork: pin existing missing expert pages; resident experts keep their VRAM addresses.
static size_t moe_cache_mem_available() {
#if defined(__linux__)
    FILE * f = fopen("/proc/meminfo", "r");
    char line[256];
    size_t kb = 0;
    while (f && fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) {
            break;
        }
    }
    if (f) {
        fclose(f);
    }
    return kb <= SIZE_MAX / 1024 ? kb * 1024 : 0;
#else
    return 0;
#endif
}

static bool moe_cache_cgroup_headroom(size_t extra, size_t registered = 0, uint64_t * peak_current = nullptr) {
#if defined(__linux__)
    FILE * f = fopen("/proc/self/cgroup", "r");
    char line[4096];
    std::string path;
    while (f && fgets(line, sizeof(line), f)) {
        if (strncmp(line, "0::/", 4) == 0) {
            path = std::string("/sys/fs/cgroup") + (line + 3);
            if (!path.empty() && path.back() == '\n') {
                path.pop_back();
            }
            break;
        }
    }
    if (f) {
        fclose(f);
    }
    if (path.empty()) {
        return false;
    }
    auto read = [&](const char * name, uint64_t & value) {
        FILE * input = fopen((path + name).c_str(), "r");
        char text[64] = {};
        const bool ok = input && fgets(text, sizeof(text), input);
        if (input) {
            fclose(input);
        }
        if (!ok) {
            return false;
        }
        if (strncmp(text, "max", 3) == 0) {
            value = UINT64_MAX;
            return true;
        }
        char * end = nullptr;
        errno = 0;
        value = strtoull(text, &end, 10);
        return errno == 0 && end != text;
    };
    // Check every ancestor limit, including the sysguard service.
    for (;;) {
        // The root cgroup has no memory.max file or service limit.
        if (path == "/sys/fs/cgroup") {
            break;
        }
        uint64_t current = 0, limit = 0;
        if (!read("/memory.current", current) || !read("/memory.max", limit)) {
            return false;
        }
        if (limit != UINT64_MAX) {
            uint64_t file = 0, inactive = 0, shmem = 0, dirty = 0, writeback = 0;
            FILE * stat = fopen((path + "/memory.stat").c_str(), "r");
            if (!stat) {
                return false;
            }
            char name[128];
            unsigned long long value;
            while (fscanf(stat, "%127s %llu", name, &value) == 2) {
                if (strcmp(name, "file") == 0) { file = value; }
                else if (strcmp(name, "inactive_file") == 0) { inactive = value; }
                else if (strcmp(name, "shmem") == 0) { shmem = value; }
                else if (strcmp(name, "file_dirty") == 0) { dirty = value; }
                else if (strcmp(name, "file_writeback") == 0) { writeback = value; }
            }
            fclose(stat);
            const uint64_t credit = ggml_cuda_moe_cache_host_file_credit(file, inactive, shmem, dirty, writeback, registered);
            const uint64_t charged = current - std::min(current, credit);
            if (peak_current) {
                uint64_t peak = current;
                (void) read("/memory.peak", peak);
                *peak_current = std::max(*peak_current, std::max(current, peak));
            }
            MOE_CACHE_LOG("[moe-cache] host service budget bytes: current=%llu limit=%llu clean_file_credit=%llu registered=%zu extra=%zu path=%s\n",
                    (unsigned long long) current, (unsigned long long) limit, (unsigned long long) credit,
                    registered, extra, path.c_str());
            if (charged > limit || extra > limit - charged) {
                return false;
            }
        }
        const size_t slash = path.find_last_of('/');
        if (slash < strlen("/sys/fs/cgroup")) {
            break;
        }
        path.resize(slash);
    }
    return true;
#else
    GGML_UNUSED(extra);
    GGML_UNUSED(registered);
    GGML_UNUSED(peak_current);
    return false;
#endif
}

// Tokens never depend on allocator addresses: even a session/table allocation
// reused at the same address must invalidate a previously captured executable.
static std::atomic<uint64_t> g_moe_cache_capture_generation{1};

uint64_t ggml_cuda_moe_cache_device_capture_generation(int logical) {
    const char * enabled = getenv("GGML_MOE_CACHE_DEVICE_CAPTURE");
    if (!enabled || strcmp(enabled, "1") != 0 || g_session_suppressed || g_session_stack.empty()) {
        return 0;
    }
    const auto * session = g_session_stack.back().active;
    if (!session || !session->device_ready.load(std::memory_order_acquire) ||
            session->devices.size() != 1 || session->devices[0]->logical != logical) {
        return 0;
    }
    return session->device_capture_generation;
}

void ggml_cuda_moe_cache_device_graph_record(int logical, bool captured, const ggml_cgraph * graph) {
    if (!ggml_cuda_moe_cache_device_capture_generation(logical)) {
        return;
    }
    auto * session = g_session_stack.back().active;
    (captured ? session->device_captures : session->device_replays).fetch_add(1, std::memory_order_relaxed);
    if (session->prefetch_enabled && graph) {
        // Replay bypasses the host dispatch hooks. Only count staged projections
        // present in this successfully launched graph, including fused gate/up.
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & copy : session->predictions) {
            if (!copy->published || copy->used) {
                continue;
            }
            for (int i = 0; i < graph->n_nodes; ++i) {
                const auto * node = graph->nodes[i];
                if (node->op == GGML_OP_MUL_MAT_ID && node->src[0] && node->src[0]->data == copy->tensor) {
                    copy->used = true;
                    break;
                }
            }
        }
    }
}

// Initial resident experts need no host pin. Before evicting one, retain an
// exact mapped copy of that expert only. A separate copy avoids overlapping
// page registrations and keeps all previously published fallbacks stable.
static bool moe_cache_device_fallback(moe_cache_session & session,
        moe_cache_device_table & table, int32_t expert) {
    if (table.fallback[expert]) {
        return true;
    }
    const size_t bytes = table.desc.expert_size;
    const size_t floor = 10ull << 30, margin = 2ull << 30;
    size_t registered = 0;
    for (const auto & range : session.mapped_ranges) {
        if (range.registered) {
            registered += range.end - range.begin;
        }
    }
    if (bytes > SIZE_MAX - floor - margin ||
            moe_cache_mem_available() < bytes + floor + margin ||
            !moe_cache_cgroup_headroom(bytes + margin, registered)) {
        return false;
    }
    moe_cache_mapped_range range;
    range.begin = (uintptr_t)table.desc.data + (size_t)expert * table.host_stride;
    range.end = range.begin + bytes;
    if (cudaHostAlloc(&range.staging, bytes, cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaHostGetDevicePointer(&range.device, range.staging, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        (void) cudaFreeHost(range.staging);
        return false;
    }
    memcpy(range.staging, (const void *)range.begin, bytes);
    try {
        session.mapped_ranges.push_back(range);
    } catch (const std::bad_alloc &) {
        (void) cudaFreeHost(range.staging);
        return false;
    }
    table.fallback[expert] = range.device;
    table.copy_source[expert] = range.staging;
    session.device_fallback_staging_bytes += bytes;
    return true;
}

// Called with session.mu held, after the last graph scope has left. A single
// boundary synchronization protects all captured table/slab references; no
// allocation they point at is replaced, and no layer waits for host routing.
static void moe_cache_device_adapt(moe_cache_session & session) {
    if (!session.device_expert_counts || session.devices.size() != 1 ||
            session.active_scopes || session.active_nodes) {
        return;
    }
    auto & device = *session.devices[0];
    const auto start = std::chrono::steady_clock::now();
    const size_t copy_limit = session.device_adapt_copy_cap;
    size_t copied = 0, pinned_source_bytes = 0, original_source_bytes = 0;
    int fills = 0, fallback_skips = 0, attempts = 0;
    try {
        std::vector<unsigned long long> counts(session.device_previous_counts.size());
        ggml_cuda_set_device(device.logical);
        if (cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(counts.data(), session.device_expert_counts,
                    counts.size() * sizeof(counts[0]), cudaMemcpyDeviceToHost) != cudaSuccess) {
            MOE_CACHE_LOG("[moe-cache] device adaptation skipped: counter read failed\n");
            (void) cudaGetLastError();
            return;
        }
        std::vector<unsigned long long> recent(counts.size());
        for (size_t i = 0; i < counts.size(); ++i) {
            recent[i] = counts[i] - session.device_previous_counts[i];
        }
        session.device_previous_counts.swap(counts);
        struct candidate {
            moe_cache_device_table * table;
            moe_cache_pool * pool;
            int32_t expert;
            uint64_t count;
        };
        struct victim { int slot; uint64_t count; };
        std::vector<candidate> missing;
        std::unordered_map<moe_cache_pool *, std::vector<victim>> victims;
        std::unordered_map<moe_cache_pool *, size_t> next_victim;
        for (auto & pool_ptr : device.pools) {
            auto & pool = *pool_ptr;
            auto & ranked = victims[&pool];
            for (int i = pool.lru_head; i >= 0; i = pool.slots[i].next) {
                auto & slot = pool.slots[i];
                const auto table = session.device_tables.find(slot.key.tensor);
                if (slot.state != moe_cache_slot_state::valid || slot.readers || slot.staging ||
                        table == session.device_tables.end()) {
                    continue;
                }
                const uint64_t used = recent[table->second.count_offset + slot.key.expert];
                slot.uses = (uint32_t)std::min<uint64_t>(UINT32_MAX, (uint64_t)slot.uses + used);
                ranked.push_back({i, used});
            }
            std::stable_sort(ranked.begin(), ranked.end(), [](const victim & a, const victim & b) {
                return a.count < b.count;
            });
            next_victim.emplace(&pool, 0);
        }
        for (auto & entry : session.device_tables) {
            auto & table = entry.second;
            moe_cache_pool * pool = nullptr;
            for (auto & p : device.pools) {
                if (p->slab && p->expert_size == table.desc.expert_size && p->wtype == table.desc.type) {
                    pool = p.get();
                    break;
                }
            }
            if (!pool) {
                continue;
            }
            for (int32_t e = 0; e < table.desc.n_expert; ++e) {
                const uint64_t used = recent[table.count_offset + e];
                if (used >= (uint64_t)std::max(1, session.config.readmit_after) &&
                        pool->map.find({table.desc.data, e}) == pool->map.end()) {
                    missing.push_back({&table, pool, e, used});
                }
            }
        }
        std::stable_sort(missing.begin(), missing.end(), [](const candidate & a, const candidate & b) {
            return a.count > b.count;
        });
        // Preallocate the publication list before any slot bytes change.
        std::vector<moe_cache_device_table *> changed;
        changed.reserve(session.device_tables.size());
        size_t max_experts = 0;
        for (const auto & entry : session.device_tables) {
            max_experts = std::max(max_experts, entry.second.addresses.size());
        }
        std::vector<uint8_t> hot(max_experts);
        for (auto & pool : device.pools) {
            pool->free_slots.reserve(pool->n_slots);
        }
        struct pending_fill {
            moe_cache_pool * pool;
            int index;
            moe_cache_device_table * table;
            moe_cache_key key;
            uint32_t uses;
            char * destination;
            const void * source;
            size_t bytes;
        };
        std::vector<pending_fill> queued;
        queued.reserve(256);
        auto mark_changed = [&](moe_cache_device_table * table) {
            if (std::find(changed.begin(), changed.end(), table) == changed.end()) {
                changed.push_back(table);
            }
        };
        for (const auto & c : missing) {
            if (++attempts > 256) {
                break;
            }
            auto & pool = *c.pool;
            if (pool.expert_size > copy_limit - copied) {
                continue;
            }
            int index;
            const bool free = !pool.free_slots.empty();
            if (free) {
                index = pool.free_slots.back();
            } else {
                auto & cursor = next_victim.at(&pool);
                const auto & ranked = victims.at(&pool);
                if (cursor == ranked.size() || ranked[cursor].count >= c.count) {
                    continue;
                }
                index = ranked[cursor++].slot;
                const auto & old_key = pool.slots[index].key;
                auto & old_table = session.device_tables.at(old_key.tensor);
                if (!moe_cache_device_fallback(session, old_table, old_key.expert)) {
                    // Preserve the resident and its table when no safe fallback fits.
                    ++fallback_skips;
                    continue;
                }
            }
            const moe_cache_key key{c.table->desc.data, c.expert};
            // A map allocation failure happens before the old slot is changed.
            try {
                if (!pool.map.emplace(key, index).second) {
                    continue;
                }
            } catch (const std::bad_alloc &) {
                break; // Publish all earlier replacements before returning.
            }
            auto & slot = pool.slots[index];
            if (free) {
                pool.free_slots.pop_back();
            } else {
                auto & old_table = session.device_tables.at(slot.key.tensor);
                old_table.addresses[slot.key.expert] = old_table.fallback[slot.key.expert];
                mark_changed(&old_table);
                device.evictions++;
            }
            moe_cache_slot_reset(pool, index, false);
            char * destination = pool.slab + (size_t)index * pool.expert_size;
            const void * source = c.table->copy_source[c.expert];
            if (source) {
                pinned_source_bytes += pool.expert_size;
            } else {
                source = (const char *)key.tensor + (size_t)key.expert * c.table->host_stride;
                original_source_bytes += pool.expert_size;
            }
            // Publish this slot only after the whole batch has landed. One
            // synchronous copy per expert measured about 2 ms, far below the
            // link, because each call waited before the next began.
            try {
                queued.push_back({&pool, index, c.table, key,
                        (uint32_t) std::min<uint64_t>(c.count, UINT32_MAX),
                        destination, source, pool.expert_size});
            } catch (const std::bad_alloc &) {
                pool.map.erase(key);
                pool.free_slots.push_back(index);
                break;
            }
            copied += pool.expert_size;
        }
        if (!queued.empty()) {
            if (!session.device_adapt_stream) {
                ggml_cuda_set_device(device.logical);
                if (cudaStreamCreateWithFlags(&session.device_adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
                    session.device_adapt_stream = nullptr;
                    (void) cudaGetLastError();
                }
            }
            cudaError_t copy_error = cudaSuccess;
            if (!session.device_adapt_stream) {
                for (const auto & fill : queued) {
                    copy_error = cudaMemcpy(fill.destination, fill.source, fill.bytes, cudaMemcpyHostToDevice);
                    if (copy_error != cudaSuccess) {
                        break;
                    }
                }
            } else {
                for (const auto & fill : queued) {
                    copy_error = cudaMemcpyAsync(fill.destination, fill.source, fill.bytes,
                            cudaMemcpyHostToDevice, session.device_adapt_stream);
                    if (copy_error != cudaSuccess) {
                        break;
                    }
                }
                if (copy_error == cudaSuccess) {
                    copy_error = cudaStreamSynchronize(session.device_adapt_stream);
                }
            }
            if (copy_error != cudaSuccess) {
                MOE_CACHE_LOG("[moe-cache] device adaptation batched copy failed: %s\n", cudaGetErrorString(copy_error));
                if (session.device_adapt_stream) {
                    (void) cudaStreamSynchronize(session.device_adapt_stream);
                }
                (void) cudaGetLastError();
                for (const auto & fill : queued) {
                    fill.pool->map.erase(fill.key);
                    fill.pool->free_slots.push_back(fill.index);
                    device.fill_failures++;
                }
            } else {
                for (const auto & fill : queued) {
                    auto & filled = fill.pool->slots[fill.index];
                    filled.key = fill.key;
                    filled.generation++;
                    filled.uses = fill.uses;
                    filled.state = moe_cache_slot_state::valid;
                    moe_cache_lru_push_back(*fill.pool, fill.index);
                    fill.table->addresses[fill.key.expert] = fill.destination;
                    mark_changed(fill.table);
                    device.fills++;
                    device.inserts++;
                    fills++;
                }
            }
        }
        for (auto * table : changed) {
            // Captured graphs retain this allocation; only its entries change.
            CUDA_CHECK(cudaMemcpy(table->device, table->addresses.data(),
                    table->addresses.size() * sizeof(void *), cudaMemcpyHostToDevice));
            if (table->staging_device) {
                CUDA_CHECK(cudaMemcpy(table->staging_device, table->addresses.data(),
                        table->addresses.size() * sizeof(void *), cudaMemcpyHostToDevice));
            }
            if (table->hot) {
                for (size_t e = 0; e < table->addresses.size(); ++e) {
                    hot[e] = table->addresses[e] != table->fallback[e];
                }
                CUDA_CHECK(cudaMemcpy(table->hot, hot.data(), table->addresses.size(), cudaMemcpyHostToDevice));
            }
        }
        MOE_CACHE_LOG("[moe-cache] device adaptation: scopes=%llu counter_bytes=%zu fills=%d copied_bytes=%zu cap_bytes=%zu fallback_staging_bytes=%zu fallback_skips=%d pinned_source_bytes=%zu original_source_bytes=%zu elapsed_ms=%.3f; stable tables\n",
                (unsigned long long)session.device_adapt_scopes,
                recent.size() * sizeof(recent[0]), fills, copied, copy_limit,
                session.device_fallback_staging_bytes, fallback_skips, pinned_source_bytes, original_source_bytes, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    } catch (const std::bad_alloc &) {
        MOE_CACHE_LOG("[moe-cache] device adaptation skipped: host allocation failed\n");
    }
}

static void moe_cache_device_release(moe_cache_session & session, bool retain_sources) {
    session.device_ready.store(false);
    if (session.device_adapt_stream) {
        if (!session.devices.empty()) {
            ggml_cuda_set_device(session.devices[0]->logical);
        }
        (void) cudaStreamDestroy(session.device_adapt_stream);
        session.device_adapt_stream = nullptr;
    }
    const uint64_t capture_generation = session.device_capture_generation;
    session.device_capture_generation = 0;
    const uint64_t captures = session.device_captures.exchange(0);
    const uint64_t replays = session.device_replays.exchange(0);
    if (captures || replays) {
        MOE_CACHE_LOG("[moe-cache] device graphs: captures=%llu replays=%llu generation=%llu\n",
                (unsigned long long)captures, (unsigned long long)replays,
                (unsigned long long)capture_generation);
    }
    if (!retain_sources) {
        session.mapped_sources.clear();
    }
    if (session.devices.empty() || (session.device_tables.empty() && session.mapped_ranges.empty())) {
        session.device_frozen = false;
        return;
    }
    ggml_cuda_set_device(session.devices[0]->logical);
    // Tables can be in flight on a scheduler stream, not the legacy cache stream.
    (void) cudaDeviceSynchronize();
    unsigned long long counts[3] = {};
    if (session.device_route_counts) {
        if (cudaMemcpy(counts, session.device_route_counts, sizeof(counts), cudaMemcpyDeviceToHost) == cudaSuccess) {
            MOE_CACHE_LOG("[moe-cache] device routes: hits=%llu misses=%llu dispatches=%llu; no host routing syncs\n",
                    counts[0], counts[1], counts[2]);
        }
        session.device_route_counts = nullptr;
    }
    if (session.device_expert_counts) {
        session.device_expert_counts = nullptr;
    }
    session.device_previous_counts.clear();
    session.device_adapt_scopes = 0;
    session.device_fallback_staging_bytes = 0;
    session.device_stats_scopes = 0;
    const uint64_t dispatches = session.device_dispatches.exchange(0);
    if (dispatches) {
        MOE_CACHE_LOG("[moe-cache] device dispatch: calls=%llu; legacy hit/miss counters exclude these routes\n",
                (unsigned long long)dispatches);
    }
    // Table pointers are views of the stable device table arena.
    if (!retain_sources) {
        for (auto & range : session.mapped_ranges) {
            if (range.registered) {
                (void) cudaHostUnregister((void *) range.begin);
            }
            if (range.staging) {
                (void) cudaFreeHost(range.staging);
            }
        }
        session.mapped_ranges.clear();
    }
    session.device_tables.clear();
    session.device_frozen = false;
    (void) cudaGetLastError();
}

static bool moe_cache_device_op_fits(const moe_cache_session & session, const ggml_tensor * op, int logical) {
    if (!session.device_ready.load() || session.devices.size() != 1 ||
            session.devices[0]->logical != logical || !op || op->op != GGML_OP_MUL_MAT_ID ||
            !op->src[0] || !op->src[1] || !op->src[2]) {
        return false;
    }
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * a = op->src[1];
    const ggml_tensor * ids = op->src[2];
    const auto found = session.device_tables.find(w->data);
    if (found == session.device_tables.end()) {
        return false;
    }
    const auto & d = found->second.desc;
    return w->type == d.type && w->ne[0] == d.n_in && w->ne[1] == d.n_out &&
        w->ne[2] == d.n_expert && w->ne[3] == 1 && ggml_is_contiguous_2(w) &&
        w->nb[1] == ggml_row_size(w->type, w->ne[0]) && w->nb[2] == found->second.host_stride &&
        a->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(a) && ggml_is_contiguous(op) &&
        ids->nb[0] == sizeof(int32_t) && ids->nb[1] % sizeof(int32_t) == 0 &&
        ids->nb[1] / sizeof(int32_t) <= INT_MAX / MMVQ_MAX_BATCH_SIZE && ids->nb[1] / sizeof(int32_t) >= (size_t)ids->ne[0] &&
        ids->ne[0] > 0 && ids->ne[0] <= 64 && ids->ne[1] > 0 &&
        ids->ne[1] <= get_mmvq_mmid_max_batch(w->type, ggml_cuda_info().devices[logical].cc) &&
        ids->ne[2] == 1 && ids->ne[3] == 1 && a->ne[0] == d.n_in &&
        a->ne[1] > 0 && a->ne[1] <= ids->ne[0] && a->ne[2] == ids->ne[1] && a->ne[3] == 1 &&
        op->ne[0] == d.n_out && op->ne[1] == ids->ne[0] && op->ne[2] == ids->ne[1] && op->ne[3] == 1 &&
        ggml_get_op_params_i32(op, 3) != GGML_PREC_F32;
}

static int moe_cache_device_supports(void * opaque, void * backend, const ggml_tensor * op) {
    auto * session = (moe_cache_session *) opaque;
    auto * b = (ggml_backend_t) backend;
    if (!session || !b || !ggml_backend_is_cuda(b)) {
        return 0;
    }
    auto * ctx = (ggml_backend_cuda_context *) b->context;
    return moe_cache_device_op_fits(*session, op, ctx->device);
}

bool ggml_cuda_moe_cache_device_supports(const ggml_tensor * op, int logical) {
    if (!op || op->op != GGML_OP_MUL_MAT_ID || !getenv("GGML_MOE_CACHE_DEVICE")) {
        return false;
    }
    if (g_session_suppressed > 0) {
        return false;
    }
    // Active scopes keep tables alive while invalidation waits with the registry locked.
    if (!g_session_stack.empty()) {
        const auto * session = g_session_stack.back().active;
        return session && moe_cache_device_op_fits(*session, op, logical);
    }
    std::lock_guard<std::mutex> lock(g_registry_mu);
    for (const auto * session : g_sessions) {
        if (moe_cache_device_op_fits(*session, op, logical)) {
            return true;
        }
    }
    return false;
}

// Per-expert counters remain on the card between bounded idle boundaries.
static __global__ void moe_cache_count_routes(const int32_t * ids, const uint8_t * hot,
        unsigned long long * counts, unsigned long long * expert_counts, int n, int experts, int n_ids, int ids_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && counts) {
        atomicAdd(counts + 2, 1ULL);
    }
    if (i < n) {
        const int32_t expert = ids[(i / n_ids) * ids_stride + i % n_ids];
        if (expert >= 0 && expert < experts) {
            atomicAdd(expert_counts + expert, 1ULL);
            if (counts) {
                atomicAdd(counts + (hot[expert] ? 0 : 1), 1ULL);
            }
        }
    }
}

bool ggml_cuda_moe_cache_device_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    if (g_session_stack.empty() || !g_session_stack.back().active ||
            !moe_cache_device_op_fits(*g_session_stack.back().active, op, ctx.device)) {
        return false;
    }
    auto & session = *g_session_stack.back().active;
    const auto & table = session.device_tables.at(op->src[0]->data);
    const ggml_tensor * a = op->src[1];
    const ggml_tensor * ids = op->src[2];
    session.device_dispatches.fetch_add(1, std::memory_order_relaxed);
    if (session.device_expert_counts) {
        const int n = (int)(ids->ne[0] * ids->ne[1]);
        moe_cache_count_routes<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(
                (const int32_t *)ids->data, table.hot, session.device_route_counts,
                session.device_expert_counts + table.count_offset, n, table.desc.n_expert, (int)ids->ne[0], (int)(ids->nb[1] / sizeof(int32_t)));
    }
    const int64_t padded = GGML_PAD(a->ne[0], MATRIX_ROW_PADDING);
    const size_t q8_bytes = (size_t)(padded / QK8_1) * a->ne[1] * a->ne[2] * sizeof(block_q8_1);
    ggml_cuda_pool_alloc<char> q8(ctx.pool());
    char * quantized = q8.alloc(q8_bytes);
    quantize_row_q8_1_cuda((const float *) a->data, nullptr, quantized, (ggml_type) table.desc.type,
            a->ne[0], a->nb[1] / sizeof(float), a->nb[2] / sizeof(float), a->nb[3] / sizeof(float),
            padded, a->ne[1], a->ne[2], 1, ctx.stream());
    ggml_cuda_moe_table_mmv(table.staging_device ? table.staging_device : table.device, (ggml_type) table.desc.type, quantized,
            (const int32_t *) ids->data, (float *) op->data,
            a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1], ids->nb[1] / sizeof(int32_t), ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    if (table.staging_device) {
        std::lock_guard<std::mutex> lock(session.mu);
        for (const auto & copy : session.predictions) {
            if (copy->tensor == op->src[0]->data && copy->published) {
                copy->used = true;
            }
        }
    }
    return true;
}

// fork: one activation quantization and one MMV retain both exact expert projections.
bool ggml_cuda_moe_cache_device_fused_compute(ggml_backend_cuda_context & ctx,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * glu) {
    const char * enabled = getenv("GGML_MOE_CACHE_DEVICE_FUSED");
    if (!enabled || strcmp(enabled, "1") != 0 || g_session_stack.empty() || !g_session_stack.back().active ||
            !up || !gate || !glu || glu->op != GGML_OP_GLU ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
            glu->src[0] != gate || glu->src[1] != up ||
            !moe_cache_device_op_fits(*g_session_stack.back().active, up, ctx.device) ||
            !moe_cache_device_op_fits(*g_session_stack.back().active, gate, ctx.device) ||
            up->src[0]->type != gate->src[0]->type || up->src[1] != gate->src[1] || up->src[2] != gate->src[2] ||
            !ggml_are_same_shape(up, gate) || !ggml_are_same_shape(up, glu) ||
            glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return false;
    }
    auto & session = *g_session_stack.back().active;
    const auto & up_table = session.device_tables.at(up->src[0]->data);
    const auto & gate_table = session.device_tables.at(gate->src[0]->data);
    const ggml_tensor * a = up->src[1];
    const ggml_tensor * ids = up->src[2];
    session.device_dispatches.fetch_add(2, std::memory_order_relaxed);
    if (session.device_expert_counts) {
        const int n = (int)(ids->ne[0] * ids->ne[1]);
        for (const auto * table : {&up_table, &gate_table}) {
            moe_cache_count_routes<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(
                    (const int32_t *)ids->data, table->hot, session.device_route_counts,
                    session.device_expert_counts + table->count_offset, n, table->desc.n_expert,
                    (int)ids->ne[0], (int)(ids->nb[1] / sizeof(int32_t)));
        }
    }
    const int64_t padded = GGML_PAD(a->ne[0], MATRIX_ROW_PADDING);
    const size_t q8_bytes = (size_t)(padded / QK8_1) * a->ne[1] * a->ne[2] * sizeof(block_q8_1);
    ggml_cuda_pool_alloc<char> q8(ctx.pool());
    char * quantized = q8.alloc(q8_bytes);
    quantize_row_q8_1_cuda((const float *)a->data, nullptr, quantized, (ggml_type)up_table.desc.type,
            a->ne[0], a->nb[1] / sizeof(float), a->nb[2] / sizeof(float), a->nb[3] / sizeof(float),
            padded, a->ne[1], a->ne[2], 1, ctx.stream());
    ggml_cuda_moe_table_mmv_fused(up_table.staging_device ? up_table.staging_device : up_table.device,
            gate_table.staging_device ? gate_table.staging_device : gate_table.device, (ggml_type)up_table.desc.type, quantized,
            (const int32_t *)ids->data, (float *)glu->data, a->ne[0], glu->ne[0], ids->ne[0], ids->ne[1],
            a->ne[1], ids->nb[1] / sizeof(int32_t), ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    if (up_table.staging_device || gate_table.staging_device) {
        std::lock_guard<std::mutex> lock(session.mu);
        for (const auto & copy : session.predictions) {
            if (copy->published && (copy->tensor == up->src[0]->data || copy->tensor == gate->src[0]->data)) {
                copy->used = true;
            }
        }
    }
    return true;
}

// Host-source mapping coverage index. Page-rounded ranges and exact victim
// copies may overlap, so a prefix maximum is required rather than just the
// immediately preceding range. The device addresses stay fixed across turns.
class moe_cache_mapping_index {
public:
    explicit moe_cache_mapping_index(const std::vector<moe_cache_mapped_range> & ranges) : ranges(ranges) { reset(); }
    void reset() {
        order.resize(ranges.size());
        for (size_t i = 0; i < order.size(); ++i) { order[i] = i; }
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ranges[a].begin < ranges[b].begin; });
        prefix.resize(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            prefix[i] = i && ranges[prefix[i - 1]].end > ranges[order[i]].end ? prefix[i - 1] : order[i];
        }
    }
    const void * find(uintptr_t begin, size_t bytes, size_t * found = nullptr) const {
        const auto end = std::upper_bound(order.begin(), order.end(), begin,
                [&](uintptr_t address, size_t index) { return address < ranges[index].begin; });
        if (end == order.begin()) { return nullptr; }
        const size_t index = prefix[(size_t)(end - order.begin() - 1)];
        const auto & range = ranges[index];
        if (!range.device || begin < range.begin || begin >= range.end || bytes > range.end - begin) {
            return nullptr;
        }
        if (found) { *found = index; }
        return (const char *)range.device + (begin - range.begin);
    }
private:
    const std::vector<moe_cache_mapped_range> & ranges;
    std::vector<size_t> order, prefix;
};

static int moe_cache_device_prepare(void * opaque, const ggml_moe_cache_tensor_desc * tensors, int n_tensors) {
#if defined(__linux__) && CUDART_VERSION >= 11010 && !defined(GGML_USE_HIP)
    auto * session = (moe_cache_session *) opaque;
    const char * enabled = getenv("GGML_MOE_CACHE_DEVICE");
    if (!session || !enabled || strcmp(enabled, "1") != 0 || !tensors || n_tensors < 1 || n_tensors > 4096 ||
            session->devices.size() != 1 || session->dormant || session->stopping) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(session->mu);
    if (session->device_ready.load()) {
        return 1;
    }
    if (session->device_attempted || session->active_nodes || session->active_scopes) {
        return 0;
    }
    session->device_attempted = true;
    // Only the bounded idle-boundary refill schedule changes; kernels and
    // weight bytes stay exact. Defaults preserve the original policy.
    session->device_adapt_interval = 64;
    session->device_adapt_copy_cap = 64u << 20;
    int64_t adaptation_value = 0;
    if (moe_cache_env_i64("GGML_MOE_CACHE_DEVICE_ADAPT_INTERVAL", 1, 1024, adaptation_value)) {
        session->device_adapt_interval = (uint64_t) adaptation_value;
    }
    if (moe_cache_env_i64("GGML_MOE_CACHE_DEVICE_ADAPT_MB", 1, 1024, adaptation_value)) {
        session->device_adapt_copy_cap = (size_t) adaptation_value << 20;
    }
    MOE_CACHE_LOG("[moe-cache] device adaptation policy: interval_scopes=%llu cap_bytes=%zu; recent-generation-counts only\n",
            (unsigned long long) session->device_adapt_interval, session->device_adapt_copy_cap);
    auto & device = *session->devices[0];
    if (device.dead.load()) {
        return 0;
    }
    // Freeze only after the worker has drained all queued prompt fills.
    const bool idle = session->idle_cv.wait_for(lock, std::chrono::seconds(30), [&] {
        return device.queue.empty() && !device.inflight;
    });
    if (!idle) {
        MOE_CACHE_LOG("[moe-cache] device tables skipped: fill queue did not drain\n");
        return 0;
    }
    if (session->prefetch_enabled) {
        for (auto & pool : device.pools) {
            for (int i = 0; i < 4; ++i) {
                (void) moe_prefetch_stage_slot(*pool, device, false, true);
            }
        }
    }
    session->device_frozen = true;
    auto fail = [&]() {
        moe_cache_device_release(*session);
        MOE_CACHE_LOG("[moe-cache] device tables unavailable; continuing legacy exact CPU/cache mode\n");
        return 0;
    };
    try {
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
        if (!page || (page & (page - 1))) {
            return fail();
        }
        if (!session->mapped_sources.empty()) {
            bool identical = session->mapped_sources.size() == (size_t)n_tensors;
            for (int i = 0; i < n_tensors && identical; ++i) {
                const auto found = session->mapped_sources.find(tensors[i].data);
                if (found == session->mapped_sources.end()) { identical = false; break; }
                const auto & old = found->second;
                const auto & now = tensors[i];
                identical = old.expert_size == now.expert_size && old.n_in == now.n_in &&
                        old.n_out == now.n_out && old.n_expert == now.n_expert && old.type == now.type;
            }
            if (!identical) {
                MOE_CACHE_LOG("[moe-cache] retained source mapping identity changed; releasing old exact copies\n");
                moe_cache_device_release(*session);
                session->device_frozen = true;
            }
        }
        size_t retained_ranges = session->mapped_ranges.size();
        size_t retained_staging = 0, retained_registered = 0;
        std::vector<uint8_t> retained_needed(retained_ranges, 0);
        moe_cache_mapping_index mappings(session->mapped_ranges);
        std::vector<moe_cache_mapped_range> pending_ranges;
        const char * stats_env = getenv("GGML_MOE_CACHE_DEVICE_STATS");
        const bool route_stats = stats_env && strcmp(stats_env, "1") == 0;
        size_t table_bytes = route_stats ? 3 * sizeof(unsigned long long) : 0, hot_bytes = 0;
        size_t count_entries = 0, arena_bytes = 0;
        for (int i = 0; i < n_tensors; ++i) {
            const auto & d = tensors[i];
            // The legacy cache supports IQ1_M; the mapped pointer specialization
            // is disabled until its card-side read failure has a validated fix.
            if (d.type == GGML_TYPE_IQ1_M) {
                MOE_CACHE_LOG("[moe-cache] device tables skipped: IQ1_M mapped dispatch is not supported; legacy cache retained\n");
                return fail();
            }
            if (!d.data || !ggml_moe_cache_wtype_supported(d.type) || d.n_in < 1 || d.n_in > INT_MAX ||
                    d.n_out < 1 || d.n_out > INT_MAX || d.n_expert < 1 || d.n_expert > 65536 ||
                    !d.expert_size || d.expert_size > SIZE_MAX / (size_t) d.n_expert ||
                    d.n_in % ggml_blck_size((ggml_type)d.type) != 0 ||
                    ggml_row_size((ggml_type)d.type, d.n_in) > d.expert_size / (size_t)d.n_out) {
                return fail();
            }
            const uintptr_t base = (uintptr_t) d.data;
            const size_t expert_bytes = ggml_row_size((ggml_type)d.type, d.n_in) * (size_t)d.n_out;
            const size_t bytes = d.expert_size * (size_t)(d.n_expert - 1) + expert_bytes;
            if (base > UINTPTR_MAX - bytes || base + bytes > UINTPTR_MAX - page ||
                    session->device_tables.count(d.data)) {
                return fail();
            }
            moe_cache_device_table table;
            table.desc = d;
            table.desc.expert_size = expert_bytes;
            table.host_stride = d.expert_size;
            cudaPointerAttributes attributes{};
            if (cudaPointerGetAttributes(&attributes, d.data) == cudaSuccess && attributes.type == cudaMemoryTypeHost) {
                void * alias = nullptr;
                if (cudaHostGetDevicePointer(&alias, const_cast<void *>(d.data), 0) == cudaSuccess) {
                    table.borrowed = alias;
                }
            }
            (void)cudaGetLastError();
            table.addresses.resize(d.n_expert);
            table.fallback.resize(d.n_expert);
            table.copy_source.resize(d.n_expert);
            table.count_offset = count_entries;
            count_entries += d.n_expert;
            for (int32_t e = 0; e < d.n_expert; ++e) {
                const void * address = nullptr;
                for (const auto & pool : device.pools) {
                    if (pool->expert_size != expert_bytes || pool->wtype != d.type) {
                        continue;
                    }
                    const auto found = pool->map.find({d.data, e});
                    if (found != pool->map.end() && pool->slots[found->second].state == moe_cache_slot_state::valid &&
                            !pool->slots[found->second].staging) {
                        address = pool->slab + (size_t) found->second * expert_bytes;
                        hot_bytes += expert_bytes;
                        break;
                    }
                }
                const uintptr_t begin = base + (size_t)e * d.expert_size;
                size_t source_range = 0;
                table.fallback[e] = table.borrowed ? (const char *)table.borrowed + (size_t)e * table.host_stride :
                    mappings.find(begin, expert_bytes, &source_range);
                if (!table.borrowed && !address && table.fallback[e]) { retained_needed[source_range] = 1; }
                if (!address && !table.fallback[e]) {
                    pending_ranges.push_back({begin & ~(page - 1),
                            (begin + expert_bytes + page - 1) & ~(page - 1), nullptr, nullptr, false});
                }
                table.addresses[e] = address;
            }
            table_bytes += (size_t) d.n_expert * (sizeof(void *) * (session->prefetch_enabled ? 2 : 1) + sizeof(unsigned long long) + (route_stats ? sizeof(uint8_t) : 0));
            if (!moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert * sizeof(void *)) ||
                    (session->prefetch_enabled && !moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert * sizeof(void *))) ||
                    (route_stats && !moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert))) {
                return fail();
            }
            session->device_tables.emplace(d.data, std::move(table));
        }
        if (!moe_cache_table_bytes_add(arena_bytes, count_entries * sizeof(unsigned long long)) ||
                (route_stats && !moe_cache_table_bytes_add(arena_bytes, 3 * sizeof(unsigned long long)))) {
            return fail();
        }
        // Retire ranges no current miss needs before adding new exact copies.
        // Old kernels and tables were drained by the idle rerank; no published
        // address can observe this pruning. Unchanged cold sets copy nothing.
        size_t kept = 0, pruned_bytes = 0;
        for (size_t i = 0; i < retained_ranges; ++i) {
            auto & range = session->mapped_ranges[i];
            if (!retained_needed[i]) {
                if (range.registered) { (void)cudaHostUnregister((void *)range.begin); }
                if (range.staging) { (void)cudaFreeHost(range.staging); }
                pruned_bytes += range.end - range.begin;
                continue;
            }
            if (range.staging) { retained_staging += range.end - range.begin; }
            if (range.registered) { retained_registered += range.end - range.begin; }
            if (kept != i) { session->mapped_ranges[kept] = range; }
            ++kept;
        }
        session->mapped_ranges.resize(kept);
        retained_ranges = kept;
        mappings.reset();
        MOE_CACHE_LOG("[moe-cache] source mappings prune: released_bytes=%zu retained_ranges=%zu policy=current-nonresident-experts-only\n",
                pruned_bytes, retained_ranges);
        auto & ranges = pending_ranges;
        std::sort(ranges.begin(), ranges.end(), [](const moe_cache_mapped_range & a, const moe_cache_mapped_range & b) {
            return a.begin < b.begin;
        });
        const size_t staging_chunk_cap = 64u << 20;
        size_t merged = 0;
        for (size_t i = 0; i < ranges.size(); ++i) {
            if (ranges[i].end - ranges[i].begin > staging_chunk_cap) {
                return fail();
            }
            if (merged && ggml_cuda_moe_cache_host_merge(ranges[merged - 1].begin, ranges[merged - 1].end,
                        ranges[i].begin, ranges[i].end, staging_chunk_cap)) {
                ranges[merged - 1].end = std::max(ranges[merged - 1].end, ranges[i].end);
            } else {
                ranges[merged++] = ranges[i];
            }
        }
        ranges.resize(merged);
        size_t pin_bytes = 0, new_bytes = 0;
        for (const auto & range : ranges) {
            const size_t bytes = range.end - range.begin;
            if (bytes > SIZE_MAX - pin_bytes) {
                return fail();
            }
            pin_bytes += bytes;
            std::vector<unsigned char> residency(bytes / page);
            if (mincore((void *)range.begin, bytes, residency.data()) != 0) {
                return fail();
            }
            for (auto resident : residency) {
                new_bytes += (resident & 1) ? 0 : page;
            }
        }
        const size_t margin = ranges.empty() ? 0 : 2ull << 30;
        const size_t floor = 7ull << 30;
        const size_t available = moe_cache_mem_available();
        MOE_CACHE_LOG("[moe-cache] device preflight: hot=%zu MiB map=%zu MiB nonresident=%zu MiB ranges=%zu available=%zu MiB\n",
                hot_bytes >> 20, pin_bytes >> 20, new_bytes >> 20, ranges.size(), available >> 20);
        if (pin_bytes > SIZE_MAX - floor - margin || available < pin_bytes + floor + margin ||
                new_bytes > SIZE_MAX - margin || !moe_cache_cgroup_headroom(new_bytes + margin, retained_registered)) {
            MOE_CACHE_LOG("[moe-cache] device tables skipped: pinning would exceed RAM floor or service headroom\n");
            return fail();
        }
        ggml_cuda_set_device(device.logical);
        size_t free_bytes = 0, total_bytes = 0;
        // Tables are mandatory generation workspace, not optional expert
        // slots. They may spend a bounded part of the configured generation
        // reserve. Keep at least 1 GiB physically free on the home card so
        // its driver allocations also fit below the 23,552 MiB card limit.
        const size_t table_workspace_cap = 64u << 20;
        const size_t physical_free_floor = 1ull << 30;
        moe_cache_memory_view table_memory;
        const cudaError_t table_query = cudaMemGetInfo(&free_bytes, &total_bytes);
        const bool budget_query = table_query == cudaSuccess && moe_cache_memory_query(device, table_memory);
        const char * table_failure = table_query != cudaSuccess ? "cuda-memory-query" :
                !budget_query ? "budget-query" : table_bytes > table_workspace_cap ? "table-cap" :
                arena_bytes > device.table_arena_cap ? "arena-not-materialized" :
                free_bytes < physical_free_floor ? "physical-free-floor" : nullptr;
        MOE_CACHE_LOG("[moe-cache] CUDA%d device table workspace bytes: requested=%zu arena_required=%zu arena_allocated=%zu cap=%zu free=%zu total=%zu reserve=%zu other_outstanding=%zu physical_free_floor=%zu failure=%s policy=materialized-before-expert-pools\n",
                device.physical, table_bytes, arena_bytes, device.table_arena_cap, table_workspace_cap,
                free_bytes, total_bytes, table_memory.reserve, table_memory.other_outstanding,
                physical_free_floor, table_failure ? table_failure : "none");
        if (table_failure) {
            return fail();
        }
        size_t arena_used = 0;
        auto table_take = [&](size_t bytes) -> void * {
            const size_t padded = (bytes + 255) & ~(size_t)255;
            if (arena_used > device.table_arena_cap || padded > device.table_arena_cap - arena_used) {
                return nullptr;
            }
            void * pointer = device.table_arena + arena_used;
            arena_used += padded;
            return pointer;
        };
        MOE_CACHE_LOG("[moe-cache] source mappings reuse: retained_ranges=%zu retained_copied_bytes=%zu retained_registered_bytes=%zu new_ranges=%zu newly_uncovered_bytes=%zu\n",
                retained_ranges, retained_staging, retained_registered, ranges.size(), pin_bytes);
        session->mapped_ranges.insert(session->mapped_ranges.end(), ranges.begin(), ranges.end());
        int readonly_supported = 0;
        if (cudaDeviceGetAttribute(&readonly_supported, cudaDevAttrHostRegisterReadOnlySupported,
                    device.logical) != cudaSuccess) {
            (void) cudaGetLastError();
            readonly_supported = 0;
        }
        const unsigned int registration_flags = cudaHostRegisterPortable | cudaHostRegisterMapped |
                (readonly_supported ? cudaHostRegisterReadOnly : 0);
        MOE_CACHE_LOG("[moe-cache] mapped registration: driver read-only protection=%d; kernels only read expert weights\n",
                readonly_supported);
        size_t staging_bytes = retained_staging, registered_bytes = retained_registered;
        uint64_t peak_service_current = 0;
        const auto mapping_start = std::chrono::steady_clock::now();
        size_t registration_calls = 0, staging_calls = 0;
        double registration_ms = 0, staging_alloc_ms = 0, source_copy_ms = 0;
        for (size_t range_index = retained_ranges; range_index < session->mapped_ranges.size(); ++range_index) {
            auto & range = session->mapped_ranges[range_index];
            if (moe_cache_mem_available() < floor + margin) {
                return fail();
            }
            const size_t bytes = range.end - range.begin;
            // Page rounding must never register a neighboring heap allocation.
            // Transfers spanning registered and unregistered host pages are not
            // valid CUDA copies, even when the expert kernels only read weights.
            bool contained = false;
            for (const auto & entry : session->device_tables) {
                const auto & d = entry.second.desc;
                const uintptr_t bank_begin = (uintptr_t)d.data;
                const size_t bank_bytes = entry.second.host_stride * (size_t)(d.n_expert - 1) + d.expert_size;
                if (range.begin >= bank_begin && range.end - bank_begin <= bank_bytes) {
                    contained = true;
                    break;
                }
            }
            const auto registration_start = std::chrono::steady_clock::now();
            const cudaError_t err = contained
                ? cudaHostRegister((void *)range.begin, bytes, registration_flags)
                : cudaErrorInvalidValue;
            registration_calls += contained;
            registration_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - registration_start).count();
            if (!contained) {
                MOE_CACHE_LOG("[moe-cache] mapped registration boundary copy: bytes=%zu; neighboring host pages preserved\n", bytes);
            }
            if (err == cudaSuccess) {
                range.registered = true;
                registered_bytes += bytes;
                if (cudaHostGetDevicePointer(&range.device, (void *)range.begin, 0) != cudaSuccess) {
                    return fail();
                }
                continue;
            }
            {
                static int registration_failures_logged = 0;
                if (registration_failures_logged < 1) {
                    registration_failures_logged++;
                    MOE_CACHE_LOG("[moe-cache] mapped registration failed: %s bytes=%zu; a private writable file view is required to pin in place\n",
                            cudaGetErrorString(err), bytes);
                }
            }
            (void) cudaGetLastError();
            // Read-only file maps cannot be registered on this driver. A small
            // pinned copy is exact. Copying the whole cold expert set (tens of
            // GiB) is slower than the legacy cache, so refuse that and keep it.
            const size_t staging_cap = 1ull << 30;
            if (bytes > staging_cap || staging_bytes - retained_staging > staging_cap - bytes) {
                MOE_CACHE_LOG("[moe-cache] mapped staging refused: next copy would exceed the %zu MiB fallback cap after %zu MiB; legacy exact cache remains\n",
                        staging_cap >> 20, (staging_bytes - retained_staging) >> 20);
                return fail();
            }
            // Read-only file maps cannot be registered on some drivers. Keep a bounded pinned copy of missing experts.
            if (moe_cache_mem_available() < bytes + floor + margin ||
                    !moe_cache_cgroup_headroom(bytes + margin, registered_bytes, &peak_service_current)) {
                MOE_CACHE_LOG("[moe-cache] mapped staging skipped: insufficient RAM for %zu MiB after registration failure (%s)\n",
                        bytes >> 20, cudaGetErrorString(err));
                return fail();
            }
            const auto staging_start = std::chrono::steady_clock::now();
            const cudaError_t staging_error = cudaHostAlloc(&range.staging, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
            ++staging_calls;
            staging_alloc_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - staging_start).count();
            if (staging_error != cudaSuccess) {
                return fail();
            }
            const auto copy_start = std::chrono::steady_clock::now();
            memcpy(range.staging, (const void *)range.begin, bytes);
            source_copy_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - copy_start).count();
            if (cudaHostGetDevicePointer(&range.device, range.staging, 0) != cudaSuccess) {
                return fail();
            }
            staging_bytes += bytes;
            if (moe_cache_mem_available() < floor + margin ||
                    !moe_cache_cgroup_headroom(margin, registered_bytes, &peak_service_current)) {
                return fail();
            }
        }
        MOE_CACHE_LOG("[moe-cache] missing expert mapping: pinned staging=%zu MiB, %.3f ms; source bytes preserved\n",
                staging_bytes >> 20, std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - mapping_start).count());
        MOE_CACHE_LOG("[moe-cache] source mappings setup: new_copy_bytes=%zu new_registered_bytes=%zu retained_copy_bytes=%zu total_ranges=%zu\n",
                staging_bytes - retained_staging, registered_bytes - retained_registered,
                retained_staging, session->mapped_ranges.size());
        MOE_CACHE_LOG("[moe-cache] source mappings timing: registration_calls=%zu registration_ms=%.3f staging_calls=%zu staging_alloc_ms=%.3f source_copy_ms=%.3f\n",
                registration_calls, registration_ms, staging_calls, staging_alloc_ms, source_copy_ms);
        mappings.reset();
        session->device_previous_counts.assign(count_entries, 0);
        session->device_expert_counts = (unsigned long long *)table_take(count_entries * sizeof(unsigned long long));
        if (!session->device_expert_counts ||
                cudaMemset(session->device_expert_counts, 0, count_entries * sizeof(unsigned long long)) != cudaSuccess) {
            return fail();
        }
        if (route_stats) {
            session->device_route_counts = (unsigned long long *)table_take(3 * sizeof(unsigned long long));
            if (!session->device_route_counts ||
                    cudaMemset(session->device_route_counts, 0, 3 * sizeof(unsigned long long)) != cudaSuccess) {
                return fail();
            }
        }
        for (auto & entry : session->device_tables) {
            auto & table = entry.second;
            if (route_stats) {
                std::vector<uint8_t> hot(table.addresses.size());
                for (size_t e = 0; e < hot.size(); ++e) {
                    hot[e] = table.addresses[e] != nullptr;
                }
                table.hot = (uint8_t *)table_take(hot.size());
                if (!table.hot || cudaMemcpy(table.hot, hot.data(), hot.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
                    return fail();
                }
            }
            for (size_t e = 0; e < table.addresses.size(); ++e) {
                const uintptr_t host = (uintptr_t)table.desc.data + e * table.host_stride;
                size_t source_range = 0;
                table.fallback[e] = table.borrowed ? (const char *)table.borrowed + e * table.host_stride :
                    mappings.find(host, table.desc.expert_size, &source_range);
                table.copy_source[e] = table.borrowed ? (const void *)host : nullptr;
                if (table.fallback[e] && !table.borrowed) {
                    // find() proved complete containment in this descriptor-bound
                    // retained range. Never use its CUDA alias as a host pointer.
                    const auto & range = session->mapped_ranges[source_range];
                    if (!range.staging && !range.registered) { return fail(); }
                    table.copy_source[e] = range.staging
                        ? (const char *) range.staging + (host - range.begin)
                        : (const void *) host;
                }
                if (!table.addresses[e]) {
                    if (!table.fallback[e]) { return fail(); }
                    table.addresses[e] = table.fallback[e];
                }
            }
            const size_t bytes = table.addresses.size() * sizeof(void *);
            if (session->prefetch_enabled) {
                table.staging_device = (const void **) table_take(bytes);
                if (!table.staging_device || cudaMemcpy(table.staging_device, table.addresses.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                    return fail();
                }
            }
            table.device = (const void **)table_take(bytes);
            if (!table.device || cudaMemcpy(table.device, table.addresses.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                return fail();
            }

        }
        if (moe_cache_mem_available() < floor ||
                !moe_cache_cgroup_headroom(margin, registered_bytes, &peak_service_current)) {
            return fail();
        }
        MOE_CACHE_LOG("[moe-cache] host staging bytes: copied=%zu registered=%zu chunk_cap=%zu peak_service_current=%llu available=%zu\n",
                staging_bytes, registered_bytes, staging_chunk_cap,
                (unsigned long long) peak_service_current, moe_cache_mem_available());
        session->mapped_sources.clear();
        for (int i = 0; i < n_tensors; ++i) { session->mapped_sources.emplace(tensors[i].data, tensors[i]); }
        session->device_capture_generation = g_moe_cache_capture_generation.fetch_add(1, std::memory_order_relaxed);
        GGML_ASSERT(session->device_capture_generation != 0);
        session->device_ready.store(true, std::memory_order_release);
        MOE_CACHE_LOG("[moe-cache] device tables active: %zu tensors, %zu MiB mapped existing pages, %zu KiB device tables; stable slots with boundary adaptation; no per-layer host routing or result copies\n",
                session->device_tables.size(), (pin_bytes + retained_staging + retained_registered) >> 20, table_bytes >> 10);
        return 1;
    } catch (const std::exception & e) {
        MOE_CACHE_LOG("[moe-cache] device preparation failed: %s\n", e.what());
        return fail();
    }
#else
    GGML_UNUSED(opaque);
    GGML_UNUSED(tensors);
    GGML_UNUSED(n_tensors);
    return 0;
#endif
}

static void moe_cache_register(const void * owner) {
    g_moe_cache_owner = owner;
    ggml_moe_cache_api api = {};
    api.owner = owner;
    api.session_status = moe_cache_session_status;
    api.query_config = moe_cache_query_config;
    api.query_device = moe_cache_query_device;
    api.query_shape = moe_cache_query_shape;
    api.session_create = moe_cache_session_create;
    api.session_destroy = moe_cache_session_destroy;
    api.prefetch_configure = moe_cache_prefetch_configure;
    api.prefetch = moe_cache_prefetch;
    api.prefetch_observe = moe_cache_prefetch_observe;
    api.session_enter = moe_cache_session_enter;
    api.session_leave = moe_cache_session_leave;
    api.begin = moe_cache_begin;
    api.plan = moe_cache_plan;
    api.dispatch = moe_cache_dispatch;
    api.collect = moe_cache_collect;
    api.end = moe_cache_end;
    api.fused_begin = moe_cache_fused_begin;
    api.invalidate = moe_cache_invalidate;
    api.warm = moe_cache_warm;
    api.print_stats = moe_cache_print_stats;
    api.device_prepare = moe_cache_device_prepare;
    api.device_supports = moe_cache_device_supports;
    api.trim = moe_cache_lend;
    api.restore = moe_cache_return;
    ggml_moe_cache_register(&api);
}

#endif // !defined(GGML_USE_MUSA)

#if defined(GGML_USE_MUSA)
void ggml_cuda_moe_cache_register(void * reg) {
    (void) reg;
    moe_cache_register_stub(reg);
}
#else
void ggml_cuda_moe_cache_register(void * reg) {
    moe_cache_register(reg);
}
#endif
