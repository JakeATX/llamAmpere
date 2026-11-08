#include "moe-cache.cuh"
#include "ai4-mm.cuh"

#if defined(GGML_USE_MUSA)

extern "C" size_t ggml_moe_cache_trim(int device) {
    (void) device;
    return 0;
}

uint64_t ggml_cuda_moe_cache_device_capture_generation(int) { return 0; }
void ggml_cuda_moe_cache_device_graph_record(int, bool, const ggml_cgraph *) {}
bool ggml_cuda_moe_cache_device_supports(const ggml_tensor *, int) { return false; }
bool ggml_cuda_moe_cache_device_compute(ggml_backend_cuda_context &, ggml_tensor *) { return false; }
bool ggml_cuda_moe_cache_mixed_compute(ggml_backend_cuda_context &, const ggml_tensor *, ggml_tensor *, ggml_tensor *, const ggml_tensor *, ggml_tensor *,
        const ggml_cuda_moe_mixed_reduce *) { return false; }
bool ggml_cuda_moe_cache_device_fused_compute(ggml_backend_cuda_context &, ggml_tensor *, ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_moe_cache_prefetch_async_active(void) { return false; }
void ggml_cuda_moe_cache_device_predict(ggml_backend_cuda_context &, const ggml_tensor *) {}
void ggml_cuda_moe_cache_table_graph_begin(ggml_backend_cuda_context &, bool) {}
void ggml_cuda_moe_cache_table_graph_end(ggml_backend_cuda_context &) {}
void ggml_cuda_moe_cache_hf_graph_end(ggml_backend_cuda_context &) {}
void ggml_cuda_moe_cache_cs_graph_scan(ggml_backend_cuda_context &, const ggml_cgraph *) {}

bool ggml_cuda_moe_skipmass_supports(const ggml_tensor *) { return false; }
bool ggml_cuda_moe_skipmass_compute(ggml_backend_cuda_context &, ggml_tensor *) { return false; }

static void moe_cache_register_stub(const void * owner) {
    (void) owner;
}

#else

#include "common.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "moe-weighted-reduction.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "../ggml-backend-moe-cache.h"
#include "../ggml-moe-prefetch.h"
#include "../ggml-moe-generation-profile.h"
#include "../ggml-moe-skipmass.h"

#if !defined(GGML_USE_HIP)
#include <cuda/atomic>
#include "moe-table-state.cuh"
#endif
#include "moe-cpu-split.h"
#include "moe-cs-greedy.h"
#include "moe-hf-copy.cuh"
#include "unary.cuh"

#include <algorithm>
#include <array>
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
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
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

// Host-to-device copy from an expert source. On Windows (WDDM) cudaMemcpyAsync rejects a host range that mixes
// cudaHostRegister'ed pages with unregistered ones (cudaErrorInvalidValue) instead of staging it like Linux does.
// Fall back to a per-thread pinned bounce buffer in that case. Other errors and other copy kinds pass through.
static cudaError_t moe_h2d_async(void * dst, const void * src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (err != cudaErrorInvalidValue || kind != cudaMemcpyHostToDevice || !dst || !src || !bytes) {
        return err;
    }
    (void) cudaGetLastError();
    static thread_local void * bounce = nullptr;
    static constexpr size_t bounce_bytes = 8u << 20;
    if (!bounce && cudaHostAlloc(&bounce, bounce_bytes, cudaHostAllocPortable) != cudaSuccess) {
        bounce = nullptr;
        (void) cudaGetLastError();
        return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
    for (size_t off = 0; off < bytes; off += bounce_bytes) {
        const size_t n = bytes - off < bounce_bytes ? bytes - off : bounce_bytes;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) { return err; }
        memcpy(bounce, (const char *) src + off, n);
        err = cudaMemcpyAsync((char *) dst + off, bounce, n, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) { return err; }
    }
    return cudaStreamSynchronize(stream);
}

// fork (missjoin): device counters for opt-in miss staging; see moe_missjoin_stage_kernel.
static unsigned long long * g_missjoin_counts = nullptr;
static constexpr int moe_missjoin_count_entries = 10;
// Overlap mode: persistent per-projection scratch and table copies, filled on a side
// stream so each projection's matrix-vector pass waits only for its own copies.
static cudaStream_t g_mj_side = nullptr;
static cudaEvent_t g_mj_fork = nullptr;
static cudaEvent_t g_mj_done[3] = {};
static char * g_mj_scratch = nullptr;
static const void ** g_mj_eff = nullptr;
static size_t g_mj_stride = 0;
static int64_t g_mj_experts = 0;
static int g_mj_slots = 0;
// fork (missjoin hit-first, GGML_MOE_MISSJOIN_HITFIRST=1): one miss plan per layer on a high-priority side
// stream, then one copy launch per projection; the resident routes of each projection compute while the
// copies land. See moe_hf_begin.
static cudaStream_t g_hf_side = nullptr;
static cudaEvent_t g_hf_fork = nullptr;
static cudaEvent_t g_hf_done[3] = {};
static char * g_hf_scratch = nullptr;
static const void ** g_hf_eff = nullptr;
static const void ** g_hf_src = nullptr;
static int * g_hf_nstage = nullptr;
// fork MOEGAP MISSLIST: GGML_MOE_MISS_LIST=1|check. The plan kernel writes the miss passes' expert lists
// (moe_miss_list, mmvq.cuh: 0 up, 1 gate, 2 down, 3 fused gate+up); a miss pass is then one launch over the
// experts it computes instead of a grid over every route. check = also run the old pass and trap on any bit change.
static moe_miss_list * g_hf_miss = nullptr;
static int moe_miss_list_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_MOE_MISS_LIST");
        if (!e) { return 0; }
        // auto (the Flash-Next default): on, except for a layer whose experts have an AI+ bank (a timing race with
        // the bank copy makes the answer differ from run to run there; 3XS output is the same either way)
        if (strcmp(e, "auto") == 0) { return 3; }
        return strcmp(e, "check") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}
// fork AIPFIX (GGML_MOE_BANK_MIXED_HITFIRST, default 1): split banked mixed GEMVs around the copy wait.
static bool moe_mixed_hit_first() {
    static const bool v = [] {
        const char * e = getenv("GGML_MOE_BANK_MIXED_HITFIRST");
        return !e || atoi(e) != 0;
    }();
    return v;
}

// fork AIPFIX P2LIST: GGML_MOE_BANK_P2_LIST=1 runs a banked layer's copied (cold) routes through the 3XS miss-list
// kernel from the plan's list (one weight read per expert for every token that routes to it, gate and up fused)
// instead of the mixed kernel's phase 2. Needs the hit-first split and GGML_MOE_MISS_LIST. Skipped routes are zeroed
// as the 3XS path does (tail zero; the CPU join rewrites host routes); the routed sum is then its own launch.
// Default on (still needs GGML_MOE_MISS_LIST): paired Taichi coding 100K on top of P1HITG+P1CONC, -0.38 ms/round
// width-matched (p=.04). GGML_MOE_BANK_P2_LIST=0 opts out.
static bool moe_bank_p2_list() {
    static const bool v = [] {
        const char * e = getenv("GGML_MOE_BANK_P2_LIST");
        return e ? atoi(e) != 0 : true;
    }();
    return v;
}

// fork AIPFIX P1HITG: GGML_MOE_BANK_P1_HITG=1 runs a banked layer's cached (3XS) gate/up routes through the 3XS
// fused hit kernel (one weight read per expert for every token routed to it, SwiGLU in the store), before the mixed
// phase 1, which then takes the bank routes only. Needs the hit-first split and the fused SwiGLU (GGML_MOE_HIT_FUSED=1).
// Bit 0: gate/up, bit 1: down. Default 3 (both): paired Taichi coding 100K, keep-2000 bank, -0.84 ms/round (p=.0025).
// GGML_MOE_BANK_P1_HITG=0 opts out.
static int moe_bank_p1_hitg_bits() {
    static const int v = [] {
        const char * e = getenv("GGML_MOE_BANK_P1_HITG");
        return e ? atoi(e) & 3 : 3;
    }();
    return v;
}
static bool moe_bank_p1_hitg() {
    return (moe_bank_p1_hitg_bits() & 1) != 0;
}
static bool moe_bank_p1_hitg_down() {
    return (moe_bank_p1_hitg_bits() & 2) != 0;
}

// fork AIPFIX P2SAME: with P1HITG, a banked layer's copied (3XS) routes run through the same 3XS kernel as its cached
// routes (the hit kernel again, with the miss masks), as a 3XS layer's hit and miss passes do, instead of the mixed
// kernel's phase 2 or the miss list. Then a route's bits do not depend on whether its expert was resident or copied
// this step (the run-to-run split of the bank path: same input, different prompt-time cache fill, different tokens).
// Default off (it did not make reruns identical, and it takes the copied routes away from P2LIST's miss-list
// kernel); GGML_MOE_BANK_P2_SAME=1 opts in, and LLAMA_MOE_BANK_DET=1 (identical reruns) turns it on at model load.
static bool moe_bank_p2_same() {
    static const bool v = [] {
        const char * e = getenv("GGML_MOE_BANK_P2_SAME");
        return e && atoi(e) != 0;
    }();
    return v;
}
static std::atomic<unsigned long long> g_st_p2same{0};

// fork AIPFIX P1CONC: GGML_MOE_BANK_P1_CONC (staged default ON, =0 opts out; needs P1HITG and the bank zero skip). With P1HITG the
// 3XS hit kernel and the bank-only mixed phase 1 each fill a fraction of the card (ncu: the width-5 mixed launch runs
// at 22% occupancy). The bank phase now runs on a second stream at the same time as the hit kernel, which skips bank
// routes without writing them, so the two launches write disjoint rows. Same kernels, same inputs, same per-row
// arithmetic: the output is bit-identical to P1HITG alone.
static bool moe_bank_p1_conc() {
    static const bool v = [] {
        const char * e = getenv("GGML_MOE_BANK_P1_CONC");
        return e ? atoi(e) != 0 : true;
    }();
    // Only a loaded bank (zero skip active) uses the second stream; a plain 3XS run never reaches this and logs nothing.
    static bool logged = false;
    const bool on = v && ggml_cuda_moe_bank_zero_active();
    if (on && !logged) {
        logged = true;
        GGML_LOG_INFO("[p1-conc] bank phase 1 on a second stream beside the 3XS hit kernel\n");
    }
    return on;
}
static constexpr int MOE_P1_CONC_STREAM = GGML_CUDA_MAX_STREAMS - 1;
static std::atomic<unsigned long long> g_st_p1conc{0};
// fork AIPFIX P1PRIO: GGML_MOE_BANK_P1_PRIO (default 0). Traces of the ship stack show the 3XS hit kernels are the
// longer side of both concurrent pairs (gate/up hitg ends ~21 us after the bank mixed GEMV; the down hit kernel starts
// ~11 us after its inputs are ready because the bank down launch, issued first, takes the SMs, then ends ~14 us after
// it). 1 = issue the hit kernel first, on the side stream created one step above default priority (graph node
// priority needs GGML_CUDA_GRAPH_NODE_PRIORITY=1, part of the ship stack), with the bank launch on the main stream;
// 2 = the same swap at default priority (issue order only). Scheduling only: same kernels, inputs and rows.
static int moe_bank_p1_prio() {
    static const int v = [] {
        const char * e = getenv("GGML_MOE_BANK_P1_PRIO");
        const int m = e ? atoi(e) : 0;
        if (m) {
            GGML_LOG_INFO("[p1-prio] mode %d: hit kernels issued first on the side stream%s\n", m,
                    m == 1 ? " at raised priority" : "");
        }
        return m == 1 || m == 2 ? m : 0;
    }();
    return v;
}
// fork: run(): the bank launch, issued on the side stream; then main(): the hit kernel with bank rows unwritten.
template <typename B, typename H>
static void moe_p1_conc_run(ggml_backend_cuda_context & ctx, B && bank_launch, H && hit_launch) {
    static cudaEvent_t ev[GGML_CUDA_MAX_DEVICES][2] = {};
    const int dev = ctx.device;
    if (!ev[dev][0]) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[dev][0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev[dev][1], cudaEventDisableTiming));
    }
    const int prio = moe_bank_p1_prio();
    if (prio == 1 && ctx.streams[dev][MOE_P1_CONC_STREAM] == nullptr) {
        int least = 0, greatest = 0;
        ggml_cuda_set_device(dev);
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        // one step above default, below the hit-first copy stream (greatest), so the copies still go first
        const int p = greatest < least ? least - 1 : least;
        CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.streams[dev][MOE_P1_CONC_STREAM], cudaStreamNonBlocking, p));
    }
    const int saved = ctx.curr_stream_no;
    cudaStream_t main_s = ctx.stream();
    cudaStream_t side_s = ctx.stream(dev, MOE_P1_CONC_STREAM);
    CUDA_CHECK(cudaEventRecord(ev[dev][0], main_s));
    CUDA_CHECK(cudaStreamWaitEvent(side_s, ev[dev][0], 0));
    ctx.curr_stream_no = MOE_P1_CONC_STREAM;
    if (prio) {
        ggml_cuda_moe_set_bank_zero_nowrite(true);
        hit_launch();
        ggml_cuda_moe_set_bank_zero_nowrite(false);
        ctx.curr_stream_no = saved;
        CUDA_CHECK(cudaEventRecord(ev[dev][1], side_s));
        bank_launch();
    } else {
        bank_launch();
        ctx.curr_stream_no = saved;
        CUDA_CHECK(cudaEventRecord(ev[dev][1], side_s));
        ggml_cuda_moe_set_bank_zero_nowrite(true);
        hit_launch();
        ggml_cuda_moe_set_bank_zero_nowrite(false);
    }
    CUDA_CHECK(cudaStreamWaitEvent(main_s, ev[dev][1], 0));
    g_st_p1conc.fetch_add(1, std::memory_order_relaxed);
}

// fork MOEGAP DOWNONE: GGML_MOE_DOWN_ONE=1|check. On a hit-first layer the down projection waits for its copy
// first and then computes every route the card owns in ONE launch over the effective table (resident experts at
// their cache address, staged ones at their copy), instead of a hit pass and a miss pass. The down copy is normally
// done by then (it is issued with the gate/up copies at the layer's start), so this drops one launch per layer
// without losing overlap. Each route is still computed by the same kernel with the same per-route arithmetic, so the
// output keeps its bits. check = also run the old two passes into a copy and trap on any bit change.
static int moe_down_one_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_MOE_DOWN_ONE");
        if (!e) { return 0; }
        return strcmp(e, "check") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}
static std::atomic<unsigned long long> g_st_one{0};
static uint8_t * g_hf_zero = nullptr;   // [g_hf_experts] zeros: the single down pass's filter when no expert is on the CPU
// --moe-skip-tail (LOSSY route-rank cutoff), hit-first layers only. Per layer the side stream writes
// [0..2][g_hf_experts] = resident in that projection OR left out, and [3][g_hf_experts] = left out, before the plan;
// the plan and the miss passes then treat a left-out expert as if resident (no copy, no CPU work, no miss math) and
// a zero kernel writes its output rows. Counters: [0] tensor-expert rows left out, [1] layer windows, [2] experts.
static uint8_t * g_tail_mask = nullptr;
// [0] rows [1] layer windows [2] experts left out [3..13] tokens by left-out routes (0..10)
#define MOE_TAIL_COUNTS 14
static unsigned long long * g_tail_counts = nullptr;
// fork SWAPPOL: per (layer, expert) decode-window statistics, written by the skip kernel (GGML_MOE_SWAP_STATS=path,
// dumped at device release). Row of MOE_SWST_COLS: routes by rank 0..9, summed weight x10240, routes skipped, routes
// hit (all three projections resident), windows present.
#define MOE_SWST_LAYERS 48
#define MOE_SWST_COLS 14
static unsigned long long * g_swst = nullptr;
// fork SWAPPOL: swap score source (GGML_MOE_SWAP_SCORE=count|kept|weight|weightkept, GGML_MOE_SWAP_DROPW). Non-count modes accumulate per
// (layer, expert) in units of 1/1024 route: kept drops the routes the skip budget left out, weight counts a route as
// 10x its router weight. Default: weight with DROPW 0.5 when the skip budget is on (its kernel fills the scores; SWAPPOL
// r37x+r43x, 29 Taichi pairs: -0.96% ms/round, p=.026), count otherwise. GGML_MOE_SWAP_SCORE=count restores the old policy.
static unsigned long long * g_swscore = nullptr;
static unsigned long long * g_swscore_host = nullptr;
static bool moe_swap_score_default_weight() {
    static const bool on = [] {
        const char * v = getenv("GGML_MOE_SWAP_SCORE");
        const char * p = getenv("GGML_MOE_SKIP_POLICY");
        return (!v || !*v) && p && *p;
    }();
    return on;
}
static int moe_swap_score_mode() {
    static const int mode = [] {
        const char * v = getenv("GGML_MOE_SWAP_SCORE");
        if (moe_swap_score_default_weight()) { return 2; }
        if (!v || !*v || strcmp(v, "count") == 0) { return 0; }
        if (strcmp(v, "kept") == 0) { return 1; }
        if (strcmp(v, "weight") == 0) { return 2; }
        if (strcmp(v, "weightkept") == 0) { return 3; }
        GGML_ABORT("GGML_MOE_SWAP_SCORE must be count, kept, weight or weightkept");
    }();
    // GGML_MOE_SWAP_DROPW alone (count scores) still needs the score buffer: mode 4.
    static const bool skipw_set = getenv("GGML_MOE_SWAP_DROPW") && *getenv("GGML_MOE_SWAP_DROPW");
    return mode ? mode : (skipw_set ? 4 : 0);
}
// fork SWAPPOL: GGML_MOE_SWAP_DROPW=<0..1> scales the score of routes the skip budget left out (not resident, rank
// past the floor): 1 counts them fully, 0 ignores them (same as kept). Returned in 1/1024 units.
static unsigned moe_swap_skipw() {
    static const unsigned w = [] {
        if (moe_swap_score_mode() & 1) { return 0u; }
        const char * v = getenv("GGML_MOE_SWAP_DROPW");
        if (!v || !*v) { return moe_swap_score_default_weight() ? 512u : 1024u; }
        const float f = strtof(v, nullptr);
        if (!(f >= 0.0f && f <= 1.0f)) { GGML_ABORT("GGML_MOE_SWAP_DROPW must be in [0, 1]"); }
        return (unsigned) (f * 1024.0f + 0.5f);
    }();
    return w;
}
static const char * moe_swst_path() {
    static const char * path = [] { const char * v = getenv("GGML_MOE_SWAP_STATS"); return v && *v ? v : nullptr; }();
    return path;
}
static size_t g_hf_stride = 0;
static int64_t g_hf_experts = 0;
static int g_hf_slots = 0;
static std::atomic<unsigned long long> g_hf_layers{0};
// fork MOEGAP: how often each new switch took its path (counted where the host issues the work, so a CUDA graph
// counts once per capture, not per replay)
static std::atomic<unsigned long long> g_st_hitg{0}, g_st_ml{0}, g_st_pc{0}, g_st_q8hit{0}, g_st_q8claim{0}, g_st_pub_plan{0},
        g_st_p2list{0}, g_st_p1hitg{0};
// fork PARFIX: graphs that ended with a layer's side copies still unjoined (see ggml_cuda_moe_cache_hf_graph_end).
static std::atomic<unsigned long long> g_hf_graph_end_joins{0};
// fork MOEGAP (GGML_MOE_CPU_SPLIT=1): the host CPU computes part of each hit-first layer's missed experts
// (moe-cpu-split.h). The plan kernel picks the CPU experts and writes this card-side plan; the publish kernel copies
// the input rows into mapped memory and raises the job; the down projection's join kernel adds the CPU's rows.
struct moe_cs_plan_dev {
    int n_exp, n_route;
    int expert[MOE_CS_MAX_EXPERTS];
    int first[MOE_CS_MAX_EXPERTS + 1];
    int route[MOE_CS_MAX_ROUTES];          // route index (token * n_ids + slot) of each CPU row
    unsigned long long seq;                // job number of this layer's job (0 = none)
    unsigned long long counter;            // last job number issued
    unsigned ticket;
    unsigned long long stat_experts, stat_routes, stat_layers;
    unsigned long long stat_wait_ns, stat_joins;  // card time spent waiting for the host in the join
    float route_ns_live;                   // GGML_MOE_CPU_SPLIT_ADAPT=1: measured cost per route (0 = use the probe's)
    unsigned long long stat_adapt;         // jobs folded into route_ns_live
    unsigned long long stat_parts;         // missed projections (expert x up/gate/down) the CPU took
    unsigned long long stat_partial;       // CPU experts that were resident in one or two projections
};
static bool g_cs_on = false;
static moe_cs_job * g_cs_job_h = nullptr;
static moe_cs_job * g_cs_job_d = nullptr;
static float * g_cs_x_h = nullptr, * g_cs_x_d = nullptr;
static float * g_cs_out_h = nullptr, * g_cs_out_d = nullptr;
static size_t g_cs_x_stride = 0, g_cs_out_stride = 0;
static moe_cs_plan_dev * g_cs_plan = nullptr;
static uint8_t * g_cs_mask = nullptr;      // [3][g_hf_experts]: resident or on the CPU (the miss passes skip both)
// fork MOEGAP DOWNONE: g_cs_mask has a 4th row, [3][e] = 1 when the host computes expert e this layer (written only
// with GGML_MOE_DOWN_ONE on); the single down pass skips exactly those.
static float g_cs_route_ns = 0.0f, g_cs_fixed_ns = 0.0f, g_cs_ns_per_byte = 0.0f;
static int g_cs_max_exp = 8, g_cs_max_route = 24;
static float g_cs_share = -1.0f;
static int g_cs_adapt = 0;   // GGML_MOE_CPU_SPLIT_ADAPT=1: the join corrects the per-route cost from the host's measured job times
// fork MOEGAP CPU-first (GGML_MOE_CPU_SPLIT_FIRST=1 sets all three; each can be set alone):
//  PARTIAL=1: an expert missing in only one or two projections may go to the CPU too (the CPU computes the whole
//    expert from host RAM; the card's hit pass still computes the resident projections, whose down rows the join
//    then overwrites, so nothing is counted twice);
//  FLOOR_US=<us per distinct expert of the layer>: card time the hit math takes anyway. The CPU may fill it at no
//    cost, so the planner keeps handing experts to the CPU while max(CPU, max(floor, link)) does not grow;
//  LINK_WEIGHT=<x>: weight on the link side of the balance (>1 hands the CPU more; default 1.0).
static int g_cs_partial = 0;
static float g_cs_floor_ns = 0.0f, g_cs_link_w = 1.0f;
static std::atomic<unsigned long long> g_cs_layers_published{0};

// fork PREFETCHFIX: asynchronous look-ahead prefetch (GGML_MOE_PREFETCH_ASYNC=1 copies,
// =2 predicts and measures accuracy only). The graph writes each prediction into a ring in
// mapped host memory; a host thread copies the predicted cold experts with the copy engine
// into a small dedicated buffer and then writes a tag. A resolve kernel before each expert
// dispatch points the dispatch at a tagged copy instead of the slow host read. Nothing on
// the compute stream ever waits for the copies, and no cached expert is evicted.
#define MOE_PF_MAX_K 64
#define MOE_PF_RING 512
#define MOE_PF_MAX_SLOTS 32
#define MOE_PF_STATS 16
struct moe_pf_entry {
    unsigned seq;
    int layer;
    unsigned stamp;
    int n;
    int32_t e[MOE_PF_MAX_K];
    uint8_t mask[MOE_PF_MAX_K];
};
struct moe_pf_state {
    std::atomic<bool> active{false};
    bool alloc_failed = false;
    std::atomic<int> requested{0};   // --moe-prefetch value seen by configure
    std::atomic<const void *> owner{nullptr}; // one live session owns captured scratch
    int device = -1;
    int n_layers = 0;
    int n_expert = 0;
    int sets = 2;
    int kpf = 8;
    size_t kind_bytes[3] = {};
    size_t kind_off[3] = {};
    size_t slot_stride = 0;
    unsigned * d_stamp = nullptr;          // [n_layers] bumped by predict and by the last consumer
    unsigned * d_pred_stamp = nullptr;     // [n_layers]
    int32_t * d_pred = nullptr;            // [n_layers][MOE_PF_MAX_K]
    unsigned long long * d_tags = nullptr; // [sets][kpf]
    char * d_buf = nullptr;                // [sets][kpf] slots of [up|gate|down]
    const uint8_t ** d_hot = nullptr;      // [n_layers][3]
    unsigned * d_head = nullptr;
    unsigned long long * d_stats = nullptr;
    moe_pf_entry * h_ring = nullptr;
    moe_pf_entry * d_ring = nullptr;
    std::mutex mu;                         // guards src/bytes/src_ready for the worker
    std::vector<const void *> src;         // [(layer*3+k)*n_expert + e] pinned host source
    std::vector<size_t> bytes;             // [layer*3+k]
    bool src_ready = false;
    cudaStream_t stream = nullptr;
    // GGML_MOE_PF_YIELD=1: demand reads go first. The resolve kernel raises mark[0] when its
    // dispatch still has host reads, a tiny kernel after the staging sets mark[1]; the worker
    // keeps at most two experts in flight and waits (bounded) while mark[0] != mark[1].
    bool yield = false;
    unsigned * h_mark = nullptr;
    unsigned * d_mark = nullptr;
    unsigned * d_mark_seq = nullptr;       // [0] counter, [1] seq of the latest raised mark
    cudaEvent_t inflight[2] = {};
    std::atomic<unsigned long long> yield_waits{0}, yield_us{0}, yield_timeouts{0};
    std::thread worker;
    std::atomic<bool> stop{false};
    std::atomic<unsigned long long> entries{0}, experts{0}, copies{0}, copy_bytes{0}, truncated{0},
            no_source{0}, overruns{0}, errors{0};
};
// Never destroyed: the worker is stopped and joined by an exit handler registered after CUDA init.
static moe_pf_state & g_pf = *new moe_pf_state();
static int moe_pf_mode() {
    static const int mode = [] {
        const char * v = getenv("GGML_MOE_PREFETCH_ASYNC");
        const int m = v ? atoi(v) : 0;
        return m == 1 || m == 2 ? m : 0;
    }();
    return mode;
}
static void moe_pf_log(const char * when);
static void moe_pf_release(const void * session);
static void moe_pf_forget(const void * session);
static void moe_missjoin_log(const char * when) {
    if (!g_missjoin_counts) {
        return;
    }
    unsigned long long c[moe_missjoin_count_entries] = {};
    if (cudaMemcpy(c, g_missjoin_counts, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    MOE_CACHE_LOG("[moe-cache] missjoin %s: dispatches=%llu distinct_experts=%llu distinct_misses=%llu staged=%llu miss_routes=%llu unstaged=%llu\n",
            when, c[0], c[1], c[2], c[3], c[4], c[5]);
    if (c[6]) {
        MOE_CACHE_LOG("[moe-cache] missjoin copy layout %s: expert_cta_dispatches=%llu staged_bytes=%llu\n",
                when, c[6], c[7]);
    }
    if (c[8]) {
        MOE_CACHE_LOG("[moe-cache] missjoin reuse-only %s: bypassed_experts=%llu bypassed_bytes=%llu\n",
                when, c[8], c[9]);
    }
    if (const unsigned long long hf = g_hf_layers.load(std::memory_order_relaxed)) {
        MOE_CACHE_LOG("[moe-cache] missjoin hit-first %s: hf_layers=%llu graph_end_joins=%llu\n", when, hf,
                g_hf_graph_end_joins.load(std::memory_order_relaxed));
    }
    if (g_st_hitg.load() || g_st_ml.load() || g_st_pc.load() || g_st_q8hit.load() || g_st_q8claim.load() ||
            ggml_cuda_moe_down_sw_count() || g_st_one.load() || g_st_pub_plan.load() || g_st_p2list.load() ||
            g_st_p1hitg.load() || g_st_p2same.load()) {
        MOE_CACHE_LOG("[moe-cache] path counts %s: hit_fused_layers=%llu miss_list_passes=%llu plan_count_layers=%llu "
                "q8_share_hits=%llu q8_share_claims=%llu down_sw_passes=%llu down_one_layers=%llu pub_plan_layers=%llu bank_p2_list=%llu bank_p1_hitg=%llu bank_p1_conc=%llu bank_p2_same=%llu (issue-time counts)\n",
                when, g_st_hitg.load(), g_st_ml.load(), g_st_pc.load(), g_st_q8hit.load(), g_st_q8claim.load(),
                ggml_cuda_moe_down_sw_count(), g_st_one.load(), g_st_pub_plan.load(), g_st_p2list.load(),
                g_st_p1hitg.load(), g_st_p1conc.load(), g_st_p2same.load());
    }
    if (g_cs_on && g_cs_plan) {
        moe_cs_plan_dev plan{};
        unsigned long long host[9] = {};
        moe_cs_stats(host);
        if (cudaMemcpy(&plan, g_cs_plan, sizeof(plan), cudaMemcpyDeviceToHost) != cudaSuccess) {
            (void) cudaGetLastError();
        }
        MOE_CACHE_LOG("[moe-cache] cpu split %s: planned_layers=%llu published_layers=%llu cpu_experts=%llu cpu_routes=%llu "
                "host_jobs=%llu host_experts=%llu host_routes=%llu host_busy_ms=%.1f (phases ms %.1f/%.1f/%.1f/%.1f) host_null_src=%llu card_timeouts=%llu card_joins=%llu card_wait_us_per_join=%.1f adapt_jobs=%llu route_us_live=%.1f\n",
                when, plan.stat_layers, g_cs_layers_published.load(std::memory_order_relaxed), plan.stat_experts,
                plan.stat_routes, host[0], host[1], host[2], host[3] / 1e6, host[5] / 1e6, host[6] / 1e6, host[7] / 1e6,
                host[8] / 1e6, host[4],
                g_cs_job_h ? (unsigned long long) g_cs_job_h->gpu_timeouts : 0ull, plan.stat_joins,
                plan.stat_joins ? plan.stat_wait_ns / 1e3 / plan.stat_joins : 0.0, plan.stat_adapt,
                plan.route_ns_live / 1e3);
        MOE_CACHE_LOG("[moe-cache] cpu split share %s: cpu_parts=%llu of missed_parts=%llu (%.1f%%), partial_experts=%llu, "
                "parts over the link=%llu\n", when, plan.stat_parts, c[2], c[2] ? 100.0 * plan.stat_parts / c[2] : 0.0,
                plan.stat_partial, c[3]);
        GGML_LOG_INFO("[moe-cpu-split] live %s: CPU-served=%llu staged=%llu missed=%llu CPU share=%.3f (missed projections)\n",
                when, plan.stat_parts, c[3], c[2], c[2] ? (double) plan.stat_parts / c[2] : 0.0);
        moe_cs_wstats_report(when);
    }
}

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
    // GGML_MOE_CACHE_DET: scope serial of the plan that queued this fill (a same-scope fill is never a hit).
    uint64_t det_scope = 0;
    uint64_t det_serial = 0;   // GGML_MOE_CACHE_DET: enqueue order, used to publish deferred fills in a fixed order
};

// fork AIPFIX: GGML_MOE_CACHE_DET=1 makes legacy (pre-freeze) hit/miss choices independent of fill-worker
// timing: every scope starts with the fill queue drained, a fill queued in the current scope is never a
// hit, and the queue limits count this scope's jobs instead of the live queue.
static bool moe_cache_det() {
    static const bool det = [] { const char * e = getenv("GGML_MOE_CACHE_DET"); return e && atoi(e) != 0; }();
    return det;
}
// fork AIPFIX DETWARM: set while moe_cache_warm queues its ranked fills. The warm queues in a fixed rank order and
// waits for live queue space itself, so the per-scope det limits must not bind there: with them the warm stopped
// after queue_mb (1 GiB) of fills, leaving a half-loaded expert, and the whole-only device tables were refused.
static thread_local bool g_moe_cache_in_warm = false;

static uint64_t moe_cache_lending_uses(const moe_cache_slot & slot) {
    return (uint64_t)slot.warm_uses + slot.uses;
}

// fork: prompt loans retain physical prefix pages instead of refilling survivors from RAM.
static bool moe_cache_keep_prompt() {
#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP)
    const char * e = getenv("LLAMA_PP_KEEP_EXPERTS");
    return e && strcmp(e, "1") == 0;
#else
    return false;
#endif
}

// time spent mapping/unmapping prompt slab chunks, reported by the loan and return log lines
static std::atomic<int64_t> g_prompt_slab_us{0};

struct moe_cache_prompt_slab {
#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP)
    static constexpr size_t capacity = 1ull << 35;
    CUdeviceptr address = 0;
    size_t mapped = 0;
    size_t chunk = 0;
    int physical = 0;

    bool init(int logical) {
        if (!ggml_cuda_info().devices[logical].vmm) { return false; }
        physical = ggml_cuda_info().devices[logical].physical_device;
        const size_t gran = ggml_cuda_info().devices[logical].vmm_granularity;
        if (!gran || gran > capacity) { return false; }
        chunk = ((16u << 20) + gran - 1) / gran * gran;
        return cuMemAddressReserve(&address, capacity, chunk, 0, 0) == CUDA_SUCCESS;
    }
    bool resize(size_t bytes) {
        if (!address || bytes > capacity || bytes > SIZE_MAX - chunk + 1) { return false; }
        const size_t wanted = (bytes + chunk - 1) / chunk * chunk;
        const size_t before = mapped;
        const int64_t t0 = ggml_time_us();
        // GGML_MOE_CACHE_VMM_BATCH=1: one access call (and one unmap) over the whole changed range instead of one
        // per 16 MiB chunk; same chunks, same addresses
        const bool batch = vmm_batch();
        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = physical;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        while (mapped < wanted) {
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location = access.location;
            CUmemGenericAllocationHandle handle;
            if (cuMemCreate(&handle, chunk, &prop, 0) != CUDA_SUCCESS) { resize(before); return false; }
            const CUresult result = cuMemMap(address + mapped, chunk, 0, handle, 0);
            CU_CHECK(cuMemRelease(handle));
            if (result != CUDA_SUCCESS) { resize(before); return false; }
            if (!batch && cuMemSetAccess(address + mapped, chunk, &access, 1) != CUDA_SUCCESS) {
                CU_CHECK(cuMemUnmap(address + mapped, chunk));
                resize(before);
                return false;
            }
            mapped += chunk;
        }
        if (batch && mapped > before && cuMemSetAccess(address + before, mapped - before, &access, 1) != CUDA_SUCCESS) {
            resize(before);
            return false;
        }
        if (batch && mapped > wanted && cuMemUnmap(address + wanted, mapped - wanted) == CUDA_SUCCESS) {
            mapped = wanted;
        }
        while (mapped > wanted) {
            CU_CHECK(cuMemUnmap(address + mapped - chunk, chunk));
            mapped -= chunk;
        }
        g_prompt_slab_us.fetch_add(ggml_time_us() - t0);
        return true;
    }
    static bool vmm_batch() {
        static const bool on = [] { const char * e = getenv("GGML_MOE_CACHE_VMM_BATCH"); return e && atoi(e) != 0; }();
        return on;
    }
    char * data() const { return (char *) address; }
    size_t mapped_size() const { return mapped; }
    size_t mapped_size(size_t bytes) const { return (bytes + chunk - 1) / chunk * chunk; }
    ~moe_cache_prompt_slab() {
        if (address) {
            resize(0);
            CU_CHECK(cuMemAddressFree(address, capacity));
        }
    }
#else
    bool init(int) { return false; }
    bool resize(size_t) { return false; }
    char * data() const { return nullptr; }
    size_t mapped_size() const { return 0; }
    size_t mapped_size(size_t bytes) const { return bytes; }
#endif
};

struct moe_cache_pool {
    size_t expert_size = 0;
    int wtype = -1;
    char * slab = nullptr;
    std::unique_ptr<moe_cache_prompt_slab> prompt_slab;
    int prompt_slots = -1;
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
    size_t pool_plan_slots = SIZE_MAX;
};

// fork: loaded profile ownership is independent of temporary warm count buffers.
struct moe_cache_profile_global_tensor {
    std::string name;
    const void * data = nullptr;
    size_t expert_size = 0;
    size_t host_stride = 0;
    int wtype = -1;
    std::vector<uint64_t> counts;
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
    std::unordered_map<moe_cache_key, size_t, moe_cache_key_hash> profile_global_ranks;
    std::vector<size_t> profile_global_slots;
    std::unordered_set<moe_cache_key, moe_cache_key_hash> profile_global_selected;
    size_t profile_global_bytes = 0;
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
#if !defined(GGML_USE_HIP)
    std::unique_ptr<moe_tp::State> table_prefetch;
    char * table_prefetch_workspace = nullptr;
    size_t table_prefetch_capacity = 0;
    bool table_prefetch_prepared = false;
#endif

    std::deque<moe_cache_job> queue;
    size_t queued_bytes = 0;
    // GGML_MOE_CACHE_DET: jobs/bytes queued since the scope began (the queue is empty at scope start).
    int det_jobs = 0;
    size_t det_bytes = 0;
    uint64_t det_serial = 0;
    // GGML_MOE_CACHE_DET: finished fills (serial, job) held back until the next scope start or the freeze.
    std::vector<std::pair<uint64_t, moe_cache_job>> det_done;
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

struct moe_missjoin_retained_state;

struct moe_cache_device_table {
    ggml_moe_cache_tensor_desc desc{}; // compact expert bytes
    // Shared draft tensors can be destroyed before their target cache session.
    std::string name;
    size_t host_stride = 0;
    const void * borrowed = nullptr; // model-owned registered arena; never unregistered here
    bool permanent_sources = false;
    const void ** device = nullptr;
    const void ** staging_device = nullptr;
    uint8_t * hot = nullptr;
    std::vector<const void *> addresses;
    std::vector<const void *> fallback;
    // Host addresses are distinct from the CUDA aliases in fallback.
    std::vector<const void *> copy_source;
    // fork CACHEHIT: 1 where the expert lives in permanent residency (never swapped).
    std::vector<uint8_t> permanent;
    // 1 where the expert belongs to the always-resident bank: its entry points at a shared zero block
    // (the regular call gives it weight 0) and it is marked permanent, so no miss, swap or CPU split touches it.
    std::vector<uint8_t> bank;
    int n_bank = 0; // fork AIPNEXTK BANKSKIP: bank entries in this table
    // fork AIPFIX ONEPATH: device pointers to the bank rows of the bank experts (nullptr elsewhere), a device flag
    // (1 = bank expert) and the rows' type; set only with LLAMA_MOE_BANK_ONEPATH=1 and AI+ unit bank rows
    const void ** ai_bank = nullptr;
    uint8_t * ai_bank_flag = nullptr;
    int ai_bank_type = -1;
    std::vector<const void *> ai_bank_rows; // host copies of ai_bank / ai_bank_flag (LLAMA_MOE_BANK_ONEPATH_CHECK)
    std::vector<uint8_t> ai_bank_host;
    struct moe_cache_pool * swap_pool = nullptr;
    size_t count_offset = 0;
    // fork (missjoin): layer and projection (0 up, 1 gate, 2 down) parsed from the tensor name.
    int mj_layer = -1;
    int mj_kind = -1;
    // fork: demand copies survive rounds, independently of residency and prefetch.
    moe_missjoin_retained_state * mj_retained = nullptr;
    char * mj_retained_data = nullptr;
    // fork PREFETCHFIX: the same, filled only when asynchronous prefetch is on.
    int pf_layer = -1;
    int pf_kind = -1;
    int tp_layer = -1, tp_kind = -1;
};

// fork CACHEHIT: one queued table write, applied on the compute stream by a tiny kernel.
struct moe_swap_write {
    unsigned long long dst;
    unsigned long long value;
    unsigned int bytes;
    unsigned int pad;
};

struct moe_swap_fill {
    struct moe_cache_pool * pool;
    int index;
    moe_cache_device_table * table;
    int32_t expert;
    char * destination;
};

struct moe_swap_pair {
    float gain;
    moe_cache_pool * pool;
    int slot;
    bool free;
    moe_cache_device_table * table;
    int32_t expert;
};

struct moe_swap_decision {
    std::vector<moe_swap_pair> pairs;
    size_t candidates = 0;
    double ms = 0;
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
    // --moe-skip-tail cutoff for the current window (0 = exact; guarded by mu, read at graph build time) and the
    // missed tensor-expert rows the host path left out because of it
    int skip_tail = 0;
    bool skip_mass = false;
    // GGML_MOE_SKIP_POLICY/_THRESHOLD/_RENORM, read once per window in set_skip_tail (mode 0 rank, 1 threshold,
    // 2 cumulative, -1 invalid) so the per-expert plan never calls getenv
    int skip_mode = 0;
    int skip_rank_floor = 0;
    float skip_threshold = 0.0f;
    // fork AIPFIX SKIPBANK: GGML_MOE_SKIP_BANK_FLOOR / _BUDGET replace the rank floor / cumulative budget in layers
    // that carry AI+ bank experts (-1 = same as the other layers); read with the policy, so graphs bake them in
    int skip_bank_floor = -1;
    float skip_bank_threshold = -1.0f;
    bool skip_renorm = false;
    bool skip_logged = false;
    int mass_layer = -1;
    const void * mass_ids = nullptr;
    void * mass_trace = nullptr;
    unsigned * mass_trace_counts = nullptr;
    uint64_t tail_skipped_rows = 0;
    int tail_layer = -1;   // hit-first layer whose cutoff masks are live (-1 none); compute thread only
    std::atomic<uint64_t> device_captures{0};
    std::atomic<uint64_t> device_replays{0};
    unsigned long long * device_expert_counts = nullptr;
    std::vector<unsigned long long> device_previous_counts;
    uint64_t device_adapt_scopes = 0;
    std::string device_profile_path;
    uint64_t device_profile_signature = 0;
    std::vector<ggml_moe_cache_tensor_desc> device_profile_tensors;
    std::vector<ggml_moe_generation_profile_slice> device_profile_slices;
    uint64_t device_adapt_interval = 64;
    size_t device_adapt_copy_cap = 64u << 20;
    bool device_publish_async = false;
    // One stream for idle-boundary expert copies. Payloads finish before the
    // address tables are published and before the next graph is launched.
    cudaStream_t device_adapt_stream = nullptr;
    size_t device_fallback_staging_bytes = 0;
    unsigned long long * device_route_counts = nullptr;
    // Diagnostic only; guarded by mu and unused without DEVICE_STATS.
    uint64_t device_stats_scopes = 0;
    std::atomic<uint64_t> device_dispatches{0};
    // fork CACHEHIT swap policy (GGML_MOE_CACHE_SWAP=1). Guarded by mu except
    // swap_streams, which only the compute thread appends to.
    bool swap_enabled = false;
    bool swap_broken = false;
    bool swap_whole = false;
    bool swap_layer = false;
    bool swap_top_only = false;
    bool swap_free_probe = false;
    bool swap_cold_prune = false;
    bool swap_decay_after = false;
    std::vector<std::array<moe_cache_device_table *, 3>> swap_groups;
    int swap_interval = 1;
    int swap_max = 256;
    size_t swap_cap_bytes = 96u << 20;
    float swap_decay = 0.95f;
    float swap_hyst = 0.1f;
    float swap_margin = 2.0f;
    uint64_t swap_scopes = 0;
    cudaStream_t swap_streams[4] = {};
    std::atomic<int> swap_n_streams{0};
    std::atomic<bool> swap_stream_overflow{false};
    cudaStream_t swap_copy_stream = nullptr;
    cudaEvent_t swap_counts_event = nullptr;
    cudaEvent_t swap_pub_event = nullptr;
    cudaEvent_t swap_copy_event = nullptr;
    cudaEvent_t swap_join_event[4] = {};
    unsigned long long * swap_counts_host = nullptr;
    bool swap_counts_pending = false;
    std::unique_ptr<moe_cache_session> swap_snapshot;
    std::future<moe_swap_decision> swap_decision;
    double swap_async_ms_total = 0;
    moe_swap_write * swap_writes_host[2] = {};
    cudaEvent_t swap_writes_event[2] = {};
    int swap_writes_cap = 0;
    int swap_writes_flip = 0;
    std::vector<unsigned long long> swap_prev;
    std::vector<float> swap_score;
    std::vector<float> swap_observed_score;
    std::vector<float> swap_profile_rate;
    std::vector<float> swap_profile_seed;
    std::unordered_map<const void *, float> swap_profile_table_ema;
    std::unordered_set<const void *> swap_profile_calibrated_tables;
    float swap_profile_strength = 0;
    bool swap_profile_decay = false;
    std::vector<moe_swap_fill> swap_inflight;
    size_t swap_inflight_bytes = 0;
    uint64_t swap_updates = 0, swap_total = 0, swap_bytes_total = 0, swap_commits = 0;
    double swap_host_ms_total = 0.0;
    std::chrono::steady_clock::time_point swap_issue_time;
    double swap_land_ms_total = 0.0;
    // fork SWAPPOL: score-mode counts per count index (1/1024 route units) and their scale into routes.
    std::vector<unsigned long long> swap_policy_counts;
    float swap_count_scale = 1.0f;
    // Guarded by mu: startup restores must retain space for the not-yet-created helper.
    bool prompt_loaned = false;
    bool profile_load_attempted = false;
    bool profile_ranked = false;
    std::unordered_map<moe_cache_key, uint64_t, moe_cache_key_hash> profile_counts;
    std::vector<moe_cache_profile_global_tensor> profile_global_tensors;
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
    // fork (missjoin overlap): per-layer up/gate/down tables and the group staged last.
    std::vector<std::array<const moe_cache_device_table *, 3>> mj_groups;
    int mj_group = -1;
    const void * mj_ids = nullptr;
    unsigned mj_pending = 0;
    char * mj_retained_arena = nullptr;
    unsigned long long * mj_retained_counts = nullptr;
    cudaStream_t mj_retained_stream = nullptr;
    bool mj_retained_stream_set = false;
    // fork (missjoin hit-first): the order projections are dispatched in per layer, learned at run time so
    // the copies are issued in the order the math needs them.
    std::vector<std::array<int8_t, 3>> hf_order;
    std::vector<int8_t> hf_seq;
    // fork MOEGAP PLANCOUNT: the layer (and ids) whose route counts the plan kernel adds; bit k = kind k's count
    // still owed by the plan (its own count kernel is skipped).
    int pc_layer = -1;
    const void * pc_ids = nullptr;
    unsigned pc_mask = 0;
    // fork MOEGAP: layers the host can compute (registered) and whose graph has the plain up/gate -> SwiGLU ->
    // down shape (checked per graph by ggml_cuda_moe_cache_cs_graph_scan); the layer whose CPU job is in flight.
    std::vector<uint8_t> cs_registered;
    std::vector<uint8_t> cs_valid;
    // AIPFIX GGML_MOE_CPU_SPLIT_PROBE_PER_TYPE=1: each registered layer's own host cost (ns; empty = the global one)
    std::vector<float> cs_route_l, cs_fixed_l;
    int cs_layer = -1;
    // fork MOEGAP MISSLIST: the layer whose plan wrote the miss lists (-1: none)
    int ml_layer = -1;
    int active_scopes = 0;
    int active_nodes = 0;
    uint64_t det_scope = 0;   // GGML_MOE_CACHE_DET scope serial
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
static void moe_cache_swap_step(moe_cache_session & session);
static void moe_cache_swap_commit(moe_cache_session & session, bool ok, bool publish);
static void moe_cache_swap_note_stream(moe_cache_session & session, ggml_backend_cuda_context & ctx);
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

// Physical free-memory floor kept by the device-table and missjoin-retain
// checks after the expert pools are filled. The old fixed 1 GiB made every
// reserve at or below ~1 GiB fall back to legacy host routing (VRAMFIT,
// 2026-10-22). The default is now 512 MiB so a 1,024 MiB reserve keeps the
// device tables; the reserve itself still decides how much stays free.
// GGML_CUDA_MOE_CACHE_FREE_FLOOR_MB overrides it (1024 = old behavior).
static size_t moe_cache_physical_free_floor() {
    static const size_t floor = [] {
        int64_t value = 0;
        if (moe_cache_env_i64("GGML_CUDA_MOE_CACHE_FREE_FLOOR_MB", 0, 8192, value)) {
            MOE_CACHE_LOG("[moe-cache] physical free floor override: %lld MiB\n", (long long) value);
            return (size_t) value << 20;
        }
        return (size_t) 512 << 20;
    }();
    return floor;
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
        // fork AIPFIX D: AI+ unit experts are cached in slots of their own size and computed by the AI+ direct
        // GEMV over the slot table (moe_cache_device_compute); no hit-first overlap or CPU split for them yet
        case GGML_TYPE_AIT2_U:
        case GGML_TYPE_AIT2P_U:
        case GGML_TYPE_AIT3_U:
        case GGML_TYPE_AIT4_U:
        case GGML_TYPE_AIT2_U2:
        case GGML_TYPE_AIT3_U2:
            return getenv("GGML_MOE_CACHE_AI") == nullptr || atoi(getenv("GGML_MOE_CACHE_AI")) != 0;
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

// A finished fill becomes a valid slot at the LRU tail. Under GGML_MOE_CACHE_DET (before the freeze) this is
// held back and done at the next scope start in enqueue order, so LRU order and demand counts never depend
// on when the worker finished a copy.
static void moe_cache_fill_publish(moe_cache_device & device, moe_cache_pool & pool, const moe_cache_job & job, bool defer) {
    moe_cache_slot & slot = pool.slots[job.slot];
    if (defer) {
        device.det_done.emplace_back(slot.det_serial, job);
        return;
    }
    slot.state = moe_cache_slot_state::valid;
    moe_cache_lru_push_back(pool, job.slot);
    device.demand_count.erase(job.key);
}

static void moe_cache_det_publish_locked(moe_cache_device & device) {
    std::sort(device.det_done.begin(), device.det_done.end(),
        [](const std::pair<uint64_t, moe_cache_job> & a, const std::pair<uint64_t, moe_cache_job> & b) { return a.first < b.first; });
    for (const auto & item : device.det_done) {
        const moe_cache_job & job = item.second;
        if (job.pool < 0 || job.pool >= (int) device.pools.size()) { continue; }
        moe_cache_pool & pool = *device.pools[job.pool];
        if (job.slot < 0 || job.slot >= pool.n_slots) { continue; }
        const moe_cache_slot & slot = pool.slots[job.slot];
        if (slot.state == moe_cache_slot_state::copying && slot.generation == job.generation && slot.key == job.key) {
            moe_cache_fill_publish(device, pool, job, false);
        }
    }
    device.det_done.clear();
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
    const char * trace = getenv("GGML_MOE_CACHE_MEMORY_COMPONENTS");
    if (trace && strcmp(trace, "1") == 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d allocation bytes: operation=%s requested=%zu free=%zu total=%zu reserve=%zu other_outstanding=%zu headroom=%zu owned=%zu grant=%zu pool_bytes=%zu table_bytes=%zu input_bytes=%zu quant_bytes=%zu output_bytes=%zu\n",
                device.physical, operation, requested, view.free, view.total, view.reserve, view.other_outstanding,
                view.headroom, device.coordinator_allocated_bytes, device.budget_limit,
                device.allocated_bytes, device.table_arena ? device.table_arena_cap : 0,
                device.d_input ? device.d_input_cap : 0, device.d_act_q8 ? device.act_q8_cap : 0,
                device.d_out ? device.d_out_cap : 0);
    } else {
        MOE_CACHE_LOG("[moe-cache] CUDA%d allocation bytes: operation=%s requested=%zu free=%zu total=%zu reserve=%zu other_outstanding=%zu headroom=%zu owned=%zu grant=%zu\n",
            device.physical, operation, requested, view.free, view.total, view.reserve, view.other_outstanding,
            view.headroom, device.coordinator_allocated_bytes, device.budget_limit);
    }
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
    const char * trace = getenv("GGML_MOE_CACHE_MEMORY_COMPONENTS");
    if (trace && strcmp(trace, "1") == 0) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d budget refresh bytes: phase=%s old=%zu grant=%zu owned=%zu free=%zu total=%zu reserve=%zu other_outstanding=%zu participants=%d headroom=%zu pool_bytes=%zu table_bytes=%zu input_bytes=%zu quant_bytes=%zu output_bytes=%zu\n",
                device.physical, phase, old_limit, device.budget_limit, device.coordinator_allocated_bytes,
                view.free, view.total, view.reserve, other_outstanding, remaining_participants, headroom,
                device.allocated_bytes, device.table_arena ? device.table_arena_cap : 0,
                device.d_input ? device.d_input_cap : 0, device.d_act_q8 ? device.act_q8_cap : 0,
                device.d_out ? device.d_out_cap : 0);
    } else {
        MOE_CACHE_LOG("[moe-cache] CUDA%d budget refresh bytes: phase=%s old=%zu grant=%zu owned=%zu free=%zu total=%zu reserve=%zu other_outstanding=%zu participants=%d headroom=%zu\n",
            device.physical, phase, old_limit, device.budget_limit, device.coordinator_allocated_bytes,
            view.free, view.total, view.reserve, other_outstanding, remaining_participants, headroom);
    }
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

// fork: these original bytes belong to the model, never to a lendable cache session.
struct moe_resident {
    int device = -1;
    void * slab = nullptr;
    size_t bytes = 0;
    uint64_t signature = 0;
    std::vector<ggml_moe_cache_tensor_desc> tensors;
    std::vector<ggml_moe_resident_span> spans;
    std::vector<uint8_t> restore_stage;
    bool dropped = false;
    int (*on_restore)(void *) = nullptr;
    void * user_data = nullptr;
};

static std::mutex g_resident_mu;
static std::vector<moe_resident *> g_residents;

static const void * moe_resident_lookup_device(const void * host, size_t bytes, int device) {
    const uintptr_t begin = (uintptr_t) host;
    if (!bytes || bytes > UINTPTR_MAX - begin) { return nullptr; }
    std::lock_guard<std::mutex> lock(g_resident_mu);
    for (const auto * resident : g_residents) {
        if (resident->device != device) { continue; }
        for (const auto & span : resident->spans) {
            const uintptr_t base = (uintptr_t) span.host;
            if (begin >= base && begin - base <= span.size && bytes <= span.size - (begin - base)) {
                return (const char *) span.device + (begin - base);
            }
        }
    }
    return nullptr;
}

static int moe_resident_host_restore(const void * host, size_t bytes) {
    const uintptr_t begin = (uintptr_t) host;
    if (!bytes || bytes > UINTPTR_MAX - begin) { return bytes == 0; }
    std::lock_guard<std::mutex> lock(g_resident_mu);
    for (auto * resident : g_residents) {
        if (!resident->dropped) { continue; }
        bool overlaps = false;
        for (const auto & span : resident->spans) {
            const uintptr_t base = (uintptr_t) span.host;
            overlaps = overlaps || (begin < base + span.size && base < begin + bytes);
        }
        if (!overlaps) { continue; }
        ggml_cuda_set_device(resident->device);
        for (const auto & span : resident->spans) {
            for (size_t offset = 0; offset < span.size;) {
                const size_t part = std::min(span.size - offset, resident->restore_stage.size());
                if (!part || cudaMemcpy(resident->restore_stage.data(), (const char *)span.device + offset,
                            part, cudaMemcpyDeviceToHost) != cudaSuccess) { return 0; }
                // A hot span may cross registered cold pages and unregistered holes.
                // CUDA cannot use that mixed host range as one D2H destination.
                memcpy((char *)const_cast<void *>(span.host) + offset, resident->restore_stage.data(), part);
                offset += part;
            }
        }
        if (!resident->on_restore || !resident->on_restore(resident->user_data)) { return 0; }
        resident->dropped = false;
        MOE_CACHE_LOG("[moe-cache] permanent host restore: bytes=%zu reason=host-source-read copied_from=GPU disk_bytes=0\n", resident->bytes);
    }
    return 1;
}

static bool moe_resident_tensor(const void * host, int device) {
    std::lock_guard<std::mutex> lock(g_resident_mu);
    for (const auto * resident : g_residents) {
        if (resident->device != device) { continue; }
        for (const auto & tensor : resident->tensors) {
            if (tensor.data == host) { return true; }
        }
    }
    return false;
}

static const void * moe_resident_lookup(const void * host, size_t bytes, void * device) {
    if (!device || ggml_backend_dev_backend_reg((ggml_backend_dev_t)device) != g_moe_cache_owner) { return nullptr; }
    auto * reg = ggml_backend_dev_backend_reg((ggml_backend_dev_t)device);
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        if (ggml_backend_reg_dev_get(reg, i) == device) { return moe_resident_lookup_device(host, bytes, (int)i); }
    }
    return nullptr;
}

static int moe_resident_copy(void * opaque, ggml_tensor * dst, size_t offset, const void * host, size_t size) {
    auto * backend = (ggml_backend_t)opaque;
    if (!backend || !dst || !dst->buffer || offset > ggml_nbytes(dst) || size > ggml_nbytes(dst) - offset ||
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)) != g_moe_cache_owner ||
            ggml_backend_buffer_is_host(dst->buffer) ||
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(dst->buffer)) != ggml_backend_get_device(backend)) { return 0; }
    if (moe_cache_keep_prompt() && host && size) {
        auto * ctx = (ggml_backend_cuda_context *)backend->context;
        std::lock_guard<std::mutex> registry_lock(g_registry_mu);
        for (auto * session : g_sessions) {
            std::lock_guard<std::mutex> lock(session->mu);
            if (!session->prompt_loaned || session->stopping) { continue; }
            for (const auto & device : session->devices) {
                if (device->logical != ctx->device || device->dead.load()) { continue; }
                for (const auto & entry : device->seen_tensors) {
                    const uintptr_t base = (uintptr_t)entry.first;
                    const uintptr_t begin = (uintptr_t)host;
                    if (begin < base) { continue; }
                    const auto host_stride = device->host_strides.find(entry.first);
                    const size_t stride = host_stride == device->host_strides.end() ? entry.second.expert_size : host_stride->second;
                    if (!stride) { continue; }
                    const size_t expert = (begin - base) / stride;
                    const size_t source_offset = (begin - base) % stride;
                    if (expert >= (size_t)entry.second.n_expert || source_offset >= entry.second.expert_size || size > entry.second.expert_size - source_offset) { continue; }
                    for (const auto & pool : device->pools) {
                        if (pool->prompt_slots < 0 || pool->expert_size != entry.second.expert_size || pool->wtype != entry.second.wtype) { continue; }
                        const auto found = pool->map.find({entry.first, (int32_t)expert});
                        if (found == pool->map.end() || pool->slots[found->second].state != moe_cache_slot_state::valid) { continue; }
                        // The B1 coordinator joins this copy stream before returning or resizing the loan.
                        ggml_cuda_set_device(ctx->device);
                        if (cudaMemcpyAsync((char *)dst->data + offset, pool->slab + (size_t)found->second * pool->expert_size + source_offset,
                                    size, cudaMemcpyDeviceToDevice, ctx->stream()) != cudaSuccess) { GGML_ABORT("prompt resident copy failed"); }
                        return 1;
                    }
                }
            }
        }
    }
    // fork AIPNEXTK BANKDROP: a range that touches dropped bank-expert bytes is uploaded around them; the dropped
    // parts are written as zeros on the device (those experts only ever run here with routing weight 0)
    if (host && size && size <= UINTPTR_MAX - (uintptr_t) host && backend->iface.set_tensor_async) {
        size_t n_holes = 0;
        const ggml_moe_bank_hole * holes = ggml_moe_bank_holes(&n_holes);
        const uintptr_t begin = (uintptr_t) host, end = begin + size;
        const ggml_moe_bank_hole * it = holes ? std::lower_bound(holes, holes + n_holes, begin,
                [](const ggml_moe_bank_hole & h, uintptr_t v) { return h.end <= v; }) : nullptr;
        if (it && it != holes + n_holes && it->begin < end) {
            auto * ctx = (ggml_backend_cuda_context *) backend->context;
            ggml_cuda_set_device(ctx->device);
            const char * bulk_env = getenv("GGML_MOE_BANK_HOLES_BULK");
            const bool bulk = bulk_env && strcmp(bulk_env, "1") == 0;
            // fork: overwrite cold spans on the same stream after one clear instead of clearing every hole.
            if (bulk) {
                CUDA_CHECK(cudaMemsetAsync((char *) dst->data + offset, 0, size, ctx->stream()));
            }
            uintptr_t cursor = begin;
            for (; it != holes + n_holes && it->begin < end; ++it) {
                const uintptr_t first = std::max(begin, it->begin), last = std::min(end, it->end);
                if (cursor < first) {
                    backend->iface.set_tensor_async(backend, dst, (const void *) cursor, offset + (cursor - begin), first - cursor);
                }
                if (!bulk && cudaMemsetAsync((char *) dst->data + offset + (first - begin), 0, last - first, ctx->stream()) != cudaSuccess) {
                    GGML_ABORT("bank expert zero upload failed");
                }
                cursor = last;
            }
            if (cursor < end) {
                backend->iface.set_tensor_async(backend, dst, (const void *) cursor, offset + (cursor - begin), end - cursor);
            }
            return 1;
        }
    }
    const char * upload = getenv("GGML_MOE_RESIDENT_UPLOAD");
    if (upload && strcmp(upload, "1") == 0) {
        const uintptr_t begin = (uintptr_t)host;
        if (!host || !size || size > UINTPTR_MAX - begin || !backend->iface.set_tensor_async) { return 0; }
        const uintptr_t end = begin + size;
        struct copy_span { uintptr_t begin, end; const char * device; };
        std::vector<copy_span> spans;
        auto * ctx = (ggml_backend_cuda_context *)backend->context;
        std::lock_guard<std::mutex> lock(g_resident_mu);
        for (const auto * resident : g_residents) {
            for (const auto & span : resident->spans) {
                const uintptr_t base = (uintptr_t)span.host;
                if (span.size > UINTPTR_MAX - base) { return 0; }
                const uintptr_t first = std::max(begin, base);
                const uintptr_t last = std::min(end, base + span.size);
                if (first >= last) { continue; }
                if (resident->device != ctx->device) { return 0; }
                spans.push_back({first, last, (const char *)span.device + (first - base)});
            }
        }
        if (spans.empty()) { return 0; }
        std::sort(spans.begin(), spans.end(), [](const copy_span & a, const copy_span & b) { return a.begin < b.begin; });
        for (size_t i = 1; i < spans.size(); ++i) {
            if (spans[i].begin < spans[i - 1].end) { return 0; }
        }
        ggml_cuda_set_device(ctx->device);
        uintptr_t cursor = begin;
        for (const auto & span : spans) {
            if (cursor < span.begin) {
                backend->iface.set_tensor_async(backend, dst, (const void *)cursor,
                        offset + (cursor - begin), span.begin - cursor);
            }
            if (cudaMemcpyAsync((char *)dst->data + offset + (span.begin - begin), span.device,
                        span.end - span.begin, cudaMemcpyDeviceToDevice, ctx->stream()) != cudaSuccess) {
                GGML_ABORT("permanent expert device upload failed");
            }
            cursor = span.end;
        }
        if (cursor < end) {
            backend->iface.set_tensor_async(backend, dst, (const void *)cursor, offset + (cursor - begin), end - cursor);
        }
        return 1;
    }
    const void * source = moe_resident_lookup(host, size, ggml_backend_get_device(backend));
    if (!source) { return 0; }
    auto * ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_set_device(ctx->device);
    if (cudaMemcpyAsync((char *)dst->data + offset, source, size, cudaMemcpyDeviceToDevice, ctx->stream()) != cudaSuccess) {
        GGML_ABORT("permanent expert device copy failed");
    }
    return 1;
}

static size_t moe_resident_spans(void * opaque, ggml_moe_resident_span * spans, size_t capacity) {
    const auto * resident = (const moe_resident *) opaque;
    if (!resident) { return 0; }
    if (spans && capacity >= resident->spans.size()) {
        std::copy(resident->spans.begin(), resident->spans.end(), spans);
    }
    return resident->spans.size();
}

static void moe_resident_dropped(void * opaque, int (*on_restore)(void *), void * user_data) {
    auto * resident = (moe_resident *) opaque;
    std::lock_guard<std::mutex> lock(g_resident_mu);
    resident->on_restore = on_restore;
    resident->user_data = user_data;
    resident->dropped = true;
}

static void moe_resident_destroy(void * opaque) {
    auto * resident = (moe_resident *) opaque;
    if (!resident) { return; }
    std::lock_guard<std::mutex> lock(g_resident_mu);
    g_residents.erase(std::remove(g_residents.begin(), g_residents.end(), resident), g_residents.end());
    ggml_cuda_set_device(resident->device);
    if (resident->slab) { (void) cudaFree(resident->slab); }
    ggml_moe_cache_resident_notify(false);
    delete resident;
}

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

// [PREFILLFAST] opt-in switch helper.
static bool moe_cache_env_on(const char * name) {
    const char * value = getenv(name);
    return value && atoi(value) != 0;
}

// Whole experts only. Partial residency (some of an expert's gate/up/down slices on the GPU, the rest on the host)
// is removed: a partly resident expert still pays the host miss, and the measured whole-only plan held 163 more
// complete experts at flat speed (home, 2026-10-27). GGML_MOE_CACHE_WHOLE_ONLY=0 is ignored with a warning.
static bool moe_cache_whole_only() {
    static const bool once = [] {
        const char * value = getenv("GGML_MOE_CACHE_WHOLE_ONLY");
        if (value && atoi(value) == 0) {
            GGML_LOG_WARN("[moe-cache] GGML_MOE_CACHE_WHOLE_ONLY=0 ignored: partial expert residency was removed\n");
        }
        return true;
    }();
    return once;
}

// [PREFILLFAST] GGML_MOE_CACHE_FILL_BATCH=N (2..256): the worker takes up to N queued demand fills
// (no look-ahead prediction) at once, copies each host source straight to its slot on the fill stream
// and waits once for the whole batch. The single-job path copies through a pinned stage and waits per
// expert, which leaves the link about two thirds idle during the post-prompt cache warm. Same bytes,
// same slots, same publication order; only the copy scheduling changes. Off by default.
static int moe_cache_fill_batch_n() {
    // default 64 (owner ruling 2026-10-21); GGML_MOE_CACHE_FILL_BATCH=0 turns it off
    static const int n = [] { int64_t v = 64; moe_cache_env_i64("GGML_MOE_CACHE_FILL_BATCH", 0, 256, v); return (int) v; }();
    return n;
}

static void moe_cache_worker_fill_batch(moe_cache_session * session, moe_cache_device * device, cudaStream_t stream,
        std::vector<moe_cache_job> & jobs) {
    const int64_t copy_start = ggml_time_us();
    ggml_cuda_set_device(device->logical);
    std::vector<char *> destinations(jobs.size(), nullptr);
    std::vector<cudaError_t> errors(jobs.size(), cudaSuccess);
    {
        std::lock_guard<std::mutex> lock(session->mu);
        for (size_t i = 0; i < jobs.size(); ++i) {
            const auto & job = jobs[i];
            if (job.pool >= 0 && job.pool < (int)device->pools.size()) {
                moe_cache_pool * pool = device->pools[job.pool].get();
                if (pool->slab && job.slot >= 0 && job.slot < pool->n_slots) {
                    destinations[i] = pool->slab + (size_t)job.slot * pool->expert_size;
                }
            }
        }
    }
    cudaError_t first_error = (device->dead.load() || moe_cache_fail(*session, "insert")) ? cudaErrorUnknown : cudaSuccess;
    size_t copied_bytes = 0;
    {
        std::unique_lock<std::mutex> fill_lock(session->fill_mu, std::defer_lock);
        if (session->config.serial_fill) {
            fill_lock.lock();
        }
        for (size_t i = 0; i < jobs.size(); ++i) {
            const auto & job = jobs[i];
            if (first_error != cudaSuccess) {
                errors[i] = first_error;
                continue;
            }
            if (!destinations[i] || !moe_resident_host_restore(job.source, job.bytes)) {
                errors[i] = first_error = cudaErrorInvalidValue;
                continue;
            }
            errors[i] = moe_h2d_async(destinations[i], job.source, job.bytes, cudaMemcpyHostToDevice, stream);
            if (errors[i] != cudaSuccess) {
                first_error = errors[i];
            } else {
                copied_bytes += job.bytes;
            }
        }
        const cudaError_t sync_error = cudaStreamSynchronize(stream);
        if (sync_error != cudaSuccess) {
            for (auto & e : errors) {
                if (e == cudaSuccess) {
                    e = sync_error;
                }
            }
            if (first_error == cudaSuccess) {
                first_error = sync_error;
            }
            copied_bytes = 0;
        }
    }
    {
        std::lock_guard<std::mutex> lock(session->mu);
        const int64_t finished = ggml_time_us();
        if (copied_bytes > 0 && first_error == cudaSuccess && finished > copy_start) {
            const double rate = (double) copied_bytes / (double) (finished - copy_start);
            device->copy_bytes_per_us = device->copy_samples == 0 ? rate :
                0.875 * device->copy_bytes_per_us + 0.125 * rate;
            device->copy_samples++;
        }
        for (size_t i = 0; i < jobs.size(); ++i) {
            auto & job = jobs[i];
            moe_cache_pool * pool = job.pool >= 0 && job.pool < (int)device->pools.size() ? device->pools[job.pool].get() : nullptr;
            if (!job.prediction && pool && job.slot >= 0 && job.slot < pool->n_slots &&
                    pool->slots[job.slot].generation == job.generation) {
                job.prediction = pool->slots[job.slot].prediction;
                if (job.prediction && job.prediction->start == 0) {
                    job.prediction->start = copy_start; // A queued prediction promoted to demand.
                }
            }
            if (job.prediction) {
                job.prediction->done = errors[i] != cudaSuccess ? -1 : finished;
                moe_prefetch_log(*job.prediction);
            }
            if (pool && job.slot >= 0 && job.slot < pool->n_slots) {
                moe_cache_slot & slot = pool->slots[job.slot];
                if (slot.state == moe_cache_slot_state::copying &&
                    slot.generation == job.generation && slot.key == job.key) {
                    if (errors[i] == cudaSuccess && (!slot.prediction || !slot.prediction->closed)) {
                        moe_cache_fill_publish(*device, *pool, job, moe_cache_det() && !session->device_frozen);
                        device->fills++;
                    } else {
                        moe_cache_slot_reset(*pool, job.slot, true);
                        if (errors[i] != cudaSuccess) {
                            device->fill_failures++;
                        }
                    }
                }
            }
        }
        device->inflight = false;
        device->inflight_prefetch = false;
        device->inflight_source = nullptr;
        device->inflight_bytes = 0;
        session->idle_cv.notify_all();
    }
    if (first_error != cudaSuccess) {
        moe_cache_cuda_ok(*device, first_error, "expert fill", true);
        moe_cache_trim_session(*session, device->physical);
    }
}

static void moe_cache_worker(moe_cache_session * session, moe_cache_device * device, cudaStream_t stream) {
    char * stage = nullptr;
    size_t stage_capacity = 0;
    std::vector<moe_cache_job> batch;

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
            const int fill_batch = moe_cache_fill_batch_n();
            if (fill_batch > 1 && !job.prediction && !device->queue.empty() && !device->queue.front().prediction) {
                batch.clear();
                batch.push_back(job);
                uintptr_t lo = (uintptr_t) job.source, hi = lo + job.bytes;
                while ((int) batch.size() < fill_batch && !device->queue.empty() && !device->queue.front().prediction) {
                    const moe_cache_job & next = device->queue.front();
                    lo = std::min(lo, (uintptr_t) next.source);
                    hi = std::max(hi, (uintptr_t) next.source + next.bytes);
                    device->queued_bytes = next.bytes <= device->queued_bytes ? device->queued_bytes - next.bytes : 0;
                    batch.push_back(next);
                    device->queue.pop_front();
                }
                // the source-release wait checks this span: cover every source of the batch
                device->inflight_source = (const void *) lo;
                device->inflight_bytes = hi - lo;
            }
        }
        if (batch.size() > 1) {
            moe_cache_worker_fill_batch(session, device, stream, batch);
            batch.clear();
            continue;
        }
        batch.clear();

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
            if (!expired && error == cudaSuccess && !moe_resident_host_restore(job.source, job.bytes)) {
                error = cudaErrorInvalidValue;
            }
            // [PREFILLFAST] GGML_MOE_CACHE_FILL_DIRECT=1: copy the host source straight to the slot.
            // Arena sources are page-locked, so this skips one host memcpy per fill; a pageable
            // source still works (the driver stages it). Off by default.
            static const bool fill_direct = moe_cache_env_on("GGML_MOE_CACHE_FILL_DIRECT");
            if (!expired && error == cudaSuccess && fill_direct) {
                error = moe_h2d_async(destination, job.source, job.bytes, cudaMemcpyHostToDevice, stream);
                if (error == cudaSuccess) {
                    error = cudaStreamSynchronize(stream);
                }
            } else if (!expired && error == cudaSuccess && stage && stage_capacity >= job.bytes) {
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
                        moe_cache_fill_publish(*device, *pool, job, moe_cache_det() && !session->device_frozen);
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
static int moe_missjoin_mode();

static bool moe_cache_table_stats_enabled() {
    const char * value = getenv("GGML_MOE_CACHE_DEVICE_STATS");
    return value && strcmp(value, "1") == 0;
}

static bool moe_cache_table_mask_enabled() {
    const char * mask = getenv("GGML_MOE_MISSJOIN_MASK");
    // [AUTOMASK] GGML_MOE_MISSJOIN_AUTOMASK=1: MISSJOIN=stage turns the resident mask on by itself (the lossless staging path
    // needs it; without it the mode logs "inactive" and does nothing). Off by default. MISSJOIN_MASK=0 still wins.
    static const bool automask = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_AUTOMASK");
        return v && strcmp(v, "1") == 0;
    }();
    return moe_cache_table_stats_enabled() || (mask && strcmp(mask, "1") == 0 && moe_missjoin_mode()) ||
            (automask && moe_missjoin_mode() == 1 && !(mask && strcmp(mask, "0") == 0));
}

// fork AIPFIX ONEPATH (LLAMA_MOE_BANK_ONEPATH=1, the graph side is in llama-graph.cpp): a banked layer's decode-size
// batches run as a plain layer. The regular passes write zero for the bank experts' routes, as they do beside the bank
// call; one AI+ table GEMV over the bank rows then writes those routes (gate/up paired with the SwiGLU).
static bool moe_onepath() {
    static const bool on = [] { const char * v = getenv("LLAMA_MOE_BANK_ONEPATH"); return v && atoi(v) != 0; }();
    return on;
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
    size_t required = 0, entries = 0, max_count = 0;
    const bool stats = moe_cache_table_stats_enabled();
    const bool mask = moe_cache_table_mask_enabled();
    auto add_table = [&](size_t count) {
        if (count > SIZE_MAX / sizeof(void *) || count > SIZE_MAX - entries ||
                !moe_cache_table_bytes_add(required, count * sizeof(void *)) ||
                (session.prefetch_enabled && !moe_cache_table_bytes_add(required, count * sizeof(void *))) ||
                (mask && !moe_cache_table_bytes_add(required, count)) ||
                // fork AIPFIX ONEPATH: bank row pointers and flags (reserved for every table, used by banked ones)
                (moe_onepath() && (!moe_cache_table_bytes_add(required, count * sizeof(void *)) ||
                    !moe_cache_table_bytes_add(required, count)))) {
            return false;
        }
        entries += count;
        return true;
    };
    for (const auto & entry : device.seen_tensors) {
        const size_t count = (size_t)entry.second.n_expert;
        if (!add_table(count)) { return false; }
        max_count = std::max(max_count, count);
    }
    // Shared MTP adds gate/up/down tables after the target pools are built.
    const char * shared = getenv("GGML_MOE_CACHE_MTP_SHARED");
    if (shared && strcmp(shared, "1") == 0 && max_count) {
        for (int role = 0; role < 3; ++role) {
            if (!add_table(max_count)) { return false; }
        }
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

static int moe_table_prefetch_mode() {
    const char * value = getenv("GGML_MOE_TABLE_PREFETCH");
    if (!value || !*value || strcmp(value, "0") == 0 || strcmp(value, "off") == 0) return 0;
    if (strcmp(value, "pred") == 0 || strcmp(value, "1") == 0) return 1;
    if (strcmp(value, "stub") == 0 || strcmp(value, "2") == 0) return 2;
    if (strcmp(value, "on") == 0 || strcmp(value, "3") == 0) return 3;
    GGML_ABORT("invalid GGML_MOE_TABLE_PREFETCH (expected off, pred, stub or on)");
}

static size_t moe_table_prefetch_capacity(const moe_cache_device & device) {
#if !defined(GGML_USE_HIP)
    return device.table_prefetch_capacity;
#else
    GGML_UNUSED(device);
    return 0;
#endif
}

static bool moe_table_prefetch_reserve(moe_cache_session & session, moe_cache_device & device) {
    const int mode = moe_table_prefetch_mode();
    if (!mode) return true;
#if defined(GGML_USE_HIP) || !defined(__linux__) || CUDART_VERSION < 11070
    GGML_UNUSED(session); GGML_UNUSED(device);
    GGML_ABORT("table prefetch requires Linux CUDA >= 11.7");
#else
    if (session.devices.size() != 1 || !moe_cache_table_stats_enabled() ||
            session.prefetch_enabled || moe_pf_mode() != 0) {
        GGML_ABORT("table prefetch requires one device, DEVICE_STATS=1 and legacy prefetch off");
    }
    const char * device_on = getenv("GGML_MOE_CACHE_DEVICE");
    const char * fused = getenv("GGML_MOE_CACHE_DEVICE_FUSED");
    const char * overlap = getenv("GGML_MOE_MISSJOIN_OVERLAP");
    const char * missjoin = getenv("GGML_MOE_MISSJOIN");
    if (!device_on || strcmp(device_on, "1") != 0 ||
            (fused && strcmp(fused, "0") != 0) || (overlap && strcmp(overlap, "0") != 0) ||
            !missjoin || strcmp(missjoin, "stage") != 0) {
        GGML_ABORT("table prefetch requires DEVICE=1, DEVICE_FUSED=0, MISSJOIN=stage and MISSJOIN_OVERLAP=0");
    }
    if (!device.table_prefetch) {
        const char * path = getenv("GGML_MOE_TABLE_MODEL");
        if (!path || !*path) GGML_ABORT("table prefetch requires GGML_MOE_TABLE_MODEL");
        try {
            std::unique_ptr<moe_tp::State> state(new moe_tp::State());
            state->model = moe_tp::load_model(path);
            state->mode = mode; state->device = device.logical;
            int64_t cap = 20, budget = 28800000LL * state->model.lead;
            if (getenv("GGML_MOE_TABLE_CAP") &&
                    !moe_cache_env_i64("GGML_MOE_TABLE_CAP", 15, 20, cap)) {
                GGML_ABORT("invalid table prefetch cold cap");
            }
            if (cap != 15 && cap != 20) GGML_ABORT("table prefetch cold cap must be 15 or 20");
            if (getenv("GGML_MOE_TABLE_BUDGET_BYTES") &&
                    !moe_cache_env_i64("GGML_MOE_TABLE_BUDGET_BYTES", 1, 256LL << 20, budget)) {
                GGML_ABORT("invalid table prefetch byte budget");
            }
            state->cold_limit = unsigned(cap); state->byte_budget = uint64_t(budget);
            device.table_prefetch = std::move(state);
        } catch (const std::exception & error) {
            GGML_ABORT("table prefetch model load failed: %s", error.what());
        }
    }
    auto & state = *device.table_prefetch;
    GGML_ASSERT(state.mode == mode);
    const size_t bytes = state.workspace_bytes();
    if ((session.device_ready.load() && bytes > device.table_prefetch_capacity) ||
            !moe_cache_grow_device(device, (void **) &device.table_prefetch_workspace,
                device.table_prefetch_capacity, bytes, "reserved table prefetch workspace", true)) {
        GGML_ABORT("table prefetch workspace unavailable");
    }
    MOE_CACHE_LOG("[moe-table-prefetch] reserve: mode=%d lead=%u cold_cap=%u budget_bytes=%llu workspace_bytes=%zu policy=before-expert-pools\n",
        mode, state.model.lead, state.cold_limit, (unsigned long long) state.byte_budget, bytes);
    return true;
#endif
}

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
    slot_count = std::min({slot_count, shape.pool_plan_slots, (size_t)INT_MAX / stride_blocks});
    if (slot_count > INT_MAX) {
        slot_count = INT_MAX;
    }
    if (slot_count < moe_cache_pool_slots_min) {
        return false;
    }

    ggml_cuda_set_device(device.logical);
    char * slab = nullptr;
    std::unique_ptr<moe_cache_prompt_slab> prompt_slab;
    if (moe_cache_keep_prompt()) {
        prompt_slab.reset(new moe_cache_prompt_slab());
        if (!prompt_slab->init(device.logical)) { prompt_slab.reset(); }
    }
    auto release_slab = [&]() {
        if (prompt_slab) { prompt_slab->resize(0); } else { (void)cudaFree(slab); }
        slab = nullptr;
    };
    cudaError_t error = cudaSuccess;
    while (slot_count >= moe_cache_pool_slots_min) {
        if (moe_cache_fail(session, "slab")) {
            error = cudaErrorMemoryAllocation;
        } else {
            if (prompt_slab) {
                error = prompt_slab->resize(slot_count * shape.expert_size) ? cudaSuccess : cudaErrorMemoryAllocation;
                slab = error == cudaSuccess ? prompt_slab->data() : nullptr;
            } else {
                error = cudaMalloc((void **)&slab, slot_count * shape.expert_size);
            }
        }
        if (error == cudaSuccess) {
            size_t deficit = 0;
            const bool measured = moe_cache_slab_preserves_reserve(
                    device, slot_count * shape.expert_size, "new expert pool", deficit);
            if (measured && !deficit) {
                break;
            }
            release_slab();
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
        release_slab();
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
        device.pools.back()->prompt_slab = std::move(prompt_slab);
    } catch (...) {
        release_slab();
        moe_cache_budget_allocation(device, allocated, false);
        return false;
    }

    shape.pool = (int)device.pools.size() - 1;
    device.allocated_bytes += allocated;

    if (!moe_cache_start_worker(session, device)) {
        if (device.pools.back()->prompt_slab) { device.pools.back()->prompt_slab.reset(); }
        else { (void)cudaFree(device.pools.back()->slab); }
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

static bool moe_cache_resize_pool(moe_cache_device & device, moe_cache_pool & pool, int slots);

// Always-resident expert bank (ggml_moe_bank_*): those experts are computed by the bank call, never by the cache.
// fork AIPNEXTK BANKSKIP: the hit/miss passes write zero for bank experts' zero-weight routes instead of
// computing them from the zero block. GGML_MOE_BANK_ZERO_SKIP=0 computes them as before.
static bool moe_bank_skip_enabled() {
    static const bool on = [] { const char * v = getenv("GGML_MOE_BANK_ZERO_SKIP"); return v == nullptr || atoi(v) != 0; }();
    return on;
}

static bool moe_bank_has(const void * data, int32_t expert) {
    const uint8_t * in_bank = ggml_moe_bank_lookup(data);
    return in_bank && expert >= 0 && expert < 512 && in_bank[expert];
}

// One zero-filled device block per card, shared by every bank expert entry of the device tables. All-zero quant
// bytes decode to zero weights, so the regular call's weight-0 bank slots read finite zeros instead of host memory.
// A larger request allocates a new block; the old one stays allocated because earlier tables may still use it.
static const void * moe_bank_zero_block(int device, size_t bytes) {
    static std::mutex mutex;
    static std::unordered_map<int, std::pair<void *, size_t>> blocks;
    std::lock_guard<std::mutex> guard(mutex);
    auto & block = blocks[device];
    if (block.first && block.second >= bytes) { return block.first; }
    void * pointer = nullptr;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess || cudaMemset(pointer, 0, bytes) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
        if (pointer) { (void) cudaFree(pointer); }
        (void) cudaGetLastError();
        return nullptr;
    }
    block = {pointer, bytes};
    return pointer;
}

static bool moe_cache_build_global_profile(moe_cache_session & session, moe_cache_device & device,
        const std::vector<moe_cache_shape *> & pending, size_t budget, bool resize = false) {
    if (session.profile_global_tensors.empty() || (!resize && !device.pools.empty()) || pending.empty()) {
        return false;
    }
    if (resize) {
        for (const auto * shape : pending) {
            if (shape->pool < 0 || (size_t)shape->pool >= device.pools.size()) { return false; }
        }
    }
    std::vector<size_t> tensors;
    for (size_t i = 0; i < session.profile_global_tensors.size(); ++i) {
        if (device.seen_tensors.count(session.profile_global_tensors[i].data)) { tensors.push_back(i); }
    }
    if (tensors.size() != device.seen_tensors.size()) { return false; }
    auto layer = [&](size_t i) {
        int value = INT_MAX;
        moe_cache_layer_number(session.profile_global_tensors[i].name.c_str(), value);
        return value;
    };
    std::stable_sort(tensors.begin(), tensors.end(), [&](size_t a, size_t b) {
        return layer(a) < layer(b);
    });
    std::vector<size_t> caps, entries(pending.size(), 0), tensor_counts(pending.size(), 0);
    for (const auto * shape : pending) {
        const size_t block_bytes = ggml_type_size((ggml_type)shape->wtype);
        if (!block_bytes || shape->expert_size % block_bytes) { return false; }
        const size_t stride = shape->expert_size / block_bytes;
        if (!stride) { return false; }
        caps.push_back(std::min<size_t>((size_t)std::min<uint64_t>(shape->n_entries, INT_MAX), INT_MAX / stride));
    }
    // fork: constrain the existing whole-expert planner; never raise its live memory budget.
    const char * pool_plan = getenv("GGML_MOE_CACHE_POOL_PLAN");
    if (pool_plan && *pool_plan) {
        if (pending.size() > 4096) { GGML_ABORT("too many expert pools for pool plan"); }
        std::vector<ggml_cuda_moe_pool_limit> pools;
        for (size_t p = 0; p < pending.size(); ++p) {
            pools.push_back({(uint64_t)pending[p]->wtype, pending[p]->expert_size, caps[p]});
        }
        std::unique_ptr<FILE, int (*)(FILE *)> file(fopen(pool_plan, "rb"), fclose);
        std::vector<uint8_t> data(16 + 24 * pending.size());
        if (!file || fread(data.data(), 1, data.size(), file.get()) != data.size() || fgetc(file.get()) != EOF ||
                ferror(file.get()) || !ggml_cuda_moe_cache_pool_plan(data.data(), data.size(), pools, moe_cache_pool_slots_min, caps)) {
            GGML_ABORT("invalid expert pool plan: %s", pool_plan);
        }
        MOE_CACHE_LOG("[moe-cache] CUDA%d pool plan loaded: %s pools=%zu\n", device.physical, pool_plan, caps.size());
        for (size_t p = 0; p < pending.size(); ++p) { pending[p]->pool_plan_slots = caps[p]; }
    }
    std::vector<ggml_cuda_moe_profile_item> items;
    std::vector<moe_cache_key> keys;
    size_t group = 0, previous_tensor = SIZE_MAX;
    for (size_t i : tensors) {
        const auto & tensor = session.profile_global_tensors[i];
        size_t pool = 0;
        while (pool < pending.size() && (pending[pool]->expert_size != tensor.expert_size || pending[pool]->wtype != tensor.wtype)) { ++pool; }
        if (pool == pending.size()) { return false; }
        const auto seen = device.seen_tensors.find(tensor.data);
        if (seen == device.seen_tensors.end() || seen->second.n_expert != (int64_t)tensor.counts.size() ||
                tensor.counts.size() > SIZE_MAX - entries[pool]) { return false; }
        entries[pool] += tensor.counts.size();
        ++tensor_counts[pool];
        if (previous_tensor != SIZE_MAX) {
            if (layer(i) == INT_MAX || layer(i) != layer(previous_tensor)) { ++group; }
            else if (tensor.counts.size() != session.profile_global_tensors[previous_tensor].counts.size()) { return false; }
        }
        for (size_t expert = 0; expert < tensor.counts.size(); ++expert) {
            // a bank expert never needs a slot: count 0 so the plan gives its budget to the next regular expert
            const uint64_t count = moe_bank_has(tensor.data, (int32_t)expert) ? 0 : tensor.counts[expert];
            items.push_back({i, (int32_t)expert, group, pool, tensor.expert_size, count});
            keys.push_back({tensor.data, (int32_t)expert});
        }
        previous_tensor = i;
    }
    for (size_t p = 0; p < pending.size(); ++p) {
        if (entries[p] != pending[p]->n_entries || tensor_counts[p] != (size_t)pending[p]->n_tensors) { return false; }
    }
    const bool whole_only = moe_cache_whole_only();
    ggml_cuda_moe_profile_plan plan;
    int64_t density_value = 0;
    const bool density = !session.profile_ranked && moe_cache_env_i64("GGML_MOE_CACHE_PROFILE_GLOBAL_DENSITY", 0, 1, density_value) && density_value;
    int64_t ranked_limit = 0;
    if (session.profile_ranked) {
        moe_cache_env_i64("GGML_MOE_CACHE_PROFILE_RANKED_LIMIT", 1, INT_MAX, ranked_limit);
    }
    if (!ggml_cuda_moe_cache_profile_plan(items, caps, budget, moe_cache_pool_slots_min, plan, density, (size_t)ranked_limit)) { return false; }
    if (ranked_limit) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d ranked prefix: requested=%lld fitted=%zu bytes=%zu; no lower-rank backfill\n",
                device.physical, (long long)ranked_limit, plan.bundles, plan.bytes);
    }
    auto publish_plan = [&]() {
        device.profile_global_ranks.clear();
        device.profile_global_selected.clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            device.profile_global_ranks.emplace(keys[i], plan.ranks[i]);
            if (whole_only && plan.selected[i]) { device.profile_global_selected.insert(keys[i]); }
        }
        device.profile_global_slots = plan.slots;
        device.profile_global_bytes = plan.bytes;
    };
    // fork: an unchanged capacity needs new ranks, not a destructive slab rebuild.
    // [ALLOCSIM] opt-in: GGML_MOE_CACHE_PLAN_REFRESH=1 (default off keeps the slab rebuild).
    static const bool plan_refresh = moe_cache_env_on("GGML_MOE_CACHE_PLAN_REFRESH");
    if (plan_refresh && whole_only && resize) {
        bool same_capacity = true;
        for (size_t p = 0; p < pending.size(); ++p) {
            const int pool = pending[p]->pool;
            same_capacity = same_capacity && pool >= 0 && (size_t)pool < device.pools.size() &&
                (size_t)device.pools[pool]->n_slots == plan.slots[p];
        }
        if (same_capacity) {
            publish_plan();
            MOE_CACHE_LOG("[moe-cache] CUDA%d whole-only plan reused: bundles=%zu slices=%zu actual_pool_bytes=%zu\n",
                    device.physical, plan.bundles, device.profile_global_selected.size(), device.allocated_bytes);
            return true;
        }
    }
    if (whole_only) {
        device.profile_global_ranks.clear();
        device.profile_global_selected.clear();
        device.profile_global_slots.clear();
        device.profile_global_bytes = 0;
    } else {
        publish_plan();
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d global profile plan: budget=%zu selected_bytes=%zu slack=%zu bundles=%zu slices=%zu minimum_slots=%d; %s, selected bundles first\n",
            device.physical, budget, plan.bytes, budget - plan.bytes, plan.bundles,
            (size_t)std::count(plan.selected.begin(), plan.selected.end(), (uint8_t)1), moe_cache_pool_slots_min,
            density ? "logical route count per real byte" : "raw bytes-weighted use");
    if (resize) {
        // Whole plans must release every old slab before allocator-rounded replacement.
        // Keeping recipient slabs alive can leave a donor below its minimum capacity.
        for (size_t p = 0; p < pending.size(); ++p) {
            const int pool = pending[p]->pool;
            if (pool < 0 || (size_t)pool >= device.pools.size()) { return false; }
            if ((whole_only && !moe_cache_keep_prompt()) || (size_t)device.pools[pool]->n_slots > plan.slots[p]) {
                if (!moe_cache_resize_pool(device, *device.pools[pool], whole_only && !moe_cache_keep_prompt() ? 0 : (int)plan.slots[p])) {
                    MOE_CACHE_LOG("[moe-cache] CUDA%d global profile replan stopped: donor pool resize failed; no recipient growth\n", device.physical);
                    return true;
                }
            }
        }
    }
    if (whole_only && !(resize && moe_cache_keep_prompt())) {
        // A logical grant can outlive allocations made by the prompt/drafter.
        // Query after releasing old slabs; never count their payload twice.
        budget = std::min(budget, moe_cache_allocation_headroom(device, "whole profile rebuild", budget));
        if (!ggml_cuda_moe_cache_profile_plan(items, caps, budget, moe_cache_pool_slots_min,
                plan, density, (size_t)ranked_limit)) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d whole-only plan refused: live headroom cannot hold the minimum complete plan\n", device.physical);
            return true;
        }
        // Establish every pool's minimum before optional growth. Actual CUDA
        // slab rounding must not let earlier pools starve the final type.
        for (size_t p = 0; p < pending.size(); ++p) {
            if (resize) {
                if (!moe_cache_resize_pool(device, *device.pools[pending[p]->pool], moe_cache_pool_slots_min)) { return true; }
            } else {
                moe_cache_allocate_pool(session, device, *pending[p], moe_cache_pool_slots_min * pending[p]->expert_size);
            }
            const int pool = pending[p]->pool;
            if (pool < 0 || device.pools[pool]->n_slots < moe_cache_pool_slots_min) {
                MOE_CACHE_LOG("[moe-cache] CUDA%d whole-only plan refused: live allocation cannot hold every pool minimum\n", device.physical);
                return true;
            }
        }
        const size_t room = moe_cache_allocation_headroom(device, "whole profile growth", budget);
        budget = std::min(budget, device.allocated_bytes + room);
        if (!ggml_cuda_moe_cache_profile_plan(items, caps, budget, moe_cache_pool_slots_min,
                plan, density, (size_t)ranked_limit)) { return true; }
        MOE_CACHE_LOG("[moe-cache] CUDA%d whole-only physical plan: budget=%zu minimum_pool_bytes=%zu bundles=%zu\n",
                device.physical, budget, device.allocated_bytes, plan.bundles);
    }
    for (size_t p = 0; p < pending.size(); ++p) {
        const size_t requested = plan.slots[p] * pending[p]->expert_size;
        if (resize || whole_only) {
            auto & pool = *device.pools[pending[p]->pool];
            if ((size_t)pool.n_slots < plan.slots[p]) {
                const size_t current = (size_t)pool.n_slots * pool.expert_size;
                const size_t other = device.allocated_bytes >= current ? device.allocated_bytes - current : 0;
                const size_t room = other < budget ? budget - other : 0;
                const size_t target = std::min(plan.slots[p], room / pool.expert_size);
                static const bool grow_stepdown = moe_cache_env_on("GGML_MOE_CACHE_GROW_STEPDOWN");
                if (target >= moe_cache_pool_slots_min && !grow_stepdown) {
                    moe_cache_resize_pool(device, pool, (int)target);
                } else if (target >= moe_cache_pool_slots_min) {
                    // [ALLOCSIM] opt-in GGML_MOE_CACHE_GROW_STEPDOWN=1. A rejected growth trims by payload bytes, but cudaMalloc pads in whole
                    // chunks, so one retry can land far below the established pool. Step the
                    // target down (never under the slots held before growth) instead.
                    const int held = pool.n_slots;
                    int want = (int)target;
                    for (;;) {
                        moe_cache_resize_pool(device, pool, want);
                        if (pool.n_slots >= want || want <= held) { break; }
                        want = std::max(held, want - std::max(1, want / 16));
                    }
                }
            }
        } else {
            moe_cache_allocate_pool(session, device, *pending[p], requested);
        }
        const int pool = pending[p]->pool;
        const int actual = pool >= 0 ? device.pools[pool]->n_slots : 0;
        MOE_CACHE_LOG("[moe-cache] CUDA%d global profile capacity: type=%s expert_bytes=%zu planned_slots=%zu actual_slots=%d\n",
                device.physical, ggml_type_name((ggml_type)pending[p]->wtype), pending[p]->expert_size, plan.slots[p], actual);
    }
    // fork: allocator headroom can shorten a pool; keep only bundles that fit all actual pools.
    if (whole_only) {
        for (size_t p = 0; p < pending.size(); ++p) {
            const int pool = pending[p]->pool;
            caps[p] = pool >= 0 ? std::min(caps[p], (size_t)device.pools[pool]->n_slots) : 0;
        }
        if (!ggml_cuda_moe_cache_profile_plan(items, caps, budget, moe_cache_pool_slots_min,
                plan, density, (size_t)ranked_limit)) {
            MOE_CACHE_LOG("[moe-cache] CUDA%d whole-only plan refused: actual pools cannot hold the minimum complete plan\n", device.physical);
            return true;
        }
        publish_plan();
        MOE_CACHE_LOG("[moe-cache] CUDA%d whole-only plan reconciled: bundles=%zu slices=%zu actual_pool_bytes=%zu\n",
                device.physical, plan.bundles, device.profile_global_selected.size(), device.allocated_bytes);
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
    if (!moe_cache_reserve_table_arena(session, device) || !moe_table_prefetch_reserve(session, device)) {
        return;
    }
    if (!moe_cache_refresh_budget(session, device, "build pools")) {
        return;
    }
    MOE_CACHE_LOG("[moe-cache] CUDA%d runtime workspace bytes: input=%zu q8=%zu output=%zu; policy=materialize-before-expert-pools\n",
            device.physical, device.d_input_cap, device.act_q8_cap, device.d_out_cap);
    const size_t scratch_reserve =
        moe_cache_scratch_total(device.scratch_reserve) + device.table_arena_cap + moe_table_prefetch_capacity(device);
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

    // [PREFILLFAST] GGML_MOE_CACHE_PLAN_GRANULE=1: cudaMalloc rounds each slab up to 2 MiB, so a plan
    // that spends the whole budget leaves the last pool short; the prompt warm then replans at the
    // allocated size and resizes several pools (~0.7 s at 100K). Planning 2 MiB per pool under the
    // budget makes every slab fit as planned, and the replan reproduces the same plan (no resize).
    // Changes which experts start cached by a few MiB, never the math. Off by default.
    size_t plan_budget = remaining;
    static const bool plan_granule = moe_cache_env_on("GGML_MOE_CACHE_PLAN_GRANULE");
    if (plan_granule && !session.profile_global_tensors.empty()) {
        const size_t pad = pending.size() * ((size_t)2 << 20);
        plan_budget = remaining > pad ? remaining - pad : 0;
    }
    if (moe_cache_build_global_profile(session, device, pending, plan_budget)) {
        return;
    }
    const char * pool_plan = getenv("GGML_MOE_CACHE_POOL_PLAN");
    if (pool_plan && *pool_plan) { GGML_ABORT("expert pool plan requires a complete global profile plan"); }
    if (!session.profile_global_tensors.empty()) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d global profile plan unavailable: tensors=%zu seen=%zu pending_shapes=%zu budget=%zu; fallback=fixed-pool-shares\n",
                device.physical, session.profile_global_tensors.size(), device.seen_tensors.size(), pending.size(), remaining);
    }

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

// Resident census: complete experts (every projection valid in a cache slot) plus bank experts, counted once each,
// over the registered main-model layers.
static void moe_cache_log_census(const moe_cache_device & device) {
    auto resident = [](void * user, const void * data, int32_t expert) -> bool {
        for (const auto & pool : ((const moe_cache_device *) user)->pools) {
            const auto found = pool->map.find({data, expert});
            if (found != pool->map.end() && pool->slots[found->second].state == moe_cache_slot_state::valid &&
                    !pool->slots[found->second].staging) {
                return true;
            }
        }
        return false;
    };
    if (!ggml_moe_bank_any() && !getenv("GGML_MOE_CACHE_CENSUS")) {
        return; // default runs keep the integ-r2 log; GGML_MOE_CACHE_CENSUS=1 prints it without a bank
    }
    int32_t first = 0, last = 0;
    size_t cache = 0, bank = 0, both = 0;
    if (ggml_moe_bank_census(resident, (void *) &device, &first, &last, &cache, &bank, &both)) {
        MOE_CACHE_LOG("[moe-cache] CUDA%d resident complete experts layers %d-%d: cache %zu + bank %zu = %zu (cache copies of bank experts: %zu)\n",
                device.physical, first, last, cache, bank, cache + bank, both);
    }
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
    MOE_CACHE_LOG("[moe-cache] configured: mode=%s devices=%zu budget=%s reserve=%zu MiB min-slab=%zu MiB min-expert=%zu KiB max-batch=%d admit=%s hot=%d cpu-overlap=%s fills=%s queue=%d/%zu MiB det=%d\n",
            session.config.automatic ? "auto" : "on",
            session.devices.size(), budget.c_str(),
            session.config.reserve_mb,
            session.config.minimum_slab_bytes >> 20,
            session.config.min_expert_bytes >> 10,
            session.config.max_batch,
            admission.c_str(),
            session.config.hot_uses,
            overlap_cpu_rows.c_str(),
            session.config.serial_fill ? "serial" : "parallel",
            session.config.queue_max, session.config.queue_mb, moe_cache_det() ? 1 : 0);
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

        const char * device_profile = getenv("GGML_MOE_CACHE_DEVICE_PROFILE_SAVE");
        if (device_profile && *device_profile) {
            const char * profile_load = getenv("GGML_MOE_CACHE_PROFILE_LOAD");
            const char * profile_save = getenv("GGML_MOE_CACHE_PROFILE_SAVE");
            if (!ggml_moe_generation_profile_same_file(device_profile, profile_load) &&
                    !ggml_moe_generation_profile_same_file(device_profile, profile_save)) {
                session->device_profile_path = device_profile;
            } else {
                MOE_CACHE_LOG("[moe-cache] device profile export refused: output aliases a startup profile\n");
            }
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
            if (pool_ptr->prompt_slab) { pool_ptr->prompt_slab.reset(); }
            else { (void)cudaFree(pool_ptr->slab); }
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
#if !defined(GGML_USE_HIP)
    GGML_ASSERT(!device.table_prefetch || !device.table_prefetch->copy_stream);
    if (device.table_prefetch_workspace) {
        (void) cudaFree(device.table_prefetch_workspace);
        moe_cache_budget_allocation(device, device.table_prefetch_capacity, false);
        device.table_prefetch_workspace = nullptr;
        device.table_prefetch_capacity = 0;
    }
    device.table_prefetch_prepared = false;
    device.table_prefetch.reset();
#endif
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
    if (session->tail_skipped_rows > 0) {
        MOE_CACHE_LOG("[moe-cache] skip-tail (host path): %llu missed tensor-expert rows left out\n",
                (unsigned long long) session->tail_skipped_rows);
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

    // The last swap batch is usually still copying at shutdown; its slots count as missing until they are marked
    // valid. Wait for the copies and mark them (host state only, no table publish) so the final census counts every
    // expert the run actually held. Device release does the same commit right after; nothing runs in between.
    if (!session->swap_inflight.empty() && !session->devices.empty()) {
        ggml_cuda_set_device(session->devices[0]->logical);
        const bool landed = cudaDeviceSynchronize() == cudaSuccess;
        (void) cudaGetLastError();
        moe_cache_swap_commit(*session, landed, false);
    }
    for (auto & device_ptr : session->devices) {
        if (device_ptr->nodes > 0 || device_ptr->dispatch_failures > 0 ||
            device_ptr->collect_failures > 0) {
            moe_cache_log_stats(*device_ptr);
        }
        moe_cache_log_census(*device_ptr);
    }

    moe_cache_device_release(*session);
    for (auto & device_ptr : session->devices) {
        std::unique_lock<std::mutex> dispatch_lock(device_ptr->dispatch_mu);
        moe_cache_free_device(*device_ptr);
        moe_cache_budget_unregister(*device_ptr);
    }

    moe_pf_forget(session);
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
    std::unique_lock<std::mutex> lock(session->mu);
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
    if (moe_cache_det() && !session->device_frozen && session->active_scopes == 1) {
        // Drain prompt-phase fills so this scope's hits do not depend on how far the worker got.
        const bool drained = session->idle_cv.wait_for(lock, std::chrono::seconds(10), [&] {
            for (const auto & d : session->devices) {
                if (!d->dead.load() && (!d->queue.empty() || d->inflight)) { return false; }
            }
            return true;
        });
        static int warned = 0;
        if (!drained && warned++ < 4) {
            MOE_CACHE_LOG("[moe-cache] det: fill queue did not drain within 10 s at scope start\n");
        }
        session->det_scope++;
        for (auto & d : session->devices) { moe_cache_det_publish_locked(*d); d->det_jobs = 0; d->det_bytes = 0; }
        static bool logged = false;
        if (!logged) { logged = true; MOE_CACHE_LOG("[moe-cache] det: on (drain fills at scope start, same-scope fills miss, per-scope queue limits)\n"); }
    }
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
                // fork (cachehit): staged slots are read only by compute streams. With swaps on,
                // wait for those streams, not for the swap copy stream (a whole-device sync here
                // would stall every scope on the next round's fills).
                const int n_streams = active->swap_enabled ? active->swap_n_streams.load(std::memory_order_acquire) : 0;
                if (n_streams > 0 && !active->swap_stream_overflow.load()) {
                    for (int i = 0; i < n_streams; ++i) {
                        CUDA_CHECK(cudaStreamSynchronize(active->swap_streams[i]));
                    }
                } else {
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
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
        active->mj_group = -1;
        active->mj_ids = nullptr;
        active->mj_pending = 0;
        active->pc_mask = 0;
        moe_cache_calibration_on_leave(*active);
        // This is outside CUDA graph capture/replay. The opt-in aggregate
        // diagnostic is separate from the default bounded adaptation below;
        // neither performs a host routing round trip at every layer.
        if (active->device_route_counts && active->device_ready.load()) {
            const uint64_t scopes = ++active->device_stats_scopes;
            // fork (cachehit): GGML_MOE_CACHE_DEVICE_STATS_EVERY=N (default 64) sets the snapshot
            // interval; 0 keeps the counters and resident mask (MISSJOIN) but never syncs here.
            static const int64_t stats_every = [] { int64_t v = 64; moe_cache_env_i64("GGML_MOE_CACHE_DEVICE_STATS_EVERY", 0, 1 << 30, v); return v; }();
            if (stats_every > 0 && (scopes == 1 || scopes % (uint64_t) stats_every == 0)) {
                ggml_cuda_set_device(active->devices[0]->logical);
                unsigned long long counts[3] = {};
                const int64_t t_sync = ggml_time_us();
                if (cudaDeviceSynchronize() == cudaSuccess &&
                        cudaMemcpy(counts, active->device_route_counts, sizeof(counts), cudaMemcpyDeviceToHost) == cudaSuccess) {
                    MOE_CACHE_LOG("[moe-cache] device routes: hits=%llu misses=%llu dispatches=%llu; graph snapshot scopes=%llu (diagnostic sync) sync_us=%lld\n",
                            counts[0], counts[1], counts[2],
                            (unsigned long long)scopes, (long long)(ggml_time_us() - t_sync));
                    moe_missjoin_log("graph snapshot");
            moe_pf_log("graph snapshot");
                } else {
                    MOE_CACHE_LOG("[moe-cache] device route snapshot unavailable\n");
                    (void) cudaGetLastError();
                }
            }
        }
        if (active->active_scopes > 0) {
            active->active_scopes--;
        }
        if (active->active_scopes == 0 && active->active_nodes == 0 && active->device_ready.load() && active->swap_enabled) {
            moe_cache_swap_step(*active);
        } else if (active->active_scopes == 0 && active->active_nodes == 0 &&
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
                    candidate.scratch_reserve, &scratch_requirements) + moe_table_prefetch_capacity(candidate);
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
                    candidate.scratch_reserve, &scratch_requirements) + moe_table_prefetch_capacity(candidate);
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
                selected->scratch_reserve, &scratch_requirements) + moe_table_prefetch_capacity(*selected);
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
    const size_t stride = moe_cache_host_stride(device, host_base, expert_size);
    if (moe_resident_lookup_device((const char *)host_base + (size_t)expert * stride, expert_size, device.logical)) { return -1; }
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
        if (moe_cache_det() && pool.slots[found->second].det_scope == session.det_scope) {
            return -1;
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
    const bool det = moe_cache_det() && !g_moe_cache_in_warm;
    const int queued_jobs = det ? device.det_jobs : (int)device.queue.size();
    const size_t queued_bytes = det ? device.det_bytes : device.queued_bytes;
    if (inserts_left <= 0 || queued_jobs >= session.config.queue_max ||
        expert_size > queue_limit - std::min(queue_limit, queued_bytes)) {
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
    slot.det_scope = session.det_scope;
    slot.det_serial = ++device.det_serial;
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
        device.det_jobs++;
        device.det_bytes += expert_size;
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

        // --moe-skip-tail: a missed expert that no token of the window ranks in the head is left out, with no fill.
        if (session.skip_tail > 0 && session.skip_mode == 0 &&
                ggml_moe_cache_tail_rank_ok(ids, n_ids, (int) node->n_tokens, index, session.skip_tail)) {
            const auto found = pool.map.find(moe_cache_key{node->host_base, expert});
            const bool cached = found != pool.map.end() && pool.slots[found->second].state == moe_cache_slot_state::valid;
            const size_t stride = moe_cache_host_stride(device, node->host_base, node->expert_size);
            const bool resident = moe_resident_lookup_device(
                    (const char *) node->host_base + (size_t) expert * stride, node->expert_size, device.logical);
            if (!cached && !resident) {
                slot_indices[index] = GGML_MOE_CACHE_SKIP;
                session.tail_skipped_rows++;
                device.misses++;
                continue;
            }
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

// Expert keep policy probe (#215/#217): read-only residency lookup, no fills, demand counts, LRU or pin changes.
static int moe_cache_resident(
        const ggml_moe_cache_tensor_desc * tensor, const int32_t * ids, int n_ids, uint8_t * resident) {
    if (!resident || n_ids <= 0) {
        return 0;
    }
    memset(resident, 0, (size_t) n_ids);
    if (!tensor || !tensor->data || !ids || tensor->expert_size == 0 ||
        g_session_suppressed > 0 || g_session_stack.empty()) {
        return 0;
    }
    moe_cache_session * session = g_session_stack.back().active;
    if (!session || session->stopping || session->dormant) {
        return 0;
    }
    int count = 0;
    std::lock_guard<std::mutex> lock(session->mu);
    if (session->stopping) {
        return 0;
    }
    for (const auto & device_ptr : session->devices) {
        const moe_cache_device & device = *device_ptr;
        if (device.dead.load()) {
            continue;
        }
        const int pool_index = moe_cache_find_pool(device, tensor->expert_size, tensor->type);
        if (pool_index < 0 || !device.pools[pool_index]->slab) {
            continue;
        }
        const moe_cache_pool & pool = *device.pools[pool_index];
        for (int index = 0; index < n_ids; index++) {
            const int32_t expert = ids[index];
            if (resident[index] || expert < 0 || expert >= tensor->n_expert) {
                continue;
            }
            const auto found = pool.map.find({tensor->data, expert});
            if (found != pool.map.end() &&
                pool.slots[found->second].state == moe_cache_slot_state::valid) {
                resident[index] = 1;
                count++;
            }
        }
    }
    return count;
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
        up->type == GGML_TYPE_NVFP4 || ggml_ai4_is_unit((ggml_type) up->type) ||
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
    if (session->skip_tail > 0 || session->skip_mass) {
        // The unfused plan applies rank or weight-budget masks.
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
            moe_cache_device_release(session, false);
            session.device_attempted = false;
            break;
        }
    }
    session.tensor_devices.erase(base);
    for (const auto & tensor : session.profile_global_tensors) {
        if (moe_cache_ranges_overlap(tensor.data, tensor.host_stride * tensor.counts.size(), base, size)) {
            session.profile_global_tensors.clear();
            for (const auto & device : session.devices) {
                device->profile_global_ranks.clear();
                device->profile_global_slots.clear();
                device->profile_global_selected.clear();
                device->profile_global_bytes = 0;
            }
            break;
        }
    }
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
    if (base && size && ggml_moe_cache_has_residents()) {
        std::lock_guard<std::mutex> lock(g_resident_mu);
        for (const auto * resident : g_residents) {
            for (const auto & span : resident->spans) {
                if (moe_cache_ranges_overlap(base, size, span.host, span.size)) {
                    GGML_ABORT("permanent expert storage is immutable; unload it before changing weights");
                }
            }
        }
    }
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
    {
        std::lock_guard<std::mutex> lock(g_resident_mu);
        for (const auto * resident : g_residents) {
            if (resident->tensors.size() != (size_t)n_tensors) { continue; }
            bool same = true;
            for (int i = 0; i < n_tensors; ++i) {
                const auto & a = tensors[i];
                const auto & b = resident->tensors[i];
                same = same && a.data == b.data && a.expert_size == b.expert_size &&
                    a.type == b.type && a.n_in == b.n_in && a.n_out == b.n_out && a.n_expert == b.n_expert &&
                    strcmp(a.name, b.name) == 0;
            }
            if (same) { return resident->signature; }
        }
    }
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
        if (!moe_resident_host_restore(t.data, t.expert_size * (size_t)t.n_expert)) {
            GGML_ABORT("cannot restore permanent experts for profile identity");
        }
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
        std::vector<std::vector<uint64_t>> & storage, std::vector<const uint64_t *> & counts, bool * ranked = nullptr) {
    if (ranked) { *ranked = false; }
    std::unique_ptr<FILE, int (*)(FILE *)> file(fopen(path, "rb"), fclose);
    uint64_t header[5] = {};
    const uint64_t magic = UINT64_C(0x31464f5250454f4d); // MOEPROF1, native uint64_t format
    if (!file) { return false; }
    const size_t header_bytes = fread(header, 1, sizeof(header), file.get());
    if (header_bytes < 4) { return false; }
    if (memcmp(header, "STRP", 4) == 0) {
        size_t layers = 0, experts = 0;
        std::vector<int> tensor_layers;
        for (int i = 0; i < n_tensors; ++i) {
            int layer = -1;
            if (!moe_cache_layer_number(tensors[i].name, layer) || layer < 0 || layer >= 4096 ||
                    tensors[i].n_expert <= 0 || tensors[i].n_expert > 4096 ||
                    (experts && experts != (size_t)tensors[i].n_expert)) { return false; }
            layers = std::max(layers, (size_t)layer + 1);
            experts = (size_t)tensors[i].n_expert;
            tensor_layers.push_back(layer);
        }
        if (!layers || !experts || fseek(file.get(), 0, SEEK_END) != 0) { return false; }
        const long length = ftell(file.get());
        if (length < 24 || (uint64_t)length > 24 + layers * experts * 8 || fseek(file.get(), 0, SEEK_SET) != 0) { return false; }
        std::vector<uint8_t> data((size_t)length);
        std::vector<std::vector<uint64_t>> scores;
        if (fread(data.data(), 1, data.size(), file.get()) != data.size() || fgetc(file.get()) != EOF || ferror(file.get()) ||
                !ggml_cuda_moe_cache_ranked_profile(data.data(), data.size(), layers, experts, scores)) { return false; }
        storage.resize(n_tensors);
        counts.clear();
        counts.reserve(n_tensors);
        for (int i = 0; i < n_tensors; ++i) { storage[i] = scores[tensor_layers[i]]; }
        for (const auto & tensor : storage) { counts.push_back(tensor.data()); }
        if (ranked) { *ranked = true; }
        return true;
    }
    if (header_bytes != sizeof(header) || header[0] != magic || header[1] != (uint64_t) n_tensors ||
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

static void * moe_resident_create(void * device, const ggml_moe_cache_tensor_desc * tensors, int n_tensors, size_t budget) {
#if !defined(__linux__)
    GGML_UNUSED_VARS(device, tensors, n_tensors, budget);
    return nullptr;
#else
    const char * profile = getenv("GGML_MOE_CACHE_PROFILE_LOAD");
    if (!device || !tensors || n_tensors <= 0 || n_tensors > 4096 || !profile || !*profile ||
            budget < (1u << 20) || budget > (4ull << 30) ||
            ggml_backend_dev_backend_reg((ggml_backend_dev_t)device) != g_moe_cache_owner) { return nullptr; }
    ggml_moe_cache_config config{};
    ggml_moe_cache_device_caps caps{};
    config.min_compute_capability = moe_cache_cc_forced_min;
    if (!moe_cache_query_device(device, &config, &caps)) { return nullptr; }
    std::unique_ptr<moe_resident> resident(new moe_resident);
    resident->device = caps.logical_device;
    size_t entries = 0;
    for (int i = 0; i < n_tensors; ++i) {
        const auto & t = tensors[i];
        if (!t.name || !t.data || !ggml_moe_cache_wtype_supported(t.type) ||
                t.n_in <= 0 || t.n_in > INT_MAX || t.n_out <= 0 || t.n_out > INT_MAX || t.n_expert <= 0 || t.n_expert > 4096 ||
                t.n_in % ggml_blck_size((ggml_type)t.type)) { return nullptr; }
        const size_t row = ggml_row_size((ggml_type)t.type, t.n_in);
        if (!row || (uint64_t)t.n_out > SIZE_MAX / row) { return nullptr; }
        const size_t bytes = row * (size_t)t.n_out;
        if (t.expert_size < bytes || (uint64_t)t.n_expert > SIZE_MAX / t.expert_size ||
                (size_t)t.n_expert * t.expert_size > UINTPTR_MAX - (uintptr_t)t.data ||
                (size_t)t.n_expert > SIZE_MAX - entries) { return nullptr; }
        entries += (size_t)t.n_expert;
    }
    resident->signature = moe_cache_profile_signature(tensors, n_tensors);
    std::vector<std::vector<uint64_t>> storage;
    std::vector<const uint64_t *> counts;
    if (!moe_cache_profile_load(profile, resident->signature, entries, tensors, n_tensors, storage, counts)) {
        MOE_CACHE_LOG("[moe-cache] permanent residency refused: invalid profile\n");
        return nullptr;
    }
    struct candidate { const void * host; size_t bytes; long double priority; };
    std::vector<candidate> ranked;
    ranked.reserve(entries);
    int last_layer = -1;
    for (int i = 0; i < n_tensors; ++i) {
        int layer = -1;
        if (moe_cache_layer_number(tensors[i].name, layer)) { last_layer = std::max(last_layer, layer); }
    }
    for (int i = 0; i < n_tensors; ++i) {
        const auto & t = tensors[i];
        const size_t bytes = ggml_row_size((ggml_type)t.type, t.n_in) * (size_t)t.n_out;
        long double total = 0;
        for (int64_t e = 0; e < t.n_expert; ++e) { total += counts[i][e]; }
        int layer = 0;
        moe_cache_layer_number(t.name, layer);
        if (layer == last_layer) { continue; }
        for (int64_t e = 0; e < t.n_expert; ++e) {
            if (!counts[i][e] || moe_bank_has(t.data, (int32_t)e)) { continue; }
            ranked.push_back({(const char *)t.data + (size_t)e * t.expert_size, bytes,
                    counts[i][e] / std::max(1.0L, total) * (1 + 4.0L / (1.0L + layer)) / bytes});
        }
    }
    std::stable_sort(ranked.begin(), ranked.end(), [](const candidate & a, const candidate & b) { return a.priority > b.priority; });
    std::vector<candidate> selected;
    for (const auto & c : ranked) {
        if (c.bytes <= budget - resident->bytes) {
            selected.push_back(c);
            resident->bytes += c.bytes;
        }
    }
    // A profile with too little coverage cannot silently turn the requested RAM cut into a smaller one.
    if (resident->bytes < budget - std::min<size_t>(16u << 20, budget / 16)) { return nullptr; }
    ggml_cuda_set_device(resident->device);
    size_t free = 0, total = 0;
    const size_t card_cap = 23552ull << 20;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess || total < free ||
            free < resident->bytes + (1ull << 30) ||
            total - free > card_cap || resident->bytes > card_cap - (total - free)) {
        MOE_CACHE_LOG("[moe-cache] permanent residency refused: VRAM capacity\n");
        return nullptr;
    }
    resident->tensors.assign(tensors, tensors + n_tensors);
    resident->spans.reserve(selected.size());
    size_t restore_bytes = 0;
    for (const auto & c : selected) { restore_bytes = std::max(restore_bytes, c.bytes); }
    resident->restore_stage.resize(std::min<size_t>(restore_bytes, 8u << 20));
    if (cudaMalloc(&resident->slab, resident->bytes) != cudaSuccess) { return nullptr; }
    size_t offset = 0;
    for (const auto & c : selected) {
        void * destination = (char *)resident->slab + offset;
        if (cudaMemcpy(destination, c.host, c.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            (void) cudaFree(resident->slab);
            return nullptr;
        }
        resident->spans.push_back({c.host, destination, c.bytes});
        offset += c.bytes;
    }
    {
        std::lock_guard<std::mutex> lock(g_resident_mu);
        for (const auto * other : g_residents) {
            for (const auto & t : other->tensors) {
                for (int i = 0; i < n_tensors; ++i) {
                    if (t.data == tensors[i].data) {
                        (void) cudaFree(resident->slab);
                        return nullptr;
                    }
                }
            }
        }
        try {
            g_residents.push_back(resident.get());
        } catch (...) {
            (void) cudaFree(resident->slab);
            throw;
        }
        ggml_moe_cache_resident_notify(true);
    }
    MOE_CACHE_LOG("[moe-cache] permanent residency: original_bytes=%zu experts=%zu restore_staging_bytes=%zu profile_identity=verified lifetime=model\n",
            resident->bytes, resident->spans.size(), resident->restore_stage.size());
    return resident.release();
#endif
}

// fork: reranking after a prompt requires retiring every prepared address table.
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

static bool moe_cache_resize_prompt_pool(moe_cache_device & device, moe_cache_pool & pool, int slots) {
    ggml_cuda_set_device(device.logical);
    const int old_slots = pool.n_slots;
    if (slots < 0 || !pool.expert_size ||
            (size_t)std::max(slots, old_slots) > SIZE_MAX / pool.expert_size) { return false; }
    const size_t before = (size_t)old_slots * pool.expert_size;
    if (slots > old_slots) {
        const size_t headroom = moe_cache_allocation_headroom(device, "grow retained prompt pool", (size_t)(slots - old_slots) * pool.expert_size);
        slots = old_slots + (int)std::min((size_t)(slots - old_slots), headroom / pool.expert_size);
    }
    if (slots == old_slots) { return true; }
    const bool prefix_only = slots > old_slots || std::none_of(pool.slots.begin() + slots, pool.slots.end(),
            [](const moe_cache_slot & slot) { return slot.state == moe_cache_slot_state::valid; });
    if (prefix_only) {
        // fork: growth and an empty-tail shrink preserve every live address and lookup key.
        pool.slots.reserve(slots);
        pool.free_slots.reserve(slots);
        size_t bytes = (size_t)slots * pool.expert_size;
        if (slots < old_slots) {
            if (!pool.prompt_slab->resize(bytes)) { return false; }
            for (int i = slots; i < old_slots; ++i) { moe_cache_map_erase(pool, i); }
        }
        while (slots > old_slots) {
            bytes = (size_t)slots * pool.expert_size;
            if (!pool.prompt_slab->resize(bytes)) {
                // fork: a failed tail mapping leaves the retained prefix intact; retry a smaller physical growth.
                const size_t chunk = pool.prompt_slab->mapped_size(1);
                const size_t mapped = pool.prompt_slab->mapped_size();
                const size_t wanted = pool.prompt_slab->mapped_size(bytes);
                if (wanted <= mapped + chunk) { return false; }
                const size_t target_bytes = mapped + (wanted - mapped) / chunk / 2 * chunk;
                const size_t target = std::min((size_t)(slots - 1), target_bytes / pool.expert_size);
                if (target <= (size_t)old_slots) { return false; }
                slots = (int)target;
                continue;
            }
            size_t deficit = 0;
            const bool measured = moe_cache_slab_preserves_reserve(device, bytes - before, "grow retained prompt pool", deficit);
            if (measured && !deficit) { break; }
            const size_t mapped = pool.prompt_slab->mapped_size();
            pool.prompt_slab->resize(before);
            if (!measured) { return false; }
            // A retry must remove physical chunks, not just payload slots.
            const size_t chunk = pool.prompt_slab->mapped_size(1);
            const size_t allowed = (mapped - std::min(mapped, deficit)) / chunk * chunk;
            const size_t target = std::min((size_t)(slots - 1), allowed / pool.expert_size);
            if (target <= (size_t)old_slots) { return false; }
            slots = (int)target;
        }
        pool.slots.resize(slots);
        pool.free_slots.clear();
        int head = -1, tail = -1;
        for (int i = 0; i < slots; ++i) {
            auto & slot = pool.slots[i];
            if (slot.state != moe_cache_slot_state::valid) {
                moe_cache_map_erase(pool, i);
                slot = {};
                pool.free_slots.push_back(i);
                continue;
            }
            slot.prev = tail;
            slot.next = -1;
            slot.readers = 0;
            if (tail >= 0) { pool.slots[tail].next = i; } else { head = i; }
            tail = i;
        }
        pool.n_slots = slots;
        pool.lru_head = head;
        pool.lru_tail = tail;
        pool.covers_all_entries = false;
        for (const auto & shape : device.shapes) {
            if (shape.expert_size == pool.expert_size && shape.wtype == pool.wtype) {
                pool.covers_all_entries = (uint64_t)slots >= shape.n_entries;
                break;
            }
        }
        if (bytes >= before) {
            moe_cache_budget_allocation(device, bytes - before, true);
            device.allocated_bytes += bytes - before;
        } else {
            moe_cache_budget_allocation(device, before - bytes, false);
            device.allocated_bytes -= before - bytes;
        }
        return true;
    }
    std::vector<int> ranked;
    for (int i = 0; i < old_slots; ++i) {
        if (pool.slots[i].state == moe_cache_slot_state::valid) { ranked.push_back(i); }
    }
    if (ranked.size() > (size_t)slots) {
        std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
            return moe_cache_lending_uses(pool.slots[a]) > moe_cache_lending_uses(pool.slots[b]);
        });
        device.evictions += ranked.size() - slots;
        ranked.resize(slots);
    }
    std::vector<moe_cache_slot> replacement(slots);
    for (int i : ranked) { if (i < slots) { replacement[i] = pool.slots[i]; } }
    std::vector<std::pair<int, int>> moves;
    moves.reserve(ranked.size());
    int hole = 0;
    for (int i : ranked) {
        if (i < slots) { continue; }
        while (hole < slots && replacement[hole].state == moe_cache_slot_state::valid) { ++hole; }
        GGML_ASSERT(hole < slots);
        moves.emplace_back(hole, i);
        replacement[hole++] = pool.slots[i];
    }
    std::unordered_map<moe_cache_key, int, moe_cache_key_hash> map;
    std::vector<int> free_slots;
    map.reserve(slots);
    free_slots.reserve(slots);
    int head = -1, tail = -1;
    for (int i = 0; i < slots; ++i) {
        auto & slot = replacement[i];
        if (slot.state != moe_cache_slot_state::valid) { free_slots.push_back(i); continue; }
        slot.prev = tail;
        slot.next = -1;
        slot.readers = 0;
        if (tail >= 0) { replacement[tail].next = i; } else { head = i; }
        tail = i;
        map.emplace(slot.key, i);
    }
    for (size_t i = 0; i < moves.size();) {
        size_t end = i + 1;
        while (end < moves.size() && moves[end].first == moves[i].first + (int)(end - i) &&
                moves[end].second == moves[i].second + (int)(end - i)) { ++end; }
        if (!moe_cache_cuda_ok(device, cudaMemcpyAsync(pool.slab + (size_t)moves[i].first * pool.expert_size,
                pool.slab + (size_t)moves[i].second * pool.expert_size, (end - i) * pool.expert_size,
                cudaMemcpyDeviceToDevice, device.compute_stream), "prompt survivor compact", true)) { GGML_ABORT("prompt compaction failed"); }
        i = end;
    }
    if (!moves.empty() && !moe_cache_cuda_ok(device, cudaStreamSynchronize(device.compute_stream), "prompt compaction join", true)) {
        GGML_ABORT("prompt compaction join failed");
    }
    const size_t bytes = (size_t)slots * pool.expert_size;
    if (!pool.prompt_slab->resize(bytes)) { return false; }
    pool.n_slots = slots;
    pool.slots.swap(replacement);
    pool.map.swap(map);
    pool.free_slots.swap(free_slots);
    pool.lru_head = head;
    pool.lru_tail = tail;
    pool.covers_all_entries = false;
    for (const auto & shape : device.shapes) {
        if (shape.expert_size == pool.expert_size && shape.wtype == pool.wtype) {
            pool.covers_all_entries = (uint64_t)slots >= shape.n_entries;
            break;
        }
    }
    moe_cache_budget_allocation(device, before - bytes, false);
    device.allocated_bytes -= before - bytes;
    return true;
}

// Contiguous slabs cannot shrink in place. Recopy retained exact weights after freeing the old slab.
static bool moe_cache_resize_pool(moe_cache_device & device, moe_cache_pool & pool, int slots) {
    for (const auto & shape : device.shapes) {
        if (shape.expert_size == pool.expert_size && shape.wtype == pool.wtype) {
            slots = (int)std::min((size_t)slots, shape.pool_plan_slots);
            break;
        }
    }
    if (slots == pool.n_slots) {
        return true;
    }
    if (pool.prompt_slab) { return moe_cache_resize_prompt_pool(device, pool, slots); }
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
    const int64_t slab_us0 = g_prompt_slab_us.load();
    if (!moe_cache_profile_rerank(*session, start + std::chrono::seconds(30))) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(session->mu);
    // A second compute may have started while the idle wait released the mutex.
    if (session->active_scopes || session->active_nodes || session->stopping) {
        return 0;
    }
    if (moe_cache_keep_prompt()) {
        size_t freed = 0, retained = 0;
        for (auto & device : session->devices) {
            if (device->dead.load() || !device->queue.empty() || device->inflight) { continue; }
            bool supported = !device->pools.empty();
            for (const auto & pool : device->pools) { supported = supported && pool->prompt_slab != nullptr; }
            if (!supported) { continue; }
            struct layer_entry { int layer; size_t bytes = 0; uint64_t uses = 0; std::vector<size_t> slots; };
            std::unordered_map<const void *, int> layers;
            for (const auto & tensor : session->profile_global_tensors) {
                int layer = -1;
                if (moe_cache_layer_number(tensor.name.c_str(), layer)) { layers.emplace(tensor.data, layer); }
            }
            std::map<int, layer_entry> entries;
            std::vector<size_t> kept_slots(device->pools.size(), 0);
            for (size_t p = 0; p < device->pools.size(); ++p) {
                const auto & pool = device->pools[p];
                if (pool->prompt_slots < 0) { pool->prompt_slots = pool->n_slots; }
                for (const auto & slot : pool->slots) {
                    if (slot.state != moe_cache_slot_state::valid) { continue; }
                    ++kept_slots[p];
                    const auto found = layers.find(slot.key.tensor);
                    if (found == layers.end()) { continue; }
                    auto & entry = entries[found->second];
                    if (entry.slots.empty()) { entry.slots.resize(device->pools.size(), 0); }
                    ++entry.slots[p];
                    entry.layer = found->second;
                    entry.bytes += pool->expert_size;
                    entry.uses += moe_cache_lending_uses(slot);
                }
            }
            std::vector<layer_entry> ranked;
            for (const auto & entry : entries) { ranked.push_back(entry.second); }
            std::stable_sort(ranked.begin(), ranked.end(), [](const layer_entry & a, const layer_entry & b) {
                return (long double)a.uses / a.bytes < (long double)b.uses / b.bytes;
            });
            std::unordered_set<int> evicted;
            // Count unmapped chunks, including unused slots, before selecting another whole layer.
            auto released_bytes = [&]() {
                size_t released = 0;
                for (size_t p = 0; p < device->pools.size(); ++p) {
                    const auto & pool = device->pools[p];
                    released += pool->prompt_slab->mapped_size() - pool->prompt_slab->mapped_size(kept_slots[p] * pool->expert_size);
                }
                return released;
            };
            for (const auto & entry : ranked) {
                if (released_bytes() >= bytes - std::min(bytes, freed)) { break; }
                evicted.insert(entry.layer);
                for (size_t p = 0; p < kept_slots.size(); ++p) { kept_slots[p] -= entry.slots[p]; }
            }
            for (const auto & pool : device->pools) {
                int kept = 0;
                for (int i = 0; i < pool->n_slots; ++i) {
                    const auto found = layers.find(pool->slots[i].key.tensor);
                    if (pool->slots[i].state == moe_cache_slot_state::valid && found != layers.end() && evicted.count(found->second)) {
                        moe_cache_slot_reset(*pool, i, true);
                    }
                    kept += pool->slots[i].state == moe_cache_slot_state::valid;
                }
                const size_t before = pool->prompt_slab->mapped_size();
                if (!moe_cache_resize_pool(*device, *pool, kept)) { continue; }
                freed += before - pool->prompt_slab->mapped_size();
                retained += (size_t)pool->n_slots * pool->expert_size;
            }
            MOE_CACHE_LOG("[moe-cache] prompt whole-layer loan: evicted_layers=%zu retained_bytes=%zu released_bytes=%zu ms=%.3f vmm_ms=%.3f\n",
                    evicted.size(), retained, freed, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                    (g_prompt_slab_us.load() - slab_us0) / 1000.0);
        }
        if (freed) { session->prompt_loaned = true; }
        return freed;
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
    if (freed) { session->prompt_loaned = true; }
    MOE_CACHE_LOG("[moe-cache] lent %zu MiB requested=%zu MiB; least-use eviction and exact survivor refill %.3f ms\n",
            freed >> 20, bytes >> 20, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return freed;
}

static void moe_cache_prompt_resume(void * opaque) {
    auto * session = (moe_cache_session *) opaque;
    if (!session) { return; }
    std::lock_guard<std::mutex> lock(session->mu);
    session->prompt_loaned = true;
}

static size_t moe_cache_return(void * opaque) {
    auto * session = (moe_cache_session *)opaque;
    if (!session || session->dormant || session->stopping) {
        return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    const int64_t slab_us0 = g_prompt_slab_us.load();
    if (!moe_cache_profile_rerank(*session, start + std::chrono::seconds(30))) {
        return 0;
    }
    const auto rerank_end = std::chrono::steady_clock::now();
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
        // A real prompt loan distinguishes this return from startup warmup restores.
        int64_t return_reserve_mb = 0;
        if (session->prompt_loaned && device->budget_registered && moe_cache_env_i64("GGML_CUDA_MOE_CACHE_RETURN_RESERVE_MB",
                    1536, (int64_t) session->config.reserve_mb, return_reserve_mb)) {
            const size_t reserve_bytes = (size_t) return_reserve_mb << 20;
            if (reserve_bytes < device->budget_reserve_bytes) {
                std::lock_guard<std::mutex> budget_lock(g_budget_mu);
                auto found = g_physical_budgets.find(device->physical);
                if (found != g_physical_budgets.end()) {
                    auto & reserves = found->second.reserves;
                    auto previous = reserves.find(device->budget_reserve_bytes);
                    if (previous != reserves.end()) {
                        try {
                            reserves.insert(reserve_bytes);
                            reserves.erase(previous);
                            MOE_CACHE_LOG("[moe-cache] CUDA%d prompt-return reserve bytes: startup=%zu previous=%zu decode=%zu\n",
                                    device->physical, session->config.reserve_mb << 20, device->budget_reserve_bytes, reserve_bytes);
                            device->budget_reserve_bytes = reserve_bytes;
                        } catch (const std::bad_alloc &) {
                            MOE_CACHE_LOG("[moe-cache] CUDA%d keeping startup reserve after allocation failure\n", device->physical);
                        }
                    }
                }
            }
        }
        if (!device->budget_claimed || !moe_cache_reserve_table_arena(*session, *device) ||
                !moe_table_prefetch_reserve(*session, *device) ||
                !moe_cache_refresh_budget(*session, *device, "return prompt scratch")) {
            continue;
        }
        const size_t scratch = moe_cache_scratch_total(device->scratch_reserve) + device->table_arena_cap + moe_table_prefetch_capacity(*device);
        const size_t slab_limit = device->budget_limit > scratch ? device->budget_limit - scratch : 0;
        size_t remaining = slab_limit > device->allocated_bytes ? slab_limit - device->allocated_bytes : 0;
        if (moe_cache_keep_prompt()) {
            for (const auto & pool : device->pools) {
                if (pool->prompt_slots < 0) { continue; }
                const size_t before = (size_t)pool->n_slots * pool->expert_size;
                const size_t target = std::min((size_t)pool->prompt_slots, (size_t)pool->n_slots + remaining / pool->expert_size);
                if (target > (size_t)pool->n_slots) { moe_cache_resize_pool(*device, *pool, (int)target); }
                const size_t after = (size_t)pool->n_slots * pool->expert_size;
                restored += after - before;
                remaining -= std::min(remaining, after - before);
                pool->prompt_slots = -1;
            }
            MOE_CACHE_LOG("[moe-cache] retained prompt pools returned: restored_bytes=%zu survivor_H2D_bytes=0 ms=%.3f rerank_ms=%.3f vmm_ms=%.3f\n",
                    restored, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                    std::chrono::duration<double, std::milli>(rerank_end - start).count(), (g_prompt_slab_us.load() - slab_us0) / 1000.0);
            continue;
        }
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
            const size_t max_slots = (size_t)std::min<uint64_t>(shape.n_entries, INT_MAX / blocks);
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
        size_t slots = 0;
        for (const auto & pool : device->pools) { slots += pool->n_slots; }
        MOE_CACHE_LOG("[moe-cache] CUDA%d prompt-return capacity bytes: expert_slots=%zu expert_bytes=%zu grant=%zu reserve=%zu\n",
                device->physical, slots, device->allocated_bytes, device->budget_limit, device->budget_reserve_bytes);
    }
    MOE_CACHE_LOG("[moe-cache] returned %zu MiB to expert slots; exact survivor refill %.3f ms\n",
            restored >> 20, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    return restored;
}

// [PREFILLFAST] GGML_MOE_CACHE_WARM_SINGLE=1: a prompt warm restores the saved profile ranks first.
// With this switch that nested restore only records the ranks; the prompt warm then fills every
// slot once in its own rank order, instead of filling all slots from the profile and replacing
// most of them right away. Off by default.
static thread_local bool g_warm_rank_only = false;
static std::atomic<uint64_t> g_moe_cache_fill_epoch{0};

uint64_t ggml_cuda_moe_cache_fill_epoch(void) {
    return g_moe_cache_fill_epoch.load();
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
    // [PREFILLFAST] GGML_MOE_CACHE_FRESH_NO_REPLAN=1: when the profile load just below runs in this same call
    // (a session rebuilt after B1 suspension), its global plan was made a moment ago from the same profile.
    // The same-session replan then only re-fits the pools to allocator padding (a few slots per pool) at
    // the cost of resizing every pool (about 0.66 s at 12.6 GB). Skip it in that case. Off by default.
    bool fresh_profile_plan = false;
    if (!loading && getenv("GGML_MOE_CACHE_PROFILE_LOAD")) {
        bool attempted_before = true;
        {
            std::lock_guard<std::mutex> lock(session->mu);
            attempted_before = session->profile_load_attempted;
        }
        // Suspension creates a new session. Restore its saved fallback rankings before prompt reranking.
        static const bool rank_only = moe_cache_env_on("GGML_MOE_CACHE_WARM_SINGLE");
        g_warm_rank_only = rank_only;
        const int profile_queued = moe_cache_warm(opaque, tensors, nullptr, n_tensors);
        g_warm_rank_only = false;
        static const bool no_replan = [] { const char * v = getenv("GGML_MOE_CACHE_FRESH_NO_REPLAN"); return v == nullptr || atoi(v) != 0; }();
        fresh_profile_plan = no_replan && !rank_only && !attempted_before && profile_queued > 0;
    }
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
            bool ranked = false;
            if (!moe_cache_profile_load(path, signature, entries, tensors, n_tensors, loaded_storage, loaded_counts, &ranked)) {
                MOE_CACHE_LOG("[moe-cache] profile rejected: %s (missing, mismatched, or corrupt)\n", path);
                return 0;
            }
            session->profile_ranked = ranked;
            if (!session->device_profile_path.empty()) {
                session->device_profile_signature = signature;
                session->device_profile_tensors.assign(tensors, tensors + n_tensors);
            }
            counts = loaded_counts.data();
            MOE_CACHE_LOG("[moe-cache] profile loaded: %s, %zu entries; %s\n", path, entries,
                    ranked ? "ranked-pair data, shape and uniqueness verified; no weight signature" : "sampled weight identity verified");
        } else {
            const bool saved = moe_cache_profile_save(path, signature, entries, tensors, counts, n_tensors);
            MOE_CACHE_LOG("[moe-cache] profile %s: %s, %zu usage counts\n", saved ? "saved" : "save failed", path, entries);
        }
    } else if (loading) {
        return 0;
    }
    int64_t global_flag = 0;
    const bool whole_only = moe_cache_whole_only();
    if (whole_only || (loading && moe_cache_env_i64("GGML_MOE_CACHE_PROFILE_GLOBAL", 0, 1, global_flag) && global_flag == 1)) {
        std::lock_guard<std::mutex> lock(session->mu);
        // A first live-count warm has no saved profile to load. Plan complete bundles
        // from its descriptors too, but preserve an already loaded profile's ranks.
        // [ALLOCSIM] GGML_MOE_CACHE_PLAN_REFRESH=1 also replans prompt-only (no saved profile) sessions from live counts.
        static const bool plan_refresh = moe_cache_env_on("GGML_MOE_CACHE_PLAN_REFRESH");
        if (loading || session->profile_global_tensors.empty() || (plan_refresh && observed && session->profile_counts.empty())) {
            session->profile_global_tensors.clear();
            session->profile_global_tensors.reserve(n_tensors);
            for (int i = 0; i < n_tensors; ++i) {
                moe_cache_profile_global_tensor tensor;
                tensor.name = tensors[i].name;
                tensor.data = tensors[i].data;
                tensor.expert_size = ggml_row_size((ggml_type)tensors[i].type, tensors[i].n_in) * (size_t)tensors[i].n_out;
                tensor.host_stride = tensors[i].expert_size;
                tensor.wtype = tensors[i].type;
                tensor.counts.assign(counts[i], counts[i] + tensors[i].n_expert);
                session->profile_global_tensors.push_back(std::move(tensor));
            }
        }
    }
    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::seconds(30);
    if (!loading && !moe_cache_profile_rerank(*session, deadline)) {
        return 0;
    }
    if (fresh_profile_plan) {
        MOE_CACHE_LOG("[moe-cache] global profile same-session replan skipped: plan made by this call's profile load (GGML_MOE_CACHE_FRESH_NO_REPLAN)\n");
    }
    if (!loading && !fresh_profile_plan && !session->profile_global_tensors.empty()) {
        std::lock_guard<std::mutex> lock(session->mu);
        if (!session->active_scopes && !session->active_nodes && !session->device_ready.load() && !session->device_frozen) {
            for (const auto & device : session->devices) {
                if (device->dead.load() || !device->queue.empty() || device->inflight) { continue; }
                std::unique_lock<std::mutex> workspace_lock(device->dispatch_mu, std::try_to_lock);
                if (!workspace_lock.owns_lock()) { continue; }
                std::vector<moe_cache_shape *> shapes;
                for (auto & shape : device->shapes) { if (shape.n_tensors > 0 && shape.pool >= 0) { shapes.push_back(&shape); } }
                const size_t budget = device->allocated_bytes;
                if (moe_cache_build_global_profile(*session, *device, shapes, budget, true)) {
                    MOE_CACHE_LOG("[moe-cache] CUDA%d global profile same-session replan: budget=%zu actual_bytes=%zu; idle tables released, shrink before grow\n",
                            device->physical, budget, device->allocated_bytes);
                }
            }
        }
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
    std::unordered_map<moe_cache_key, size_t, moe_cache_key_hash> global_ranks;
    std::unordered_set<moe_cache_key, moe_cache_key_hash> selected;
    {
        std::lock_guard<std::mutex> lock(session->mu);
        for (const auto & device : session->devices) {
            global_ranks.insert(device->profile_global_ranks.begin(), device->profile_global_ranks.end());
            if (whole_only) {
                selected.insert(device->profile_global_selected.begin(), device->profile_global_selected.end());
                for (const auto & pool : device->pools) {
                    for (int i = 0; i < pool->n_slots; ++i) {
                        const auto & slot = pool->slots[i];
                        if (!session->device_frozen && !slot.readers && slot.state == moe_cache_slot_state::valid &&
                                !device->profile_global_selected.count(slot.key)) {
                            moe_cache_slot_reset(*pool, i, true);
                        }
                    }
                }
            }
        }
    }
    const bool global_profile = global_ranks.size() == entries;
    if (whole_only && (!global_profile || selected.empty())) {
        MOE_CACHE_LOG("[moe-cache] whole-only warm refused: a complete global profile plan is required\n");
        return 0;
    }
    if (whole_only) {
        MOE_CACHE_LOG("[moe-cache] whole-only warm: selected_slices=%zu; no partial backfill\n", selected.size());
    }
    if (global_profile) {
        MOE_CACHE_LOG("[moe-cache] global profile warm: pure saved-profile bundle order overrides prompt counts and PROFILE_BLEND_PCT; pool plan preserved\n");
    }
    else if (!session->profile_global_tensors.empty()) {
        MOE_CACHE_LOG("[moe-cache] global profile warm unavailable: planned_entries=%zu expected=%zu; fallback=default-warm-ranking\n",
                global_ranks.size(), entries);
    }
    struct candidate { int tensor; int32_t expert; uint64_t count; bool current; long double startup_score; bool first_layer; };
    std::vector<candidate> ranked;
    ranked.reserve(entries);
    int64_t blend_pct = 0;
    const bool blending = !global_profile && !loading && !session->profile_counts.empty() &&
        moe_cache_env_i64("GGML_MOE_CACHE_PROFILE_BLEND_PCT", 0, 100, blend_pct) && blend_pct > 0;
    if (blending) {
        MOE_CACHE_LOG("[moe-cache] post-prompt profile blend: profile_pct=%lld; normalized per tensor, decode swap scores unchanged\n",
                (long long) blend_pct);
    }
    for (int i = 0; i < n_tensors; ++i) {
        long double observed_total = 0;
        for (int32_t e = 0; e < tensors[i].n_expert; ++e) { observed_total += counts[i][e]; }
        long double profile_total = 0;
        if (blending) {
            for (int32_t e = 0; e < tensors[i].n_expert; ++e) {
                const auto prior = session->profile_counts.find({tensors[i].data, e});
                if (prior != session->profile_counts.end()) { profile_total += prior->second; }
            }
        }
        for (int32_t e = 0; e < tensors[i].n_expert; ++e) {
            const moe_cache_key key{tensors[i].data, e};
            if (loading && counts[i][e]) {
                session->profile_counts[key] = counts[i][e];
            }
            if (global_profile) {
                if (whole_only && !selected.count(key)) { continue; }
                const auto prior = session->profile_counts.find(key);
                static const bool plan_refresh = moe_cache_env_on("GGML_MOE_CACHE_PLAN_REFRESH");
                const uint64_t heat = (plan_refresh && session->profile_counts.empty()) ? counts[i][e] :
                    prior == session->profile_counts.end() ? 0 : prior->second;
                ranked.push_back({i, e, heat, true, 0, false});
                continue;
            }
            int layer = 0;
            moe_cache_layer_number(tensors[i].name, layer);
            const size_t bytes = ggml_row_size((ggml_type)tensors[i].type, tensors[i].n_in) * (size_t)tensors[i].n_out;
            // Output pruning can observe fewer tokens in the last layer. Compare
            // conditional use rates, preserving the original counts on disk.
            const long double rate = counts[i][e] / std::max(1.0L, observed_total);
            const long double priority = rate * (1 + 4.0L / (layer + 1)) / bytes;
            if (blending) {
                const auto prior = session->profile_counts.find(key);
                const long double profile_rate = prior == session->profile_counts.end() ? 0 :
                    prior->second / std::max(1.0L, profile_total);
                const long double mixed = ((100 - blend_pct) * rate + blend_pct * profile_rate) / 100;
                // Keep lending survivors in the same order as the normalized seed.
                const uint64_t heat = (uint64_t) (mixed * 1000000000.0L);
                ranked.push_back({i, e, heat, true, mixed, false});
                continue;
            }
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
    if (loading && g_warm_rank_only) {
        MOE_CACHE_LOG("[moe-cache] profile ranks restored for %zu entries; fills left to the prompt warm (GGML_MOE_CACHE_WARM_SINGLE)\n", ranked.size());
        return 0;
    }
    std::stable_sort(ranked.begin(), ranked.end(), [&](const candidate & a, const candidate & b) {
        if (global_profile) {
            return global_ranks.at({tensors[a.tensor].data, a.expert}) < global_ranks.at({tensors[b.tensor].data, b.expert});
        }
        if (blending && a.startup_score != b.startup_score) { return a.startup_score > b.startup_score; }
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
    // LOADFAST (opt-in GGML_MOE_CACHE_WARM_BATCH=1, startup profile only): the slot choice below is
    // unchanged, but each fill it queues is taken back from the worker queue and copied here in
    // batches of async copies straight from the (pinned) source, one wait per batch, two batches
    // in flight, instead of the worker's stage memcpy + copy + wait per expert.
    // PP-1: every pool of this warm has its final size here; only fills follow. A caller that overlaps the
    // fills with other GPU work waits for this count to move before it allocates anything of its own.
    g_moe_cache_fill_epoch.fetch_add(1);
    struct warm_fill { moe_cache_device * device; moe_cache_job job; char * destination; };
    struct warm_batch { std::vector<warm_fill> fills; size_t bytes = 0; cudaEvent_t done = nullptr; bool issued = false; };
    const char * warm_batch_env = getenv("GGML_MOE_CACHE_WARM_BATCH");
    // [PPWARM] GGML_MOE_CACHE_WARM_BATCH=2 also batches the post-prompt warm (B1 return), which otherwise
    // goes through the worker's per-expert stage copy and wait inside prompt time. Off by default.
    const int warm_batch_mode = warm_batch_env ? atoi(warm_batch_env) : 0;
    const bool batch_warm = loading ? warm_batch_mode != 0 : warm_batch_mode >= 2;
    // [PPWARM] GGML_MOE_CACHE_PROMPT_WARM_CAP=N: queue at most N hottest fills in the post-prompt warm;
    // the remaining free slots are left to decode-time admission. Test knob, off (0) by default.
    static const int64_t prompt_warm_cap = [] { const char * v = getenv("GGML_MOE_CACHE_PROMPT_WARM_CAP"); return v ? std::max<int64_t>(0, atoll(v)) : 0; }();
    bool warm_capped = false;
    const size_t batch_bytes_max = (size_t) 256 << 20;
    std::vector<std::pair<moe_cache_device *, cudaStream_t>> warm_streams;
    warm_batch batches[2];
    int filling = 0;
    size_t batch_copies = 0, batch_count = 0;
    auto warm_stream = [&](moe_cache_device * device) -> cudaStream_t {
        for (const auto & s : warm_streams) { if (s.first == device) { return s.second; } }
        cudaStream_t stream = nullptr;
        ggml_cuda_set_device(device->logical);
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { (void) cudaGetLastError(); stream = nullptr; }
        warm_streams.emplace_back(device, stream);
        return stream;
    };
    // Same slot bookkeeping as the worker after a copy.
    auto warm_finish = [&](warm_batch & batch, bool copied) {
        std::lock_guard<std::mutex> lock(session->mu);
        for (auto & f : batch.fills) {
            auto * device = f.device;
            if (f.job.pool < 0 || f.job.pool >= (int) device->pools.size()) { continue; }
            auto & pool = *device->pools[f.job.pool];
            if (f.job.slot < 0 || f.job.slot >= pool.n_slots) { continue; }
            moe_cache_slot & slot = pool.slots[f.job.slot];
            if (slot.state != moe_cache_slot_state::copying || slot.generation != f.job.generation || !(slot.key == f.job.key)) { continue; }
            if (copied && (!slot.prediction || !slot.prediction->closed)) {
                slot.state = moe_cache_slot_state::valid;
                moe_cache_lru_push_back(pool, f.job.slot);
                device->demand_count.erase(f.job.key);
                device->fills++;
            } else {
                moe_cache_slot_reset(pool, f.job.slot, true);
                device->fill_failures++;
            }
        }
        batch.fills.clear();
        batch.bytes = 0;
        batch.issued = false;
        session->idle_cv.notify_all();
    };
    auto warm_wait = [&](warm_batch & batch) {
        if (!batch.issued) { return; }
        bool ok = true;
        if (batch.done) {
            ok = cudaEventSynchronize(batch.done) == cudaSuccess;
        } else {
            for (const auto & s : warm_streams) { if (s.second) { ok = cudaStreamSynchronize(s.second) == cudaSuccess && ok; } }
        }
        if (!ok) { (void) cudaGetLastError(); }
        warm_finish(batch, ok);
    };
    auto warm_issue = [&](warm_batch & batch) {
        if (batch.fills.empty()) { return; }
        bool ok = true;
        std::unique_lock<std::mutex> fill_lock(session->fill_mu, std::defer_lock);
        if (session->config.serial_fill) { fill_lock.lock(); }
        cudaStream_t last = nullptr;
        for (auto & f : batch.fills) {
            cudaStream_t stream = warm_stream(f.device);
            ggml_cuda_set_device(f.device->logical);
            ok = ok && !f.device->dead.load() && !moe_cache_fail(*session, "insert") &&
                    moe_resident_host_restore(f.job.source, f.job.bytes) &&
                    moe_h2d_async(f.destination, f.job.source, f.job.bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
            last = stream;
            ++batch_copies;
        }
        ++batch_count;
        // one device in practice; with several, wait on every warm stream instead of one event
        if (ok && warm_streams.size() == 1 && last) {
            if (!batch.done && cudaEventCreateWithFlags(&batch.done, cudaEventDisableTiming) != cudaSuccess) { batch.done = nullptr; }
            if (batch.done && cudaEventRecord(batch.done, last) != cudaSuccess) { ok = false; }
        } else if (batch.done) {
            (void) cudaEventDestroy(batch.done);
            batch.done = nullptr;
        }
        batch.issued = true;
        if (!ok) {
            (void) cudaGetLastError();
            for (const auto & s : warm_streams) { if (s.second) { (void) cudaStreamSynchronize(s.second); } }
            warm_finish(batch, false);
        }
    };
    auto warm_flush = [&]() {
        auto & current = batches[filling];
        auto & other = batches[1 - filling];
        warm_wait(other);       // keep at most two batches in flight
        warm_issue(current);
        filling = 1 - filling;
    };
    int queued = 0;
    std::unordered_set<moe_cache_pool *> settled;
    struct warm_flag { warm_flag() { g_moe_cache_in_warm = true; } ~warm_flag() { g_moe_cache_in_warm = false; } } warm_flag_guard;
    for (const auto & c : ranked) {
        if (std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if (moe_bank_has(tensors[c.tensor].data, c.expert)) {
            continue; // computed by the always-resident bank
        }
        if (!loading && prompt_warm_cap > 0 && queued >= prompt_warm_cap) {
            warm_capped = true;
            break;
        }
        std::unique_ptr<moe_cache_node, decltype(&moe_cache_end)> node(begin(c.tensor), moe_cache_end);
        if (!node) {
            continue;
        }
        if (moe_resident_lookup_device((const char *)node->host_base + (size_t)c.expert *
                moe_cache_host_stride(*node->device, node->host_base, node->expert_size), node->expert_size, node->device->logical)) { continue; }
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
                    if (batch_warm && inserts_left == 0) {
                        // take the fill just queued back from the worker (it has not seen it: we hold mu)
                        const auto job = std::find_if(device.queue.begin(), device.queue.end(),
                                [&](const moe_cache_job & j) { return j.key == key && !j.prediction; });
                        if (job != device.queue.end() && pool.slab && job->slot >= 0 && job->slot < pool.n_slots) {
                            batches[filling].fills.push_back({ &device, *job, pool.slab + (size_t) job->slot * pool.expert_size });
                            batches[filling].bytes += job->bytes;
                            device.queued_bytes = job->bytes <= device.queued_bytes ? device.queued_bytes - job->bytes : 0;
                            device.queue.erase(job);
                            if (device.queue.empty()) { wake = false; }
                        }
                    }
                }
            }
        }
        if (wake) {
            session->cv.notify_all();
        }
        if (batch_warm && batches[filling].bytes >= batch_bytes_max) {
            warm_flush();
        }
    }
    if (batch_warm) {
        warm_flush();
        warm_wait(batches[0]);
        warm_wait(batches[1]);
        for (auto & b : batches) { if (b.done) { (void) cudaEventDestroy(b.done); } }
        for (const auto & s : warm_streams) { if (s.second) { (void) cudaStreamDestroy(s.second); } }
        MOE_CACHE_LOG("[moe-cache] [LOADFAST] %s warm batched: %zu copies in %zu batches (direct from host source, one wait per batch)\n",
                loading ? "profile" : "prompt", batch_copies, batch_count);
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
    if (warm_capped) {
        MOE_CACHE_LOG("[moe-cache] [PPWARM] prompt warm capped at %lld fills (GGML_MOE_CACHE_PROMPT_WARM_CAP)\n", (long long) prompt_warm_cap);
    }
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
    // fork PREFETCHFIX: the asynchronous path replaces the staging-table prefetch entirely.
    if (mode != 0 && moe_pf_mode() != 0) {
        std::lock_guard<std::mutex> pf_lock(g_pf.mu);
        const void * owner = g_pf.owner.load(std::memory_order_acquire);
        if (owner == nullptr || owner == session) {
            g_pf.requested.store(mode, std::memory_order_relaxed);
            g_pf.owner.store(session, std::memory_order_release);
        } else {
            // Graph replay bypasses host hooks; scratch cannot change owners
            // until the old live session and its captured tables are drained.
            MOE_CACHE_LOG("[moe-prefetch-async] off: another live cache session owns prediction buffers\n");
        }
    } else if (mode == 0) {
        std::lock_guard<std::mutex> pf_lock(g_pf.mu);
        if (g_pf.owner.load(std::memory_order_acquire) == session) {
            g_pf.requested.store(0, std::memory_order_relaxed);
            g_pf.active.store(false, std::memory_order_release);
            g_pf.src_ready = false;
        }
    }
    session->prefetch_enabled = mode != 0 && moe_pf_mode() == 0;
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
                if (ids[rank] >= 0 && ids[rank] < desc.n_expert && desc.data) {
                    // fork (cachehit): the device table's permanent mask came from the same lookup; the span
                    // scan costs ~3 us per call and runs for every rank and projection of every layer.
                    const auto known = session->device_tables.find(desc.data);
                    const bool permanent = known != session->device_tables.end() && (size_t) ids[rank] < known->second.permanent.size() ?
                        known->second.permanent[ids[rank]] != 0 :
                        moe_resident_lookup_device((const char *)desc.data + (size_t)ids[rank] * source_strides[t], desc.expert_size, device.logical) != nullptr;
                    if (permanent) { continue; }
                }
                if (ids[rank] < 0 || ids[rank] >= desc.n_expert || !desc.data || desc.expert_size > budget ||
                        moe_bank_has(desc.data, ids[rank])) {
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
#elif defined(_WIN32)
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (size_t) st.ullAvailPhys : 0;
#else
    return 0;
#endif
}

#if defined(__linux__)
// Hosts without a cgroup v2 hierarchy (Docker on cgroup v1, older distributions) used to fail the headroom check and
// silently lose the device route tables. v1: the memory controller's limit minus its usage, with the kernel's
// "no limit" value (a huge number) meaning unlimited. No limit anywhere: nothing to charge against, like bare metal
// (the caller still checks MemAvailable and the RAM floor).
static bool moe_cache_cgroup_v1_headroom(size_t extra, size_t registered, uint64_t * peak_current) {
    std::string path;
    FILE * f = fopen("/proc/self/cgroup", "r");
    char line[4096];
    while (f && fgets(line, sizeof(line), f)) {
        // "<id>:<controllers>:<path>", the memory controller is listed among the comma-separated controllers
        const char * c1 = strchr(line, ':');
        const char * c2 = c1 ? strchr(c1 + 1, ':') : nullptr;
        if (!c2) {
            continue;
        }
        const std::string controllers(c1 + 1, c2);
        const std::string list = "," + controllers + ",";
        if (list.find(",memory,") != std::string::npos) {
            path = c2 + 1;
            if (!path.empty() && path.back() == '\n') {
                path.pop_back();
            }
            break;
        }
    }
    if (f) {
        fclose(f);
    }
    auto read = [](const std::string & file, uint64_t & value) {
        FILE * input = fopen(file.c_str(), "r");
        char text[64] = {};
        const bool ok = input && fgets(text, sizeof(text), input);
        if (input) {
            fclose(input);
        }
        if (!ok) {
            return false;
        }
        char * end = nullptr;
        errno = 0;
        value = strtoull(text, &end, 10);
        return errno == 0 && end != text;
    };
    // a container usually sees its own group as the controller root; try the reported path, then the mount itself
    const std::string roots[2] = { "/sys/fs/cgroup/memory" + path, "/sys/fs/cgroup/memory" };
    for (const auto & root : roots) {
        uint64_t limit = 0, usage = 0;
        if (!read(root + "/memory.limit_in_bytes", limit) || !read(root + "/memory.usage_in_bytes", usage)) {
            continue;
        }
        if (limit >= (1ull << 60)) {
            break; // unlimited
        }
        uint64_t file = 0, inactive = 0, shmem = 0, dirty = 0, writeback = 0;
        if (FILE * stat = fopen((root + "/memory.stat").c_str(), "r")) {
            char name[128];
            unsigned long long value;
            while (fscanf(stat, "%127s %llu", name, &value) == 2) {
                if (strcmp(name, "total_cache") == 0) { file = value; }
                else if (strcmp(name, "total_inactive_file") == 0) { inactive = value; }
                else if (strcmp(name, "total_shmem") == 0) { shmem = value; }
                else if (strcmp(name, "total_dirty") == 0) { dirty = value; }
                else if (strcmp(name, "total_writeback") == 0) { writeback = value; }
            }
            fclose(stat);
        }
        const uint64_t credit = ggml_cuda_moe_cache_host_file_credit(file, inactive, shmem, dirty, writeback, registered);
        const uint64_t charged = usage - std::min(usage, credit);
        if (peak_current) {
            *peak_current = std::max(*peak_current, usage);
        }
        MOE_CACHE_LOG("[moe-cache] host service budget source: cgroup v1 limit minus usage (usage=%llu limit=%llu clean_file_credit=%llu extra=%zu path=%s)\n",
                (unsigned long long) usage, (unsigned long long) limit, (unsigned long long) credit, extra, root.c_str());
        return charged <= limit && extra <= limit - charged;
    }
    MOE_CACHE_LOG("[moe-cache] host service budget source: no cgroup memory limit, MemAvailable only (extra=%zu)\n", extra);
    return true;
}
#endif

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
        return moe_cache_cgroup_v1_headroom(extra, registered, peak_current);
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
            // a "0::/" line whose group has no v2 files here (containers that mount a v1 memory controller): use v1 / no-limit rules
            return moe_cache_cgroup_v1_headroom(extra, registered, peak_current);
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
#elif defined(_WIN32)
    // No cgroups on Windows; the physical RAM floor check in the caller is the only guard.
    GGML_UNUSED(extra);
    GGML_UNUSED(registered);
    if (peak_current) { *peak_current = 0; }
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
    const size_t floor = 7ull << 30, margin = 2ull << 30;
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
    if (moe_cache_whole_only()) { return; }
    if (!session.device_expert_counts || session.devices.size() != 1 ||
            session.active_scopes || session.active_nodes) {
        return;
    }
    auto & device = *session.devices[0];
    const auto start = std::chrono::steady_clock::now();
    const size_t copy_limit = session.device_adapt_copy_cap;
    size_t copied = 0, pinned_source_bytes = 0, original_source_bytes = 0;
    int fills = 0, fallback_skips = 0;
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
        if (!session.device_profile_slices.empty()) {
            const bool saved = ggml_moe_generation_profile_write(session.device_profile_path, session.device_profile_signature,
                    session.device_profile_slices, session.device_previous_counts,
                    session.device_capture_generation, session.device_adapt_scopes);
            MOE_CACHE_LOG("[moe-cache] device profile %s: scopes=%llu tail_incomplete=1\n",
                    saved ? "saved" : "save failed", (unsigned long long) session.device_adapt_scopes);
        }
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
                // fork (cachehit): cheap count test first; the permanent mask was filled by the
                // same lookup at prepare, so rescanning every span here (~220 ms/update) is gone.
                if (used < (uint64_t)std::max(1, session.config.readmit_after)) { continue; }
                if (e < (int32_t) table.permanent.size() ? table.permanent[e] != 0 :
                        moe_resident_lookup_device((const char *)table.desc.data + (size_t)e * table.host_stride,
                        table.desc.expert_size, device.logical) != nullptr) { continue; }
                if (pool->map.find({table.desc.data, e}) == pool->map.end()) {
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
        std::vector<uint8_t> hot(session.device_publish_async ? recent.size() : max_experts);
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
            if (queued.size() == 256) {
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
                    copy_error = moe_h2d_async(fill.destination, fill.source, fill.bytes,
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
        const bool publish_async = session.device_publish_async && session.device_adapt_stream;
        auto publish = [&](void * destination, const void * source, size_t bytes) {
            if (publish_async) {
                CUDA_CHECK(moe_h2d_async(destination, source, bytes, cudaMemcpyHostToDevice, session.device_adapt_stream));
            } else {
                CUDA_CHECK(cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice));
            }
        };
        for (auto * table : changed) {
            // Captured graphs retain this allocation; only its entries change.
            publish(table->device, table->addresses.data(), table->addresses.size() * sizeof(void *));
            if (table->staging_device) {
                publish(table->staging_device, table->addresses.data(), table->addresses.size() * sizeof(void *));
            }
            if (table->hot) {
                uint8_t * table_hot = hot.data() + (publish_async ? table->count_offset : 0);
                for (size_t e = 0; e < table->addresses.size(); ++e) {
                    table_hot[e] = table->addresses[e] != table->fallback[e];
                }
                publish(table->hot, table_hot, table->addresses.size());
            }
        }
        // All host rows and slot addresses stay live until publication finishes.
        if (publish_async) {
            CUDA_CHECK(cudaStreamSynchronize(session.device_adapt_stream));
        }
        MOE_CACHE_LOG("[moe-cache] device adaptation: scopes=%llu counter_bytes=%zu fills=%d copied_bytes=%zu cap_bytes=%zu fallback_staging_bytes=%zu fallback_skips=%d pinned_source_bytes=%zu original_source_bytes=%zu elapsed_ms=%.3f; stable tables publish_async=%d\n",
                (unsigned long long)session.device_adapt_scopes,
                recent.size() * sizeof(recent[0]), fills, copied, copy_limit,
                session.device_fallback_staging_bytes, fallback_skips, pinned_source_bytes, original_source_bytes, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(), publish_async);
    } catch (const std::bad_alloc &) {
        MOE_CACHE_LOG("[moe-cache] device adaptation skipped: host allocation failed\n");
    }
}

// fork CACHEHIT: decayed-score swap policy.
//
// Every swap_interval scopes the per-expert route counters are read back with
// an async copy on the compute stream (no device sync). One scope later the
// host folds them into decayed scores, picks per-pool candidate/victim pairs,
// orders all pairs globally by gain and takes a bounded batch. Victims are
// switched to their exact host fallback by a tiny kernel on the compute
// stream; the copy stream waits for that kernel, then copies the candidates'
// exact bytes. A later scope sees the copy event done and publishes the new
// addresses the same way. Readers always see either the old slot (still
// intact), the host fallback, or the new slot after its bytes landed, so the
// math is unchanged. Only table contents change; captured graphs stay valid.
static __global__ void moe_swap_apply(const moe_swap_write * writes, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const moe_swap_write w = writes[i];
        if (w.bytes == 8) {
            *(unsigned long long *)(uintptr_t)w.dst = w.value;
        } else {
            *(uint8_t *)(uintptr_t)w.dst = (uint8_t)w.value;
        }
    }
}

static void moe_cache_swap_note_stream(moe_cache_session & session, ggml_backend_cuda_context & ctx) {
    if (!session.swap_enabled) {
        return;
    }
    cudaStream_t stream = ctx.stream(ctx.device, 0);
    const int n = session.swap_n_streams.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        if (session.swap_streams[i] == stream) {
            return;
        }
    }
    if (n >= 4) {
        session.swap_stream_overflow.store(true);
        return;
    }
    session.swap_streams[n] = stream;
    session.swap_n_streams.store(n + 1, std::memory_order_release);
}

static void moe_cache_swap_free(moe_cache_session & session) {
    for (cudaEvent_t * e : {&session.swap_counts_event, &session.swap_pub_event, &session.swap_copy_event,
            &session.swap_join_event[0], &session.swap_join_event[1], &session.swap_join_event[2], &session.swap_join_event[3],
            &session.swap_writes_event[0], &session.swap_writes_event[1]}) {
        if (*e) { (void) cudaEventDestroy(*e); *e = nullptr; }
    }
    if (session.swap_copy_stream) { (void) cudaStreamDestroy(session.swap_copy_stream); session.swap_copy_stream = nullptr; }
    if (session.swap_counts_host) { (void) cudaFreeHost(session.swap_counts_host); session.swap_counts_host = nullptr; }
    for (auto *& w : session.swap_writes_host) {
        if (w) { (void) cudaFreeHost(w); w = nullptr; }
    }
    session.swap_writes_cap = 0;
    session.swap_counts_pending = false;
    (void) cudaGetLastError();
}

static bool moe_cache_swap_groups(moe_cache_session & session, bool census = false) {
    auto & groups = session.swap_groups;
    groups.clear();
    for (auto & entry : session.device_tables) {
        auto & table = entry.second;
        int layer = -1;
        if (!table.desc.name || strncmp(table.desc.name, "blk.", 4) != 0 ||
                !moe_cache_layer_number(table.desc.name, layer) || layer >= 1024) { return false; }
        const char * suffix = strchr(table.desc.name + 4, '.');
        if (!suffix) { return false; }
        ++suffix;
        const int k = strcmp(suffix, "ffn_up_exps.weight") == 0 ? 0 :
            strcmp(suffix, "ffn_gate_exps.weight") == 0 ? 1 : strcmp(suffix, "ffn_down_exps.weight") == 0 ? 2 : -1;
        if (k < 0) { return false; }
        if (groups.size() <= (size_t) layer) { groups.resize(layer + 1, {nullptr, nullptr, nullptr}); }
        if (groups[layer][k]) { return false; }
        groups[layer][k] = &table;
        for (auto & pool : session.devices[0]->pools) {
            if (pool->slab && pool->expert_size == table.desc.expert_size && pool->wtype == table.desc.type) {
                table.swap_pool = pool.get();
                break;
            }
        }
    }
    size_t partial = 0, complete = 0, bank_complete = 0;
    for (size_t layer = 0; layer < groups.size(); ++layer) {
        const auto & group = groups[layer];
        if (!group[0] && !group[1] && !group[2]) { continue; }
        if (!group[0] || !group[1] || !group[2] || group[0]->desc.n_expert != group[1]->desc.n_expert ||
                group[0]->desc.n_expert != group[2]->desc.n_expert) { return false; }
        const size_t previous_complete = complete, previous_partial = partial;
        size_t resident_bytes = 0;
        for (int32_t e = 0; e < group[0]->desc.n_expert; ++e) {
            int resident = 0;
            for (auto * table : group) {
                bool present = table->permanent[e];
                if (!present && table->swap_pool) {
                    const auto found = table->swap_pool->map.find({table->desc.data, e});
                    present = found != table->swap_pool->map.end() && (!census ||
                            table->swap_pool->slots[found->second].state == moe_cache_slot_state::valid);
                }
                resident += present;
                if (present) { resident_bytes += table->desc.expert_size; }
            }
            partial += resident > 0 && resident < 3;
            complete += resident == 3;
            bank_complete += resident == 3 && group[0]->bank[e] && group[1]->bank[e] && group[2]->bank[e];
        }
        if (census) {
            MOE_CACHE_LOG("[moe-cache] final residency: layer=%zu complete=%zu partial=%zu resident_slice_bytes=%zu\n",
                    layer, complete - previous_complete, partial - previous_partial, resident_bytes);
        }
    }
    MOE_CACHE_LOG("[moe-cache] whole swap %s: complete=%zu partial=%zu layers=%zu\n",
            census ? "final" : "initial", complete, partial, groups.size());
    if (bank_complete) {
        MOE_CACHE_LOG("[moe-cache] whole swap %s resident complete experts: cache %zu + bank %zu = %zu\n",
                census ? "final" : "initial", complete - bank_complete, bank_complete, complete);
    }
    return !groups.empty() && partial == 0;
}

static bool moe_cache_swap_init(moe_cache_session & session) {
    if (session.swap_counts_host) {
        return true;
    }
    const size_t n = session.device_previous_counts.size();
    const int writes_cap = 2 * 4096 + 64;
    bool ok = cudaStreamCreateWithFlags(&session.swap_copy_stream, cudaStreamNonBlocking) == cudaSuccess &&
        cudaHostAlloc((void **)&session.swap_counts_host, n * sizeof(unsigned long long), cudaHostAllocPortable) == cudaSuccess &&
        cudaHostAlloc((void **)&session.swap_writes_host[0], writes_cap * sizeof(moe_swap_write), cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess &&
        cudaHostAlloc((void **)&session.swap_writes_host[1], writes_cap * sizeof(moe_swap_write), cudaHostAllocPortable | cudaHostAllocMapped) == cudaSuccess;
    for (cudaEvent_t * e : {&session.swap_counts_event, &session.swap_pub_event, &session.swap_copy_event,
            &session.swap_join_event[0], &session.swap_join_event[1], &session.swap_join_event[2], &session.swap_join_event[3],
            &session.swap_writes_event[0], &session.swap_writes_event[1]}) {
        ok = ok && cudaEventCreateWithFlags(e, cudaEventDisableTiming) == cudaSuccess;
    }
    if (!ok) {
        MOE_CACHE_LOG("[moe-cache] swap policy disabled: stream/event/pinned allocation failed\n");
        moe_cache_swap_free(session);
        return false;
    }
    if (session.swap_whole && !moe_cache_swap_groups(session)) {
        if (moe_cache_whole_only()) {
            MOE_CACHE_LOG("[moe-cache] whole-only swaps disabled: incomplete projections, names or resident bundles\n");
            moe_cache_swap_free(session);
            return false;
        }
        MOE_CACHE_LOG("[moe-cache] whole swap refused: incomplete projections, names or resident bundles; using slice policy\n");
        session.swap_whole = false;
        session.swap_groups.clear();
    }
    session.swap_writes_cap = writes_cap;
    if (g_swscore && g_hf_experts > 0) {
        // fork SWAPPOL: score counts restart with swap_prev; the readback below orders after this on the same stream.
        const size_t bytes = (size_t) MOE_SWST_LAYERS * g_hf_experts * sizeof(unsigned long long);
        if (cudaMemsetAsync(g_swscore, 0, bytes, session.swap_streams[0]) != cudaSuccess) {
            (void) cudaGetLastError();
            moe_cache_swap_free(session);
            return false;
        }
        memset(g_swscore_host, 0, bytes);
        MOE_CACHE_LOG("[moe-cache] swap score: mode=%d (1 kept, 2 weight, 3 weight+kept, 4 count) skipw=%u/1024\n", moe_swap_score_mode(), moe_swap_skipw());
    }
    session.swap_prev.assign(n, 0);
    session.swap_score.assign(n, 0.0f);
    int64_t prior_pct = 0;
    if (!session.profile_counts.empty() &&
            moe_cache_env_i64("GGML_MOE_CACHE_PROFILE_PRIOR_PCT", 0, 1000, prior_pct) && prior_pct > 0) {
        session.swap_profile_strength = (float) prior_pct / 100;
        const char * decay = getenv("GGML_MOE_CACHE_PROFILE_PRIOR_DECAY");
        session.swap_profile_decay = decay && strcmp(decay, "1") == 0;
        session.swap_observed_score.assign(n, 0.0f);
        session.swap_profile_rate.assign(n, 0.0f);
        session.swap_profile_seed.assign(n, 0.0f);
        for (const auto & entry : session.device_tables) {
            const auto & table = entry.second;
            long double total = 0;
            for (int32_t e = 0; e < table.desc.n_expert; ++e) {
                const auto count = session.profile_counts.find({table.desc.data, e});
                if (count != session.profile_counts.end()) { total += count->second; }
            }
            if (total == 0) { continue; }
            for (int32_t e = 0; e < table.desc.n_expert; ++e) {
                const auto count = session.profile_counts.find({table.desc.data, e});
                if (count != session.profile_counts.end()) {
                    session.swap_profile_rate[table.count_offset + e] = (float) (count->second / total);
                }
            }
        }
        MOE_CACHE_LOG("[moe-cache] swap profile prior: pct=%lld mode=%s; scale from first observed per-table route snapshot\n",
                (long long) prior_pct, session.swap_profile_decay ? "decaying-seed" : "persistent");
    }
    return true;
}

// Launch queued table writes on the first compute stream. The host buffer is
// double-buffered and reused only after its previous kernel completed.
static bool moe_cache_swap_launch(moe_cache_session & session, const std::vector<moe_swap_write> & writes) {
    if (writes.empty()) {
        return true;
    }
    cudaStream_t main = session.swap_streams[0];
    size_t done = 0;
    while (done < writes.size()) {
        const int k = session.swap_writes_flip;
        session.swap_writes_flip ^= 1;
        if (cudaEventSynchronize(session.swap_writes_event[k]) != cudaSuccess) {
            return false;
        }
        const int n = (int)std::min<size_t>(writes.size() - done, (size_t)session.swap_writes_cap);
        memcpy(session.swap_writes_host[k], writes.data() + done, (size_t)n * sizeof(moe_swap_write));
        moe_swap_apply<<<(n + 127) / 128, 128, 0, main>>>(session.swap_writes_host[k], n);
        if (cudaGetLastError() != cudaSuccess || cudaEventRecord(session.swap_writes_event[k], main) != cudaSuccess) {
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

static void moe_swap_push_address(std::vector<moe_swap_write> & writes, moe_cache_device_table & table, int32_t expert,
        const void * address) {
    writes.push_back({(unsigned long long)(uintptr_t)(table.device + expert), (unsigned long long)(uintptr_t)address, 8, 0});
    if (table.staging_device) {
        // With prefetch the kernels read the staging copy; it must never keep a victim's slot.
        writes.push_back({(unsigned long long)(uintptr_t)(table.staging_device + expert), (unsigned long long)(uintptr_t)address, 8, 0});
    }
    if (table.hot) {
        writes.push_back({(unsigned long long)(uintptr_t)(table.hot + expert),
                (unsigned long long)(address != table.fallback[expert]), 1, 0});
    }
}

// Mark landed fills valid. With publish=false (after a device sync, tables
// about to be rebuilt) only host state changes.
static void moe_cache_swap_commit(moe_cache_session & session, bool ok, bool publish) {
    if (session.swap_inflight.empty()) {
        return;
    }
    auto & device = *session.devices[0];
    std::vector<moe_swap_write> writes;
    // Reserve address, staging address and hot flag writes before slot changes.
    writes.reserve(session.swap_inflight.size() * 3);
    for (const auto & fill : session.swap_inflight) {
        auto & pool = *fill.pool;
        auto & slot = pool.slots[fill.index];
        if (slot.state != moe_cache_slot_state::copying) {
            continue;
        }
        if (ok) {
            slot.state = moe_cache_slot_state::valid;
            slot.generation++;
            moe_cache_lru_push_back(pool, fill.index);
            fill.table->addresses[fill.expert] = fill.destination;
            if (publish) {
                moe_swap_push_address(writes, *fill.table, fill.expert, fill.destination);
            }
            device.fills++;
            device.inserts++;
        } else {
            moe_cache_slot_reset(pool, fill.index, true);
            device.fill_failures++;
        }
    }
    session.swap_commits += session.swap_inflight.size();
    session.swap_inflight.clear();
    session.swap_inflight_bytes = 0;
    if (publish && !moe_cache_swap_launch(session, writes)) {
        MOE_CACHE_LOG("[moe-cache] swap publish failed; swaps stop, tables stay exact\n");
        (void) cudaGetLastError();
        session.swap_broken = true;
    }
}

// fork SWAPPOL: swap payback per (layer, expert), host only (GGML_MOE_SWAP_STATS). One projection (kind 0) stands
// for the expert; uses = routes counted for it between swap-in and eviction.
struct moe_swtrack {
    std::vector<long long> in_count;   // route count at swap-in; -1 when not resident through a tracked swap
    std::vector<unsigned long long> in_scope, swap_in, swap_out, uses_evicted;
    unsigned long long hist_uses[2][10] = {};   // [evicted, still resident at release] by use bucket
    unsigned long long hist_life[10] = {};      // evicted, by rounds resident
    unsigned long long evict_initial = 0, rounds = 0, updates = 0, slices = 0, bytes = 0;
};
static moe_swtrack g_swtrack;

// 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, 129+
static int moe_swtrack_bucket(unsigned long long v) {
    if (v <= 2) { return (int) v; }
    int b = 3;
    for (unsigned long long hi = 4; v > hi && b < 9; hi *= 2) { ++b; }
    return b;
}

static void moe_swtrack_event(const moe_cache_session & session, const moe_cache_device_table & table, int32_t e, bool in) {
    if (!g_swst || table.mj_kind != 0 || table.mj_layer < 0 || table.mj_layer >= MOE_SWST_LAYERS || e < 0 || e >= g_hf_experts) {
        return;
    }
    auto & t = g_swtrack;
    const size_t rows = (size_t) MOE_SWST_LAYERS * g_hf_experts;
    if (t.in_count.size() != rows) {
        t.in_count.assign(rows, -1);
        for (auto * v : {&t.in_scope, &t.swap_in, &t.swap_out, &t.uses_evicted}) { v->assign(rows, 0); }
    }
    const size_t i = (size_t) table.mj_layer * g_hf_experts + e;
    const long long now = session.swap_counts_host ? (long long) session.swap_counts_host[table.count_offset + e] : 0;
    if (in) {
        t.swap_in[i]++;
        t.in_count[i] = now;
        t.in_scope[i] = session.swap_scopes;
        return;
    }
    t.swap_out[i]++;
    if (t.in_count[i] < 0) {
        t.evict_initial++;
        return;
    }
    const unsigned long long uses = now > t.in_count[i] ? (unsigned long long) (now - t.in_count[i]) : 0;
    t.uses_evicted[i] += uses;
    t.hist_uses[0][moe_swtrack_bucket(uses)]++;
    t.hist_life[moe_swtrack_bucket(session.swap_scopes - t.in_scope[i])]++;
    t.in_count[i] = -1;
}

// Called at device release after its device sync: close the still-resident swaps, then rewrite the cumulative dump.
static void moe_swtrack_dump(const moe_cache_session & session) {
    const char * path = moe_swst_path();
    if (!path || !g_swst || g_hf_experts <= 0) {
        return;
    }
    auto & t = g_swtrack;
    const size_t rows = (size_t) MOE_SWST_LAYERS * g_hf_experts;
    if (t.in_count.size() == rows) {
        for (const auto & entry : session.device_tables) {
            const auto & table = entry.second;
            if (table.mj_kind != 0 || table.mj_layer < 0 || table.mj_layer >= MOE_SWST_LAYERS || table.desc.n_expert != g_hf_experts) { continue; }
            for (int32_t e = 0; e < table.desc.n_expert; ++e) {
                const size_t i = (size_t) table.mj_layer * g_hf_experts + e;
                if (t.in_count[i] < 0) { continue; }
                const long long now = session.swap_counts_host ? (long long) session.swap_counts_host[table.count_offset + e] : t.in_count[i];
                t.hist_uses[1][moe_swtrack_bucket(now > t.in_count[i] ? (unsigned long long) (now - t.in_count[i]) : 0)]++;
                t.in_count[i] = -1;
            }
        }
    }
    t.rounds += session.swap_scopes;
    t.updates += session.swap_updates;
    t.slices += session.swap_total;
    t.bytes += session.swap_bytes_total;
    std::vector<unsigned long long> st(rows * MOE_SWST_COLS);
    if (cudaMemcpy(st.data(), g_swst, st.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        MOE_CACHE_LOG("[moe-cache] swap stats: device read failed; no dump\n");
        return;
    }
    FILE * f = fopen(path, "w");
    if (!f) {
        MOE_CACHE_LOG("[moe-cache] swap stats: cannot write %s\n", path);
        return;
    }
    fprintf(f, "# swappol-stats v1 experts=%lld rounds=%llu updates=%llu slices=%llu bytes=%llu evict_initial=%llu score_mode=%d\n",
            (long long) g_hf_experts, t.rounds, t.updates, t.slices, t.bytes, t.evict_initial, moe_swap_score_mode());
    fprintf(f, "# use buckets: 0 1 2 3-4 5-8 9-16 17-32 33-64 65-128 129+\n");
    const char * names[3] = {"hist_uses_evicted", "hist_uses_resident_at_end", "hist_rounds_evicted"};
    for (int h = 0; h < 3; ++h) {
        fprintf(f, "%s", names[h]);
        for (int b = 0; b < 10; ++b) { fprintf(f, " %llu", h < 2 ? t.hist_uses[h][b] : t.hist_life[b]); }
        fprintf(f, "\n");
    }
    fprintf(f, "layer expert r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 weight_x10240 skipped hit windows swap_in swap_out uses_evicted\n");
    for (size_t i = 0; i < rows; ++i) {
        const unsigned long long * row = st.data() + i * MOE_SWST_COLS;
        const unsigned long long sin = t.swap_in.empty() ? 0 : t.swap_in[i], sout = t.swap_out.empty() ? 0 : t.swap_out[i];
        bool any = sin || sout;
        for (int c = 0; c < MOE_SWST_COLS; ++c) { any |= row[c] != 0; }
        if (!any) { continue; }
        fprintf(f, "%zu %zu", i / (size_t) g_hf_experts, i % (size_t) g_hf_experts);
        for (int c = 0; c < MOE_SWST_COLS; ++c) { fprintf(f, " %llu", row[c]); }
        fprintf(f, " %llu %llu %llu\n", sin, sout, t.uses_evicted.empty() ? 0ULL : t.uses_evicted[i]);
    }
    fclose(f);
    MOE_CACHE_LOG("[moe-cache] swap stats: wrote %s (rounds=%llu slices=%llu)\n", path, t.rounds, t.slices);
}

static std::vector<moe_swap_pair> moe_cache_swap_whole_pairs(moe_cache_session & session, size_t & n_cands) {
    struct entry { float score; size_t layer; int32_t expert; std::array<int, 3> slots; };
    struct batch { float gain; entry candidate; entry victim; bool free; };
    const auto & groups = session.swap_groups;
    std::vector<batch> batches;
    std::unordered_map<moe_cache_pool *, size_t> free_used;
    std::vector<bool> grouped(groups.size(), false);
    auto compatible = [&](size_t a, size_t b) {
        if (session.swap_layer && a != b) { return false; }
        for (int k = 0; k < 3; ++k) {
            const auto & x = groups[a][k]->desc;
            const auto & y = groups[b][k]->desc;
            if (x.type != y.type || x.expert_size != y.expert_size || x.n_in != y.n_in || x.n_out != y.n_out) { return false; }
        }
        return true;
    };
    auto order = [](const entry & a, const entry & b) {
        return a.layer != b.layer ? a.layer < b.layer : a.expert < b.expert;
    };
    for (size_t base = 0; base < groups.size(); ++base) {
        if (grouped[base] || !groups[base][0]) { continue; }
        std::vector<entry> candidates, victims;
        for (size_t layer = base; layer < groups.size(); ++layer) {
            if (!groups[layer][0] || !compatible(base, layer)) { continue; }
            grouped[layer] = true;
            const auto & group = groups[layer];
            for (int32_t e = 0; e < group[0]->desc.n_expert; ++e) {
                entry item{0, layer, e, {-1, -1, -1}};
                if (session.swap_cold_prune) {
                    for (int k = 0; k < 3; ++k) { item.score += session.swap_score[group[k]->count_offset + e]; }
                    item.score /= 3.0f;
                }
                int present = 0;
                bool usable = true;
                for (int k = 0; k < 3; ++k) {
                    auto & table = *group[k];
                    if (!session.swap_cold_prune) { item.score += session.swap_score[table.count_offset + e]; }
                    auto * pool = table.swap_pool;
                    if (!pool || table.permanent[e]) { usable = false; break; }
                    const auto found = pool->map.find({table.desc.data, e});
                    if (found != pool->map.end()) {
                        const auto & slot = pool->slots[found->second];
                        if (slot.state != moe_cache_slot_state::valid || slot.readers || slot.staging || !table.fallback[e]) { usable = false; break; }
                        item.slots[k] = found->second;
                        ++present;
                    } else if (session.swap_cold_prune && k == 0 && !(item.score >= 2.0f && item.score > session.swap_margin)) {
                        // fork: an absent first projection rules out a complete resident victim.
                        usable = false;
                        break;
                    }
                }
                if (!session.swap_cold_prune) { item.score /= 3.0f; }
                if (!usable) { continue; }
                if (present == 3) { victims.push_back(item); }
                else if (present == 0 && item.score >= 2.0f && item.score > session.swap_margin) { candidates.push_back(item); }
            }
        }
        n_cands += candidates.size();
        auto candidate_order = [&](const entry & a, const entry & b) { return a.score != b.score ? a.score > b.score : order(a, b); };
        auto victim_order = [&](const entry & a, const entry & b) { return a.score != b.score ? a.score < b.score : order(a, b); };
        if (session.swap_top_only) {
            // fork: one round admits at most this many complete experts per compatible group.
            const size_t take = (size_t) std::max(1, session.swap_max / 3);
            std::partial_sort(candidates.begin(), candidates.begin() + std::min(take, candidates.size()), candidates.end(), candidate_order);
            // fork: each candidate consumes at most one resident victim.
            const size_t vtake = std::min({take, candidates.size(), victims.size()});
            if (!candidates.empty()) { std::partial_sort(victims.begin(), victims.begin() + vtake, victims.end(), victim_order); }
        } else {
            std::sort(candidates.begin(), candidates.end(), candidate_order);
            std::sort(victims.begin(), victims.end(), victim_order);
        }
        size_t victim = 0, admitted = 0;
        for (const auto & candidate : candidates) {
            entry free_slots{0, candidate.layer, -1, {-1, -1, -1}};
            std::unordered_map<moe_cache_pool *, size_t> next_free;
            if (!session.swap_free_probe) { next_free = free_used; }
            bool free = true;
            for (int k = 0; k < 3; ++k) {
                auto * pool = groups[candidate.layer][k]->swap_pool;
                size_t used = 0;
                if (session.swap_free_probe) {
                    const auto found = free_used.find(pool);
                    if (found != free_used.end()) { used = found->second; }
                    // fork: earlier projections can reserve slots in the same pool.
                    for (int j = 0; j < k; ++j) { used += groups[candidate.layer][j]->swap_pool == pool; }
                } else {
                    used = next_free[pool]++;
                }
                if (used >= pool->free_slots.size()) { free = false; break; }
                free_slots.slots[k] = pool->free_slots[pool->free_slots.size() - 1 - used];
            }
            if (free) {
                if (session.swap_free_probe) {
                    for (int k = 0; k < 3; ++k) { ++free_used[groups[candidate.layer][k]->swap_pool]; }
                } else {
                    free_used = std::move(next_free);
                }
                batches.push_back({candidate.score, candidate, free_slots, true});
            } else {
                if (victim == victims.size() || candidate.score <= victims[victim].score * (1.0f + session.swap_hyst) + session.swap_margin) { break; }
                batches.push_back({candidate.score - victims[victim].score, candidate, victims[victim++], false});
            }
            if (++admitted >= (size_t) session.swap_max / 3) { break; }
        }
    }
    std::sort(batches.begin(), batches.end(), [&](const batch & a, const batch & b) { return a.gain != b.gain ? a.gain > b.gain : order(a.candidate, b.candidate); });
    std::vector<moe_swap_pair> result;
    if (session.swap_top_only) {
        result.reserve(batches.size() * 3);
    }
    for (const auto & batch : batches) {
        for (int k = 0; k < 3; ++k) {
            auto * table = groups[batch.candidate.layer][k];
            result.push_back({batch.gain, table->swap_pool, batch.victim.slots[k], batch.free, table, batch.candidate.expert});
        }
    }
    return result;
}

static std::vector<moe_swap_pair> moe_cache_swap_plan(moe_cache_session & session, size_t & n_cands) {
    auto & device = *session.devices[0];
    const size_t n = session.swap_prev.size();
    const unsigned long long * counts = session.swap_policy_counts.empty() ? session.swap_counts_host : session.swap_policy_counts.data();
    const float scale = session.swap_policy_counts.empty() ? 1.0f : session.swap_count_scale;
    const float decay = session.swap_decay_after ? 1.0f : session.swap_decay;
    if (session.swap_profile_strength > 0) {
        for (const auto & entry : session.device_tables) {
            const auto & table = entry.second;
            long double recent_total = 0;
            for (int32_t e = 0; e < table.desc.n_expert; ++e) {
                const size_t index = table.count_offset + e;
                recent_total += (long double) (counts[index] - session.swap_prev[index]) * scale;
            }
            auto & table_ema = session.swap_profile_table_ema[table.desc.data];
            const bool calibrated = session.swap_profile_calibrated_tables.count(table.desc.data) != 0;
            if (!calibrated && recent_total == 0) { continue; }
            // Counts and the prior use the same steady EMA units. Calibrate
            // causally; the profile's training corpus length is irrelevant.
            table_ema = calibrated ?
                decay * table_ema + (float) recent_total :
                (float) recent_total / (session.swap_decay < 1 ? 1 - session.swap_decay : 1);
            for (int32_t e = 0; e < table.desc.n_expert; ++e) {
                const size_t index = table.count_offset + e;
                const float recent = (float) (counts[index] - session.swap_prev[index]) * scale;
                session.swap_prev[index] = counts[index];
                session.swap_observed_score[index] = session.swap_observed_score[index] * decay + recent;
                float prior = session.swap_profile_rate[index] * table_ema * session.swap_profile_strength;
                if (session.swap_profile_decay) {
                    session.swap_profile_seed[index] = calibrated ?
                        session.swap_profile_seed[index] * decay : prior;
                    prior = session.swap_profile_seed[index];
                }
                session.swap_score[index] = session.swap_observed_score[index] + prior;
            }
            session.swap_profile_calibrated_tables.insert(table.desc.data);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            const unsigned long long recent = counts[i] - session.swap_prev[i];
            session.swap_prev[i] = counts[i];
            session.swap_score[i] = session.swap_score[i] * decay + (float)recent * scale;
        }
    }
    struct cand { float score; moe_cache_device_table * table; int32_t expert; };
    struct vict { float score; int slot; };
    using pair = moe_swap_pair;
    std::unordered_map<moe_cache_pool *, std::vector<cand>> cands;
    n_cands = 0;
    if (!session.swap_whole) {
        for (auto & entry : session.device_tables) {
            auto & table = entry.second;
            if (!table.swap_pool) {
                for (auto & p : device.pools) {
                    if (p->slab && p->expert_size == table.desc.expert_size && p->wtype == table.desc.type) {
                        table.swap_pool = p.get();
                        break;
                    }
                }
            }
            if (!table.swap_pool) {
                continue;
            }
            auto & pool = *table.swap_pool;
            auto & list = cands[&pool];
            for (int32_t e = 0; e < (int32_t)table.desc.n_expert; ++e) {
                const float score = session.swap_score[table.count_offset + e];
                if (score <= session.swap_margin || table.permanent[e] || pool.map.count({table.desc.data, e})) {
                    continue;
                }
                list.push_back({score, &table, e});
                ++n_cands;
            }
        }
    }
    std::vector<pair> pairs;
    const size_t limit = (size_t)session.swap_max;
    for (auto & item : cands) {
        auto & pool = *item.first;
        auto & list = item.second;
        if (list.empty()) {
            continue;
        }
        const size_t take = std::min(list.size(), limit);
        std::partial_sort(list.begin(), list.begin() + take, list.end(),
                [](const cand & a, const cand & b) { return a.score > b.score; });
        std::vector<vict> victims;
        victims.reserve(pool.n_slots);
        for (int i = 0; i < pool.n_slots; ++i) {
            const auto & slot = pool.slots[i];
            if (slot.state != moe_cache_slot_state::valid || slot.readers || slot.staging) {
                continue;
            }
            const auto found = session.device_tables.find(slot.key.tensor);
            if (found == session.device_tables.end()) {
                continue;
            }
            victims.push_back({session.swap_score[found->second.count_offset + slot.key.expert], i});
        }
        const size_t vtake = std::min(victims.size(), take);
        std::partial_sort(victims.begin(), victims.begin() + vtake, victims.end(),
                [](const vict & a, const vict & b) { return a.score < b.score; });
        size_t free_used = 0, v = 0;
        for (size_t c = 0; c < take; ++c) {
            if (free_used < pool.free_slots.size()) {
                const int slot = pool.free_slots[pool.free_slots.size() - 1 - free_used++];
                pairs.push_back({list[c].score, &pool, slot, true, list[c].table, list[c].expert});
                continue;
            }
            if (v >= vtake || list[c].score <= victims[v].score * (1.0f + session.swap_hyst) + session.swap_margin) {
                break;
            }
            pairs.push_back({list[c].score - victims[v].score, &pool, victims[v].slot, false, list[c].table, list[c].expert});
            ++v;
        }
    }
    if (session.swap_whole) {
        pairs = moe_cache_swap_whole_pairs(session, n_cands);
    } else {
        std::stable_sort(pairs.begin(), pairs.end(), [](const pair & a, const pair & b) { return a.gain > b.gain; });
    }
    if (session.swap_decay_after) {
        for (auto & score : session.swap_score) { score *= session.swap_decay; }
        for (auto & score : session.swap_observed_score) { score *= session.swap_decay; }
        for (auto & score : session.swap_profile_seed) { score *= session.swap_decay; }
        for (auto & entry : session.swap_profile_table_ema) { entry.second *= session.swap_decay; }
    }
    return pairs;
}

static void moe_cache_swap_apply_decision(moe_cache_session & session, const std::vector<moe_swap_pair> & pairs,
        size_t n_cands, double plan_ms) {
    auto & device = *session.devices[0];
    const auto start = std::chrono::steady_clock::now();
    const size_t limit = (size_t) session.swap_max;
    std::vector<moe_swap_write> writes;
    writes.reserve(limit * 3);
    session.swap_inflight.reserve(limit);
    size_t bytes = 0;
    int evictions = 0;
    const size_t step = session.swap_whole ? 3 : 1;
    for (size_t index = 0; index < pairs.size(); index += step) {
        if (session.swap_whole) {
            if (limit - session.swap_inflight.size() < 3) { break; }
            size_t batch_bytes = 0;
            bool valid = true;
            for (size_t k = 0; k < 3; ++k) {
                const auto & p = pairs[index + k];
                auto & pool = *p.pool;
                const auto & slot = pool.slots[p.slot];
                if (pool.expert_size > session.swap_cap_bytes - bytes - batch_bytes) { valid = false; break; }
                batch_bytes += pool.expert_size;
                if (slot.staging || slot.readers || pool.map.count({p.table->desc.data, p.expert})) { valid = false; break; }
                if (p.free) {
                    if (slot.state != moe_cache_slot_state::free || std::find(pool.free_slots.begin(), pool.free_slots.end(), p.slot) == pool.free_slots.end()) { valid = false; break; }
                } else {
                    const auto found = session.device_tables.find(slot.key.tensor);
                    if (slot.state != moe_cache_slot_state::valid || found == session.device_tables.end() ||
                            found->second.permanent[slot.key.expert] || !found->second.fallback[slot.key.expert]) { valid = false; break; }
                }
            }
            if (!valid) { continue; }
            size_t inserted = 0;
            try {
                for (; inserted < 3; ++inserted) {
                    const auto & p = pairs[index + inserted];
                    if (!p.pool->map.emplace(moe_cache_key{p.table->desc.data, p.expert}, p.slot).second) { break; }
                }
            } catch (const std::bad_alloc &) {
                // Roll back reservations before any resident slot or table changes.
            }
            if (inserted != 3) {
                for (size_t k = 0; k < inserted; ++k) {
                    const auto & p = pairs[index + k];
                    p.pool->map.erase({p.table->desc.data, p.expert});
                }
                continue;
            }
        }
        for (size_t k = 0; k < step; ++k) {
            const auto & p = pairs[index + k];
            if (session.swap_inflight.size() >= limit) {
                break;
            }
            auto & pool = *p.pool;
            if (pool.expert_size > session.swap_cap_bytes - bytes) {
                continue;
            }
            auto & slot = pool.slots[p.slot];
            if (p.free) {
                const auto it = std::find(pool.free_slots.begin(), pool.free_slots.end(), p.slot);
                if (it == pool.free_slots.end() || slot.state != moe_cache_slot_state::free || slot.staging) {
                    continue;
                }
                pool.free_slots.erase(it);
            } else {
                auto & old_table = session.device_tables.at(slot.key.tensor);
                const int32_t old_expert = slot.key.expert;
                if (!old_table.fallback[old_expert]) {
                    continue; // keep a resident with no exact host fallback
                }
                old_table.addresses[old_expert] = old_table.fallback[old_expert];
                moe_swtrack_event(session, old_table, old_expert, false);
                moe_swap_push_address(writes, old_table, old_expert, old_table.fallback[old_expert]);
                moe_cache_slot_reset(pool, p.slot, false);
                device.evictions++;
                ++evictions;
            }
            const moe_cache_key key{p.table->desc.data, p.expert};
            slot.key = key;
            slot.state = moe_cache_slot_state::copying;
            slot.uses = (uint32_t)std::min(p.gain, 4.0e9f);
            pool.map[key] = p.slot;
            moe_swtrack_event(session, *p.table, p.expert, true);
            session.swap_inflight.push_back({&pool, p.slot, p.table, p.expert, pool.slab + (size_t)p.slot * pool.expert_size});
            bytes += pool.expert_size;
        }
    }
    const double host_ms = plan_ms + std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    session.swap_updates++;
    session.swap_host_ms_total += host_ms;
    if (session.swap_inflight.empty()) {
        if (session.swap_updates <= 4 || session.swap_updates % 64 == 0) {
            MOE_CACHE_LOG("[moe-cache] swap update: scope=%llu swaps=0 candidates=%zu pairs=%zu bytes=0 host_ms=%.3f\n",
                    (unsigned long long)session.swap_scopes, n_cands, pairs.size(), host_ms);
        }
        return;
    }
    // Victims leave the tables first; the copy stream waits for that publication
    // on every compute stream, and later work on the other streams waits too.
    cudaStream_t main = session.swap_streams[0];
    bool ok = moe_cache_swap_launch(session, writes) &&
        cudaEventRecord(session.swap_pub_event, main) == cudaSuccess &&
        cudaStreamWaitEvent(session.swap_copy_stream, session.swap_pub_event, 0) == cudaSuccess;
    const int n_streams = session.swap_n_streams.load(std::memory_order_acquire);
    for (int i = 1; ok && i < n_streams; ++i) {
        ok = cudaEventRecord(session.swap_join_event[i], session.swap_streams[i]) == cudaSuccess &&
            cudaStreamWaitEvent(session.swap_copy_stream, session.swap_join_event[i], 0) == cudaSuccess &&
            cudaStreamWaitEvent(session.swap_streams[i], session.swap_pub_event, 0) == cudaSuccess;
    }
    for (const auto & fill : session.swap_inflight) {
        if (!ok) {
            break;
        }
        const void * source = fill.table->copy_source[fill.expert];
        if (!source) {
            source = (const char *)fill.table->desc.data + (size_t)fill.expert * fill.table->host_stride;
        }
        ok = moe_h2d_async(fill.destination, source, fill.pool->expert_size, cudaMemcpyHostToDevice,
                session.swap_copy_stream) == cudaSuccess;
    }
    ok = ok && cudaEventRecord(session.swap_copy_event, session.swap_copy_stream) == cudaSuccess;
    session.swap_inflight_bytes = bytes;
    session.swap_issue_time = std::chrono::steady_clock::now();
    session.swap_total += session.swap_inflight.size();
    session.swap_bytes_total += bytes;
    MOE_CACHE_LOG("[moe-cache] swap update: scope=%llu swaps=%zu evictions=%d candidates=%zu pairs=%zu bytes=%zu host_ms=%.3f\n",
            (unsigned long long)session.swap_scopes, session.swap_inflight.size(), evictions, n_cands, pairs.size(), bytes, host_ms);
    if (!ok) {
        MOE_CACHE_LOG("[moe-cache] swap issue failed: %s; draining and stopping swaps\n", cudaGetErrorString(cudaGetLastError()));
        (void) cudaDeviceSynchronize();
        (void) cudaGetLastError();
        moe_cache_swap_commit(session, false, false);
        session.swap_broken = true;
    }
}

// The worker owns this host snapshot; it never reads live slots or calls CUDA.
static std::unique_ptr<moe_cache_session> moe_cache_swap_snapshot(const moe_cache_session & session,
        std::unique_ptr<moe_cache_session> copy = {}) {
    if (!copy) { copy.reset(new moe_cache_session); }
    copy->swap_whole = session.swap_whole;
    copy->swap_layer = session.swap_layer;
    copy->swap_decay_after = session.swap_decay_after;
    copy->swap_max = session.swap_max;
    copy->swap_decay = session.swap_decay;
    copy->swap_hyst = session.swap_hyst;
    copy->swap_margin = session.swap_margin;
    copy->swap_profile_strength = session.swap_profile_strength;
    copy->swap_profile_decay = session.swap_profile_decay;
    copy->swap_prev = session.swap_prev;
    copy->swap_score = session.swap_score;
    copy->swap_observed_score = session.swap_observed_score;
    copy->swap_profile_rate = session.swap_profile_rate;
    copy->swap_profile_seed = session.swap_profile_seed;
    copy->swap_profile_table_ema = session.swap_profile_table_ema;
    copy->swap_profile_calibrated_tables = session.swap_profile_calibrated_tables;
    copy->swap_policy_counts = session.swap_policy_counts;
    copy->swap_count_scale = session.swap_count_scale;
    copy->device_previous_counts.assign(session.swap_counts_host, session.swap_counts_host + session.swap_prev.size());
    copy->swap_counts_host = copy->device_previous_counts.data();
    if (copy->devices.empty()) { copy->devices.emplace_back(new moe_cache_device(0, 0)); }
    auto & cloned = copy->devices[0]->pools;
    const auto & source = session.devices[0]->pools;
    while (cloned.size() < source.size()) { cloned.emplace_back(new moe_cache_pool); }
    cloned.resize(source.size());
    std::unordered_map<const moe_cache_pool *, moe_cache_pool *> pools;
    for (size_t i = 0; i < source.size(); ++i) {
        // The planner snapshot never owns card memory: copy the slot table, leave prompt_slab empty.
        moe_cache_pool & dst = *cloned[i];
        const moe_cache_pool & src = *source[i];
        dst.expert_size = src.expert_size;
        dst.wtype = src.wtype;
        dst.slab = src.slab;
        dst.prompt_slab.reset();
        dst.prompt_slots = src.prompt_slots;
        dst.n_slots = src.n_slots;
        dst.covers_all_entries = src.covers_all_entries;
        dst.slots = src.slots;
        dst.free_slots = src.free_slots;
        dst.map = src.map;
        dst.lru_head = src.lru_head;
        dst.lru_tail = src.lru_tail;
        pools[source[i].get()] = cloned[i].get();
    }
    copy->device_tables = session.device_tables;
    for (auto & entry : copy->device_tables) {
        if (entry.second.swap_pool) { entry.second.swap_pool = pools.at(entry.second.swap_pool); }
    }
    copy->swap_groups = session.swap_groups;
    for (auto & group : copy->swap_groups) {
        for (auto & table : group) {
            if (table) { table = &copy->device_tables.at(table->desc.data); }
        }
    }
    return copy;
}

static moe_swap_decision moe_cache_swap_run_plan(moe_cache_session & session) {
    const auto start = std::chrono::steady_clock::now();
    moe_swap_decision result;
    result.pairs = moe_cache_swap_plan(session, result.candidates);
    result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}

static bool moe_cache_swap_take_plan(moe_cache_session & session, moe_swap_decision & result) {
    auto & copy = *session.swap_snapshot;
    bool valid = true;
    for (auto & pair : result.pairs) {
        auto * saved_pool = pair.pool;
        const auto & saved_slot = saved_pool->slots[pair.slot];
        pair.table = &session.device_tables.at(pair.table->desc.data);
        size_t pool = 0;
        while (copy.devices[0]->pools.at(pool).get() != saved_pool) { ++pool; }
        pair.pool = session.devices[0]->pools.at(pool).get();
        const auto & slot = pair.pool->slots.at(pair.slot);
        valid &= slot.generation == saved_slot.generation && slot.key == saved_slot.key &&
            slot.state == saved_slot.state && !slot.readers && !slot.staging &&
            !pair.pool->map.count({pair.table->desc.data, pair.expert});
    }
    session.swap_prev.swap(copy.swap_prev);
    session.swap_score.swap(copy.swap_score);
    session.swap_observed_score.swap(copy.swap_observed_score);
    session.swap_profile_seed.swap(copy.swap_profile_seed);
    session.swap_profile_table_ema.swap(copy.swap_profile_table_ema);
    session.swap_profile_calibrated_tables.swap(copy.swap_profile_calibrated_tables);
    session.swap_async_ms_total += result.ms;
    if (!valid) { result.pairs.clear(); }
    return valid;
}

static void moe_cache_swap_finish_plan(moe_cache_session & session) {
    auto result = session.swap_decision.get();
    moe_cache_swap_take_plan(session, result);
    moe_cache_swap_apply_decision(session, result.pairs, result.candidates, 0);
}

// Runs the real planner and snapshot path without a CUDA context or model weights.
bool ggml_cuda_moe_cache_swap_async_test() {
    bool ok = true;
    int keys[3] = {};
    for (bool whole : {false, true}) {
        for (bool prior : {false, true}) {
            auto check = [&](bool pass, const char * step, int round = -1) {
                if (!pass) {
                    fprintf(stderr, "cache-swap-async whole=%d prior=%d round=%d: %s failed\n", whole, prior, round, step);
                    ok = false;
                }
            };
            moe_cache_session seed;
            seed.devices.emplace_back(new moe_cache_device(0, 0));
            seed.devices[0]->pools.emplace_back(new moe_cache_pool);
            auto & pool = *seed.devices[0]->pools[0];
            pool.expert_size = 64; pool.wtype = GGML_TYPE_Q4_0; pool.n_slots = 9;
            pool.slots.resize(9); pool.free_slots = {6, 7, 8};
            seed.swap_whole = whole; seed.swap_max = 6; seed.swap_margin = 0;
            seed.swap_decay_after = prior; seed.swap_profile_strength = prior ? 1 : 0;
            seed.swap_profile_decay = prior;
            seed.swap_prev.assign(18, 0); seed.swap_score.assign(18, 0);
            seed.swap_observed_score.assign(18, 0); seed.swap_profile_seed.assign(18, 0);
            seed.swap_profile_rate.assign(18, 1.0f / 6);
            seed.device_previous_counts.assign(18, 0);
            seed.swap_counts_host = seed.device_previous_counts.data();
            std::array<moe_cache_device_table *, 3> group;
            for (int k = 0; k < 3; ++k) {
                auto & table = seed.device_tables[&keys[k]];
                table.desc.data = &keys[k]; table.desc.n_expert = 6;
                table.desc.expert_size = 64; table.desc.type = GGML_TYPE_Q4_0;
                table.desc.n_in = 16; table.desc.n_out = 8;
                table.count_offset = k * 6; table.swap_pool = &pool;
                table.permanent.assign(6, 0); table.permanent[5] = 1; table.fallback.assign(6, &keys[k]);
                group[k] = &table;
                for (int e = 0; e < 2; ++e) {
                    const int index = k * 2 + e;
                    pool.slots[index].key = {&keys[k], e};
                    pool.slots[index].state = moe_cache_slot_state::valid;
                    pool.map[pool.slots[index].key] = index;
                }
            }
            seed.swap_groups.push_back(group);
            auto sync = moe_cache_swap_snapshot(seed);
            auto async = moe_cache_swap_snapshot(seed);
            size_t swaps = 0, free_swaps = 0, evictions = 0;
            auto apply = [](const moe_swap_decision & result) {
                for (const auto & p : result.pairs) {
                    auto & slot = p.pool->slots[p.slot];
                    if (p.free) {
                        auto & free = p.pool->free_slots;
                        free.erase(std::find(free.begin(), free.end(), p.slot));
                    } else { p.pool->map.erase(slot.key); }
                    slot.key = {p.table->desc.data, p.expert};
                    slot.state = moe_cache_slot_state::valid; ++slot.generation;
                    p.pool->map[slot.key] = p.slot;
                }
            };
            for (int round = 0; round < 16; ++round) {
                for (int k = 0; k < 3; ++k) {
                    const size_t index = k * 6 + (round + 2) % 6;
                    sync->device_previous_counts[index] += 100 + round;
                    async->device_previous_counts[index] += 100 + round;
                }
                auto expected = moe_cache_swap_run_plan(*sync);
                async->swap_snapshot = moe_cache_swap_snapshot(*async, std::move(async->swap_snapshot));
                auto * copy = async->swap_snapshot.get();
                std::promise<void> begin;
                auto ready = begin.get_future();
                async->swap_decision = std::async(std::launch::async, [copy, ready = std::move(ready)]() mutable {
                    ready.wait(); return moe_cache_swap_run_plan(*copy);
                });
                ok &= async->swap_decision.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
                async->device_previous_counts[0] += 17;
                begin.set_value();
                // A later scope can change live counts; the captured inputs stay fixed.
                auto actual = async->swap_decision.get();
                async->device_previous_counts[0] -= 17;
                check(moe_cache_swap_take_plan(*async, actual), "take plan", round);
                check(expected.pairs.size() == actual.pairs.size() && expected.candidates == actual.candidates, "pair count", round);
                check(sync->swap_prev == async->swap_prev && sync->swap_score == async->swap_score, "scores", round);
                check(sync->swap_profile_seed == async->swap_profile_seed && sync->swap_profile_table_ema == async->swap_profile_table_ema, "prior", round);
                for (size_t i = 0; i < std::min(expected.pairs.size(), actual.pairs.size()); ++i) {
                    const auto & a = expected.pairs[i]; const auto & b = actual.pairs[i];
                    check(a.table->desc.data == b.table->desc.data && a.expert == b.expert &&
                        a.slot == b.slot && a.free == b.free && a.gain == b.gain, "pair", round);
                }
                swaps += expected.pairs.size();
                for (const auto & p : actual.pairs) { p.free ? ++free_swaps : ++evictions; }
                apply(expected); apply(actual);
            }
            check(swaps > 0 && free_swaps > 0 && evictions > 0, "free and victim coverage");
            // Pick a missing expert; a large count on a resident cannot produce a stale pair.
            int missing = 0;
            while (missing < 5 && async->devices[0]->pools[0]->map.count({&keys[0], missing})) { ++missing; }
            check(missing < 5, "missing expert");
            if (missing == 5) { continue; }
            for (int k = 0; k < (whole ? 3 : 1); ++k) {
                check(!async->devices[0]->pools[0]->map.count({&keys[k], missing}), "missing projection");
                async->device_previous_counts[k * 6 + missing] += 10000;
            }
            async->swap_snapshot = moe_cache_swap_snapshot(*async, std::move(async->swap_snapshot));
            auto stale = moe_cache_swap_run_plan(*async->swap_snapshot);
            check(!stale.pairs.empty(), "stale pair coverage");
            for (auto & slot : async->devices[0]->pools[0]->slots) { ++slot.generation; }
            async->device_previous_counts[missing] += 10000;
            check(!moe_cache_swap_take_plan(*async, stale) && stale.pairs.empty(), "stale rejection");
            check(async->swap_prev[missing] + 10000 == async->device_previous_counts[missing], "stale counts");
        }
    }
    return ok;
}

// Called with mu held after the last scope left; never waits for the card.
// fork SWAPPOL: map the per (layer, expert) score readback onto count indices for the planner.
static void moe_swap_fill_policy_counts(moe_cache_session & session) {
    if (!g_swscore_host || g_hf_experts <= 0) {
        session.swap_policy_counts.clear();
        return;
    }
    session.swap_policy_counts.assign(session.swap_prev.size(), 0);
    session.swap_count_scale = 1.0f / 1024.0f;
    for (const auto & entry : session.device_tables) {
        const auto & table = entry.second;
        if (table.mj_layer < 0 || table.mj_layer >= MOE_SWST_LAYERS || table.desc.n_expert != g_hf_experts ||
                table.count_offset + table.desc.n_expert > session.swap_policy_counts.size()) {
            continue;
        }
        for (int32_t e = 0; e < table.desc.n_expert; ++e) {
            session.swap_policy_counts[table.count_offset + e] = g_swscore_host[(size_t) table.mj_layer * g_hf_experts + e];
        }
    }
}

static void moe_cache_swap_step(moe_cache_session & session) {
    if (!session.device_expert_counts || session.devices.size() != 1 || session.swap_broken ||
            session.swap_n_streams.load(std::memory_order_acquire) == 0) {
        return;
    }
    if (session.swap_stream_overflow.load()) {
        if (!session.swap_broken) {
            MOE_CACHE_LOG("[moe-cache] swap policy stopped: more than 4 compute streams\n");
        }
        session.swap_broken = true;
        return;
    }
    auto & device = *session.devices[0];
    ggml_cuda_set_device(device.logical);
    if (!moe_cache_swap_init(session)) {
        session.swap_broken = true;
        return;
    }
    const uint64_t scope = ++session.swap_scopes;
    // 1) Publish fills whose bytes landed (normally during the graph just enqueued).
    if (!session.swap_inflight.empty()) {
        const cudaError_t q = cudaEventQuery(session.swap_copy_event);
        if (q == cudaSuccess) {
            session.swap_land_ms_total += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - session.swap_issue_time).count();
            moe_cache_swap_commit(session, true, true);
        } else if (q != cudaErrorNotReady) {
            MOE_CACHE_LOG("[moe-cache] swap copy failed: %s; draining and stopping swaps\n", cudaGetErrorString(q));
            (void) cudaDeviceSynchronize();
            (void) cudaGetLastError();
            moe_cache_swap_commit(session, false, false);
            session.swap_broken = true;
            return;
        }
    }
    if (session.swap_broken) {
        return;
    }
    // A ready plan is applied one scope later; an unfinished worker never stalls decode.
    if (session.swap_decision.valid() && session.swap_decision.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        try {
            moe_cache_swap_finish_plan(session);
        } catch (const std::exception & error) {
            MOE_CACHE_LOG("[moe-cache] async swap plan failed: %s; stopping swaps\n", error.what());
            session.swap_snapshot.reset();
            session.swap_broken = true;
            return;
        }
    }
    if (session.swap_broken) {
        return;
    }
    // 2) Decide from the counters read back at an earlier scope.
    if (session.swap_counts_pending && session.swap_inflight.empty() && !session.swap_decision.valid()) {
        const cudaError_t q = cudaEventQuery(session.swap_counts_event);
        if (q != cudaErrorNotReady) {
            session.swap_counts_pending = false;
            if (q == cudaSuccess) {
                moe_swap_fill_policy_counts(session);
                static const bool async = getenv("GGML_MOE_SWAP_ASYNC") && atoi(getenv("GGML_MOE_SWAP_ASYNC")) != 0;
                if (async) {
                    try {
                        const auto start = std::chrono::steady_clock::now();
                        session.swap_snapshot = moe_cache_swap_snapshot(session, std::move(session.swap_snapshot));
                        auto * copy = session.swap_snapshot.get();
                        session.swap_decision = std::async(std::launch::async, [copy] { return moe_cache_swap_run_plan(*copy); });
                        session.swap_host_ms_total += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    } catch (const std::exception & error) {
                        MOE_CACHE_LOG("[moe-cache] async swap launch failed: %s; stopping swaps\n", error.what());
                        session.swap_snapshot.reset();
                        session.swap_broken = true;
                        return;
                    }
                } else {
                    auto result = moe_cache_swap_run_plan(session);
                    moe_cache_swap_apply_decision(session, result.pairs, result.candidates, result.ms);
                }
            } else {
                (void) cudaGetLastError();
            }
        }
    }
    if (session.swap_broken) {
        return;
    }
    // 3) Read back this scope's counters for the next decision (async, compute stream).
    if (!session.swap_counts_pending && scope % (uint64_t)session.swap_interval == 0) {
        cudaStream_t main = session.swap_streams[0];
        if (cudaMemcpyAsync(session.swap_counts_host, session.device_expert_counts,
                    session.swap_prev.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost, main) == cudaSuccess &&
                (!g_swscore_host || cudaMemcpyAsync(g_swscore_host, g_swscore, (size_t) MOE_SWST_LAYERS * g_hf_experts *
                    sizeof(unsigned long long), cudaMemcpyDeviceToHost, main) == cudaSuccess) &&
                cudaEventRecord(session.swap_counts_event, main) == cudaSuccess) {
            session.swap_counts_pending = true;
        } else {
            (void) cudaGetLastError();
        }
    }
}

// Called by device release after its full device sync.
static void moe_cache_swap_release(moe_cache_session & session) {
    if (session.swap_decision.valid()) {
        session.swap_decision.wait();
        session.swap_decision = {};
    }
    session.swap_snapshot.reset();
    moe_swtrack_dump(session);
    if (session.swap_updates || session.swap_commits) {
        MOE_CACHE_LOG("[moe-cache] swap totals: updates=%llu swaps=%llu bytes=%llu commits=%llu host_ms_total=%.3f land_ms_total=%.3f scopes=%llu async_plan_ms_total=%.3f\n",
                (unsigned long long)session.swap_updates, (unsigned long long)session.swap_total,
                (unsigned long long)session.swap_bytes_total, (unsigned long long)session.swap_commits,
                session.swap_host_ms_total, session.swap_land_ms_total, (unsigned long long)session.swap_scopes, session.swap_async_ms_total);
    }
    moe_cache_swap_commit(session, true, false);
    moe_cache_swap_free(session);
    session.swap_groups.clear();
    session.swap_prev.clear();
    session.swap_score.clear();
    session.swap_observed_score.clear();
    session.swap_profile_rate.clear();
    session.swap_profile_seed.clear();
    session.swap_profile_table_ema.clear();
    session.swap_profile_calibrated_tables.clear();
    session.swap_profile_strength = 0;
    session.swap_scopes = 0;
    session.swap_broken = false;
}

static void moe_table_prefetch_release(moe_cache_session & session) {
#if !defined(GGML_USE_HIP)
    for (auto & device : session.devices) {
        if (!device->table_prefetch) continue;
        auto & state = *device->table_prefetch;
        if (!state.copy_stream) { device->table_prefetch_prepared = false; continue; }
        ggml_cuda_set_device(device->logical);
        // This also covers partial preparation before tables were published.
        CUDA_CHECK(cudaDeviceSynchronize());
        if (device->table_prefetch_prepared) {
            std::vector<moe_tp::Stats> counts(state.model.layers.size());
            CUDA_CHECK(cudaMemcpy(counts.data(), state.stats, counts.size() * sizeof(counts[0]), cudaMemcpyDeviceToHost));
            moe_tp::word progress = 0;
            CUDA_CHECK(cudaMemcpy(&progress, state.progress, sizeof(progress), cudaMemcpyDeviceToHost));
            moe_tp::Stats total{};
            for (size_t layer = 0; layer < counts.size(); ++layer) {
                const auto & c = counts[layer];
                for (int k = 0; k < 3; ++k) {
                    total.useful_bytes[k] += c.useful_bytes[k]; total.copied_bytes[k] += c.copied_bytes[k];
                }
                total.observed_kernel_span_ns += c.observed_kernel_span_ns; total.copy_launches += c.copy_launches;
                total.served_routes += c.served_routes;
                MOE_CACHE_LOG("[moe-table-prefetch] layer=%zu useful_bytes=%llu,%llu,%llu copied_bytes=%llu,%llu,%llu observed_kernel_span_ns=%llu copy_launches=%llu served_routes=%llu\n",
                    layer, c.useful_bytes[0], c.useful_bytes[1], c.useful_bytes[2], c.copied_bytes[0], c.copied_bytes[1], c.copied_bytes[2],
                    c.observed_kernel_span_ns, c.copy_launches, c.served_routes);
            }
            const auto useful = total.useful_bytes[0] + total.useful_bytes[1] + total.useful_bytes[2];
            const auto copied = total.copied_bytes[0] + total.copied_bytes[1] + total.copied_bytes[2];
            MOE_CACHE_LOG("[moe-table-prefetch] total: rounds=%u useful_bytes=%llu copied_bytes=%llu useful_pct=%.6f observed_kernel_span_ns=%llu copy_launches=%llu served_routes=%llu; span excludes FIFO waits and is not CE busy\n",
                unsigned(progress >> 32), useful, copied, copied ? 100.0 * double(useful) / double(copied) : 0.0,
                total.observed_kernel_span_ns, total.copy_launches, total.served_routes);
        }
        state.release_streams(); device->table_prefetch_prepared = false;
    }
#else
    GGML_UNUSED(session);
#endif
}

static void moe_table_prefetch_prepare(moe_cache_session & session) {
#if !defined(GGML_USE_HIP)
    auto & device = *session.devices[0];
    if (!device.table_prefetch) return;
    auto & state = *device.table_prefetch;
    std::vector<std::array<moe_tp::Input, 3>> inputs(state.model.layers.size());
    for (auto & entry : session.device_tables) {
        auto & table = entry.second;
        int layer = -1, parsed = 0;
        char kind[16] = {};
        if (!table.desc.name || sscanf(table.desc.name, "blk.%d.ffn_%15[a-z]_exps%n", &layer, kind, &parsed) != 2 ||
                parsed <= 0 || (table.desc.name[parsed] && strcmp(table.desc.name + parsed, ".weight") != 0) ||
                layer < 0 || size_t(layer) >= inputs.size()) {
            GGML_ABORT("table prefetch requires complete named expert layers");
        }
        const int k = strcmp(kind, "up") == 0 ? 0 : strcmp(kind, "gate") == 0 ? 1 : strcmp(kind, "down") == 0 ? 2 : -1;
        if (k < 0 || table.desc.n_expert != 512 || !table.device || !table.hot || inputs[layer][k].table) {
            GGML_ABORT("table prefetch expert geometry mismatch");
        }
        const size_t bytes = (size_t) table.desc.n_out * ggml_row_size((ggml_type) table.desc.type, table.desc.n_in);
        if (!bytes || bytes > table.desc.expert_size) GGML_ABORT("table prefetch compact expert exceeds source stride");
        table.tp_layer = layer; table.tp_kind = k;
        inputs[layer][k] = {table.device, table.hot, bytes};
    }
    try {
        state.prepare(device.table_prefetch_workspace, device.table_prefetch_capacity, inputs);
    } catch (const std::exception & error) {
        GGML_ABORT("table prefetch preparation failed: %s", error.what());
    }
    device.table_prefetch_prepared = true;
    char digest[65];
    for (int i = 0; i < 32; ++i) snprintf(digest + 2 * i, 3, "%02x", state.model.payload_sha256[i]);
    MOE_CACHE_LOG("[moe-table-prefetch] active: mode=%d lead=%u layers=%zu payload_sha256=%s; device-only lookup, FIFO pull, no target waits\n",
        state.mode, state.model.lead, inputs.size(), digest);
#else
    GGML_UNUSED(session);
#endif
}

void ggml_cuda_moe_cache_table_graph_begin(ggml_backend_cuda_context & ctx, bool allow_single_stream) {
#if !defined(GGML_USE_HIP)
    if (g_session_suppressed || g_session_stack.empty() || !g_session_stack.back().active) return;
    auto & session = *g_session_stack.back().active;
    if (!session.device_ready.load() || session.devices.size() != 1 || session.devices[0]->logical != ctx.device) return;
    auto & device = *session.devices[0];
    if (device.table_prefetch_prepared) device.table_prefetch->graph_begin(&ctx, ctx.stream(), allow_single_stream);
#else
    GGML_UNUSED(ctx); GGML_UNUSED(allow_single_stream);
#endif
}

void ggml_cuda_moe_cache_table_graph_end(ggml_backend_cuda_context & ctx) {
#if !defined(GGML_USE_HIP)
    if (g_session_suppressed || g_session_stack.empty() || !g_session_stack.back().active) return;
    auto & session = *g_session_stack.back().active;
    if (!session.device_ready.load() || session.devices.size() != 1 || session.devices[0]->logical != ctx.device) return;
    auto & device = *session.devices[0];
    if (device.table_prefetch_prepared) device.table_prefetch->graph_end(&ctx);
#else
    GGML_UNUSED(ctx);
#endif
}

static void moe_cache_device_release(moe_cache_session & session, bool retain_sources) {
    if (session.mass_trace) {
        ggml_cuda_set_device(session.devices[0]->logical);
        unsigned counts[2];
        CUDA_CHECK(cudaMemcpy(counts, session.mass_trace_counts, sizeof(counts), cudaMemcpyDeviceToHost));
        const size_t bytes = (size_t) std::min(counts[0], 16384u) * 736;
        std::vector<char> rows(bytes);
        CUDA_CHECK(cudaMemcpy(rows.data(), session.mass_trace, bytes, cudaMemcpyDeviceToHost));
        FILE * f = fopen(getenv("GGML_MOE_SKIP_TRACE"), "ab");
        const uint32_t header[] = {0x31574d53, 1, 736, std::min(counts[0], 16384u), counts[0] > 16384u, 16};
        GGML_ASSERT(f && fwrite(header, sizeof(header), 1, f) == 1 && fwrite(rows.data(), 1, bytes, f) == bytes);
        fclose(f);
        CUDA_CHECK(cudaFree(session.mass_trace));
        CUDA_CHECK(cudaFree(session.mass_trace_counts));
        session.mass_trace = nullptr;
        session.mass_trace_counts = nullptr;
    }
    moe_pf_release(&session); // fork PREFETCHFIX: tables below are about to be freed
    session.device_ready.store(false);
    ggml_moe_bank_onepath_set_live(false);
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
    moe_table_prefetch_release(session);
    if (session.devices.empty() || (session.device_tables.empty() && session.mapped_ranges.empty())) {
        session.device_frozen = false;
        return;
    }
    ggml_cuda_set_device(session.devices[0]->logical);
    // Tables can be in flight on a scheduler stream, not the legacy cache stream.
    CUDA_CHECK(cudaDeviceSynchronize());
    if (session.mj_retained_arena) {
        unsigned long long counts[7] = {};
        if (cudaMemcpy(counts, session.mj_retained_counts, sizeof(counts), cudaMemcpyDeviceToHost) == cudaSuccess) {
            MOE_CACHE_LOG("[moe-cache] missjoin retain final: dispatches=%llu hit_experts=%llu hit_routes=%llu copies=%llu copy_bytes=%llu evictions=%llu fallback_misses=%llu\n",
                    counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]);
        }
        (void) cudaFree(session.mj_retained_arena);
        session.mj_retained_arena = nullptr;
        session.mj_retained_counts = nullptr;
    }
    session.mj_retained_stream = nullptr;
    session.mj_retained_stream_set = false;
    moe_cache_swap_release(session);
    if (g_tail_counts) {
        unsigned long long tail[MOE_TAIL_COUNTS] = {};
        if (cudaMemcpy(tail, g_tail_counts, sizeof(tail), cudaMemcpyDeviceToHost) == cudaSuccess && tail[1] > 0) {
            MOE_CACHE_LOG("[moe-cache] skip-tail: %llu missed tensor-expert rows left out (%llu experts) over %llu layer windows (process total)\n",
                    tail[0], tail[2], tail[1]);
            MOE_CACHE_LOG("[moe-cache] skip-tail: token-layers by routes left out 0..10: %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
                    tail[3], tail[4], tail[5], tail[6], tail[7], tail[8], tail[9], tail[10], tail[11], tail[12], tail[13]);
        }
    }
    unsigned long long counts[3] = {};
    if (session.device_route_counts) {
        if (cudaMemcpy(counts, session.device_route_counts, sizeof(counts), cudaMemcpyDeviceToHost) == cudaSuccess) {
            MOE_CACHE_LOG("[moe-cache] device routes: hits=%llu misses=%llu dispatches=%llu; no host routing syncs\n",
                    counts[0], counts[1], counts[2]);
            moe_missjoin_log("final");
            moe_pf_log("final");
        }
        session.device_route_counts = nullptr;
    }
    if (session.device_expert_counts) {
        session.device_expert_counts = nullptr;
    }
    session.device_previous_counts.clear();
    session.device_profile_slices.clear();
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
                ggml_cuda_host_unregister((void *) range.begin);
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
        (ggml_ai4_is_unit(w->type) ? ggml_cuda_moe_ai_table_supported(w->type, w->ne[0], w->ne[1], ids->ne[1]) :
            ids->ne[1] <= get_mmvq_mmid_max_batch(w->type, ggml_cuda_info().devices[logical].cc)) &&
        ids->ne[2] == 1 && ids->ne[3] == 1 && a->ne[0] == d.n_in &&
        a->ne[1] > 0 && a->ne[1] <= ids->ne[0] && a->ne[2] == ids->ne[1] && a->ne[3] == 1 &&
        op->ne[0] == d.n_out && op->ne[1] == ids->ne[0] && op->ne[2] == ids->ne[1] && op->ne[3] == 1 &&
        ggml_get_op_params_i32(op, 3) != GGML_PREC_F32;
}

// fork: keep decode microbatches within both the host-plan and device-kernel limits.
static int moe_cache_decode_capacity(void * opaque, int64_t n_expert_used, const int32_t * types, int n_types) {
    auto * session = (moe_cache_session *) opaque;
    const char * enabled = getenv("GGML_MOE_CACHE_DEVICE");
    // The first concurrent verify may exceed the eight-token prepare gate.
    // Bound it from model types first, leaving allocation at its safe decode boundary.
    if (!session || !enabled || strcmp(enabled, "1") != 0 || session->devices.size() != 1 ||
            session->dormant || session->stopping || !types || n_types <= 0 ||
            n_expert_used <= 0 || n_expert_used > moe_cache_node_rows_max ||
            (session->device_attempted && !session->device_ready.load())) {
        return 0;
    }
    int capacity = std::min(session->config.max_batch, (int) (moe_cache_node_rows_max / n_expert_used));
    const int cc = ggml_cuda_info().devices[session->devices[0]->logical].cc;
    for (int i = 0; i < n_types; ++i) {
        if (!ggml_moe_cache_wtype_supported((ggml_type) types[i])) {
            return 0;
        }
        capacity = std::min(capacity, get_mmvq_mmid_max_batch((ggml_type) types[i], cc));
    }
    return capacity;
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

// fork (missjoin): opt-in miss staging for the device-table path.
// GGML_MOE_MISSJOIN=stage    : lossless. Each distinct missed expert of a dispatch is
//                              copied once from its mapped host alias into a VRAM scratch
//                              slot with wide coalesced loads; the unchanged MMV then reads
//                              the same bytes from VRAM (bit-identical results).
// GGML_MOE_MISSJOIN=diag_hot : DIAGNOSTIC ONLY, LOSSY. Misses are redirected to a resident
//                              expert of the same tensor, so no host bytes are read. Wrong
//                              outputs by design; used only to measure the no-miss ceiling.
#define MOE_MISSJOIN_MAX_ROUTES 1024
#define MOE_MISSJOIN_MAX_SLOTS 256

static int moe_missjoin_mode() {
    static const int mode = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN");
        if (!v) { return 0; }
        if (strcmp(v, "stage") == 0) { return 1; }
        if (strcmp(v, "diag_hot") == 0) { return 2; }
        return 0;
    }();
    return mode;
}

static int moe_missjoin_slots() {
    static const int slots = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_SLOTS");
        const int s = v ? atoi(v) : 64;
        return std::max(1, std::min(s, MOE_MISSJOIN_MAX_SLOTS));
    }();
    return slots;
}

static int moe_missjoin_blocks(int device) {
    static const int blocks = [device] {
        const char * v = getenv("GGML_MOE_MISSJOIN_BLOCKS");
        const int b = v ? atoi(v) : 2 * ggml_cuda_info().devices[device].nsm;
        return std::max(1, std::min(b, 4096));
    }();
    return blocks;
}

static bool moe_missjoin_expert_cta() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_MOE_MISSJOIN_EXPERT_CTA");
        return value && strcmp(value, "1") == 0;
    }();
    return enabled;
}

static bool moe_missjoin_reuse_only() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_MOE_MISSJOIN_REUSE_ONLY");
        return value && strcmp(value, "1") == 0;
    }();
    return enabled;
}

static constexpr int MOE_MISSJOIN_RETAIN_MAX = 8;

struct moe_missjoin_retained_state {
    int tags[MOE_MISSJOIN_RETAIN_MAX];
    unsigned long long ages[MOE_MISSJOIN_RETAIN_MAX];
    unsigned long long clock;
    int n_copy;
    const int4 * sources[MOE_MISSJOIN_MAX_SLOTS];
    int4 * destinations[MOE_MISSJOIN_MAX_SLOTS];
};

static int moe_missjoin_retained_slots() {
    static const int slots = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_RETAIN");
        return v ? std::max(0, std::min(atoi(v), MOE_MISSJOIN_RETAIN_MAX)) : 0;
    }();
    return slots;
}

// One planner owns tags; the following copy kernel reads an immutable plan.
// The same stream completes every copy and MMV before this table is planned again.
static __global__ void moe_missjoin_retained_plan_kernel(const void * const * table, const uint8_t * hot,
        const int32_t * ids, int n_ids, int n_routes, int ids_stride, int n_expert,
        const void ** eff, char * scratch, size_t stride, size_t copy_bytes, int slots,
        moe_missjoin_retained_state * state, char * retained, int retained_slots,
        unsigned long long * counts, unsigned long long * retained_counts) {
    __shared__ int selected[MOE_MISSJOIN_MAX_ROUTES];
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        selected[i] = ids[(i / n_ids) * ids_stride + i % n_ids];
    }
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        eff[e] = table[e];
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    bool protected_slot[MOE_MISSJOIN_RETAIN_MAX] = {};
    for (int i = 0; i < n_routes; ++i) {
        const int e = selected[i];
        if (e >= 0 && e < n_expert && !hot[e]) {
            for (int s = 0; s < retained_slots; ++s) {
                protected_slot[s] |= state->tags[s] == e;
            }
        }
    }
    unsigned long long distinct = 0, misses = 0, miss_routes = 0, hit_experts = 0, hit_routes = 0, evictions = 0;
    int n_copy = 0, n_scratch = 0, fallback = 0;
    for (int i = 0; i < n_routes; ++i) {
        const int e = selected[i];
        if (e < 0 || e >= n_expert) {
            continue;
        }
        miss_routes += !hot[e];
        bool first = true;
        for (int j = 0; j < i; ++j) {
            if (selected[j] == e) { first = false; break; }
        }
        if (!first) { continue; }
        ++distinct;
        if (hot[e]) { continue; }
        ++misses;
        int slot = -1;
        for (int s = 0; s < retained_slots; ++s) {
            if (state->tags[s] == e) { slot = s; break; }
        }
        if (slot >= 0) {
            eff[e] = retained + (size_t) slot * stride;
            state->ages[slot] = ++state->clock;
            ++hit_experts;
            for (int j = i; j < n_routes; ++j) { hit_routes += selected[j] == e; }
            continue;
        }
        if (((uintptr_t) table[e] & 15) != 0 || n_copy == slots) {
            ++fallback;
            continue;
        }
        for (int s = 0; s < retained_slots; ++s) {
            if (!protected_slot[s] && (slot < 0 || state->tags[s] < 0 || state->ages[s] < state->ages[slot])) {
                slot = s;
                if (state->tags[s] < 0) { break; }
            }
        }
        char * destination;
        if (slot >= 0) {
            evictions += state->tags[slot] >= 0;
            state->tags[slot] = e;
            state->ages[slot] = ++state->clock;
            protected_slot[slot] = true;
            destination = retained + (size_t) slot * stride;
        } else {
            destination = scratch + (size_t) n_scratch++ * stride;
        }
        eff[e] = destination;
        state->sources[n_copy] = (const int4 *) table[e];
        state->destinations[n_copy++] = (int4 *) destination;
    }
    state->n_copy = n_copy;
    if (counts) {
        atomicAdd(counts + 0, 1ULL);
        atomicAdd(counts + 1, distinct);
        atomicAdd(counts + 2, misses);
        atomicAdd(counts + 3, (unsigned long long) n_copy);
        atomicAdd(counts + 4, miss_routes);
        atomicAdd(counts + 5, (unsigned long long) fallback);
    }
    atomicAdd(retained_counts + 0, 1ULL);
    atomicAdd(retained_counts + 1, hit_experts);
    atomicAdd(retained_counts + 2, hit_routes);
    atomicAdd(retained_counts + 3, (unsigned long long) n_copy);
    atomicAdd(retained_counts + 4, (unsigned long long) n_copy * copy_bytes);
    atomicAdd(retained_counts + 5, evictions);
    atomicAdd(retained_counts + 6, (unsigned long long) fallback);
}

static __global__ void moe_missjoin_retained_copy_kernel(const moe_missjoin_retained_state * state, size_t copy_bytes) {
    const size_t chunks = copy_bytes / 16;
    const size_t total = chunks * (size_t) state->n_copy;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    size_t c = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; c + 3 * step < total; c += 4 * step) {
        int4 values[4];
        size_t rows[4], offsets[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const size_t ck = c + k * step;
            rows[k] = ck / chunks;
            offsets[k] = ck - rows[k] * chunks;
            values[k] = __ldcs(state->sources[rows[k]] + offsets[k]);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            state->destinations[rows[k]][offsets[k]] = values[k];
        }
    }
    for (; c < total; c += step) {
        const size_t row = c / chunks;
        const size_t offset = c - row * chunks;
        state->destinations[row][offset] = __ldcs(state->sources[row] + offset);
    }
}

// counts: 0 dispatches, 1 distinct experts, 2 distinct misses, 3 staged,
//         4 miss routes (with duplicates), 5 unstaged misses (overflow, unaligned, or bypassed).
// 6 expert-CTA dispatches, 7 bytes staged by that copy layout.
// 8 bypassed single-use images, 9 full-image bytes not staged for those images.
// Retained hits are counted separately and do not increment staged or unstaged.
template <bool expert_cta, bool reuse_only>
static __global__ void moe_missjoin_stage_kernel(const void * const * table, const uint8_t * hot,
        const int32_t * ids, int n_ids, int n_routes, int ids_stride, int n_expert,
        const void ** eff, char * scratch, size_t stride, size_t copy_bytes, int slots, int diag,
        unsigned long long * counts) {
    __shared__ int s_e[MOE_MISSJOIN_MAX_ROUTES];
    __shared__ unsigned char s_flag[MOE_MISSJOIN_MAX_ROUTES];
    __shared__ int s_stage[MOE_MISSJOIN_MAX_SLOTS];
    __shared__ int s_n_stage;
    __shared__ int s_sub;
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        s_e[i] = ids[(i / n_ids) * ids_stride + i % n_ids];
    }
    if (threadIdx.x == 0) { s_sub = INT_MAX; }
    __syncthreads();
    // bit0 first occurrence, bit1 miss, bit2 stageable, bit3 bypassed single-use image.
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        const int e = s_e[i];
        unsigned char f = 0;
        if (e >= 0 && e < n_expert) {
            bool first = true;
            for (int j = 0; j < i; ++j) {
                if (s_e[j] == e) { first = false; break; }
            }
            if (first) { f |= 1; }
            if (!hot[e]) {
                f |= 2;
                if (((uintptr_t) table[e] & 15) == 0) { f |= 4; }
            }
        }
        if constexpr (reuse_only) {
            if (!diag && f == 7) {
                bool reused = false;
                for (int j = i + 1; j < n_routes; ++j) {
                    if (s_e[j] == e) { reused = true; break; }
                }
                if (!reused) { f = (f & ~4) | 8; }
            }
        }
        s_flag[i] = f;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        if (s_flag[i] == 7) {
            int rank = 0;
            for (int j = 0; j < i; ++j) { rank += s_flag[j] == 7; }
            if (rank < slots) { s_stage[rank] = s_e[i]; }
        }
    }
    if (threadIdx.x == 0) {
        int stageable = 0;
        for (int i = 0; i < n_routes; ++i) { stageable += s_flag[i] == 7; }
        s_n_stage = min(stageable, slots);
    }
    if (diag && blockIdx.x == 0) {
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
            if (hot[e]) { atomicMin(&s_sub, e); }
        }
    }
    __syncthreads();
    const int n_stage = s_n_stage;
    if (blockIdx.x == 0) {
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
            eff[e] = table[e];
        }
        __syncthreads();
        if (diag) {
            if (s_sub != INT_MAX) {
                for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
                    if ((s_flag[i] & 3) == 3) { eff[s_e[i]] = table[s_sub]; }
                }
            }
        } else {
            for (int r = threadIdx.x; r < n_stage; r += blockDim.x) {
                eff[s_stage[r]] = scratch + (size_t) r * stride;
            }
        }
        if (threadIdx.x == 0 && counts) {
            unsigned long long distinct = 0, dmiss = 0, mroutes = 0, bypassed = 0;
            for (int i = 0; i < n_routes; ++i) {
                distinct += (s_flag[i] & 1) != 0;
                dmiss += (s_flag[i] & 3) == 3;
                mroutes += (s_flag[i] & 2) != 0;
                if constexpr (reuse_only) { bypassed += (s_flag[i] & 8) != 0; }
            }
            atomicAdd(counts + 0, 1ULL);
            atomicAdd(counts + 1, distinct);
            atomicAdd(counts + 2, dmiss);
            atomicAdd(counts + 3, diag ? 0ULL : (unsigned long long) n_stage);
            atomicAdd(counts + 4, mroutes);
            atomicAdd(counts + 5, diag ? 0ULL : dmiss - (unsigned long long) n_stage);
            if constexpr (reuse_only) {
                if (bypassed) {
                    atomicAdd(counts + 8, bypassed);
                    atomicAdd(counts + 9, bypassed * copy_bytes);
                }
            }
            if constexpr (expert_cta) {
                if (!diag) {
                    atomicAdd(counts + 6, 1ULL);
                    atomicAdd(counts + 7, (unsigned long long) n_stage * copy_bytes);
                }
            }
        }
    }
    if (diag || n_stage == 0) {
        return;
    }
    const size_t chunks = copy_bytes / 16;
    if constexpr (expert_cta) {
        // A block stays within one expert image, eliminating division by the
        // runtime image size from every vector copy. Distribute all blocks
        // evenly, including nondivisible grids and grids smaller than n_stage.
        const int groups = min(n_stage, (int) gridDim.x);
        const int group = blockIdx.x % groups;
        const int block = blockIdx.x / groups;
        const int blocks = ((int) gridDim.x - 1 - group) / groups + 1;
        const size_t step = (size_t) blocks * blockDim.x;
        for (int r = group; r < n_stage; r += groups) {
            const int4 * source = (const int4 *) table[s_stage[r]];
            int4 * destination = (int4 *) (scratch + (size_t) r * stride);
            size_t c = (size_t) block * blockDim.x + threadIdx.x;
            for (; c + 3 * step < chunks; c += 4 * step) {
                int4 values[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    values[k] = __ldcs(source + c + k * step);
                }
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    destination[c + k * step] = values[k];
                }
            }
            for (; c < chunks; c += step) {
                destination[c] = __ldcs(source + c);
            }
        }
        return;
    }
    const size_t total = chunks * (size_t) n_stage;
    const size_t step = (size_t) gridDim.x * blockDim.x;
    size_t c = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    // Four loads in flight per thread before the stores keep the link busy.
    for (; c + 3 * step < total; c += 4 * step) {
        int4 v[4];
        size_t r[4], o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const size_t ck = c + k * step;
            r[k] = ck / chunks;
            o[k] = ck - r[k] * chunks;
            v[k] = __ldcs((const int4 *) table[s_stage[r[k]]] + o[k]);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            ((int4 *) (scratch + r[k] * stride))[o[k]] = v[k];
        }
    }
    for (; c < total; c += step) {
        const size_t r = c / chunks;
        const size_t o = c - r * chunks;
        ((int4 *) (scratch + r * stride))[o] = __ldcs((const int4 *) table[s_stage[r]] + o);
    }
}

static void moe_missjoin_stage(const void * const * table, const uint8_t * hot,
        const int32_t * ids, int n_ids, int n_routes, int ids_stride, int n_expert,
        const void ** eff, char * scratch, size_t stride, size_t copy_bytes, int slots, int diag,
        unsigned long long * counts, int blocks, cudaStream_t stream) {
    auto launch = [&](auto copy_tag, auto reuse_tag) {
        constexpr bool expert_cta = decltype(copy_tag)::value;
        constexpr bool reuse_only = decltype(reuse_tag)::value;
        moe_missjoin_stage_kernel<expert_cta, reuse_only><<<blocks, 256, 0, stream>>>(
                table, hot, ids, n_ids, n_routes, ids_stride, n_expert,
                eff, scratch, stride, copy_bytes, slots, diag, counts);
    };
    auto select_reuse = [&](auto copy_tag) {
        // Retention has its own planner and keeps its existing placement policy.
        if (moe_missjoin_reuse_only() && !moe_missjoin_retained_slots()) {
            launch(copy_tag, std::true_type{});
        } else {
            launch(copy_tag, std::false_type{});
        }
    };
    if (moe_missjoin_expert_cta()) {
        select_reuse(std::true_type{});
    } else {
        select_reuse(std::false_type{});
    }
}

static bool moe_missjoin_overlap_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_OVERLAP");
        return v && strcmp(v, "1") == 0;
    }();
    return on && moe_missjoin_mode() == 1;
}

// fork (missjoin hit-first): lossless scheduling variant of stage mode. Needs the stage mode, and refuses
// the fused up/gate path and asynchronous prefetch, whose dispatch it does not reorder.
static bool moe_missjoin_hitfirst_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_HITFIRST");
        if (v && strcmp(v, "0") == 0) {  // default on with MISSJOIN=stage (owner 2026-10-22); 0 turns it off
            return false;
        }
        const char * fused = getenv("GGML_MOE_CACHE_DEVICE_FUSED");
        const char * pf = getenv("GGML_MOE_PREFETCH_ASYNC");
        const char * ov = getenv("GGML_MOE_MISSJOIN_OVERLAP");
        return !(fused && strcmp(fused, "1") == 0) && !(pf && atoi(pf) != 0) && !(ov && strcmp(ov, "1") == 0);
    }();
    return on && moe_missjoin_mode() == 1;
}

static int moe_hf_slots() {
    static const int slots = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_SLOTS");
        const int s = v ? atoi(v) : 32;
        return std::max(1, std::min(s, MOE_MISSJOIN_MAX_SLOTS));
    }();
    return slots;
}

static int moe_hf_blocks() {
    static const int blocks = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_HF_BLOCKS");
        const int b = v ? atoi(v) : 24;
        return std::max(1, std::min(b, 4096));
    }();
    return blocks;
}

static size_t moe_missjoin_copy_bytes(const moe_cache_device_table & t) {
    return (size_t) t.desc.n_out * ggml_row_size((ggml_type) t.desc.type, t.desc.n_in);
}

static void moe_missjoin_retained_prepare(moe_cache_session & session) {
    const int slots = moe_missjoin_retained_slots();
    if (!slots || moe_missjoin_mode() != 1) {
        return;
    }
    if (moe_missjoin_overlap_enabled()) {
        MOE_CACHE_LOG("[moe-cache] missjoin retain disabled: overlap uses separate staging buffers\n");
        return;
    }
    const size_t state_bytes = GGML_PAD(sizeof(moe_missjoin_retained_state), 256);
    const size_t cap = 512u << 20;
    size_t bytes = 256;
    for (const auto & entry : session.device_tables) {
        const auto & table = entry.second;
        const size_t copy_bytes = moe_missjoin_copy_bytes(table);
        if (!table.hot || !copy_bytes || copy_bytes % 16 != 0) { continue; }
        const size_t stride = GGML_PAD(copy_bytes, 256);
        if (stride > cap / slots || bytes > cap - state_bytes || stride * slots > cap - bytes - state_bytes) {
            MOE_CACHE_LOG("[moe-cache] missjoin retain disabled: arena exceeds %zu bytes\n", cap);
            return;
        }
        bytes += state_bytes + stride * slots;
    }
    if (bytes == 256) { return; }
    size_t free_bytes = 0, total_bytes = 0;
    const size_t free_floor = moe_cache_physical_free_floor();
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes < bytes || free_bytes - bytes < free_floor) {
        (void) cudaGetLastError();
        MOE_CACHE_LOG("[moe-cache] missjoin retain disabled: physical free floor\n");
        return;
    }
    char * arena = nullptr;
    bool ok = cudaMalloc((void **) &arena, bytes) == cudaSuccess;
    if (ok) { ok = cudaMemset(arena, 0, 7 * sizeof(unsigned long long)) == cudaSuccess; }
    size_t offset = 256, tables = 0;
    for (auto & entry : session.device_tables) {
        auto & table = entry.second;
        const size_t copy_bytes = moe_missjoin_copy_bytes(table);
        if (!ok || !table.hot || !copy_bytes || copy_bytes % 16 != 0) { continue; }
        table.mj_retained = (moe_missjoin_retained_state *) (arena + offset);
        table.mj_retained_data = arena + offset + state_bytes;
        ok = cudaMemset(table.mj_retained, 0, sizeof(moe_missjoin_retained_state)) == cudaSuccess &&
             cudaMemset(table.mj_retained->tags, 0xff, sizeof(table.mj_retained->tags)) == cudaSuccess;
        offset += state_bytes + GGML_PAD(copy_bytes, 256) * slots;
        ++tables;
    }
    // Compute streams are nonblocking, so publish tags only after initialization completes.
    if (ok) { ok = cudaStreamSynchronize(nullptr) == cudaSuccess; }
    if (!ok) {
        for (auto & entry : session.device_tables) {
            entry.second.mj_retained = nullptr;
            entry.second.mj_retained_data = nullptr;
        }
        if (arena) { (void) cudaFree(arena); }
        (void) cudaGetLastError();
        MOE_CACHE_LOG("[moe-cache] missjoin retain disabled: allocation failed\n");
        return;
    }
    session.mj_retained_arena = arena;
    session.mj_retained_counts = (unsigned long long *) arena;
    MOE_CACHE_LOG("[moe-cache] missjoin retain: slots=%d tables=%zu arena_bytes=%zu cap_bytes=%zu free_before=%zu free_floor=%zu\n",
            slots, tables, bytes, cap, free_bytes, free_floor);
}

static bool moe_missjoin_retained_use(ggml_backend_cuda_context & ctx, moe_cache_session & session,
        const moe_cache_device_table & table) {
    if (!table.mj_retained || ctx.curr_stream_no != 0) { return false; }
    std::lock_guard<std::mutex> lock(session.mu);
    if (!session.mj_retained_stream_set) {
        session.mj_retained_stream = ctx.stream();
        session.mj_retained_stream_set = true;
    }
    return session.mj_retained_stream == ctx.stream();
}

// fork MOEGAP: GGML_MOE_CPU_SPLIT=1 turns the CPU expert split on (default off).
static bool moe_cs_enabled() {
    static const bool on = [] { const char * v = getenv("GGML_MOE_CPU_SPLIT"); return v && strcmp(v, "1") == 0; }();
    return on;
}

static float moe_cs_env_float(const char * name, float def) {
    const char * v = getenv(name);
    return v && *v ? (float) atof(v) : def;
}

// AIPFIX: GGML_MOE_CPU_SPLIT_PROBE_PER_TYPE=1 probes one layer per distinct host type triple (up, gate, down) and the
// planner uses each layer's own cost (default off: one probe on the first layer, its cost for every layer).
static bool moe_cs_per_type_enabled() {
    static const bool on = [] { const char * v = getenv("GGML_MOE_CPU_SPLIT_PROBE_PER_TYPE"); return v && strcmp(v, "1") == 0; }();
    return on;
}

// Host-to-card copy rate of pinned memory, bytes per second (best of three 32 MiB copies); 0 when it fails.
static double moe_cs_probe_link(void * dst, size_t cap) {
    const size_t bytes = std::min(cap, (size_t) 32u << 20);
    void * src = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaStream_t st = nullptr;
    double best = 0.0;
    if (cudaHostAlloc(&src, bytes, cudaHostAllocDefault) == cudaSuccess &&
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) == cudaSuccess &&
            cudaEventCreate(&e0) == cudaSuccess && cudaEventCreate(&e1) == cudaSuccess) {
        memset(src, 1, bytes);
        for (int rep = 0; rep < 4; ++rep) {
            float ms = 0.0f;
            if (cudaEventRecord(e0, st) != cudaSuccess || moe_h2d_async(dst, src, bytes, cudaMemcpyHostToDevice, st) != cudaSuccess ||
                    cudaEventRecord(e1, st) != cudaSuccess || cudaEventSynchronize(e1) != cudaSuccess ||
                    cudaEventElapsedTime(&ms, e0, e1) != cudaSuccess) {
                best = 0.0;
                break;
            }
            if (rep > 0 && ms > 0.0f) { best = std::max(best, bytes / (ms * 1e-3)); }
        }
    }
    if (e0) { (void) cudaEventDestroy(e0); }
    if (e1) { (void) cudaEventDestroy(e1); }
    if (st) { (void) cudaStreamDestroy(st); }
    if (src) { (void) cudaFreeHost(src); }
    (void) cudaGetLastError();
    return best;
}

static void moe_cs_log_inactive(const char * reason) {
    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("[moe-cpu-split] startup: CPU split off (%s); measured link=unavailable CPU probe=unavailable CPU share=0\n", reason);
        logged = true;
    }
}

// Mapped job memory, the host pool, the layer registrations and the startup probe (once per process; the layers
// are registered again for every session). Needs the hit-first buffers (g_hf_scratch is the link probe's target).
static void moe_cs_prepare(moe_cache_session & session) {
    session.cs_registered.assign(session.mj_groups.size(), 0);
    session.cs_valid.assign(session.mj_groups.size(), 0);
    session.cs_layer = -1;
    session.cs_route_l.clear();
    session.cs_fixed_l.clear();
    if (!moe_cs_enabled() || session.devices.size() != 1) {
        moe_cs_log_inactive(!moe_cs_enabled() ? "disabled" : "requires one device");
        return;
    }
    int64_t max_ff = 0, max_embd = 0;
    for (const auto & g : session.mj_groups) {
        if (g[0] && g[1] && g[2]) {
            max_ff = std::max(max_ff, g[0]->desc.n_out);
            max_embd = std::max(max_embd, g[0]->desc.n_in);
        }
    }
    if (!max_ff || !max_embd) {
        moe_cs_log_inactive("no host layer registered");
        return;
    }
    static bool failed = false;
    if (!g_cs_job_h && !failed) {
        g_cs_x_stride = (size_t) GGML_PAD(max_embd, 64);
        g_cs_out_stride = (size_t) GGML_PAD(max_embd, 64);
        void * p = nullptr;
        bool ok = cudaHostAlloc(&p, sizeof(moe_cs_job), cudaHostAllocMapped) == cudaSuccess;
        if (ok) { g_cs_job_h = (moe_cs_job *) p; memset(p, 0, sizeof(moe_cs_job)); ok = cudaHostGetDevicePointer((void **) &g_cs_job_d, p, 0) == cudaSuccess; }
        const size_t xb = MOE_CS_MAX_ROUTES * g_cs_x_stride * sizeof(float);
        const size_t ob = MOE_CS_MAX_ROUTES * g_cs_out_stride * sizeof(float);
        if (ok) { ok = cudaHostAlloc(&p, xb, cudaHostAllocMapped) == cudaSuccess; }
        if (ok) { g_cs_x_h = (float *) p; memset(p, 0, xb); ok = cudaHostGetDevicePointer((void **) &g_cs_x_d, p, 0) == cudaSuccess; }
        if (ok) { ok = cudaHostAlloc(&p, ob, cudaHostAllocMapped) == cudaSuccess; }
        if (ok) { g_cs_out_h = (float *) p; memset(p, 0, ob); ok = cudaHostGetDevicePointer((void **) &g_cs_out_d, p, 0) == cudaSuccess; }
        if (ok) {
            ok = cudaMalloc((void **) &g_cs_plan, sizeof(moe_cs_plan_dev)) == cudaSuccess &&
                 cudaMemset(g_cs_plan, 0, sizeof(moe_cs_plan_dev)) == cudaSuccess &&
                 cudaMalloc((void **) &g_cs_mask, 4 * (size_t) g_hf_experts) == cudaSuccess &&
                 cudaMemset(g_cs_mask, 1, 4 * (size_t) g_hf_experts) == cudaSuccess;
        }
        ok = ok && moe_cs_start(g_cs_job_h, g_cs_x_h, g_cs_x_stride, g_cs_out_h, g_cs_out_stride, max_ff, max_embd);
        if (!ok) {
            (void) cudaGetLastError();
            failed = true;
            g_cs_job_h = nullptr;
            MOE_CACHE_LOG("[moe-cache] cpu split: setup failed, off\n");
            moe_cs_log_inactive("host setup failed");
            return;
        }
    }
    if (!g_cs_job_h) {
        moe_cs_log_inactive("host workers unavailable");
        return;
    }
    size_t layers = 0;
    int host_alt_layers = 0;
    for (size_t l = 0; l < session.mj_groups.size(); ++l) {
        const auto & g = session.mj_groups[l];
        if (!g[0] || !g[1] || !g[2]) { continue; }
        moe_cs_role roles[3];
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            const auto & t = *g[k];
            roles[k].type = t.desc.type;
            roles[k].n_in = t.desc.n_in;
            roles[k].n_out = t.desc.n_out;
            roles[k].n_expert = t.desc.n_expert;
            roles[k].row_bytes = ggml_row_size((ggml_type) t.desc.type, t.desc.n_in);
            roles[k].src = t.copy_source.data();
            ok = ok && t.copy_source.size() == (size_t) t.desc.n_expert && t.desc.n_expert <= g_hf_experts;
        }
        // AIPFAST route 2: an AI+ layer computes its host misses from the base file's 3XS copy when one is named
        // (GGML_MOE_CPU_SPLIT_HOST_GGUF); the card keeps the AI+ bytes. All three projections or none: a layer with only
        // down in AI+ (gate/up already 3XS) also swaps gate/up, which points them at the same 3XS bytes in the base file.
        moe_cs_role own[3] = { roles[0], roles[1], roles[2] };
        bool swapped = false;
        if (ok && (ggml_ai4_is_unit((ggml_type) roles[0].type) || ggml_ai4_is_unit((ggml_type) roles[2].type))) {
            static const char * kind[3] = {"up", "gate", "down"};
            moe_cs_role alt[3];
            bool have = true;
            for (int k = 0; k < 3 && have; ++k) {
                char name[64];
                snprintf(name, sizeof(name), "blk.%d.ffn_%s_exps.weight", (int) l, kind[k]);
                have = moe_cs_host_alt(name, roles[k].n_in, roles[k].n_out, roles[k].n_expert, &alt[k]);
            }
            if (have) {
                char nm[3][64];
                const char * nms[3];
                for (int k = 0; k < 3; ++k) {
                    snprintf(nm[k], sizeof(nm[k]), "blk.%d.ffn_%s_exps.weight", (int) l, kind[k]);
                    nms[k] = nm[k];
                }
                have = moe_cs_host_alt_keep((int) l, nms);   // over the RAM budget: keep the AI+ decode, no disk reads
            }
            if (have) {
                for (int k = 0; k < 3; ++k) { roles[k] = alt[k]; }
                swapped = true;
            }
        }
        bool reg = ok && moe_cs_register_layer((int) l, roles);
        if (ok && !reg && swapped) {
            // the 3XS copy has no host row kernel in this build: keep the AI+ decode rather than lose the layer's host pool
            fprintf(stderr, "[moe-cpu-split] layer %zu: host 3XS copy not usable (type %s), host misses keep the AI+ decode\n",
                    l, ggml_type_name((ggml_type) roles[0].type));
            swapped = false;
            reg = moe_cs_register_layer((int) l, own);
        }
        host_alt_layers += reg && swapped;
        if (reg) {
            session.cs_registered[l] = 1;
            ++layers;
        }
    }
    static bool probed = false;
    static std::map<std::array<int, 3>, std::array<double, 2>> type_cost;   // AIPFIX per-type: raw probes (ns)
    if (!probed && layers) {
        probed = true;
        double route_ns = 0.0, fixed_ns = 0.0;
        int timed_layer = -1;
        const bool cpu_ok = moe_cs_probe(&route_ns, &fixed_ns, &timed_layer);
        if (cpu_ok && moe_cs_per_type_enabled()) {
            // reuse the first probe for the type triple of the layer it actually timed (lowest pool-ok layer)
            int t[3];
            if (moe_cs_layer_types(timed_layer, t)) { type_cost[{t[0], t[1], t[2]}] = {route_ns, fixed_ns}; }
        }
        const double link = moe_cs_probe_link(g_hf_scratch, 3 * (size_t) g_hf_slots * g_hf_stride);
        // overrides: microseconds per (expert, route) on the CPU, microseconds per job, link GB/s
        g_cs_route_ns = moe_cs_env_float("GGML_MOE_CPU_SPLIT_ROUTE_US", cpu_ok ? (float) (route_ns / 1000.0) : 60.0f) * 1000.0f;
        g_cs_fixed_ns = moe_cs_env_float("GGML_MOE_CPU_SPLIT_FIXED_US", cpu_ok ? (float) (fixed_ns / 1000.0) : 10.0f) * 1000.0f +
                        moe_cs_env_float("GGML_MOE_CPU_SPLIT_HANDSHAKE_US", 8.0f) * 1000.0f;
        const float gbps = moe_cs_env_float("GGML_MOE_CPU_SPLIT_GBPS", link > 0.0 ? (float) (link / 1e9) : 20.0f);
        g_cs_ns_per_byte = gbps > 0.0f ? 1.0f / gbps : 0.0f;
        g_cs_max_exp = std::max(1, std::min((int) moe_cs_env_float("GGML_MOE_CPU_SPLIT_MAX_EXPERTS", 8.0f), MOE_CS_MAX_EXPERTS));
        g_cs_max_route = std::max(1, std::min((int) moe_cs_env_float("GGML_MOE_CPU_SPLIT_MAX_ROUTES", 24.0f), MOE_CS_MAX_ROUTES));
        const char * share = getenv("LLAMA_MOE_CPU_SHARE");
        if (share && *share) {
            char * end = nullptr;
            const float value = strtof(share, &end);
            if (end == share || *end || !(value >= 0.0f && value <= 1.0f)) {
                GGML_LOG_WARN("[moe-cpu-split] ignoring invalid LLAMA_MOE_CPU_SHARE: expected a number in [0, 1]\n");
            } else {
                g_cs_share = value;
            }
        }
        const bool first = moe_cs_env_float("GGML_MOE_CPU_SPLIT_FIRST", 0.0f) != 0.0f;
        g_cs_adapt = moe_cs_env_float("GGML_MOE_CPU_SPLIT_ADAPT", first ? 1.0f : 0.0f) != 0.0f ? 1 : 0;
        g_cs_partial = moe_cs_env_float("GGML_MOE_CPU_SPLIT_PARTIAL", first ? 1.0f : 0.0f) != 0.0f ? 1 : 0;
        if (g_cs_share >= 0.0f) { g_cs_partial = 1; }
        g_cs_floor_ns = std::max(0.0f, moe_cs_env_float("GGML_MOE_CPU_SPLIT_FLOOR_US", first ? 4.0f : 0.0f)) * 1000.0f;
        g_cs_link_w = std::max(0.05f, moe_cs_env_float("GGML_MOE_CPU_SPLIT_LINK_WEIGHT", 1.0f));
        char share_text[64] = "auto (chosen per layer)";
        if (g_cs_share >= 0.0f) { snprintf(share_text, sizeof(share_text), "%.3f (override, within work caps)", g_cs_share); }
        GGML_LOG_INFO("[moe-cpu-split] startup: measured link=%.2f GB/s CPU probe=%.2f us CPU share=%s\n",
                link / 1e9, cpu_ok ? route_ns / 1000.0 : -1.0, share_text);
        MOE_CACHE_LOG("[moe-cache] cpu split: partial experts %s, card floor %.1f us per distinct expert, link weight %.2f\n",
                g_cs_partial ? "on" : "off", g_cs_floor_ns / 1000.0f, g_cs_link_w);
        MOE_CACHE_LOG("[moe-cache] cpu split: probe link %.1f GB/s -> using %.1f GB/s; CPU %.1f us per (expert, route), "
                "%.1f us per job incl. handshake; at most %d experts / %d routes per layer; per-route cost %s\n",
                link / 1e9, gbps, g_cs_route_ns / 1000.0f, g_cs_fixed_ns / 1000.0f, g_cs_max_exp, g_cs_max_route,
                g_cs_adapt ? "corrected from measured jobs (ADAPT)" : "fixed");
    }
    g_cs_on = layers > 0 && g_cs_ns_per_byte > 0.0f;
    if (!g_cs_on) { moe_cs_log_inactive(layers ? "link disabled" : "no host layer registered"); }
    if (g_cs_on && moe_cs_per_type_enabled()) {
        std::vector<std::array<int, 3>> key(session.cs_registered.size(), std::array<int, 3>{-1, -1, -1});
        for (size_t l = 0; l < key.size(); ++l) {
            int t[3];
            if (session.cs_registered[l] && moe_cs_layer_types((int) l, t)) { key[l] = {t[0], t[1], t[2]}; }
        }
        const float hs_ns = moe_cs_env_float("GGML_MOE_CPU_SPLIT_HANDSHAKE_US", 8.0f) * 1000.0f;
        const bool route_env = getenv("GGML_MOE_CPU_SPLIT_ROUTE_US") && *getenv("GGML_MOE_CPU_SPLIT_ROUTE_US");
        const bool fixed_env = getenv("GGML_MOE_CPU_SPLIT_FIXED_US") && *getenv("GGML_MOE_CPU_SPLIT_FIXED_US");
        std::vector<double> rl, fl;
        moe_cs_type_costs(key, [](int l, double * r, double * f) { return moe_cs_probe_at(l, r, f); },
                (double) g_cs_route_ns, (double) (g_cs_fixed_ns - hs_ns), type_cost, rl, fl);
        session.cs_route_l.assign(key.size(), 0.0f);
        session.cs_fixed_l.assign(key.size(), 0.0f);
        std::map<std::array<int, 3>, int> n_of;
        for (size_t l = 0; l < key.size(); ++l) {
            if (key[l][0] < 0) { continue; }
            session.cs_route_l[l] = route_env ? g_cs_route_ns : (float) rl[l];
            session.cs_fixed_l[l] = fixed_env ? g_cs_fixed_ns : (float) fl[l] + hs_ns;
            ++n_of[key[l]];
        }
        for (const auto & kv : n_of) {
            const auto & c = type_cost[kv.first];
            MOE_CACHE_LOG("[moe-cache] cpu split per-type probe: up %s gate %s down %s, %d layers: %.1f us per (expert, route), "
                    "%.1f us per job incl. handshake (x%.2f the reference)%s\n",
                    ggml_type_name((ggml_type) kv.first[0]), ggml_type_name((ggml_type) kv.first[1]),
                    ggml_type_name((ggml_type) kv.first[2]), kv.second,
                    route_env ? g_cs_route_ns / 1000.0f : (float) (c[0] / 1000.0),
                    fixed_env ? g_cs_fixed_ns / 1000.0f : (float) ((c[1] + hs_ns) / 1000.0),
                    route_env || g_cs_route_ns <= 0.0f ? 1.0f : (float) (c[0] / g_cs_route_ns),
                    route_env || fixed_env ? " (env override)" : "");
        }
    }
    MOE_CACHE_LOG("[moe-cache] cpu split: %s, %zu of %zu layers on the host pool (%d threads)\n",
            g_cs_on ? "on" : "off", layers, session.mj_groups.size(), moe_cs_threads());
    if (host_alt_layers) {
        fprintf(stderr, "[moe-cpu-split] host misses of %d AI+ layers use the 3XS copies in GGML_MOE_CPU_SPLIT_HOST_GGUF\n", host_alt_layers);
        moe_cs_host_alt_warm();
    }
}

// Called at the end of device preparation, outside graph capture.
static void moe_missjoin_overlap_prepare(moe_cache_session & session) {
    session.mj_groups.clear();
    session.mj_group = -1;
    session.mj_ids = nullptr;
    session.mj_pending = 0;
    session.pc_mask = 0;
    session.hf_order.clear();
    session.hf_seq.clear();
    const bool hitfirst = moe_missjoin_hitfirst_enabled();
    if (!hitfirst) {
        moe_cs_log_inactive(!moe_cs_enabled() ? "disabled" : "hit-first disabled");
    }
    if (!moe_missjoin_overlap_enabled() && !hitfirst) {
        return;
    }
    size_t stride = 0;
    int64_t experts = 0;
    for (auto & entry : session.device_tables) {
        auto & t = entry.second;
        t.mj_layer = -1;
        t.mj_kind = -1;
        int layer = -1;
        char kind[16] = {};
        if (!t.desc.name || sscanf(t.desc.name, "blk.%d.ffn_%15[a-z]_exps", &layer, kind) != 2 || layer < 0 || layer >= 1024) {
            continue;
        }
        const int k = strcmp(kind, "up") == 0 ? 0 : strcmp(kind, "gate") == 0 ? 1 : strcmp(kind, "down") == 0 ? 2 : -1;
        const size_t bytes = moe_missjoin_copy_bytes(t);
        if (k < 0 || !t.hot || bytes % 16 != 0 || t.desc.n_expert > INT_MAX) {
            continue;
        }
        if ((int) session.mj_groups.size() <= layer) {
            session.mj_groups.resize(layer + 1, {nullptr, nullptr, nullptr});
        }
        if (session.mj_groups[layer][k]) {
            continue;
        }
        session.mj_groups[layer][k] = &t;
        t.mj_layer = layer;
        t.mj_kind = k;
        stride = std::max(stride, (size_t) GGML_PAD(bytes, 256));
        experts = std::max(experts, t.desc.n_expert);
    }
    if (hitfirst) {
        const int slots = moe_hf_slots();
        bool ok = !session.mj_groups.empty();
        if (ok && !g_hf_side) {
            int least = 0, greatest = 0;
            ok = cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess &&
                 cudaStreamCreateWithPriority(&g_hf_side, cudaStreamNonBlocking, greatest) == cudaSuccess &&
                 cudaEventCreateWithFlags(&g_hf_fork, cudaEventDisableTiming) == cudaSuccess;
            for (int k = 0; ok && k < 3; ++k) {
                ok = cudaEventCreateWithFlags(&g_hf_done[k], cudaEventDisableTiming) == cudaSuccess;
            }
        }
        if (ok && !g_hf_scratch) {
            ok = cudaMalloc((void **) &g_hf_scratch, 3 * (size_t) slots * stride) == cudaSuccess &&
                 cudaMalloc((void **) &g_hf_eff, 3 * (size_t) experts * sizeof(void *)) == cudaSuccess &&
                 cudaMalloc((void **) &g_hf_src, 3 * (size_t) slots * sizeof(void *)) == cudaSuccess &&
                 cudaMalloc((void **) &g_hf_nstage, 3 * sizeof(int)) == cudaSuccess &&
                 cudaMalloc((void **) &g_hf_miss, 4 * sizeof(moe_miss_list)) == cudaSuccess &&
                 cudaMemset(g_hf_miss, 0, 4 * sizeof(moe_miss_list)) == cudaSuccess &&
                 cudaMalloc((void **) &g_hf_zero, (size_t) experts) == cudaSuccess &&
                 cudaMemset(g_hf_zero, 0, (size_t) experts) == cudaSuccess &&
                 cudaMalloc((void **) &g_tail_mask, 4 * (size_t) experts) == cudaSuccess &&
                 cudaMemset(g_tail_mask, 0, 4 * (size_t) experts) == cudaSuccess &&
                 cudaMalloc((void **) &g_tail_counts, MOE_TAIL_COUNTS * sizeof(unsigned long long)) == cudaSuccess &&
                 cudaMemset(g_tail_counts, 0, MOE_TAIL_COUNTS * sizeof(unsigned long long)) == cudaSuccess;
            if (ok) {
                g_hf_stride = stride;
                g_hf_experts = experts;
                g_hf_slots = slots;
            }
            const size_t sw_rows = (size_t) MOE_SWST_LAYERS * experts;
            if (ok && moe_swst_path()) {
                ok = cudaMalloc((void **) &g_swst, sw_rows * MOE_SWST_COLS * sizeof(unsigned long long)) == cudaSuccess &&
                     cudaMemset(g_swst, 0, sw_rows * MOE_SWST_COLS * sizeof(unsigned long long)) == cudaSuccess;
            }
            if (ok && moe_swap_score_mode()) {
                ok = cudaMalloc((void **) &g_swscore, sw_rows * sizeof(unsigned long long)) == cudaSuccess &&
                     cudaMemset(g_swscore, 0, sw_rows * sizeof(unsigned long long)) == cudaSuccess &&
                     cudaHostAlloc((void **) &g_swscore_host, sw_rows * sizeof(unsigned long long), cudaHostAllocPortable) == cudaSuccess;
                if (ok) { memset(g_swscore_host, 0, sw_rows * sizeof(unsigned long long)); }
            }
        }
        ok = ok && stride <= g_hf_stride && experts <= g_hf_experts && slots <= g_hf_slots;
        if (!ok) {
            moe_cs_log_inactive(session.mj_groups.empty() ? "no host layer registered" : "hit-first setup failed");
            (void) cudaGetLastError();
            session.mj_groups.clear();
            for (auto & entry : session.device_tables) { entry.second.mj_layer = -1; entry.second.mj_kind = -1; }
        } else {
            session.hf_order.assign(session.mj_groups.size(), std::array<int8_t, 3>{-1, -1, -1});
            moe_cs_prepare(session);
        }
        size_t complete_hf = 0;
        for (const auto & g : session.mj_groups) { complete_hf += g[0] && g[1] && g[2]; }
        MOE_CACHE_LOG("[moe-cache] missjoin hit-first: %s, %zu complete layers, slots=%d, copy blocks=%d, arena=%zu MiB\n",
                ok ? "on" : "off (setup failed)", complete_hf, slots, moe_hf_blocks(),
                ok ? (3 * (size_t) g_hf_slots * g_hf_stride) >> 20 : 0);
        return;
    }
    const int slots = moe_missjoin_slots();
    bool ok = !session.mj_groups.empty();
    if (ok && !g_mj_side) {
        ok = cudaStreamCreateWithFlags(&g_mj_side, cudaStreamNonBlocking) == cudaSuccess &&
             cudaEventCreateWithFlags(&g_mj_fork, cudaEventDisableTiming) == cudaSuccess;
        for (int k = 0; ok && k < 3; ++k) {
            ok = cudaEventCreateWithFlags(&g_mj_done[k], cudaEventDisableTiming) == cudaSuccess;
        }
    }
    if (ok && !g_mj_scratch) {
        ok = cudaMalloc((void **) &g_mj_scratch, 3 * (size_t) slots * stride) == cudaSuccess &&
             cudaMalloc((void **) &g_mj_eff, 3 * (size_t) experts * sizeof(void *)) == cudaSuccess;
        if (ok) {
            g_mj_stride = stride;
            g_mj_experts = experts;
            g_mj_slots = slots;
        }
    }
    ok = ok && stride <= g_mj_stride && experts <= g_mj_experts && slots <= g_mj_slots;
    if (!ok) {
        (void) cudaGetLastError();
        session.mj_groups.clear();
        for (auto & entry : session.device_tables) { entry.second.mj_layer = -1; entry.second.mj_kind = -1; }
    }
    size_t complete = 0;
    for (const auto & g : session.mj_groups) { complete += g[0] && g[1] && g[2]; }
    MOE_CACHE_LOG("[moe-cache] missjoin overlap: %s, %zu complete layers, scratch=%zu MiB\n",
            ok ? "on" : "off (setup failed)", complete, ok ? (3 * (size_t) g_mj_slots * g_mj_stride) >> 20 : 0);
}

// Stages all three projections of a layer on the side stream; returns the table copy for this one.
static const void * const * moe_missjoin_overlap_table(ggml_backend_cuda_context & ctx, moe_cache_session & session,
        const moe_cache_device_table & table, const ggml_tensor * ids) {
    if (table.mj_layer < 0 || table.mj_layer >= (int) session.mj_groups.size()) {
        return nullptr;
    }
    const auto & group = session.mj_groups[table.mj_layer];
    if (!group[0] || !group[1] || !group[2]) {
        return nullptr;
    }
    const int routes = (int)(ids->ne[0] * ids->ne[1]);
    if (routes > MOE_MISSJOIN_MAX_ROUTES) {
        return nullptr;
    }
    const int k = table.mj_kind;
    cudaStream_t main = ctx.stream();
    if (!(session.mj_group == table.mj_layer && session.mj_ids == ids->data && (session.mj_pending >> k & 1u))) {
        // Join any copies a previous group left unconsumed before reusing the buffers.
        for (int j = 0; j < 3; ++j) {
            if (session.mj_pending >> j & 1u) {
                CUDA_CHECK(cudaStreamWaitEvent(main, g_mj_done[j], 0));
            }
        }
        CUDA_CHECK(cudaEventRecord(g_mj_fork, main));
        CUDA_CHECK(cudaStreamWaitEvent(g_mj_side, g_mj_fork, 0));
        for (int j = 0; j < 3; ++j) {
            const auto & m = *group[j];
            moe_missjoin_stage(
                    m.staging_device ? m.staging_device : m.device, m.hot, (const int32_t *) ids->data,
                    (int) ids->ne[0], routes, (int)(ids->nb[1] / sizeof(int32_t)), (int) m.desc.n_expert,
                    g_mj_eff + (size_t) j * g_mj_experts, g_mj_scratch + (size_t) j * g_mj_slots * g_mj_stride,
                    g_mj_stride, moe_missjoin_copy_bytes(m), g_mj_slots, 0, session.device_route_counts ? g_missjoin_counts : nullptr,
                    moe_missjoin_blocks(ctx.device), g_mj_side);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(g_mj_done[j], g_mj_side));
        }
        session.mj_group = table.mj_layer;
        session.mj_ids = ids->data;
        session.mj_pending = 7u;
    }
    CUDA_CHECK(cudaStreamWaitEvent(main, g_mj_done[k], 0));
    session.mj_pending &= ~(1u << k);
    return g_mj_eff + (size_t) k * g_mj_experts;
}

// ---------------------------------------------------------------------------------------------
// fork (missjoin hit-first): per-layer miss plan, per-projection copy, resident-first math.
//
// Order of work in one layer, for projections P1, P2, P3 in dispatch order (up, gate, down):
//   main stream:  fork | q8(P1) | hit(P1) | wait copy(P1) | miss(P1) | q8(P2) | hit(P2) | wait copy(P2) | ...
//   side stream:  plan(all three) | copy(P1) | copy(P2) | copy(P3)
// The plan removes duplicate routes once for the layer and lists each projection's distinct missed
// experts; each copy launch then moves exactly those experts. A projection's resident routes run while its
// copy (and every later copy) is in flight; its missed routes run after its own event. Every route is
// computed once by the same kernel from the same bytes, so results equal the serial staging path bit for bit.
// Buffer lifetime: the next layer's plan is ordered after this layer's last miss pass (its fork event is
// recorded on the main stream after it), so the persistent arena is never overwritten while read.

struct moe_hf_plan_args {
    const void * const * table[3];
    const uint8_t * hot[3];
    const void ** eff[3];
    const void ** src[3];
    int * nstage;
    char * scratch;
    size_t stride;
    // fork MOEGAP: CPU expert split (cs null: off for this layer; only the fast plan kernel reads these)
    moe_cs_plan_dev * cs;
    uint8_t * cs_mask;            // [3][cs_mask_stride]: nonzero = the miss pass skips the expert
    uint8_t * cs_cpu;             // fork MOEGAP DOWNONE: [cs_mask_stride] 1 = on the CPU this layer (null: off)
    int64_t cs_mask_stride;
    float cs_route_ns, cs_fixed_ns, cs_ns_per_byte;
    float cs_adapt_scale;         // AIPFIX per-type probe: this layer's route cost / the reference cost (1 = off)
    int cs_adapt;
    float cs_bytes[3];            // copy bytes of one expert per projection
    int cs_max_exp, cs_max_route;
    int cs_partial;               // fork MOEGAP CPU-first (see g_cs_partial)
    float cs_floor_ns, cs_link_w, cs_share;
    moe_miss_list * miss;         // fork MOEGAP MISSLIST: [4] lists to write (null: off; fast kernel only)
    unsigned long long * pc_exp[3];   // fork MOEGAP PLANCOUNT: per-kind expert counts (null: off; fast kernel only)
    unsigned long long * pc_route;    // [0] hot routes, [1] cold routes, [2] dispatches (may be null)
    // fork MOEGAP PUBPLAN: the fast plan kernel publishes the host job itself (cs_job null: the publish kernel does)
    moe_cs_job * cs_job;
    const char * cs_x;
    size_t cs_nb1, cs_nb2;
    int cs_x_per_slot, cs_n_ids, cs_n_embd, cs_layer;
    float * cs_xh;
    size_t cs_xh_stride;
};

// counts: 0 dispatches, 1 distinct experts, 2 distinct misses, 3 staged, 4 miss routes, 5 unstaged misses
// (the same meanings as moe_missjoin_stage_kernel, summed over the three projections of the layer).
static __global__ void moe_hf_plan_kernel(const moe_hf_plan_args args, const int32_t * ids, int n_ids, int n_routes,
        int ids_stride, int n_expert, int slots, unsigned aligned_mask, unsigned long long * counts) {
    __shared__ int s_e[MOE_MISSJOIN_MAX_ROUTES];
    __shared__ unsigned char s_first[MOE_MISSJOIN_MAX_ROUTES];
    __shared__ unsigned char s_flag[3][MOE_MISSJOIN_MAX_ROUTES];
    __shared__ int s_stage[3][MOE_MISSJOIN_MAX_SLOTS];
    __shared__ int s_total[3];
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        s_e[i] = ids[(i / n_ids) * ids_stride + i % n_ids];
    }
    __syncthreads();
    // The duplicate removal happens once here for all three projections.
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        const int e = s_e[i];
        unsigned char first = 0;
        if (e >= 0 && e < n_expert) {
            first = 1;
            for (int j = 0; j < i; ++j) {
                if (s_e[j] == e) { first = 0; break; }
            }
        }
        s_first[i] = first;
    }
    __syncthreads();
    // bit0 distinct, bit1 miss, bit2 stageable (miss, 16-byte aligned source, copy size a multiple of 16)
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        const int e = s_e[i];
        for (int k = 0; k < 3; ++k) {
            unsigned char f = 0;
            if (s_first[i]) {
                f |= 1;
                if (!args.hot[k][e]) {
                    f |= 2;
                    if ((aligned_mask >> k & 1u) && ((uintptr_t) args.table[k][e] & 15) == 0) { f |= 4; }
                }
            }
            s_flag[k][i] = f;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        for (int k = 0; k < 3; ++k) {
            if (s_flag[k][i] == 7) {
                int rank = 0;
                for (int j = 0; j < i; ++j) { rank += s_flag[k][j] == 7; }
                if (rank < slots) { s_stage[k][rank] = s_e[i]; }
            }
        }
    }
    if (threadIdx.x < 3) {
        int stageable = 0;
        for (int i = 0; i < n_routes; ++i) { stageable += s_flag[threadIdx.x][i] == 7; }
        s_total[threadIdx.x] = stageable;
        args.nstage[threadIdx.x] = min(stageable, slots);
    }
    __syncthreads();
    for (int k = 0; k < 3; ++k) {
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
            args.eff[k][e] = args.table[k][e];
        }
    }
    __syncthreads();
    for (int k = 0; k < 3; ++k) {
        const int n_stage = min(s_total[k], slots);
        for (int r = threadIdx.x; r < n_stage; r += blockDim.x) {
            args.eff[k][s_stage[k][r]] = args.scratch + ((size_t) k * slots + r) * args.stride;
            args.src[k][r] = args.table[k][s_stage[k][r]];
        }
    }
    if (threadIdx.x == 0 && counts) {
        unsigned long long distinct = 0, dmiss = 0, mroutes = 0, staged = 0;
        for (int i = 0; i < n_routes; ++i) { distinct += s_first[i]; }
        for (int k = 0; k < 3; ++k) {
            staged += min(s_total[k], slots);
            for (int i = 0; i < n_routes; ++i) {
                dmiss += (s_flag[k][i] & 3) == 3;
                // routes (with duplicates) whose expert is not resident
                if (s_e[i] >= 0 && s_e[i] < n_expert && !args.hot[k][s_e[i]]) { ++mroutes; }
            }
        }
        atomicAdd(counts + 0, 3ULL);
        atomicAdd(counts + 1, 3 * distinct);
        atomicAdd(counts + 2, dmiss);
        atomicAdd(counts + 3, staged);
        atomicAdd(counts + 4, mroutes);
        atomicAdd(counts + 5, dmiss - staged);
    }
}

// [VPLAN] GGML_MOE_HF_PLAN_FAST=1 (default off): the same plan with fewer dependent global-memory round trips.
// moe_hf_plan_kernel waits on memory about a dozen times in a row (per-projection flag loads, the 1536-entry
// address table copied one load at a time, the stats summed by one thread); this one loads every route's three
// resident flags and addresses at once, ranks the staged experts in shared memory, and fills the address table
// with the loads batched ahead of the stores. It writes the same nstage, src, eff and stats values, so every
// later kernel reads exactly what it read before. Small batches only (<= MOE_HF_FAST_MAX_ROUTES routes,
// <= MOE_HF_FAST_MAX_EXPERT experts); anything larger takes moe_hf_plan_kernel.
#define MOE_HF_FAST_MAX_ROUTES 256
#define MOE_HF_FAST_MAX_EXPERT 1024

// fork MOEGAP PLANCOUNT: GGML_MOE_PLAN_COUNT=1 has the hit-first plan kernel (side stream) add the per-expert
// and hot/cold route counts of all three projections of its layer, instead of three moe_cache_count_routes launches
// on the main stream. Integer atomics, so the totals (and the swap policy that reads them) are the same. Default off.
static bool moe_plan_count_enabled() {
    static const bool on = [] { const char * v = getenv("GGML_MOE_PLAN_COUNT"); return v != nullptr && atoi(v) != 0; }();
    return on;
}

static bool moe_plan_count_legacy_off() {
    static const bool off = [] { const char * v = getenv("GGML_MOE_CACHE_DEVICE_FUSED"); return !(v && strcmp(v, "1") == 0); }();
    return off;
}

// True when the plan kernel already adds this projection's counts (consumes the owed bit).
static bool moe_plan_count_take(moe_cache_session & session, const moe_cache_device_table & table, const ggml_tensor * ids) {
    const int k = table.mj_kind;
    if (k < 0 || k > 2 || session.pc_layer < 0 || session.pc_layer != table.mj_layer || session.pc_ids != ids->data ||
            !(session.pc_mask >> k & 1u)) {
        return false;
    }
    session.pc_mask &= ~(1u << k);
    return true;
}

// fork MOEGAP PUBPLAN: GGML_MOE_CPU_SPLIT_PUB_PLAN (default on, =0 turns it off; only matters when GGML_MOE_CPU_SPLIT=1) moves the CPU job's publish (input rows to
// mapped host memory, then the job and its number) into the fast plan kernel, right after the CPU pick and before
// the miss lists, staging and address table. The host starts one launch (and one wait for free SM slots) earlier;
// the rows, the job and the order of the writes are those of moe_cs_publish_kernel, so the results are the same.
static bool moe_cs_pub_plan_enabled() {
    static const bool on = [] { const char * v = getenv("GGML_MOE_CPU_SPLIT_PUB_PLAN"); return v == nullptr || atoi(v) != 0; }();
    return on;
}
static bool moe_hf_plan_fast_enabled() {
    static const bool on = [] { const char * v = getenv("GGML_MOE_HF_PLAN_FAST"); return v == nullptr || atoi(v) != 0; }();
    return on;
}

static __global__ void __launch_bounds__(256) moe_hf_plan_fast_kernel(const moe_hf_plan_args args, const int32_t * ids,
        int n_ids, int n_routes, int ids_stride, int n_expert, int slots, unsigned aligned_mask, unsigned long long * counts) {
    __shared__ int s_e[MOE_HF_FAST_MAX_ROUTES];
    __shared__ unsigned char s_flag[3][MOE_HF_FAST_MAX_ROUTES];
    __shared__ const void * s_src[3][MOE_HF_FAST_MAX_ROUTES];
    __shared__ short s_slot[3][MOE_HF_FAST_MAX_EXPERT];  // expert -> staging slot, -1 when not staged
    __shared__ int s_total[3];
    __shared__ unsigned s_n[3];  // distinct experts, distinct misses, miss routes (all three projections)
    // fork MOEGAP: CPU split state
    __shared__ unsigned char s_cpu[MOE_HF_FAST_MAX_EXPERT];   // expert -> CPU slot + 1, 0 = card
    __shared__ short s_cnt[MOE_HF_FAST_MAX_ROUTES];           // at a candidate's first route: its route count
    __shared__ short s_order[MOE_HF_FAST_MAX_ROUTES];         // candidates by (routes, first route)
    __shared__ int s_nk[3];                                   // distinct misses per projection
    __shared__ int s_ncand;
    __shared__ int s_cs_first[MOE_CS_MAX_EXPERTS + 1];
    __shared__ int s_ml[4];                                   // fork MOEGAP MISSLIST: entries per list
    const int tid = threadIdx.x;
    for (int i = tid; i < n_routes; i += blockDim.x) {
        s_e[i] = ids[(i / n_ids) * ids_stride + i % n_ids];
    }
    for (int j = tid; j < 3 * n_expert; j += blockDim.x) {
        s_slot[j / n_expert][j % n_expert] = -1;
    }
    if (args.cs) {
        for (int j = tid; j < n_expert; j += blockDim.x) { s_cpu[j] = 0; }
    }
    if (tid < 3) {
        s_total[tid] = 0;
        s_n[tid] = 0;
        s_nk[tid] = 0;
    }
    if (tid == 0) { s_ncand = 0; }
    if (tid < 4) { s_ml[tid] = 0; }
    if (tid == 0 && args.pc_exp[0] && args.pc_route) { atomicAdd(args.pc_route + 2, 3ULL); }
    __syncthreads();
    // bit0 distinct, bit1 miss, bit2 stageable (as in moe_hf_plan_kernel)
    for (int i = tid; i < n_routes; i += blockDim.x) {
        const int e = s_e[i];
        const bool valid = e >= 0 && e < n_expert;
        uint8_t hot[3] = { 1, 1, 1 };
        const void * src[3] = { nullptr, nullptr, nullptr };
        if (valid) {
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                hot[k] = args.hot[k][e];
                src[k] = args.table[k][e];
            }
        }
        bool first = valid;
        for (int j = 0; j < i && first; ++j) {
            first = s_e[j] != e;
        }
        unsigned dmiss = 0, mroutes = 0;
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            unsigned char f = 0;
            if (first) {
                f |= 1;
                if (!hot[k]) {
                    f |= 2;
                    if ((aligned_mask >> k & 1u) && ((uintptr_t) src[k] & 15) == 0) { f |= 4; }
                }
            }
            s_flag[k][i] = f;
            s_src[k][i] = src[k];
            dmiss += (f & 3) == 3;
            mroutes += valid && !hot[k];
            if (args.cs && (f & 3) == 3) { atomicAdd(&s_nk[k], 1); }
            // fork MOEGAP PLANCOUNT: what moe_cache_count_routes adds for this route and projection
            if (valid && args.pc_exp[0]) {
                atomicAdd(args.pc_exp[k] + e, 1ULL);
                if (args.pc_route) { atomicAdd(args.pc_route + (hot[k] ? 0 : 1), 1ULL); }
            }
        }
        if (first) { atomicAdd(&s_n[0], 1u); }
        if (dmiss) { atomicAdd(&s_n[1], dmiss); }
        if (mroutes) { atomicAdd(&s_n[2], mroutes); }
    }
    __syncthreads();
    if (args.cs && (args.cs_partial ? (s_nk[0] || s_nk[1] || s_nk[2]) : (s_nk[0] && s_nk[1] && s_nk[2]))) {
        // fork MOEGAP: hand the host CPU the experts missing in all three projections (with PARTIAL, in any),
        // fewest routes first (the CPU pays per route, the link per expert; at equal routes, more missed
        // projections first), while that does not raise max(CPU time, max(card floor, link time)).
        for (int i = tid; i < n_routes; i += blockDim.x) {
            short c = 0;
            const int nm = ((s_flag[0][i] & 3) == 3) + ((s_flag[1][i] & 3) == 3) + ((s_flag[2][i] & 3) == 3);
            if (nm == 3 || (args.cs_partial && nm > 0)) {
                const int e = s_e[i];
                for (int j = i; j < n_routes; ++j) { c += s_e[j] == e; }
                c = (short) (c * 4 + (3 - nm));   // sort key; the route count is c / 4
            }
            s_cnt[i] = c;
        }
        __syncthreads();
        for (int i = tid; i < n_routes; i += blockDim.x) {
            const int c = s_cnt[i];
            if (c) {
                int rank = 0;
                for (int j = 0; j < n_routes; ++j) {
                    const int cj = s_cnt[j];
                    rank += cj && (cj < c || (cj == c && j < i));
                }
                s_order[rank] = (short) i;
                atomicAdd(&s_ncand, 1);
            }
        }
        __syncthreads();
        if (tid == 0) {
            moe_cs_plan_dev * plan = args.cs;
            const float w = args.cs_link_w * args.cs_ns_per_byte;
            float gpu = 0.0f;
#pragma unroll
            for (int k = 0; k < 3; ++k) { gpu += (float) s_nk[k] * args.cs_bytes[k]; }
            gpu *= w;
            const float floor_ns = args.cs_floor_ns * (float) s_n[0];
            // AIPFIX per-type: the live (ADAPT) cost is kept in the reference layer's units; cs_adapt_scale = this
            // layer's probe cost / the reference's (1 when the per-type probe is off)
            const float route_ns = args.cs_adapt && plan->route_ns_live > 0.0f ? plan->route_ns_live * args.cs_adapt_scale : args.cs_route_ns;
            const int target = moe_cs_target_parts(args.cs_share, s_nk[0] + s_nk[1] + s_nk[2]);
            const moe_cs_greedy_cfg gc = { route_ns, args.cs_fixed_ns, w, floor_ns,
                    { args.cs_bytes[0], args.cs_bytes[1], args.cs_bytes[2] }, args.cs_max_exp, args.cs_max_route, target };
            const unsigned char * const fl[3] = { s_flag[0], s_flag[1], s_flag[2] };
            int take[MOE_CS_MAX_EXPERTS];
            int nr = 0;
            unsigned long long parts = 0, partial = 0;
            const int ne = moe_cs_greedy(gc, gpu, s_ncand, s_order, s_cnt, fl, take, s_cs_first, &nr, &parts, &partial);
            for (int j = 0; j < ne; ++j) {
                plan->expert[j] = s_e[take[j]];
                s_cpu[s_e[take[j]]] = (unsigned char) (j + 1);
            }
            plan->stat_parts += parts;
            plan->stat_partial += partial;
            for (int j = 0; j <= ne; ++j) { plan->first[j] = s_cs_first[j]; }
            plan->n_exp = ne;
            plan->n_route = nr;
            plan->seq = 0;   // the publish kernel sets it when there is a job
            plan->stat_layers += 1;
            plan->stat_experts += (unsigned long long) ne;
            plan->stat_routes += (unsigned long long) nr;
        }
        __syncthreads();
    } else if (args.cs && tid == 0) {
        args.cs->n_exp = 0;
        args.cs->n_route = 0;
        args.cs->seq = 0;
        args.cs->stat_layers += 1;
    }
    if (args.cs) {
        // mask for the miss passes (and no staging for CPU experts)
        for (int i = tid; i < n_routes; i += blockDim.x) {
            const int e = s_e[i];
            if (e < 0 || e >= n_expert) { continue; }
            const int slot = s_cpu[e];
            if (args.cs_cpu) { args.cs_cpu[e] = (unsigned char) (slot != 0); }
            if (slot) {
                // the CPU computes this expert: never staged, its routes listed in route order
#pragma unroll
                for (int k = 0; k < 3; ++k) { s_flag[k][i] &= (unsigned char) ~4u; }
                int pos = s_cs_first[slot - 1];
                for (int j = 0; j < i; ++j) { pos += s_e[j] == e; }
                args.cs->route[pos] = i;
            }
            if (s_flag[0][i] & 1) {
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    args.cs_mask[k * args.cs_mask_stride + e] = (unsigned char) (!(s_flag[k][i] & 2) || slot);
                }
            }
        }
        __syncthreads();
        if (args.cs_job && args.cs->n_route > 0) {
            // fork MOEGAP PUBPLAN: moe_cs_publish_kernel's work, by this block (plan->route and n_route were
            // written by this block before the barrier above)
            moe_cs_plan_dev * plan = args.cs;
            const int n_route = plan->n_route;
            const int n_embd = args.cs_n_embd;
            for (int b = 0; b < n_route; ++b) {
                const int r = plan->route[b];
                const int t = r / args.cs_n_ids, j = r % args.cs_n_ids;
                const float * src = (const float *) (args.cs_x + (args.cs_x_per_slot ? (size_t) j * args.cs_nb1 : 0) +
                                                     (size_t) t * args.cs_nb2);
                float * d = args.cs_xh + (size_t) b * args.cs_xh_stride;
                if ((n_embd & 3) == 0 && ((uintptr_t) src & 15) == 0 && ((uintptr_t) d & 15) == 0) {
                    for (int c = tid; c < n_embd / 4; c += blockDim.x) { ((float4 *) d)[c] = ((const float4 *) src)[c]; }
                } else {
                    for (int c = tid; c < n_embd; c += blockDim.x) { d[c] = src[c]; }
                }
            }
            __threadfence_system();
            __syncthreads();
            if (tid == 0) {
                moe_cs_job * job = args.cs_job;
                const int n_exp = plan->n_exp;
                job->layer = args.cs_layer;
                job->n_exp = n_exp;
                job->n_route = n_route;
                for (int b = 0; b < n_route; ++b) {
                    job->input_row[b] = args.cs_x_per_slot ? plan->route[b] : plan->route[b] / args.cs_n_ids;
                }
                for (int i = 0; i < n_exp; ++i) { job->expert[i] = plan->expert[i]; }
                for (int i = 0; i <= n_exp; ++i) { job->first[i] = plan->first[i]; }
                __threadfence_system();
                const unsigned long long seq = plan->counter + 1;
                plan->counter = seq;
                plan->seq = seq;
                job->seq = seq;
                __threadfence_system();
            }
        }
    }
    if (args.miss) {
        // fork MOEGAP MISSLIST: the experts each miss pass computes (missed, not on the CPU), in first-route order,
        // with their route masks; the same set the masked miss passes pick (cs_mask / hot)
        for (int i = tid; i < n_routes; i += blockDim.x) {
            if (!(s_flag[0][i] & 1)) { continue; }
            const int e = s_e[i];
            const bool cpu = args.cs && s_cpu[e];
            bool need[4];
#pragma unroll
            for (int k = 0; k < 3; ++k) { need[k] = (s_flag[k][i] & 2) && !cpu; }
            need[3] = need[0] || need[1];
            if (!(need[0] || need[1] || need[2])) { continue; }
            int rank[4] = { 0, 0, 0, 0 };
            unsigned long long routes = 1ull << i;
            for (int j = 0; j < i; ++j) {
                const int ej = s_e[j];
                if (ej == e) { routes |= 1ull << j; }
                if (!(s_flag[0][j] & 1)) { continue; }
                const bool cj = args.cs && s_cpu[ej];
                const bool n0 = (s_flag[0][j] & 2) && !cj, n1 = (s_flag[1][j] & 2) && !cj;
                rank[0] += n0;
                rank[1] += n1;
                rank[2] += (s_flag[2][j] & 2) && !cj;
                rank[3] += n0 || n1;
            }
            for (int j = i + 1; j < n_routes; ++j) {
                if (s_e[j] == e) { routes |= 1ull << j; }
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (need[k]) {
                    args.miss[k].expert[rank[k]] = e;
                    args.miss[k].routes[rank[k]] = routes;
                    atomicAdd(&s_ml[k], 1);
                }
            }
        }
        __syncthreads();
        if (tid < 4) { args.miss[tid].n = s_ml[tid]; }
    }
    // staging slots in route order, as in moe_hf_plan_kernel
    for (int i = tid; i < n_routes; i += blockDim.x) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (s_flag[k][i] == 7) {
                int rank = 0;
                for (int j = 0; j < i; ++j) { rank += s_flag[k][j] == 7; }
                atomicAdd(&s_total[k], 1);
                if (rank < slots) {
                    s_slot[k][s_e[i]] = (short) rank;
                    args.src[k][rank] = s_src[k][i];
                }
            }
        }
    }
    __syncthreads();
    if (tid < 3) {
        args.nstage[tid] = min(s_total[tid], slots);
    }
    // effective addresses: the resident table, with staged experts pointing at their scratch slot; loads first
    constexpr int U = 8;
    const int total = 3 * n_expert;
    for (int base = 0; base < total; base += U * blockDim.x) {
        const void * v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = base + u * blockDim.x + tid;
            v[u] = j < total ? args.table[j / n_expert][j % n_expert] : nullptr;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = base + u * blockDim.x + tid;
            if (j < total) {
                const int k = j / n_expert, e = j % n_expert;
                const int r = s_slot[k][e];
                args.eff[k][e] = r >= 0 ? (const void *) (args.scratch + ((size_t) k * slots + r) * args.stride) : v[u];
            }
        }
    }
    if (tid == 0 && counts) {
        unsigned long long staged = 0;
        for (int k = 0; k < 3; ++k) { staged += min(s_total[k], slots); }
        const unsigned long long dmiss = s_n[1];
        atomicAdd(counts + 0, 3ULL);
        atomicAdd(counts + 1, 3ULL * s_n[0]);
        atomicAdd(counts + 2, dmiss);
        atomicAdd(counts + 3, staged);
        atomicAdd(counts + 4, (unsigned long long) s_n[2]);
        atomicAdd(counts + 5, dmiss - staged);
    }
}

// fork MOEGAP: copies the CPU routes' input rows into mapped host memory, then (last block) the job, then
// raises job->seq. One block per possible CPU route; runs on the side stream right after the plan.
static __global__ void moe_cs_publish_kernel(moe_cs_plan_dev * plan, moe_cs_job * job, const char * x, size_t nb1,
        size_t nb2, int x_per_slot, int n_ids, int n_embd, float * xh, size_t xh_stride, int layer) {
    const int n_route = plan->n_route;
    if (n_route == 0) {
        return;
    }
    const int b = blockIdx.x;
    if (b < n_route) {
        const int r = plan->route[b];
        const int t = r / n_ids, j = r % n_ids;
        const float * src = (const float *) (x + (x_per_slot ? (size_t) j * nb1 : 0) + (size_t) t * nb2);
        float * d = xh + (size_t) b * xh_stride;
        for (int c = threadIdx.x; c < n_embd; c += blockDim.x) { d[c] = src[c]; }
    }
    __threadfence_system();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) {
        last = atomicAdd(&plan->ticket, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last || threadIdx.x != 0) {
        return;
    }
    plan->ticket = 0;
    const int n_exp = plan->n_exp;
    job->layer = layer;
    job->n_exp = n_exp;
    job->n_route = n_route;
    for (int b = 0; b < n_route; ++b) { job->input_row[b] = x_per_slot ? plan->route[b] : plan->route[b] / n_ids; }
    for (int i = 0; i < n_exp; ++i) { job->expert[i] = plan->expert[i]; }
    for (int i = 0; i <= n_exp; ++i) { job->first[i] = plan->first[i]; }
    __threadfence_system();
    const unsigned long long seq = plan->counter + 1;
    plan->counter = seq;
    plan->seq = seq;
    job->seq = seq;
    __threadfence_system();
}

static __device__ __forceinline__ unsigned long long moe_cs_ld_sys(const volatile unsigned long long * p) {
    unsigned long long v;
    asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

static __device__ __forceinline__ unsigned long long moe_cs_now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// fork MOEGAP: after the down projection's miss pass, waits for the host job and writes the CPU's down rows
// into the projection output (those routes were skipped by both card passes). One block per possible CPU route.
static __global__ void moe_cs_join_kernel(moe_cs_plan_dev * plan, moe_cs_job * job, const float * out_h,
        size_t out_stride, char * dst, size_t nb1, size_t nb2, int n_ids, int n_out, float adapt_route0, float adapt_fixed,
        const int32_t * bank_ids = nullptr, size_t bank_stride = 0, float adapt_scale = 1.0f) {
    const unsigned long long seq = plan->seq;
    const int n_route = plan->n_route;
    if (seq == 0 || (int) blockIdx.x >= n_route) {
        return;
    }
    if (threadIdx.x == 0) {
        const unsigned long long t0 = moe_cs_now_ns();
        while (moe_cs_ld_sys(&job->done) < seq) {
            __nanosleep(256);
            if (moe_cs_now_ns() - t0 > 2000000000ULL) {
                if (blockIdx.x == 0) { job->gpu_timeouts = job->gpu_timeouts + 1; }
                break;
            }
        }
        if (blockIdx.x == 0) {
            plan->stat_wait_ns += moe_cs_now_ns() - t0;
            plan->stat_joins += 1;
            if (adapt_route0 > 0.0f && moe_cs_ld_sys(&job->done) >= seq) {
                // Measured cost per route of this job (the host's busy time less the job's fixed cost; the fixed
                // cost already carries the handshake allowance), folded in at 1/16 per job.
                const float busy = (float) moe_cs_ld_sys(&job->busy_ns);
                // AIPFIX per-type: adapt_scale = this layer's probe cost / the reference's; the sample is folded in
                // the reference's units (adapt_scale 1 = off: unchanged)
                const float smp = fminf(fmaxf((busy + 8000.0f - adapt_fixed) / (float) n_route / adapt_scale, 500.0f), 500000.0f);
                const float cur = plan->route_ns_live > 0.0f ? plan->route_ns_live : adapt_route0;
                plan->route_ns_live = cur + (smp - cur) * 0.0625f;
                plan->stat_adapt += 1;
            }
        }
        __threadfence_system();
    }
    __syncthreads();
    if (!dst) {
        return;   // drain only (a graph ended with the job unjoined)
    }
    const int r = plan->route[blockIdx.x];
    const int t = r / n_ids, j = r % n_ids;
    // A mixed projection already wrote bank routes in their own format.
    if (bank_ids && bank_ids[t*bank_stride + j] >= 0) {
        return;
    }
    float * d = (float *) (dst + (size_t) j * nb1 + (size_t) t * nb2);
    const float * src = out_h + (size_t) blockIdx.x * out_stride;
    for (int c = threadIdx.x; c < n_out; c += blockDim.x) { d[c] = __ldcv(src + c); }
}

// True when this dispatch takes the hit-first path. Forks the layer's plan and copies on the first
// projection of a layer; later projections only join their own event (moe_hf_wait).
// fork MOEGAP MISSLIST: run a miss pass from the plan's list when this layer has one; old_pass(dst) is the masked
// pass it replaces (run as well into a copy in check mode, then the two outputs are compared bit for bit).
template <typename F>
static bool moe_miss_pass_list(ggml_backend_cuda_context & ctx, const moe_cache_session & session, int layer, int list,
        const void * const * eff, const void * const * gate_eff, ggml_type type, const char * quantized, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels, F && old_pass) {
    if (session.ml_layer < 0 || session.ml_layer != layer ||
            !ggml_cuda_moe_table_mmv_list_supported(type, gate_eff != nullptr, n_in, n_ids, n_tokens, channels)) {
        return false;
    }
    const int64_t count = n_out * n_ids * n_tokens;
    ggml_cuda_pool_alloc<float> reference(ctx.pool());
    if (moe_miss_list_mode() == 2) {
        float * ref = reference.alloc((size_t) count);
        CUDA_CHECK(cudaMemcpyAsync(ref, dst, (size_t) count * sizeof(float), cudaMemcpyDeviceToDevice, ctx.stream()));
        old_pass(ref);
    }
    ggml_cuda_moe_table_mmv_list(ctx, g_hf_miss + list, eff, gate_eff, type, quantized, dst, n_in, n_out, n_ids,
            n_tokens, channels);
    g_st_ml.fetch_add(1, std::memory_order_relaxed);
    if (moe_miss_list_mode() == 2) {
        ggml_cuda_moe_bits_check(reference.get(), dst, count, ctx.stream());
    }
    return true;
}

// --moe-skip-tail: decide once per layer which routed experts are left out. An expert is left out when it is resident
// in none of the three projections and no token of the window ranks it below `rank` (ids rows are in descending
// router score order). Writes the masks described at g_tail_mask; one block.
#define MOE_TAIL_MAX_EXPERT 1024
static __global__ void __launch_bounds__(256) moe_tail_prep_kernel(const uint8_t * hot0, const uint8_t * hot1,
        const uint8_t * hot2, const int32_t * ids, int n_ids, int n_routes, int ids_stride, int n_expert, int rank,
        uint8_t * out, int64_t out_stride, unsigned long long * counts) {
    __shared__ unsigned char s_routed[MOE_TAIL_MAX_EXPERT];
    __shared__ unsigned char s_head[MOE_TAIL_MAX_EXPERT];
    __shared__ unsigned s_rows, s_experts;
    const int tid = threadIdx.x;
    for (int e = tid; e < n_expert; e += blockDim.x) {
        s_routed[e] = 0;
        s_head[e] = 0;
    }
    if (tid == 0) {
        s_rows = 0;
        s_experts = 0;
    }
    __syncthreads();
    for (int i = tid; i < n_routes; i += blockDim.x) {
        const int slot = i % n_ids;
        const int e = ids[(i / n_ids) * ids_stride + slot];
        if (e >= 0 && e < n_expert) {
            s_routed[e] = 1;
            if (slot < rank) {
                s_head[e] = 1;
            }
        }
    }
    __syncthreads();
    for (int e = tid; e < n_expert; e += blockDim.x) {
        const uint8_t h0 = hot0[e], h1 = hot1[e], h2 = hot2[e];
        const uint8_t skip = s_routed[e] && !s_head[e] && !h0 && !h1 && !h2;
        out[e] = h0 | skip;
        out[out_stride + e] = h1 | skip;
        out[2 * out_stride + e] = h2 | skip;
        out[3 * out_stride + e] = skip;
        if (skip) {
            atomicAdd(&s_experts, 1u);
        }
    }
    __syncthreads();
    for (int i = tid; i < n_routes; i += blockDim.x) {
        const int e = ids[(i / n_ids) * ids_stride + i % n_ids];
        if (e >= 0 && e < n_expert && s_routed[e] && !s_head[e] && !hot0[e] && !hot1[e] && !hot2[e]) {
            atomicAdd(&s_rows, 1u);
        }
    }
    for (int token = tid; counts && n_ids <= 10 && token < n_routes / n_ids; token += blockDim.x) {
        int left = 0;
        for (int j = 0; j < n_ids; ++j) {
            const int e = ids[token * ids_stride + j];
            left += e >= 0 && e < n_expert && s_routed[e] && !s_head[e] && !hot0[e] && !hot1[e] && !hot2[e];
        }
        atomicAdd(counts + 3 + left, 1ULL);
    }
    __syncthreads();
    if (tid == 0 && counts) {
        atomicAdd(counts + 0, 3ULL * s_rows);
        atomicAdd(counts + 1, 1ULL);
        atomicAdd(counts + 2, (unsigned long long) s_experts);
    }
}

// Writes zero output rows for the routes whose expert the cutoff left out (skip[e] != 0). One block per route.
static __global__ void moe_tail_zero_kernel(const uint8_t * skip, const int32_t * ids, int n_ids, int ids_stride,
        int n_expert, char * out, size_t nb1, size_t nb2, int n_out) {
    const int route = blockIdx.x;
    const int token = route / n_ids, slot = route % n_ids;
    const int e = ids[token * ids_stride + slot];
    if (e < 0 || e >= n_expert || !skip[e]) {
        return;
    }
    float * row = (float *) (out + (size_t) token * nb2 + (size_t) slot * nb1);
    for (int i = threadIdx.x; i < n_out; i += blockDim.x) {
        row[i] = 0.0f;
    }
}

// fork: sampled router weights and residency, without host registration.
struct moe_mass_record {
    uint32_t round, layer, n_used, n_tokens;
    int32_t ids[80];
    float weights[80];
    uint8_t hot[80];
};
static_assert(sizeof(moe_mass_record) == 736, "skip trace layout");

bool ggml_cuda_moe_skipmass_supports(const ggml_tensor * op) {
    int layer = -1, end = 0;
    return op && op->op == GGML_OP_MAP_CUSTOM2 && sscanf(op->name, "moe_skipmass-%d%n", &layer, &end) == 1 &&
        op->name[end] == 0 && layer >= 0 && layer < 48 && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) &&
        op->src[1] && op->src[1]->type == GGML_TYPE_I32 && op->src[1]->nb[0] == sizeof(int32_t) &&
        op->src[1]->ne[0] == 10 && op->src[1]->ne[1] >= 1 && op->src[1]->ne[1] <= 8 &&
        op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 && ggml_is_contiguous(op) &&
        ggml_are_same_shape(op, op->src[0]) && op->ne[0] == 1 && op->ne[1] == 10 &&
        op->ne[2] == op->src[1]->ne[1] && op->ne[3] == 1;
}

static __global__ void moe_mass_kernel(const float * weights, float * out_weights, const int32_t * ids,
        int stride, int width, int layer, int mode, float threshold, bool renorm, int rank_floor, int experts,
        const uint8_t * hot0, const uint8_t * hot1, const uint8_t * hot2,
        uint8_t * masks, int mask_stride, unsigned long long * counts,
        moe_mass_record * trace, unsigned * trace_counts,
        unsigned long long * swst, unsigned long long * swscore, int score_mode, unsigned skipw) {
    __shared__ unsigned char drop[80], hot[80];
    __shared__ int expert[80];
    __shared__ float scale[8];
    __shared__ bool rescale[8];
    const int tid = threadIdx.x;
    const int routes = width * 10;
    if (mode) {
        for (int e = tid; e < experts; e += blockDim.x) {
            masks[e] = hot0[e]; masks[mask_stride + e] = hot1[e];
            masks[2 * mask_stride + e] = hot2[e]; masks[3 * mask_stride + e] = 0;
        }
    }
    for (int i = tid; i < routes; i += blockDim.x) {
        drop[i] = 0;
        expert[i] = ids[(i / 10) * stride + i % 10];
        const int e = expert[i];
        hot[i] = e >= 0 && e < experts ? (hot0[e] != 0) | ((hot1[e] != 0) << 1) | ((hot2[e] != 0) << 2) : 7;
    }
    __syncthreads();
    if (trace && tid == 0) {
        const unsigned round = layer == 0 ? ++trace_counts[1] : trace_counts[1];
        if (round % 16 == 0) {
            const unsigned index = trace_counts[0]++;
            if (index < 16384) {
                auto & rec = trace[index];
                rec.round = round; rec.layer = layer; rec.n_used = 10; rec.n_tokens = width;
                for (int i = 0; i < 80; ++i) {
                    rec.ids[i] = i < routes ? expert[i] : -1;
                    rec.weights[i] = i < routes ? weights[i] : 0.0f;
                    rec.hot[i] = i < routes ? hot[i] : 7;
                }
            }
        }
    }
    for (int token = tid; token < width; token += blockDim.x) {
        const bool changed = mode && ggml_moe_skipmass_select(weights + 10 * token, hot + 10 * token, 10, mode, threshold, drop + 10 * token, rank_floor);
        rescale[token] = false;
        if (changed && renorm) {
            float total = 0.0f, kept = 0.0f;
            for (int j = 0; j < 10; ++j) {
                const int i = 10 * token + j;
                total += weights[i]; kept += drop[i] ? 0.0f : weights[i];
            }
            rescale[token] = kept > 0.0f;
            if (rescale[token]) { scale[token] = total / kept; }
        }
    }
    __syncthreads();
    // fork: all routes can write after the per-token drop decisions are visible.
    for (int i = tid; i < routes; i += blockDim.x) {
        out_weights[i] = drop[i] ? 0.0f : (rescale[i / 10] ? weights[i] * scale[i / 10] : weights[i]);
    }
    // fork SWAPPOL: statistics and swap scores from the drop decisions above (integer atomics: order-independent).
    for (int i = tid; (swst || swscore) && i < routes; i += blockDim.x) {
        const int e = expert[i];
        if (e < 0 || e >= experts) { continue; }
        const unsigned long long wfx = (unsigned long long) (fminf(fmaxf(weights[i], 0.0f), 1.0f) * 10240.0f + 0.5f);
        if (swst) {
            unsigned long long * row = swst + ((size_t) layer * experts + e) * MOE_SWST_COLS;
            atomicAdd(row + i % 10, 1ULL);
            atomicAdd(row + 10, wfx);
            if (drop[i]) { atomicAdd(row + 11, 1ULL); }
            if (hot[i] == 7) { atomicAdd(row + 12, 1ULL); }
            bool first = true;
            for (int j = 0; j < i; ++j) { first &= expert[j] != e; }
            if (first) { atomicAdd(row + 13, 1ULL); }
        }
        if (swscore) {
            const unsigned long long v = (score_mode & 2 ? wfx : 1024ULL) * (drop[i] ? skipw : 1024u) / 1024u;
            if (v) { atomicAdd(swscore + (size_t) layer * experts + e, v); }
        }
    }
    if (!mode) { return; }
    for (int token = tid; counts && token < width; token += blockDim.x) {
        int left = 0;
        for (int j = 0; j < 10; ++j) { left += drop[10 * token + j]; }
        atomicAdd(counts + 3 + left, 1ULL);
    }
    // fork: only the first dropped occurrence can omit an expert for the window.
    for (int i = tid; i < routes; i += blockDim.x) {
        if (hot[i] || !drop[i]) { continue; }
        const int e = expert[i];
        unsigned long long rows = 0;
        bool skip = true;
        for (int j = 0; j < routes; ++j) {
            if (expert[j] != e) { continue; }
            if (j < i || !drop[j]) { skip = false; break; }
            ++rows;
        }
        if (!skip) { continue; }
        masks[e] = masks[mask_stride + e] = masks[2 * mask_stride + e] = masks[3 * mask_stride + e] = 1;
        if (counts) {
            atomicAdd(counts + 2, 1ULL);
            atomicAdd(counts, 3ULL * rows);
        }
    }
    if (counts && tid == 0) { atomicAdd(counts + 1, 1ULL); }
}

bool ggml_cuda_moe_skipmass_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    GGML_ASSERT(ggml_cuda_moe_skipmass_supports(op));
    auto * active_session = !g_session_suppressed && !g_session_stack.empty() ? g_session_stack.back().active : nullptr;
    // Warmup and cache-disabled evaluation have no device residency to consult. Keep the original weights.
    if (!active_session || !active_session->device_ready.load(std::memory_order_acquire)) {
        if (active_session) { active_session->mass_layer = -1; active_session->mass_ids = nullptr; }
        CUDA_CHECK(cudaMemcpyAsync(op->data, op->src[0]->data, ggml_nbytes(op), cudaMemcpyDeviceToDevice, ctx.stream()));
        return true;
    }
    auto & session = *active_session;
    GGML_ASSERT(session.device_ready.load() && session.devices.size() == 1 && session.devices[0]->logical == ctx.device);
    const int mode = session.skip_mode;
    GGML_ASSERT(mode >= 0);
    const float threshold = session.skip_threshold;
    GGML_ASSERT(!mode || ggml_moe_skipmass_budget_valid(mode, threshold));
    const bool active = mode && session.skip_mass;
    if (!active && !session.mass_trace && !g_swst && !g_swscore) {
        // fork: exact evaluation needs no route selection or residency reads.
        CUDA_CHECK(cudaMemcpyAsync(op->data, op->src[0]->data, ggml_nbytes(op), cudaMemcpyDeviceToDevice, ctx.stream()));
        session.mass_layer = -1;
        session.mass_ids = nullptr;
        return true;
    }
    const int layer = atoi(op->name + strlen("moe_skipmass-"));
    GGML_ASSERT(layer < (int) session.mj_groups.size());
    const auto & group = session.mj_groups[layer];
    GGML_ASSERT(group[0] && group[1] && group[2] && g_tail_mask && moe_missjoin_hitfirst_enabled());
    const bool renorm = session.skip_renorm;
    // fork AIPFIX SKIPBANK: a layer with AI+ bank experts may use its own floor / budget
    const bool bank_layer = group[0]->n_bank > 0 || group[1]->n_bank > 0 || group[2]->n_bank > 0;
    const int rank_floor = bank_layer && session.skip_bank_floor >= 0 ? session.skip_bank_floor : session.skip_rank_floor;
    const float layer_threshold = bank_layer && session.skip_bank_threshold >= 0.0f ? session.skip_bank_threshold : threshold;
    const int experts = (int) group[0]->desc.n_expert;
    GGML_ASSERT(experts <= g_hf_experts && experts <= MOE_TAIL_MAX_EXPERT);
    moe_mass_kernel<<<1, 256, 0, ctx.stream()>>>((const float *) op->src[0]->data, (float *) op->data,
        (const int32_t *) op->src[1]->data, (int) (op->src[1]->nb[1] / sizeof(int32_t)), (int) op->ne[2],
        layer, active ? mode : 0, layer_threshold, renorm, rank_floor, experts, group[0]->hot, group[1]->hot, group[2]->hot,
        g_tail_mask, g_hf_experts, g_tail_counts, (moe_mass_record *) session.mass_trace, session.mass_trace_counts,
        experts == g_hf_experts ? g_swst : nullptr, experts == g_hf_experts ? g_swscore : nullptr, moe_swap_score_mode(), moe_swap_skipw());
    CUDA_CHECK(cudaGetLastError());
    session.mass_layer = active ? layer : -1;
    session.mass_ids = op->src[1]->data;
    return true;
}

// True when the --moe-skip-tail masks of this projection's layer are live (written by this layer's moe_hf_begin).
static bool moe_tail_active(const moe_cache_session & session, const moe_cache_device_table & table) {
    return session.tail_layer >= 0 && session.tail_layer == table.mj_layer;
}

static void moe_tail_zero(ggml_backend_cuda_context & ctx, const ggml_tensor * ids, int n_expert, ggml_tensor * out) {
    GGML_ASSERT(out->type == GGML_TYPE_F32 && out->nb[0] == sizeof(float));
    const int routes = (int) (ids->ne[0] * ids->ne[1]);
    moe_tail_zero_kernel<<<routes, 128, 0, ctx.stream()>>>(g_tail_mask + 3 * (size_t) g_hf_experts,
            (const int32_t *) ids->data, (int) ids->ne[0], (int) (ids->nb[1] / sizeof(int32_t)), n_expert,
            (char *) out->data, out->nb[1], out->nb[2], (int) out->ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

// fork AIPFIX D3: a projection can run the hit/miss passes when its filtered table GEMV takes this many tokens
// (MMVQ types: the per-type mmid batch limit; AI+ unit experts: the AI+ direct GEMV over the slot table).
static bool moe_hf_type_ok(const moe_cache_device_table & t, int64_t n_tok, int cc) {
    const ggml_type type = (ggml_type) t.desc.type;
    if (ggml_ai4_is_unit(type)) {
        return ggml_cuda_moe_ai_table_supported(type, t.desc.n_in, t.desc.n_out, n_tok);
    }
    return ggml_cuda_moe_table_mmv_filtered_supported(type) && n_tok <= get_mmvq_mmid_max_batch(type, cc);
}

static bool moe_hf_begin(ggml_backend_cuda_context & ctx, moe_cache_session & session,
        const moe_cache_device_table & table, const ggml_tensor * a, const ggml_tensor * ids) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!moe_missjoin_hitfirst_enabled() || session.mj_groups.empty() || table.staging_device || !table.hot ||
            table.mj_layer < 0 || table.mj_layer >= (int) session.mj_groups.size() || !g_hf_scratch ||
            ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 ||
            !moe_hf_type_ok(table, ids->ne[1], cc)) {
        return false;
    }
    const auto & group = session.mj_groups[table.mj_layer];
    if (!group[0] || !group[1] || !group[2]) {
        return false;
    }
    const int routes = (int)(ids->ne[0] * ids->ne[1]);
    if (routes < 1 || routes > MOE_MISSJOIN_MAX_ROUTES || ids->ne[2] != 1) {
        return false;
    }
    // fork PARFIX: the layer takes this path only when all three projections fit the card's table path for this
    // many tokens (the same limit moe_cache_device_op_fits applies per weight type: on Ampere IQ3_S stops at 6,
    // IQ3_XXS/IQ2_S at 7, the rest at 8). With --parallel 2 a 6-8 token verify batch could otherwise start a layer
    // whose down (or gate) projection then runs on the CPU, leaving that projection's side copy unjoined across
    // the graph split. One sequence (<= 5 tokens) always fits every type, so this never declines there.
    for (int j = 0; j < 3; ++j) {
        if (!moe_hf_type_ok(*group[j], ids->ne[1], cc) || group[j]->staging_device ||
                group[j]->desc.n_expert != group[0]->desc.n_expert) {
            return false;
        }
    }
    const int k = table.mj_kind;
    cudaStream_t main = ctx.stream();
    if (session.mj_group == table.mj_layer && session.mj_ids == ids->data && (session.mj_pending >> k & 1u)) {
        // Same layer, copy already issued by its first projection.
        session.hf_seq.push_back((int8_t) k);
        if (session.hf_seq.size() == 3) {
            std::array<int8_t, 3> order{session.hf_seq[0], session.hf_seq[1], session.hf_seq[2]};
            if (order[0] != order[1] && order[0] != order[2] && order[1] != order[2]) {
                session.hf_order[table.mj_layer] = order;
            }
        }
        return true;
    }
    // New layer: join copies a previous group left unconsumed, then fork.
    for (int j = 0; j < 3; ++j) {
        if (session.mj_pending >> j & 1u) {
            CUDA_CHECK(cudaStreamWaitEvent(main, g_hf_done[j], 0));
        }
    }
    CUDA_CHECK(cudaEventRecord(g_hf_fork, main));
    CUDA_CHECK(cudaStreamWaitEvent(g_hf_side, g_hf_fork, 0));
    const int slots = g_hf_slots;
    moe_hf_plan_args args = {};
    unsigned aligned = 0;
    for (int j = 0; j < 3; ++j) {
        const auto & m = *group[j];
        args.table[j] = m.device;
        args.hot[j] = m.hot;
        args.eff[j] = g_hf_eff + (size_t) j * g_hf_experts;
        args.src[j] = g_hf_src + (size_t) j * slots;
        aligned |= (moe_missjoin_copy_bytes(m) % 16 == 0 ? 1u : 0u) << j;
    }
    args.nstage = g_hf_nstage;
    args.scratch = g_hf_scratch;
    args.stride = g_hf_stride;
    const int n_expert = (int) group[0]->desc.n_expert;
    // --moe-skip-tail: the cutoff masks go first on the side stream; the plan then sees left-out experts as resident.
    const bool mass = session.skip_mass && session.skip_mode != 0 &&
            session.mass_layer == table.mj_layer && session.mass_ids == ids->data;
    const bool tail = (mass || (session.skip_tail > 0 && session.skip_mode == 0)) && g_tail_mask && n_expert <= MOE_TAIL_MAX_EXPERT &&
            n_expert <= g_hf_experts && (mass || session.skip_tail < (int) ids->ne[0]);
    session.tail_layer = tail ? table.mj_layer : -1;
    if (tail && !mass) {
        moe_tail_prep_kernel<<<1, 256, 0, g_hf_side>>>(group[0]->hot, group[1]->hot, group[2]->hot,
                (const int32_t *) ids->data, (int) ids->ne[0], routes, (int) (ids->nb[1] / sizeof(int32_t)), n_expert,
                session.skip_tail, g_tail_mask, g_hf_experts, g_tail_counts);
        CUDA_CHECK(cudaGetLastError());
    }
    if (tail) {
        for (int j = 0; j < 3; ++j) {
            args.hot[j] = g_tail_mask + (size_t) j * g_hf_experts;
        }
    }
    const bool fast = moe_hf_plan_fast_enabled() && routes <= MOE_HF_FAST_MAX_ROUTES && n_expert <= MOE_HF_FAST_MAX_EXPERT;
    // fork MOEGAP: CPU split for this layer (decided here, at capture time; the plan picks the experts)
    const int L = table.mj_layer;
    const bool cs = g_cs_on && fast && k != 2 && L < (int) session.cs_registered.size() && session.cs_registered[L] &&
            session.cs_valid[L] && a && a->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float) &&
            a->ne[0] == group[0]->desc.n_in && a->ne[0] <= (int64_t) g_cs_x_stride &&
            (a->ne[1] == 1 || a->ne[1] == ids->ne[0]) && a->ne[2] == ids->ne[1] &&
            group[2]->desc.n_out <= (int64_t) g_cs_out_stride;
    session.cs_layer = cs ? L : -1;
    // fork MOEGAP MISSLIST
    const int ml_mode = moe_miss_list_mode();
    const bool ml = ml_mode != 0 && (ml_mode != 3 || !ggml_moe_bank_lookup(group[0]->desc.data)) &&
            fast && routes <= MOE_MISS_LIST_MAX && g_hf_miss;
    session.ml_layer = ml ? L : -1;
    if (ml) { args.miss = g_hf_miss; }
    session.pc_layer = -1;
    session.pc_mask = 0;
    bool pc_tp = false;   // the table-prefetch dispatch adds its own counts
#if !defined(GGML_USE_HIP)
    pc_tp = session.devices.empty() || session.devices[0]->table_prefetch_prepared;
#endif
    if (fast && !pc_tp && moe_plan_count_enabled() && moe_plan_count_legacy_off() && session.device_expert_counts) {
        for (int j = 0; j < 3; ++j) { args.pc_exp[j] = session.device_expert_counts + group[j]->count_offset; }
        args.pc_route = session.device_route_counts;
        session.pc_layer = L;
        session.pc_ids = ids->data;
        session.pc_mask = 7u;
        g_st_pc.fetch_add(1, std::memory_order_relaxed);
    }
    if (cs) {
        args.cs = g_cs_plan;
        args.cs_mask = g_cs_mask;
        args.cs_mask_stride = g_hf_experts;
        args.cs_cpu = moe_down_one_mode() != 0 ? g_cs_mask + 3 * (size_t) g_hf_experts : nullptr;
        args.cs_route_ns = g_cs_route_ns;
        args.cs_adapt = g_cs_adapt;
        args.cs_fixed_ns = g_cs_fixed_ns;
        args.cs_adapt_scale = 1.0f;
        if (L < (int) session.cs_route_l.size() && session.cs_route_l[L] > 0.0f) {   // AIPFIX per-type probe
            args.cs_route_ns = session.cs_route_l[L];
            args.cs_fixed_ns = session.cs_fixed_l[L];
            args.cs_adapt_scale = g_cs_route_ns > 0.0f ? session.cs_route_l[L] / g_cs_route_ns : 1.0f;
        }
        args.cs_ns_per_byte = g_cs_ns_per_byte;
        for (int j = 0; j < 3; ++j) { args.cs_bytes[j] = (float) moe_missjoin_copy_bytes(*group[j]); }
        args.cs_max_exp = g_cs_max_exp;
        args.cs_max_route = g_cs_max_route;
        args.cs_partial = g_cs_partial;
        args.cs_floor_ns = g_cs_floor_ns;
        args.cs_link_w = g_cs_link_w;
        args.cs_share = g_cs_share;
    }
    // fork MOEGAP PUBPLAN
    const bool pub_plan = cs && fast && moe_cs_pub_plan_enabled();
    if (pub_plan) {
        args.cs_job = g_cs_job_d;
        args.cs_x = (const char *) a->data;
        args.cs_nb1 = a->nb[1];
        args.cs_nb2 = a->nb[2];
        args.cs_x_per_slot = a->ne[1] > 1 ? 1 : 0;
        args.cs_n_ids = (int) ids->ne[0];
        args.cs_n_embd = (int) a->ne[0];
        args.cs_layer = L;
        args.cs_xh = g_cs_x_d;
        args.cs_xh_stride = g_cs_x_stride;
        g_st_pub_plan.fetch_add(1, std::memory_order_relaxed);
    }
    if (fast) {
        moe_hf_plan_fast_kernel<<<1, 256, 0, g_hf_side>>>(args, (const int32_t *) ids->data, (int) ids->ne[0], routes,
                (int)(ids->nb[1] / sizeof(int32_t)), n_expert, slots, aligned, g_missjoin_counts);
    } else {
        moe_hf_plan_kernel<<<1, 256, 0, g_hf_side>>>(args, (const int32_t *) ids->data, (int) ids->ne[0], routes,
                (int)(ids->nb[1] / sizeof(int32_t)), n_expert, slots, aligned, g_missjoin_counts);
    }
    CUDA_CHECK(cudaGetLastError());
    if (cs && !pub_plan) {
        moe_cs_publish_kernel<<<g_cs_max_route, 128, 0, g_hf_side>>>(g_cs_plan, g_cs_job_d, (const char *) a->data,
                a->nb[1], a->nb[2], a->ne[1] > 1 ? 1 : 0, (int) ids->ne[0], (int) a->ne[0], g_cs_x_d, g_cs_x_stride, L);
        CUDA_CHECK(cudaGetLastError());
    }
    if (cs) {
        g_cs_layers_published.fetch_add(1, std::memory_order_relaxed);
    }
    std::array<int8_t, 3> order = session.hf_order[table.mj_layer];
    if (order[0] < 0) { order = {0, 1, 2}; }
    static const bool copy_rows = [] {
        const char * v = getenv("GGML_MOE_MISSJOIN_HF_COPY_ROWS");
        return v && atoi(v) != 0;
    }();
    for (int idx = 0; idx < 3; ++idx) {
        const int j = order[idx];
        if (copy_rows) {
            moe_hf_copy_kernel<true><<<moe_hf_blocks(), 256, 0, g_hf_side>>>(args.src[j], g_hf_nstage + j,
                    g_hf_scratch + (size_t) j * slots * g_hf_stride, g_hf_stride, moe_missjoin_copy_bytes(*group[j]));
        } else {
            moe_hf_copy_kernel<false><<<moe_hf_blocks(), 256, 0, g_hf_side>>>(args.src[j], g_hf_nstage + j,
                    g_hf_scratch + (size_t) j * slots * g_hf_stride, g_hf_stride, moe_missjoin_copy_bytes(*group[j]));
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(g_hf_done[j], g_hf_side));
    }
    session.mj_group = table.mj_layer;
    session.mj_ids = ids->data;
    session.mj_pending = 7u;
    session.hf_seq.assign(1, (int8_t) k);
    g_hf_layers.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Joins this projection's copy and returns its table of effective expert addresses.
static const void * const * moe_hf_wait(ggml_backend_cuda_context & ctx, moe_cache_session & session,
        const moe_cache_device_table & table) {
    const int k = table.mj_kind;
    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), g_hf_done[k], 0));
    session.mj_pending &= ~(1u << k);
    return g_hf_eff + (size_t) k * g_hf_experts;
}

// fork MOEGAP: checks, per layer, that the graph is the plain expert block the CPU split reproduces:
// down = mul_mat_id(down, swiglu(mul_mat_id(gate, x, ids), mul_mat_id(up, x, ids)), ids), with up, gate and the
// SwiGLU each read only by the next step (the CPU routes leave those rows unwritten; only down rows are joined).
// Layers this graph does not touch keep their earlier verdict.
void ggml_cuda_moe_cache_cs_graph_scan(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph) {
    if (!g_cs_on || g_session_suppressed || g_session_stack.empty() || !g_session_stack.back().active) return;
    auto & session = *g_session_stack.back().active;
    if (!session.device_ready.load() || session.devices.size() != 1 || session.devices[0]->logical != ctx.device) return;
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph *>(cgraph));
    auto table_of = [&](const ggml_tensor * t) -> const moe_cache_device_table * {
        if (!t || t->op != GGML_OP_MUL_MAT_ID || !t->src[0]) return nullptr;
        const auto it = session.device_tables.find(t->src[0]->data);
        return it == session.device_tables.end() ? nullptr : &it->second;
    };
    std::vector<int8_t> seen(session.cs_valid.size(), 0), good(session.cs_valid.size(), 0);
    std::vector<const ggml_tensor *> watch;   // up, gate, glu of the candidate layers
    std::vector<int> watch_layer, watch_uses;
    for (int n = 0; n < n_nodes; ++n) {
        const ggml_tensor * d = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), n);
        const auto * td = table_of(d);
        if (!td || td->mj_layer < 0 || td->mj_layer >= (int) seen.size()) continue;
        const int L = td->mj_layer;
        seen[L] = 1;
        if (td->mj_kind != 2) continue;
        const ggml_tensor * g = d->src[1];
        bool ok = d->type == GGML_TYPE_F32 && g && g->op == GGML_OP_GLU && ggml_get_glu_op(g) == GGML_GLU_OP_SWIGLU &&
                ggml_get_op_params_i32(g, 1) == 0 && g->src[0] && g->src[1] && g->type == GGML_TYPE_F32;
        const ggml_tensor * gate = ok ? g->src[0] : nullptr;
        const ggml_tensor * up = ok ? g->src[1] : nullptr;
        const ggml_tensor * gate_out = gate, * up_out = up;
        const ggml_tensor * mixed_gate = ggml_cuda_moe_mixed_base(gate);
        const ggml_tensor * mixed_up = ggml_cuda_moe_mixed_base(up);
        const bool mixed = mixed_gate && mixed_up;
        if (mixed) { gate = mixed_gate; up = mixed_up; }
        const auto * tg = table_of(gate);
        const auto * tu = table_of(up);
        ok = ok && tg && tu && tg->mj_layer == L && tu->mj_layer == L && tg->mj_kind == 1 && tu->mj_kind == 0 &&
                up->src[1] == gate->src[1] && up->src[2] == gate->src[2] && d->src[2] == up->src[2] &&
                ggml_are_same_shape(up, gate) && ggml_are_same_shape(up, g) && ggml_is_contiguous(g);
        if (ok) {
            good[L] = 1;
            watch.push_back(up_out); watch_layer.push_back(L); watch_uses.push_back(1);
            watch.push_back(gate_out); watch_layer.push_back(L); watch_uses.push_back(1);
            watch.push_back(g); watch_layer.push_back(L); watch_uses.push_back(mixed ? 2 : 1);
        }
    }
    if (!watch.empty()) {
        std::vector<int> uses(watch.size(), 0);
        std::unordered_map<const ggml_tensor *, size_t> index;
        for (size_t w = 0; w < watch.size(); ++w) { index.emplace(watch[w], w); }
        for (int n = 0; n < n_nodes; ++n) {
            const ggml_tensor * t = ggml_graph_node(const_cast<ggml_cgraph *>(cgraph), n);
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (!t->src[s]) continue;
                const auto it = index.find(t->src[s]);
                if (it != index.end()) { ++uses[it->second]; }
            }
            // a view of a watched tensor is another reader
            if (t->view_src) {
                const auto it = index.find(t->view_src);
                if (it != index.end()) { uses[it->second] += 2; }
            }
        }
        for (size_t w = 0; w < watch.size(); ++w) {
            if (uses[w] != watch_uses[w]) { good[watch_layer[w]] = 0; }
        }
    }
    size_t valid = 0;
    for (size_t L = 0; L < seen.size(); ++L) {
        if (seen[L]) { session.cs_valid[L] = good[L]; }
        valid += session.cs_valid[L] != 0;
    }
    static size_t logged = SIZE_MAX;
    if (valid != logged) {
        logged = valid;
        MOE_CACHE_LOG("[moe-cache] cpu split: graph check, %zu layers eligible\n", valid);
    }
}

// fork PARFIX: called at the end of every eager or captured evaluation of a CUDA graph (one scheduler split),
// before cudaStreamEndCapture. A layer whose projections did not all take the miss-join path (for example one
// projection fell back to another path) leaves side-stream copies whose events nothing in this graph joins.
// Joining them here keeps a capture closed (no unjoined side work) and keeps any later cudaStreamWaitEvent from
// waiting on an event recorded by another graph or capture, which fails with "invalid argument". Nothing is
// pending when every layer consumed its three projections, the normal case, and then this does nothing.
void ggml_cuda_moe_cache_hf_graph_end(ggml_backend_cuda_context & ctx) {
    if (g_session_suppressed || g_session_stack.empty() || !g_session_stack.back().active) return;
    auto & session = *g_session_stack.back().active;
    if (session.devices.size() != 1 || session.devices[0]->logical != ctx.device) return;
    if (session.cs_layer >= 0) {
        // fork MOEGAP: a CPU-split layer whose down projection never ran here; wait for the host job so the
        // next layer cannot overwrite its rows. Its CPU routes have no output: that would be a wrong result.
        moe_cs_join_kernel<<<g_cs_max_route, 32, 0, ctx.stream()>>>(g_cs_plan, g_cs_job_d, g_cs_out_d, g_cs_out_stride,
                nullptr, 0, 0, 1, 0, 0.0f, 0.0f);
        CUDA_CHECK(cudaGetLastError());
        session.cs_layer = -1;
        MOE_CACHE_LOG("[moe-cache] cpu split: WARNING a graph ended with a CPU job unjoined (results are wrong)\n");
    }
    if (!session.mj_pending) return;
    cudaEvent_t * done = moe_missjoin_hitfirst_enabled() ? g_hf_done : g_mj_done;
    for (int j = 0; j < 3; ++j) {
        if ((session.mj_pending >> j & 1u) && done[j]) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), done[j], 0));
        }
    }
    session.mj_pending = 0;
    session.pc_mask = 0;
    session.mj_group = -1;
    session.mj_ids = nullptr;
    session.hf_seq.clear();
    if (g_hf_graph_end_joins.fetch_add(1, std::memory_order_relaxed) == 0) {
        MOE_CACHE_LOG("[moe-cache] missjoin: joined side copies left pending at the end of a graph\n");
    }
}

// ---------------------------------------------------------------------------------------------
// fork PREFETCHFIX: asynchronous look-ahead prefetch. See moe_pf_state for the overview.
//
// Ordering argument (single compute stream, graph order): predict(T) runs before every
// dispatch of layer T and after every dispatch of layers <= T - sets. A slot of set T % sets is
// therefore never rewritten while a dispatch that matched it can still read it. Tags carry
// the stamp of their prediction; the down dispatch bumps the stamp, so a tag is usable only
// between its own prediction and the end of that layer, and late copies are simply ignored.
// Slot bytes are exact copies of the immutable host expert bytes, so results are bit-identical.

static constexpr unsigned long long moe_pf_tag(unsigned stamp, int layer, unsigned mask, int expert) {
    return ((unsigned long long) stamp << 32) | ((unsigned long long) (layer & 0x3ff) << 22) |
           ((unsigned long long) (mask & 7) << 19) | (unsigned long long) (expert & 0x7ffff);
}

// stats: 0 accuracy dispatches, 1 distinct actual, 2 actual in prediction, 3 distinct misses,
// 4 misses in prediction, 5 misses served by prefetch, 6 miss routes served, 7 resolve calls,
// 8 predictions, 9 candidates, 10 cold candidates, 11 cold projections, 12 tags matched,
// 13 tagged but already resident
static __global__ void moe_pf_predict_kernel(const int32_t * ids, int k_ids, int layer, int n_expert,
        const uint8_t * const * hot, unsigned * stamp, unsigned * pred_stamp, int32_t * pred,
        moe_pf_entry * ring, unsigned * head, int write_ring, unsigned long long * stats) {
    __shared__ int s_e[MOE_PF_MAX_K];
    __shared__ unsigned char s_m[MOE_PF_MAX_K];
    __shared__ unsigned s_stamp;
    const int i = threadIdx.x;
    int e = -1;
    unsigned m = 0;
    if (i < k_ids) {
        e = ids[i];
        if (e < 0 || e >= n_expert) {
            e = -1;
        }
    }
    if (e >= 0) {
        for (int k = 0; k < 3; ++k) {
            const uint8_t * h = hot[layer * 3 + k];
            if (h && !h[e]) {
                m |= 1u << k;
            }
        }
    }
    if (i < MOE_PF_MAX_K) {
        s_e[i] = e;
        s_m[i] = (unsigned char) m;
        pred[(size_t) layer * MOE_PF_MAX_K + i] = e;
    }
    if (i == 0) {
        s_stamp = stamp[layer] + 1;
        stamp[layer] = s_stamp;
        pred_stamp[layer] = s_stamp;
    }
    __syncthreads();
    if (i != 0) {
        return;
    }
    int n = 0;
    unsigned long long projections = 0;
    unsigned pos = 0;
    moe_pf_entry * entry = nullptr;
    if (write_ring) {
        pos = atomicAdd(head, 1u);
        entry = ring + pos % MOE_PF_RING;
        *(volatile unsigned *) &entry->seq = 0;
        __threadfence_system();
    }
    for (int j = 0; j < k_ids && j < MOE_PF_MAX_K; ++j) {
        if (s_m[j]) {
            if (entry) {
                entry->e[n] = s_e[j];
                entry->mask[n] = s_m[j];
            }
            projections += __popc(s_m[j]);
            ++n;
        }
    }
    if (entry) {
        entry->layer = layer;
        entry->stamp = s_stamp;
        entry->n = n;
        __threadfence_system();
        *(volatile unsigned *) &entry->seq = pos + 1;
    }
    if (stats) {
        atomicAdd(stats + 8, 1ULL);
        atomicAdd(stats + 9, (unsigned long long) min(k_ids, MOE_PF_MAX_K));
        atomicAdd(stats + 10, (unsigned long long) n);
        atomicAdd(stats + 11, projections);
    }
}

static __global__ void moe_pf_mark_end_kernel(const unsigned * mark_seq, unsigned * mark) {
    __threadfence();
    *(volatile unsigned *) (mark + 1) = mark_seq[1];
}

static __global__ void moe_pf_tag_kernel(unsigned long long * tag, unsigned long long value) {
#if defined(GGML_USE_HIP)
    __threadfence();
    atomicExch(tag, value);
#else
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(*tag).store(value, cuda::memory_order_release);
#endif
}

// One block: copies the dispatch's pointer table and resident mask, then marks tagged
// prefetched experts resident and points them at their slot. Stage and matrix-vector
// kernels read the copies, so every block of the stage kernel sees the same decision.
static __global__ void moe_pf_resolve_kernel(const void * const * table, const uint8_t * hot,
        const int32_t * ids, int n_ids, int n_routes, int ids_stride, int n_expert, int layer, int kind,
        unsigned * stamp, const unsigned * pred_stamp, const int32_t * pred, int k_pred,
        unsigned long long * tags, int kpf, char * slots, size_t slot_stride, size_t kind_off,
        const void ** out_table, uint8_t * out_hot, unsigned long long * stats, unsigned * mark, unsigned * mark_seq) {
    __shared__ int s_slot_e[MOE_PF_MAX_SLOTS];
    __shared__ int s_route_e[MOE_MISSJOIN_MAX_ROUTES];
    __shared__ unsigned s_stamp;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        out_table[e] = table[e];
        out_hot[e] = hot[e];
    }
    for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
        s_route_e[i] = ids[(i / n_ids) * ids_stride + i % n_ids];
    }
    if (threadIdx.x == 0) {
        s_stamp = stamp[layer];
    }
    __syncthreads();
    const unsigned cur = s_stamp;
    if (threadIdx.x < kpf) {
        const int j = threadIdx.x;
#if defined(GGML_USE_HIP)
        const unsigned long long t = atomicAdd(tags + j, 0ULL);
        __threadfence();
#else
        const unsigned long long t = cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>(tags[j]).load(cuda::memory_order_acquire);
#endif
        int e = -1;
        if (t != 0 && (unsigned) (t >> 32) == cur && (int) ((t >> 22) & 0x3ff) == layer && ((t >> 19) >> kind & 1)) {
            e = (int) (t & 0x7ffff);
            if (e >= n_expert) {
                e = -1;
            }
        }
        s_slot_e[j] = e;
        if (e >= 0 && !hot[e]) {
            out_table[e] = slots + (size_t) j * slot_stride + kind_off;
            out_hot[e] = 1;
        }
    }
    __shared__ unsigned s_acc[8];
    if (threadIdx.x < 8) {
        s_acc[threadIdx.x] = 0;
    }
    __syncthreads();
    if (stats) {
        const bool valid = pred_stamp[layer] == cur;
        for (int i = threadIdx.x; i < n_routes; i += blockDim.x) {
            const int e = s_route_e[i];
            if (e < 0 || e >= n_expert) {
                continue;
            }
            bool slot = false;
            for (int j = 0; j < kpf; ++j) {
                slot |= s_slot_e[j] == e;
            }
            const bool miss = !hot[e];
            if (miss && slot) {
                atomicAdd(&s_acc[6], 1u);
            }
            bool first = true;
            for (int j = 0; j < i; ++j) {
                if (s_route_e[j] == e) { first = false; break; }
            }
            if (!first) {
                continue;
            }
            bool predicted = false;
            if (valid) {
                for (int j = 0; j < k_pred; ++j) {
                    predicted |= pred[(size_t) layer * MOE_PF_MAX_K + j] == e;
                }
                atomicAdd(&s_acc[1], 1u);
                if (predicted) { atomicAdd(&s_acc[2], 1u); }
                if (miss) { atomicAdd(&s_acc[3], 1u); }
                if (miss && predicted) { atomicAdd(&s_acc[4], 1u); }
            }
            if (miss && slot) {
                atomicAdd(&s_acc[5], 1u);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) {
        return;
    }
    if (kind == 2) {
        // The down projection is the layer's last reader: later copies for this stamp are dropped.
        stamp[layer] = cur + 1;
    }
    if (mark) {
        bool demand = false;
        for (int i = 0; i < n_routes && !demand; ++i) {
            const int e = s_route_e[i];
            demand = e >= 0 && e < n_expert && !out_hot[e];
        }
        if (demand) {
            const unsigned seq = atomicAdd(mark_seq, 1u) + 1;
            mark_seq[1] = seq;
            *(volatile unsigned *) mark = seq;
        }
    }
    if (!stats) {
        return;
    }
    unsigned long long matched = 0, resident = 0;
    for (int j = 0; j < kpf; ++j) {
        if (s_slot_e[j] >= 0) {
            ++matched;
            resident += hot[s_slot_e[j]] != 0;
        }
    }
    atomicAdd(stats + 7, 1ULL);
    atomicAdd(stats + 12, matched);
    atomicAdd(stats + 13, resident);
    if (pred_stamp[layer] == cur) {
        if (kind == 0) {
            atomicAdd(stats + 0, 1ULL);
            atomicAdd(stats + 1, (unsigned long long) s_acc[1]);
            atomicAdd(stats + 2, (unsigned long long) s_acc[2]);
        }
        atomicAdd(stats + 3, (unsigned long long) s_acc[3]);
        atomicAdd(stats + 4, (unsigned long long) s_acc[4]);
    }
    atomicAdd(stats + 5, (unsigned long long) s_acc[5]);
    atomicAdd(stats + 6, (unsigned long long) s_acc[6]);
}

static void moe_pf_log(const char * when) {
    if (!g_pf.d_stats) {
        return;
    }
    unsigned long long c[MOE_PF_STATS] = {};
    if (cudaMemcpy(c, g_pf.d_stats, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    auto pct = [](unsigned long long a, unsigned long long b) { return b ? 100.0 * (double) a / (double) b : 0.0; };
    const unsigned long long copies = g_pf.copies.load();
    MOE_CACHE_LOG("[moe-prefetch-async] %s: predictions=%llu candidates=%llu cold_candidates=%llu cold_projections=%llu | "
            "accuracy: actual_in_prediction=%.1f%% (%llu/%llu) misses_in_prediction=%.1f%% (%llu/%llu) | "
            "served: miss_projections=%llu (%.1f%% of misses) miss_routes=%llu tags_matched=%llu already_resident=%llu resolves=%llu | "
            "host: entries=%llu experts=%llu projection_copies=%llu MiB=%.1f useful_copies=%.1f%% truncated=%llu no_source=%llu overruns=%llu errors=%llu yield_waits=%llu yield_ms=%.1f yield_timeouts=%llu\n",
            when, c[8], c[9], c[10], c[11],
            pct(c[2], c[1]), c[2], c[1], pct(c[4], c[3]), c[4], c[3],
            c[5], pct(c[5], c[3]), c[6], c[12], c[13], c[7],
            g_pf.entries.load(), g_pf.experts.load(), copies, (double) g_pf.copy_bytes.load() / (1024.0 * 1024.0),
            pct(c[5], copies), g_pf.truncated.load(), g_pf.no_source.load(), g_pf.overruns.load(), g_pf.errors.load(),
            g_pf.yield_waits.load(), (double) g_pf.yield_us.load() / 1000.0, g_pf.yield_timeouts.load());
}

static void moe_pf_worker_main() {
    ggml_cuda_set_device(g_pf.device);
    unsigned tail = 0;
    auto last = std::chrono::steady_clock::now();
    while (!g_pf.stop.load(std::memory_order_relaxed)) {
        volatile moe_pf_entry * slot = g_pf.h_ring + tail % MOE_PF_RING;
        const unsigned seq = slot->seq;
        if (seq != tail + 1) {
            if (seq != 0 && (int) (seq - (tail + 1)) > 0) {
                // The device lapped the ring; resume at the entry that overwrote this one.
                g_pf.overruns.fetch_add(seq - (tail + 1), std::memory_order_relaxed);
                tail = seq - 1;
                continue;
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - last < std::chrono::milliseconds(2)) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
            continue;
        }
        moe_pf_entry entry;
        entry.layer = slot->layer;
        entry.stamp = slot->stamp;
        entry.n = std::min<int>((int) slot->n, MOE_PF_MAX_K);
        for (int j = 0; j < entry.n; ++j) {
            entry.e[j] = slot->e[j];
            entry.mask[j] = slot->mask[j];
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (slot->seq != seq) {
            continue; // overwritten while reading; the lap check above resynchronizes
        }
        ++tail;
        last = std::chrono::steady_clock::now();
        g_pf.entries.fetch_add(1, std::memory_order_relaxed);
        if (moe_pf_mode() != 1) {
            continue;
        }
        std::lock_guard<std::mutex> lock(g_pf.mu);
        if (!g_pf.src_ready || entry.layer < 0 || entry.layer >= g_pf.n_layers) {
            continue;
        }
        const int set = entry.layer % g_pf.sets;
        const int n = std::min(entry.n, g_pf.kpf);
        if (entry.n > n) {
            g_pf.truncated.fetch_add(entry.n - n, std::memory_order_relaxed);
        }
        for (int j = 0; j < n; ++j) {
            const int e = entry.e[j];
            if (e < 0 || e >= g_pf.n_expert) {
                continue;
            }
            if (g_pf.yield) {
                // At most two experts in flight, so a demand read never queues behind many copies.
                cudaEvent_t ev = g_pf.inflight[j & 1];
                while (cudaEventQuery(ev) == cudaErrorNotReady) {
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#endif
                }
                const volatile unsigned * mark = g_pf.h_mark;
                if (mark[0] != mark[1]) {
                    const auto t0 = std::chrono::steady_clock::now();
                    auto t = t0;
                    while (mark[0] != mark[1] && t - t0 < std::chrono::microseconds(300)) {
#if defined(__x86_64__) || defined(__i386__)
                        __builtin_ia32_pause();
#endif
                        t = std::chrono::steady_clock::now();
                    }
                    g_pf.yield_waits.fetch_add(1, std::memory_order_relaxed);
                    g_pf.yield_us.fetch_add((unsigned long long) std::chrono::duration_cast<std::chrono::microseconds>(t - t0).count(),
                            std::memory_order_relaxed);
                    if (mark[0] != mark[1]) {
                        g_pf.yield_timeouts.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            char * dst = g_pf.d_buf + ((size_t) set * g_pf.kpf + j) * g_pf.slot_stride;
            unsigned copied = 0;
            for (int k = 0; k < 3; ++k) {
                if (!(entry.mask[j] >> k & 1)) {
                    continue;
                }
                const size_t idx = (size_t) entry.layer * 3 + k;
                const void * src = g_pf.src[idx * g_pf.n_expert + e];
                const size_t bytes = g_pf.bytes[idx];
                if (!src || !bytes) {
                    g_pf.no_source.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (moe_h2d_async(dst + g_pf.kind_off[k], src, bytes, cudaMemcpyHostToDevice, g_pf.stream) != cudaSuccess) {
                    (void) cudaGetLastError();
                    g_pf.errors.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                copied |= 1u << k;
                g_pf.copies.fetch_add(1, std::memory_order_relaxed);
                g_pf.copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
            }
            if (copied) {
                moe_pf_tag_kernel<<<1, 1, 0, g_pf.stream>>>(g_pf.d_tags + (size_t) set * g_pf.kpf + j,
                        moe_pf_tag(entry.stamp, entry.layer, copied, e));
                if (cudaGetLastError() != cudaSuccess) {
                    g_pf.errors.fetch_add(1, std::memory_order_relaxed);
                }
                g_pf.experts.fetch_add(1, std::memory_order_relaxed);
                if (g_pf.yield) {
                    (void) cudaEventRecord(g_pf.inflight[j & 1], g_pf.stream);
                }
            }
        }
    }
}

// Session teardown: the worker stops copying and kernels see no resident masks.
static void moe_pf_release(const void * session) {
    if (!g_pf.d_stats || session != g_pf.owner) {
        return;
    }
    g_pf.active.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> lock(g_pf.mu);
    g_pf.src_ready = false;
    if (g_pf.stream) {
        (void) cudaStreamSynchronize(g_pf.stream);
    }
    if (g_pf.d_hot) {
        (void) cudaMemset(g_pf.d_hot, 0, (size_t) g_pf.n_layers * 3 * sizeof(uint8_t *));
    }
    (void) cudaGetLastError();
}

// Table refreshes retain ownership; only full session destruction hands it off.
static void moe_pf_forget(const void * session) {
    std::lock_guard<std::mutex> lock(g_pf.mu);
    if (g_pf.owner.load(std::memory_order_relaxed) == session) {
        g_pf.requested.store(0, std::memory_order_relaxed);
        g_pf.owner.store(nullptr, std::memory_order_release);
    }
}

// Called at the end of device preparation, outside graph capture.
static void moe_pf_prepare(moe_cache_session & session) {
    const int requested = g_pf.requested.load(std::memory_order_relaxed);
    if (moe_pf_mode() == 0 || requested == 0 || g_pf.alloc_failed || &session != g_pf.owner ||
            session.devices.size() != 1) {
        return;
    }
    const int device = session.devices[0]->logical;
    int n_layers = 0;
    int n_expert = 0;
    size_t kind_bytes[3] = {};
    bool uniform = true;
    for (auto & entry : session.device_tables) {
        auto & t = entry.second;
        t.pf_layer = -1;
        t.pf_kind = -1;
        int layer = -1;
        char kind[16] = {};
        if (!t.desc.name || sscanf(t.desc.name, "blk.%d.ffn_%15[a-z]_exps", &layer, kind) != 2 || layer < 0 || layer >= 1024) {
            continue;
        }
        const int k = strcmp(kind, "up") == 0 ? 0 : strcmp(kind, "gate") == 0 ? 1 : strcmp(kind, "down") == 0 ? 2 : -1;
        const size_t bytes = moe_missjoin_copy_bytes(t);
        if (k < 0 || !t.hot || bytes % 16 != 0 || t.desc.n_expert <= 0 || t.desc.n_expert > 0x7ffff) {
            continue;
        }
        if (n_expert == 0) {
            n_expert = (int) t.desc.n_expert;
        }
        uniform &= n_expert == (int) t.desc.n_expert;
        t.pf_layer = layer;
        t.pf_kind = k;
        n_layers = std::max(n_layers, layer + 1);
        kind_bytes[k] = std::max(kind_bytes[k], bytes);
    }
    const auto disable = [&](const char * why) {
        for (auto & entry : session.device_tables) { entry.second.pf_layer = -1; entry.second.pf_kind = -1; }
        g_pf.active.store(false, std::memory_order_release);
        MOE_CACHE_LOG("[moe-prefetch-async] off: %s\n", why);
    };
    if (!uniform || n_layers == 0) {
        return disable("expert tables are not uniform");
    }
    if (!g_pf.d_stats) {
        const char * v = getenv("GGML_MOE_PF_SLOTS");
        g_pf.kpf = std::max(1, std::min(v ? atoi(v) : 8, MOE_PF_MAX_SLOTS));
        const int lead = requested < 0 ? 1 : requested;
        g_pf.sets = lead + 1;
        g_pf.device = device;
        g_pf.n_layers = n_layers;
        g_pf.n_expert = n_expert;
        size_t off = 0;
        for (int k = 0; k < 3; ++k) {
            g_pf.kind_bytes[k] = kind_bytes[k];
            g_pf.kind_off[k] = off;
            off += GGML_PAD(kind_bytes[k], 256);
        }
        g_pf.slot_stride = off;
        const size_t n_slots = (size_t) g_pf.sets * g_pf.kpf;
        bool ok = cudaMalloc((void **) &g_pf.d_stamp, n_layers * sizeof(unsigned)) == cudaSuccess &&
            cudaMalloc((void **) &g_pf.d_pred_stamp, n_layers * sizeof(unsigned)) == cudaSuccess &&
            cudaMalloc((void **) &g_pf.d_pred, (size_t) n_layers * MOE_PF_MAX_K * sizeof(int32_t)) == cudaSuccess &&
            cudaMalloc((void **) &g_pf.d_tags, n_slots * sizeof(unsigned long long)) == cudaSuccess &&
            cudaMalloc((void **) &g_pf.d_hot, (size_t) n_layers * 3 * sizeof(uint8_t *)) == cudaSuccess &&
            cudaMalloc((void **) &g_pf.d_head, sizeof(unsigned)) == cudaSuccess &&
            (moe_pf_mode() != 1 || cudaMalloc((void **) &g_pf.d_buf, n_slots * g_pf.slot_stride) == cudaSuccess) &&
            cudaHostAlloc((void **) &g_pf.h_ring, MOE_PF_RING * sizeof(moe_pf_entry), cudaHostAllocMapped) == cudaSuccess &&
            cudaHostGetDevicePointer((void **) &g_pf.d_ring, g_pf.h_ring, 0) == cudaSuccess &&
            cudaStreamCreateWithFlags(&g_pf.stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaMalloc((void **) &g_pf.d_stats, MOE_PF_STATS * sizeof(unsigned long long)) == cudaSuccess;
        {
            const char * v = getenv("GGML_MOE_PF_YIELD");
            g_pf.yield = v && strcmp(v, "1") == 0 && moe_pf_mode() == 1;
        }
        if (ok && g_pf.yield) {
            ok = cudaHostAlloc((void **) &g_pf.h_mark, 2 * sizeof(unsigned), cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer((void **) &g_pf.d_mark, g_pf.h_mark, 0) == cudaSuccess &&
                cudaMalloc((void **) &g_pf.d_mark_seq, 2 * sizeof(unsigned)) == cudaSuccess &&
                cudaMemset(g_pf.d_mark_seq, 0, 2 * sizeof(unsigned)) == cudaSuccess &&
                cudaEventCreateWithFlags(&g_pf.inflight[0], cudaEventDisableTiming) == cudaSuccess &&
                cudaEventCreateWithFlags(&g_pf.inflight[1], cudaEventDisableTiming) == cudaSuccess;
            if (ok) {
                g_pf.h_mark[0] = g_pf.h_mark[1] = 0;
            }
        }
        ok = ok && cudaMemset(g_pf.d_stamp, 0, n_layers * sizeof(unsigned)) == cudaSuccess &&
            cudaMemset(g_pf.d_pred_stamp, 0xff, n_layers * sizeof(unsigned)) == cudaSuccess &&
            cudaMemset(g_pf.d_tags, 0, n_slots * sizeof(unsigned long long)) == cudaSuccess &&
            cudaMemset(g_pf.d_hot, 0, (size_t) n_layers * 3 * sizeof(uint8_t *)) == cudaSuccess &&
            cudaMemset(g_pf.d_head, 0, sizeof(unsigned)) == cudaSuccess &&
            cudaMemset(g_pf.d_stats, 0, MOE_PF_STATS * sizeof(unsigned long long)) == cudaSuccess;
        if (ok) {
            memset(g_pf.h_ring, 0, MOE_PF_RING * sizeof(moe_pf_entry));
        }
        if (!ok) {
            (void) cudaGetLastError();
            g_pf.alloc_failed = true;
            g_pf.d_stats = nullptr;
            return disable("allocation failed");
        }
        g_pf.worker = std::thread(moe_pf_worker_main);
        // Registered after the CUDA runtime's own teardown, so it runs first at exit.
        atexit([] {
            g_pf.stop.store(true);
            if (g_pf.worker.joinable()) {
                g_pf.worker.join();
            }
        });
        MOE_CACHE_LOG("[moe-prefetch-async] on: mode=%s%s lead=%d sets=%d slots_per_set=%d slot=%.2f MiB buffer=%.1f MiB layers=%d experts=%d\n",
                moe_pf_mode() == 1 ? "copy" : "predict-only", g_pf.yield ? " (demand first)" : "", lead, g_pf.sets, g_pf.kpf,
                (double) g_pf.slot_stride / (1024.0 * 1024.0),
                moe_pf_mode() == 1 ? (double) (n_slots * g_pf.slot_stride) / (1024.0 * 1024.0) : 0.0, n_layers, n_expert);
    } else if ((requested < 0 ? 1 : requested) + 1 != g_pf.sets ||
            device != g_pf.device || n_layers > g_pf.n_layers || n_expert != g_pf.n_expert ||
            kind_bytes[0] > g_pf.kind_bytes[0] || kind_bytes[1] > g_pf.kind_bytes[1] || kind_bytes[2] > g_pf.kind_bytes[2]) {
        return disable("tables changed shape after the first preparation");
    }
    std::vector<const uint8_t *> hot((size_t) g_pf.n_layers * 3, nullptr);
    std::lock_guard<std::mutex> lock(g_pf.mu);
    g_pf.src.assign((size_t) g_pf.n_layers * 3 * g_pf.n_expert, nullptr);
    g_pf.bytes.assign((size_t) g_pf.n_layers * 3, 0);
    for (auto & entry : session.device_tables) {
        const auto & t = entry.second;
        if (t.pf_layer < 0) {
            continue;
        }
        const size_t idx = (size_t) t.pf_layer * 3 + t.pf_kind;
        hot[idx] = t.hot;
        g_pf.bytes[idx] = moe_missjoin_copy_bytes(t);
        for (int e = 0; e < g_pf.n_expert && (size_t) e < t.copy_source.size(); ++e) {
            g_pf.src[idx * g_pf.n_expert + e] = t.copy_source[e];
        }
    }
    if (cudaMemcpy(g_pf.d_hot, hot.data(), hot.size() * sizeof(uint8_t *), cudaMemcpyHostToDevice) != cudaSuccess) {
        (void) cudaGetLastError();
        g_pf.src_ready = false;
        return disable("table upload failed");
    }
    g_pf.src_ready = true;
    g_pf.active.store(true, std::memory_order_release);
}

bool ggml_cuda_moe_cache_prefetch_async_active(void) {
    return g_pf.active.load(std::memory_order_relaxed);
}

void ggml_cuda_moe_cache_device_predict(ggml_backend_cuda_context & ctx, const ggml_tensor * node) {
    if (!g_pf.active.load(std::memory_order_acquire) || ctx.device != g_pf.device || node->type != GGML_TYPE_I32 ||
            moe_prefetch_session(nullptr) != g_pf.owner) {
        return;
    }
    int source = -1, target = -1;
    if (sscanf(node->name, "moe_prefetch-%d-%d", &source, &target) != 2 || target < 1 || target >= g_pf.n_layers) {
        return;
    }
    const int k_ids = (int) std::min<int64_t>(node->ne[0], MOE_PF_MAX_K);
    moe_pf_predict_kernel<<<1, MOE_PF_MAX_K, 0, ctx.stream()>>>((const int32_t *) node->data, k_ids, target, g_pf.n_expert,
            g_pf.d_hot, g_pf.d_stamp, g_pf.d_pred_stamp, g_pf.d_pred, g_pf.d_ring, g_pf.d_head, 1, g_pf.d_stats);
    CUDA_CHECK(cudaGetLastError());
}

// fork: each projection keeps its effective table and payload alive through its last reader.
struct moe_cache_effective_table {
    ggml_cuda_pool_alloc<char> pf_mem;
    ggml_cuda_pool_alloc<char> mj_scratch;
    ggml_cuda_pool_alloc<const void *> mj_eff;
    bool pf_marked = false;

    explicit moe_cache_effective_table(ggml_cuda_pool & pool) : pf_mem(pool), mj_scratch(pool), mj_eff(pool) {}
};

static const void * const * moe_cache_resolve_table(ggml_backend_cuda_context & ctx, moe_cache_session & session,
        const moe_cache_device_table & table, const ggml_tensor * ids, moe_cache_effective_table & mem) {
    const void * const * mmv_table = table.staging_device ? table.staging_device : table.device;
    const int mj_mode = moe_missjoin_mode();
    const int mj_routes = (int)(ids->ne[0] * ids->ne[1]);
    const size_t mj_bytes = moe_missjoin_copy_bytes(table);
    const bool mj_overlap_on = moe_missjoin_overlap_enabled() && !session.mj_groups.empty();
    const uint8_t * stage_hot = table.hot;
#if !defined(GGML_USE_HIP)
    // fork PREFETCHRW (opt-in): the table-prefetch snapshot replaces the published table for this dispatch.
    if (!session.devices.empty() && session.devices[0]->table_prefetch_prepared) {
        const auto & snapshot = session.devices[0]->table_prefetch->snapshots[table.tp_layer][table.tp_kind];
        mmv_table = snapshot.pointers; stage_hot = snapshot.hot;
    }
#endif
    // fork PREFETCHFIX: substitute tagged prefetched copies for host reads (a per-dispatch table copy).
    if (!mj_overlap_on && table.pf_layer >= 1 && table.pf_layer < g_pf.n_layers && table.hot &&
            mj_routes <= MOE_MISSJOIN_MAX_ROUTES && g_pf.active.load(std::memory_order_relaxed) && &session == g_pf.owner &&
            ctx.device == g_pf.device && table.desc.n_expert == g_pf.n_expert) {
        const size_t n_e = (size_t) table.desc.n_expert;
        char * m = mem.pf_mem.alloc(n_e * sizeof(void *) + GGML_PAD(n_e, 16));
        const void ** pf_table = (const void **) m;
        uint8_t * pf_hot = (uint8_t *) (m + n_e * sizeof(void *));
        const int set = table.pf_layer % g_pf.sets;
        moe_pf_resolve_kernel<<<1, 256, 0, ctx.stream()>>>(mmv_table, table.hot, (const int32_t *) ids->data,
                (int) ids->ne[0], mj_routes, (int)(ids->nb[1] / sizeof(int32_t)), (int) n_e, table.pf_layer, table.pf_kind,
                g_pf.d_stamp, g_pf.d_pred_stamp, g_pf.d_pred, MOE_PF_MAX_K, g_pf.d_tags + (size_t) set * g_pf.kpf, g_pf.kpf,
                g_pf.d_buf ? g_pf.d_buf + (size_t) set * g_pf.kpf * g_pf.slot_stride : nullptr, g_pf.slot_stride,
                g_pf.kind_off[table.pf_kind], pf_table, pf_hot, g_pf.d_stats,
                g_pf.yield ? g_pf.d_mark : nullptr, g_pf.d_mark_seq);
        mmv_table = pf_table;
        stage_hot = pf_hot;
        mem.pf_marked = g_pf.yield;
    }
    const void * const * mj_overlap = mj_overlap_on ? moe_missjoin_overlap_table(ctx, session, table, ids) : nullptr;
    if (mj_overlap) {
        mmv_table = mj_overlap;
    } else if (mj_mode && table.hot && mj_routes <= MOE_MISSJOIN_MAX_ROUTES && mj_bytes % 16 == 0 &&
            table.desc.n_expert <= INT_MAX) {
        const int slots = moe_missjoin_slots();
        const size_t stride = GGML_PAD(mj_bytes, 256);
        char * scratch = mj_mode == 1 ? mem.mj_scratch.alloc(stride * (size_t) slots) : nullptr;
        const void ** eff = mem.mj_eff.alloc((size_t) table.desc.n_expert);
        if (mj_mode == 1 && moe_missjoin_retained_use(ctx, session, table)) {
            moe_missjoin_retained_plan_kernel<<<1, 256, 0, ctx.stream()>>>(
                    mmv_table, stage_hot, (const int32_t *) ids->data, (int) ids->ne[0], mj_routes,
                    (int)(ids->nb[1] / sizeof(int32_t)), (int) table.desc.n_expert, eff, scratch, stride, mj_bytes,
                    slots, table.mj_retained, table.mj_retained_data, moe_missjoin_retained_slots(),
                    session.device_route_counts ? g_missjoin_counts : nullptr, session.mj_retained_counts);
            CUDA_CHECK(cudaGetLastError());
            moe_missjoin_retained_copy_kernel<<<moe_missjoin_blocks(ctx.device), 256, 0, ctx.stream()>>>(table.mj_retained, mj_bytes);
        } else {
            moe_missjoin_stage(
                    mmv_table, stage_hot, (const int32_t *) ids->data, (int) ids->ne[0], mj_routes,
                    (int)(ids->nb[1] / sizeof(int32_t)), (int) table.desc.n_expert, eff, scratch, stride, mj_bytes,
                    slots, mj_mode == 2, session.device_route_counts ? g_missjoin_counts : nullptr,
                    mj_mode == 1 ? moe_missjoin_blocks(ctx.device) : 1, ctx.stream());
        }
        mmv_table = eff;
    }
    return mmv_table;
}

// fork MOEGAP Q8SHARE (GGML_MOE_Q8_SHARE=1|check, default off): the routed experts' up/gate activation goes
// through the mmvq shared-quantize cache, keyed on the tensor the expert input views (the FFN input). The shared
// expert's up/gate reads that same tensor, so it finds the buffer and skips its own quantize, and the gate projection
// of an unfused layer skips its second one. It is the same quantize_row_q8_1_cuda call on the same floats, so the
// bytes are the same; check mode quantizes again into a pool buffer and traps on any byte difference.
static int moe_q8_share_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_MOE_Q8_SHARE");
        if (!e) { return 0; }
        return strcmp(e, "check") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}

// fork AIPFIX: with a bank, its AI activations (bank_q, bank_s) are written too, by the same launch as the q8_1
// buffer when that one is quantized here (GGML_MOE_BANK_MIXED_DUALQ), on their own otherwise.
static char * moe_quantize_act(ggml_backend_cuda_context & ctx, const ggml_tensor * a, ggml_type type, bool share,
        ggml_cuda_pool_alloc<char> & local, const ggml_tensor * bank = nullptr,
        ggml_cuda_pool_alloc<int8_t> * bank_q = nullptr, ggml_cuda_pool_alloc<float> * bank_s = nullptr) {
    if (ggml_ai4_is_unit(type)) {
        // fork AIPFIX D: AI+ cached experts read the AI activation layout (no q8_1, no bank beside them)
        GGML_ASSERT(!bank && ggml_is_contiguous(a));
        const int64_t rows = a->ne[1] * a->ne[2];
        const size_t ai_bytes = ggml_cuda_moe_ai_act_bytes(rows, a->ne[0]);
        // fork AIPFIX D4: up and gate read one AI activation (the shared q8 cache keyed on the input tensor;
        // ne10_padded = -K marks the AI layout so it never matches a q8_1 entry). Same kernel on the same floats.
        if (share && moe_q8_share_mode() && a->type == GGML_TYPE_F32 && a->ne[3] == 1 &&
                ggml_cuda_q8_cacheable(ctx, ai_bytes)) {
            char * hit = ggml_cuda_q8_cache_lookup(ctx, a, type, ai_bytes, -a->ne[0]);
            if (hit) {
                g_st_q8hit.fetch_add(1, std::memory_order_relaxed);
                return hit;
            }
            char * buf = ggml_cuda_q8_cache_claim(ctx, a, type, ai_bytes, -a->ne[0]);
            g_st_q8claim.fetch_add(1, std::memory_order_relaxed);
            ggml_cuda_moe_ai_quantize((const float *) a->data, rows, a->ne[0], buf, ctx.stream());
            return buf;
        }
        char * buf = local.alloc(ai_bytes);
        ggml_cuda_moe_ai_quantize((const float *) a->data, rows, a->ne[0], buf, ctx.stream());
        return buf;
    }
    const int64_t padded = GGML_PAD(a->ne[0], MATRIX_ROW_PADDING);
    const size_t q8_bytes = (size_t)(padded / QK8_1) * a->ne[1] * a->ne[2] * sizeof(block_q8_1);
    GGML_ASSERT(!bank || (bank_q && bank_s));
    bool bank_done = false;
    auto quantize = [&](char * out) {
        if (bank && !bank_done && ggml_cuda_moe_mixed_quantize_dual(ctx, bank, a, out, type, *bank_q, *bank_s)) {
            bank_done = true;
            return;
        }
        quantize_row_q8_1_cuda((const float *) a->data, nullptr, out, type,
                a->ne[0], a->nb[1] / sizeof(float), a->nb[2] / sizeof(float), a->nb[3] / sizeof(float),
                padded, a->ne[1], a->ne[2], 1, ctx.stream());
    };
    auto finish = [&](char * out) {
        if (bank && !bank_done) {
            ggml_cuda_moe_mixed_quantize_x(ctx, bank, *bank_q, *bank_s);
        }
        return out;
    };
    const int mode = share ? moe_q8_share_mode() : 0;
    if (mode && a->type == GGML_TYPE_F32 && a->ne[3] == 1 && ggml_is_contiguous(a) && ggml_cuda_q8_cacheable(ctx, q8_bytes)) {
        // key on the viewed tensor when the input is a whole-tensor reshape of it (the shared expert reads that one)
        const ggml_tensor * base = a;
        const ggml_tensor * v = a->view_src;
        if (v && a->view_offs == 0 && a->data == v->data && v->type == GGML_TYPE_F32 && ggml_is_contiguous(v) &&
                ggml_nelements(v) == ggml_nelements(a) && v->ne[0] == a->ne[0]) {
            base = v;
        }
        char * hit = ggml_cuda_q8_cache_lookup(ctx, base, type, q8_bytes, padded);
        if (hit) {
            g_st_q8hit.fetch_add(1, std::memory_order_relaxed);
            if (mode == 2) {
                char * ref = local.alloc(q8_bytes);
                quantize(ref);
                ggml_cuda_moe_bits_check((const float *) ref, (const float *) hit, (int64_t) (q8_bytes / sizeof(float)),
                        ctx.stream());
            }
            return finish(hit);
        }
        char * buf = ggml_cuda_q8_cache_claim(ctx, base, type, q8_bytes, padded);
        g_st_q8claim.fetch_add(1, std::memory_order_relaxed);
        quantize(buf);
        return finish(buf);
    }
    char * buf = local.alloc(q8_bytes);
    quantize(buf);
    return finish(buf);
}

// fork AIPFIX ONEPATH: the bank experts' routes of one projection, from their bank rows, after a regular pass wrote
// zero there (the bank experts' entries are the zero block). The AI+ table GEMV writes only routes with flag 1.
static void moe_onepath_log(int type, const char * what) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        fprintf(stderr, "[moe-cache-ai] one path: first bank-row %s launch, type=%d\n", what, type);
    }
}

// fork AIPFIX ONEPATH check (LLAMA_MOE_BANK_ONEPATH_CHECK=1; synchronizes, so run it with GGML_CUDA_DISABLE_GRAPHS=1):
// the bank experts' routes of out against an f32 reference, the reconstructed bank rows times the unquantized input
// (with gate: out = silu(gate x) * (table x)). Prints the worst relative route error per call, first 300 calls.
// =2: every route of every call, the other experts against their host rows (f32 dequant); prints only routes over 5%
// and a summary every 2000 calls. Gate/up routes of the CPU-split experts are skipped (the host computes them; the
// card writes only their down rows, after the join); their down rows are checked against the whole expert recomputed
// from the layer input the gate/up check saved (the card's SwiGLU rows of those routes are never written).
static int moe_onepath_check_mode() {
    static const int v = [] { const char * e = getenv("LLAMA_MOE_BANK_ONEPATH_CHECK"); return e ? atoi(e) : 0; }();
    return v;
}

static void moe_onepath_check(ggml_backend_cuda_context & ctx, const moe_cache_device_table & table,
        const moe_cache_device_table * gate, const ggml_tensor * a, const ggml_tensor * ids, const float * out,
        int64_t M, const char * where) {
    static std::atomic<int> calls{0};
    const int mode = moe_onepath_check_mode();
    if (mode == 0 || (mode == 1 && (table.ai_bank_rows.empty() || calls.fetch_add(1) >= 300))) {
        return;
    }
    // mode 2 checks the finished outputs only (the bank and pair passes leave the other routes to the miss pass)
    if (mode == 2 && strcmp(where, "final") != 0) {
        return;
    }
    const int call = mode == 2 ? calls.fetch_add(1) : 0;
    GGML_ASSERT(a->type == GGML_TYPE_F32 && ggml_is_contiguous(a));
    cudaStream_t stream = ctx.stream();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int64_t K = a->ne[0], n_used = ids->ne[0], n_tok = ids->ne[1], ne11 = a->ne[1];
    const int64_t ids_stride = ids->nb[1] / sizeof(int32_t);
    std::vector<int32_t> hid((size_t) (ids_stride*(n_tok - 1) + n_used));
    CUDA_CHECK(cudaMemcpy(hid.data(), ids->data, hid.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
    std::vector<float> hx((size_t) (K*ne11*n_tok));
    CUDA_CHECK(cudaMemcpy(hx.data(), a->data, hx.size()*sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> hy((size_t) (M*n_used*n_tok));
    CUDA_CHECK(cudaMemcpy(hy.data(), out, hy.size()*sizeof(float), cudaMemcpyDeviceToHost));
    ggml_cuda_pool_alloc<float> w(ctx.pool(), (size_t) (M*K));
    std::vector<float> hw((size_t) (M*K));
    std::vector<double> ref((size_t) M), g((size_t) M);
    auto gemv = [&](const void * rows, ggml_type type, const float * x, std::vector<double> & y) {
        ggml_cuda_moe_ai_reconstruct(rows, type, w.get(), K, M, stream);
        CUDA_CHECK(cudaMemcpyAsync(hw.data(), w.get(), hw.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int64_t m = 0; m < M; ++m) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += (double) hw[(size_t) (m*K + k)] * x[k];
            }
            y[(size_t) m] = acc;
        }
    };
    // the host rows of an expert outside the bank (its regular copy), dequantized to f32
    std::vector<float> hrow((size_t) K);
    auto host_gemv = [&](const moe_cache_device_table & tb, int32_t e, const float * x, std::vector<double> & y) {
        const ggml_type type = (ggml_type) tb.desc.type;
        const size_t row_bytes = ggml_row_size(type, K);
        const char * base = (const char *) tb.desc.data + (size_t) e*tb.host_stride;
        const auto * traits = ggml_get_type_traits(type);
        GGML_ASSERT(traits->to_float && "onepath check 2: no host dequant for this type");
        for (int64_t m = 0; m < M; ++m) {
            traits->to_float(base + (size_t) m*row_bytes, hrow.data(), K);
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += (double) hrow[(size_t) k] * x[k];
            }
            y[(size_t) m] = acc;
        }
    };
    // CPU-split experts of this layer (the plan of the layer's first projection)
    std::vector<int32_t> cpu_exp;
    if (mode == 2 && g_cs_on && g_cs_plan) {
        moe_cs_plan_dev plan;
        CUDA_CHECK(cudaMemcpy(&plan, g_cs_plan, sizeof(plan), cudaMemcpyDeviceToHost));
        for (int i = 0; i < std::min(plan.n_exp, (int) MOE_CS_MAX_EXPERTS); ++i) {
            cpu_exp.push_back(plan.expert[i]);
        }
    }
    // the layer input of the last gate/up check (mode 2), for the CPU-split experts' down check
    struct saved_in {
        int layer = -1;
        int64_t K = 0, M = 0, ne11 = 0, n_tok = 0, n_used = 0, ids_stride = 0;
        std::vector<float> x;
        std::vector<int32_t> ids;
        const moe_cache_device_table * up = nullptr, * gate = nullptr;
    };
    static saved_in s_in;
    if (mode == 2 && gate) {
        s_in.layer = table.mj_layer; s_in.K = K; s_in.M = M; s_in.ne11 = ne11; s_in.n_tok = n_tok; s_in.n_used = n_used;
        s_in.ids_stride = ids_stride; s_in.x = hx; s_in.ids = hid; s_in.up = &table; s_in.gate = gate;
    }
    // one host row-dot over any table and shape
    auto host_rows = [&](const moe_cache_device_table & tb, int32_t e, int64_t rows, int64_t cols, const float * x,
            std::vector<double> & y) {
        const ggml_type type = (ggml_type) tb.desc.type;
        const size_t row_bytes = ggml_row_size(type, cols);
        const char * base = (const char *) tb.desc.data + (size_t) e*tb.host_stride;
        const auto * traits = ggml_get_type_traits(type);
        std::vector<float> row((size_t) cols);
        y.assign((size_t) rows, 0.0);
        for (int64_t m = 0; m < rows; ++m) {
            traits->to_float(base + (size_t) m*row_bytes, row.data(), cols);
            double acc = 0.0;
            for (int64_t k = 0; k < cols; ++k) {
                acc += (double) row[(size_t) k] * x[k];
            }
            y[(size_t) m] = acc;
        }
    };
    static std::atomic<long long> s_routes{0}, s_bad{0}, s_cpu_checked{0};
    static double s_worst_bank = 0.0, s_worst_other = 0.0;
    double worst = 0.0;
    int routes = 0;
    for (int64_t t = 0; t < n_tok; ++t) {
        for (int64_t u = 0; u < n_used; ++u) {
            const int32_t e = hid[(size_t) (t*ids_stride + u)];
            const bool banked = e >= 0 && e < (int32_t) table.ai_bank_host.size() && table.ai_bank_host[e];
            if (e < 0 || e >= (int32_t) table.desc.n_expert || (!banked && mode != 2)) {
                continue;
            }
            const bool cpu = std::find(cpu_exp.begin(), cpu_exp.end(), e) != cpu_exp.end();
            const bool released = (size_t) e < table.bank.size() && table.bank[e] != 0; // bank-path control: host rows gone
            if (!banked && ((cpu && table.mj_kind != 2) || released)) {
                continue;
            }
            // a CPU-split expert's down: recompute the whole expert from the saved layer input
            const bool full = !banked && cpu && table.mj_kind == 2;
            if (full && !(s_in.layer == table.mj_layer && s_in.n_tok == n_tok && s_in.n_used == n_used &&
                    s_in.ids[(size_t) (t*s_in.ids_stride + u)] == e && s_in.M == K)) {
                continue;
            }
            const float * x = hx.data() + (t*ne11 + (ne11 == 1 ? 0 : u))*K;
            std::vector<float> xg;
            if (full) {
                const float * x0 = s_in.x.data() + (t*s_in.ne11 + (s_in.ne11 == 1 ? 0 : u))*s_in.K;
                std::vector<double> uo, go;
                host_rows(*s_in.up, e, s_in.M, s_in.K, x0, uo);
                host_rows(*s_in.gate, e, s_in.M, s_in.K, x0, go);
                xg.resize((size_t) s_in.M);
                for (int64_t m = 0; m < s_in.M; ++m) {
                    xg[(size_t) m] = (float) (go[(size_t) m] / (1.0 + exp(-go[(size_t) m])) * uo[(size_t) m]);
                }
                x = xg.data();
                ++s_cpu_checked;
            }
            if (banked) {
                gemv(table.ai_bank_rows[e], (ggml_type) table.ai_bank_type, x, ref);
            } else {
                host_gemv(table, e, x, ref);
            }
            if (gate && !banked) {
                host_gemv(*gate, e, x, g);
                for (int64_t m = 0; m < M; ++m) {
                    ref[(size_t) m] *= g[(size_t) m] / (1.0 + exp(-g[(size_t) m]));
                }
            } else if (gate) {
                gemv(gate->ai_bank_rows[e], (ggml_type) gate->ai_bank_type, x, g);
                for (int64_t m = 0; m < M; ++m) {
                    ref[(size_t) m] *= g[(size_t) m] / (1.0 + exp(-g[(size_t) m]));
                }
            }
            const float * y = hy.data() + (t*n_used + u)*M;
            double d2 = 0.0, r2 = 0.0;
            for (int64_t m = 0; m < M; ++m) {
                d2 += (y[m] - ref[(size_t) m])*(y[m] - ref[(size_t) m]);
                r2 += ref[(size_t) m]*ref[(size_t) m];
            }
            const double rel = sqrt(d2/std::max(r2, 1e-30));
            worst = std::max(worst, rel);
            ++routes;
            if (mode == 2) {
                double & w2 = banked ? s_worst_bank : s_worst_other;
                w2 = std::max(w2, rel);
                ++s_routes;
                if (!(rel <= 0.05)) {
                    ++s_bad;
                    fprintf(stderr, "[onepath-check2] BAD call=%d %s layer=%d kind=%d gate=%d tok=%lld slot=%lld expert=%d "
                            "bank=%d cpu=%d full=%d rel=%.4g |ref|=%.4g\n", call, where, table.mj_layer, table.mj_kind,
                            gate ? 1 : 0, (long long) t, (long long) u, e, banked ? 1 : 0, cpu ? 1 : 0, full ? 1 : 0, rel,
                            sqrt(r2));
                }
            }
        }
    }
    if (mode == 2) {
        if (call % 2000 == 0) {
            fprintf(stderr, "[onepath-check2] calls=%d routes=%lld bad=%lld cpu_checked=%lld worst_bank=%.4g "
                    "worst_other=%.4g\n", call + 1, s_routes.load(), s_bad.load(), s_cpu_checked.load(), s_worst_bank,
                    s_worst_other);
        }
        return;
    }
    fprintf(stderr, "[onepath-check] %s layer=%d kind=%d tokens=%lld routes=%d worst_rel=%.4g\n", where, table.mj_layer,
            table.mj_kind, (long long) n_tok, routes, worst);
}

static void moe_onepath_bank(ggml_backend_cuda_context & ctx, const moe_cache_device_table & table,
        const ggml_tensor * a, const ggml_tensor * ids, float * out, int64_t M, bool share) {
    if (!table.ai_bank) {
        // a banked tensor without bank rows here would leave its bank routes at zero
        GGML_ASSERT(!(moe_onepath() && table.n_bank > 0) && "one path: banked tensor without AI+ bank rows");
        return;
    }
    const ggml_type wtype = (ggml_type) table.ai_bank_type;
    moe_onepath_log((int) wtype, "GEMV");
    ggml_cuda_pool_alloc<char> local(ctx.pool());
    const char * act = moe_quantize_act(ctx, a, wtype, share, local);
    const bool ok = ggml_cuda_moe_ai_table_mmv(table.ai_bank, wtype, act, (const int32_t *) ids->data, out,
            a->ne[0], M, ids->ne[0], ids->ne[1], a->ne[1], ids->nb[1] / sizeof(int32_t), (int) table.desc.n_expert,
            table.ai_bank_flag, 1, ctx.stream());
    GGML_ASSERT(ok);
    moe_onepath_check(ctx, table, nullptr, a, ids, out, M, "bank-pass");
}

static __global__ void moe_onepath_swiglu(const float * gate, const float * up, float * dst, const int32_t * ids,
        const uint8_t * flag, int M, int n_used, int64_t ids_stride, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t r = i / M;
    if (flag[ids[(r / n_used)*ids_stride + r % n_used]]) {
        dst[i] = ggml_cuda_op_silu_single(gate[i]) * up[i];
    }
}

// gate/up of the bank experts' routes and their SwiGLU into glu (one paired launch when the two bank types match)
static void moe_onepath_bank_pair(ggml_backend_cuda_context & ctx, const moe_cache_device_table & up_table,
        const moe_cache_device_table & gate_table, const ggml_tensor * a, const ggml_tensor * ids, ggml_tensor * glu) {
    GGML_ASSERT(!up_table.ai_bank == !gate_table.ai_bank);
    if (!up_table.ai_bank) {
        GGML_ASSERT(!(moe_onepath() && up_table.n_bank > 0) && "one path: banked tensor without AI+ bank rows");
        return;
    }
    const ggml_type ut = (ggml_type) up_table.ai_bank_type;
    const ggml_type gt = (ggml_type) gate_table.ai_bank_type;
    const int64_t M = glu->ne[0], n = ggml_nelements(glu), ids_stride = ids->nb[1] / sizeof(int32_t);
    ggml_cuda_pool_alloc<char> local(ctx.pool());
    const char * act = moe_quantize_act(ctx, a, ut, true, local);
    ggml_cuda_pool_alloc<float> up_out(ctx.pool(), n);
    ggml_cuda_pool_alloc<float> gate_out(ctx.pool(), n);
    static const bool group_pair = [] {
        const char * v = getenv("LLAMA_MOE_BANK_ONEPATH_GROUP_PAIR");
        return v && atoi(v) != 0;
    }();
    auto run = [&](const moe_cache_device_table & t, ggml_type type, float * out, const void * const * t2, float * out2,
            const uint8_t * f2, float * g) {
        return ggml_cuda_moe_ai_table_mmv(t.ai_bank, type, act, (const int32_t *) ids->data, out, a->ne[0], M,
                ids->ne[0], ids->ne[1], a->ne[1], ids_stride, (int) t.desc.n_expert, t.ai_bank_flag, 1, ctx.stream(),
                t2, out2, f2, g, nullptr, nullptr, !t2 || group_pair);
    };
    // f3n: the bank-route errors of the 100K config came from the grouped AI+ GEMV's paired launch over the bank rows,
    // not the fused SwiGLU (grouping off with the fused epilogue checked clean; the unpaired grouped downs are clean).
    // The paired launch stays ungrouped (LLAMA_MOE_BANK_ONEPATH_GROUP_PAIR=1 groups it) and the fused epilogue is the
    // default again; LLAMA_MOE_BANK_ONEPATH_GLU=0 runs the SwiGLU after, LLAMA_MOE_BANK_ONEPATH_PAIR=0 runs gate and
    // up as two unpaired (grouped) launches (diagnostics).
    static const bool fused = [] {
        const char * v = getenv("LLAMA_MOE_BANK_ONEPATH_GLU");
        return !v || atoi(v) != 0;
    }();
    static const bool paired = [] {
        const char * v = getenv("LLAMA_MOE_BANK_ONEPATH_PAIR");
        return !v || atoi(v) != 0;
    }();
    if (ut == gt && paired && fused && run(up_table, ut, up_out.get(), gate_table.ai_bank, gate_out.get(),
            gate_table.ai_bank_flag, (float *) glu->data)) {
        moe_onepath_log((int) ut, "paired gate/up+SwiGLU");
        moe_onepath_check(ctx, up_table, &gate_table, a, ids, (const float *) glu->data, M, "pair-pass");
        return;
    }
    if (ut == gt && paired && run(up_table, ut, up_out.get(), gate_table.ai_bank, gate_out.get(), gate_table.ai_bank_flag,
            nullptr)) {
        moe_onepath_log((int) ut, "paired gate/up, SwiGLU after");
    } else {
        moe_onepath_log((int) ut, "gate/up (two launches)");
        GGML_ASSERT(run(up_table, ut, up_out.get(), nullptr, nullptr, nullptr, nullptr));
        GGML_ASSERT(run(gate_table, gt, gate_out.get(), nullptr, nullptr, nullptr, nullptr));
    }
    moe_onepath_swiglu<<<(n + 255)/256, 256, 0, ctx.stream()>>>(gate_out.get(), up_out.get(), (float *) glu->data,
            (const int32_t *) ids->data, up_table.ai_bank_flag, (int) M, (int) ids->ne[0], ids_stride, n);
    CUDA_CHECK(cudaGetLastError());
    moe_onepath_check(ctx, up_table, &gate_table, a, ids, (const float *) glu->data, M, "pair-pass");
}

static bool moe_cache_device_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op, const ggml_tensor * bank,
        const ggml_tensor * route_map = nullptr, ggml_tensor * route_mask = nullptr,
        const ggml_cuda_moe_mixed_reduce * red = nullptr) {
    if (g_session_stack.empty() || !g_session_stack.back().active ||
            !moe_cache_device_op_fits(*g_session_stack.back().active, op, ctx.device)) {
        return false;
    }
    auto & session = *g_session_stack.back().active;
    const auto & table = session.device_tables.at(op->src[0]->data);
    const ggml_tensor * a = op->src[1];
    const ggml_tensor * ids = op->src[2];
    session.device_dispatches.fetch_add(1, std::memory_order_relaxed);
    moe_cache_swap_note_stream(session, ctx);
    // fork (missjoin hit-first): the first projection of a layer starts the layer's miss copies right here,
    // before this dispatch's own work, so the link is busy while the resident math runs.
    const bool hf_on = moe_hf_begin(ctx, session, table, a, ids);
#if !defined(GGML_USE_HIP)
    auto & tp_device = *session.devices[0];
    const bool table_prefetch = tp_device.table_prefetch_prepared;
    if (table_prefetch) {
        GGML_ASSERT(table.tp_layer >= 0 && table.tp_kind >= 0 && session.device_expert_counts && table.hot);
        tp_device.table_prefetch->dispatch(&ctx, unsigned(table.tp_layer), unsigned(table.tp_kind),
            (const int32_t *) ids->data, int(ids->ne[0]), int(ids->ne[1]), int(ids->nb[1] / sizeof(int32_t)),
            table.device, table.hot, session.device_route_counts, session.device_expert_counts + table.count_offset);
    } else
#endif
    if (session.device_expert_counts && !moe_plan_count_take(session, table, ids)) {
        const int n = (int)(ids->ne[0] * ids->ne[1]);
        moe_cache_count_routes<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(
                (const int32_t *)ids->data, table.hot, session.device_route_counts,
                session.device_expert_counts + table.count_offset, n, table.desc.n_expert, (int)ids->ne[0], (int)(ids->nb[1] / sizeof(int32_t)));
    }
    ggml_cuda_pool_alloc<char> q8(ctx.pool());
    ggml_cuda_pool_alloc<int8_t> bank_q(ctx.pool());
    ggml_cuda_pool_alloc<float> bank_s(ctx.pool());
    // fork: overlap both activation quantizers with expert copies before joining the side stream.
    char * quantized = moe_quantize_act(ctx, a, (ggml_type) table.desc.type, table.mj_kind == 0 || table.mj_kind == 1, q8,
            bank && hf_on ? bank : nullptr, &bank_q, &bank_s);
    if (hf_on) {
        // Resident routes first, then the routes whose experts were just copied; together they compute
        // every route exactly once with the same kernel.
        const int64_t ids_stride = ids->nb[1] / sizeof(int32_t);
        // fork MOEGAP: on a CPU-split layer the miss pass also skips the experts the host computes
        const bool cs = session.cs_layer >= 0 && session.cs_layer == table.mj_layer;
        // --moe-skip-tail: outside CPU-split layers the miss pass skips the left-out experts through the cutoff
        // masks (a CPU-split plan already folded them into g_cs_mask)
        const bool tail = moe_tail_active(session, table);
        const uint8_t * miss_skip = cs ? g_cs_mask + (size_t) table.mj_kind * g_hf_experts :
                tail ? g_tail_mask + (size_t) table.mj_kind * g_hf_experts : table.hot;
        const void * const * eff = nullptr;
        // fork AIPFIX D3: AI+ unit experts run both passes through the AI+ direct GEMV (same per-route kernel,
        // filtered by expert, so the two passes write disjoint routes as the MMVQ ones do)
        const ggml_type wtype = (ggml_type) table.desc.type;
        const bool ai = ggml_ai4_is_unit(wtype);
        GGML_ASSERT(!ai || !bank);
        auto ai_pass = [&](const void * const * tbl, float * out, const uint8_t * filter, int filter_val) {
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true)) {
                fprintf(stderr, "[moe-cache-ai] first cached AI+ slot GEMV: type=%d filter=%d\n", (int) wtype, filter_val);
            }
            const bool ok = ggml_cuda_moe_ai_table_mmv(tbl, wtype, quantized, (const int32_t *) ids->data, out,
                    a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1], ids_stride, (int) table.desc.n_expert,
                    filter, filter_val, ctx.stream());
            GGML_ASSERT(ok);
        };
        auto hit_pass = [&](float * out) {
            if (ai) {
                ai_pass(table.device, out, table.hot, 1);
                return;
            }
            const bool hit_ok = ggml_cuda_moe_table_mmv_d1_try(ctx, table.device, (ggml_type) table.desc.type, quantized,
                    (const int32_t *) ids->data, out, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1],
                    ids_stride, table.hot, 1) ||
                ggml_cuda_moe_table_mmv_filtered(table.device, (ggml_type) table.desc.type, quantized,
                    (const int32_t *) ids->data, out, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1],
                    ids_stride, table.hot, 1, ctx.stream());
            GGML_ASSERT(hit_ok);
        };
        auto old_miss = [&](float * out) {
            if (ai) {
                ai_pass(eff, out, miss_skip, 0);
                return;
            }
            const bool miss_ok = ggml_cuda_moe_table_mmv_d1_try(ctx, eff, (ggml_type) table.desc.type, quantized,
                    (const int32_t *) ids->data, out, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1],
                    ids_stride, miss_skip, 0) ||
                ggml_cuda_moe_table_mmv_filtered(eff, (ggml_type) table.desc.type, quantized,
                    (const int32_t *) ids->data, out, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1],
                    ids_stride, miss_skip, 0, ctx.stream());
            GGML_ASSERT(miss_ok);
        };
        auto miss_pass = [&](float * out) {
            if (!(!ai && ids->ne[2] == 1 && table.mj_kind >= 0 && table.mj_kind < 3 &&
                    moe_miss_pass_list(ctx, session, table.mj_layer, table.mj_kind, eff, nullptr, (ggml_type) table.desc.type,
                        quantized, out, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1], old_miss))) {
                old_miss(out);
            }
        };
        // The down projection uses one pass only when no tail rows need zeroing.
        const int one = table.mj_kind == 2 && !tail ? moe_down_one_mode() : 0;
        if (bank) {
            const uint8_t * mixed_skip = cs ? g_cs_mask + (size_t) table.mj_kind*g_hf_experts :
                    tail ? g_tail_mask + (size_t) table.mj_kind*g_hf_experts : nullptr;
            // fork AIPFIX: bank and resident routes run before the copy wait, as the hit pass does;
            // the resident table equals the effective one for those experts.
            const bool split = moe_mixed_hit_first();
            // the CPU join below rewrites routes after the launch, so a CPU-split layer sums them afterwards
            // fork AIPFIX P2LIST: the copied routes take the 3XS miss-list kernel (the sum is then its own launch)
            const bool p2list = split && moe_bank_p2_list() && ids->ne[2] == 1 && table.mj_kind >= 0 &&
                    table.mj_kind < 3 && session.ml_layer >= 0 && session.ml_layer == table.mj_layer &&
                    ggml_cuda_moe_table_mmv_list_supported(wtype, false, a->ne[0], ids->ne[0], ids->ne[1], a->ne[1]);
            // fork AIPFIX P1HITG (down): the cached 3XS down routes take the 3XS hit kernel (one weight read per
            // expert for all tokens routed to it; bank routes get zeros there), phase 1 then takes the bank routes
            // only, and the routed sum is its own launch, as P2LIST.
            const bool p1hd = split && !ai && table.mj_kind == 2 && ids->ne[2] == 1 && moe_bank_p1_hitg_down();
            const ggml_cuda_moe_mixed_reduce * kred = (cs && table.mj_kind == 2) || p2list || p1hd ? nullptr : red;
            if (split) {
                auto phase1 = [&]() {
                    ggml_cuda_mul_mat_id_mixed(ctx, bank, op, op, quantized, table.device, nullptr, table.hot,
                            route_map, route_mask, bank_q.get(), bank_s.get(), p1hd ? 3 : 1, nullptr, kred);
                    CUDA_CHECK(cudaGetLastError());
                };
                if (p1hd) {
                    g_st_p1hitg.fetch_add(1, std::memory_order_relaxed);
                }
                // fork AIPFIX P1CONC: bank-only phase 3 on the side stream while the 3XS hit pass runs
                if (p1hd && moe_bank_p1_conc()) {
                    moe_p1_conc_run(ctx, phase1, [&]() { hit_pass((float *) op->data); });
                } else {
                    if (p1hd) {
                        hit_pass((float *) op->data);
                    }
                    phase1();
                }
            }
            eff = moe_hf_wait(ctx, session, table);
            auto mixed_rest = [&](float * out) {
                ggml_tensor o = *op;
                o.data = out;
                ggml_cuda_mul_mat_id_mixed(ctx, bank, op, &o, quantized, eff, mixed_skip, table.hot, route_map,
                        route_mask, bank_q.get(), bank_s.get(), split ? 2 : 0, nullptr, kred);
            };
            // fork AIPFIX P2SAME: with P1HITG down, the copied routes take the 3XS miss pass (same kernel as the
            // hit pass above, miss masks; bank routes are hot and skipped), not the mixed kernel's phase 2
            if (p1hd && moe_bank_p2_same()) {
                old_miss((float *) op->data);
                g_st_p2same.fetch_add(1, std::memory_order_relaxed);
            } else if (!p2list || !moe_miss_pass_list(ctx, session, table.mj_layer, table.mj_kind, eff, nullptr, wtype,
                        quantized, (float *) op->data, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1],
                        mixed_rest)) {
                mixed_rest((float *) op->data);
            }
            if (p2list || p1hd) {
                CUDA_CHECK(cudaGetLastError());
                g_st_p2list.fetch_add(p2list ? 1 : 0, std::memory_order_relaxed);
                if (tail) {
                    moe_tail_zero(ctx, ids, (int) table.desc.n_expert, op);
                }
                // CPU-split down layers sum after the join below
                if (red && !(cs && table.mj_kind == 2)) {
                    ggml_cuda_op_moe_mixed_reduction(ctx, op, red->weights, red->bank_ids, red->dst);
                }
            }
        } else if (one != 0) {
            eff = moe_hf_wait(ctx, session, table);
            const int64_t count = op->ne[0] * ids->ne[0] * ids->ne[1];
            GGML_ASSERT(op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_nelements(op) == count);
            ggml_cuda_pool_alloc<float> reference(ctx.pool());
            if (one == 2) {
                float * ref = reference.alloc((size_t) count);
                CUDA_CHECK(cudaMemcpyAsync(ref, op->data, (size_t) count * sizeof(float), cudaMemcpyDeviceToDevice,
                        ctx.stream()));
                hit_pass(ref);
                miss_pass(ref);
            }
            // every expert the card owns (CPU-split layers skip the host's experts, flagged by the plan)
            const uint8_t * cpu = cs ? g_cs_mask + 3 * (size_t) g_hf_experts : g_hf_zero;
            const bool one_ok = (ai && (ai_pass(eff, (float *) op->data, cpu, 0), true)) ||
                ggml_cuda_moe_table_mmv_d1_try(ctx, eff, (ggml_type) table.desc.type, quantized,
                    (const int32_t *) ids->data, (float *) op->data, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1],
                    a->ne[1], ids_stride, cpu, 0) ||
                ggml_cuda_moe_table_mmv_filtered(eff, (ggml_type) table.desc.type, quantized,
                    (const int32_t *) ids->data, (float *) op->data, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1],
                    a->ne[1], ids_stride, cpu, 0, ctx.stream());
            GGML_ASSERT(one_ok);
            g_st_one.fetch_add(1, std::memory_order_relaxed);
            // with CPU_SPLIT_PARTIAL the old hit pass also wrote the resident rows of the host's experts (the join
            // overwrites them right after), so the two buffers are only comparable without it
            if (one == 2 && !(cs && g_cs_partial)) {
                ggml_cuda_moe_bits_check(reference.get(), (const float *) op->data, count, ctx.stream());
            }
            moe_onepath_bank(ctx, table, a, ids, (float *) op->data, op->ne[0], table.mj_kind == 0 || table.mj_kind == 1);
        } else {
            hit_pass((float *) op->data);
            CUDA_CHECK(cudaGetLastError());
            // fork AIPFIX ONEPATH: bank experts are resident (hot), so the hit pass wrote their zeros; before the wait
            moe_onepath_bank(ctx, table, a, ids, (float *) op->data, op->ne[0], table.mj_kind == 0 || table.mj_kind == 1);
            eff = moe_hf_wait(ctx, session, table);
            miss_pass((float *) op->data);
            if (tail) {
                moe_tail_zero(ctx, ids, (int) table.desc.n_expert, op);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        if (cs && table.mj_kind == 2) {
            GGML_ASSERT(op->type == GGML_TYPE_F32 && op->nb[0] == sizeof(float));
            // AIPFIX per-type probe: this layer's fixed cost, and its route cost relative to the reference
            const int csl = table.mj_layer;
            const bool own = csl >= 0 && csl < (int) session.cs_route_l.size() && session.cs_route_l[csl] > 0.0f && g_cs_route_ns > 0.0f;
            moe_cs_join_kernel<<<g_cs_max_route, 128, 0, ctx.stream()>>>(g_cs_plan, g_cs_job_d, g_cs_out_d,
                    g_cs_out_stride, (char *) op->data, op->nb[1], op->nb[2], (int) ids->ne[0], (int) op->ne[0],
                    g_cs_adapt ? g_cs_route_ns : 0.0f, own ? session.cs_fixed_l[csl] : g_cs_fixed_ns,
                    bank ? (const int32_t *) bank->src[2]->data : nullptr,
                    bank ? bank->src[2]->nb[1]/sizeof(int32_t) : 0,
                    own ? session.cs_route_l[csl] / g_cs_route_ns : 1.0f);
            CUDA_CHECK(cudaGetLastError());
            session.cs_layer = -1;
        }
        if (red && bank && cs && table.mj_kind == 2) {
            ggml_cuda_op_moe_mixed_reduction(ctx, op, red->weights, red->bank_ids, red->dst);
        }
        if (!bank) {
            moe_onepath_check(ctx, table, nullptr, a, ids, (const float *) op->data, op->ne[0], "final");
        }
        return true;
    }
    moe_cache_effective_table effective(ctx.pool());
    const void * const * mmv_table = moe_cache_resolve_table(ctx, session, table, ids, effective);
    if (ggml_ai4_is_unit((ggml_type) table.desc.type)) {
        GGML_ASSERT(!bank);
        const bool ok = ggml_cuda_moe_ai_table_mmv(mmv_table, (ggml_type) table.desc.type, quantized,
                (const int32_t *) ids->data, (float *) op->data, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1],
                ids->nb[1] / sizeof(int32_t), (int) table.desc.n_expert, nullptr, 0, ctx.stream());
        GGML_ASSERT(ok);
    } else if (bank) {
        ggml_cuda_mul_mat_id_mixed(ctx, bank, op, op, quantized, mmv_table, nullptr, nullptr, route_map, route_mask,
                nullptr, nullptr, 0, nullptr, red);
    } else if (!ggml_cuda_moe_table_mmv_grouped_try(ctx, mmv_table, nullptr, (ggml_type)table.desc.type, quantized,
            (const int32_t *)ids->data, (float *)op->data, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1],
            a->ne[1], ids->nb[1] / sizeof(int32_t)) &&
        !ggml_cuda_moe_table_mmv_d1_try(ctx, mmv_table, (ggml_type)table.desc.type, quantized,
            (const int32_t *)ids->data, (float *)op->data, a->ne[0], op->ne[0], ids->ne[0], ids->ne[1],
            a->ne[1], ids->nb[1] / sizeof(int32_t), nullptr, 0)) {
        ggml_cuda_moe_table_mmv(mmv_table, (ggml_type)table.desc.type, quantized,
                (const int32_t *)ids->data, (float *)op->data,
                a->ne[0], op->ne[0], ids->ne[0], ids->ne[1], a->ne[1], ids->nb[1] / sizeof(int32_t), ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
    if (!bank) {
        moe_onepath_bank(ctx, table, a, ids, (float *) op->data, op->ne[0], table.mj_kind == 0 || table.mj_kind == 1);
    }
    if (effective.pf_marked) {
        moe_pf_mark_end_kernel<<<1, 1, 0, ctx.stream()>>>(g_pf.d_mark_seq, g_pf.d_mark);
    }
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

bool ggml_cuda_moe_cache_device_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    return moe_cache_device_compute(ctx, op, nullptr);
}

bool ggml_cuda_moe_cache_mixed_compute(ggml_backend_cuda_context & ctx, const ggml_tensor * bank,
        ggml_tensor * base, ggml_tensor * dst, const ggml_tensor * route_map, ggml_tensor * route_mask,
        const ggml_cuda_moe_mixed_reduce * red) {
    if (!ggml_cuda_moe_mixed_supported(bank, base)) {
        return false;
    }
    ggml_tensor output = *base;
    output.data = dst->data;
    return moe_cache_device_compute(ctx, &output, bank, route_map, route_mask, red);
}

// fork AIPFIX (GGML_MOE_BANK_MIXED_PAIR): up and gate share the activation quantizers, the copy wait and one
// mixed launch per phase; every route of each projection runs the same per-route code as separate launches.
bool ggml_cuda_moe_cache_mixed_pair_compute(ggml_backend_cuda_context & ctx,
        const ggml_tensor * bank, ggml_tensor * base, ggml_tensor * dst,
        const ggml_tensor * bank2, ggml_tensor * base2, ggml_tensor * dst2,
        const ggml_tensor * route_map, ggml_tensor * route_mask, ggml_tensor * glu) {
    if (!ggml_cuda_moe_mixed_pair_supported(bank, base, bank2, base2) ||
            g_session_stack.empty() || !g_session_stack.back().active) {
        return false;
    }
    auto & session = *g_session_stack.back().active;
    if (!moe_cache_device_op_fits(session, base, ctx.device) || !moe_cache_device_op_fits(session, base2, ctx.device)) {
        return false;
    }
#if !defined(GGML_USE_HIP)
    if (!session.devices.empty() && session.devices[0]->table_prefetch_prepared) {
        return false;
    }
#endif
    const auto & first = session.device_tables.at(base->src[0]->data);
    const auto & second = session.device_tables.at(base2->src[0]->data);
    const ggml_tensor * a = base->src[1];
    const ggml_tensor * ids = base->src[2];
    if (first.mj_layer < 0 || first.mj_layer != second.mj_layer || first.desc.type != second.desc.type ||
            first.mj_kind < 0 || first.mj_kind > 1 || second.mj_kind < 0 || second.mj_kind > 1 ||
            first.mj_kind == second.mj_kind) {
        return false;
    }
    // A declined begin has no side effects; the caller then runs both projections separately.
    if (!moe_hf_begin(ctx, session, first, a, ids)) {
        return false;
    }
    GGML_ASSERT(moe_hf_begin(ctx, session, second, a, ids));
    session.device_dispatches.fetch_add(2, std::memory_order_relaxed);
    moe_cache_swap_note_stream(session, ctx);
    if (session.device_expert_counts) {
        const int n = (int)(ids->ne[0] * ids->ne[1]);
        for (const auto * table : {&first, &second}) {
            if (moe_plan_count_take(session, *table, ids)) { continue; }
            moe_cache_count_routes<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(
                    (const int32_t *)ids->data, table->hot, session.device_route_counts,
                    session.device_expert_counts + table->count_offset, n, table->desc.n_expert,
                    (int)ids->ne[0], (int)(ids->nb[1] / sizeof(int32_t)));
        }
    }
    ggml_cuda_pool_alloc<char> q8(ctx.pool());
    ggml_cuda_pool_alloc<int8_t> bank_q(ctx.pool());
    ggml_cuda_pool_alloc<float> bank_s(ctx.pool());
    char * quantized = moe_quantize_act(ctx, a, (ggml_type) first.desc.type, true, q8, bank, &bank_q, &bank_s);
    ggml_tensor op = *base;
    op.data = dst->data;
    ggml_tensor op2 = *base2;
    op2.data = dst2->data;
    const bool cs = session.cs_layer >= 0 && session.cs_layer == first.mj_layer;
    const bool tail = moe_tail_active(session, first);
    auto skip_of = [&](const moe_cache_device_table & table) -> const uint8_t * {
        return cs ? g_cs_mask + (size_t) table.mj_kind*g_hf_experts :
                tail ? g_tail_mask + (size_t) table.mj_kind*g_hf_experts : nullptr;
    };
    const bool split = moe_mixed_hit_first();
    // fork AIPFIX P2LIST: the copied routes take the 3XS miss-list kernel (fused up * silu(gate) into glu, or one
    // list launch per projection without glu). The fused list covers every route either projection missed, so
    // phase 1 then takes a cached route only when both projections hold it (no half-counted SwiGLU tile).
    const ggml_type wtype = (ggml_type) first.desc.type;
    const bool p2list = split && moe_bank_p2_list() && ids->ne[2] == 1 && session.ml_layer >= 0 &&
            session.ml_layer == first.mj_layer &&
            ggml_cuda_moe_table_mmv_list_supported(wtype, glu != nullptr, a->ne[0], ids->ne[0], ids->ne[1], a->ne[1]);
    // fork AIPFIX P1HITG: cached routes (resident for both projections) take the 3XS fused hit kernel first; it
    // writes zeros for bank routes (their table entry is the bank zero block), which phase 1 then overwrites.
    const int64_t ids_stride = ids->nb[1] / sizeof(int32_t);
    const bool p1hitg = split && glu && moe_bank_p1_hitg() && ids->ne[2] == 1 && a->type == GGML_TYPE_F32 &&
            ggml_cuda_moe_table_mmv_hitg_supported(wtype, a->ne[0], glu->ne[0], ids->ne[0], ids->ne[1], a->ne[1],
                    ids_stride);
    const bool both = (p2list && glu != nullptr) || p1hitg;
    ggml_cuda_moe_mixed_pair resident = { bank2, base2, &op2, second.device, nullptr, second.hot, glu, both };
    resident.hot_only = p1hitg;
    auto phase1 = [&]() {
        ggml_cuda_mul_mat_id_mixed(ctx, bank, &op, &op, quantized, first.device, nullptr, first.hot,
                route_map, route_mask, bank_q.get(), bank_s.get(), 1, &resident);
        CUDA_CHECK(cudaGetLastError());
    };
    if (p1hitg) {
        const auto & up_t = first.mj_kind == 0 ? first : second;
        const auto & gate_t = first.mj_kind == 0 ? second : first;
        auto hit = [&]() {
            ggml_cuda_moe_table_mmv_hitg(ctx, up_t.device, gate_t.device, wtype, quantized, (const int32_t *) ids->data,
                    (float *) glu->data, a->ne[0], glu->ne[0], ids->ne[0], ids->ne[1], a->ne[1], ids_stride,
                    up_t.hot, gate_t.hot, 1);
        };
        g_st_p1hitg.fetch_add(1, std::memory_order_relaxed);
        if (moe_bank_p1_conc()) {
            moe_p1_conc_run(ctx, phase1, hit);   // split is set whenever p1hitg is
        } else {
            hit();
            phase1();
        }
    } else if (split) {
        phase1();
    }
    const void * const * eff = moe_hf_wait(ctx, session, first);
    const void * const * eff2 = moe_hf_wait(ctx, session, second);
    // fork AIPFIX P2SAME: the copied routes take the hit kernel again with the miss masks (the 3XS layer's miss pass):
    // bank routes are hot (skipped), phase 1 wrote them; resident routes were written by the hit pass above.
    if (p1hitg && moe_bank_p2_same()) {
        const bool up_first = first.mj_kind == 0;
        const auto & up_t = up_first ? first : second;
        const auto & gate_t = up_first ? second : first;
        const uint8_t * skip_up = skip_of(up_t);
        const uint8_t * skip_gate = skip_of(gate_t);
        ggml_cuda_moe_table_mmv_hitg(ctx, up_first ? eff : eff2, up_first ? eff2 : eff, wtype, quantized,
                (const int32_t *) ids->data, (float *) glu->data, a->ne[0], glu->ne[0], ids->ne[0], ids->ne[1],
                a->ne[1], ids_stride, skip_up ? skip_up : up_t.hot, skip_gate ? skip_gate : gate_t.hot, 0);
        CUDA_CHECK(cudaGetLastError());
        g_st_p2same.fetch_add(1, std::memory_order_relaxed);
        if (tail) {
            moe_tail_zero(ctx, ids, (int) first.desc.n_expert, glu);
        }
        return true;
    }
    bool listed = false;
    if (p2list) {
        const bool up_first = first.mj_kind == 0;
        const void * const * up_eff = up_first ? eff : eff2;
        const void * const * gate_eff = up_first ? eff2 : eff;
        ggml_tensor & up_op = up_first ? op : op2;
        ggml_tensor & gate_op = up_first ? op2 : op;
        const int64_t n_in = a->ne[0], n_out = op.ne[0], n_ids = ids->ne[0], n_tok = ids->ne[1], ch = a->ne[1];
        auto mixed_rest = [&](ggml_tensor * d1, ggml_tensor * d2, ggml_tensor * g) {
            const ggml_cuda_moe_mixed_pair rest = { bank2, base2, d2, eff2, skip_of(second), second.hot, g, both };
            ggml_cuda_mul_mat_id_mixed(ctx, bank, &op, d1, quantized, eff, skip_of(first), first.hot,
                    route_map, route_mask, bank_q.get(), bank_s.get(), 2, &rest);
        };
        if (glu) {
            // check mode: the mixed phase 2 into a copy of glu (its up/gate stores land in op/op2 as usual)
            listed = moe_miss_pass_list(ctx, session, first.mj_layer, 3, up_eff, gate_eff, wtype, quantized,
                    (float *) glu->data, n_in, n_out, n_ids, n_tok, ch, [&](float * out) {
                        ggml_tensor g = *glu;
                        g.data = out;
                        mixed_rest(&op, &op2, &g);
                    });
        } else {
            listed = moe_miss_pass_list(ctx, session, first.mj_layer, 0, up_eff, nullptr, wtype, quantized,
                    (float *) up_op.data, n_in, n_out, n_ids, n_tok, ch, [&](float * out) {
                        ggml_tensor o = up_op;
                        o.data = out;
                        mixed_rest(up_first ? &o : &op, up_first ? &op2 : &o, nullptr);
                    });
            // the up list ran, so the gate list (same checks) runs too
            const bool gate_listed = listed && moe_miss_pass_list(ctx, session, first.mj_layer, 1, gate_eff, nullptr,
                    wtype, quantized, (float *) gate_op.data, n_in, n_out, n_ids, n_tok, ch, [&](float * out) {
                        ggml_tensor o = gate_op;
                        o.data = out;
                        mixed_rest(up_first ? &op : &o, up_first ? &o : &op2, nullptr);
                    });
            GGML_ASSERT(gate_listed == listed);
        }
        if (listed) {
            CUDA_CHECK(cudaGetLastError());
            g_st_p2list.fetch_add(1, std::memory_order_relaxed);
            if (tail) {
                if (glu) {
                    moe_tail_zero(ctx, ids, (int) first.desc.n_expert, glu);
                } else {
                    moe_tail_zero(ctx, ids, (int) first.desc.n_expert, &op);
                    moe_tail_zero(ctx, ids, (int) first.desc.n_expert, &op2);
                }
            }
            return true;
        }
    }
    const ggml_cuda_moe_mixed_pair rest = { bank2, base2, &op2, eff2, skip_of(second), second.hot, glu, both };
    ggml_cuda_mul_mat_id_mixed(ctx, bank, &op, &op, quantized, eff, skip_of(first), first.hot,
            route_map, route_mask, bank_q.get(), bank_s.get(), split ? 2 : 0, &rest);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// fork AIPFIX D4: cached AI+ gate/up run as one paired AI+ GEMV per pass (gate in the upper half of the grid) and one
// SwiGLU (the GLU op's formula), instead of two GEMVs per pass and the GLU op. GGML_MOE_CACHE_AI_PAIR=0 opts out.
static bool moe_ai_pair_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_MOE_CACHE_AI_PAIR");
        return !e || atoi(e) != 0;
    }();
    return on;
}

// fork AIPFIX D6: the paired launches also write the SwiGLU (no separate kernel). GGML_MOE_CACHE_AI_PAIR_GLU=0 opts out.
static bool moe_ai_pair_glu_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_MOE_CACHE_AI_PAIR_GLU");
        return !e || atoi(e) != 0;
    }();
    return on;
}

static __global__ void moe_ai_swiglu(const float * gate, const float * up, float * dst, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = ggml_cuda_op_silu_single(gate[i]) * up[i];
    }
}

// fork: one activation quantization and one MMV retain both exact expert projections.
bool ggml_cuda_moe_cache_device_fused_compute(ggml_backend_cuda_context & ctx,
        ggml_tensor * up, ggml_tensor * gate, ggml_tensor * glu) {
    const char * enabled = getenv("GGML_MOE_CACHE_DEVICE_FUSED");
    const bool legacy_fused = enabled && strcmp(enabled, "1") == 0;
    // fork MOEGAP HITG: GGML_MOE_HIT_FUSED takes the hit-first layers through one fused gate/up/SwiGLU pass
    const bool hitg = ggml_cuda_moe_hit_fused_mode() != 0;
    if ((!legacy_fused && !hitg) || g_session_stack.empty() || !g_session_stack.back().active ||
            !up || !gate || !glu || glu->op != GGML_OP_GLU ||
            ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 ||
            glu->src[0] != gate || glu->src[1] != up ||
            !moe_cache_device_op_fits(*g_session_stack.back().active, up, ctx.device) ||
            !moe_cache_device_op_fits(*g_session_stack.back().active, gate, ctx.device) ||
            up->src[0]->type != gate->src[0]->type || up->src[1] != gate->src[1] || up->src[2] != gate->src[2] ||
            !ggml_are_same_shape(up, gate) || !ggml_are_same_shape(up, glu) ||
            glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu) ||
            (ggml_ai4_is_unit(up->src[0]->type) && (!hitg || !moe_ai_pair_enabled()))) {
        return false;
    }
    const bool ai = ggml_ai4_is_unit(up->src[0]->type);
    auto & session = *g_session_stack.back().active;
#if !defined(GGML_USE_HIP)
    // fork PREFETCHRW: table prefetch dispatches per projection; use the unfused path while it is active.
    if (!session.devices.empty() && session.devices[0]->table_prefetch_prepared) {
        return false;
    }
#endif
    const auto & up_table = session.device_tables.at(up->src[0]->data);
    const auto & gate_table = session.device_tables.at(gate->src[0]->data);
    const ggml_tensor * a = up->src[1];
    const ggml_tensor * ids = up->src[2];
    if (hitg) {
        const int64_t ids_stride = ids->nb[1] / sizeof(int32_t);
        const bool ok = up_table.mj_kind == 0 && gate_table.mj_kind == 1 &&
                up_table.mj_layer >= 0 && up_table.mj_layer == gate_table.mj_layer &&
                a->type == GGML_TYPE_F32 && ids->ne[2] == 1 &&
                (ai ? ggml_cuda_moe_ai_table_supported((ggml_type) up_table.desc.type, a->ne[0], glu->ne[0], ids->ne[1]) :
                ggml_cuda_moe_table_mmv_hitg_supported((ggml_type) up_table.desc.type, a->ne[0], glu->ne[0],
                    ids->ne[0], ids->ne[1], a->ne[1], ids_stride));
        if (ok) {
            moe_cache_swap_note_stream(session, ctx);
            if (moe_hf_begin(ctx, session, up_table, a, ids)) {
                GGML_ASSERT(moe_hf_begin(ctx, session, gate_table, a, ids));
                session.device_dispatches.fetch_add(2, std::memory_order_relaxed);
                if (session.device_expert_counts) {
                    const int n = (int)(ids->ne[0] * ids->ne[1]);
                    for (const auto * table : {&up_table, &gate_table}) {
                        if (moe_plan_count_take(session, *table, ids)) { continue; }
                        moe_cache_count_routes<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(
                                (const int32_t *)ids->data, table->hot, session.device_route_counts,
                                session.device_expert_counts + table->count_offset, n, table->desc.n_expert,
                                (int)ids->ne[0], (int) ids_stride);
                    }
                }
                ggml_cuda_pool_alloc<char> q8(ctx.pool());
                char * quantized = moe_quantize_act(ctx, a, (ggml_type) up_table.desc.type, true, q8);
                g_st_hitg.fetch_add(1, std::memory_order_relaxed);
                if (ai) {
                    const ggml_type wtype = (ggml_type) up_table.desc.type;
                    const int64_t n = ggml_nelements(glu);
                    static std::atomic<bool> logged{false};
                    if (!logged.exchange(true)) {
                        fprintf(stderr, "[moe-cache-ai] first cached AI+ paired gate/up launch: type=%d tokens=%lld glu=%d\n",
                                (int) wtype, (long long) ids->ne[1], (int) moe_ai_pair_glu_enabled());
                    }
                    ggml_cuda_pool_alloc<float> up_out(ctx.pool(), n);
                    ggml_cuda_pool_alloc<float> gate_out(ctx.pool(), n);
                    const bool fglu = moe_ai_pair_glu_enabled() &&
                            glu->ne[0]/16*ids->ne[0]*ids->ne[1] <= 65536;
                    float * glu_out = fglu ? (float *) glu->data : nullptr;
                    auto pair = [&](const void * const * ut, const void * const * gt, const uint8_t * uf,
                            const uint8_t * gf, int val, const uint8_t * up_prev, const uint8_t * gate_prev) {
                        const bool pok = ggml_cuda_moe_ai_table_mmv(ut, wtype, quantized, (const int32_t *) ids->data,
                                up_out.get(), a->ne[0], glu->ne[0], ids->ne[0], ids->ne[1], a->ne[1], ids_stride,
                                (int) up_table.desc.n_expert, uf, val, ctx.stream(), gt, gate_out.get(), gf,
                                glu_out, up_prev, gate_prev);
                        GGML_ASSERT(pok);
                    };
                    pair(up_table.device, gate_table.device, up_table.hot, gate_table.hot, 1, nullptr, nullptr);
                    const void * const * up_eff = moe_hf_wait(ctx, session, up_table);
                    const void * const * gate_eff = moe_hf_wait(ctx, session, gate_table);
                    const bool cs = session.cs_layer >= 0 && session.cs_layer == up_table.mj_layer;
                    const bool tail = moe_tail_active(session, up_table);
                    const uint8_t * skip_up = cs ? g_cs_mask + (size_t) up_table.mj_kind * g_hf_experts :
                            tail ? g_tail_mask + (size_t) up_table.mj_kind * g_hf_experts : up_table.hot;
                    const uint8_t * skip_gate = cs ? g_cs_mask + (size_t) gate_table.mj_kind * g_hf_experts :
                            tail ? g_tail_mask + (size_t) gate_table.mj_kind * g_hf_experts : gate_table.hot;
                    pair(up_eff, gate_eff, skip_up, skip_gate, 0, up_table.hot, gate_table.hot);
                    if (!fglu) {
                        moe_ai_swiglu<<<(n + 255)/256, 256, 0, ctx.stream()>>>(gate_out.get(), up_out.get(),
                                (float *) glu->data, n);
                    }
                    CUDA_CHECK(cudaGetLastError());
                    if (tail) {
                        moe_tail_zero(ctx, ids, (int) up_table.desc.n_expert, glu);
                    }
                    return true;
                }
                // hit pass: experts resident for both projections
                ggml_cuda_moe_table_mmv_hitg(ctx, up_table.device, gate_table.device, (ggml_type) up_table.desc.type,
                        quantized, (const int32_t *) ids->data, (float *) glu->data, a->ne[0], glu->ne[0], ids->ne[0],
                        ids->ne[1], a->ne[1], ids_stride, up_table.hot, gate_table.hot, 1);
                moe_onepath_bank_pair(ctx, up_table, gate_table, a, ids, glu);
                const void * const * up_eff = moe_hf_wait(ctx, session, up_table);
                const void * const * gate_eff = moe_hf_wait(ctx, session, gate_table);
                // miss pass: every other expert the card computes (a CPU-split layer skips the host's experts)
                const bool cs = session.cs_layer >= 0 && session.cs_layer == up_table.mj_layer;
                const bool tail = moe_tail_active(session, up_table);
                const uint8_t * skip_up = cs ? g_cs_mask + (size_t) up_table.mj_kind * g_hf_experts :
                        tail ? g_tail_mask + (size_t) up_table.mj_kind * g_hf_experts : up_table.hot;
                const uint8_t * skip_gate = cs ? g_cs_mask + (size_t) gate_table.mj_kind * g_hf_experts :
                        tail ? g_tail_mask + (size_t) gate_table.mj_kind * g_hf_experts : gate_table.hot;
                auto old_miss = [&](float * out) {
                    ggml_cuda_moe_table_mmv_hitg(ctx, up_eff, gate_eff, (ggml_type) up_table.desc.type,
                            quantized, (const int32_t *) ids->data, out, a->ne[0], glu->ne[0], ids->ne[0],
                            ids->ne[1], a->ne[1], ids_stride, skip_up, skip_gate, 0);
                };
                if (!moe_miss_pass_list(ctx, session, up_table.mj_layer, 3, up_eff, gate_eff,
                        (ggml_type) up_table.desc.type, quantized, (float *) glu->data, a->ne[0], glu->ne[0],
                        ids->ne[0], ids->ne[1], a->ne[1], old_miss)) {
                    old_miss((float *) glu->data);
                }
                CUDA_CHECK(cudaGetLastError());
                if (tail) {
                    moe_tail_zero(ctx, ids, (int) up_table.desc.n_expert, glu);
                }
                moe_onepath_check(ctx, up_table, &gate_table, a, ids, (const float *) glu->data, glu->ne[0], "final");
                return true;
            }
        }
        if (!legacy_fused) {
            return false;
        }
    }
    session.device_dispatches.fetch_add(2, std::memory_order_relaxed);
    moe_cache_swap_note_stream(session, ctx);
    if (session.device_expert_counts) {
        const int n = (int)(ids->ne[0] * ids->ne[1]);
        for (const auto * table : {&up_table, &gate_table}) {
            if (moe_plan_count_take(session, *table, ids)) { continue; }
            moe_cache_count_routes<<<(n + 127) / 128, 128, 0, ctx.stream()>>>(
                    (const int32_t *)ids->data, table->hot, session.device_route_counts,
                    session.device_expert_counts + table->count_offset, n, table->desc.n_expert,
                    (int)ids->ne[0], (int)(ids->nb[1] / sizeof(int32_t)));
        }
    }
    ggml_cuda_pool_alloc<char> q8(ctx.pool());
    char * quantized = moe_quantize_act(ctx, a, (ggml_type) up_table.desc.type, true, q8);
    // Keep both projections in allocation order; the pool releases them in reverse order.
    moe_cache_effective_table up_effective(ctx.pool());
    moe_cache_effective_table gate_effective(ctx.pool());
    const void * const * up_mmv_table = moe_cache_resolve_table(ctx, session, up_table, ids, up_effective);
    const void * const * gate_mmv_table = moe_cache_resolve_table(ctx, session, gate_table, ids, gate_effective);
    if (!ggml_cuda_moe_table_mmv_grouped_try(ctx, up_mmv_table, gate_mmv_table, (ggml_type)up_table.desc.type, quantized,
            (const int32_t *)ids->data, (float *)glu->data, a->ne[0], glu->ne[0], ids->ne[0], ids->ne[1],
            a->ne[1], ids->nb[1] / sizeof(int32_t))) {
        ggml_cuda_moe_table_mmv_fused(up_mmv_table, gate_mmv_table, (ggml_type)up_table.desc.type, quantized,
                (const int32_t *)ids->data, (float *)glu->data, a->ne[0], glu->ne[0], ids->ne[0], ids->ne[1],
                a->ne[1], ids->nb[1] / sizeof(int32_t), ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
    moe_onepath_bank_pair(ctx, up_table, gate_table, a, ids, glu);
    if (up_effective.pf_marked || gate_effective.pf_marked) {
        // Overflow and unaligned misses can still read host memory after staging.
        moe_pf_mark_end_kernel<<<1, 1, 0, ctx.stream()>>>(g_pf.d_mark_seq, g_pf.d_mark);
    }
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

#if (defined(__linux__) || defined(_WIN32)) && CUDART_VERSION >= 11010 && !defined(GGML_USE_HIP)
static void * moe_cache_host_alias(const void * source, size_t bytes) {
    const uintptr_t begin = (uintptr_t)source;
    if (!source || !bytes || bytes > UINTPTR_MAX - begin ||
            !ggml_cuda_host_pin_contains(source, bytes)) {
        return nullptr;
    }
    // Endpoint attributes alone cannot exclude an unregistered interior hole.
    const void * last = (const void *)(begin + bytes - 1);
    cudaPointerAttributes first_attributes{}, last_attributes{};
    void * alias = nullptr;
    void * last_alias = nullptr;
    const bool mapped = cudaPointerGetAttributes(&first_attributes, source) == cudaSuccess &&
            first_attributes.type == cudaMemoryTypeHost &&
            cudaPointerGetAttributes(&last_attributes, last) == cudaSuccess &&
            last_attributes.type == cudaMemoryTypeHost &&
            cudaHostGetDevicePointer(&alias, const_cast<void *>(source), 0) == cudaSuccess &&
            cudaHostGetDevicePointer(&last_alias, const_cast<void *>(last), 0) == cudaSuccess &&
            alias && bytes - 1 <= UINTPTR_MAX - (uintptr_t)alias &&
            (uintptr_t)last_alias == (uintptr_t)alias + bytes - 1;
    (void)cudaGetLastError();
    return mapped ? alias : nullptr;
}
#endif

static int moe_cache_device_prepare(void * opaque, const ggml_moe_cache_tensor_desc * tensors, int n_tensors) {
#if (defined(__linux__) || defined(_WIN32)) && CUDART_VERSION >= 11010 && !defined(GGML_USE_HIP)
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
    session->device_publish_async = moe_cache_env_i64("GGML_MOE_CACHE_DEVICE_PUBLISH_ASYNC", 0, 1, adaptation_value) && adaptation_value;
    // fork CACHEHIT: decayed-score swap policy with no host stall. Default off.
    // With prefetch on, swaps also rewrite the staging tables and never touch staging slots.
    session->swap_enabled = moe_cache_env_i64("GGML_MOE_CACHE_SWAP", 0, 1, adaptation_value) && adaptation_value;
    if (session->swap_enabled) {
        if (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_INTERVAL", 1, 1024, adaptation_value)) { session->swap_interval = (int)adaptation_value; }
        if (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_MAX", 1, 4096, adaptation_value)) { session->swap_max = (int)adaptation_value; }
        if (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_MB", 1, 4096, adaptation_value)) { session->swap_cap_bytes = (size_t)adaptation_value << 20; }
        if (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_DECAY_PCT", 0, 100, adaptation_value)) { session->swap_decay = (float)adaptation_value / 100.0f; }
        if (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_HYST_PCT", 0, 1000, adaptation_value)) { session->swap_hyst = (float)adaptation_value / 100.0f; }
        if (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_MARGIN", 0, 100000, adaptation_value)) { session->swap_margin = (float)adaptation_value; }
        if (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_MARGIN_MILLI", 0, 100000000, adaptation_value)) { session->swap_margin = (float) adaptation_value / 1000.0f; }
        session->swap_whole = moe_cache_whole_only() ||
            (moe_cache_env_i64("GGML_MOE_CACHE_SWAP_WHOLE", 0, 1, adaptation_value) && adaptation_value);
        session->swap_layer = moe_cache_env_i64("GGML_MOE_CACHE_SWAP_LAYER", 0, 1, adaptation_value) && adaptation_value;
        session->swap_top_only = moe_cache_env_i64("GGML_MOE_CACHE_SWAP_TOP_ONLY", 0, 1, adaptation_value) && adaptation_value;
        session->swap_free_probe = moe_cache_env_i64("GGML_MOE_CACHE_SWAP_FREE_PROBE", 0, 1, adaptation_value) && adaptation_value;
        session->swap_cold_prune = moe_cache_env_i64("GGML_MOE_CACHE_SWAP_COLD_PRUNE", 0, 1, adaptation_value) && adaptation_value;
        session->swap_decay_after = moe_cache_env_i64("GGML_MOE_CACHE_SWAP_DECAY_AFTER", 0, 1, adaptation_value) && adaptation_value;
        MOE_CACHE_LOG("[moe-cache] swap modes: whole=%d layer=%d decay_after=%d\n", session->swap_whole, session->swap_layer, session->swap_decay_after);
        MOE_CACHE_LOG("[moe-cache] swap policy: interval_scopes=%d max_swaps=%d cap_bytes=%zu decay=%.2f hyst=%.2f margin=%.1f; async publication, no device sync\n",
                session->swap_interval, session->swap_max, session->swap_cap_bytes, session->swap_decay,
                session->swap_hyst, session->swap_margin);
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
    for (auto & d : session->devices) { moe_cache_det_publish_locked(*d); }
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
#ifdef _WIN32
        const size_t page = 4096;  // x64 Windows page size (no sysconf)
#else
        const size_t page = (size_t) sysconf(_SC_PAGESIZE);
#endif
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
        const bool resident_mask = moe_cache_table_mask_enabled();
        const bool mask_only = resident_mask && !route_stats;
        size_t table_bytes = route_stats ? 3 * sizeof(unsigned long long) : 0, hot_bytes = 0;
        size_t count_entries = 0, arena_bytes = 0;
        size_t bank_zero_bytes = 0, bank_entries = 0;
        for (int i = 0; i < n_tensors; ++i) {
            const auto & d = tensors[i];
            if (d.data && ggml_moe_bank_lookup(d.data) && d.n_in > 0 && d.n_out > 0) {
                bank_zero_bytes = std::max(bank_zero_bytes, ggml_row_size((ggml_type)d.type, d.n_in) * (size_t)d.n_out);
            }
        }
        ggml_cuda_set_device(device.logical);
        const void * bank_zero = bank_zero_bytes ? moe_bank_zero_block(device.logical, bank_zero_bytes) : nullptr;
        ggml_cuda_moe_set_bank_zero(moe_bank_skip_enabled() ? bank_zero : nullptr);
        if (bank_zero_bytes && !bank_zero) {
            MOE_CACHE_LOG("[moe-cache] device tables skipped: bank zero block allocation failed\n");
            return fail();
        }
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
            table.permanent_sources = moe_resident_tensor(d.data, device.logical);
            // A bank tensor can span pinned cold ranges and unregistered holes.
            const bool bank_sources = ggml_moe_bank_lookup(d.data) != nullptr;
            // fork: session-owned registrations must pass through retained_needed before pruning.
            if (!table.permanent_sources && !mappings.find(base, bytes)) {
                table.borrowed = moe_cache_host_alias(d.data, bytes);
            }
            (void)cudaGetLastError();
            table.addresses.resize(d.n_expert);
            table.permanent.assign(d.n_expert, 0);
            table.bank.assign(d.n_expert, 0);
            table.fallback.resize(d.n_expert);
            table.copy_source.resize(d.n_expert);
            table.count_offset = count_entries;
            count_entries += d.n_expert;
            for (int32_t e = 0; e < d.n_expert; ++e) {
                const void * permanent = moe_resident_lookup_device((const char *)d.data + (size_t)e * d.expert_size,
                        expert_bytes, device.logical);
                const void * address = permanent;
                if (permanent) { hot_bytes += expert_bytes; table.permanent[e] = 1; }
                if (!permanent && bank_zero && moe_bank_has(d.data, e)) {
                    address = bank_zero;
                    table.permanent[e] = 1;
                    table.bank[e] = 1;
                    ++table.n_bank;
                    ++bank_entries;
                }
                for (const auto & pool : device.pools) {
                    if (permanent) { break; }
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
                if (table.permanent_sources) {
                    if (!permanent) {
                        void * alias = moe_cache_host_alias((const void *)begin, expert_bytes);
                        if (!alias) { return fail(); }
                        table.fallback[e] = alias;
                    }
                } else {
                    // Bank release leaves the cold experts pinned in separate ranges.
                    void * alias = nullptr;
                    if (!table.borrowed && bank_sources && !table.bank[e] && !mappings.find(begin, expert_bytes) &&
                            (alias = moe_cache_host_alias((const void *)begin, expert_bytes))) {
                        table.fallback[e] = alias;
                        table.copy_source[e] = (const void *)begin;
                    } else {
                        table.fallback[e] = table.borrowed ? (const char *)table.borrowed + (size_t)e * table.host_stride :
                            mappings.find(begin, expert_bytes, &source_range);
                    }
                    (void)cudaGetLastError();
                }
                if (!table.permanent_sources && !table.borrowed && !table.copy_source[e] && !address && table.fallback[e]) { retained_needed[source_range] = 1; }
                if (!address && !table.fallback[e]) {
                    pending_ranges.push_back({begin & ~(page - 1),
                            (begin + expert_bytes + page - 1) & ~(page - 1), nullptr, nullptr, false});
                }
                table.addresses[e] = address;
            }
            table_bytes += (size_t) d.n_expert * (sizeof(void *) * (session->prefetch_enabled ? 2 : 1) + sizeof(unsigned long long) + (resident_mask ? sizeof(uint8_t) : 0));
            if (!moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert * sizeof(void *)) ||
                    (session->prefetch_enabled && !moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert * sizeof(void *))) ||
                    (resident_mask && !moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert)) ||
                    (moe_onepath() && table.n_bank > 0 &&
                        (!moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert * sizeof(void *)) ||
                         !moe_cache_table_bytes_add(arena_bytes, (size_t)d.n_expert)))) {
                return fail();
            }
            auto & stored = session->device_tables.emplace(d.data, std::move(table)).first->second;
            stored.name = d.name ? d.name : "";
            stored.desc.name = d.name ? stored.name.c_str() : nullptr;
        }
        if (!moe_cache_table_bytes_add(arena_bytes, count_entries * sizeof(unsigned long long)) ||
                (route_stats && !moe_cache_table_bytes_add(arena_bytes, 3 * sizeof(unsigned long long)))) {
            return fail();
        }
        // Retire ranges no current miss needs before adding new exact copies.
        // Old kernels and tables were drained by the idle rerank; no published
        // address can observe this pruning. Unchanged cold sets copy nothing.
        size_t kept = 0, pruned_bytes = 0;
        if (retained_ranges) { CUDA_CHECK(cudaDeviceSynchronize()); }
        for (size_t i = 0; i < retained_ranges; ++i) {
            auto & range = session->mapped_ranges[i];
            if (!retained_needed[i]) {
                if (range.registered) { ggml_cuda_host_unregister((void *) range.begin); }
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
#ifdef _WIN32
            new_bytes += bytes;  // no mincore: count every page as newly resident (conservative)
#else
            std::vector<unsigned char> residency(bytes / page);
            if (mincore((void *)range.begin, bytes, residency.data()) != 0) {
                return fail();
            }
            for (auto resident : residency) {
                new_bytes += (resident & 1) ? 0 : page;
            }
#endif
        }
        // GGML_MOE_DEVICE_RAM_FLOOR_MIB / GGML_MOE_DEVICE_RAM_MARGIN_MIB override the pinning headroom (defaults 7 GiB / 2 GiB).
        const char * floor_env = getenv("GGML_MOE_DEVICE_RAM_FLOOR_MIB");
        const char * margin_env = getenv("GGML_MOE_DEVICE_RAM_MARGIN_MIB");
        const size_t margin = ranges.empty() ? 0 : (margin_env ? (size_t) atoll(margin_env) << 20 : 2ull << 30);
        const size_t floor = floor_env ? (size_t) atoll(floor_env) << 20 : 7ull << 30;
        const size_t available = moe_cache_mem_available();
        if (bank_entries) {
            MOE_CACHE_LOG("[moe-cache] device tables: %zu bank expert entries point at a %zu KiB zero block (computed by the bank call)\n",
                    bank_entries, bank_zero_bytes >> 10);
        }
        MOE_CACHE_LOG("[moe-cache] device preflight: hot=%zu MiB map=%zu MiB nonresident=%zu MiB ranges=%zu available=%zu MiB\n",
                hot_bytes >> 20, pin_bytes >> 20, new_bytes >> 20, ranges.size(), available >> 20);
        // The RAM floor guards new pinning only: with no ranges there is nothing to pin, so a host just under
        // 7 GiB free must not fall back to legacy host routing (~35% slower writing at home, 2026-10-22).
        if (pin_bytes > SIZE_MAX - floor - margin || (!ranges.empty() && available < pin_bytes + floor + margin) ||
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
        const size_t physical_free_floor = moe_cache_physical_free_floor();
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
                ? ggml_cuda_host_register((void *)range.begin, bytes, registration_flags)
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
        if (moe_missjoin_mode() && !g_missjoin_counts) {
            if (!mask_only && (cudaMalloc((void **) &g_missjoin_counts, moe_missjoin_count_entries * sizeof(unsigned long long)) != cudaSuccess ||
                    cudaMemset(g_missjoin_counts, 0, moe_missjoin_count_entries * sizeof(unsigned long long)) != cudaSuccess)) {
                g_missjoin_counts = nullptr;
                (void) cudaGetLastError();
            }
            MOE_CACHE_LOG("[moe-cache] missjoin mode=%s slots=%d%s\n", moe_missjoin_mode() == 1 ? "stage" : "diag_hot",
                    moe_missjoin_slots(), moe_missjoin_mode() == 2 ? " (DIAGNOSTIC, LOSSY: misses redirected)" : "");
            if (!resident_mask) {
                MOE_CACHE_LOG("[moe-cache] missjoin needs GGML_MOE_MISSJOIN_MASK=1, GGML_MOE_MISSJOIN_AUTOMASK=1 or GGML_MOE_CACHE_DEVICE_STATS=1; inactive\n");
            }
        }
        if (mask_only) {
            MOE_CACHE_LOG("[moe-cache] missjoin resident_mask=1 route_stats=0\n");
        }
        for (auto & entry : session->device_tables) {
            auto & table = entry.second;
            if (resident_mask) {
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
                const bool borrowed_expert = table.copy_source[e] != nullptr;
                if (!table.permanent_sources && !borrowed_expert) {
                    table.fallback[e] = table.borrowed ? (const char *)table.borrowed + e * table.host_stride :
                        mappings.find(host, table.desc.expert_size, &source_range);
                }
                table.copy_source[e] = borrowed_expert || table.borrowed || (table.permanent_sources && table.fallback[e]) ? (const void *)host : nullptr;
                if (table.fallback[e] && !borrowed_expert && !table.borrowed && !table.permanent_sources) {
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
            // fork AIPFIX ONEPATH: the bank experts' own rows, for the AI+ table GEMV beside the regular passes
            const void * rows = nullptr;
            int32_t rows_type = -1;
            size_t rows_nb2 = 0;
            const int16_t * rows_slot = nullptr;
            if (moe_onepath() && table.n_bank > 0 &&
                    ggml_moe_bank_rows(table.desc.data, &rows, &rows_type, &rows_nb2, &rows_slot) &&
                    ggml_ai4_is_unit((ggml_type) rows_type)) {
                std::vector<const void *> ai(table.addresses.size(), nullptr);
                std::vector<uint8_t> flag(table.addresses.size(), 0);
                for (size_t e = 0; e < ai.size(); ++e) {
                    if (table.bank[e] && rows_slot[e] >= 0) {
                        ai[e] = (const char *) rows + (size_t) rows_slot[e] * rows_nb2;
                        flag[e] = 1;
                    }
                }
                table.ai_bank = (const void **) table_take(bytes);
                table.ai_bank_flag = (uint8_t *) table_take(flag.size());
                if (!table.ai_bank || !table.ai_bank_flag ||
                        cudaMemcpy(table.ai_bank, ai.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
                        cudaMemcpy(table.ai_bank_flag, flag.data(), flag.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
                    return fail();
                }
                table.ai_bank_type = rows_type;
                table.ai_bank_rows = std::move(ai);
                table.ai_bank_host = std::move(flag);
            }

        }
        // Same rule as the preflight: the floor guards pages pinned here; no new ranges means nothing was pinned.
        if ((!ranges.empty() && moe_cache_mem_available() < floor) ||
                !moe_cache_cgroup_headroom(margin, registered_bytes, &peak_service_current)) {
            return fail();
        }
        MOE_CACHE_LOG("[moe-cache] host staging bytes: copied=%zu registered=%zu chunk_cap=%zu peak_service_current=%llu available=%zu\n",
                staging_bytes, registered_bytes, staging_chunk_cap,
                (unsigned long long) peak_service_current, moe_cache_mem_available());
        moe_missjoin_overlap_prepare(*session);
        moe_missjoin_retained_prepare(*session);
        moe_pf_prepare(*session);
        moe_table_prefetch_prepare(*session);
        session->mapped_sources.clear();
        for (int i = 0; i < n_tensors; ++i) { session->mapped_sources.emplace(tensors[i].data, tensors[i]); }
        if (!session->device_profile_path.empty() && session->device_profile_tensors.size() == (size_t) n_tensors) {
            bool same = true;
            for (int i = 0; i < n_tensors; ++i) {
                const auto & a = tensors[i];
                const auto & b = session->device_profile_tensors[i];
                same = same && a.data == b.data && a.expert_size == b.expert_size && a.type == b.type &&
                    a.n_in == b.n_in && a.n_out == b.n_out && a.n_expert == b.n_expert && strcmp(a.name, b.name) == 0;
            }
            if (same) {
                for (int i = 0; i < n_tensors; ++i) {
                    const auto & table = session->device_tables.at(tensors[i].data);
                    session->device_profile_slices.push_back({table.count_offset, (size_t) tensors[i].n_expert});
                }
            }
        }
        if (moe_cache_whole_only() && !moe_cache_swap_groups(*session, true)) {
            MOE_CACHE_LOG("[moe-cache] whole-only device tables refused: resident bundles are incomplete\n");
            return fail();
        }
        session->device_capture_generation = g_moe_cache_capture_generation.fetch_add(1, std::memory_order_relaxed);
        GGML_ASSERT(session->device_capture_generation != 0);
        if (getenv("GGML_MOE_SKIP_TRACE")) {
            CUDA_CHECK(cudaMalloc(&session->mass_trace, 16384 * sizeof(moe_mass_record)));
            CUDA_CHECK(cudaMalloc((void **) &session->mass_trace_counts, 2 * sizeof(unsigned)));
            CUDA_CHECK(cudaMemset(session->mass_trace_counts, 0, 2 * sizeof(unsigned)));
        }
        session->device_ready.store(true, std::memory_order_release);
        if (moe_onepath()) {
            // fork AIPFIX ONEPATH: the graph takes the one path only while every banked table has its bank rows
            size_t banked = 0, with_rows = 0;
            for (const auto & entry : session->device_tables) {
                banked += entry.second.n_bank > 0;
                with_rows += entry.second.n_bank > 0 && entry.second.ai_bank != nullptr;
            }
            ggml_moe_bank_onepath_set_live(banked > 0 && with_rows == banked);
            MOE_CACHE_LOG("[moe-cache] one path %s: %zu of %zu banked tensors have bank rows\n",
                    banked > 0 && with_rows == banked ? "live" : "off (bank graph kept)", with_rows, banked);
        }
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

static void moe_cache_set_skip_tail(void * opaque, int rank, bool mass) {
    auto * session = (moe_cache_session *) opaque;
    if (!session || rank < 0 || rank > 9) {
        return;
    }
    const char * policy = getenv("GGML_MOE_SKIP_POLICY");
    int mode = !policy || strcmp(policy, "rank") == 0 ? 0 : strcmp(policy, "threshold") == 0 ? 1 : strcmp(policy, "cumulative") == 0 ? 2 : -1;
    const char * t = getenv("GGML_MOE_SKIP_THRESHOLD");
    char * end = nullptr;
    const float threshold = mode > 0 && t ? strtof(t, &end) : 0.0f;
    if (mode > 0 && !(t && end != t && *end == 0 && std::isfinite(threshold) && ggml_moe_skipmass_budget_valid(mode, threshold))) {
        mode = -1; // asserted where the drop kernel would use it
    }
    const char * r = getenv("GGML_MOE_SKIP_RENORM");
    const bool renorm = r && strcmp(r, "1") == 0;
    const char * f = getenv("GGML_MOE_SKIP_RANK_FLOOR");
    char * floor_end = nullptr;
    const long floor_value = mode == 2 && f ? strtol(f, &floor_end, 10) : 0;
    if (mode == 2 && f && !(floor_end != f && *floor_end == 0 && floor_value >= 0 && floor_value <= 10)) {
        mode = -1;
    }
    const int rank_floor = mode == 2 ? (int) floor_value : 0;
    // fork AIPFIX SKIPBANK: optional per-bank-layer floor and budget (cumulative policy only; invalid values = off)
    int bank_floor = -1;
    float bank_threshold = -1.0f;
    if (mode == 2) {
        const char * bf = getenv("GGML_MOE_SKIP_BANK_FLOOR");
        const char * bb = getenv("GGML_MOE_SKIP_BANK_BUDGET");
        char * e1 = nullptr;
        char * e2 = nullptr;
        const long v1 = bf ? strtol(bf, &e1, 10) : -1;
        const float v2 = bb ? strtof(bb, &e2) : -1.0f;
        bank_floor = bf && e1 != bf && *e1 == 0 && v1 >= 0 && v1 <= 10 ? (int) v1 : -1;
        bank_threshold = bb && e2 != bb && *e2 == 0 && std::isfinite(v2) && ggml_moe_skipmass_budget_valid(2, v2) ? v2 : -1.0f;
    }
    // A positive legacy rank still enables an environment-selected mass policy.
    mass = mode != 0 && (mass || rank > 0);
    std::lock_guard<std::mutex> lock(session->mu);
    const bool policy_changed = session->skip_mass != mass || session->skip_mode != mode || session->skip_threshold != threshold || session->skip_renorm != renorm || session->skip_rank_floor != rank_floor ||
        session->skip_bank_floor != bank_floor || session->skip_bank_threshold != bank_threshold;
    if (policy_changed) {
        session->skip_logged = false;
    }
    session->skip_mode = mode;
    session->skip_mass = mass;
    session->skip_rank_floor = rank_floor;
    session->skip_threshold = threshold;
    session->skip_bank_floor = bank_floor;
    session->skip_bank_threshold = bank_threshold;
    session->skip_renorm = renorm;
    if ((rank > 0 || mass) && !session->skip_logged) {
        static const char * names[] = {"invalid", "rank", "threshold", "cumulative"};
        MOE_CACHE_LOG("[moe-skip] policy=%s tail=%d threshold=%g renorm=%d rank_floor=%d\n", names[mode + 1], rank, threshold, renorm ? 1 : 0, rank_floor);
        if (bank_floor >= 0 || bank_threshold >= 0.0f) {
            MOE_CACHE_LOG("[moe-skip] bank layers: rank_floor=%d threshold=%g (-1 = same as other layers)\n", bank_floor, bank_threshold);
        }
        session->skip_logged = true;
    }
    if (session->skip_tail == rank && !policy_changed) {
        return;
    }
    session->skip_tail = rank;
    if (session->device_ready.load()) {
        // The rank decides which kernels a graph contains; never replay a graph built under the other policy.
        session->device_capture_generation = g_moe_cache_capture_generation.fetch_add(1, std::memory_order_relaxed);
    }
}

static void moe_cache_register(const void * owner) {
    g_moe_cache_owner = owner;
    ggml_moe_cache_api api = {};
    api.owner = owner;
    api.resident_create = moe_resident_create;
    api.resident_destroy = moe_resident_destroy;
    api.resident_spans = moe_resident_spans;
    api.resident_dropped = moe_resident_dropped;
    api.host_restore = moe_resident_host_restore;
    api.resident_lookup = moe_resident_lookup;
    api.resident_copy = moe_resident_copy;
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
    api.decode_capacity = moe_cache_decode_capacity;
    api.trim = moe_cache_lend;
    api.restore = moe_cache_return;
    api.prompt_resume = moe_cache_prompt_resume;
    api.resident = moe_cache_resident;
    api.set_skip_tail = moe_cache_set_skip_tail;
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
