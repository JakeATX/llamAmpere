#pragma once

// fork (INKCACHEUX): expert-cache policy and user controls for llama_moe_cache, all off by default.
//
//   GS1 user controls (read once when the cache is built):
//     LLAMA_MOE_CACHE_DEV_MIB=a,b,..  cache MiB per GPU (cache devices in backend order; one value = every GPU);
//                                     replaces the --moe-cache-mib total and its split
//     LLAMA_MOE_CACHE_SLOTS=N         fixed slot count (experts) per layer group instead of a byte budget;
//                                     wins over LLAMA_MOE_CACHE_DEV_MIB
//     LLAMA_MOE_CACHE_LAYERS=0,4-20   only these layers are cached (a looped layer's visits follow its first visit)
//     LLAMA_MOE_CACHE_REQ_STATS=1     one hit/miss/upload line per request (WARN level, so the default server log shows it) (llama_moe_cache_request_end)
//     LLAMA_MOE_CACHE_DRAFT_MIB=N     (common/speculative.cpp) a draft model context gets its own N MiB cache
//   GS4 heat profile:
//     LLAMA_MOE_CACHE_HEAT=path       decode use rate per expert; loaded at start (warm slots), saved at exit and at
//                                     a request end at most every LLAMA_MOE_CACHE_HEAT_SAVE_S (60) seconds; the saved
//                                     rate is old*(1-w) + session*w, w = LLAMA_MOE_CACHE_HEAT_WEIGHT (0.5)
//   #249 + GS9 miss split (needs --moe-cache-cpu-share > 0, which builds the CPU part of the graph):
//     LLAMA_MOE_CACHE_SPLIT=minimax   per lookup, choose how many misses go over PCIe from a fitted cost model
//                                     (CPU a + c*cpu + d*pcie, GPU g0 + g*pcie, minimize the max)
//     LLAMA_MOE_CACHE_SPLIT=sched     one PCIe fraction for all lookups, start 1-cpu_share clamped to [0.10, 0.60],
//                                     at most 3 updates after 48, 96 and 96 decode windows from the same fit
//     LLAMA_MOE_CACHE_SPLIT_Q256=N    fixed PCIe fraction N/256 (no tuning)
//   SPLITCAL calibrated prior (with minimax or sched; off when unset, then nothing above changes):
//     LLAMA_MOE_CACHE_SPLIT_CAL=1     at boot, time the CPU expert path (the real expert types on the host experts,
//                                     GGML_CPU_AI4_AVX2 as set) and an expert upload, and seed the fit with them:
//                                     slopes c and g from the timings (weight ~8 samples, forgotten like real ones),
//                                     intercepts and contention left to the samples; the fit is ready at once (no
//                                     cpu_share fallback). SCHED starts at the fraction of the update rule, and its
//                                     clamp widens to [0.05, 0.95]
//     LLAMA_MOE_CACHE_SPLIT_CAL=c,g   the same from given ms per expert (CPU, PCIe) instead of boot timings
//     LLAMA_MOE_CACHE_FEWROW_CPU=auto (llama-moe-cache.cpp) R = most rows per expert whose boot CPU time is below
//                                     the upload time (0 = every miss uploads); logged with [SPLITCAL]
//   GS8 decode boundary overlap:
//     LLAMA_MOE_CACHE_BOUNDARY=1      the per-window cache work (round admission ranking and swaps, split-policy
//                                     window close) runs when a decode step has been submitted, while the device still
//                                     computes its tail, instead of at the first lookup of the next step

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_tensor;

// "0,4-20" -> mask[n_layer]; false and err on a bad list or a layer out of range
bool llama_moe_cache_parse_layers(const char * text, int32_t n_layer, std::vector<uint8_t> & mask, std::string & err);
// "a,b,c" MiB -> values; false on a bad list
bool llama_moe_cache_parse_mib_list(const char * text, std::vector<double> & mib);

struct llama_moe_cache_controls {
    std::vector<double>  dev_mib;   // empty = off
    int32_t              slots = 0; // 0 = off
    std::vector<uint8_t> layers;    // empty = every layer
    bool                 req_stats = false;
    bool                 boundary  = false;

    void from_env(int32_t n_layer);
    bool any() const { return !dev_mib.empty() || slots > 0 || !layers.empty(); }
    bool layer_selected(int32_t il) const { return layers.empty() || (il >= 0 && il < (int32_t) layers.size() && layers[il]); }
    // MiB for cache device id, -1 if the list is off
    double dev_budget_mib(size_t id) const;
};

// GS4: decode use rate per (layer, expert) key; key = il*n_expert + e as in moe_cache_lru
struct llama_moe_cache_heat {
    std::string path;
    int32_t  n_layer  = 0;
    int32_t  n_expert = 0;
    uint64_t fingerprint = 0;
    double   weight = 0.5;
    int32_t  save_s = 60;

    std::vector<float>    rate_old;  // loaded rate per key, empty if nothing was loaded
    std::vector<uint64_t> uses;      // this session: decode rows routed to each key
    std::vector<uint64_t> rows;      // this session: decode rows seen by each layer
    uint64_t rows_saved = 0;         // session rows at the last save
    std::chrono::steady_clock::time_point t_save;

    bool on() const { return !path.empty(); }
    // reads LLAMA_MOE_CACHE_HEAT*; off if unset
    void init(int32_t n_layer, int32_t n_expert, uint64_t fingerprint);
    bool load();                                   // false: no file, or it does not match this model and cache
    void observe(int32_t il, const int32_t * ids, size_t n_ids);
    void add_rows(int32_t il, int64_t n_tokens) { rows[il] += (uint64_t) n_tokens; }
    std::vector<float> rates() const;              // blended old and session rates
    bool save();
    void maybe_save();                             // at a request end: if save_s elapsed and new rows were seen
};

// FNV-1a over the layout of the cached layers (callers feed names, types and shapes)
struct llama_moe_cache_fp {
    uint64_t h = 1469598103934665603ull;
    void add(const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *) p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    }
    void add_tensor(int32_t il, const ggml_tensor * t);
};

// #249 + GS9: how many of a lookup's misses go over PCIe once the cache is full
struct llama_moe_split_policy {
    enum mode_t { OFF, MINIMAX, SCHED, FIXED };
    mode_t   mode = OFF;
    uint32_t share_q_cfg = 0; // configured CPU share in 1/65536 (fallback while the model is not ready)
    double   frac = 0.0;      // SCHED/FIXED: PCIe fraction

    // online fit, exponentially weighted least squares (forgetting 0.995), ms
    struct ewls {
        int k = 0;
        double A[3][3] = {};
        double b[3] = {};
        double theta[3] = {};
        int64_t n = 0;
        void add(const double * x, double y);
        void solve(bool cpu_branch);
        double predict(const double * x) const { double r = 0; for (int i = 0; i < k; ++i) r += theta[i]*x[i]; return r; }
    };
    ewls cpu; // x = [1, n_cpu, n_pcie]
    ewls gpu; // x = [1, n_pcie]
    int64_t n_gpu_pcie = 0; // GPU samples with PCIe work

    // the lookup in flight (its misses) and the timing of the previous one
    int32_t prev_il = -1;
    int64_t prev_rows = 0;
    bool    have_pending = false;
    size_t  pend_n = 0, pend_m = 0;
    std::chrono::steady_clock::time_point t_prev;

    int64_t n_windows = 0, n_lookups = 0, n_explore = 0, n_obs = 0, n_drop = 0;
    int32_t n_updates = 0;
    std::vector<double> frac_hist;

    // SPLITCAL (LLAMA_MOE_CACHE_SPLIT_CAL): cal_c, cal_g given ms per expert (0 = time them at boot)
    bool   cal_want = false, cal_seeded = false;
    double cal_c = 0.0, cal_g = 0.0;

    void init(float cpu_share);  // reads LLAMA_MOE_CACHE_SPLIT*; mode OFF if unset or no CPU share
    // SPLITCAL: prior from c (CPU) and g (PCIe) ms per expert
    void seed(double c_ms, double g_ms);
    bool active() const { return mode != OFF; }
    bool ready() const { return cpu.n >= 8 && gpu.n >= 8 && n_gpu_pcie >= 4; }
    // routing by the fit waits for real samples on both branches (the SPLITCAL seed sets n but has no intercepts)
    int64_t n_real_cpu = 0, n_real_gpu = 0;
    bool route_ready() const { return ready() && n_real_cpu >= 4 && n_real_gpu >= 4; }

    // right after the ids of layer il (n_rows rows) are on the host: closes the previous lookup's interval
    void on_ids(int32_t il, int64_t n_rows);
    // CPU count for n_miss misses; carry is the layer's accumulator in 1/65536 (fallback and SCHED/FIXED)
    size_t n_cpu(size_t n_miss, uint32_t & carry);
    // what plan() did with this lookup (the zero slots can send CPU misses back to PCIe)
    void record(size_t n_cpu, size_t n_pcie);
    // a decode window ended (GS8 boundary, or the next window's first lookup)
    void window_end();
    void log_summary() const;
};
