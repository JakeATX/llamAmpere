// [#22/#29] Bounded f16 prefill (GGML_CUDA_PREFILL_KV_MIB): on/off comparison of FLASH_ATTN_EXT outputs.
//
// GGML_CUDA_PREFILL_KV_MIB is read once per process, so every budget arm runs in its own forked child. The parent never
// touches a ggml backend (CUDA does not survive fork): it builds the inputs (quantized with ggml_quantize_chunk) and the
// f64 CPU reference, then forks one child per arm. The child sets the env, loads the backends and runs every case.
// Arm "unset" runs first and sends its outputs back through a pipe. Later children inherit those outputs and compare
// against them.
//
// Layout: the model's. Q is the permuted view [D, n_q, n_head] of [D, n_head, n_q]. K and V are the permuted views
// [D, n_kv, n_head_kv] of the cache [D, n_head_kv, n_kv], so a head slice is strided, as in llama-server. The mask is
// f16 causal over the last n_q positions. Results are f32 (GGML_PREC_F32).
//
// Cases:
//   identity  131,072 KV / 1,024 Q, D 256, 4 KV heads, GQA 4 and 6, K/V q8_0/q8_0 and tq5_0/turbo4. The f16 copies
//             are 512 MiB, so every budget arm below takes the bounded plan here. f64 reference on every 32nd query row
//             plus the last.
//   prod      the ATX attention shape (server log: 24 query heads, 4 KV heads, GQA 6, D 256, tq5_0/turbo4) at
//             262,144 KV / 1,024 Q. Every budget arm takes 1 KV head per group. Same reference rows as identity.
//   tolerance 8,192 KV / 128 Q, the same pairs, GQA 6 and 4, plus a sinks case. The f16 copies are 32 MiB, so
//             budgets 1/20/27 give 1/2/3 KV heads per group, and 256/272/410 fit the full copies (plan off, output
//             identical). f64 reference on every 4th query row plus the last.
//   excluded  ALiBi (max_bias 8), D 128, and a single KV head (8,192 / 128). Gate: identical to "unset" and no bounded
//             ledger row.
// Gates for every bounded case. The bounded route runs the unbounded kernel once per group with fewer KV heads, so only
// the grid differs. Whether that changes the summation order is decided by mma_grid() below, which mirrors the
// stream-k choice in launch_fattn:
//   same order (neither the unbounded launch nor any group launch uses stream-k): output bytes equal to arm "unset".
//   order differs, or cannot be derived (unknown GPU): with R = max |reference| and errors taken on the reference rows,
//     a: max |bounded - ref| <= max |unbounded - ref| * 1.10
//     b: max |bounded - unbounded| <= max |unbounded - ref| + 5e-3 * R
//   The CASE line prints max |bounded - unbounded| (all rows and reference rows), both errors vs the reference, and R.
// dst view: the first tolerance case with dst as a view of a plain tensor. Gate: identical to the reserved run in the
// same child, and the ledger shows scratch=bounded-pool.
// For every bounded case the fallback ledger ("cuda.fattn" f16_convert rows) must show exactly one live bounded row,
// counted once, with "heads=h/n" equal to the expected heads per group (computed independently below) and the case's
// KV head count, and no full-copy row. Ledger slots outlive ggml_ledger_reset (it zeroes the counts), so rows with
// count 0 belong to earlier cases and are skipped. Arm "unset" must show no bounded row.
// PLAN lines: ggml_backend_buft_get_alloc_size for the ATX FLASH_ATTN_EXT shape (24 query heads, 4 KV heads, D 256,
// n_q 1,024, tq5_0 K / turbo4 V) at n_kv 100,352 / 131,072 / 245,760 / 262,144. For bounded arms the size must equal
// pad128(nbytes(dst)) + max(heads * (2 * kv_per_head + out_per_head), budget).
// Graph arms (CUDA graphs on): the same 8,192 / 128 q8_0 graph computed 4 times. With the env unset a capture must happen.
// With budget 1 the graph must stay eager with the ledger outcome "eager:incompatible_fattn_bounded_prefill".
// Needs a GPU backend; skips (exit 0) without one, and on Windows (no fork).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ledger.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

enum case_kind { KIND_IDENTITY, KIND_TOLERANCE, KIND_EXCLUDED };

struct case_def {
    case_kind kind;
    ggml_type tk;
    ggml_type tv;
    int64_t   D;
    int64_t   n_kv;
    int64_t   n_q;
    int64_t   n_head;
    int64_t   n_head_kv;
    bool      sinks;
    float     max_bias;
    const char * what;
};

static std::vector<case_def> make_cases() {
    std::vector<case_def> c;
    const std::pair<ggml_type, ggml_type> pairs[2] = { { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0 }, { GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0 } };
    for (const auto & p : pairs) {
        for (int64_t gqa : { 4, 6 }) {
            c.push_back({ KIND_IDENTITY, p.first, p.second, 256, 131072, 1024, 4*gqa, 4, false, 0.0f, "identity" });
        }
    }
    c.push_back({ KIND_IDENTITY, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0, 256, 262144, 1024, 24, 4, false, 0.0f, "identity:prod" });
    for (const auto & p : pairs) {
        for (int64_t gqa : { 6, 4 }) {
            c.push_back({ KIND_TOLERANCE, p.first, p.second, 256, 8192, 128, 4*gqa, 4, false, 0.0f, "tolerance" });
        }
    }
    c.push_back({ KIND_TOLERANCE, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0, 256, 8192, 128, 24, 4, true, 0.0f, "tolerance+sinks" });
    c.push_back({ KIND_EXCLUDED, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 8192, 128, 24, 4, false, 8.0f, "excluded:alibi" });
    c.push_back({ KIND_EXCLUDED, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 128, 8192, 128, 16, 4, false, 0.0f, "excluded:D128" });
    c.push_back({ KIND_EXCLUDED, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 256, 8192, 128,  6, 1, false, 0.0f, "excluded:1kvhead" });
    return c;
}

// the plan's decision, re-derived here from the documented rule (not from the library)
static int expected_heads(const case_def & c, size_t budget_mib) {
    if (budget_mib == 0 || c.D != 256 || c.max_bias != 0.0f || c.n_head_kv < 2 || c.n_q <= 8) {
        return 0;
    }
    const size_t budget = budget_mib << 20;
    const size_t kv     = size_t(c.n_kv) * 256 * 2;
    const size_t out    = size_t(c.n_q) * size_t(c.n_head / c.n_head_kv) * 256 * 4;
    const size_t full   = 2 * kv * size_t(c.n_head_kv);
    const size_t floor1 = 2 * kv + out;
    if (full <= budget || floor1 >= full) {
        return 0;
    }
    return (int) std::min<size_t>(std::max<size_t>(budget / floor1, 1), size_t(c.n_head_kv));
}

// Summation order of one MMA f16 launch, re-derived from the source for D 256 on Ampere (sm_86), causal mask,
// max_bias 0, n_kv % 256 == 0, no sparse path (op_params[4] is 0 here):
//   ncols2 = 8 if gqa > 4, else 4 if gqa > 2 (use_gqa_opt)       fattn.cu ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2
//   ncols1 = 64 / ncols2 when n_q > 32 / ncols2                    fattn.cu ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1
//   ntiles = ceil(n_q / ncols1) * ceil(gqa / ncols2) * kv_heads    fattn-common.cuh launch_fattn (ntiles_dst)
//   stream-k when 100 * ntiles / (max_blocks * waves) < 75, with max_blocks = blocks/SM * n_SM (should_use_stream_k)
// Without stream-k every block owns one output tile and walks the whole KV range, so the per-element order does not
// depend on the grid. With stream-k a tile's KV range is split over blocks at grid-dependent points and merged by a
// fixup kernel, so a different tile count gives a different order. blocks/SM = 2 is the occupancy of the (8, 8) and
// (16, 4) D 256 kernels in this build (255 registers x 128 threads, cuobjdump of libggml-cuda.so, sm_86). Shapes
// outside this table return known = false and are treated as "order may differ".
struct grid_info {
    bool    known    = false;
    bool    stream_k = false;
    int64_t ntiles   = 0;
    int64_t nblocks  = 0;
};

static grid_info mma_grid(const case_def & c, int64_t kv_heads, int n_sm) {
    grid_info g;
    const int64_t gqa = c.n_head / c.n_head_kv;
    if (n_sm <= 0 || c.D != 256 || c.max_bias != 0.0f || c.n_kv % 256 != 0 || gqa <= 2) {
        return g;
    }
    const int64_t ncols2 = gqa > 4 ? 8 : 4;
    if (c.n_q <= 32 / ncols2) {
        return g;
    }
    const int64_t ncols1     = 64 / ncols2;
    const int64_t max_blocks = 2 * int64_t(n_sm);
    const int64_t nbatch_fa  = 32; // ggml_cuda_fattn_mma_get_config, (256, 256, 64) on Ampere
    g.ntiles = (c.n_q + ncols1 - 1) / ncols1 * ((gqa + ncols2 - 1) / ncols2) * kv_heads;
    const int64_t waves = (g.ntiles + max_blocks - 1) / max_blocks;
    g.stream_k = 100 * g.ntiles / (max_blocks * waves) < 75;
    g.nblocks  = g.ntiles;
    if (g.stream_k) {
        const int64_t raw     = std::min(max_blocks, (c.n_kv + nbatch_fa - 1) / nbatch_fa * g.ntiles);
        const int64_t rounded = raw / g.ntiles * g.ntiles;
        const int64_t loss    = rounded > 0 ? 100 * (raw - rounded) / raw : 100;
        g.nblocks = loss <= 5 ? rounded : raw;
    }
    g.known = true;
    return g;
}

// true when the unbounded launch and every group launch (groups of h KV heads, the last one possibly smaller) run
// without stream-k; `desc` gets the tile counts, "/sk<blocks>" marks a stream-k launch
static bool same_order(const case_def & c, int h, int n_sm, std::string & desc) {
    const grid_info u = mma_grid(c, c.n_head_kv, n_sm);
    if (!u.known) {
        desc = " order=unknown";
        return false;
    }
    auto launch = [](const grid_info & g) {
        return std::to_string(g.ntiles) + (g.stream_k ? "/sk" + std::to_string(g.nblocks) : std::string());
    };
    bool same = !u.stream_k;
    std::string tiles = " tiles[unbounded " + launch(u) + " groups";
    for (int64_t first = 0; first < c.n_head_kv; first += h) {
        const grid_info g = mma_grid(c, std::min<int64_t>(h, c.n_head_kv - first), n_sm);
        same = same && g.known && !g.stream_k;
        tiles += " " + launch(g);
    }
    desc = std::string(" order=") + (same ? "same" : "differs") + tiles + "]";
    return same;
}

// SM count for the order table; 0 (unknown GPU) makes every bounded case take the tolerance gates
static int known_sm_count(const char * description) {
    return description != nullptr && strstr(description, "RTX 3090 Ti") != nullptr ? 84 : 0;
}

struct case_data {
    std::vector<float>         q;  // [D, n_head, n_q]
    const std::vector<uint8_t> * k = nullptr; // [D, n_head_kv, n_kv] quantized
    const std::vector<uint8_t> * v = nullptr;
    const std::vector<ggml_fp16_t> * m = nullptr; // [n_kv, n_q]
    std::vector<float>         s;  // sinks [n_head]
    // f64 reference on the subset rows (tolerance cases): ref[(row_index*n_head + h)*D + d]
    std::vector<int64_t>       ref_rows;
    std::vector<double>        ref;
    double                     ref_max = 0.0;
};

static std::map<std::string, std::vector<uint8_t>>     g_kv_store;
static std::map<std::string, std::vector<ggml_fp16_t>> g_mask_store;

// rows of D values; a pool of distinct quantized rows is tiled over the tensor (the identity cases are large)
static const std::vector<uint8_t> & get_kv(ggml_type type, int64_t D, int64_t n_head_kv, int64_t n_kv, uint32_t seed) {
    char key[128];
    snprintf(key, sizeof(key), "%d/%lld/%lld/%lld/%u", (int) type, (long long) D, (long long) n_head_kv, (long long) n_kv, seed);
    auto it = g_kv_store.find(key);
    if (it != g_kv_store.end()) {
        return it->second;
    }
    const int64_t rows      = n_head_kv * n_kv;
    const int64_t pool_rows = std::min<int64_t>(rows, 32768);
    const size_t  row_bytes = ggml_row_size(type, D);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> f(size_t(pool_rows * D));
    for (float & x : f) {
        x = nd(rng);
    }
    std::vector<uint8_t> pool(size_t(pool_rows) * row_bytes);
    ggml_quantize_chunk(type, f.data(), pool.data(), 0, pool_rows, D, nullptr);
    std::vector<uint8_t> & out = g_kv_store[key];
    out.resize(size_t(rows) * row_bytes);
    uint64_t h = seed * 0x9e3779b97f4a7c15ull + 1;
    for (int64_t r = 0; r < rows; ++r) {
        int64_t src = r;
        if (rows > pool_rows) {
            h ^= h << 13; h ^= h >> 7; h ^= h << 17;
            src = int64_t(h % uint64_t(pool_rows));
        }
        memcpy(out.data() + size_t(r) * row_bytes, pool.data() + size_t(src) * row_bytes, row_bytes);
    }
    return out;
}

static const std::vector<ggml_fp16_t> & get_mask(int64_t n_kv, int64_t n_q) {
    const std::string key = std::to_string(n_kv) + "/" + std::to_string(n_q);
    auto it = g_mask_store.find(key);
    if (it != g_mask_store.end()) {
        return it->second;
    }
    std::vector<ggml_fp16_t> & m = g_mask_store[key];
    m.resize(size_t(n_kv) * size_t(n_q));
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t j = 0; j < n_q; ++j) {
        for (int64_t i = 0; i < n_kv; ++i) {
            m[size_t(j)*n_kv + i] = i <= n_kv - n_q + j ? zero : ninf;
        }
    }
    return m;
}

// f64 reference: dequantized K/V (to_float, the same rotated domain the CUDA converters produce), f32 Q, exact softmax.
// Rows: every `step`-th query row plus the last. One task per (row, KV head) covers that KV head's gqa query heads, so
// each K/V row is read once per task. Tasks run on up to 16 threads, all joined before any fork.
static void build_reference(const case_def & c, case_data & d, int64_t step) {
    const int64_t D = c.D;
    const ggml_type_traits * tk = ggml_get_type_traits(c.tk);
    const ggml_type_traits * tv = ggml_get_type_traits(c.tv);
    GGML_ASSERT(tk->to_float && tv->to_float);
    const size_t rbk = ggml_row_size(c.tk, D);
    const size_t rbv = ggml_row_size(c.tv, D);
    std::vector<float> kf(size_t(c.n_kv * c.n_head_kv * D));
    std::vector<float> vf(kf.size());
    for (int64_t r = 0; r < c.n_kv * c.n_head_kv; ++r) {
        tk->to_float(d.k->data() + size_t(r) * rbk, kf.data() + size_t(r) * D, D);
        tv->to_float(d.v->data() + size_t(r) * rbv, vf.data() + size_t(r) * D, D);
    }
    for (int64_t j = 0; j < c.n_q; j += step) {
        d.ref_rows.push_back(j);
    }
    if (d.ref_rows.back() != c.n_q - 1) {
        d.ref_rows.push_back(c.n_q - 1);
    }
    const double  scale = 1.0 / std::sqrt((double) D);
    const int64_t gqa   = c.n_head / c.n_head_kv;
    d.ref.assign(d.ref_rows.size() * size_t(c.n_head) * D, 0.0);

    const size_t n_tasks = d.ref_rows.size() * size_t(c.n_head_kv);
    std::atomic<size_t> next(0);
    auto worker = [&]() {
        std::vector<double> logit(static_cast<size_t>(gqa * c.n_kv));
        std::vector<double> acc(static_cast<size_t>(gqa * D));
        std::vector<double> mx(static_cast<size_t>(gqa));
        std::vector<double> sum(static_cast<size_t>(gqa));
        for (size_t t = next++; t < n_tasks; t = next++) {
            const size_t  ri = t / size_t(c.n_head_kv);
            const int64_t hk = int64_t(t % size_t(c.n_head_kv));
            const int64_t j  = d.ref_rows[ri];
            const float * q0 = d.q.data() + size_t((j*c.n_head + hk*gqa) * D); // gqa consecutive heads
            std::fill(mx.begin(), mx.end(), -INFINITY);
            for (int64_t i = 0; i < c.n_kv; ++i) {
                const float mval = ggml_fp16_to_fp32((*d.m)[size_t(j)*c.n_kv + i]);
                if (std::isinf(mval)) {
                    for (int64_t g = 0; g < gqa; ++g) {
                        logit[size_t(g*c.n_kv + i)] = -INFINITY;
                    }
                    continue;
                }
                const float * kr = kf.data() + size_t((i*c.n_head_kv + hk) * D);
                for (int64_t g = 0; g < gqa; ++g) {
                    const float * qv = q0 + g*D;
                    double dot = 0.0;
                    for (int64_t e = 0; e < D; ++e) {
                        dot += double(qv[e]) * double(kr[e]);
                    }
                    const double l = dot * scale + mval;
                    logit[size_t(g*c.n_kv + i)] = l;
                    mx[g] = std::max(mx[g], l);
                }
            }
            for (int64_t g = 0; g < gqa; ++g) {
                if (c.sinks) {
                    mx[g] = std::max(mx[g], (double) d.s[size_t(hk*gqa + g)]);
                }
                sum[g] = 0.0;
            }
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int64_t i = 0; i < c.n_kv; ++i) {
                const float * vr = vf.data() + size_t((i*c.n_head_kv + hk) * D);
                for (int64_t g = 0; g < gqa; ++g) {
                    const double l = logit[size_t(g*c.n_kv + i)];
                    if (std::isinf(l)) {
                        continue;
                    }
                    const double p = std::exp(l - mx[g]);
                    sum[g] += p;
                    double * a = acc.data() + g*D;
                    for (int64_t e = 0; e < D; ++e) {
                        a[e] += p * double(vr[e]);
                    }
                }
            }
            for (int64_t g = 0; g < gqa; ++g) {
                const int64_t h = hk*gqa + g;
                if (c.sinks) {
                    sum[g] += std::exp(double(d.s[size_t(h)]) - mx[g]);
                }
                double * o = d.ref.data() + (ri*c.n_head + h) * D;
                for (int64_t e = 0; e < D; ++e) {
                    o[e] = acc[size_t(g*D + e)] / sum[g];
                }
            }
        }
    };
    const size_t n_threads = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), 16);
    std::vector<std::thread> pool;
    for (size_t i = 1; i < std::min(n_threads, n_tasks); ++i) {
        pool.emplace_back(worker);
    }
    worker();
    for (std::thread & th : pool) {
        th.join();
    }
    for (const double o : d.ref) {
        d.ref_max = std::max(d.ref_max, std::fabs(o));
    }
}

static void build_inputs(const case_def & c, case_data & d, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    d.q.resize(size_t(c.D * c.n_head * c.n_q));
    for (float & x : d.q) {
        x = nd(rng);
    }
    d.k = &get_kv(c.tk, c.D, c.n_head_kv, c.n_kv, 1000 + (uint32_t) c.tk * 7 + (uint32_t) c.D);
    d.v = &get_kv(c.tv, c.D, c.n_head_kv, c.n_kv, 2000 + (uint32_t) c.tv * 7 + (uint32_t) c.D);
    d.m = &get_mask(c.n_kv, c.n_q);
    if (c.sinks) {
        d.s.resize(size_t(c.n_head));
        for (float & x : d.s) {
            x = nd(rng);
        }
    }
    if (c.kind != KIND_EXCLUDED) {
        build_reference(c, d, c.kind == KIND_TOLERANCE ? 4 : 32);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// child side

struct ledger_counts {
    int64_t bounded_reserved = 0;
    int64_t bounded_pool     = 0;
    int64_t full_reserved    = 0;
    int64_t full_pool        = 0;
    int     heads            = -1; // from the bounded row's "heads=h/nh": h
    int     heads_nkv        = -1; // nh
    int     heads_rows       = 0;  // live bounded keys seen
    int64_t heads_count      = 0;  // bounded launches counted on those keys
    std::map<std::string, int64_t> graph; // cuda.graph outcome -> count
};

static void ledger_cb(const char * site, const char * key, int64_t count, void * user_data) {
    ledger_counts * c = (ledger_counts *) user_data;
    if (count == 0) {
        return; // a slot from an earlier case: ggml_ledger_reset zeroes counts but keeps the slots
    }
    if (strcmp(site, "cuda.graph") == 0) {
        const char * sp = strchr(key, ' ');
        c->graph[sp ? std::string(key, sp - key) : std::string(key)] += count;
        return;
    }
    if (strcmp(site, "cuda.fattn") != 0 || strncmp(key, "f16_convert ", 12) != 0) {
        return;
    }
    if (strstr(key, "scratch=bounded-reserved") != nullptr) {
        c->bounded_reserved += count;
    } else if (strstr(key, "scratch=bounded-pool") != nullptr) {
        c->bounded_pool += count;
    } else if (strstr(key, "scratch=reserved") != nullptr) {
        c->full_reserved += count;
    } else if (strstr(key, "scratch=pool") != nullptr) {
        c->full_pool += count;
    }
    const char * h = strstr(key, "scratch=bounded-") != nullptr ? strstr(key, "heads=") : nullptr;
    if (h != nullptr) {
        if (sscanf(h + 6, "%d/%d", &c->heads, &c->heads_nkv) != 2) {
            c->heads = c->heads_nkv = -1;
        }
        c->heads_rows++;
        c->heads_count += count;
    }
}

struct built_graph {
    ggml_context *        ctx_in  = nullptr;
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf_in  = nullptr;
    ggml_backend_buffer_t buf_out = nullptr;
    ggml_tensor *         out     = nullptr;
    ggml_cgraph *         gf      = nullptr;

    void release() {
        if (buf_out) { ggml_backend_buffer_free(buf_out); }
        if (ctx)     { ggml_free(ctx); }
        if (buf_in)  { ggml_backend_buffer_free(buf_in); }
        if (ctx_in)  { ggml_free(ctx_in); }
        *this = built_graph();
    }
};

static built_graph build_graph(ggml_backend_t backend, const case_def & c, const case_data & d, bool dst_view) {
    built_graph g;
    ggml_init_params pi = { ggml_tensor_overhead() * 16, nullptr, true };
    g.ctx_in = ggml_init(pi);
    ggml_tensor * qb      = ggml_new_tensor_3d(g.ctx_in, GGML_TYPE_F32, c.D, c.n_head, c.n_q);
    ggml_tensor * kb      = ggml_new_tensor_3d(g.ctx_in, c.tk, c.D, c.n_head_kv, c.n_kv);
    ggml_tensor * vb      = ggml_new_tensor_3d(g.ctx_in, c.tv, c.D, c.n_head_kv, c.n_kv);
    ggml_tensor * m       = ggml_new_tensor_2d(g.ctx_in, GGML_TYPE_F16, c.n_kv, c.n_q);
    ggml_tensor * s       = c.sinks ? ggml_new_tensor_1d(g.ctx_in, GGML_TYPE_F32, c.n_head) : nullptr;
    ggml_tensor * backing = ggml_new_tensor_3d(g.ctx_in, GGML_TYPE_F32, c.D, c.n_head, c.n_q);
    ggml_tensor * q       = ggml_permute(g.ctx_in, qb, 0, 2, 1, 3); // [D, n_q, n_head]
    ggml_tensor * k       = ggml_permute(g.ctx_in, kb, 0, 2, 1, 3); // [D, n_kv, n_head_kv], strided head slices
    ggml_tensor * v       = ggml_permute(g.ctx_in, vb, 0, 2, 1, 3);
    g.buf_in = ggml_backend_alloc_ctx_tensors(g.ctx_in, backend);
    GGML_ASSERT(g.buf_in);

    ggml_backend_tensor_set(qb, d.q.data(), 0, ggml_nbytes(qb));
    GGML_ASSERT(d.k->size() == ggml_nbytes(kb) && d.v->size() == ggml_nbytes(vb));
    ggml_backend_tensor_set(kb, d.k->data(), 0, ggml_nbytes(kb));
    ggml_backend_tensor_set(vb, d.v->data(), 0, ggml_nbytes(vb));
    ggml_backend_tensor_set(m, d.m->data(), 0, ggml_nbytes(m));
    if (s) {
        ggml_backend_tensor_set(s, d.s.data(), 0, ggml_nbytes(s));
    }
    std::vector<float> zero(size_t(ggml_nelements(backing)), 0.0f);
    ggml_backend_tensor_set(backing, zero.data(), 0, ggml_nbytes(backing));

    ggml_init_params pg = { ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true };
    g.ctx = ggml_init(pg);
    g.out = ggml_flash_attn_ext(g.ctx, q, k, v, m, 1.0f/sqrtf((float) c.D), c.max_bias, 0.0f);
    if (s) {
        ggml_flash_attn_ext_add_sinks(g.out, s);
    }
    ggml_prec_set_acc(g.out, GGML_PREC_F32);
    GGML_ASSERT(ggml_nbytes(g.out) == ggml_nbytes(backing));
    if (dst_view) {
        g.out->view_src  = backing;
        g.out->view_offs = 0;
        GGML_ASSERT(ggml_backend_view_init(g.out) == GGML_STATUS_SUCCESS);
    } else {
        g.buf_out = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
        GGML_ASSERT(g.buf_out);
    }
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return g;
}

static void run_case(ggml_backend_t backend, const case_def & c, const case_data & d, bool dst_view,
                     std::vector<float> & out, ledger_counts & l) {
    ggml_ledger_reset();
    built_graph g = build_graph(backend, c, d, dst_view);
    GGML_ASSERT(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS);
    out.resize(size_t(ggml_nelements(g.out)));
    ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
    g.release();
    l = ledger_counts();
    ggml_ledger_foreach(ledger_cb, &l);
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double mx = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double dd = std::fabs(double(a[i]) - double(b[i]));
        if (!(dd <= mx)) {
            mx = std::isnan(dd) ? INFINITY : dd;
        }
    }
    return mx;
}

static double max_abs_vs_ref(const case_def & c, const case_data & d, const std::vector<float> & o) {
    double mx = 0.0;
    for (size_t ri = 0; ri < d.ref_rows.size(); ++ri) {
        const int64_t j = d.ref_rows[ri];
        for (int64_t h = 0; h < c.n_head; ++h) {
            for (int64_t e = 0; e < c.D; ++e) {
                const double dd = std::fabs(double(o[size_t((j*c.n_head + h)*c.D + e)]) - d.ref[(ri*c.n_head + h)*c.D + e]);
                if (!(dd <= mx)) {
                    mx = std::isnan(dd) ? INFINITY : dd;
                }
            }
        }
    }
    return mx;
}

// max |a - b| on the reference rows only, the same rows the errors vs the reference are taken on
static double max_abs_diff_ref_rows(const case_def & c, const case_data & d, const std::vector<float> & a,
                                    const std::vector<float> & b) {
    double mx = 0.0;
    for (const int64_t j : d.ref_rows) {
        const size_t o = size_t(j * c.n_head * c.D);
        for (size_t i = o; i < o + size_t(c.n_head * c.D); ++i) {
            const double dd = std::fabs(double(a[i]) - double(b[i]));
            if (!(dd <= mx)) {
                mx = std::isnan(dd) ? INFINITY : dd;
            }
        }
    }
    return mx;
}

struct arm_def {
    const char * name;
    const char * env;    // nullptr = unset
    size_t       mib;
    bool         graphs; // graph arm: CUDA graphs on, only the graph check runs
};

static const arm_def g_arms[] = {
    { "unset", nullptr, 0,   false },
    { "1",     "1",     1,   false },
    { "20",    "20",    20,  false },
    { "27",    "27",    27,  false },
    { "256",   "256",   256, false },
    { "272",   "272",   272, false },
    { "410",   "410",   410, false },
    { "graph-unset", nullptr, 0, true },
    { "graph-1",     "1",     1, true },
};

static std::vector<case_def>          g_cases;
static std::vector<case_data>         g_data;
static std::vector<std::vector<float>> g_ref_out; // arm "unset" outputs, per case

static size_t first_tolerance_case() {
    size_t ci = 0;
    while (g_cases[ci].kind != KIND_TOLERANCE) {
        ci++;
    }
    return ci;
}

static void plan_lines(ggml_backend_dev_t dev, const arm_def & arm, int & n_fail) {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    for (int64_t n_kv : { 100352, 131072, 245760, 262144 }) {
        ggml_init_params pi = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(pi);
        ggml_tensor * qb  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 24, 1024);
        ggml_tensor * kb  = ggml_new_tensor_3d(ctx, GGML_TYPE_TQ5_0, 256, 4, n_kv);
        ggml_tensor * vb  = ggml_new_tensor_3d(ctx, GGML_TYPE_TURBO4_0, 256, 4, n_kv);
        ggml_tensor * m   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1024);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, ggml_permute(ctx, qb, 0, 2, 1, 3), ggml_permute(ctx, kb, 0, 2, 1, 3),
                                                ggml_permute(ctx, vb, 0, 2, 1, 3), m, 1.0f/16.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        const size_t size = ggml_backend_buft_get_alloc_size(buft, out);
        const size_t nb   = ggml_nbytes(out);
        const case_def c  = { KIND_TOLERANCE, GGML_TYPE_TQ5_0, GGML_TYPE_TURBO4_0, 256, n_kv, 1024, 24, 4, false, 0.0f, "plan" };
        const int h = expected_heads(c, arm.mib);
        std::string verdict = "";
        if (h > 0) {
            const size_t kv   = size_t(n_kv) * 256 * 2;
            const size_t outb = size_t(1024) * 6 * 256 * 4;
            const size_t want = ((nb + 127) / 128) * 128 + std::max(size_t(h) * (2*kv + outb), arm.mib << 20);
            verdict = size == want ? " OK" : " FAIL:size!=plan";
            n_fail += size != want;
        }
        printf("PLAN arm=%s n_kv=%lld n_q=1024 K=tq5_0 V=turbo4 heads/group=%d reserve_behind_dst=%.1f MiB full_f16_copies=%.1f MiB%s\n",
               arm.name, (long long) n_kv, h, (size - nb) / 1048576.0, 2.0 * n_kv * 256 * 2 * 4 / 1048576.0, verdict.c_str());
        ggml_free(ctx);
    }
}

static int child_main(const arm_def & arm, int out_fd) {
    if (arm.env) {
        setenv("GGML_CUDA_PREFILL_KV_MIB", arm.env, 1);
    } else {
        unsetenv("GGML_CUDA_PREFILL_KV_MIB");
    }
    if (arm.graphs) {
        unsetenv("GGML_CUDA_DISABLE_GRAPHS");
    } else {
        setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1); // every compute launches on the host so the ledger counts it
    }
    ggml_ledger_set_enabled(true);

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (dev == nullptr) {
        return 77;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    int n_fail = 0;

    if (arm.graphs) {
        // the first tolerance case (q8_0/q8_0, GQA 6, 8,192 / 128), same graph computed 4 times
        const size_t ci = first_tolerance_case();
        const case_def & c = g_cases[ci];
        const bool bounded = expected_heads(c, arm.mib) > 0;
        ggml_ledger_reset();
        built_graph g = build_graph(backend, c, g_data[ci], false);
        for (int it = 0; it < 4; ++it) {
            GGML_ASSERT(ggml_backend_graph_compute(backend, g.gf) == GGML_STATUS_SUCCESS);
        }
        std::vector<float> out(size_t(ggml_nelements(g.out)));
        ggml_backend_tensor_get(g.out, out.data(), 0, ggml_nbytes(g.out));
        g.release();
        ledger_counts l;
        ggml_ledger_foreach(ledger_cb, &l);
        int64_t captures = 0;
        int64_t incompat = 0;
        std::string outcomes;
        for (const auto & kv : l.graph) {
            outcomes += " " + kv.first + "=" + std::to_string(kv.second);
            captures += kv.first.rfind("capture:", 0) == 0 ? kv.second : 0;
            incompat += kv.first == "eager:incompatible_fattn_bounded_prefill" ? kv.second : 0;
        }
        std::string why;
        if (l.graph.empty()) {
            why = " no_cuda_graph_rows(graphs_not_compiled?)";
        } else if (bounded) {
            if (captures != 0 || incompat == 0) {
                why = " bounded_graph_captured_or_not_marked";
            }
        } else if (captures == 0) {
            why = " control_never_captured";
        }
        const bool same = out.size() == g_ref_out[ci].size() && memcmp(out.data(), g_ref_out[ci].data(), out.size()*sizeof(float)) == 0;
        printf("GRAPH arm=%-11s K=%s V=%s n_kv=%lld n_q=%lld outcomes:%s  bounded_rows=%lld  vs_unset_eager=%s  %s%s\n",
               arm.name, ggml_type_name(c.tk), ggml_type_name(c.tv), (long long) c.n_kv, (long long) c.n_q, outcomes.c_str(),
               (long long) (l.bounded_reserved + l.bounded_pool), bounded ? (same ? "identical" : "differs(reported)") : "n/a",
               why.empty() ? "OK" : "FAIL:", why.c_str());
        n_fail += !why.empty();
        ggml_backend_free(backend);
        fflush(stdout);
        return n_fail == 0 ? 0 : 1;
    }

    plan_lines(dev, arm, n_fail);

    const char * dev_desc = ggml_backend_dev_description(dev);
    const int    n_sm     = known_sm_count(dev_desc);
    printf("DEVICE arm=%s \"%s\" n_sm=%d%s\n", arm.name, dev_desc ? dev_desc : "?", n_sm,
           n_sm > 0 ? "" : " (not in the order table: every bounded case takes the tolerance gates)");

    for (size_t ci = 0; ci < g_cases.size(); ++ci) {
        const case_def & c = g_cases[ci];
        const case_data & d = g_data[ci];
        const int h = expected_heads(c, arm.mib);
        std::vector<float> out;
        ledger_counts l;
        run_case(backend, c, d, false, out, l);

        std::string why;
        std::string info;
        char buf[256];
        if (arm.env == nullptr) {
            // control arm: send outputs to the parent; must never take the bounded route
            const size_t bytes = out.size() * sizeof(float);
            size_t off = 0;
            while (off < bytes) {
                const ssize_t w = write(out_fd, (const char *) out.data() + off, bytes - off);
                if (w <= 0) {
                    perror("write");
                    return 2;
                }
                off += size_t(w);
            }
            if (l.bounded_reserved + l.bounded_pool != 0) {
                why += " unset_took_bounded";
            }
            if (c.kind != KIND_EXCLUDED && l.full_reserved == 0) {
                why += " no_full_f16_copy(route_changed?)";
            }
            if (c.kind == KIND_IDENTITY) {
                std::vector<float> again;
                ledger_counts l2;
                run_case(backend, c, d, false, again, l2);
                const bool rep = memcmp(again.data(), out.data(), out.size()*sizeof(float)) == 0;
                info += rep ? " unbounded_repeatable" : " unbounded_NOT_repeatable";
            }
            if (c.kind != KIND_EXCLUDED) {
                snprintf(buf, sizeof(buf), " max|unbounded-ref|=%.3e R=%.3e", max_abs_vs_ref(c, d, out), d.ref_max);
                info += buf;
            }
        } else {
            const std::vector<float> & ref = g_ref_out[ci];
            GGML_ASSERT(ref.size() == out.size());
            const bool same = memcmp(ref.data(), out.data(), out.size()*sizeof(float)) == 0;
            const double dmax = same ? 0.0 : max_abs_diff(out, ref);
            if (h == 0) {
                if (l.bounded_reserved + l.bounded_pool != 0) {
                    why += " bounded_row_on_excluded_or_fitting_shape";
                }
                if (!same) {
                    snprintf(buf, sizeof(buf), " differs_from_unset(max %.3e)", dmax);
                    why += buf;
                }
            } else {
                if (l.bounded_reserved == 0) {
                    why += " no_bounded_reserved_row";
                }
                if (l.bounded_pool != 0 || l.full_reserved != 0 || l.full_pool != 0) {
                    why += " unexpected_scratch_rows";
                }
                // one live bounded row, counted once, naming this plan's heads per group and the case's KV heads
                if (l.heads_rows != 1 || l.heads_count != 1 || l.heads != h || l.heads_nkv != c.n_head_kv) {
                    snprintf(buf, sizeof(buf), " heads=%d/%d(rows %d, launches %lld)!=expected_%d/%lld", l.heads, l.heads_nkv,
                             l.heads_rows, (long long) l.heads_count, h, (long long) c.n_head_kv);
                    why += buf;
                }
                std::string order;
                const bool order_same = same_order(c, h, n_sm, order);
                const double e_b   = max_abs_vs_ref(c, d, out);
                const double e_u   = max_abs_vs_ref(c, d, ref);
                const double drows = same ? 0.0 : max_abs_diff_ref_rows(c, d, out, ref);
                snprintf(buf, sizeof(buf), " %s max|bounded-unbounded|=%.3e (ref rows %.3e) max|bounded-ref|=%.3e "
                         "max|unbounded-ref|=%.3e R=%.3e", same ? "identical" : "differs", dmax, drows, e_b, e_u, d.ref_max);
                info += order + buf;
                if (order_same) {
                    if (!same) {
                        why += " bytes_differ(same_order)";
                    }
                } else {
                    if (!(e_b <= e_u * 1.10)) {
                        why += " gate_a(bounded_ref>1.10*unbounded_ref)";
                    }
                    if (!(drows <= e_u + 5e-3 * d.ref_max)) {
                        why += " gate_b(bounded_unbounded>unbounded_ref+5e-3*R)";
                    }
                }
                // dst as a view: the executor must fall back to the pool and give the same bytes
                if (ci == first_tolerance_case()) {
                    std::vector<float> outv;
                    ledger_counts lv;
                    run_case(backend, c, d, true, outv, lv);
                    const bool vsame = memcmp(outv.data(), out.data(), out.size()*sizeof(float)) == 0;
                    snprintf(buf, sizeof(buf), " dst_view:%s pool_rows=%lld reserved_rows=%lld",
                             vsame ? "identical" : "differs", (long long) lv.bounded_pool, (long long) lv.bounded_reserved);
                    info += buf;
                    if (!vsame || lv.bounded_pool == 0 || lv.bounded_reserved != 0) {
                        why += " dst_view";
                    }
                }
            }
        }
        printf("CASE arm=%-5s %-16s K=%-8s V=%-8s D=%lld n_kv=%lld n_q=%lld n_head=%lld n_head_kv=%lld expect_heads=%d "
               "rows[bres=%lld bpool=%lld full_res=%lld full_pool=%lld]%s  %s%s\n",
               arm.name, c.what, ggml_type_name(c.tk), ggml_type_name(c.tv), (long long) c.D, (long long) c.n_kv,
               (long long) c.n_q, (long long) c.n_head, (long long) c.n_head_kv, h, (long long) l.bounded_reserved,
               (long long) l.bounded_pool, (long long) l.full_reserved, (long long) l.full_pool, info.c_str(),
               why.empty() ? "OK" : "FAIL:", why.c_str());
        fflush(stdout);
        n_fail += !why.empty();
    }

    ggml_backend_free(backend);
    fflush(stdout);
    return n_fail == 0 ? 0 : 1;
}

int main() {
#ifdef _WIN32
    printf("test-fattn-bounded-prefill: needs fork(), skipping on Windows\n");
    return 0;
#else
    // parent: inputs and references only, no backend
    g_cases = make_cases();
    g_data.resize(g_cases.size());
    for (size_t ci = 0; ci < g_cases.size(); ++ci) {
        build_inputs(g_cases[ci], g_data[ci], 17 + 31 * (uint32_t) ci);
    }
    printf("test-fattn-bounded-prefill: %zu cases, inputs and f64 references built\n", g_cases.size());

    int n_fail = 0;
    int n_arms = 0;
    for (const arm_def & arm : g_arms) {
        int fds[2] = { -1, -1 };
        const bool control = arm.env == nullptr && !arm.graphs;
        if (control && pipe(fds) != 0) {
            perror("pipe");
            return 1;
        }
        fflush(stdout);
        fflush(stderr);
        const pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid == 0) {
            if (control) {
                close(fds[0]);
            }
            const int rc = child_main(arm, control ? fds[1] : -1);
            fflush(stdout);
            fflush(stderr);
            _exit(rc);
        }
        if (control) {
            close(fds[1]);
            g_ref_out.resize(g_cases.size());
            bool ok = true;
            for (size_t ci = 0; ci < g_cases.size() && ok; ++ci) {
                const case_def & c = g_cases[ci];
                g_ref_out[ci].resize(size_t(c.D * c.n_head * c.n_q));
                const size_t bytes = g_ref_out[ci].size() * sizeof(float);
                size_t off = 0;
                while (off < bytes) {
                    const ssize_t r = read(fds[0], (char *) g_ref_out[ci].data() + off, bytes - off);
                    if (r <= 0) {
                        ok = false;
                        break;
                    }
                    off += size_t(r);
                }
            }
            close(fds[0]);
            if (!ok) {
                printf("ARM unset: short read of the control outputs\n");
            }
        }
        int status = 0;
        waitpid(pid, &status, 0);
        const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
        if (rc == 77) {
            printf("test-fattn-bounded-prefill: no GPU backend, skipping\n");
            return 0;
        }
        printf("ARM %s rc=%d\n", arm.name, rc);
        fflush(stdout);
        n_arms++;
        n_fail += rc != 0;
        if (control && rc != 0) {
            printf("test-fattn-bounded-prefill: control arm failed, stopping\n");
            return 1;
        }
    }
    printf("test-fattn-bounded-prefill: %d/%d arms passed\n", n_arms - n_fail, n_arms);
    return n_fail == 0 ? 0 : 1;
#endif
}
