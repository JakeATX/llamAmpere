#include "llama-lsa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"

#include "gguf.h"

#include <array>
extern "C" {
#include "../vendor/hash/sha256/sha256.h"
}

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static std::string llama_lsa_describe_cfg(const llama_lsa_config & c);

ggml_tensor * llama_lsa_build_recall_pick(ggml_context * ctx, ggml_tensor * scores, ggml_tensor * mask,
        ggml_tensor * cell_pos, ggml_tensor * q_pos, int block, int refine, float tau,
        int window, int sinks, int coarse_keep, int fine_keep, int budget_pct) {
    GGML_ASSERT(block == 128 && (refine == 16 || refine == 128));
    GGML_ASSERT((budget_pct == 100 || budget_pct == 110 || budget_pct == 125 || budget_pct == 150) && coarse_keep >= 1 && fine_keep >= 1);
    coarse_keep = (int) std::min<int64_t>(scores->ne[0]/block, ((int64_t) coarse_keep*budget_pct + 99)/100);
    fine_keep = (int) std::min<int64_t>(scores->ne[0]/refine, ((int64_t) fine_keep*budget_pct + 99)/100);
    ggml_tensor * gate = ggml_lsa_block_select_cells(ctx, scores, mask, cell_pos, q_pos,
            block, tau, window, sinks, coarse_keep, 0);
    return ggml_lsa_block_select_cells_gated_ext(ctx, scores, mask, cell_pos, q_pos,
            refine, tau, window, sinks, fine_keep, gate, block/refine, 0);
}

static int64_t lsa_env_i64(const char * name, int64_t def) {
    const char * v = getenv(name);
    return v && *v ? strtoll(v, nullptr, 10) : def;
}

static double lsa_env_f64(const char * name, double def) {
    const char * v = getenv(name);
    return v && *v ? strtod(v, nullptr) : def;
}

static std::string lsa_env_str(const char * name) {
    const char * v = getenv(name);
    return v ? std::string(v) : std::string();
}

const llama_lsa_config & llama_lsa_cfg() {
    static const llama_lsa_config cfg = [] {
        llama_lsa_config c;
        const std::string mode = lsa_env_str("LLAMA_LSA_MODE");
        if (mode == "key" || mode == "lsa" || mode == "1") {
            c.mode = llama_lsa_config::KEY;
        } else if (mode == "pos" || mode == "lsa-pos") {
            c.mode = llama_lsa_config::POS;
        } else if (mode == "oracle" || mode == "lsa-oracle") {
            c.mode = llama_lsa_config::ORACLE;
        } else if (mode == "mass" || mode == "lsa-mass") {
            c.mode = llama_lsa_config::MASS; // diagnostic: rank cells by summed attention probability
        } else if (mode == "quest" || mode == "lsa-quest") {
            c.mode = llama_lsa_config::QUEST; // real-K block bound, no learned map
        } else if (!mode.empty() && mode != "0" && mode != "off" && mode != "none") {
            LLAMA_LOG_WARN("%s: unknown LLAMA_LSA_MODE=%s, LSA stays off\n", __func__, mode.c_str());
        }
        c.map_path  = lsa_env_str("LLAMA_LSA_MAP");
        c.keep      = lsa_env_i64("LLAMA_LSA_KEEP",      c.keep);
        c.keep_frac = lsa_env_f64("LLAMA_LSA_KEEP_FRAC", c.keep_frac);
        c.sinks     = lsa_env_i64("LLAMA_LSA_SINKS",     c.sinks);
        c.window    = lsa_env_i64("LLAMA_LSA_WINDOW",    c.window);
        c.bucket    = lsa_env_i64("LLAMA_LSA_BUCKET",    c.bucket);
        c.max_rows  = lsa_env_i64("LLAMA_LSA_MAX_ROWS",  c.max_rows);
        c.min_pos   = lsa_env_i64("LLAMA_LSA_MIN_POS",   c.min_pos);
        c.fa_sparse = lsa_env_i64("LLAMA_LSA_FA_SPARSE", 0) != 0;
        c.gather    = lsa_env_i64("LLAMA_LSA_GATHER",    0) != 0;
        c.shared    = lsa_env_i64("LLAMA_LSA_SHARED",    0) != 0;
        c.shared_max = lsa_env_i64("LLAMA_LSA_SHARED",   0) == 2;
        c.shared_union = lsa_env_i64("LLAMA_LSA_SHARED", 0) == 3;
        c.shared    = c.shared && !c.shared_union; // SHARED=3 picks per row (then OR), not the summed shared pick
        c.smax_chain = lsa_env_i64("LLAMA_LSA_SMAX_CHAIN", 0) != 0;
        c.elig_drop = lsa_env_i64("LLAMA_LSA_ELIG",      0) != 0;
        c.pick_every = std::max<int64_t>(1, lsa_env_i64("LLAMA_LSA_PICK_EVERY", 1));
        c.pick_check = lsa_env_i64("LLAMA_LSA_PICK_CHECK", 0) != 0;
        c.blist_lean = lsa_env_i64("LLAMA_LSA_BLIST_LEAN", 0) != 0;
        c.packed    = lsa_env_i64("LLAMA_LSA_PACKED",    0) != 0;
        c.blk_list  = lsa_env_i64("LLAMA_LSA_BLKLIST",   0) != 0;
        c.vk        = lsa_env_i64("LLAMA_LSA_VK",        0) != 0;
        c.packed    = c.packed || c.blk_list; // the block list keeps the packed I32 KQ mask for the rest of the context
        c.fast      = lsa_env_i64("LLAMA_LSA_FAST",      0) != 0;
        c.idx_cache = lsa_env_i64("LLAMA_LSA_IDX_CACHE", 0) != 0;
        c.idx_q8    = lsa_env_i64("LLAMA_LSA_IDX_Q8",    0) != 0 && c.idx_cache;
        c.sj_compact = lsa_env_i64("LLAMA_LSA_SJ_COMPACT", 0) != 0;
        if (c.sj_compact) {
            setenv("GGML_SJKVARN_COMPACT_MAX_BOUND", "262144", 0); // the kept set is 10-20K cells; ggml's default cap is 4096
            setenv("GGML_SJKVARN_COMPACT_HEADS4", "1", 0); // ggml takes 4 KV heads only when asked
        }
        c.per_head  = lsa_env_i64("LLAMA_LSA_PER_HEAD",  0) != 0;
        c.topp      = lsa_env_f64("LLAMA_LSA_TOPP",      c.topp);
        c.topp_tau  = lsa_env_f64("LLAMA_LSA_TOPP_TAU",  c.topp_tau);
        c.topp_min  = lsa_env_f64("LLAMA_LSA_TOPP_MIN",  c.topp_min);
        c.stats     = lsa_env_i64("LLAMA_LSA_STATS",     0) != 0;
        c.spec_recall = lsa_env_i64("LLAMA_LSA_SPEC_RECALL", 0) != 0;
        c.j3        = lsa_env_i64("LLAMA_LSA_J3",       0) != 0;
        c.j3_a3     = (int) lsa_env_i64("LLAMA_LSA_J3_A3",    150);
        c.j3_a1     = (int) lsa_env_i64("LLAMA_LSA_J3_A1",    150);
        c.j3_prev   = lsa_env_i64("LLAMA_LSA_J3_PREV",  1) != 0;
        c.j3_first  = (int) std::max<int64_t>(0, lsa_env_i64("LLAMA_LSA_J3_FIRST", 3));
        c.recall    = lsa_env_i64("LLAMA_LSA_RECALL",    0) != 0;
        c.block     = lsa_env_i64("LLAMA_LSA_BLOCK",     0);
        c.block_tau = lsa_env_f64("LLAMA_LSA_BLOCK_TAU", c.block_tau);
        c.block_sum = lsa_env_i64("LLAMA_LSA_BLOCK_SUM", 0) != 0 && c.block > 1 && c.idx_cache && c.mode == llama_lsa_config::KEY;
        c.block_sum_mode = lsa_env_i64("LLAMA_LSA_BLOCK_SUM", 0) == 2 ? 2 : 1;
        c.sum_op    = c.block_sum && lsa_env_i64("LLAMA_LSA_SUM_OP", 0) != 0;
        if (c.block_sum && c.idx_q8) {
            LLAMA_LOG_WARN("%s: LLAMA_LSA_IDX_Q8 is not supported with LLAMA_LSA_BLOCK_SUM, index cache stays f16\n", __func__);
            c.idx_q8 = false;
        }
        const int64_t comp_lvl = lsa_env_i64("LLAMA_LSA_COMP", 0);
        c.comp       = comp_lvl != 0;
        c.comp_fast  = comp_lvl == 2 && c.idx_cache;
        if (comp_lvl == 2 && !c.idx_cache) {
            LLAMA_LOG_WARN("%s: LLAMA_LSA_COMP=2 needs LLAMA_LSA_IDX_CACHE=1, using the dense-score compensation probe\n", __func__);
        }
        c.comp_bias  = lsa_env_f64("LLAMA_LSA_COMP_BIAS", c.comp_bias);
        c.comp_max_t = lsa_env_i64("LLAMA_LSA_COMP_MAX_T", c.comp_max_t);
        c.comp_lb    = lsa_env_i64("LLAMA_LSA_COMP_LB", 0) != 0;
        c.blk_mask_legacy = lsa_env_i64("LLAMA_LSA_BLK_MASK_LEGACY", 0) != 0;
        c.comp_cnt_legacy = lsa_env_i64("LLAMA_LSA_COMP_CNT_LEGACY", 0) != 0;
        c.comp_cnt_exp    = lsa_env_i64("LLAMA_LSA_COMP_CNT_EXP", 0) != 0;
        c.rec_blk         = lsa_env_i64("LLAMA_LSA_REC_BLK", 0) != 0;
        c.blk_lse         = lsa_env_i64("LLAMA_LSA_BLK_LSE", 0) != 0;
        c.fused_sel       = lsa_env_i64("LLAMA_LSA_FUSED_SEL", 0) != 0;
        c.graph_reuse     = lsa_env_i64("LLAMA_LSA_GRAPH_REUSE", 0) != 0;
        c.unit_op         = lsa_env_i64("LLAMA_LSA_UNIT_OP", 0) != 0;
        c.slice           = std::max<int64_t>(0, lsa_env_i64("LLAMA_LSA_SLICE", 0));
        c.fused_comp      = lsa_env_i64("LLAMA_LSA_FUSED_COMP", 0) != 0;
        c.ring_legacy     = lsa_env_i64("LLAMA_LSA_RING_LEGACY", 0) != 0;
        {
            const int64_t Bc = c.block > 1 ? c.block : 128;
            const int64_t rr = std::max<int64_t>(lsa_env_i64("LLAMA_LSA_RING_ROWS", 4*Bc), 2*Bc);
            c.ring_rows = (rr + Bc - 1)/Bc*Bc;
        }
        const int64_t sj_skip_lvl = lsa_env_i64("LLAMA_LSA_SJ_SKIP", 0);
        c.sj_skip   = sj_skip_lvl != 0;
        if (sj_skip_lvl >= 2 && c.mode != llama_lsa_config::OFF) {
            setenv("GGML_SJKVARN_LIVE_STRIPS", "1", 0); // 2 = also skip the dead 16-cell strips of a live record
        }
        if (c.sj_skip && c.mode != llama_lsa_config::OFF) {
            setenv("GGML_SJKVARN_LIVE_RECORDS", "1", 0);
        }
        c.refine      = lsa_env_i64("LLAMA_LSA_REFINE", 0);
        c.refine_mult = std::max(1.0, lsa_env_f64("LLAMA_LSA_REFINE_MULT", c.refine_mult));
        c.refine_sh   = lsa_env_i64("LLAMA_LSA_REFINE_SH", 0) != 0;
        c.blist_min_unit = lsa_env_i64("LLAMA_LSA_BLKLIST_FINE", 0) != 0 ? 1 : 16;
        if (c.refine > 0 && (c.block <= 1 || c.block % c.refine != 0)) {
            LLAMA_LOG_WARN("%s: LLAMA_LSA_REFINE=%lld needs LLAMA_LSA_BLOCK > 1 and a divisor of it, refine stays off\n",
                    __func__, (long long) c.refine);
            c.refine = 0;
        }
        c.dump        = lsa_env_str("LLAMA_LSA_DUMP");
        c.dump_min_kv = lsa_env_i64("LLAMA_LSA_DUMP_MIN_KV", c.dump_min_kv);
        c.score_log        = lsa_env_str("LLAMA_LSA_SCORE_LOG");
        c.score_log_min_kv = lsa_env_i64("LLAMA_LSA_SCORE_LOG_MIN_KV", c.score_log_min_kv);
        c.resident         = std::min(100.0, std::max(0.0, lsa_env_f64("LLAMA_LSA_RESIDENT", 0.0)));
        c.layers    = lsa_env_str("LLAMA_LSA_LAYERS");
        c.keep_frac_layers = lsa_env_str("LLAMA_LSA_KEEP_FRAC_LAYERS");
        c.keep_sched       = lsa_env_str("LLAMA_LSA_KEEP_SCHED");
        c.resident_mib     = std::max(0.0, lsa_env_f64("LLAMA_LSA_RESIDENT_MIB", 0.0));
        c.resident_sticky  = lsa_env_f64("LLAMA_LSA_RESIDENT_STICKY", 0.0);
        c.resident_auto        = lsa_env_str("LLAMA_LSA_RESIDENT") == "auto" && c.resident_mib <= 0.0;
        c.resident_reserve_mib = std::max(0.0, lsa_env_f64("LLAMA_LSA_RESIDENT_RESERVE_MIB", c.resident_reserve_mib));
        c.resident_compute_mib = std::max(0.0, lsa_env_f64("LLAMA_LSA_RESIDENT_COMPUTE_MIB", c.resident_compute_mib));
        c.resident_reserve_set = getenv("LLAMA_LSA_RESIDENT_RESERVE_MIB") != nullptr;
        c.resident_compute_set = getenv("LLAMA_LSA_RESIDENT_COMPUTE_MIB") != nullptr;
        c.resident_margin_mib  = getenv("LLAMA_LSA_RESIDENT_MARGIN_MIB") ? std::max(0.0, lsa_env_f64("LLAMA_LSA_RESIDENT_MARGIN_MIB", 0.0)) : -1.0;
        c.resident_margin_from = c.resident_margin_mib >= 0.0 ? lsa_env_str("LLAMA_LSA_RESIDENT_MARGIN_FROM") : std::string();
        c.resident_target      = std::min(100.0, std::max(1.0, lsa_env_f64("LLAMA_LSA_RESIDENT_TARGET", c.resident_target)));
        c.resident_keep_fit    = lsa_env_i64("LLAMA_LSA_RESIDENT_LOWER_KEEP", lsa_env_i64("LLAMA_LSA_RESIDENT_KEEP_FIT", 0)) != 0;
        c.resident_keep_min    = std::min(1.0, std::max(0.0, lsa_env_f64("LLAMA_LSA_RESIDENT_KEEP_MIN", c.resident_keep_min)));
        c.resident_stage_auto  = lsa_env_i64("LLAMA_LSA_RESIDENT_AUTO_STAGE", 1) != 0 && getenv("GGML_CUDA_LSA_RESIDENT_STAGE") == nullptr;
        if (c.resident_auto) {
            c.resident = 100.0; // marks residency on; the slot count comes from llama_lsa_resident_fit at cache build
        }
        if (c.resident_mib > 0.0 && c.resident <= 0.0) {
            c.resident = 100.0; // marks residency on; the slot count comes from resident_mib
        }
        if (c.bucket < 1) {
            c.bucket = 1;
        }
        if (c.j3 && c.mode != llama_lsa_config::OFF) {
            auto pct_ok = [](int p) { return p == 0 || p == 100 || p == 110 || p == 125 || p == 150; };
            const char * pf = getenv("GGML_CUDA_LSA_PREFETCH");
            if (!pct_ok(c.j3_a3) || !pct_ok(c.j3_a1)) {
                LLAMA_LOG_WARN("%s: LLAMA_LSA_J3_A3/_A1 must be 0, 100, 110, 125 or 150 (got %d/%d): J3 off\n", __func__,
                        c.j3_a3, c.j3_a1);
                c.j3 = false;
            } else if (c.spec_recall || (pf && atoi(pf) > 0)) {
                LLAMA_LOG_WARN("%s: LLAMA_LSA_J3 does not combine with LLAMA_LSA_SPEC_RECALL or GGML_CUDA_LSA_PREFETCH: J3 off\n", __func__);
                c.j3 = false;
            } else if (c.resident <= 0.0) {
                LLAMA_LOG_WARN("%s: LLAMA_LSA_J3 needs LLAMA_LSA_RESIDENT (record-page slots): J3 off\n", __func__);
                c.j3 = false;
            }
        }
        if (c.mode == llama_lsa_config::OFF) {
            c.j3 = false;
        }
        if (c.mode != llama_lsa_config::OFF && c.map_path.empty() && !c.mapless_probe()) {
            LLAMA_LOG_WARN("%s: LLAMA_LSA_MODE set but LLAMA_LSA_MAP is empty, LSA stays off\n", __func__);
            c.mode = llama_lsa_config::OFF;
        }
        if (c.mode != llama_lsa_config::OFF) {
            // exact radix select for the wide index-score rows (k = thousands of cells); without it ggml-cuda sorts
            // whole 100K-200K rows. Same kept set as the sort (ggml-cuda top-k.cu), rows it does not take are unchanged.
            // An explicit GGML_CUDA_TOPK_SELECT (0 = off) wins.
            setenv("GGML_CUDA_TOPK_SELECT", "1", 0);
            // the "LSA probe ON" WARN line is logged per context (llama_context ctor), not here: this static is first
            // built inside the server's memory-fit dry run, whose logger drops WARN
        }
        return c;
    }();
    return cfg;
}

static std::string llama_lsa_describe_cfg(const llama_lsa_config & c) {
    static const char * names[] = { "off", "key", "pos", "oracle", "mass", "quest" };
    std::ostringstream os;
    os << "mode=" << names[c.mode] << " keep=" << c.keep << " keep_frac=" << c.keep_frac << " sinks=" << c.sinks
       << " window=" << c.window << " bucket=" << c.bucket << " max_rows=" << c.max_rows << " min_pos=" << c.min_pos
       << " graph_reuse=" << (c.graph_reuse ? 1 : 0) << " fa_sparse=" << (c.fa_sparse ? 1 : 0) << " shared=" << (c.shared_union ? 3 : c.shared ? (c.shared_max ? 2 : 1) : 0) << " smax_chain=" << (c.smax_chain ? 1 : 0) << " elig=" << (c.elig_drop ? 1 : 0) << " pick_every=" << c.pick_every << " pick_check=" << (c.pick_check ? 1 : 0) << " blist_lean=" << (c.blist_lean ? 1 : 0) << " packed=" << (c.packed ? 1 : 0) << " blk_list=" << (c.blk_list ? 1 : 0) << " vk=" << (c.vk ? 1 : 0) << " fast=" << (c.fast ? 1 : 0) << " idx_cache=" << (c.idx_cache ? 1 : 0) << " idx_q8=" << (c.idx_q8 ? 1 : 0) << " sj_compact=" << (c.sj_compact ? 1 : 0) << " block=" << c.block << " block_tau=" << c.block_tau << " block_sum=" << (c.block_sum ? c.block_sum_mode : 0) << " comp=" << (c.comp ? (c.comp_fast ? 2 : 1) : 0) << " comp_bias=" << c.comp_bias << " comp_max_t=" << c.comp_max_t << " comp_lb=" << (c.comp_lb ? 1 : 0) << " blk_mask_legacy=" << (c.blk_mask_legacy ? 1 : 0) << " comp_cnt_legacy=" << (c.comp_cnt_legacy ? 1 : 0) << " comp_cnt_exp=" << (c.comp_cnt_exp ? 1 : 0) << " rec_blk=" << (c.rec_blk ? 1 : 0) << " fused_sel=" << (c.fused_sel ? 1 : 0) << " unit_op=" << (c.unit_op ? 1 : 0) << " fused_comp=" << (c.fused_comp ? 1 : 0) << " ring=" << (c.ring_legacy ? std::string("legacy") : std::to_string(c.ring_rows)) << " blk_lse=" << (c.blk_lse ? 1 : 0) << " refine=" << c.refine << " refine_mult=" << c.refine_mult << " refine_sh=" << (c.refine_sh ? 1 : 0) << " blist_fine=" << (c.blist_min_unit < 16 ? 1 : 0) << " dump=" << (c.dump.empty() ? 0 : 1) << " sj_skip=" << (c.sj_skip ? 1 : 0) << " per_head=" << (c.per_head ? 1 : 0) << " resident=" << c.resident << " res_auto=" << (c.resident_auto ? 1 : 0) << " layers=" << (c.layers.empty() ? "all" : c.layers)
       << " keep_frac_layers=" << (c.keep_frac_layers.empty() ? "none" : c.keep_frac_layers)
       << " keep_sched=" << (c.keep_sched.empty() ? "none" : c.keep_sched) << " resident_mib=" << c.resident_mib << " sticky=" << c.resident_sticky
       << " topp=" << c.topp << " topp_tau=" << c.topp_tau << " topp_min=" << c.topp_min << " stats=" << (c.stats ? 1 : 0) << " recall=" << (c.recall ? 1 : 0)
       << " spec_recall=" << (c.spec_recall ? 1 : 0)
       << " j3=" << (c.j3 ? 1 : 0) << " j3_a3=" << c.j3_a3 << " j3_a1=" << c.j3_a1 << " j3_prev=" << (c.j3_prev ? 1 : 0)
       << " j3_first=" << c.j3_first << (c.gather ? " gather=1" : "") << " map=" << c.map_path;
    return os.str();
}

static std::atomic<uint32_t> g_lsa_log_epoch{1};
static std::atomic<uint64_t> g_lsa_graph_gen{1};

static std::atomic<double>  g_lsa_keep_cap_frac{0.0};
static std::atomic<int64_t> g_lsa_keep_cap_cells{0};

void llama_lsa_set_keep_cap(double frac, int64_t cells) {
    g_lsa_keep_cap_frac.store(frac);
    g_lsa_keep_cap_cells.store(cells);
}

double llama_lsa_keep_cap_frac() {
    return g_lsa_keep_cap_frac.load(std::memory_order_relaxed);
}

int64_t llama_lsa_keep_cap_cells() {
    return g_lsa_keep_cap_cells.load(std::memory_order_relaxed);
}

static std::mutex g_lsa_compute_mtx;
static std::vector<std::pair<const void *, int64_t>> g_lsa_compute;

void llama_lsa_note_compute(const void * dev, int64_t bytes) {
    if (!dev || bytes <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lk(g_lsa_compute_mtx);
    for (auto & [d, b] : g_lsa_compute) {
        if (d == dev) {
            b = std::max(b, bytes);
            return;
        }
    }
    g_lsa_compute.emplace_back(dev, bytes);
}

int64_t llama_lsa_measured_compute(const void * dev) {
    std::lock_guard<std::mutex> lk(g_lsa_compute_mtx);
    for (const auto & [d, b] : g_lsa_compute) {
        if (d == dev) {
            return b;
        }
    }
    return 0;
}

void llama_lsa_log_epoch_bump() {
    g_lsa_log_epoch.fetch_add(1, std::memory_order_relaxed);
}

void llama_lsa_comp_note(bool on, int32_t seq, int mode, const char * why) {
    static std::mutex mtx;
    static std::map<int32_t, bool> st;
    static uint32_t ep = 0;
    std::lock_guard<std::mutex> lock(mtx);
    const uint32_t e = g_lsa_log_epoch.load(std::memory_order_relaxed);
    if (e != ep) { // a new context: say it again (the fit dry run drops WARN, see llama_lsa_log_once)
        ep = e;
        st.clear();
    }
    auto it = st.find(seq);
    if (it != st.end() && it->second == on) {
        return;
    }
    st[seq] = on;
    if (on) {
        LLAMA_LOG_WARN("%s: lsa comp on mode=%d seq=%d\n", __func__, mode, (int) seq);
    } else {
        LLAMA_LOG_WARN("%s: lsa comp OFF mode=%d seq=%d: %s; compensation off until this sequence restarts at "
                "position 0\n", __func__, mode, (int) seq, why ? why : "?");
    }
}

bool llama_lsa_log_once(uint32_t & seen) {
    const uint32_t e = g_lsa_log_epoch.load(std::memory_order_relaxed);
    if (seen == e) {
        return false;
    }
    seen = e;
    return true;
}

uint64_t llama_lsa_graph_gen() {
    return g_lsa_graph_gen.load(std::memory_order_relaxed);
}

bool llama_lsa_graph_reuse_on() {
    static const bool on = [] {
        const auto & c = llama_lsa_cfg();
        return c.graph_reuse && c.idx_cache && c.pick_every <= 1 && !c.pick_check && c.min_pos <= 0 && c.dump.empty() &&
                c.score_log.empty() && c.resident <= 0.0 && !c.resident_auto;
    }();
    return on;
}

std::string llama_lsa_check_cache(uint32_t bits_k, uint32_t bits_v, ggml_type body, ggml_type type_k, ggml_type type_v) {
    const auto & c = llama_lsa_cfg();
    if (c.mode == llama_lsa_config::OFF) {
        return {};
    }
    const bool sj = bits_k > 0 || bits_v > 0;
    const bool resident = c.resident > 0.0 || c.resident_auto || c.resident_mib > 0.0;
    // plain K types the LSA probe scores (llama-graph.cpp lsa_k_plain); V any plain type the FA node takes
    const auto k_plain = [](ggml_type t) {
        return t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_F32 || t == GGML_TYPE_Q8_0 ||
               t == GGML_TYPE_Q5_0 || t == GGML_TYPE_Q5_1 || t == GGML_TYPE_Q4_0 || t == GGML_TYPE_Q4_1 ||
               t == GGML_TYPE_IQ4_NL;
    };
    if (c.gather) {
        // the gather path is not part of this build: plain f16 / q8_0 run the masked LSA node
        return "LLAMA_LSA_GATHER=1 is not available in this build (plain f16/q8_0 LSA runs the masked path; unset it)";
    }
    if (sj && c.fa_sparse) {
        return "LLAMA_LSA_FA_SPARSE is for plain K/V (an SJ-KVaRN node ignores the sparse bound)";
    }
    if (!sj) {
        if (!k_plain(type_k) || !k_plain(type_v)) {
            return std::string("plain K/V cache ") + ggml_type_name(type_k) + "/" + ggml_type_name(type_v) +
                   " (LSA scores plain f16/bf16/f32/q8_0/q5/q4/iq4_nl caches, SJ-KVaRN 3/2t, 3/3t and 4/4 scalar)";
        }
        if (resident || c.j3) {
            return "LLAMA_LSA_RESIDENT / LLAMA_LSA_J3 need an SJ-KVaRN body (a plain cache has no record pages)";
        }
        if (c.sj_compact) {
            return "LLAMA_LSA_SJ_COMPACT needs an SJ-KVaRN body";
        }
        if (c.sj_skip) {
            LLAMA_LOG_WARN("%s: LLAMA_LSA_SJ_SKIP has no effect on a plain cache (the pick still applies)\n", __func__);
        }
        return {};
    }
    const bool trellis = body == GGML_TYPE_I16;
    const bool lb  = trellis && bits_k == 3 && (bits_v == 2 || bits_v == 3);
    const bool s44 = body == GGML_TYPE_F32 && bits_k == 4 && bits_v == 4;
    if (!lb && !s44) {
        return "SJ-KVaRN " + std::to_string(bits_k) + "/" + std::to_string(bits_v) + (trellis ? " trellis" :
               body == GGML_TYPE_F32 ? " scalar" : " turbo4") +
               " body (LSA runs on SJ-KVaRN 3/2t, 3/3t and 4/4 scalar, plain f16 and q8_0)";
    }
    if (!c.idx_cache || c.fast || (c.mode != llama_lsa_config::KEY && c.mode != llama_lsa_config::POS)) {
        return "SJ-KVaRN needs LLAMA_LSA_IDX_CACHE=1, mode key or pos, and no LLAMA_LSA_FAST (else every layer would "
               "stay dense)";
    }
    if (lb && (c.per_head || c.sj_compact)) {
        return "LLAMA_LSA_PER_HEAD / LLAMA_LSA_SJ_COMPACT are 4/4 scalar only (on 3/2t and 3/3t use "
               "GGML_SJKVARN_TILE_SKIP=1)";
    }
    if (lb && c.sj_skip) {
        // the 8 GB presets set it; on the low-bit tile kernel it skips nothing, the pick still applies
        LLAMA_LOG_WARN("%s: LLAMA_LSA_SJ_SKIP has no effect on SJ-KVaRN %u/%u trellis (GGML_SJKVARN_TILE_SKIP=1 skips "
                       "dead tiles there)\n", __func__, bits_k, bits_v);
    }
    if (s44 && c.j3) {
        return "LLAMA_LSA_J3 needs record pages (SJ-KVaRN 3/2t or 3/3t); the 4/4 scalar body keeps 16-cell strips";
    }
    return {};
}

std::string llama_lsa_describe() {
    return llama_lsa_describe_cfg(llama_lsa_cfg());
}

bool llama_lsa_config::layer_enabled(int il) const {
    if (layers.empty()) {
        return true;
    }
    static std::once_flag once;
    static std::set<int> ids;
    std::call_once(once, [this] {
        std::stringstream ss(layers);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                ids.insert(atoi(item.c_str()));
            }
        }
    });
    return ids.count(il) != 0;
}

double llama_lsa_config::keep_frac_for(int il) const {
    return keep_frac_for(il, keep_frac);
}

double llama_lsa_config::keep_frac_sched(int64_t n_kv) const {
    if (keep_sched.empty()) {
        return -1.0;
    }
    static std::once_flag once;
    static std::vector<std::pair<int64_t, double>> pts; // (n_kv threshold, fraction), ascending
    std::call_once(once, [this] {
        std::stringstream ss(keep_sched);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t c = item.find(':');
            if (c == std::string::npos || c == 0) {
                LLAMA_LOG_WARN("%s: LLAMA_LSA_KEEP_SCHED item '%s' is not <ctx>:<frac>, ignored\n", __func__, item.c_str());
                continue;
            }
            char * end = nullptr;
            double t = strtod(item.c_str(), &end);
            if (end && (*end == 'K' || *end == 'k')) {
                t *= 1024.0;
            }
            const double f = strtod(item.c_str() + c + 1, nullptr);
            if (t <= 0.0 || f <= 0.0) {
                LLAMA_LOG_WARN("%s: LLAMA_LSA_KEEP_SCHED item '%s' needs ctx > 0 and frac > 0, ignored\n", __func__, item.c_str());
                continue;
            }
            pts.emplace_back((int64_t) t, f);
        }
        std::sort(pts.begin(), pts.end());
        std::string d;
        for (const auto & p : pts) {
            d += " " + std::to_string(p.first) + ":" + std::to_string(p.second);
        }
        LLAMA_LOG_WARN("%s: LSA keep schedule (n_kv:frac, dense below the first):%s\n", __func__, d.empty() ? " none (all dense)" : d.c_str());
    });
    double f = 0.0;
    for (const auto & p : pts) {
        if (n_kv >= p.first) {
            f = p.second;
        }
    }
    return f;
}

double llama_lsa_config::keep_frac_for(int il, double base) const {
    if (keep_frac_layers.empty()) {
        return base;
    }
    static std::once_flag once;
    static std::map<int, double> fr;
    std::call_once(once, [this] {
        std::stringstream ss(keep_frac_layers);
        std::string item;
        while (std::getline(ss, item, ',')) {
            const size_t c = item.find(':');
            if (c == std::string::npos) {
                LLAMA_LOG_WARN("%s: LLAMA_LSA_KEEP_FRAC_LAYERS item '%s' has no ':', ignored\n", __func__, item.c_str());
                continue;
            }
            fr[atoi(item.substr(0, c).c_str())] = strtod(item.c_str() + c + 1, nullptr);
        }
    });
    const auto it = fr.find(il);
    return it == fr.end() ? base : it->second;
}

namespace {

struct lsa_host_map {
    std::mutex    mtx;
    bool          tried    = false;
    bool          ok       = false;
    ggml_context * host_ctx = nullptr; // tensor data loaded by gguf (never freed: the probe keeps it for the process)
    gguf_context * gguf     = nullptr;
    std::array<uint8_t, SHA256_DIGEST_SIZE> identity = {};
    std::map<std::pair<int, ggml_backend_buffer_type_t>, std::pair<ggml_tensor *, ggml_tensor *>> dev;
};

lsa_host_map & lsa_map() {
    static lsa_host_map * m = new lsa_host_map(); // leaked on purpose: device buffers must outlive static teardown
    return *m;
}

bool lsa_load_host(lsa_host_map & m, const std::string & path) {
    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &m.host_ctx };
    m.gguf = gguf_init_from_file(path.c_str(), params);
    if (!m.gguf) {
        LLAMA_LOG_ERROR("%s: cannot read LSA map %s, LSA stays off\n", __func__, path.c_str());
        return false;
    }
    const int64_t kid = gguf_find_key(m.gguf, "lsa.index_dim");
    const int64_t idim = kid >= 0 ? (int64_t) gguf_get_val_u32(m.gguf, kid) : -1;
    int n_layers = 0;
    sha256_t hash;
    sha256_init(&hash);
    // walk the file's named tensors: with no_alloc=false the ctx also holds the gguf data blob tensor
    for (int64_t i = 0; i < gguf_get_n_tensors(m.gguf); ++i) {
        ggml_tensor * t = ggml_get_tensor(m.host_ctx, gguf_get_tensor_name(m.gguf, i));
        if (!t || t->type != GGML_TYPE_F32 || ggml_n_dims(t) != 2 || (idim > 0 && t->ne[1] != idim)) {
            LLAMA_LOG_ERROR("%s: LSA map tensor %s has an unexpected type/shape, LSA stays off\n", __func__, gguf_get_tensor_name(m.gguf, i));
            return false;
        }
        n_layers += strstr(t->name, ".wk") != nullptr;
        sha256_update(&hash, (const uint8_t *) t->name, strlen(t->name) + 1);
        sha256_update(&hash, (const uint8_t *) t->ne, sizeof(t->ne));
        sha256_update(&hash, (const uint8_t *) t->data, ggml_nbytes(t));
    }
    sha256_final(&hash, m.identity.data());
    LLAMA_LOG_INFO("%s: LSA map %s: %d layers, index_dim %lld\n", __func__, path.c_str(), n_layers, (long long) idim);
    return n_layers > 0;
}

} // namespace

static std::array<uint8_t, SHA256_DIGEST_SIZE> lsa_map_identity() {
    const auto & cfg = llama_lsa_cfg();
    std::array<uint8_t, SHA256_DIGEST_SIZE> digest;
    if (cfg.mapless_probe()) {
        const char * tag = cfg.mode == llama_lsa_config::ORACLE ? "lsa-mapless-oracle-v1" : "lsa-mapless-mass-v1";
        sha256_hash(digest.data(), (const uint8_t *) tag, strlen(tag));
        return digest;
    }
    auto & m = lsa_map();
    std::lock_guard<std::mutex> lock(m.mtx);
    if (!m.tried) {
        m.tried = true;
        m.ok = lsa_load_host(m, cfg.map_path);
    }
    if (!m.ok) {
        throw std::runtime_error("cannot identify the LSA map for prompt-cache state");
    }
    return m.identity;
}

bool llama_lsa_map_has_layer(int il) {
    const auto & cfg = llama_lsa_cfg();
    if (cfg.mode == llama_lsa_config::OFF || !cfg.layer_enabled(il)) {
        return false;
    }
    auto & m = lsa_map();
    std::lock_guard<std::mutex> lock(m.mtx);
    if (!m.tried) {
        m.tried = true;
        m.ok = lsa_load_host(m, cfg.map_path);
    }
    char name_k[64];
    snprintf(name_k, sizeof(name_k), "lsa.%d.wk", il);
    return m.ok && ggml_get_tensor(m.host_ctx, name_k) != nullptr;
}

bool llama_lsa_get_map(int il, ggml_backend_buffer_type_t buft, ggml_tensor ** wq, ggml_tensor ** wk) {
    const auto & cfg = llama_lsa_cfg();
    if (cfg.mode == llama_lsa_config::OFF || !cfg.layer_enabled(il)) {
        return false;
    }

    auto & m = lsa_map();
    std::lock_guard<std::mutex> lock(m.mtx);

    if (!m.tried) {
        m.tried = true;
        m.ok = lsa_load_host(m, cfg.map_path);
    }
    if (!m.ok) {
        return false;
    }

    const auto key = std::make_pair(il, buft);
    auto it = m.dev.find(key);
    if (it != m.dev.end()) {
        *wq = it->second.first;
        *wk = it->second.second;
        return *wq != nullptr;
    }

    char name_q[64];
    char name_k[64];
    snprintf(name_q, sizeof(name_q), "lsa.%d.wq", il);
    snprintf(name_k, sizeof(name_k), "lsa.%d.wk", il);
    ggml_tensor * hq = ggml_get_tensor(m.host_ctx, name_q);
    ggml_tensor * hk = ggml_get_tensor(m.host_ctx, name_k);
    if (!hq || !hk || !ggml_are_same_shape(hq, hk)) {
        m.dev[key] = { nullptr, nullptr };
        return false;
    }

    ggml_init_params ip = { /*.mem_size =*/ 2*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * dctx = ggml_init(ip);
    ggml_tensor * dq = ggml_dup_tensor(dctx, hq);
    ggml_tensor * dk = ggml_dup_tensor(dctx, hk);
    ggml_set_name(dq, name_q);
    ggml_set_name(dk, name_k);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(dctx, buft);
    if (!buf) {
        LLAMA_LOG_ERROR("%s: cannot allocate the LSA map for layer %d\n", __func__, il);
        m.dev[key] = { nullptr, nullptr };
        return false;
    }
    ggml_backend_tensor_set(dq, hq->data, 0, ggml_nbytes(hq));
    ggml_backend_tensor_set(dk, hk->data, 0, ggml_nbytes(hk));

    m.dev[key] = { dq, dk };
    *wq = dq;
    *wk = dk;
    return true;
}

namespace {
struct lsa_idx_entry {
    ggml_tensor *         t     = nullptr;
    ggml_tensor *         smax  = nullptr; // LLAMA_LSA_BLOCK_SUM block summaries (same buffer)
    ggml_tensor *         smin  = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    uint64_t              built = UINT64_MAX; // generation the cell contents match; UINT64_MAX = never built
    uint64_t              gen   = 0;          // bumped when this K cache's cells change outside the graph
    uint64_t              touch = 0;          // lsa_idx_store::seq at the last invalidate: a restore picks the newest
    std::string           name;               // K cache tensor name: the key in saved states
    ggml_context *        ctx   = nullptr;
};
struct lsa_idx_store {
    std::mutex                                     mtx;
    std::map<const ggml_tensor *, lsa_idx_entry>   e;
    int64_t                                        ring_hi = 0; // LLAMA_LSA_COMP=2 on SJ-KVaRN: staging ring holds positions [ring_hi - n_ring, ring_hi)
    int64_t                                        ring_lo = 0; // ... but none below ring_lo: set by a far rewind to a block start, where the
                                                                // ring rows under the rewind point still hold higher (old) positions
    uint64_t                                       seq     = 0; // invalidate counter (entry touch)
};
lsa_idx_store & lsa_idx() {
    static lsa_idx_store * s = new lsa_idx_store(); // leaked on purpose, like the map buffers
    return *s;
}
} // namespace

bool llama_lsa_idx_get(const ggml_tensor * kcache, int64_t d_idx, ggml_backend_buffer_type_t buft, llama_lsa_idx_slot * out,
        int64_t kv_size, int64_t n_head) {
    if (!llama_lsa_cfg().idx_cache) {
        return false;
    }
    if (kv_size <= 0) {
        kv_size = kcache->ne[1];
    }
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto & en = s.e[kcache];
    const ggml_type type = llama_lsa_cfg().idx_q8 && d_idx % ggml_blck_size(GGML_TYPE_Q8_0) == 0 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    if (en.t && (en.t->ne[0] != d_idx || en.t->ne[1] != kv_size || en.t->ne[2] != n_head)) {
        return false; // shape changed under the same tensor: never expected, stay on the recompute path
    }
    if (!en.t) {
        ggml_init_params ip = { /*.mem_size =*/ 3*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_3d(ctx, type, d_idx, kv_size, n_head);
        ggml_format_name(t, "lsa_idx_%s", kcache->name);
        const auto & cfg = llama_lsa_cfg();
        if (cfg.block_sum && kv_size >= cfg.block) {
            // per-KV-head cache (LLAMA_LSA_PER_HEAD=1): one summary plane per head, head-major like the keys
            en.smax = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_idx, kv_size/cfg.block, n_head);
            en.smin = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d_idx, kv_size/cfg.block, n_head);
            ggml_format_name(en.smax, "lsa_smax_%s", kcache->name);
            ggml_format_name(en.smin, "lsa_smin_%s", kcache->name);
        }
        en.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!en.buf) {
            LLAMA_LOG_ERROR("%s: cannot allocate the LSA index cache (%lld x %lld x %lld %s), using the recompute path\n",
                    __func__, (long long) d_idx, (long long) kv_size, (long long) n_head, ggml_type_name(type));
            s.e.erase(kcache);
            return false;
        }
        ggml_backend_buffer_clear(en.buf, 0);
        g_lsa_graph_gen.fetch_add(1, std::memory_order_relaxed);
        en.t = t;
        en.ctx = ctx;
        en.name = kcache->name;
        LLAMA_LOG_INFO("%s: LSA index cache for %s: %lld x %lld x %lld %s (%.1f MiB)\n", __func__, kcache->name,
                (long long) d_idx, (long long) kv_size, (long long) n_head, ggml_type_name(type), ggml_nbytes(t)/1048576.0);
    }
    out->t     = en.t;
    out->smax  = en.smax;
    out->smin  = en.smin;
    out->valid = en.built == en.gen;
    out->epoch = en.gen;
    return true;
}

void llama_lsa_idx_commit(const ggml_tensor * kcache, uint64_t epoch) {
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto it = s.e.find(kcache);
    if (it != s.e.end() && epoch == it->second.gen) {
        it->second.built = epoch;
    }
}

namespace {
struct lsa_mean_entry {
    ggml_tensor *         km    = nullptr;
    ggml_tensor *         vm    = nullptr;
    ggml_tensor *         kr    = nullptr; // SJ-KVaRN staging ring (n_ring > 0)
    ggml_tensor *         vr    = nullptr;
    ggml_backend_buffer_t buf   = nullptr;
    uint64_t              built = UINT64_MAX;
    uint64_t              gen   = 0; // same rule as lsa_idx_entry::gen
    uint64_t              touch = 0; // same rule as lsa_idx_entry::touch
    std::string           name;      // K cache tensor name (the key tensor may be gone by the time a state is read);
                                     // "#rot" appended when the means are in the rotated SJ-KVaRN basis (LLAMA_LSA_VK),
                                     // so a state holding one basis never restores into a cache of the other
    bool                  rot   = false;
    ggml_context *        ctx   = nullptr;
};
std::map<const ggml_tensor *, lsa_mean_entry> & lsa_means() { // guarded by lsa_idx().mtx
    static auto * m = new std::map<const ggml_tensor *, lsa_mean_entry>();
    return *m;
}
} // namespace

bool llama_lsa_mean_get(const ggml_tensor * kcache, int64_t dk, int64_t dv, int64_t n_blk, ggml_backend_buffer_type_t buft,
        ggml_tensor ** km, ggml_tensor ** vm, bool * valid, uint64_t * epoch,
        int64_t n_ring, ggml_tensor ** kr, ggml_tensor ** vr, bool rot) {
    if (!llama_lsa_cfg().comp_fast || n_blk <= 0 || (n_ring > 0 && (!kr || !vr))) {
        return false;
    }
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto & en = lsa_means()[kcache];
    if (en.km && (en.km->ne[0] != dk || en.vm->ne[0] != dv || en.km->ne[1] != n_blk ||
                  (en.kr ? en.kr->ne[1] : 0) != n_ring || en.rot != rot)) {
        return false;
    }
    if (!en.km) {
        ggml_init_params ip = { /*.mem_size =*/ 4*ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dk, n_blk);
        ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dv, n_blk);
        ggml_format_name(k, "lsa_kmean_%s", kcache->name);
        ggml_format_name(v, "lsa_vmean_%s", kcache->name);
        if (n_ring > 0) {
            en.kr = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dk, n_ring);
            en.vr = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dv, n_ring);
            ggml_format_name(en.kr, "lsa_kring_%s", kcache->name);
            ggml_format_name(en.vr, "lsa_vring_%s", kcache->name);
        }
        en.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!en.buf) {
            LLAMA_LOG_ERROR("%s: cannot allocate the LSA block-mean cache, compensation off for %s\n", __func__, kcache->name);
            lsa_means().erase(kcache);
            return false;
        }
        ggml_backend_buffer_clear(en.buf, 0);
        g_lsa_graph_gen.fetch_add(1, std::memory_order_relaxed);
        en.km = k;
        en.vm = v;
        en.ctx = ctx;
        en.name = std::string(kcache->name) + (rot ? "#rot" : "");
        en.rot  = rot;
        LLAMA_LOG_INFO("%s: LSA block-mean cache for %s: %lld blocks, ring %lld rows (%.1f MiB)\n", __func__, kcache->name,
                (long long) n_blk, (long long) n_ring, ggml_backend_buffer_get_size(en.buf)/1048576.0);
    }
    *km    = en.km;
    *vm    = en.vm;
    if (n_ring > 0) {
        *kr = en.kr;
        *vr = en.vr;
    }
    *valid = en.built == en.gen;
    *epoch = en.gen;
    return true;
}

void llama_lsa_mean_commit(const ggml_tensor * kcache, uint64_t epoch) {
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto it = lsa_means().find(kcache);
    if (it != lsa_means().end() && epoch == it->second.gen) {
        it->second.built = epoch;
    }
}

namespace {
struct lsa_pick_entry {
    ggml_tensor *         sel   = nullptr; // F32 [n_cap]
    ggml_backend_buffer_t buf   = nullptr;
    ggml_context *        ctx   = nullptr;
    uint64_t              gen   = 0;          // bumped by invalidate
    uint64_t              built = UINT64_MAX; // gen of the committed pick
    int64_t               pos0  = -1;         // first row position of the pick graph, then of the last reuse
    int64_t               edge  = 0;
    int64_t               nbs   = 0;
    int64_t               n_use = 0;          // graphs that used this pick (1 = the pick itself)
    int32_t               seq   = -1;         // sequence of the pick graph
};
std::map<const ggml_tensor *, lsa_pick_entry> & lsa_picks() { // guarded by lsa_idx().mtx
    static auto * m = new std::map<const ggml_tensor *, lsa_pick_entry>();
    return *m;
}
} // namespace

bool llama_lsa_pick_get(const ggml_tensor * kcache, int64_t n_cap, ggml_backend_buffer_type_t buft, int64_t pos0,
        int64_t edge, int64_t nbs, int32_t seq, llama_lsa_pick_slot * out) {
    const auto & cfg = llama_lsa_cfg();
    if (cfg.pick_every <= 1 || !cfg.idx_cache || n_cap <= 0 || nbs > n_cap) { // invalidate hooks only run with the index cache
        return false;
    }
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto & en = lsa_picks()[kcache];
    if (en.sel && en.sel->ne[0] != n_cap) {
        return false;
    }
    if (!en.sel) {
        ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_cap);
        ggml_format_name(t, "lsa_pick_%s", kcache->name);
        en.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!en.buf) {
            ggml_free(ctx);
            lsa_picks().erase(kcache);
            return false;
        }
        ggml_backend_buffer_clear(en.buf, 0);
        en.sel = t;
        en.ctx = ctx;
    }
    out->sel   = en.sel;
    out->gen   = en.gen;
    out->reuse = en.built == en.gen && en.n_use < cfg.pick_every && pos0 > en.pos0 && edge == en.edge && nbs >= en.nbs &&
                 seq == en.seq;
    out->age   = out->reuse ? en.n_use : 0;
    return true;
}

void llama_lsa_pick_commit(const ggml_tensor * kcache, uint64_t gen, bool reuse, int64_t pos0, int64_t edge, int64_t nbs,
        int32_t seq, bool expire) {
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    auto it = lsa_picks().find(kcache);
    if (it != lsa_picks().end() && expire) {
        it->second.built = UINT64_MAX;
        return;
    }
    if (it == lsa_picks().end() || gen != it->second.gen) {
        return;
    }
    auto & en = it->second;
    if (reuse) {
        en.n_use++;
        en.pos0 = pos0; // a rewind below the last reused graph re-picks
    } else {
        en.built = gen;
        en.pos0  = pos0;
        en.edge  = edge;
        en.nbs   = nbs;
        en.n_use = 1;
        en.seq   = seq;
    }
}

bool llama_lsa_idx_trace() {
    static const bool on = lsa_env_i64("LLAMA_LSA_IDX_TRACE", 0) != 0; // diagnostics at WARN (server.log drops INFO)
    return on;
}

void llama_lsa_idx_invalidate(const ggml_tensor * const * kcaches, size_t n, const char * why) {
    static const bool test_skip = lsa_env_i64("LLAMA_LSA_IDX_TEST_SKIP_INVALIDATE", 0) != 0; // negative control only
    // LLAMA_LSA_IDX_TEST_GLOBAL_INVALIDATE=1: old behaviour (any cache's restore/shift invalidates every layer of every
    // context, e.g. the MTP drafter's restore after the main one left SJ-KVaRN dense); negative control only
    static const bool test_global = lsa_env_i64("LLAMA_LSA_IDX_TEST_GLOBAL_INVALIDATE", 0) != 0;
    if (!llama_lsa_cfg().idx_cache || test_skip) {
        return;
    }
    g_lsa_graph_gen.fetch_add(1, std::memory_order_relaxed);
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    if (test_global) {
        for (auto & it : s.e)         { it.second.gen++; }
        for (auto & it : lsa_means()) { it.second.gen++; }
        for (auto & it : lsa_picks()) { it.second.gen++; }
        if (llama_lsa_idx_trace()) {
            LLAMA_LOG_WARN("%s: [lsa-trace] %s: global (test rule), %zu index + %zu mean entries\n", __func__,
                    why ? why : "?", s.e.size(), lsa_means().size());
        }
        return;
    }
    // only this K cache's layers: entries not created yet start unbuilt anyway. An empty list (a cache without
    // layers; std::vector::data() of an empty vector may be null) invalidates nothing - it used to fall into the
    // global rule above and leave every restored layer of every context dense.
    size_t n_idx = 0, n_mean = 0;
    for (size_t i = 0; kcaches != nullptr && i < n; ++i) {
        auto ie = s.e.find(kcaches[i]);
        if (ie != s.e.end()) {
            ie->second.gen++;
            ie->second.touch = ++s.seq;
            ++n_idx;
        }
        auto ip = lsa_picks().find(kcaches[i]);
        if (ip != lsa_picks().end()) {
            ip->second.gen++;
        }
        auto im = lsa_means().find(kcaches[i]);
        if (im != lsa_means().end()) {
            im->second.gen++;
            im->second.touch = ++s.seq;
            ++n_mean;
        }
    }
    if (llama_lsa_idx_trace()) {
        LLAMA_LOG_WARN("%s: [lsa-trace] %s: %zu K tensors, bumped %zu/%zu index + %zu/%zu mean entries\n", __func__,
                why ? why : "?", n, n_idx, s.e.size(), n_mean, lsa_means().size());
    }
}

void llama_lsa_idx_forget(const ggml_tensor * const * kcaches, size_t n) {
    // a K cache is being destroyed (context freed, e.g. the server's memory-fit dry runs): drop its index/mean entries
    // and free their device buffers. Without this, dead entries kept VRAM, could be picked by a restore of the same
    // name, and a later tensor at the same address would inherit them.
    g_lsa_graph_gen.fetch_add(1, std::memory_order_relaxed);
    if (kcaches == nullptr || n == 0) {
        return;
    }
    auto & s = lsa_idx();
    std::lock_guard<std::mutex> lock(s.mtx);
    size_t n_idx = 0, n_mean = 0;
    for (size_t i = 0; i < n; ++i) {
        auto ie = s.e.find(kcaches[i]);
        if (ie != s.e.end()) {
            if (ie->second.buf) { ggml_backend_buffer_free(ie->second.buf); }
            if (ie->second.ctx) { ggml_free(ie->second.ctx); }
            s.e.erase(ie);
            ++n_idx;
        }
        auto ip = lsa_picks().find(kcaches[i]);
        if (ip != lsa_picks().end()) {
            if (ip->second.buf) { ggml_backend_buffer_free(ip->second.buf); }
            if (ip->second.ctx) { ggml_free(ip->second.ctx); }
            lsa_picks().erase(ip);
        }
        auto im = lsa_means().find(kcaches[i]);
        if (im != lsa_means().end()) {
            if (im->second.buf) { ggml_backend_buffer_free(im->second.buf); }
            if (im->second.ctx) { ggml_free(im->second.ctx); }
            lsa_means().erase(im);
            ++n_mean;
        }
    }
    if (llama_lsa_idx_trace() && (n_idx || n_mean)) {
        LLAMA_LOG_WARN("%s: [lsa-trace] forget: %zu index + %zu mean entries freed, %zu + %zu left\n", __func__,
                n_idx, n_mean, s.e.size(), lsa_means().size());
    }
}

static const uint32_t LSA_IDX_STATE_MAGIC   = 0x4941534c; // "LSAI"
static const uint32_t LSA_IDX_STATE_VERSION = 2; // 2: per layer d_idx, n_head, ggml type; head-major rows (1: f16, n_head 1)
static const uint32_t LSA_IDX_STATE_VERSION_MEANS = 3; // 3 = 2 + LLAMA_LSA_COMP=2 block means and staging rings behind
                                                       // the layers; written only when compensation has valid means
static const uint32_t LSA_IDX_STATE_VERSION_MAP = 4; // flags + SHA256 of the loaded map before the layer header

namespace {
struct lsa_mean_save {
    std::string   name;
    ggml_tensor * km, * vm, * kr, * vr;
};
}

void llama_lsa_idx_state_write(llama_io_write_i & io, int64_t n_rows) {
    static const bool test_no_state = lsa_env_i64("LLAMA_LSA_IDX_TEST_NO_STATE", 0) != 0; // negative control only
    if (llama_lsa_cfg().mode == llama_lsa_config::OFF || !llama_lsa_cfg().idx_cache || n_rows <= 0 || test_no_state) {
        return;
    }
    std::vector<std::pair<std::string, ggml_tensor *>> valid;
    {
        auto & s = lsa_idx();
        std::lock_guard<std::mutex> lock(s.mtx);
        for (const auto & it : s.e) {
            const auto & en = it.second;
            if (en.t && en.built == en.gen && n_rows <= en.t->ne[1]) {
                valid.emplace_back(en.name, en.t);
            }
        }
    }
    if (llama_lsa_idx_trace()) {
        size_t n_all = 0;
        {
            auto & s = lsa_idx();
            std::lock_guard<std::mutex> lock(s.mtx);
            n_all = s.e.size();
        }
        LLAMA_LOG_WARN("%s: [lsa-trace] save: %zu/%zu index layers valid at %lld rows\n", __func__, valid.size(), n_all,
                (long long) n_rows);
    }
    if (valid.empty()) {
        return; // nothing valid: no suffix, the restore leaves the layers dense
    }
    std::sort(valid.begin(), valid.end()); // same byte layout for the size pass and the write pass
    // SJ-KVaRN compensation: the block means cannot be rebuilt from the sealed records, so they travel with the state
    std::vector<lsa_mean_save> means;
    int64_t ring_n = 0;
    static const bool test_no_means = lsa_env_i64("LLAMA_LSA_COMP_TEST_NO_STATE", 0) != 0; // negative control only
    if (!test_no_means) {
        auto & s = lsa_idx();
        std::lock_guard<std::mutex> lock(s.mtx);
        for (const auto & it : lsa_means()) {
            const auto & en = it.second;
            // the ring must still hold the tail block [B*floor((n_rows-1)/B), n_rows); rows past n_rows are rejected
            // draft rows (seq_rm after verify) and count as stale after the restore
            const int64_t B = llama_lsa_cfg().block > 1 ? llama_lsa_cfg().block : 128;
            const bool tail_in_ring = en.kr && n_rows <= s.ring_hi &&
                    ((n_rows - 1)/B)*B >= std::max<int64_t>(s.ring_hi - en.kr->ne[1], s.ring_lo);
            if (en.km && en.kr && en.built == en.gen && tail_in_ring && (ring_n == 0 || en.kr->ne[1] == ring_n)) {
                ring_n = en.kr->ne[1];
                means.push_back({ en.name, en.km, en.vm, en.kr, en.vr });
            }
        }
    }
    std::sort(means.begin(), means.end(), [](const lsa_mean_save & a, const lsa_mean_save & b) { return a.name < b.name; });
    const uint32_t version = LSA_IDX_STATE_VERSION_MAP;
    const uint32_t flags = means.empty() ? 0 : 1;
    const auto identity = lsa_map_identity();
    const uint32_t n = (uint32_t) valid.size();
    io.write(&LSA_IDX_STATE_MAGIC, sizeof(uint32_t));
    io.write(&version,             sizeof(uint32_t));
    io.write(&flags, sizeof(flags));
    io.write(identity.data(), identity.size());
    io.write(&n,      sizeof(n));
    io.write(&n_rows, sizeof(n_rows));
    for (const auto & v : valid) {
        ggml_tensor *  t      = v.second;
        const int64_t  d_idx  = t->ne[0];
        const int64_t  n_head = t->ne[2];
        const int32_t  type   = (int32_t) t->type;
        io.write_string(v.first);
        io.write(&d_idx,  sizeof(d_idx));
        io.write(&n_head, sizeof(n_head));
        io.write(&type,   sizeof(type));
        for (int64_t h = 0; h < n_head; ++h) {
            io.write_tensor(t, (size_t) h*t->nb[2], (size_t) n_rows*t->nb[1]);
        }
    }
    if (flags & 1) {
        const int64_t  B  = llama_lsa_cfg().block > 1 ? llama_lsa_cfg().block : 128;
        const uint32_t nm = (uint32_t) means.size();
        io.write(&nm,     sizeof(nm));
        io.write(&ring_n, sizeof(ring_n));
        io.write(&B,      sizeof(B));
        for (const auto & m : means) {
            const int64_t dk = m.km->ne[0], dv = m.vm->ne[0];
            const int64_t nb = std::min<int64_t>((n_rows + B - 1)/B, m.km->ne[1]); // the partial tail block too
            io.write_string(m.name);
            io.write(&dk, sizeof(dk));
            io.write(&dv, sizeof(dv));
            io.write(&nb, sizeof(nb));
            io.write_tensor(m.km, 0, (size_t) nb*m.km->nb[1]);
            io.write_tensor(m.vm, 0, (size_t) nb*m.vm->nb[1]);
            io.write_tensor(m.kr, 0, ggml_nbytes(m.kr)); // rows of the tail block that later ubatches re-read
            io.write_tensor(m.vr, 0, ggml_nbytes(m.vr));
        }
    }
}

// LLAMA_LSA_BLOCK_SUM: the summaries are not in the saved state; after a restore they are recomputed here from the
// restored index rows [0, n_rows) (whole blocks only; a partial tail block lies inside the always-kept window)
static void lsa_sum_from_host(const ggml_tensor * t, int64_t n_rows) {
    const auto & cfg = llama_lsa_cfg();
    ggml_tensor * smax = nullptr;
    ggml_tensor * smin = nullptr;
    {
        auto & s = lsa_idx();
        std::lock_guard<std::mutex> lock(s.mtx);
        for (auto & it : s.e) {
            if (it.second.t == t) {
                smax = it.second.smax;
                smin = it.second.smin;
            }
        }
    }
    if (!smax || !smin || t->type != GGML_TYPE_F16 || smax->ne[2] != t->ne[2]) {
        return; // summaries exist only for the f16 cache (IDX_Q8 is refused with BLOCK_SUM)
    }
    const int64_t d = t->ne[0], B = cfg.block;
    const int64_t nb = std::min<int64_t>(n_rows/B, smax->ne[1]);
    if (nb <= 0) {
        return;
    }
    std::vector<ggml_fp16_t> rows((size_t) nb*B*d), mx((size_t) nb*d), mn((size_t) nb*d);
    for (int64_t h = 0; h < t->ne[2]; ++h) { // one plane per KV head (1 for the shared cache)
    ggml_backend_tensor_get(t, rows.data(), (size_t) h*t->nb[2], rows.size()*sizeof(ggml_fp16_t));
    for (int64_t b = 0; b < nb; ++b) {
        for (int64_t c = 0; c < d; ++c) {
            float hi = -INFINITY, lo = INFINITY, s1 = 0.0f, s2 = 0.0f;
            for (int64_t r = 0; r < B; ++r) {
                const float x = ggml_fp16_to_fp32(rows[(size_t) (b*B + r)*d + c]);
                hi = std::max(hi, x);
                lo = std::min(lo, x);
                s1 += x;
                s2 += x*x;
            }
            if (cfg.block_sum_mode == 2) { // mean, variance (same order of operations as the graph: avg, avg of squares)
                const float mu = s1/(float) B;
                hi = mu;
                lo = std::max(0.0f, s2/(float) B - mu*mu);
            }
            mx[(size_t) b*d + c] = ggml_fp32_to_fp16(hi);
            mn[(size_t) b*d + c] = ggml_fp32_to_fp16(lo);
        }
    }
    ggml_backend_tensor_set(smax, mx.data(), (size_t) h*smax->nb[2], mx.size()*sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(smin, mn.data(), (size_t) h*smin->nb[2], mn.size()*sizeof(ggml_fp16_t));
    }
}

// version 3 tail: block means + staging rings per K cache tensor; a layer that does not match this process (LSA or
// compensation off, other -ub or block size) is skipped and its compensation stays off until a restart at position 0
static void lsa_means_state_read(llama_io_read_i & io, int64_t n_rows) {
    uint32_t nm = 0;
    int64_t  ring_n = 0, B = 0;
    io.read(&nm,     sizeof(nm));
    io.read(&ring_n, sizeof(ring_n));
    io.read(&B,      sizeof(B));
    if (nm > 4096 || ring_n <= 0 || ring_n > (1 << 20) || B <= 0 || B > 4096 || ring_n % B != 0) {
        throw std::runtime_error("corrupt LSA block-mean state header");
    }
    const auto & cfg = llama_lsa_cfg();
    const bool on = cfg.comp_fast && (cfg.block > 1 ? cfg.block : 128) == B;
    uint32_t n_ok = 0;
    std::vector<uint8_t> skip;
    for (uint32_t i = 0; i < nm; ++i) {
        std::string name;
        int64_t dk = 0, dv = 0, nb = 0;
        io.read_string(name);
        io.read(&dk, sizeof(dk));
        io.read(&dv, sizeof(dv));
        io.read(&nb, sizeof(nb));
        if (dk <= 0 || dk > 65536 || dv <= 0 || dv > 65536 || nb <= 0 || nb > (n_rows + B - 1)/B) {
            throw std::runtime_error("corrupt LSA block-mean state layer");
        }
        const size_t fp16 = sizeof(ggml_fp16_t);
        const size_t nbytes = (size_t) nb*(dk + dv)*fp16 + (size_t) ring_n*(dk + dv)*fp16;
        lsa_mean_entry * en = nullptr;
        if (on) {
            auto & s = lsa_idx();
            std::lock_guard<std::mutex> lock(s.mtx);
            // several contexts can hold a cache of the same name (MTP drafter, a fit dry run before the forget hook):
            // the K restore that just ran touched this process's live one last
            for (auto & it : lsa_means()) {
                auto & e = it.second;
                if (e.km && e.kr && name == e.name && e.km->ne[0] == dk && e.vm->ne[0] == dv &&
                        e.km->ne[1] >= nb && e.kr->ne[1] == ring_n && (!en || e.touch > en->touch)) {
                    en = &e;
                }
            }
        }
        if (en) {
            io.read_tensor(en->km, 0, (size_t) nb*en->km->nb[1]);
            io.read_tensor(en->vm, 0, (size_t) nb*en->vm->nb[1]);
            io.read_tensor(en->kr, 0, ggml_nbytes(en->kr));
            io.read_tensor(en->vr, 0, ggml_nbytes(en->vr));
            auto & s = lsa_idx();
            std::lock_guard<std::mutex> lock(s.mtx);
            en->built = en->gen; // the K restore already bumped this layer's generation
            s.ring_hi = n_rows;
            // the saved ring is only known good from the tail block on (the save rule; ring_lo is not in the state)
            s.ring_lo = ((n_rows - 1)/B)*B;
            ++n_ok;
        } else {
            for (size_t done = 0; done < nbytes; ) {
                const size_t c = std::min(nbytes - done, (size_t) 16 << 20);
                skip.resize(c);
                io.read(skip.data(), c);
                done += c;
            }
        }
    }
    LLAMA_LOG_INFO("%s: LSA block means restored for %u/%u layers (ring %lld rows)\n", __func__, n_ok, nm, (long long) ring_n);
    if (llama_lsa_idx_trace()) {
        LLAMA_LOG_WARN("%s: [lsa-trace] restore: means %u/%u layers (ring %lld rows)\n", __func__, n_ok, nm, (long long) ring_n);
    }
}

void llama_lsa_warn_restore_dense(const char * why, uint32_t n_ok, uint32_t n) {
    const auto & cfg = llama_lsa_cfg();
    if (cfg.mode == llama_lsa_config::OFF || !cfg.idx_cache) {
        return;
    }
    static std::atomic<uint32_t> n_warn{0};
    const uint32_t k = n_warn.fetch_add(1);
    if (k == 0) {
        LLAMA_LOG_WARN("%s: LSA stays OFF (dense attention) for this restored conversation: %s (index %u/%u layers); "
                "it turns on again when a prompt is processed from position 0. Further cases are not logged.\n",
                __func__, why, n_ok, n);
    }
}

void llama_lsa_idx_state_read(llama_io_read_i & io) {
    g_lsa_graph_gen.fetch_add(1, std::memory_order_relaxed);
    uint32_t magic = 0, version = 0, n = 0;
    int64_t  n_rows = 0;
    io.read(&magic, sizeof(magic));
    if (magic != LSA_IDX_STATE_MAGIC) {
        // not ours (e.g. a caller passed a longer buffer): leave the rest unread, as before this suffix existed
        LLAMA_LOG_WARN("%s: unknown bytes behind the recurrent state, ignored (LSA index cache not restored)\n", __func__);
        llama_lsa_warn_restore_dense("unknown bytes instead of the LSA index suffix", 0, 0);
        return;
    }
    io.read(&version, sizeof(version));
    if (version != 1 && version != LSA_IDX_STATE_VERSION && version != LSA_IDX_STATE_VERSION_MEANS &&
            version != LSA_IDX_STATE_VERSION_MAP) {
        throw std::runtime_error("unsupported LSA index cache state version");
    }
    const bool on = llama_lsa_cfg().mode != llama_lsa_config::OFF && llama_lsa_cfg().idx_cache;
    uint32_t flags = version == LSA_IDX_STATE_VERSION_MEANS ? 1 : 0;
    if (version == LSA_IDX_STATE_VERSION_MAP) {
        std::array<uint8_t, SHA256_DIGEST_SIZE> identity;
        io.read(&flags, sizeof(flags));
        io.read(identity.data(), identity.size());
        if (flags & ~1u) {
            throw std::runtime_error("corrupt LSA index cache state flags");
        }
        if (on && identity != lsa_map_identity()) {
            throw std::runtime_error("LSA prompt-cache map SHA256 mismatch; rebuild the prompt cache with the current map");
        }
    } else if (on) {
        throw std::runtime_error("LSA prompt cache has no map identity; rebuild the prompt cache with the current map");
    }
    io.read(&n, sizeof(n));
    io.read(&n_rows, sizeof(n_rows));
    if (n_rows <= 0 || n > 4096) {
        throw std::runtime_error("corrupt LSA index cache state header");
    }
    uint32_t n_ok = 0;
    std::vector<uint8_t> skip;
    for (uint32_t i = 0; i < n; ++i) {
        std::string name;
        int64_t d_idx = 0, n_head = 1;
        int32_t type  = GGML_TYPE_F16;
        io.read_string(name);
        io.read(&d_idx, sizeof(d_idx));
        if (version >= 2) {
            io.read(&n_head, sizeof(n_head));
            io.read(&type,   sizeof(type));
        }
        if (d_idx <= 0 || d_idx > 4096 || n_head <= 0 || n_head > 64 ||
                (type != GGML_TYPE_F16 && type != GGML_TYPE_Q8_0) || d_idx % ggml_blck_size((ggml_type) type) != 0) {
            throw std::runtime_error("corrupt LSA index cache state layer");
        }
        const size_t row_bytes = ggml_row_size((ggml_type) type, d_idx);
        const size_t nbytes    = (size_t) n_rows*row_bytes; // per head
        ggml_tensor * t = nullptr;
        if (on) {
            auto & s = lsa_idx();
            std::lock_guard<std::mutex> lock(s.mtx);
            // newest-touched entry of this name = the cache whose K restore just ran (see lsa_means_state_read)
            lsa_idx_entry * best = nullptr;
            for (auto & it : s.e) {
                auto & en = it.second;
                if (en.t && en.name == name && en.t->ne[0] == d_idx && en.t->ne[1] >= n_rows && en.t->ne[2] == n_head &&
                        en.t->type == (ggml_type) type && (!best || en.touch > best->touch)) {
                    best = &en;
                }
            }
            if (best) {
                t = best->t;
                best->built = best->gen; // the K restore already bumped this layer's generation; it matches again
            }
        }
        if (t) {
            for (int64_t h = 0; h < n_head; ++h) {
                io.read_tensor(t, (size_t) h*t->nb[2], nbytes);
            }
            lsa_sum_from_host(t, n_rows);
            ++n_ok;
        } else {
            // unknown layer (other process, LSA off): skip the bytes, the layer stays dense. Host/file readers hold
            // tensor bytes inline; a device reader only restores states this process wrote, which always match.
            for (size_t done = 0; done < nbytes*n_head; ) {
                const size_t c = std::min(nbytes*n_head - done, (size_t) 16 << 20);
                skip.resize(c);
                io.read(skip.data(), c);
                done += c;
            }
        }
    }
    LLAMA_LOG_INFO("%s: LSA index cache restored for %u/%u layers (%lld rows)\n", __func__, n_ok, n, (long long) n_rows);
    if (on && n_ok < n) {
        llama_lsa_warn_restore_dense("saved index layers did not match this context", n_ok, n);
    }
    if (llama_lsa_idx_trace()) {
        LLAMA_LOG_WARN("%s: [lsa-trace] restore: index %u/%u layers (%lld rows), version %u\n", __func__, n_ok, n,
                (long long) n_rows, version);
    }
    if (flags & 1) {
        lsa_means_state_read(io, n_rows);
    }
}

void llama_lsa_set_q_cell(ggml_tensor * q_cell, const llama_ubatch * ubatch) {
    if (!q_cell || !q_cell->buffer) {
        return;
    }
    GGML_ASSERT(q_cell->type == GGML_TYPE_I64 && q_cell->ne[0] == (int64_t) ubatch->n_tokens);
    std::vector<int64_t> rows(ubatch->n_tokens);
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        rows[i] = ubatch->pos[i];
    }
    ggml_backend_tensor_set(q_cell, rows.data(), 0, rows.size()*sizeof(int64_t));
}

void llama_lsa_set_q_blk(ggml_tensor * q_blk, const ggml_tensor * rows_i64, int64_t n_blocks) {
    if (!q_blk || !q_blk->buffer || !rows_i64 || !rows_i64->buffer) {
        return;
    }
    const auto & cfg = llama_lsa_cfg();
    GGML_ASSERT(q_blk->type == GGML_TYPE_I32 && rows_i64->type == GGML_TYPE_I64 && q_blk->ne[0] == rows_i64->ne[0]);
    GGML_ASSERT(n_blocks > 0);
    const int64_t B = cfg.block > 1 ? cfg.block : 128; // LLAMA_LSA_COMP=2 without a block mode uses 128-cell blocks
    const int64_t n = q_blk->ne[0];
    const int64_t last = n_blocks - 1;
    std::vector<int64_t> rows(n);
    ggml_backend_tensor_get(rows_i64, rows.data(), 0, n*sizeof(int64_t));
    std::vector<int32_t> blk(n);
    for (int64_t i = 0; i < n; ++i) {
        blk[i] = (int32_t) std::min<int64_t>(std::max<int64_t>(rows[i], 0)/B, last);
    }
    ggml_backend_tensor_set(q_blk, blk.data(), 0, n*sizeof(int32_t));
}

void llama_lsa_set_mean_inputs(ggml_tensor * u_blk, ggml_tensor * u_rblk, ggml_tensor * q_ring, const ggml_tensor * rows_i64,
        int64_t n_blocks, int64_t n_ring, bool ring_rows) {
    if (!u_blk || !u_blk->buffer || !rows_i64 || !rows_i64->buffer || n_blocks <= 0) {
        return;
    }
    const auto & cfg = llama_lsa_cfg();
    GGML_ASSERT(u_blk->type == GGML_TYPE_I32 && rows_i64->type == GGML_TYPE_I64);
    const int64_t B = cfg.block > 1 ? cfg.block : 128;
    const int64_t n = rows_i64->ne[0];
    std::vector<int64_t> rows(n);
    ggml_backend_tensor_get(rows_i64, rows.data(), 0, n*sizeof(int64_t));
    std::vector<int64_t> blks;
    for (int64_t r : rows) {
        const int64_t b = std::max<int64_t>(r, 0)/B;
        if (std::find(blks.begin(), blks.end(), b) == blks.end()) {
            blks.push_back(b);
        }
    }
    const int64_t n_u = u_blk->ne[0];
    GGML_ASSERT(!blks.empty());
    if ((int64_t) blks.size() > n_u) {
        // non-consecutive cells (plain cache with holes): refresh what fits, and let the next graph rebuild every block
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("%s: ubatch touches %zu blocks (> %lld); LSA block means rebuilt by the next graph\n", __func__,
                    blks.size(), (long long) n_u);
        }
        blks.resize(n_u);
        auto & s = lsa_idx();
        std::lock_guard<std::mutex> lock(s.mtx);
        for (auto & en : lsa_means()) {
            en.second.built = UINT64_MAX;
        }
    }
    if (n_ring > 0) {
        // the refresh re-reads rows [block start of p0, p0) from the staging ring: after a rewind further back than
        // the ring reaches (server prefix reuse, seq_rm) or a gap, they are gone and that block's mean would be wrong
        // for good, so the means go invalid (compensation off until a restart at position 0 or a restore)
        const int64_t p0 = std::max<int64_t>(*std::min_element(rows.begin(), rows.end()), 0);
        const int64_t p1 = *std::max_element(rows.begin(), rows.end()) + 1;
        auto & s = lsa_idx();
        std::lock_guard<std::mutex> lock(s.mtx);
        // a rewind to a block start needs no ring row: the blocks below p0 keep their means (their rows did not
        // change) and every block from p0 on is gathered from rows decoded from here on. This is the server's
        // rollback to a context checkpoint (multi-turn prompt reuse), which sits at a batch start.
        const int64_t lo  = std::max<int64_t>(s.ring_hi - n_ring, s.ring_lo);
        const bool    far = (p0/B)*B < lo;
        if (p0 == 0) {
            s.ring_lo = 0;
        } else if (far && p0 % B == 0 && p0 <= s.ring_hi) {
            s.ring_lo = p0; // ring rows for positions below p0 are stale from now on
        } else if (far || p0 > s.ring_hi) {
            bool was_on = false;
            for (auto & en : lsa_means()) {
                was_on = was_on || en.second.built != UINT64_MAX;
                en.second.built = UINT64_MAX;
            }
            if (was_on) { // once per loss, not once per process: a later sequence that loses it says so too
                LLAMA_LOG_WARN("%s: ubatch at position %lld is outside the LSA staging ring [%lld, %lld) and not at a "
                        "block start; compensation off until a sequence restarts at position 0\n", __func__, (long long) p0,
                        (long long) std::max<int64_t>(lo, 0), (long long) s.ring_hi);
            }
        }
        s.ring_hi = p1;
    }
    if (ring_rows && n_ring > 0) {
        // carry ring: the graph gathers each block row by row from [carry ring (n_ring rows) ; this ubatch's rows]
        std::vector<int32_t> ub(n_u), rr(n_u*B);
        std::unordered_map<int64_t, int64_t> at; // position -> ubatch row
        at.reserve(2*n);
        for (int64_t i = 0; i < n; ++i) {
            at[std::max<int64_t>(rows[i], 0)] = i;
        }
        for (int64_t i = 0; i < n_u; ++i) {
            const int64_t b = blks[std::min<int64_t>(i, (int64_t) blks.size() - 1)];
            ub[i] = (int32_t) std::min<int64_t>(b, n_blocks - 1);
            for (int64_t j = 0; j < B; ++j) {
                // rows before the ubatch come from the carry ring; rows past it are unwritten (stale, inside the window)
                const int64_t r  = b*B + j;
                const auto    it = at.find(r);
                rr[i*B + j] = (int32_t) (it != at.end() ? n_ring + it->second : r % n_ring);
            }
        }
        ggml_backend_tensor_set(u_blk, ub.data(), 0, n_u*sizeof(int32_t));
        if (u_rblk && u_rblk->buffer) {
            GGML_ASSERT(u_rblk->type == GGML_TYPE_I32 && u_rblk->ne[0] == n_u*B);
            ggml_backend_tensor_set(u_rblk, rr.data(), 0, rr.size()*sizeof(int32_t));
        }
        if (q_ring && q_ring->buffer) {
            const int64_t m = q_ring->ne[0];
            GGML_ASSERT(q_ring->type == GGML_TYPE_I64 && m == std::min<int64_t>(n, n_ring));
            std::vector<int64_t> qr(m);
            for (int64_t k = 0; k < m; ++k) {
                qr[k] = std::max<int64_t>(rows[n - m + k], 0) % n_ring;
            }
            ggml_backend_tensor_set(q_ring, qr.data(), 0, m*sizeof(int64_t));
        }
        return;
    }
    std::vector<int32_t> ub(n_u), ur(n_u);
    const int64_t n_rb = n_ring > 0 ? n_ring/B : 1;
    for (int64_t i = 0; i < n_u; ++i) {
        const int64_t b = blks[std::min<int64_t>(i, (int64_t) blks.size() - 1)];
        ub[i] = (int32_t) std::min<int64_t>(b, n_blocks - 1);
        ur[i] = (int32_t) (b % n_rb);
    }
    ggml_backend_tensor_set(u_blk, ub.data(), 0, n_u*sizeof(int32_t));
    if (u_rblk && u_rblk->buffer && n_ring > 0) {
        GGML_ASSERT(u_rblk->type == GGML_TYPE_I32 && u_rblk->ne[0] == n_u);
        ggml_backend_tensor_set(u_rblk, ur.data(), 0, n_u*sizeof(int32_t));
    }
    if (q_ring && q_ring->buffer && n_ring > 0) {
        GGML_ASSERT(q_ring->type == GGML_TYPE_I64 && q_ring->ne[0] == n);
        std::vector<int64_t> qr(n);
        for (int64_t i = 0; i < n; ++i) {
            qr[i] = std::max<int64_t>(rows[i], 0) % n_ring;
        }
        ggml_backend_tensor_set(q_ring, qr.data(), 0, n*sizeof(int64_t));
    }
}
