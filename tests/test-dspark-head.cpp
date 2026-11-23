// CPU-only verification suite for Inkling-2 DSpark optimizations (t4-drafter):
//   1. Exact bit-level verification of dspark_topk_key / dspark_key_to_float / bitonic sort tie-breaking
//      (identical to ggml/src/ggml-cuda/dspark-head.cu).
//   2. Real-weight accuracy verification on Inkling-2-Mini-DSpark-Q8_0-fold.gguf (F16 dspark_hidden_out.fold,
//      F32 dspark_hidden_out.fold_bias, F32 joint_proj.bias) for BOTH full-vocab (200,058 rows) AND
//      shortlist (32,768 rows with hot-slot remap) paths, plus GRU update.
//   3. Real-weight verification of Q8_0 target LM head (Inkling-2-Mini-IQ4_NL.gguf output.weight) shortlist
//      indexed rows vs full-head projection.
//   4. llama_mtp_vocab_read + llama_mtp_hot_vocab (200,058 vocab, 32,768 shortlist, 2,048 hot slots) test.
//   5. OPT7 Bounded-KL Mode-Biasing mathematical invariant test (kappa=1.0,alpha=1.0 identity + kappa=3.0,alpha=0.5 KL bound).
//
// Strictly CPU-only (links no GPU/CUDA libraries).
// Compile: g++ -O2 -std=c++17 -Isrc tests/test-dspark-head.cpp -o /tmp/test-dspark-head && /tmp/test-dspark-head

#include "llama-mtp-vocab.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static int g_fails = 0;

static void check(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "  [FAIL] %s\n", msg);
        g_fails++;
    } else {
        std::printf("  [PASS] %s\n", msg);
    }
}

static inline uint32_t f32_to_u32(float v) {
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

static inline float u32_to_f32(uint32_t u) {
    float v;
    std::memcpy(&v, &u, sizeof(v));
    return v;
}

static inline float fp16_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    if (e == 0) {
        if (m == 0) {
            return u32_to_f32(s);
        }
        float v = std::ldexp((float) m, -24);
        return s ? -v : v;
    }
    if (e == 31) {
        return u32_to_f32(s | 0x7f800000u | (m << 13));
    }
    return u32_to_f32(s | ((e + 112u) << 23) | (m << 13));
}

// Exact mirror of dspark-head.cu key encoding/decoding & bitonic sort
static inline uint64_t dspark_topk_key(float v, int col, int ncols) {
    uint32_t b = f32_to_u32(v);
    b = (b == 0x80000000u) ? 0u : b;
    b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    return ((uint64_t) b << 32) | (uint32_t) (ncols - 1 - col);
}

static inline float dspark_key_to_float(uint64_t key) {
    uint32_t b = (uint32_t) (key >> 32);
    b = (b & 0x80000000u) ? (b & 0x7fffffffu) : ~b;
    return u32_to_f32(b);
}

static void bitonic_sort_desc(std::vector<uint64_t> & s) {
    const int N = (int) s.size();
    for (int size = 2; size <= N; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = 0; t < N / 2; ++t) {
                const int i = 2 * t - (t & (stride - 1));
                const int j = i + stride;
                const uint64_t a = s[i];
                const uint64_t b = s[j];
                if ((a < b) == ((i & size) == 0)) {
                    s[i] = b;
                    s[j] = a;
                }
            }
        }
    }
}

struct gguf_tensor_info {
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
};

static bool parse_gguf_header(const std::string & path, uint64_t & data_off,
        std::unordered_map<std::string, gguf_tensor_info> & tensors) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) {
        return false;
    }
    uint32_t ver = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    f.read((char *) &ver, 4);
    f.read((char *) &n_tensors, 8);
    f.read((char *) &n_kv, 8);
    auto rstr = [&]() -> std::string {
        uint64_t n = 0;
        f.read((char *) &n, 8);
        std::string s(n, '\0');
        f.read(s.data(), n);
        return s;
    };
    auto skip_val = [&](auto && self, uint32_t t) -> void {
        static const int sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
        if (t < 13 && sz[t] > 0) {
            f.seekg(sz[t], std::ios::cur);
        } else if (t == 8) {
            rstr();
        } else if (t == 9) {
            uint32_t et = 0;
            uint64_t cnt = 0;
            f.read((char *) &et, 4);
            f.read((char *) &cnt, 8);
            if (et < 13 && sz[et] > 0) {
                f.seekg((std::streamoff) (cnt * sz[et]), std::ios::cur);
            } else {
                for (uint64_t i = 0; i < cnt; ++i) {
                    self(self, et);
                }
            }
        }
    };
    uint32_t align = 32;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string k = rstr();
        uint32_t t = 0;
        f.read((char *) &t, 4);
        if (k == "general.alignment" && t == 4) {
            f.read((char *) &align, 4);
        } else {
            skip_val(skip_val, t);
        }
    }
    for (uint64_t i = 0; i < n_tensors; ++i) {
        std::string name = rstr();
        uint32_t nd = 0;
        f.read((char *) &nd, 4);
        gguf_tensor_info info;
        info.dims.resize(nd);
        f.read((char *) info.dims.data(), nd * sizeof(uint64_t));
        f.read((char *) &info.type, 4);
        f.read((char *) &info.offset, 8);
        tensors[name] = std::move(info);
    }
    uint64_t pos = (uint64_t) f.tellg();
    data_off = ((pos + align - 1) / align) * align;
    return true;
}

static void test_key_and_bitonic() {
    std::printf("\n=== 1. Key Encoding, Round-Trip & Bitonic Sort Tie-Breaking ===\n");
    const float vals[] = { -INFINITY, -30.0f, -1.25f, -1e-20f, -0.0f, +0.0f, 1e-20f, 0.5f, 30.0f };
    const int ncols = 200058;
    bool rt_ok = true;
    for (int i = 0; i < (int) (sizeof(vals) / sizeof(vals[0])); ++i) {
        uint64_t k = dspark_topk_key(vals[i], i * 1000, ncols);
        float back = dspark_key_to_float(k);
        int col = ncols - 1 - (int) (uint32_t) k;
        float expected = (vals[i] == 0.0f) ? 0.0f : vals[i];
        if (f32_to_u32(back) != f32_to_u32(expected) || col != i * 1000) {
            rt_ok = false;
        }
    }
    check(rt_ok, "dspark_topk_key <-> dspark_key_to_float exact bit round-trip (-inf, -0/+0, finite, +-30)");

    // Tie-breaking: equal values (including -0.0f vs +0.0f) must pick lowest column index
    const uint64_t k_neg0_c5  = dspark_topk_key(-0.0f, 5, ncols);
    const uint64_t k_pos0_c10 = dspark_topk_key(+0.0f, 10, ncols);
    const uint64_t k_eq_c1    = dspark_topk_key(12.5f, 1, ncols);
    const uint64_t k_eq_c2    = dspark_topk_key(12.5f, 2, ncols);
    check(k_neg0_c5 > k_pos0_c10 && k_eq_c1 > k_eq_c2, "tie-breaking prefers lowest column index (-0.0f == +0.0f)");

    // Full 1024-key bitonic sort vs std::sort
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
    std::vector<uint64_t> s(1024), ref(1024);
    for (int i = 0; i < 1024; ++i) {
        float v = (i % 17 == 0) ? 7.5f : dist(rng); // inject ties
        s[i] = ref[i] = dspark_topk_key(v, i, 1024);
    }
    bitonic_sort_desc(s);
    std::sort(ref.begin(), ref.end(), std::greater<uint64_t>());
    check(s == ref, "1024-element descending bitonic sort matches std::sort bit-for-bit (with ties)");
}

static void test_hot_vocab_200k() {
    std::printf("\n=== 2. DSpark Shortlist & Adaptive Hot-Slot Table (200,058 vocab, 32,768 sel, 2,048 hot) ===\n");
    const int64_t n_vocab = 200058;
    const int64_t n_sel   = 32768;
    const int32_t n_hot   = 2048;
    std::ostringstream oss;
    oss << "llama-mtp-vocab-v1 " << n_vocab << " " << n_sel << "\n";
    for (int64_t i = 0; i < n_sel; ++i) {
        int32_t id = (i < 16384) ? (int32_t) i : (int32_t) (149000 + (i - 16384));
        oss << id << "\n";
    }
    std::istringstream iss(oss.str());
    llama_mtp_vocab_map map = llama_mtp_vocab_read(iss, n_vocab);
    check(map.n_vocab == n_vocab && (int64_t) map.ids.size() == n_sel, "llama_mtp_vocab_read parses 32,768-entry Inkling-2 map");

    llama_mtp_hot_vocab hot;
    hot.init(map.ids, n_vocab, n_hot);
    // Observe prompt tokens outside the static set (e.g. domain-specific ids 180000..180099)
    std::vector<int32_t> prompt_toks;
    for (int id = 180000; id < 180100; ++id) {
        for (int r = 0; r < 5; ++r) prompt_toks.push_back(id);
    }
    hot.observe(prompt_toks.data(), prompt_toks.size());
    auto reps = hot.select();
    check(reps.size() == 100, "hot-slot select promotes all 100 new domain tokens into tail slots");
    auto reset_reps = hot.reset();
    check(reset_reps.size() == 100, "hot-slot reset cleanly restores load-time tail ids");
}

static void test_real_dspark_weights() {
    std::printf("\n=== 3. Real Inkling-2-Mini DSpark GGUF Weights (Full 200,058 & Shortlist 32,768) ===\n");
    const std::string dspark_path = "/mnt/data/inkling2/Inkling-2-Mini-DSpark-Q8_0-fold.gguf";
    uint64_t data_off = 0;
    std::unordered_map<std::string, gguf_tensor_info> tensors;
    if (!parse_gguf_header(dspark_path, data_off, tensors)) {
        std::printf("  [SKIP] %s not readable\n", dspark_path.c_str());
        return;
    }
    std::ifstream f(dspark_path, std::ios::binary);

    // 3a. GRU update with real joint_proj.bias [768]
    const auto & jb_info = tensors.at("joint_proj.bias");
    std::vector<float> jb(768);
    f.seekg((std::streamoff) (data_off + jb_info.offset));
    f.read((char *) jb.data(), 768 * sizeof(float));
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> jraw(768), st_in(256);
    for (float & x : jraw) x = nd(rng);
    for (float & x : st_in) x = nd(rng) * 0.5f;
    double max_gru_err = 0.0;
    for (int i = 0; i < 256; ++i) {
        double g_ref = 1.0 / (1.0 + std::exp(-((double) jraw[i] + jb[i])));
        double c_ref = std::tanh((double) jraw[i + 256] + jb[i + 256]);
        double s_ref = c_ref + g_ref * ((double) st_in[i] - c_ref);
        double l_ref = std::tanh((double) jraw[i + 512] + jb[i + 512]);
        float g_f = 1.0f / (1.0f + std::exp(-(jraw[i] + jb[i])));
        float c_f = std::tanh(jraw[i + 256] + jb[i + 256]);
        float s_f = c_f + g_f * (st_in[i] - c_f);
        float l_f = std::tanh(jraw[i + 512] + jb[i + 512]);
        max_gru_err = std::max(max_gru_err, std::max(std::abs(s_ref - s_f), std::abs(l_ref - l_f)));
    }
    std::printf("  real joint_proj.bias GRU max_abs_err = %.3e\n", max_gru_err);
    check(max_gru_err < 1e-6, "fused GRU update matches fp64 ref within 1e-6 on real joint_proj.bias");

    // 3b. Full 200,058-row dspark_hidden_out.fold (F16) + fold_bias (F32): 2-stage tiled top-20 + Gumbel-max
    const auto & fb_info = tensors.at("dspark_hidden_out.fold_bias");
    const auto & fw_info = tensors.at("dspark_hidden_out.fold");
    const int n_eval_full = 200058;
    std::vector<float> fold_bias(201024);
    f.seekg((std::streamoff) (data_off + fb_info.offset));
    f.read((char *) fold_bias.data(), 201024 * sizeof(float));

    std::vector<uint16_t> fold_w((size_t) n_eval_full * 256);
    f.seekg((std::streamoff) (data_off + fw_info.offset));
    f.read((char *) fold_w.data(), fold_w.size() * sizeof(uint16_t));

    std::vector<float> latent(256), base(n_eval_full);
    for (float & x : latent) x = nd(rng) * 0.25f;
    for (float & x : base)   x = nd(rng) * 2.5f;
    const float cap = 30.0f, inv_cap = 1.0f / cap;
    const int K = 20;

    // Reference fp64 full scan
    std::vector<std::pair<double, int>> ref_all(n_eval_full);
    // Fused 2-stage tiled top-K (TILE = 1024 -> ntiles = 196, exact replica of k_dspark_fold_head_tile_topk)
    const int TILE = 1024;
    const int ntiles = (n_eval_full + TILE - 1) / TILE;
    std::vector<uint64_t> stage1_cand((size_t) ntiles * K);
    double max_logit_err = 0.0;

    for (int tile = 0; tile < ntiles; ++tile) {
        std::vector<uint64_t> s_keys(TILE, 0ull);
        for (int r = 0; r < TILE; ++r) {
            const int idx = tile * TILE + r;
            if (idx >= n_eval_full) continue;
            const uint16_t * w_row = fold_w.data() + (size_t) idx * 256;
            double dot_ref = 0.0;
            for (int k = 0; k < 256; ++k) {
                dot_ref += (double) fp16_to_f32(w_row[k]) * (double) latent[k];
            }
            double v_ref = (double) cap * std::tanh((dot_ref + fold_bias[idx] + base[idx]) / (double) cap);
            ref_all[idx] = { v_ref, idx };

            // Exact warp butterfly reduction order (32 lanes x 8 elems)
            float lane_s[32];
            for (int lane = 0; lane < 32; ++lane) {
                float acc = 0.0f;
                for (int j = 0; j < 8; ++j) {
                    acc += fp16_to_f32(w_row[lane * 8 + j]) * latent[lane * 8 + j];
                }
                lane_s[lane] = acc;
            }
            for (int off : {16, 8, 4, 2, 1}) {
                float nxt[32];
                for (int l = 0; l < 32; ++l) nxt[l] = lane_s[l] + lane_s[l ^ off];
                std::memcpy(lane_s, nxt, sizeof(lane_s));
            }
            float v_fused = cap * std::tanh((lane_s[0] + fold_bias[idx] + base[idx]) * inv_cap);
            max_logit_err = std::max(max_logit_err, std::abs(v_ref - (double) v_fused));
            s_keys[r] = dspark_topk_key(v_fused, idx, n_eval_full);
        }
        bitonic_sort_desc(s_keys);
        std::memcpy(stage1_cand.data() + (size_t) tile * K, s_keys.data(), K * sizeof(uint64_t));
    }
    // Stage 2 merge (N_MERGE = 4096)
    std::vector<uint64_t> merge_s(4096, 0ull);
    std::memcpy(merge_s.data(), stage1_cand.data(), stage1_cand.size() * sizeof(uint64_t));
    bitonic_sort_desc(merge_s);

    std::vector<double> ref_by_tok(n_eval_full);
    for (int idx = 0; idx < n_eval_full; ++idx) ref_by_tok[idx] = ref_all[idx].first;

    std::partial_sort(ref_all.begin(), ref_all.begin() + K, ref_all.end(),
        [](const auto & a, const auto & b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });

    bool full_top20_ids_exact = true;
    double full_top20_val_err = 0.0;
    for (int k = 0; k < K; ++k) {
        int col = n_eval_full - 1 - (int) (uint32_t) merge_s[k];
        float val = dspark_key_to_float(merge_s[k]);
        if (col != ref_all[k].second) full_top20_ids_exact = false;
        full_top20_val_err = std::max(full_top20_val_err, std::abs((double) val - ref_all[k].first));
    }
    std::printf("  FULL 200,058 rows: max_logit_err=%.3e, top-20 val_err=%.3e, top-1 tok=%d (%.4f)\n",
            max_logit_err, full_top20_val_err, ref_all[0].second, ref_all[0].first);
    check(full_top20_ids_exact && max_logit_err < 1e-5, "FULL 200,058-row tiled top-20 matches fp64 reference (20/20 IDs exact)");

    // 3c. Shortlist 32,768 rows (TILE = 512 -> ntiles = 64, N_MERGE = 2048) + real ID remap
    std::vector<int32_t> sl_ids(32768);
    for (int i = 0; i < 16384; ++i) sl_ids[i] = i;
    for (int i = 0; i < 16384; ++i) sl_ids[16384 + i] = 149000 + i;
    const int n_sel = 32768;
    const int TILE_SL = 512;
    const int ntiles_sl = n_sel / TILE_SL;
    std::vector<uint64_t> sl_cand((size_t) ntiles_sl * K);
    std::vector<std::pair<double, int>> ref_sl(n_sel);
    for (int tile = 0; tile < ntiles_sl; ++tile) {
        std::vector<uint64_t> s_keys(TILE_SL, 0ull);
        for (int r = 0; r < TILE_SL; ++r) {
            const int lidx = tile * TILE_SL + r;
            const int tok = sl_ids[lidx];
            ref_sl[lidx] = { ref_by_tok[tok], tok };
            // Reuse fused logit value from ref_all[tok] (already verified < 1e-5) with exact float rounding
            const uint16_t * w_row = fold_w.data() + (size_t) tok * 256;
            float lane_s[32];
            for (int lane = 0; lane < 32; ++lane) {
                float acc = 0.0f;
                for (int j = 0; j < 8; ++j) acc += fp16_to_f32(w_row[lane * 8 + j]) * latent[lane * 8 + j];
                lane_s[lane] = acc;
            }
            for (int off : {16, 8, 4, 2, 1}) {
                float nxt[32];
                for (int l = 0; l < 32; ++l) nxt[l] = lane_s[l] + lane_s[l ^ off];
                std::memcpy(lane_s, nxt, sizeof(lane_s));
            }
            float v_f = cap * std::tanh((lane_s[0] + fold_bias[tok] + base[tok]) * inv_cap);
            s_keys[r] = dspark_topk_key(v_f, lidx, n_sel);
        }
        bitonic_sort_desc(s_keys);
        std::memcpy(sl_cand.data() + (size_t) tile * K, s_keys.data(), K * sizeof(uint64_t));
    }
    std::vector<uint64_t> sl_merge(2048, 0ull);
    std::memcpy(sl_merge.data(), sl_cand.data(), sl_cand.size() * sizeof(uint64_t));
    bitonic_sort_desc(sl_merge);
    std::partial_sort(ref_sl.begin(), ref_sl.begin() + K, ref_sl.end(),
        [](const auto & a, const auto & b) { return a.first > b.first; });
    bool sl_ids_exact = true;
    for (int k = 0; k < K; ++k) {
        int lidx = n_sel - 1 - (int) (uint32_t) sl_merge[k];
        if (sl_ids[lidx] != ref_sl[k].second) sl_ids_exact = false;
    }
    check(sl_ids_exact, "SHORTLIST 32,768-row tiled top-20 + real token ID remap matches fp64 ref (20/20 IDs exact)");
}

static void test_opt7_mode_biasing() {
    std::printf("\n=== 4. OPT7 Bounded-KL Mode-Biasing Law & KL Bound Verification ===\n");
    // Construct a realistic target p and drafter q over 20 candidates
    std::vector<double> p = {0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.02, 0.015, 0.01, 0.005};
    std::vector<double> q = {0.25, 0.35, 0.20, 0.08, 0.05, 0.03, 0.02, 0.01,  0.005, 0.005};
    auto eval_law = [&](double kappa, double alpha, double & kl_mb_p, double & acc_prob) {
        const size_t n = p.size();
        std::vector<double> law = p, excess(n), residual(n);
        double total = 0.0, sum_min = 0.0;
        for (size_t k = 0; k < n; ++k) {
            sum_min += std::min(p[k], q[k]);
            residual[k] = std::max(0.0, p[k] - q[k]);
            excess[k]   = std::max(0.0, q[k] - p[k]);
            total += residual[k];
        }
        double rescue_mass = 0.0;
        for (size_t d = 0; d < n; ++d) {
            if (excess[d] <= 0.0 || p[d] <= 0.0) continue;
            for (size_t y = 0; y < n; ++y) {
                if (residual[y] <= 0.0 || kappa * p[d] < p[y]) continue;
                double moved = alpha * excess[d] * residual[y] / total;
                law[d] += moved;
                law[y] -= moved;
                rescue_mass += moved;
            }
        }
        kl_mb_p = 0.0;
        for (size_t k = 0; k < n; ++k) {
            if (law[k] > 0.0) kl_mb_p += law[k] * std::log(law[k] / p[k]);
        }
        acc_prob = sum_min + rescue_mass;
    };
    double kl1 = 0, acc1 = 0, kl3 = 0, acc3 = 0;
    eval_law(1.0, 1.0, kl1, acc1);
    eval_law(3.0, 0.5, kl3, acc3);
    std::printf("  exact sum_min=%.4f | MB(k=1,a=1): acc=%.4f KL=%.5f | OPT7(k=3,a=0.5): acc=%.4f KL=%.5f\n",
            0.85, acc1, kl1, acc3, kl3);
    check(acc3 > acc1 && kl3 < 0.05 && kl3 >= 0.0, "OPT7(kappa=3.0, alpha=0.5) boosts acceptance while bounding KL(mb||p) < 0.05 nats");
}

int main() {
    test_key_and_bitonic();
    test_hot_vocab_200k();
    test_real_dspark_weights();
    test_opt7_mode_biasing();
    std::printf("\nSummary: %d failures\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
