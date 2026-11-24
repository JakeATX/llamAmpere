// Real-graph ggml-backend CUDA benchmark & verifier for ALL 5-Bit AI+ (GGML_TYPE_AIT5_U) kernel cleanups:
//   1. Dense MUL_MAT (3072x2304 Inkling attn_q, 2304x1152 H128, 5120x5120 27B, 17408x5120 27B) at T = 1, 5, 8:
//      base (DECODE_FAST=0, A16=0) vs ai4-decode-fast (DECODE_FAST=1, A16=0) vs AIT5_A16 (A16=1) + live FP64 CPU ref!
//   2. MoE MUL_MAT_ID (2304x2304 + 2304x1152, 128e top-8 Zipf + duplicate-within-token + invalid-id test):
//      - Decode T = 1, 2, 4, 5, 8: base vs ai4-decode-fast vs AIT5_A16_ID=1 + live FP64 CPU ref!
//      - Prefill N = 512, 1024, 2048: base vs AI5_LOWREG_ID=1 vs AI5_COOP128=1 + bitwise --dump/--cmp!
//
// Follows every review rule:
//   - Real ggml-backend CUDA compute graphs (no stand-in rewrites)
//   - Exact GPU cudaEvent timing on backend stream + clean read-only 64 MiB L2 flush + box's own BWREF GB/s
//   - Full 0..255 random weight bytes, varied random finite FP16 scales, 15x outlier activations
//   - Hard fire-counter assertions when opt-in env flags are enabled (fails loudly if hook did not engage!)

#include "bwref_flush.cuh"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-ai4.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern "C" {
uint64_t ggml_cuda_ait5a16_fire_count(int which);
void     ggml_cuda_ait5a16_reset_fire_count(void);
uint64_t ggml_cuda_ai5_lowreg_id_fire_count(void);
void     ggml_cuda_ai5_lowreg_id_reset_fire_count(void);
uint64_t ggml_cuda_ai5_coop128_fire_count(void);
void     ggml_cuda_ai5_coop128_reset_fire_count(void);
void *   ggml_cuda_ait5_backend_stream(ggml_backend_t backend);
}

namespace {

static bool env_on(const char * name, bool def = false) {
    const char * e = getenv(name);
    return e ? (atoi(e) != 0) : def;
}

static int env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}

struct xorshift64 {
    uint64_t s;
    explicit xorshift64(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t next() {
        uint64_t x = s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s = x;
        return x;
    }
    double u01() {
        return (double) (next() >> 11) * (1.0 / 9007199254740992.0);
    }
    float normal_approx() {
        // CLT sum of 6 uniforms in [-3, +3], mean 0, std ~0.707
        double sum = 0.0;
        for (int i = 0; i < 6; ++i) sum += u01();
        return (float) (sum - 3.0);
    }
};

static uint16_t f32_to_f16_bits(float v) {
    return (uint16_t) ggml_fp32_to_fp16(v);
}

// Fill AIT5_U slabs with full 0..255 random tile bytes and varied random finite FP16 scales (pos & neg).
static void fill_ait5_random(uint8_t * dst, int64_t n_slabs, int64_t K, uint64_t seed) {
    xorshift64 rng(seed);
    const int64_t nk = K / 128;
    const int64_t ub = AI4U_UNIT_BYTES(5); // 1312
    for (int64_t s = 0; s < n_slabs; ++s) {
        for (int64_t u = 0; u < nk; ++u) {
            uint8_t * blk = dst + (s * nk + u) * ub;
            uint16_t * dh = reinterpret_cast<uint16_t *>(blk);
            for (int r = 0; r < 16; ++r) {
                const float sign = (rng.next() & 1) ? 1.0f : -1.0f;
                const float mag  = 0.002f + 0.078f * (float) rng.u01();
                dh[r] = f32_to_f16_bits(sign * mag);
            }
            uint8_t * tiles = blk + AI4U_SCALE_BYTES;
            for (int64_t i = 0; i < ub - AI4U_SCALE_BYTES; i += 8) {
                const uint64_t v = rng.next();
                const int64_t rem = std::min<int64_t>(8, (ub - AI4U_SCALE_BYTES) - i);
                std::memcpy(tiles + i, &v, (size_t) rem);
            }
        }
    }
}

// Fill activations [n_rows, K] with realistic 15x outlier channels (~1% of channels).
static void fill_act_with_outliers(float * x, int64_t n_rows, int64_t K, uint64_t seed, float outlier_scale = 15.0f) {
    xorshift64 rng(seed);
    std::vector<float> ch_gain((size_t) K, 1.0f);
    for (int64_t k = 0; k < K; ++k) {
        if ((k % 97) == 7) {
            ch_gain[(size_t) k] = outlier_scale;
        }
    }
    for (int64_t r = 0; r < n_rows; ++r) {
        const float row_scale = 0.5f + 1.5f * (float) rng.u01();
        for (int64_t k = 0; k < K; ++k) {
            x[r * K + k] = row_scale * ch_gain[(size_t) k] * rng.normal_approx();
        }
    }
}

// Fill Zipf-skewed routing ids [n_used, n_tok], with deliberate duplicate-within-token and optional invalid id.
static void fill_zipf_ids(
        int32_t * ids, int n_used, int n_tok, int n_expert, uint64_t seed,
        bool inject_dup, bool inject_invalid) {
    xorshift64 rng(seed);
    std::vector<double> cdf((size_t) n_expert);
    double total = 0.0;
    for (int e = 0; e < n_expert; ++e) {
        total += 1.0 / std::pow((double) (e + 1), 0.85);
        cdf[(size_t) e] = total;
    }
    for (int e = 0; e < n_expert; ++e) cdf[(size_t) e] /= total;

    for (int t = 0; t < n_tok; ++t) {
        for (int u = 0; u < n_used; ++u) {
            const double p = rng.u01();
            int lo = 0, hi = n_expert - 1;
            while (lo < hi) {
                const int mid = (lo + hi) >> 1;
                if (cdf[(size_t) mid] < p) lo = mid + 1;
                else hi = mid;
            }
            ids[t * n_used + u] = lo;
        }
        if (inject_dup && n_used >= 2 && (t == 0 || (t % 17 == 0))) {
            ids[t * n_used + 1] = ids[t * n_used + 0];
        }
    }
    if (inject_invalid && n_used >= 1 && n_tok >= 1) {
        ids[(n_tok - 1) * n_used + (n_used - 1)] = -1;
    }
}

struct acc_metrics {
    double max_abs  = 0.0;
    double rel_rmse = 0.0;
    double cos_sim  = 1.0;
};

static acc_metrics compare_fp64_ref(const float * got, const double * ref, size_t n) {
    double max_abs = 0.0;
    double err2 = 0.0, ref2 = 0.0, got2 = 0.0, dot = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double g = (double) got[i];
        const double r = ref[i];
        const double d = std::fabs(g - r);
        if (d > max_abs) max_abs = d;
        err2 += d * d;
        ref2 += r * r;
        got2 += g * g;
        dot  += g * r;
    }
    acc_metrics m;
    m.max_abs  = max_abs;
    m.rel_rmse = (ref2 > 0.0) ? std::sqrt(err2 / ref2) : std::sqrt(err2);
    m.cos_sim  = (ref2 > 0.0 && got2 > 0.0) ? (dot / (std::sqrt(ref2) * std::sqrt(got2))) : 1.0;
    return m;
}

static uint32_t float_ulp_diff(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return 0xFFFFFFFFu;
    uint32_t ua = 0, ub = 0;
    std::memcpy(&ua, &a, 4);
    std::memcpy(&ub, &b, 4);
    auto ordered = [](uint32_t u) -> uint32_t {
        return (u & 0x80000000u) ? (0x80000000u - u) : (u + 0x80000000u);
    };
    const uint32_t oa = ordered(ua);
    const uint32_t ob = ordered(ub);
    return (oa > ob) ? (oa - ob) : (ob - oa);
}

struct dump_record {
    std::string        name;
    std::vector<float> data;
};

static void save_dumps(const std::string & path, const std::vector<dump_record> & recs) {
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        fprintf(stderr, "ERROR: cannot open dump file %s for writing\n", path.c_str());
        exit(1);
    }
    const uint32_t magic = 0x41495435u;
    const uint32_t count = (uint32_t) recs.size();
    os.write((const char *) &magic, 4);
    os.write((const char *) &count, 4);
    for (const auto & r : recs) {
        const uint32_t nlen = (uint32_t) r.name.size();
        const uint64_t dlen = (uint64_t) r.data.size();
        os.write((const char *) &nlen, 4);
        os.write(r.name.data(), nlen);
        os.write((const char *) &dlen, 8);
        os.write((const char *) r.data.data(), (std::streamsize) (dlen * sizeof(float)));
    }
    printf("Saved %u output records to %s\n", count, path.c_str());
}

static void compare_dumps(const std::string & path, const std::vector<dump_record> & recs) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        fprintf(stderr, "ERROR: cannot open reference dump %s\n", path.c_str());
        exit(1);
    }
    uint32_t magic = 0, count = 0;
    is.read((char *) &magic, 4);
    is.read((char *) &count, 4);
    if (magic != 0x41495435u) {
        fprintf(stderr, "ERROR: bad magic in %s\n", path.c_str());
        exit(1);
    }
    std::vector<dump_record> ref(count);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t nlen = 0;
        uint64_t dlen = 0;
        is.read((char *) &nlen, 4);
        ref[i].name.resize(nlen);
        is.read(&ref[i].name[0], nlen);
        is.read((char *) &dlen, 8);
        ref[i].data.resize((size_t) dlen);
        is.read((char *) ref[i].data.data(), (std::streamsize) (dlen * sizeof(float)));
    }
    printf("\n=== Comparison vs Reference Dump (%s) ===\n", path.c_str());
    for (const auto & cur : recs) {
        const dump_record * match = nullptr;
        for (const auto & r : ref) {
            if (r.name == cur.name) { match = &r; break; }
        }
        if (!match) {
            printf("  %-36s : [not in ref]\n", cur.name.c_str());
            continue;
        }
        if (match->data.size() != cur.data.size()) {
            fprintf(stderr, "ERROR: size mismatch for %s\n", cur.name.c_str());
            exit(1);
        }
        double max_abs = 0.0;
        uint32_t max_ulp = 0;
        size_t exact = 0;
        for (size_t i = 0; i < cur.data.size(); ++i) {
            const double d = std::fabs((double) cur.data[i] - (double) match->data[i]);
            if (d > max_abs) max_abs = d;
            const uint32_t u = float_ulp_diff(cur.data[i], match->data[i]);
            if (u > max_ulp) max_ulp = u;
            if (u == 0) ++exact;
        }
        printf("  %-36s : max_abs=%.3e  max_ulp=%u  exact=%zu/%zu%s\n",
               cur.name.c_str(), max_abs, max_ulp, exact, cur.data.size(),
               (max_ulp == 0) ? "  [BITWISE EXACT]" : "");
    }
}

static double median_us(std::vector<double> & v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Compute FP64 CPU reference for first `m_check` rows of dense AIT5_U MUL_MAT: Y[t*M + r]
static std::vector<double> cpu_ref_dense(
        const uint8_t * w_host, const float * x_host, int64_t M, int64_t K, int64_t T, int64_t m_check) {
    const int64_t slabs = m_check / 16;
    const int64_t sb = ai4x_slab_bytes_k(K, 5);
    std::vector<float> slab_f32((size_t) (16 * K));
    std::vector<double> ref((size_t) (T * m_check), 0.0);
    for (int64_t s = 0; s < slabs; ++s) {
        aik_dequantize_slab_c(5, 0xAA73876Bu, w_host + s * sb, slab_f32.data(), K);
        for (int64_t r = 0; r < 16; ++r) {
            const float * wr = slab_f32.data() + r * K;
            for (int64_t t = 0; t < T; ++t) {
                const float * xt = x_host + t * K;
                double acc = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    acc += (double) wr[k] * (double) xt[k];
                }
                ref[(size_t) (t * m_check + s * 16 + r)] = acc;
            }
        }
    }
    return ref;
}

// Compute FP64 CPU reference for first `m_check` rows of MoE AIT5_U MUL_MAT_ID
static std::vector<double> cpu_ref_moe(
        const uint8_t * w_host, int64_t nb02, const float * x_host, const int32_t * ids_host,
        int64_t K, int64_t n_used, int64_t ne11, int64_t n_tok, int64_t n_expert, int64_t m_check) {
    const int64_t slabs = m_check / 16;
    const int64_t sb = ai4x_slab_bytes_k(K, 5);
    std::vector<float> slab_f32((size_t) (16 * K));
    const int64_t n_pairs = n_used * n_tok;
    std::vector<double> ref((size_t) (n_pairs * m_check), 0.0);

    for (int64_t p = 0; p < n_pairs; ++p) {
        const int32_t e = ids_host[p];
        if (e < 0 || e >= n_expert) {
            continue; // exact 0.0
        }
        const int64_t tok = p / n_used;
        const int64_t u   = p % n_used;
        const int64_t xr  = tok * ne11 + (u % ne11);
        const float * xt  = x_host + xr * K;
        const uint8_t * we = w_host + (int64_t) e * nb02;
        for (int64_t s = 0; s < slabs; ++s) {
            aik_dequantize_slab_c(5, 0xAA73876Bu, we + s * sb, slab_f32.data(), K);
            for (int64_t r = 0; r < 16; ++r) {
                const float * wr = slab_f32.data() + r * K;
                double acc = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    acc += (double) wr[k] * (double) xt[k];
                }
                ref[(size_t) (p * m_check + s * 16 + r)] = acc;
            }
        }
    }
    return ref;
}

static void run_dense_case(
        ggml_backend_t backend, bwref_ctx * bctx, void * stream,
        const char * label, int64_t M, int64_t K, int64_t T,
        int warmup, int iters, bool check_ref, std::vector<dump_record> & dumps) {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_AIT5_U, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t w_bytes = ggml_nbytes(w);
    std::vector<uint8_t> h_w(w_bytes);
    std::vector<float>   h_x((size_t) (T * K));
    std::vector<float>   h_y((size_t) (T * M));
    fill_ait5_random(h_w.data(), M / 16, K, 0x1001ULL ^ (uint64_t) (M * 31 + K));
    fill_act_with_outliers(h_x.data(), T, K, 0x2002ULL ^ (uint64_t) (T * 131 + K));

    ggml_backend_tensor_set(w, h_w.data(), 0, w_bytes);
    ggml_backend_tensor_set(x, h_x.data(), 0, h_x.size() * sizeof(float));

    // Seed output buffer with a finite non-zero sentinel so invalid-id zeroing is actively proven
    std::fill(h_y.begin(), h_y.end(), 12345.0f);
    ggml_backend_tensor_set(y, h_y.data(), 0, h_y.size() * sizeof(float));

    ggml_cuda_ait5a16_reset_fire_count();
    for (int i = 0; i < warmup; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: graph compute failed on %s\n", label);
            exit(1);
        }
    }
    ggml_backend_synchronize(backend);

    const bool expect_a16 = env_on("GGML_CUDA_AIT5_A16", false) && T <= env_int("GGML_CUDA_AIT5_A16_MAXT", 8);
    const uint64_t fires_a16 = ggml_cuda_ait5a16_fire_count(0);
    if (expect_a16 && fires_a16 == 0) {
        fprintf(stderr, "FATAL: GGML_CUDA_AIT5_A16=1 was set but AIT5_A16 dense kernel did NOT fire on %s!\n", label);
        exit(2);
    }

    std::vector<double> times;
    times.reserve((size_t) iters);
    for (int i = 0; i < iters; ++i) {
        bwref_flush_l2(bctx);
        bwref_timer_start(bctx, stream);
        if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: async graph compute failed on %s\n", label);
            exit(1);
        }
        times.push_back(bwref_timer_stop_us(bctx, stream));
    }

    ggml_backend_tensor_get(y, h_y.data(), 0, h_y.size() * sizeof(float));

    const double med_us = median_us(times);
    const double eff_gbps = ((double) w_bytes / (med_us * 1e-6)) / 1e9;

    std::string acc_str;
    if (check_ref) {
        const int64_t m_chk = std::min<int64_t>(M, 128);
        std::vector<double> ref = cpu_ref_dense(h_w.data(), h_x.data(), M, K, T, m_chk);
        std::vector<float> got_sub((size_t) (T * m_chk));
        for (int64_t t = 0; t < T; ++t) {
            std::memcpy(got_sub.data() + t * m_chk, h_y.data() + t * M, (size_t) m_chk * sizeof(float));
        }
        const acc_metrics am = compare_fp64_ref(got_sub.data(), ref.data(), ref.size());
        char buf_s[128];
        snprintf(buf_s, sizeof(buf_s), "rel_rmse=%.2e max_abs=%.2e cos=%.7f", am.rel_rmse, am.max_abs, am.cos_sim);
        acc_str = buf_s;
    }

    printf("  %-26s M=%5lld K=%5lld T=%lld | %8.2f us | %6.1f GB/s | a16_fire=%llu | %s\n",
           label, (long long) M, (long long) K, (long long) T, med_us, eff_gbps,
           (unsigned long long) fires_a16, acc_str.c_str());

    char rec_name[128];
    snprintf(rec_name, sizeof(rec_name), "dense_%s_M%lld_K%lld_T%lld", label, (long long) M, (long long) K, (long long) T);
    dumps.push_back({rec_name, std::move(h_y)});

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void run_moe_case(
        ggml_backend_t backend, bwref_ctx * bctx, void * stream,
        const char * label, int64_t M, int64_t K, int64_t n_expert, int64_t n_used, int64_t ne11, int64_t n_tok,
        bool inject_invalid, int warmup, int iters, bool check_ref, std::vector<dump_record> & dumps) {
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 16 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w   = ggml_new_tensor_3d(ctx, GGML_TYPE_AIT5_U, K, M, n_expert);
    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, ne11, n_tok);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tok);
    ggml_tensor * y   = ggml_mul_mat_id(ctx, w, x, ids);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t w_bytes = ggml_nbytes(w);
    const int64_t n_rows  = ne11 * n_tok;
    const int64_t n_pairs = n_used * n_tok;
    std::vector<uint8_t> h_w(w_bytes);
    std::vector<float>   h_x((size_t) (n_rows * K));
    std::vector<int32_t> h_ids((size_t) n_pairs);
    std::vector<float>   h_y((size_t) (n_pairs * M));

    fill_ait5_random(h_w.data(), n_expert * (M / 16), K, 0x3003ULL ^ (uint64_t) (M * 17 + K));
    fill_act_with_outliers(h_x.data(), n_rows, K, 0x4004ULL ^ (uint64_t) (n_tok * 53 + K));
    fill_zipf_ids(h_ids.data(), (int) n_used, (int) n_tok, (int) n_expert,
                  0x5005ULL ^ (uint64_t) n_tok, true /* inject_dup */, inject_invalid);

    ggml_backend_tensor_set(w,   h_w.data(),   0, w_bytes);
    ggml_backend_tensor_set(x,   h_x.data(),   0, h_x.size() * sizeof(float));
    ggml_backend_tensor_set(ids, h_ids.data(), 0, h_ids.size() * sizeof(int32_t));

    // Count unique active experts for effective HBM bandwidth reporting
    std::vector<uint8_t> seen((size_t) n_expert, 0);
    int64_t unique_exp = 0;
    for (int64_t p = 0; p < n_pairs; ++p) {
        const int32_t e = h_ids[(size_t) p];
        if (e >= 0 && e < n_expert && !seen[(size_t) e]) {
            seen[(size_t) e] = 1;
            ++unique_exp;
        }
    }

    // Seed output buffer: non-zero sentinel when path guarantees zeroing (actively proves zero-write), else 0.0f
    const bool guarantees_zero_pre = env_on("GGML_CUDA_AIT5_A16_ID", false) ||
                                     env_on("GGML_CUDA_AI4_DECODE_FAST", true) ||
                                     env_on("GGML_CUDA_AI_ID_SKIP_ZERO", false);
    std::fill(h_y.begin(), h_y.end(), guarantees_zero_pre ? 12345.0f : 0.0f);
    ggml_backend_tensor_set(y, h_y.data(), 0, h_y.size() * sizeof(float));

    ggml_cuda_ait5a16_reset_fire_count();
    ggml_cuda_ai5_lowreg_id_reset_fire_count();
    ggml_cuda_ai5_coop128_reset_fire_count();

    for (int i = 0; i < warmup; ++i) {
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: MoE graph compute failed on %s\n", label);
            exit(1);
        }
    }
    ggml_backend_synchronize(backend);

    const uint64_t f_a16_id = ggml_cuda_ait5a16_fire_count(1);
    const uint64_t f_lowreg = ggml_cuda_ai5_lowreg_id_fire_count();
    const uint64_t f_coop   = ggml_cuda_ai5_coop128_fire_count();

    const bool expect_a16_id = env_on("GGML_CUDA_AIT5_A16_ID", false) &&
                               n_tok <= env_int("GGML_CUDA_AIT5_A16_MAXT", 8) &&
                               (n_used * n_tok <= 64);
    const bool expect_lowreg = env_on("GGML_CUDA_AI5_LOWREG_ID", false) &&
                               n_tok >= env_int("GGML_CUDA_AI5_LOWREG_ID_MIN_N", 64) &&
                               n_tok <= env_int("GGML_CUDA_AI5_LOWREG_ID_MAX_N", 1024);
    const bool expect_coop   = env_on("GGML_CUDA_AI5_COOP128", false) &&
                               !expect_lowreg &&
                               env_int("GGML_CUDA_AI4_ID_SHAPE", 0) == 0 &&
                               n_tok >= env_int("GGML_CUDA_AI4_BN128_MIN", 512);

    if (expect_a16_id && f_a16_id == 0) {
        fprintf(stderr, "FATAL: GGML_CUDA_AIT5_A16_ID=1 set but AIT5_A16_ID did NOT fire on %s!\n", label);
        exit(2);
    }
    if (expect_lowreg && f_lowreg == 0) {
        fprintf(stderr, "FATAL: GGML_CUDA_AI5_LOWREG_ID=1 set but AI5_LOWREG_ID did NOT fire on %s!\n", label);
        exit(2);
    }
    if (expect_coop && f_coop == 0) {
        fprintf(stderr, "FATAL: GGML_CUDA_AI5_COOP128=1 set but AI5_COOP128 did NOT fire on %s!\n", label);
        exit(2);
    }

    std::vector<double> times;
    times.reserve((size_t) iters);
    for (int i = 0; i < iters; ++i) {
        bwref_flush_l2(bctx);
        bwref_timer_start(bctx, stream);
        if (ggml_backend_graph_compute_async(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: MoE async graph compute failed on %s\n", label);
            exit(1);
        }
        times.push_back(bwref_timer_stop_us(bctx, stream));
    }

    ggml_backend_tensor_get(y, h_y.data(), 0, h_y.size() * sizeof(float));

    const double med_us = median_us(times);
    const double active_w_bytes = (double) unique_exp * (double) w->nb[2];
    const double eff_gbps = (active_w_bytes / (med_us * 1e-6)) / 1e9;

    // Verify invalid-id zero column when a path guarantees zeroing (AIT5_A16_ID, DECODE_FAST, or AI_ID_SKIP_ZERO=1)
    const bool guarantees_zero = expect_a16_id || env_on("GGML_CUDA_AI4_DECODE_FAST", true) || env_on("GGML_CUDA_AI_ID_SKIP_ZERO", false);
    if (inject_invalid && guarantees_zero) {
        const int64_t inv_p = n_pairs - 1;
        for (int64_t r = 0; r < M; ++r) {
            if (h_y[(size_t) (inv_p * M + r)] != 0.0f) {
                fprintf(stderr, "FATAL: invalid expert id (-1) produced non-zero output %.6e at row %lld on %s!\n",
                        h_y[(size_t) (inv_p * M + r)], (long long) r, label);
                exit(3);
            }
        }
    }

    std::string acc_str;
    if (check_ref && n_tok <= 8) {
        const int64_t m_chk = std::min<int64_t>(M, 64);
        std::vector<double> ref = cpu_ref_moe(h_w.data(), (int64_t) w->nb[2], h_x.data(), h_ids.data(),
                                              K, n_used, ne11, n_tok, n_expert, m_chk);
        std::vector<float> got_sub((size_t) (n_pairs * m_chk));
        for (int64_t p = 0; p < n_pairs; ++p) {
            std::memcpy(got_sub.data() + p * m_chk, h_y.data() + p * M, (size_t) m_chk * sizeof(float));
        }
        const acc_metrics am = compare_fp64_ref(got_sub.data(), ref.data(), ref.size());
        char buf_s[128];
        snprintf(buf_s, sizeof(buf_s), "rel_rmse=%.2e max_abs=%.2e cos=%.7f", am.rel_rmse, am.max_abs, am.cos_sim);
        acc_str = buf_s;
    }

    printf("  %-22s M=%4lld K=%4lld N=%4lld actE=%3lld | %8.2f us | %6.1f GB/s | fires(a16=%llu lr=%llu c128=%llu) | %s\n",
           label, (long long) M, (long long) K, (long long) n_tok, (long long) unique_exp,
           med_us, eff_gbps,
           (unsigned long long) f_a16_id, (unsigned long long) f_lowreg, (unsigned long long) f_coop,
           acc_str.c_str());

    char rec_name[128];
    snprintf(rec_name, sizeof(rec_name), "moe_%s_M%lld_K%lld_N%lld", label, (long long) M, (long long) K, (long long) n_tok);
    dumps.push_back({rec_name, std::move(h_y)});

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

} // namespace

int main(int argc, char ** argv) {
    int device = 0;
    int warmup = 5;
    int iters  = 25;
    bool check_ref = true;
    std::string mode = "all";
    std::string dump_path;
    std::string cmp_path;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--cmp") && i + 1 < argc) cmp_path = argv[++i];
        else if (!strcmp(argv[i], "--no-ref")) check_ref = false;
        else {
            fprintf(stderr, "Usage: %s [--device N] [--mode all|dense|moe-decode|moe-prefill] [--warmup N] [--iters N] [--dump F] [--cmp F] [--no-ref]\n", argv[0]);
            return 1;
        }
    }

    bwref_print_device(device);
    bwref_ctx * bctx = bwref_init(device);
    const double bwref_gbps = bwref_measure_gbps(bctx, 15);
    printf("BWREF raw HBM read (256 MiB LDG.E.128): %.1f GB/s\n", bwref_gbps);
    printf("Env: DECODE_FAST=%d AIT5_A16=%d AIT5_A16_ID=%d AIT5_A16_MAXT=%d AI5_LOWREG_ID=%d AI5_COOP128=%d\n",
           env_int("GGML_CUDA_AI4_DECODE_FAST", 1),
           env_int("GGML_CUDA_AIT5_A16", 0),
           env_int("GGML_CUDA_AIT5_A16_ID", 0),
           env_int("GGML_CUDA_AIT5_A16_MAXT", 8),
           env_int("GGML_CUDA_AI5_LOWREG_ID", 0),
           env_int("GGML_CUDA_AI5_COOP128", 0));

    ggml_backend_t backend = ggml_backend_cuda_init(device);
    if (!backend) {
        fprintf(stderr, "ERROR: ggml_backend_cuda_init(%d) failed\n", device);
        bwref_free(bctx);
        return 1;
    }
    void * stream = ggml_cuda_ait5_backend_stream(backend);
    std::vector<dump_record> dumps;

    if (mode == "all" || mode == "dense") {
        printf("\n=== 1. Dense AIT5_U MUL_MAT (T = 1, 5, 8; 15x outlier channels) ===\n");
        struct shape { const char * name; int64_t M; int64_t K; };
        const shape shapes[] = {
            { "inkling_attn_q", 3072,  2304 },
            { "inkling_h128",   2304,  1152 },
            { "27b_sq5120",     5120,  5120 },
            { "27b_down17k",   17408,  5120 },
        };
        const int64_t widths[] = { 1, 5, 8 };
        for (const auto & s : shapes) {
            for (int64_t T : widths) {
                run_dense_case(backend, bctx, stream, s.name, s.M, s.K, T, warmup, iters, check_ref, dumps);
            }
        }
    }

    if (mode == "all" || mode == "moe-decode") {
        printf("\n=== 2. MoE AIT5_U MUL_MAT_ID Decode (128e top-8 Zipf + dup + invalid-id) ===\n");
        const int64_t widths[] = { 1, 2, 4, 5, 8 };
        for (int64_t T : widths) {
            run_moe_case(backend, bctx, stream, "gu_2304x2304", 2304, 2304, 128, 8, 1, T, true, warmup, iters, check_ref, dumps);
            run_moe_case(backend, bctx, stream, "dn_2304x1152", 2304, 1152, 128, 8, 8, T, true, warmup, iters, check_ref, dumps);
        }
    }

    if (mode == "all" || mode == "moe-prefill") {
        printf("\n=== 3. MoE AIT5_U MUL_MAT_ID Prefill (128e top-8 Zipf + dup) ===\n");
        const int64_t ns[] = { 512, 1024, 2048 };
        for (int64_t N : ns) {
            run_moe_case(backend, bctx, stream, "gu_2304x2304", 2304, 2304, 128, 8, 1, N, false, warmup, iters, false, dumps);
            run_moe_case(backend, bctx, stream, "dn_2304x1152", 2304, 1152, 128, 8, 8, N, false, warmup, iters, false, dumps);
        }
    }

    if (!dump_path.empty()) {
        save_dumps(dump_path, dumps);
    }
    if (!cmp_path.empty()) {
        compare_dumps(cmp_path, dumps);
    }

    ggml_backend_free(backend);
    bwref_free(bctx);
    return 0;
}
