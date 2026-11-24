// Pure CPU numerical verification for SJ-KVaRN 4/4 verify fast paths:
//   1. STREAM_PV16: FP16 Q*K^T (FP32 acc) + centered [-8,+7] V codes + scaled FP16 P*V accumulation
//   2. DIRECT_VERIFY: U8*S8 INT8 Q*K^T (with 7.5*eps DC correction & KBIAS ulp-2) + U8*U8 P*V (1-strip and 2-strip)
// Tested against exact FP64 SJ-KVaRN 4/4 reference AND ggml-cpu OP_FLASH_ATTN_EXT on:
//   - Native D=256, G=128 sealed records with Outl100x K/V channel outliers
//   - Inkling pair-packed D=128 -> D=256 (H_q=24, H_kv=8 -> P=4, GQA=6) with H256 rotation & TQ6_0/F16 ring/sink

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-sjkvarn.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

static inline float fp16_round(float x) {
    return GGML_FP16_TO_FP32(GGML_FP32_TO_FP16(x));
}

static inline void hadamard256_inplace(float * x) {
    for (int h = 1; h < 256; h <<= 1) {
        for (int i = 0; i < 256; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 256; ++i) {
        x[i] *= 0.0625f;
    }
}

static inline float rintf_even(float x) {
    return nearbyintf(x);
}

struct Metrics {
    double nmse = 0.0;
    double max_abs = 0.0;
    double cos_sim = 1.0;
    double sqnr_db = 100.0;
};

static Metrics compute_metrics(const std::vector<float> & test, const std::vector<double> & ref) {
    double se = 0.0, sref = 0.0, stest = 0.0, dot = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = test[i];
        const double b = ref[i];
        if (!std::isfinite(a)) {
            fprintf(stderr, "NON-FINITE value at index %zu: %g\n", i, a);
            std::abort();
        }
        const double d = a - b;
        se += d * d;
        sref += b * b;
        stest += a * a;
        dot += a * b;
        max_abs = std::max(max_abs, std::abs(d));
    }
    Metrics m;
    m.nmse = sref > 0.0 ? se / sref : se;
    m.max_abs = max_abs;
    m.cos_sim = (sref > 0.0 && stest > 0.0) ? dot / (std::sqrt(sref) * std::sqrt(stest)) : 1.0;
    m.sqnr_db = m.nmse > 0.0 ? -10.0 * std::log10(m.nmse) : 120.0;
    return m;
}

struct TestDataset {
    int D = 256;
    int G = 128;
    int S = 128;
    int cap = 256;
    int n_groups = 12; // body: 1536 tokens; total N = 128 + 1536 + 128 = 1792 (padded 2048)
    int N = 1792;
    int n_kv_pad = 2048;
    int hkv = 4;
    int nh = 24;
    int gqa = 6;
    int n_q = 5;
    bool d128_packed = false;
    bool fused_rot = false;
    bool has_sinks = false;
    bool per_head_mask = false;
    ggml_type ring_type = GGML_TYPE_TQ6_0;
    bool f16_sink = true;

    ggml_sj_kvarn::layout l;
    std::vector<float> Q;            // [D, nh, n_q] (unrotated if fused_rot, else rotated)
    std::vector<float> Q_rot;        // [D, nh, n_q] rotated
    std::vector<uint8_t> body;       // [n_groups, hkv, l.bytes]
    std::vector<ggml_fp16_t> sink_K; // [S, hkv, D]
    std::vector<ggml_fp16_t> sink_V; // [S, hkv, D]
    std::vector<float> ring_K;       // [N - B, hkv, D] decoded from ring_type
    std::vector<float> ring_V;       // [N - B, hkv, D] decoded from ring_type
    std::vector<float> mask;         // [n_kv_pad, n_q, (per_head_mask ? nh : 1)]
    std::vector<float> sinks;        // [nh]
    int B = 0;
};

static TestDataset generate_dataset(uint32_t seed, int n_q, bool d128_packed, bool outl100x, bool fused_rot, bool has_sinks, bool per_head_mask) {
    TestDataset ds;
    ds.n_q = n_q;
    ds.d128_packed = d128_packed;
    ds.fused_rot = fused_rot;
    ds.has_sinks = has_sinks;
    ds.per_head_mask = per_head_mask;
    ds.B = ds.S + ds.n_groups * ds.G;
    ds.N = ds.B + 128;
    ds.n_kv_pad = ((ds.N + 255) / 256) * 256;
    ds.l = ggml_sj_kvarn::make_layout(ds.D, ds.G, 4, 4);

    std::mt19937 rng(seed);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    const int dh = d128_packed ? 128 : 256;
    // Generate unrotated K, V across [N, hkv, 256] (if d128_packed, two 128-wide heads per packed KV head)
    std::vector<float> K_raw((size_t) ds.N * ds.hkv * ds.D);
    std::vector<float> V_raw((size_t) ds.N * ds.hkv * ds.D);

    // Persistent channel outlier gains (Outl100x: 4% of channels scaled by 100x, plus per-record drift)
    std::vector<float> k_gain(ds.hkv * ds.D, 1.0f), v_gain(ds.hkv * ds.D, 1.0f);
    if (outl100x) {
        for (int h = 0; h < ds.hkv; ++h) {
            for (int d = 0; d < ds.D; ++d) {
                if ((d % 29) == (h * 7) % 29) {
                    k_gain[h * ds.D + d] = 100.0f;
                }
                if ((d % 31) == (h * 5 + 3) % 31) {
                    v_gain[h * ds.D + d] = 100.0f;
                }
            }
        }
    }

    for (int p = 0; p < ds.N; ++p) {
        // Also vary token scale occasionally to stress dynamic Vscale/Ktok scaling
        const float tok_scale = (p % 97 == 13) ? 12.0f : (p % 53 == 7) ? 0.15f : 1.0f;
        for (int h = 0; h < ds.hkv; ++h) {
            for (int d = 0; d < ds.D; ++d) {
                K_raw[((size_t) p * ds.hkv + h) * ds.D + d] = norm(rng) * k_gain[h * ds.D + d] * 0.25f * tok_scale;
                V_raw[((size_t) p * ds.hkv + h) * ds.D + d] = norm(rng) * v_gain[h * ds.D + d] * 0.25f * tok_scale;
            }
        }
    }

    // Rotate K and V rows with H256 and round to fp16 (cache domain)
    std::vector<ggml_fp16_t> K_rot((size_t) ds.N * ds.hkv * ds.D);
    std::vector<ggml_fp16_t> V_rot((size_t) ds.N * ds.hkv * ds.D);
    std::vector<float> row(ds.D);
    for (int p = 0; p < ds.N; ++p) {
        for (int h = 0; h < ds.hkv; ++h) {
            const float * kr = K_raw.data() + ((size_t) p * ds.hkv + h) * ds.D;
            const float * vr = V_raw.data() + ((size_t) p * ds.hkv + h) * ds.D;
            memcpy(row.data(), kr, ds.D * sizeof(float));
            hadamard256_inplace(row.data());
            for (int d = 0; d < ds.D; ++d) K_rot[((size_t) p * ds.hkv + h) * ds.D + d] = GGML_FP32_TO_FP16(row[d]);
            memcpy(row.data(), vr, ds.D * sizeof(float));
            hadamard256_inplace(row.data());
            for (int d = 0; d < ds.D; ++d) V_rot[((size_t) p * ds.hkv + h) * ds.D + d] = GGML_FP32_TO_FP16(row[d]);
        }
    }

    // Seal body [S, B)
    ds.body.resize((size_t) ds.n_groups * ds.hkv * ds.l.bytes);
    for (int g = 0; g < ds.n_groups; ++g) {
        for (int h = 0; h < ds.hkv; ++h) {
            uint8_t * rec = ds.body.data() + ((size_t) g * ds.hkv + h) * ds.l.bytes;
            const size_t base = ((size_t) (ds.S + g * ds.G) * ds.hkv + h) * ds.D;
            ggml_sj_kvarn::seal_group(K_rot.data() + base, V_rot.data() + base, (size_t) ds.hkv * ds.D, ds.l, 16, rec, false);
        }
    }

    // Sink [0, S) in F16
    ds.sink_K.resize((size_t) ds.S * ds.hkv * ds.D);
    ds.sink_V.resize((size_t) ds.S * ds.hkv * ds.D);
    memcpy(ds.sink_K.data(), K_rot.data(), ds.sink_K.size() * sizeof(ggml_fp16_t));
    memcpy(ds.sink_V.data(), V_rot.data(), ds.sink_V.size() * sizeof(ggml_fp16_t));

    // Ring [B, N) quantized through TQ6_0 then dequantized (matches exact CUDA ring rows)
    const int n_ring = ds.N - ds.B;
    ds.ring_K.resize((size_t) n_ring * ds.hkv * ds.D);
    ds.ring_V.resize((size_t) n_ring * ds.hkv * ds.D);
    std::vector<block_tq6_0> qbuf(ds.D / QK_TQ6);
    for (int p = 0; p < n_ring; ++p) {
        for (int h = 0; h < ds.hkv; ++h) {
            const ggml_fp16_t * rk = K_rot.data() + ((size_t) (ds.B + p) * ds.hkv + h) * ds.D;
            const ggml_fp16_t * rv = V_rot.data() + ((size_t) (ds.B + p) * ds.hkv + h) * ds.D;
            std::vector<float> tmp(ds.D);
            ggml_fp16_to_fp32_row(rk, tmp.data(), ds.D);
            quantize_row_tq6_0_rotated_ref(tmp.data(), qbuf.data(), ds.D);
            dequantize_row_tq6_0(qbuf.data(), ds.ring_K.data() + ((size_t) p * ds.hkv + h) * ds.D, ds.D);
            ggml_fp16_to_fp32_row(rv, tmp.data(), ds.D);
            quantize_row_tq6_0_rotated_ref(tmp.data(), qbuf.data(), ds.D);
            dequantize_row_tq6_0(qbuf.data(), ds.ring_V.data() + ((size_t) p * ds.hkv + h) * ds.D, ds.D);
        }
    }

    // Q [D, nh, n_q]: unrotated if fused_rot, else rotated
    ds.Q.resize((size_t) ds.n_q * ds.nh * ds.D, 0.0f);
    ds.Q_rot.resize((size_t) ds.n_q * ds.nh * ds.D, 0.0f);
    for (int q = 0; q < ds.n_q; ++q) {
        for (int h = 0; h < ds.nh; ++h) {
            const int h_in_g = h % ds.gqa;
            float * dst_q = ds.Q.data() + ((size_t) q * ds.nh + h) * ds.D;
            if (d128_packed) {
                const int offset = (h_in_g >= ds.gqa / 2) ? 128 : 0;
                for (int d = 0; d < dh; ++d) {
                    dst_q[offset + d] = norm(rng) * 0.5f;
                }
            } else {
                for (int d = 0; d < ds.D; ++d) {
                    dst_q[d] = norm(rng) * 0.5f;
                }
            }
            memcpy(row.data(), dst_q, ds.D * sizeof(float));
            hadamard256_inplace(row.data());
            memcpy(ds.Q_rot.data() + ((size_t) q * ds.nh + h) * ds.D, row.data(), ds.D * sizeof(float));
            if (!fused_rot) {
                memcpy(dst_q, row.data(), ds.D * sizeof(float));
            }
        }
    }

    // Mask [n_kv_pad, n_q, m_heads]
    const int m_heads = per_head_mask ? ds.nh : 1;
    ds.mask.resize((size_t) m_heads * ds.n_q * ds.n_kv_pad, -INFINITY);
    for (int mh = 0; mh < m_heads; ++mh) {
        for (int q = 0; q < ds.n_q; ++q) {
            const int vis_end = ds.N - (ds.n_q - 1 - q);
            for (int p = 0; p < ds.n_kv_pad; ++p) {
                if (p < vis_end) {
                    float bias = 0.0f;
                    if (per_head_mask) {
                        bias = -0.0005f * float(vis_end - 1 - p) * (1.0f + 0.1f * (mh % 6));
                    }
                    ds.mask[((size_t) mh * ds.n_q + q) * ds.n_kv_pad + p] = fp16_round(bias);
                }
            }
        }
    }

    // Sinks
    ds.sinks.resize(ds.nh, 0.0f);
    if (has_sinks) {
        for (int h = 0; h < ds.nh; ++h) {
            ds.sinks[h] = -1.5f + 0.25f * float(h % 8);
        }
    }
    (void) uni;
    return ds;
}

// Exact FP64 reference over decoded SJ-KVaRN rows
static std::vector<double> run_fp64_reference(const TestDataset & ds, float scale) {
    const int D = ds.D;
    std::vector<double> out((size_t) ds.n_q * ds.nh * D, 0.0);
    std::vector<float> kf(D), vf(D);
    std::vector<double> scores(ds.N);

    for (int q = 0; q < ds.n_q; ++q) {
        for (int h = 0; h < ds.nh; ++h) {
            const int z_kv = h / ds.gqa;
            const int mh = ds.per_head_mask ? h : 0;
            const float * qrow = ds.Q_rot.data() + ((size_t) q * ds.nh + h) * D;
            const float * mrow = ds.mask.data() + ((size_t) mh * ds.n_q + q) * ds.n_kv_pad;

            double mx = -INFINITY;
            for (int p = 0; p < ds.N; ++p) {
                const float mk = mrow[p];
                if (mk == -INFINITY) {
                    scores[p] = -INFINITY;
                    continue;
                }
                if (p < ds.S) {
                    ggml_fp16_to_fp32_row(ds.sink_K.data() + ((size_t) p * ds.hkv + z_kv) * D, kf.data(), D);
                } else if (p < ds.B) {
                    const int g = (p - ds.S) / ds.G;
                    const int t = (p - ds.S) % ds.G;
                    const uint8_t * rec = ds.body.data() + ((size_t) g * ds.hkv + z_kv) * ds.l.bytes;
                    ggml_sj_kvarn::decode_k_row(rec, ds.l, t, kf.data(), false);
                } else {
                    const int rp = p - ds.B;
                    memcpy(kf.data(), ds.ring_K.data() + ((size_t) rp * ds.hkv + z_kv) * D, D * sizeof(float));
                }
                double s = 0.0;
                for (int d = 0; d < D; ++d) {
                    s += (double) qrow[d] * (double) scale * (double) kf[d];
                }
                s += (double) mk;
                scores[p] = s;
                mx = std::max(mx, s);
            }
            if (ds.has_sinks) {
                mx = std::max(mx, (double) ds.sinks[h]);
            }

            double sum = ds.has_sinks ? std::exp((double) ds.sinks[h] - mx) : 0.0;
            std::vector<double> acc(D, 0.0);
            for (int p = 0; p < ds.N; ++p) {
                if (scores[p] == -INFINITY) continue;
                const double w = std::exp(scores[p] - mx);
                sum += w;
                if (p < ds.S) {
                    ggml_fp16_to_fp32_row(ds.sink_V.data() + ((size_t) p * ds.hkv + z_kv) * D, vf.data(), D);
                } else if (p < ds.B) {
                    const int g = (p - ds.S) / ds.G;
                    const int t = (p - ds.S) % ds.G;
                    const uint8_t * rec = ds.body.data() + ((size_t) g * ds.hkv + z_kv) * ds.l.bytes;
                    ggml_sj_kvarn::decode_v_row(rec, ds.l, t, vf.data(), false);
                } else {
                    const int rp = p - ds.B;
                    memcpy(vf.data(), ds.ring_V.data() + ((size_t) rp * ds.hkv + z_kv) * D, D * sizeof(float));
                }
                for (int d = 0; d < D; ++d) {
                    acc[d] += w * (double) vf[d];
                }
            }

            std::vector<float> of(D);
            for (int d = 0; d < D; ++d) of[d] = (float) (acc[d] / sum);
            hadamard256_inplace(of.data());
            for (int d = 0; d < D; ++d) out[((size_t) q * ds.nh + h) * D + d] = (double) of[d];
        }
    }
    return out;
}

// Bit-faithful CPU simulation of STREAM_PV16:
//   - Q*K^T: FP16 Q'*K_code + K_zero.Q, FP32 accumulation
//   - Softmax: FP32 online softmax
//   - P*V: centered V codes (q - 8) in [-8, +7], dynamic FP16 acc16[D] with acc_scale & acc_bound,
//          record_change FP32 rescale across Vch_old / Vch_new, exact strips via FP16 MMA
static std::vector<float> run_stream_pv16_sim(const TestDataset & ds, float scale) {
    const int D = ds.D;
    std::vector<float> out((size_t) ds.n_q * ds.nh * D, 0.0f);

    for (int q = 0; q < ds.n_q; ++q) {
        for (int h = 0; h < ds.nh; ++h) {
            const int z_kv = h / ds.gqa;
            const int mh = ds.per_head_mask ? h : 0;
            const float * qrow = ds.Q_rot.data() + ((size_t) q * ds.nh + h) * D;
            const float * mrow = ds.mask.data() + ((size_t) mh * ds.n_q + q) * ds.n_kv_pad;

            std::vector<float> acc16(D, 0.0f); // stored rounded to FP16 at every strip
            float acc_scale = 0.0f;
            float acc_bound = 0.0f;
            float kq_max = -FLT_MAX / 2.0f;
            float kq_sum = 0.0f;
            float pz = 0.0f;

            // Walk body records in 16-token strips
            for (int g = 0; g < ds.n_groups; ++g) {
                const uint8_t * rec = ds.body.data() + ((size_t) g * ds.hkv + z_kv) * ds.l.bytes;
                // setup_body
                std::vector<float> qp(D), vch(D);
                float c_r = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const int ki = ggml_sj_kvarn::k_ch_idx(d, D, ds.G, 4);
                    const int vi = ggml_sj_kvarn::v_ch_idx(d, D, ds.G, 4);
                    const float ks = ggml_sj_kvarn::get_half(rec, ds.l.k_scale + 2 * ki);
                    const float kz = ggml_sj_kvarn::get_half(rec, ds.l.k_zero + 2 * ki);
                    const float qsc = fp16_round(qrow[d] * scale);
                    qp[d] = fp16_round(qsc * ks);
                    c_r += (qrow[d] * scale) * kz;
                    vch[d] = ggml_sj_kvarn::get_half(rec, ds.l.v_ch + 2 * vi);
                }

                for (int sl = 0; sl < ds.G / 16; ++sl) {
                    const int p0 = ds.S + g * ds.G + 16 * sl;
                    float kq[16], e[16], vs[16], vz[16];
                    float mx = -FLT_MAX / 2.0f;
                    for (int t = 0; t < 16; ++t) {
                        const int tok = 16 * sl + t;
                        const float tok_s = ggml_sj_kvarn::get_half(rec, ds.l.k_tok + 2 * tok);
                        // FP16 MMA dot product + FP32 acc
                        float dot = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            const float kc = (float) ggml_sj_kvarn::k_code(rec, ds.l, tok, d);
                            dot += fp16_round(kc * qp[d]);
                        }
                        kq[t] = tok_s * (dot + c_r) + mrow[p0 + t];
                        mx = std::max(mx, kq[t]);
                        vs[t] = ggml_sj_kvarn::get_half(rec, ds.l.v_scale + 2 * tok);
                        vz[t] = ggml_sj_kvarn::get_half(rec, ds.l.v_zero + 2 * tok);
                    }
                    const float nm = std::max(kq_max, mx);
                    if (nm != kq_max) {
                        const float sc = expf(kq_max - nm);
                        kq_max = nm;
                        kq_sum *= sc;
                        pz *= sc;
                        acc_scale *= sc;
                    }
                    float pv_max = 0.0f, pv_sum = 0.0f;
                    for (int t = 0; t < 16; ++t) {
                        e[t] = expf(kq[t] - nm);
                        kq_sum += e[t];
                        pz += e[t] * (vz[t] + 8.0f * vs[t]);
                        const float av = e[t] * vs[t];
                        pv_max = std::max(pv_max, av);
                        pv_sum += av;
                    }
                    if (pv_max > 0.0f) {
                        const float strip_peak = 8.0f * pv_sum;
                        const bool need_rescale = (acc_scale == 0.0f) || (pv_max > 8.0f * acc_scale) ||
                            (acc_bound + strip_peak / acc_scale > 16384.0f) || (acc_scale < 1e-20f);
                        if (need_rescale) {
                            const float old_true_bound = acc_scale * acc_bound;
                            float new_scale = std::max((old_true_bound + strip_peak) * (1.0f / 256.0f), pv_max * (1.0f / 8.0f));
                            if (old_true_bound > 0.0f) {
                                new_scale = std::max(new_scale, acc_scale * (1.0f / 1024.0f));
                            }
                            const float alpha = (acc_scale > 0.0f && old_true_bound > 0.0f) ? fp16_round(acc_scale / new_scale) : 0.0f;
                            for (int d = 0; d < D; ++d) {
                                acc16[d] = fp16_round(acc16[d] * alpha);
                            }
                            acc_bound = old_true_bound / new_scale;
                            acc_scale = new_scale;
                        }
                        acc_bound += strip_peak / acc_scale;
                        const float inv_s = 1.0f / acc_scale;
                        float p_mma[16];
                        for (int t = 0; t < 16; ++t) {
                            p_mma[t] = fp16_round(e[t] * vs[t] * inv_s);
                        }
                        for (int d = 0; d < D; ++d) {
                            // Simulates HMMA.16816.F16: 16-element sum rounded to FP16 + acc16[d] rounded to FP16
                            float sum16 = 0.0f;
                            for (int t = 0; t < 16; ++t) {
                                const int tok = 16 * sl + t;
                                const float vc = (float) ((int) ggml_sj_kvarn::v_code(rec, ds.l, tok, d) - 8);
                                sum16 += p_mma[t] * vc; // internal MMA accumulation, then rounded to FP16
                            }
                            acc16[d] = fp16_round(acc16[d] + fp16_round(sum16));
                        }
                    }
                }

                // record_change(rec_old = rec, rec_new = next or nullptr)
                const uint8_t * rec_next = (g + 1 < ds.n_groups) ? ds.body.data() + ((size_t) (g + 1) * ds.hkv + z_kv) * ds.l.bytes : nullptr;
                std::vector<float> tmp_f32(D);
                float max_d = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const float vo = vch[d];
                    const float vn = rec_next ? ggml_sj_kvarn::get_half(rec_next, ds.l.v_ch + 2 * ggml_sj_kvarn::v_ch_idx(d, D, ds.G, 4)) : 1.0f;
                    const float r = vo / vn;
                    tmp_f32[d] = (acc16[d] * acc_scale + pz) * r;
                    max_d = std::max(max_d, std::abs(tmp_f32[d]));
                }
                if (max_d > 0.0f) {
                    acc_scale = max_d * (1.0f / 256.0f);
                    const float inv = 1.0f / acc_scale;
                    for (int d = 0; d < D; ++d) {
                        acc16[d] = fp16_round(tmp_f32[d] * inv);
                    }
                    acc_bound = 256.0f;
                } else {
                    std::fill(acc16.begin(), acc16.end(), 0.0f);
                    acc_scale = 0.0f;
                    acc_bound = 0.0f;
                }
                pz = 0.0f;
            }

            // Exact strips: sink [0, S) + ring [B, N)
            const int n_ex_sink = ds.S / 16;
            const int n_ex_ring = (ds.n_kv_pad - ds.B) / 16;
            std::vector<float> kf(D), vf(D);
            for (int e_idx = 0; e_idx < n_ex_sink + n_ex_ring; ++e_idx) {
                const int p0 = e_idx < n_ex_sink ? 16 * e_idx : ds.B + 16 * (e_idx - n_ex_sink);
                float kq[16];
                float mx = -FLT_MAX / 2.0f;
                std::vector<std::vector<float>> v_strip(16, std::vector<float>(D, 0.0f));
                float v_absmax = 0.0f;
                for (int t = 0; t < 16; ++t) {
                    const int p = p0 + t;
                    if (p < ds.N) {
                        if (p < ds.S) {
                            ggml_fp16_to_fp32_row(ds.sink_K.data() + ((size_t) p * ds.hkv + z_kv) * D, kf.data(), D);
                            ggml_fp16_to_fp32_row(ds.sink_V.data() + ((size_t) p * ds.hkv + z_kv) * D, vf.data(), D);
                        } else {
                            const int rp = p - ds.B;
                            for (int d = 0; d < D; ++d) {
                                kf[d] = fp16_round(ds.ring_K[((size_t) rp * ds.hkv + z_kv) * D + d]);
                                vf[d] = fp16_round(ds.ring_V[((size_t) rp * ds.hkv + z_kv) * D + d]);
                            }
                        }
                    } else {
                        std::fill(kf.begin(), kf.end(), 0.0f);
                        std::fill(vf.begin(), vf.end(), 0.0f);
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += fp16_round(kf[d] * fp16_round(qrow[d] * scale));
                        v_strip[t][d] = vf[d];
                        v_absmax = std::max(v_absmax, std::abs(vf[d]));
                    }
                    kq[t] = dot + mrow[p];
                    mx = std::max(mx, kq[t]);
                }
                const float nm = std::max(kq_max, mx);
                if (nm != kq_max) {
                    const float sc = expf(kq_max - nm);
                    kq_max = nm;
                    kq_sum *= sc;
                    acc_scale *= sc;
                }
                float e_vals[16], e_sum = 0.0f;
                for (int t = 0; t < 16; ++t) {
                    e_vals[t] = expf(kq[t] - nm);
                    kq_sum += e_vals[t];
                    e_sum += e_vals[t];
                }
                const float strip_peak = v_absmax * e_sum;
                if (strip_peak > 0.0f) {
                    const float pv_max = v_absmax;
                    const bool need_rescale = (acc_scale == 0.0f) || (e_sum > 8.0f * acc_scale) ||
                        (acc_bound + strip_peak / acc_scale > 16384.0f) || (acc_scale < 1e-20f);
                    if (need_rescale) {
                        const float old_true_bound = acc_scale * acc_bound;
                        float new_scale = std::max((old_true_bound + strip_peak) * (1.0f / 256.0f), e_sum * pv_max * (1.0f / 8.0f));
                        if (old_true_bound > 0.0f) {
                            new_scale = std::max(new_scale, acc_scale * (1.0f / 1024.0f));
                        }
                        const float alpha = (acc_scale > 0.0f && old_true_bound > 0.0f) ? fp16_round(acc_scale / new_scale) : 0.0f;
                        for (int d = 0; d < D; ++d) acc16[d] = fp16_round(acc16[d] * alpha);
                        acc_bound = old_true_bound / new_scale;
                        acc_scale = new_scale;
                    }
                    acc_bound += strip_peak / acc_scale;
                    const float inv_s = 1.0f / acc_scale;
                    for (int d = 0; d < D; ++d) {
                        float sum16 = 0.0f;
                        for (int t = 0; t < 16; ++t) {
                            sum16 += fp16_round(e_vals[t] * inv_s) * v_strip[t][d];
                        }
                        acc16[d] = fp16_round(acc16[d] + fp16_round(sum16));
                    }
                }
            }

            if (ds.has_sinks) {
                const float nm = std::max(kq_max, ds.sinks[h]);
                const float sc = expf(kq_max - nm);
                kq_max = nm;
                kq_sum = sc * kq_sum + expf(ds.sinks[h] - nm);
                acc_scale *= sc;
            }

            std::vector<float> of(D);
            const float inv = acc_scale / kq_sum;
            for (int d = 0; d < D; ++d) of[d] = acc16[d] * inv;
            hadamard256_inplace(of.data());
            memcpy(out.data() + ((size_t) q * ds.nh + h) * D, of.data(), D * sizeof(float));
        }
    }
    return out;
}

// Bit-faithful CPU simulation of DIRECT_VERIFY (U8*S8 QK + U8*U8 PV, pair_32 = IMMA.16832.U8.U8 vs IMMA.16816.U8.U8)
static std::vector<float> run_direct_verify_sim(const TestDataset & ds, float scale, bool pair_32) {
    const int D = ds.D;
    std::vector<float> out((size_t) ds.n_q * ds.nh * D, 0.0f);

    for (int q = 0; q < ds.n_q; ++q) {
        for (int h = 0; h < ds.nh; ++h) {
            const int z_kv = h / ds.gqa;
            const int mh = ds.per_head_mask ? h : 0;
            const float * qrow = ds.Q_rot.data() + ((size_t) q * ds.nh + h) * D;
            const float * mrow = ds.mask.data() + ((size_t) mh * ds.n_q + q) * ds.n_kv_pad;

            std::vector<float> acc(D, 0.0f);
            float kq_max = -FLT_MAX / 2.0f;
            float kq_sum = 0.0f;
            float pz = 0.0f;

            for (int g = 0; g < ds.n_groups; ++g) {
                const uint8_t * rec = ds.body.data() + ((size_t) g * ds.hkv + z_kv) * ds.l.bytes;
                // setup_body: quantize Q' = fp16(q*scale)*Kscale to int8 + 7.5*sum(eps) DC correction
                std::vector<float> p_vals(D), vch(D);
                float amax = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const int ki = ggml_sj_kvarn::k_ch_idx(d, D, ds.G, 4);
                    const float ks = ggml_sj_kvarn::get_half(rec, ds.l.k_scale + 2 * ki);
                    const float qsc = fp16_round(qrow[d] * scale);
                    p_vals[d] = fp16_round(qsc * ks);
                    amax = std::max(amax, std::abs(p_vals[d]));
                    vch[d] = ggml_sj_kvarn::get_half(rec, ds.l.v_ch + 2 * ggml_sj_kvarn::v_ch_idx(d, D, ds.G, 4));
                }
                const float sq = amax > 0.0f ? amax * (1.0f / 126.5f) : 0.0f;
                const float isq = amax > 0.0f ? fp16_round(126.5f / amax) : 0.0f;
                const float nsq = fp16_round(-sq);
                std::vector<int8_t> qi(D);
                float eps_sum = 0.0f, kz_dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const int ki = ggml_sj_kvarn::k_ch_idx(d, D, ds.G, 4);
                    const float kz = ggml_sj_kvarn::get_half(rec, ds.l.k_zero + 2 * ki);
                    const float qsc = fp16_round(qrow[d] * scale);
                    const float h_val = fp16_round(p_vals[d] * isq + 1152.0f);
                    int iv = (int) (h_val - 1152.0f);
                    iv = std::max(-128, std::min(127, iv));
                    qi[d] = (int8_t) iv;
                    const float eps = fp16_round(fp16_round(h_val - 1152.0f) * nsq + p_vals[d]);
                    eps_sum += eps;
                    kz_dot += qsc * kz;
                }
                const float c_r = kz_dot + 7.5f * eps_sum;

                const int n_strips = ds.G / 16;
                const int step = pair_32 ? 2 : 1;
                for (int sl = 0; sl < n_strips; sl += step) {
                    const int count = std::min(step, n_strips - sl) * 16;
                    std::vector<float> kq(count), vs(count), vz(count), e(count);
                    float mx = -FLT_MAX / 2.0f;
                    for (int t = 0; t < count; ++t) {
                        const int tok = 16 * sl + t;
                        const int p = ds.S + g * ds.G + tok;
                        const float tok_s = ggml_sj_kvarn::get_half(rec, ds.l.k_tok + 2 * tok);
                        int32_t idot = 0;
                        for (int d = 0; d < D; ++d) {
                            idot += (int32_t) ggml_sj_kvarn::k_code(rec, ds.l, tok, d) * (int32_t) qi[d];
                        }
                        // Reproduce KBIAS_I ulp=2 truncation for low nibbles (t%16 < 8) vs 16* for high nibbles
                        float kdot_f;
                        if ((t % 16) < 8) {
                            kdot_f = 0.5f * ((float) (25165824.0f + 2.0f * (float) idot) - 25165824.0f) * sq;
                        } else {
                            kdot_f = (1.0f / 32.0f) * ((float) (25165824.0f + 32.0f * (float) idot) - 25165824.0f) * sq;
                        }
                        kq[t] = tok_s * (kdot_f + c_r) + mrow[p];
                        mx = std::max(mx, kq[t]);
                        vs[t] = ggml_sj_kvarn::get_half(rec, ds.l.v_scale + 2 * tok);
                        vz[t] = ggml_sj_kvarn::get_half(rec, ds.l.v_zero + 2 * tok);
                    }
                    const float nm = std::max(kq_max, mx);
                    if (nm != kq_max) {
                        const float sc = expf(kq_max - nm);
                        kq_max = nm;
                        kq_sum *= sc;
                        pz *= sc;
                        for (int d = 0; d < D; ++d) acc[d] *= sc;
                    }
                    float pv_max = 0.0f;
                    for (int t = 0; t < count; ++t) {
                        e[t] = expf(kq[t] - nm);
                        kq_sum += e[t];
                        pz += e[t] * vz[t];
                        pv_max = std::max(pv_max, e[t] * vs[t]);
                    }
                    const float rvs = pv_max > 0.0f ? 255.0f / pv_max : 0.0f;
                    const float f = pv_max * (1.0f / 255.0f);
                    std::vector<uint8_t> u(count);
                    for (int t = 0; t < count; ++t) {
                        const float p_vs = e[t] * vs[t];
                        float v = rintf_even(p_vs * rvs);
                        v = std::max(0.0f, std::min(255.0f, v));
                        u[t] = (uint8_t) v;
                        pz += 7.5f * (p_vs - f * (float) u[t]);
                    }
                    for (int d = 0; d < D; ++d) {
                        int32_t vdot = 0;
                        for (int t = 0; t < count; ++t) {
                            const int tok = 16 * sl + t;
                            vdot += (int32_t) ggml_sj_kvarn::v_code(rec, ds.l, tok, d) * (int32_t) u[t];
                        }
                        acc[d] += f * (float) vdot;
                    }
                }

                // record_change(rec, next)
                const uint8_t * rec_next = (g + 1 < ds.n_groups) ? ds.body.data() + ((size_t) (g + 1) * ds.hkv + z_kv) * ds.l.bytes : nullptr;
                for (int d = 0; d < D; ++d) {
                    const float vo = vch[d];
                    const float vn = rec_next ? ggml_sj_kvarn::get_half(rec_next, ds.l.v_ch + 2 * ggml_sj_kvarn::v_ch_idx(d, D, ds.G, 4)) : 1.0f;
                    acc[d] = (acc[d] + pz) * (vo / vn);
                }
                pz = 0.0f;
            }

            // Exact strips (FP16 KQ + FP16 PV -> FP32 acc, matching fattn-sjkvarn-direct.cuh)
            const int n_ex_sink = ds.S / 16;
            const int n_ex_ring = (ds.n_kv_pad - ds.B) / 16;
            std::vector<float> kf(D), vf(D);
            for (int e_idx = 0; e_idx < n_ex_sink + n_ex_ring; ++e_idx) {
                const int p0 = e_idx < n_ex_sink ? 16 * e_idx : ds.B + 16 * (e_idx - n_ex_sink);
                float kq[16];
                float mx = -FLT_MAX / 2.0f;
                std::vector<std::vector<float>> v_strip(16, std::vector<float>(D, 0.0f));
                for (int t = 0; t < 16; ++t) {
                    const int p = p0 + t;
                    if (p < ds.N) {
                        if (p < ds.S) {
                            ggml_fp16_to_fp32_row(ds.sink_K.data() + ((size_t) p * ds.hkv + z_kv) * D, kf.data(), D);
                            ggml_fp16_to_fp32_row(ds.sink_V.data() + ((size_t) p * ds.hkv + z_kv) * D, vf.data(), D);
                        } else {
                            const int rp = p - ds.B;
                            for (int d = 0; d < D; ++d) {
                                kf[d] = fp16_round(ds.ring_K[((size_t) rp * ds.hkv + z_kv) * D + d]);
                                vf[d] = fp16_round(ds.ring_V[((size_t) rp * ds.hkv + z_kv) * D + d]);
                            }
                        }
                    } else {
                        std::fill(kf.begin(), kf.end(), 0.0f);
                        std::fill(vf.begin(), vf.end(), 0.0f);
                    }
                    float dot = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        dot += fp16_round(kf[d] * fp16_round(qrow[d] * scale));
                        v_strip[t][d] = vf[d];
                    }
                    kq[t] = dot + mrow[p];
                    mx = std::max(mx, kq[t]);
                }
                const float nm = std::max(kq_max, mx);
                if (nm != kq_max) {
                    const float sc = expf(kq_max - nm);
                    kq_max = nm;
                    kq_sum *= sc;
                    for (int d = 0; d < D; ++d) acc[d] *= sc;
                }
                for (int t = 0; t < 16; ++t) {
                    const float w = expf(kq[t] - nm);
                    kq_sum += w;
                    const float wh = fp16_round(w);
                    for (int d = 0; d < D; ++d) {
                        acc[d] += wh * v_strip[t][d];
                    }
                }
            }

            if (ds.has_sinks) {
                const float nm = std::max(kq_max, ds.sinks[h]);
                const float sc = expf(kq_max - nm);
                kq_max = nm;
                kq_sum = sc * kq_sum + expf(ds.sinks[h] - nm);
                for (int d = 0; d < D; ++d) acc[d] *= sc;
            }

            std::vector<float> of(D);
            const float inv = 1.0f / kq_sum;
            for (int d = 0; d < D; ++d) of[d] = acc[d] * inv;
            hadamard256_inplace(of.data());
            memcpy(out.data() + ((size_t) q * ds.nh + h) * D, of.data(), D * sizeof(float));
        }
    }
    return out;
}

// Verify against ggml-cpu's real ggml_compute_forward_flash_attn_ext_sj_kvarn
static std::vector<float> run_ggml_cpu_ref(const TestDataset & ds, float scale) {
    const size_t mem = 128ull * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds.D, ds.n_q, ds.nh, 1);
    for (int qi = 0; qi < ds.n_q; ++qi) {
        for (int hi = 0; hi < ds.nh; ++hi) {
            memcpy((float *) q->data + ((size_t) hi * ds.n_q + qi) * ds.D,
                   ds.Q.data() + ((size_t) qi * ds.nh + hi) * ds.D,
                   ds.D * sizeof(float));
        }
    }

    // Sink + cap ring with f16_sink sidecar
    const size_t packed_row = ggml_row_size(GGML_TYPE_TQ6_0, ds.D * ds.hkv);
    const int f16_sink_rows = (ds.S * ds.D * ds.hkv * 2 + (int) packed_row - 1) / (int) packed_row;
    const int alloc_rows = ds.S + ds.cap + f16_sink_rows;
    auto make_ring = [&](const std::vector<ggml_fp16_t> & sink, const std::vector<float> & ring) {
        ggml_tensor * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, ds.D * ds.hkv, alloc_rows);
        ggml_tensor * view = ggml_view_4d(ctx, storage, ds.D, alloc_rows, ds.hkv, 1,
                packed_row, ggml_row_size(GGML_TYPE_TQ6_0, ds.D), ggml_nbytes(storage), 0);
        memset(storage->data, 0, ggml_nbytes(storage));
        for (int p = 0; p < ds.S; ++p) {
            for (int h = 0; h < ds.hkv; ++h) {
                void * dst = (char *) storage->data + (size_t) (ds.S + ds.cap) * packed_row + (size_t) (p * ds.hkv + h) * ds.D * 2;
                memcpy(dst, sink.data() + ((size_t) p * ds.hkv + h) * ds.D, ds.D * 2);
            }
        }
        std::vector<float> tmp(ds.D);
        for (int p = ds.B; p < ds.N; ++p) {
            const int row = ds.S + (p - ds.S) % ds.cap;
            for (int h = 0; h < ds.hkv; ++h) {
                void * dst = (char *) storage->data + (size_t) row * packed_row + (size_t) h * ggml_row_size(GGML_TYPE_TQ6_0, ds.D);
                // Re-quantize from raw ring values (already quantized/dequantized, close to bit-identical)
                const float * src = ring.data() + ((size_t) (p - ds.B) * ds.hkv + h) * ds.D;
                quantize_row_tq6_0_rotated_ref(src, (block_tq6_0 *) dst, ds.D);
            }
        }
        return view;
    };
    ggml_tensor * kr = make_ring(ds.sink_K, ds.ring_K);
    ggml_tensor * vr = make_ring(ds.sink_V, ds.ring_V);

    const int m_heads = ds.per_head_mask ? ds.nh : 1;
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ds.n_kv_pad, ds.n_q, m_heads, 1);
    for (size_t i = 0; i < ds.mask.size(); ++i) {
        ((ggml_fp16_t *) m->data)[i] = GGML_FP32_TO_FP16(ds.mask[i]);
    }
    ggml_tensor * bt = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) ds.body.size());
    memcpy(bt->data, ds.body.data(), ds.body.size());
    ggml_tensor * desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_SJKVARN_DESC_N_ENTRIES);
    int32_t * d = (int32_t *) desc->data;
    memset(d, 0, GGML_SJKVARN_DESC_N_ENTRIES * 4);
    d[GGML_SJKVARN_DESC_S] = ds.S;
    d[GGML_SJKVARN_DESC_CAP] = ds.cap;
    d[GGML_SJKVARN_DESC_B] = ds.B;
    d[GGML_SJKVARN_DESC_N] = ds.N;
    d[GGML_SJKVARN_DESC_QPOS0] = ds.N - ds.n_q;
    d[GGML_SJKVARN_DESC_G] = ds.G;
    d[GGML_SJKVARN_DESC_D] = ds.D;
    d[GGML_SJKVARN_DESC_RECBYTES] = (int32_t) ds.l.bytes;
    d[GGML_SJKVARN_DESC_HKV] = ds.hkv;
    d[GGML_SJKVARN_DESC_B_OLD] = ds.B;
    d[GGML_SJKVARN_DESC_TYPE_K] = GGML_TYPE_TQ6_0;
    d[GGML_SJKVARN_DESC_TYPE_V] = GGML_TYPE_TQ6_0;
    d[GGML_SJKVARN_DESC_SINK_TYPE] = GGML_TYPE_F16;

    ggml_tensor * snk = nullptr;
    if (ds.has_sinks) {
        snk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ds.nh);
        memcpy(snk->data, ds.sinks.data(), ds.nh * sizeof(float));
    }

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, kr, vr, m, scale, 0.0f, 0.0f);
    if (snk) ggml_flash_attn_ext_add_sinks(out, snk);
    ggml_flash_attn_ext_set_sj_kvarn(out, bt, desc, 4, 4, ds.n_kv_pad);
    if (ds.fused_rot) ggml_flash_attn_ext_set_sj_kvarn_rot(out, 256);
    ggml_prec_set_acc(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    // Output layout of ggml_flash_attn_ext is [D, nh, n_q]
    std::vector<float> res((size_t) ds.n_q * ds.nh * ds.D);
    memcpy(res.data(), out->data, res.size() * sizeof(float));
    if (!ds.fused_rot) {
        for (int qi = 0; qi < ds.n_q; ++qi) {
            for (int hi = 0; hi < ds.nh; ++hi) {
                hadamard256_inplace(res.data() + ((size_t) qi * ds.nh + hi) * ds.D);
            }
        }
    }
    ggml_free(ctx);
    return res;
}

} // namespace

int main() {
    struct Case {
        const char * name;
        int n_q;
        bool d128_packed;
        bool outl100x;
        bool fused_rot;
        bool has_sinks;
        bool per_head_mask;
    };

    const Case cases[] = {
        { "D256_normal_T5",            5, false, false, true,  false, false },
        { "D256_Outl100x_T5",          5, false, true,  true,  false, false },
        { "D128packed_Outl100x_T4",    4, true,  true,  true,  false, false },
        { "D128packed_Outl100x_T5",    5, true,  true,  true,  false, false },
        { "D128packed_Outl100x_T6",    6, true,  true,  true,  false, false },
        { "D128packed_Outl100x_T8",    8, true,  true,  true,  false, false },
        { "D128packed_Outl100x_T16",  16, true,  true,  true,  false, false },
        { "D128packed_Outl100x_T32",  32, true,  true,  true,  false, false },
        { "D128packed_SinksRelBias_T5",5, true,  true,  true,  true,  true  },
    };

    int failures = 0;
    const float scale = 1.0f / std::sqrt(128.0f);

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const Case & c = cases[i];
        const TestDataset ds = generate_dataset(42u + (uint32_t) i, c.n_q, c.d128_packed, c.outl100x, c.fused_rot, c.has_sinks, c.per_head_mask);
        const std::vector<double> ref64 = run_fp64_reference(ds, scale);
        const std::vector<float> cpu_ref = run_ggml_cpu_ref(ds, scale);
        const std::vector<float> pv16    = run_stream_pv16_sim(ds, scale);
        const std::vector<float> dir16   = run_direct_verify_sim(ds, scale, false);
        const std::vector<float> dir32   = run_direct_verify_sim(ds, scale, true);

        const Metrics m_cpu = compute_metrics(cpu_ref, ref64);
        const Metrics m_pv  = compute_metrics(pv16, ref64);
        const Metrics m_d16 = compute_metrics(dir16, ref64);
        const Metrics m_d32 = compute_metrics(dir32, ref64);

        printf("[%s]\n", c.name);
        printf("  ggml_cpu_ref   : nmse=%.3e sqnr=%.2f dB cos=%.6f maxabs=%.3e\n", m_cpu.nmse, m_cpu.sqnr_db, m_cpu.cos_sim, m_cpu.max_abs);
        printf("  STREAM_PV16    : nmse=%.3e sqnr=%.2f dB cos=%.6f maxabs=%.3e\n", m_pv.nmse, m_pv.sqnr_db, m_pv.cos_sim, m_pv.max_abs);
        printf("  DIRECT_U8U8_16 : nmse=%.3e sqnr=%.2f dB cos=%.6f maxabs=%.3e\n", m_d16.nmse, m_d16.sqnr_db, m_d16.cos_sim, m_d16.max_abs);
        printf("  DIRECT_U8U8_32 : nmse=%.3e sqnr=%.2f dB cos=%.6f maxabs=%.3e\n", m_d32.nmse, m_d32.sqnr_db, m_d32.cos_sim, m_d32.max_abs);

        if (m_cpu.nmse > 1e-4 || m_pv.nmse > 1e-4 || m_d16.nmse > 2.5e-3 || m_d32.nmse > std::max(2.5e-3, m_d16.nmse * 1.1) ||
            m_pv.cos_sim < 0.9999 || m_d16.cos_sim < 0.999 || m_d32.cos_sim < 0.999) {
            fprintf(stderr, "FAIL: accuracy bound exceeded on %s\n", c.name);
            ++failures;
        }
    }

    if (failures == 0) {
        printf("ALL %zu CPU VERIFICATION CASES PASSED!\n", sizeof(cases) / sizeof(cases[0]));
    }
    return failures == 0 ? 0 : 1;
}
