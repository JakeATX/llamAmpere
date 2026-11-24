// Standalone real-data, L2-flushed CUDA benchmark & numerical parity suite for SJ-KVaRN 4/4 verify paths:
//   - Baseline       : default ggml_cuda_flash_attn_ext (env switches OFF/0 -> sjkvarn_stream / rowsplit / tile)
//   - STREAM_PV16    : spill-free HMMA.16816.F16 (142T) stream kernel (fattn-sjkvarn-stream-pv16.cuh)
//   - DIRECT_VERIFY  : INT8 IMMA.16832.U8.S8 (285T) + IMMA.16832.U8.U8 (285T) verify kernel (fattn-sjkvarn-direct-verify.cuh)
//
// Builds on genuine ggml_sj_kvarn::seal_group 4/4 sealed records with Outl100x channels, H256 rotation, TQ6_0 ring,
// and F16 independent sink sidecar (Inkling D=128 pair-packed -> D=256, H_q=24, H_kv=4, GQA=6).
//
// Usage:
//   bench-sjkvarn-verify [--selfcheck-only] [--iters 20] [--warmup 5] [--nkv 32768,100096]

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml-quants.h"
#include "ggml-sjkvarn.h"

#include "../ggml/src/ggml-cuda/common.cuh"
#include "../ggml/src/ggml-cuda/fattn.cuh"
#include "../ggml/src/ggml-cuda/fattn-sjkvarn-rot.cuh"
#include "../ggml/src/ggml-cuda/fattn-sjkvarn-verify.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CK(call) do { \
    const cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1); \
    } \
} while (0)

namespace {

static void hadamard256_inplace(float * x) {
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

__global__ static void k_flush_l2(uint32_t * __restrict__ buf, const size_t n_words, const uint32_t seed) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    uint32_t acc = seed;
    for (size_t i = idx; i < n_words; i += stride) {
        const uint32_t v = buf[i];
        acc ^= v + 0x9e3779b9u;
        buf[i] = acc;
    }
}

struct Metrics {
    double nmse = 0.0;
    double max_abs = 0.0;
    double cos_sim = 1.0;
    double sqnr_db = 120.0;
};

static Metrics compute_metrics(const std::vector<float> & test, const std::vector<float> & ref) {
    double se = 0.0, sref = 0.0, stest = 0.0, dot = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = test[i];
        const double b = ref[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            fprintf(stderr, "NON-FINITE value at index %zu (test=%g ref=%g)\n", i, a, b);
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

struct HostDataset {
    int D = 256;
    int G = 128;
    int S = 128;
    int cap = 256;
    int n_groups = 0;
    int B = 0;
    int N = 0;
    int n_kv_pad = 0;
    int hkv = 4;
    int nh = 24;
    int gqa = 6;
    int n_q = 5;
    bool fused_rot = true;
    bool has_sinks = false;
    bool per_head_mask = false;
    ggml_sj_kvarn::layout l;

    std::vector<float> Q;                // [D, nh, n_q]
    std::vector<uint8_t> body;           // [n_groups, hkv, l.bytes]
    std::vector<uint8_t> k_ring_storage; // [alloc_rows * packed_row]
    std::vector<uint8_t> v_ring_storage;
    size_t packed_row = 0;
    int alloc_rows = 0;
    std::vector<ggml_fp16_t> mask;       // [n_kv_pad, n_q, m_heads]
    std::vector<float> sinks;            // [nh]
};

static HostDataset generate_dataset(uint32_t seed, int n_kv_pad_target, int n_q_max,
                                    bool outl100x, bool fused_rot, bool has_sinks, bool per_head_mask) {
    HostDataset ds;
    ds.n_q = n_q_max;
    ds.fused_rot = fused_rot;
    ds.has_sinks = has_sinks;
    ds.per_head_mask = per_head_mask;
    ds.n_kv_pad = ((n_kv_pad_target + 255) / 256) * 256;
    // Choose B = n_kv_pad - 256, N = B + 128 (sink S=128 + body n_groups*128 + 128 ring rows)
    ds.n_groups = std::max(1, (ds.n_kv_pad - ds.S - ds.cap) / ds.G);
    ds.B = ds.S + ds.n_groups * ds.G;
    ds.N = ds.B + 128;
    ds.l = ggml_sj_kvarn::make_layout(ds.D, ds.G, 4, 4);

    // Prepare persistent outlier gains
    std::vector<float> k_gain(ds.hkv * ds.D, 1.0f), v_gain(ds.hkv * ds.D, 1.0f);
    if (outl100x) {
        for (int h = 0; h < ds.hkv; ++h) {
            for (int d = 0; d < ds.D; ++d) {
                if ((d % 29) == (h * 7) % 29) k_gain[h * ds.D + d] = 100.0f;
                if ((d % 31) == (h * 5 + 3) % 31) v_gain[h * ds.D + d] = 100.0f;
            }
        }
    }

    ds.body.resize((size_t) ds.n_groups * ds.hkv * ds.l.bytes);

    // Parallel sealing per group so 100K dataset builds fast with zero repeated records
    const int n_threads = std::min<int>(std::max<int>(1, std::thread::hardware_concurrency()), 8);
    auto seal_worker = [&](const int tid) {
        std::vector<ggml_fp16_t> g_k((size_t) ds.G * ds.hkv * ds.D);
        std::vector<ggml_fp16_t> g_v((size_t) ds.G * ds.hkv * ds.D);
        std::vector<float> rk(ds.D), rv(ds.D);
        for (int g = tid; g < ds.n_groups; g += n_threads) {
            std::mt19937 grng(seed + 10007u * (uint32_t) (g + 1));
            std::normal_distribution<float> gnorm(0.0f, 1.0f);
            for (int t = 0; t < ds.G; ++t) {
                const int p = ds.S + g * ds.G + t;
                const float tok_scale = (p % 97 == 13) ? 12.0f : (p % 53 == 7) ? 0.15f : 1.0f;
                for (int h = 0; h < ds.hkv; ++h) {
                    for (int d = 0; d < ds.D; ++d) {
                        rk[d] = gnorm(grng) * k_gain[h * ds.D + d] * 0.25f * tok_scale;
                        rv[d] = gnorm(grng) * v_gain[h * ds.D + d] * 0.25f * tok_scale;
                    }
                    hadamard256_inplace(rk.data());
                    hadamard256_inplace(rv.data());
                    ggml_fp16_t * dk = g_k.data() + ((size_t) t * ds.hkv + h) * ds.D;
                    ggml_fp16_t * dv = g_v.data() + ((size_t) t * ds.hkv + h) * ds.D;
                    for (int d = 0; d < ds.D; ++d) {
                        dk[d] = GGML_FP32_TO_FP16(rk[d]);
                        dv[d] = GGML_FP32_TO_FP16(rv[d]);
                    }
                }
            }
            for (int h = 0; h < ds.hkv; ++h) {
                uint8_t * rec = ds.body.data() + ((size_t) g * ds.hkv + h) * ds.l.bytes;
                ggml_sj_kvarn::seal_group(g_k.data() + (size_t) h * ds.D,
                                          g_v.data() + (size_t) h * ds.D,
                                          (size_t) ds.hkv * ds.D, ds.l, 16, rec, false);
            }
        }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
        threads.emplace_back(seal_worker, t);
    }
    for (auto & th : threads) {
        th.join();
    }

    // Ring & Sink
    ds.packed_row = ggml_row_size(GGML_TYPE_TQ6_0, ds.D * ds.hkv);
    const int f16_sink_rows = (ds.S * ds.D * ds.hkv * 2 + (int) ds.packed_row - 1) / (int) ds.packed_row;
    ds.alloc_rows = ds.S + ds.cap + f16_sink_rows;
    ds.k_ring_storage.assign((size_t) ds.alloc_rows * ds.packed_row, 0);
    ds.v_ring_storage.assign((size_t) ds.alloc_rows * ds.packed_row, 0);

    std::mt19937 rng(seed ^ 0xA5A55A5Au);
    std::normal_distribution<float> norm(0.0f, 1.0f);
    std::vector<float> rk(ds.D), rv(ds.D);

    // F16 sink sidecar [0, S)
    for (int p = 0; p < ds.S; ++p) {
        for (int h = 0; h < ds.hkv; ++h) {
            for (int d = 0; d < ds.D; ++d) {
                rk[d] = norm(rng) * k_gain[h * ds.D + d] * 0.25f;
                rv[d] = norm(rng) * v_gain[h * ds.D + d] * 0.25f;
            }
            hadamard256_inplace(rk.data());
            hadamard256_inplace(rv.data());
            ggml_fp16_t * dk = (ggml_fp16_t *) (ds.k_ring_storage.data() + (size_t) (ds.S + ds.cap) * ds.packed_row + (size_t) (p * ds.hkv + h) * ds.D * 2);
            ggml_fp16_t * dv = (ggml_fp16_t *) (ds.v_ring_storage.data() + (size_t) (ds.S + ds.cap) * ds.packed_row + (size_t) (p * ds.hkv + h) * ds.D * 2);
            for (int d = 0; d < ds.D; ++d) {
                dk[d] = GGML_FP32_TO_FP16(rk[d]);
                dv[d] = GGML_FP32_TO_FP16(rv[d]);
            }
        }
    }
    // TQ6_0 ring [B, N)
    const size_t tq6_head_bytes = ggml_row_size(GGML_TYPE_TQ6_0, ds.D);
    for (int p = ds.B; p < ds.N; ++p) {
        const int row = ds.S + (p - ds.S) % ds.cap;
        for (int h = 0; h < ds.hkv; ++h) {
            for (int d = 0; d < ds.D; ++d) {
                rk[d] = norm(rng) * k_gain[h * ds.D + d] * 0.25f;
                rv[d] = norm(rng) * v_gain[h * ds.D + d] * 0.25f;
            }
            hadamard256_inplace(rk.data());
            hadamard256_inplace(rv.data());
            quantize_row_tq6_0_rotated_ref(rk.data(), (block_tq6_0 *) (ds.k_ring_storage.data() + (size_t) row * ds.packed_row + (size_t) h * tq6_head_bytes), ds.D);
            quantize_row_tq6_0_rotated_ref(rv.data(), (block_tq6_0 *) (ds.v_ring_storage.data() + (size_t) row * ds.packed_row + (size_t) h * tq6_head_bytes), ds.D);
        }
    }

    // Q [D, nh, n_q] (Inkling pair-packed D=128 into 256)
    ds.Q.assign((size_t) ds.n_q * ds.nh * ds.D, 0.0f);
    for (int q = 0; q < ds.n_q; ++q) {
        for (int h = 0; h < ds.nh; ++h) {
            const int offset = ((h % ds.gqa) >= ds.gqa / 2) ? 128 : 0;
            float * dst_q = ds.Q.data() + ((size_t) q * ds.nh + h) * ds.D;
            for (int d = 0; d < 128; ++d) {
                dst_q[offset + d] = norm(rng) * 0.5f;
            }
            if (!fused_rot) {
                hadamard256_inplace(dst_q);
            }
        }
    }

    // F16 mask [n_kv_pad, n_q, m_heads]
    const int m_heads = per_head_mask ? ds.nh : 1;
    ds.mask.resize((size_t) m_heads * ds.n_q * ds.n_kv_pad, GGML_FP32_TO_FP16(-INFINITY));
    for (int mh = 0; mh < m_heads; ++mh) {
        for (int q = 0; q < ds.n_q; ++q) {
            const int vis_end = ds.N - (ds.n_q - 1 - q);
            for (int p = 0; p < vis_end; ++p) {
                float bias = 0.0f;
                if (per_head_mask) {
                    bias = -0.0005f * float(vis_end - 1 - p) * (1.0f + 0.1f * (mh % 6));
                }
                ds.mask[((size_t) mh * ds.n_q + q) * ds.n_kv_pad + p] = GGML_FP32_TO_FP16(bias);
            }
        }
    }

    ds.sinks.resize(ds.nh, 0.0f);
    if (has_sinks) {
        for (int h = 0; h < ds.nh; ++h) {
            ds.sinks[h] = -1.5f + 0.25f * float(h % 8);
        }
    }
    return ds;
}

static std::vector<float> run_cpu_ref(const HostDataset & ds, const int n_q, const float scale) {
    const size_t mem = 256ull * 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds.D, n_q, ds.nh, 1);
    memcpy(q->data, ds.Q.data(), (size_t) n_q * ds.nh * ds.D * sizeof(float));

    ggml_tensor * ks = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, ds.D * ds.hkv, ds.alloc_rows);
    ggml_tensor * vs = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, ds.D * ds.hkv, ds.alloc_rows);
    memcpy(ks->data, ds.k_ring_storage.data(), ds.k_ring_storage.size());
    memcpy(vs->data, ds.v_ring_storage.data(), ds.v_ring_storage.size());
    const size_t tq6_head_bytes = ggml_row_size(GGML_TYPE_TQ6_0, ds.D);
    ggml_tensor * kr = ggml_view_4d(ctx, ks, ds.D, ds.alloc_rows, ds.hkv, 1, ds.packed_row, tq6_head_bytes, ggml_nbytes(ks), 0);
    ggml_tensor * vr = ggml_view_4d(ctx, vs, ds.D, ds.alloc_rows, ds.hkv, 1, ds.packed_row, tq6_head_bytes, ggml_nbytes(vs), 0);

    const int m_heads = ds.per_head_mask ? ds.nh : 1;
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ds.n_kv_pad, n_q, m_heads, 1);
    for (int mh = 0; mh < m_heads; ++mh) {
        memcpy((ggml_fp16_t *) m->data + (size_t) mh * n_q * ds.n_kv_pad,
               ds.mask.data() + (size_t) mh * ds.n_q * ds.n_kv_pad,
               (size_t) n_q * ds.n_kv_pad * sizeof(ggml_fp16_t));
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
    d[GGML_SJKVARN_DESC_QPOS0] = ds.N - n_q;
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

    std::vector<float> res((size_t) n_q * ds.nh * ds.D);
    memcpy(res.data(), out->data, res.size() * sizeof(float));
    ggml_free(ctx);
    return res;
}

struct CudaFixture {
    ggml_backend_t backend = nullptr;
    ggml_backend_cuda_context * cuda_ctx = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * ks = nullptr;
    ggml_tensor * vs = nullptr;
    ggml_tensor * kr = nullptr;
    ggml_tensor * vr = nullptr;
    ggml_tensor * m = nullptr;
    ggml_tensor * bt = nullptr;
    ggml_tensor * desc = nullptr;
    ggml_tensor * snk = nullptr;
    ggml_tensor * out = nullptr;
    const HostDataset * ds = nullptr;
    float scale = 1.0f / std::sqrt(128.0f);

    void init(ggml_backend_t be, const HostDataset & dataset) {
        backend = be;
        cuda_ctx = (ggml_backend_cuda_context *) be->context;
        ds = &dataset;
        ggml_init_params ip = { 32 * ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);

        q  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ds->D, ds->n_q, ds->nh, 1);
        ks = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, ds->D * ds->hkv, ds->alloc_rows);
        vs = ggml_new_tensor_2d(ctx, GGML_TYPE_TQ6_0, ds->D * ds->hkv, ds->alloc_rows);
        const size_t tq6_head_bytes = ggml_row_size(GGML_TYPE_TQ6_0, ds->D);
        kr = ggml_view_4d(ctx, ks, ds->D, ds->alloc_rows, ds->hkv, 1, ds->packed_row, tq6_head_bytes, ggml_nbytes(ks), 0);
        vr = ggml_view_4d(ctx, vs, ds->D, ds->alloc_rows, ds->hkv, 1, ds->packed_row, tq6_head_bytes, ggml_nbytes(vs), 0);

        const int m_heads = ds->per_head_mask ? ds->nh : 1;
        m    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ds->n_kv_pad, ds->n_q, m_heads, 1);
        bt   = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) ds->body.size());
        desc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, GGML_SJKVARN_DESC_N_ENTRIES);
        if (ds->has_sinks) {
            snk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ds->nh);
        }
        out = ggml_flash_attn_ext(ctx, q, kr, vr, m, scale, 0.0f, 0.0f);
        if (snk) ggml_flash_attn_ext_add_sinks(out, snk);
        ggml_flash_attn_ext_set_sj_kvarn(out, bt, desc, 4, 4, ds->n_kv_pad);
        if (ds->fused_rot) ggml_flash_attn_ext_set_sj_kvarn_rot(out, 256);
        ggml_prec_set_acc(out, GGML_PREC_F32);

        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buf != nullptr);

        ggml_backend_tensor_set(q,  ds->Q.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(ks, ds->k_ring_storage.data(), 0, ds->k_ring_storage.size());
        ggml_backend_tensor_set(vs, ds->v_ring_storage.data(), 0, ds->v_ring_storage.size());
        ggml_backend_tensor_set(m,  ds->mask.data(), 0, ggml_nbytes(m));
        ggml_backend_tensor_set(bt, ds->body.data(), 0, ds->body.size());
        if (snk) {
            ggml_backend_tensor_set(snk, ds->sinks.data(), 0, ggml_nbytes(snk));
        }
        int32_t d[GGML_SJKVARN_DESC_N_ENTRIES] = {};
        d[GGML_SJKVARN_DESC_S] = ds->S;
        d[GGML_SJKVARN_DESC_CAP] = ds->cap;
        d[GGML_SJKVARN_DESC_B] = ds->B;
        d[GGML_SJKVARN_DESC_N] = ds->N;
        d[GGML_SJKVARN_DESC_QPOS0] = ds->N - ds->n_q;
        d[GGML_SJKVARN_DESC_G] = ds->G;
        d[GGML_SJKVARN_DESC_D] = ds->D;
        d[GGML_SJKVARN_DESC_RECBYTES] = (int32_t) ds->l.bytes;
        d[GGML_SJKVARN_DESC_HKV] = ds->hkv;
        d[GGML_SJKVARN_DESC_B_OLD] = ds->B;
        d[GGML_SJKVARN_DESC_TYPE_K] = GGML_TYPE_TQ6_0;
        d[GGML_SJKVARN_DESC_TYPE_V] = GGML_TYPE_TQ6_0;
        d[GGML_SJKVARN_DESC_SINK_TYPE] = GGML_TYPE_F16;
        ggml_backend_tensor_set(desc, d, 0, sizeof(d));
    }

    void destroy() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        buf = nullptr; ctx = nullptr;
    }
};

enum class BenchMode { BASELINE, STREAM_PV16, DIRECT_VERIFY };

static bool invoke_mode(ggml_backend_cuda_context & cctx, ggml_tensor * dst, const BenchMode mode) {
    if (mode == BenchMode::BASELINE) {
        ggml_cuda_flash_attn_ext(cctx, dst);
        return true;
    }
    if (mode == BenchMode::STREAM_PV16) {
        const int n_q = (int) dst->src[0]->ne[1];
        if (n_q <= 8) {
            ggml_cuda_flash_attn_ext_sj_kvarn_stream_pv16_run(cctx, dst);
            return true;
        }
        if (n_q <= 16) {
            for (int r0 = 0; r0 < n_q; r0 += 8) {
                const int nr = std::min(8, n_q - r0);
                ggml_tensor qv = *dst->src[0];
                ggml_tensor mv = *dst->src[3];
                ggml_tensor dv = *dst;
                qv.ne[1] = nr;
                qv.data = (char *) dst->src[0]->data + (size_t) r0 * dst->src[0]->nb[1];
                mv.ne[1] = dst->src[3]->ne[1] - r0;
                mv.data = (char *) dst->src[3]->data + (size_t) r0 * dst->src[3]->nb[1];
                dv.ne[2] = nr;
                dv.data = (char *) dst->data + (size_t) r0 * dst->nb[2];
                dv.src[0] = &qv;
                dv.src[3] = &mv;
                ggml_cuda_flash_attn_ext_sj_kvarn_stream_pv16_run(cctx, &dv);
            }
            return true;
        }
        return false; // > 16 not covered by stream-pv16
    }
    ggml_cuda_flash_attn_ext_sj_kvarn_direct_verify_run(cctx, dst);
    return true;
}

static std::vector<float> run_cuda_slice(CudaFixture & fix, const HostDataset & ds, const int n_q, const BenchMode mode) {
    // Re-upload compact n_q slice of Q and mask so nb[2] matches [D, n_q, nh, 1] contiguously
    ggml_backend_tensor_set(fix.q, ds.Q.data(), 0, (size_t) n_q * ds.nh * ds.D * sizeof(float));
    const int m_heads = ds.per_head_mask ? ds.nh : 1;
    std::vector<ggml_fp16_t> mslice((size_t) m_heads * n_q * ds.n_kv_pad);
    for (int mh = 0; mh < m_heads; ++mh) {
        memcpy(mslice.data() + (size_t) mh * n_q * ds.n_kv_pad,
               ds.mask.data() + (size_t) mh * ds.n_q * ds.n_kv_pad,
               (size_t) n_q * ds.n_kv_pad * sizeof(ggml_fp16_t));
    }
    ggml_backend_tensor_set(fix.m, mslice.data(), 0, mslice.size() * sizeof(ggml_fp16_t));

    ggml_tensor qv = *fix.q;
    qv.ne[1] = n_q;
    qv.nb[2] = qv.nb[1] * n_q;
    qv.nb[3] = qv.nb[2] * qv.ne[2];
    ggml_tensor mv = *fix.m;
    mv.ne[1] = n_q;
    mv.nb[2] = mv.nb[1] * n_q;
    mv.nb[3] = mv.nb[2] * mv.ne[2];
    ggml_tensor dv = *fix.out;
    dv.ne[2] = n_q;
    dv.nb[3] = dv.nb[2] * n_q;
    dv.src[0] = &qv;
    dv.src[3] = &mv;

    // Wait: fix.Q was stored in [row, head] layout [q * nh + h] above, whereas ggml FA Q layout is [D, n_q, nh, 1] -> [head * n_q + row]!
    // CHECK THIS!! In test-sjkvarn-verify-cpu.cpp: let's match exact [D, n_q, nh, 1] layout in Q!
    std::vector<float> q_ggml((size_t) ds.nh * n_q * ds.D);
    for (int q = 0; q < n_q; ++q) {
        for (int h = 0; h < ds.nh; ++h) {
            memcpy(q_ggml.data() + ((size_t) h * n_q + q) * ds.D,
                   ds.Q.data() + ((size_t) q * ds.nh + h) * ds.D,
                   ds.D * sizeof(float));
        }
    }
    ggml_backend_tensor_set(fix.q, q_ggml.data(), 0, q_ggml.size() * sizeof(float));
    CK(cudaMemsetAsync(fix.out->data, 0, ggml_nbytes(fix.out), fix.cuda_ctx->stream()));

    const bool ok = invoke_mode(*fix.cuda_ctx, &dv, mode);
    CK(cudaStreamSynchronize(fix.cuda_ctx->stream()));
    std::vector<float> res;
    if (!ok) return res;
    res.resize((size_t) n_q * ds.nh * ds.D);
    ggml_backend_tensor_get(fix.out, res.data(), 0, res.size() * sizeof(float));
    return res;
}

static double bench_mode(CudaFixture & fix, ggml_tensor * dst, const BenchMode mode,
                         uint32_t * d_l2, const size_t l2_words, const int warmup, const int iters) {
    cudaStream_t stream = fix.cuda_ctx->stream();
    cudaEvent_t ev0, ev1;
    CK(cudaEventCreate(&ev0));
    CK(cudaEventCreate(&ev1));
    for (int i = 0; i < warmup; ++i) {
        k_flush_l2<<<656, 256, 0, stream>>>(d_l2, l2_words, (uint32_t) i);
        if (!invoke_mode(*fix.cuda_ctx, dst, mode)) {
            CK(cudaEventDestroy(ev0));
            CK(cudaEventDestroy(ev1));
            return -1.0;
        }
    }
    CK(cudaStreamSynchronize(stream));
    std::vector<double> times(iters);
    for (int i = 0; i < iters; ++i) {
        k_flush_l2<<<656, 256, 0, stream>>>(d_l2, l2_words, (uint32_t) (warmup + i));
        CK(cudaEventRecord(ev0, stream));
        invoke_mode(*fix.cuda_ctx, dst, mode);
        CK(cudaEventRecord(ev1, stream));
        CK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, ev0, ev1));
        times[i] = (double) ms;
    }
    CK(cudaEventDestroy(ev0));
    CK(cudaEventDestroy(ev1));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2]; // median ms
}

} // namespace

int main(int argc, char ** argv) {
    bool selfcheck_only = false;
    bool env_dispatch = false;
    int warmup = 5;
    int iters = 20;
    std::vector<int> nkv_list = { 32768, 100096 };
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--selfcheck-only") == 0) selfcheck_only = true;
        else if (strcmp(argv[i], "--env-dispatch") == 0) env_dispatch = true;
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--nkv") == 0 && i + 1 < argc) {
            nkv_list.clear();
            const char * s = argv[++i];
            while (*s) {
                char * end = nullptr;
                const long v = strtol(s, &end, 10);
                if (v > 0) nkv_list.push_back((int) v);
                s = (*end == ',') ? end + 1 : end;
            }
        }
    }

    if (!env_dispatch) {
        // Ensure default env flags are OFF so BASELINE runs clean unmodified sjkvarn_stream!
        unsetenv("GGML_SJKVARN_STREAM_PV16");
        unsetenv("GGML_SJKVARN_DIRECT_VERIFY");
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "Failed to initialize CUDA backend 0\n");
        return 1;
    }

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs, smpbo=%.1f KB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.sharedMemPerBlockOptin / 1024.0);

    // 1. Numerical Self-Check vs ggml-cpu reference on N_kv=4096 Outl100x
    printf("\n=== Numerical Parity Self-Check (N_kv=4096, D128 packed->256, Outl100x, Rot256) ===\n");
    int failures = 0;
    {
        const HostDataset ds = generate_dataset(777u, 4096, 32, true, true, false, false);
        CudaFixture fix;
        fix.init(backend, ds);
        const int Ts[] = { 1, 4, 5, 6, 8, 16, 32 };
        for (int T : Ts) {
            // Prepare ds copy with n_q = T and [head * T + q] Q layout for CPU ref too
            std::vector<float> q_save = ds.Q;
            // Notice: CPU ref reads Q as [D, n_q, nh, 1] -> offset [h * n_q + q]!
            HostDataset ds_t = ds;
            ds_t.n_q = T;
            ds_t.Q.resize((size_t) ds.nh * T * ds.D);
            ds_t.mask.resize((size_t) T * ds.n_kv_pad);
            for (int q = 0; q < T; ++q) {
                for (int h = 0; h < ds.nh; ++h) {
                    memcpy(ds_t.Q.data() + ((size_t) h * T + q) * ds.D,
                           q_save.data() + ((size_t) q * ds.nh + h) * ds.D, ds.D * sizeof(float));
                }
                memcpy(ds_t.mask.data() + (size_t) q * ds.n_kv_pad,
                       ds.mask.data() + (size_t) q * ds.n_kv_pad, ds.n_kv_pad * sizeof(ggml_fp16_t));
            }
            const std::vector<float> ref = run_cpu_ref(ds_t, T, 1.0f / std::sqrt(128.0f));
            // For CUDA run_cuda_slice: restore ds.Q layout or use ds_t directly!
            if (env_dispatch) {
                const std::vector<float> env_res = run_cuda_slice(fix, ds, T, BenchMode::BASELINE);
                const Metrics me = compute_metrics(env_res, ref);
                printf("T=%2d  ENV_DISPATCH  nmse=%.2e sqnr=%.1fdB cos=%.6f maxabs=%.2e\n", T, me.nmse, me.sqnr_db, me.cos_sim, me.max_abs);
                if (me.nmse > 1e-3 || me.cos_sim < 0.999) ++failures;
                continue;
            }
            const std::vector<float> base = run_cuda_slice(fix, ds, T, BenchMode::BASELINE);
            const std::vector<float> pv16 = run_cuda_slice(fix, ds, T, BenchMode::STREAM_PV16);
            const std::vector<float> dirv = run_cuda_slice(fix, ds, T, BenchMode::DIRECT_VERIFY);

            const Metrics mb = compute_metrics(base, ref);
            const Metrics md = compute_metrics(dirv, ref);
            printf("T=%2d  BASE  nmse=%.2e sqnr=%.1fdB cos=%.6f", T, mb.nmse, mb.sqnr_db, mb.cos_sim);
            if (mb.nmse > 1e-4 || mb.cos_sim < 0.9999) ++failures;
            if (!pv16.empty()) {
                const Metrics mp = compute_metrics(pv16, ref);
                printf(" | PV16  nmse=%.2e sqnr=%.1fdB cos=%.6f", mp.nmse, mp.sqnr_db, mp.cos_sim);
                if (mp.nmse > 1e-4 || mp.cos_sim < 0.9999) ++failures;
            }
            printf(" | DIRECT_VERIFY nmse=%.2e sqnr=%.1fdB cos=%.6f\n", md.nmse, md.sqnr_db, md.cos_sim);
            if (md.nmse > 2.5e-3 || md.cos_sim < 0.999) ++failures;
        }
        fix.destroy();
    }

    // Also verify Sinks + Per-Head RelBias Mask (T=5)
    {
        const HostDataset ds = generate_dataset(999u, 4096, 5, true, true, true, true);
        CudaFixture fix;
        fix.init(backend, ds);
        HostDataset ds_t = ds;
        ds_t.Q.resize((size_t) ds.nh * 5 * ds.D);
        for (int q = 0; q < 5; ++q) {
            for (int h = 0; h < ds.nh; ++h) {
                memcpy(ds_t.Q.data() + ((size_t) h * 5 + q) * ds.D,
                       ds.Q.data() + ((size_t) q * ds.nh + h) * ds.D, ds.D * sizeof(float));
            }
        }
        const std::vector<float> ref = run_cpu_ref(ds_t, 5, 1.0f / std::sqrt(128.0f));
        if (env_dispatch) {
            const std::vector<float> env_res = run_cuda_slice(fix, ds, 5, BenchMode::BASELINE);
            const Metrics me = compute_metrics(env_res, ref);
            printf("T= 5 (Sinks+PerHeadMask) | ENV_DISPATCH nmse=%.2e sqnr=%.1fdB cos=%.6f\n", me.nmse, me.sqnr_db, me.cos_sim);
            if (me.nmse > 2.5e-3 || me.cos_sim < 0.999) ++failures;
        } else {
            const std::vector<float> pv16 = run_cuda_slice(fix, ds, 5, BenchMode::STREAM_PV16);
            const std::vector<float> dirv = run_cuda_slice(fix, ds, 5, BenchMode::DIRECT_VERIFY);
            const Metrics mp = compute_metrics(pv16, ref);
            const Metrics md = compute_metrics(dirv, ref);
            printf("T= 5 (Sinks+PerHeadMask) | PV16 nmse=%.2e sqnr=%.1fdB | DIRECT_VERIFY nmse=%.2e sqnr=%.1fdB\n",
                   mp.nmse, mp.sqnr_db, md.nmse, md.sqnr_db);
            if (mp.nmse > 1e-4 || md.nmse > 2.5e-3 || mp.cos_sim < 0.9999 || md.cos_sim < 0.999) ++failures;
        }
        fix.destroy();
    }
    if (failures > 0) {
        fprintf(stderr, "SELF-CHECK FAILED (%d failures)\n", failures);
        ggml_backend_free(backend);
        return 1;
    }
    printf("ALL NUMERICAL PARITY CHECKS PASSED!\n");
    if (selfcheck_only) {
        ggml_backend_free(backend);
        return 0;
    }

    // 2. L2-Flushed Benchmarks across N_kv and T = 1, 4, 5, 6, 8, 16, 32
    const size_t l2_bytes = 64ull * 1024 * 1024;
    const size_t l2_words = l2_bytes / sizeof(uint32_t);
    uint32_t * d_l2 = nullptr;
    CK(cudaMalloc(&d_l2, l2_bytes));
    CK(cudaMemset(d_l2, 0x5A, l2_bytes));

    const int Ts[] = { 1, 4, 5, 6, 8, 16, 32 };
    for (int nkv : nkv_list) {
        printf("\n=== L2-Flushed Benchmark: N_kv=%d (D=128 packed->256, H_q=24, H_kv=4, GQA=6, Outl100x, Rot256, iters=%d) ===\n",
               nkv, iters);
        printf("  T |  BASE (ms) | PV16 (ms) | PV16 vs B | DIRECT_V (ms) | DIR vs B | DIR TFLOPS | Approx BW\n");
        printf("----+------------+-----------+-----------+---------------+----------+------------+----------\n");

        const HostDataset ds = generate_dataset(4242u + (uint32_t) nkv, nkv, 32, true, true, false, false);
        CudaFixture fix;
        fix.init(backend, ds);
        const double bytes_kv = (double) ds.body.size() + 2.0 * (double) (ds.S * 256 * 2 + (ds.N - ds.B) * ggml_row_size(GGML_TYPE_TQ6_0, 256)) * ds.hkv;

        for (int T : Ts) {
            std::vector<float> q_ggml((size_t) ds.nh * T * ds.D);
            for (int q = 0; q < T; ++q) {
                for (int h = 0; h < ds.nh; ++h) {
                    memcpy(q_ggml.data() + ((size_t) h * T + q) * ds.D,
                           ds.Q.data() + ((size_t) q * ds.nh + h) * ds.D, ds.D * sizeof(float));
                }
            }
            ggml_backend_tensor_set(fix.q, q_ggml.data(), 0, q_ggml.size() * sizeof(float));

            ggml_tensor qv = *fix.q;
            qv.ne[1] = T;
            qv.nb[2] = qv.nb[1] * T;
            qv.nb[3] = qv.nb[2] * qv.ne[2];
            ggml_tensor mv = *fix.m;
            mv.ne[1] = T;
            mv.nb[2] = mv.nb[1] * T;
            mv.nb[3] = mv.nb[2] * mv.ne[2];
            ggml_tensor dv = *fix.out;
            dv.ne[2] = T;
            dv.nb[3] = dv.nb[2] * T;
            dv.src[0] = &qv;
            dv.src[3] = &mv;

            if (env_dispatch) {
                const double t_env = bench_mode(fix, &dv, BenchMode::BASELINE, d_l2, l2_words, warmup, iters);
                const double flops = 4.0 * (double) T * (double) ds.N * (double) ds.nh * (double) ds.D;
                printf(" %2d | ENV_DISPATCH=%8.3f ms (%.1f TFLOPS, %.1f GB/s)\n",
                       T, t_env, (flops / (t_env * 1e-3)) / 1e12, (bytes_kv / (t_env * 1e-3)) / 1e9);
                continue;
            }
            const double t_base = bench_mode(fix, &dv, BenchMode::BASELINE,      d_l2, l2_words, warmup, iters);
            const double t_pv16 = bench_mode(fix, &dv, BenchMode::STREAM_PV16,   d_l2, l2_words, warmup, iters);
            const double t_dirv = bench_mode(fix, &dv, BenchMode::DIRECT_VERIFY, d_l2, l2_words, warmup, iters);

            const double flops = 4.0 * (double) T * (double) ds.N * (double) ds.nh * (double) ds.D;
            const double dir_tf = (flops / (t_dirv * 1e-3)) / 1e12;
            const double dir_bw = (bytes_kv / (t_dirv * 1e-3)) / 1e9;
            if (t_pv16 > 0.0) {
                printf(" %2d | %10.3f | %9.3f | %8.2fx | %13.3f | %7.2fx | %10.1f | %6.1f GB/s\n",
                       T, t_base, t_pv16, t_base / t_pv16, t_dirv, t_base / t_dirv, dir_tf, dir_bw);
            } else {
                printf(" %2d | %10.3f |       N/A |       N/A | %13.3f | %7.2fx | %10.1f | %6.1f GB/s\n",
                       T, t_base, t_dirv, t_base / t_dirv, dir_tf, dir_bw);
            }
        }
        fix.destroy();
    }

    CK(cudaFree(d_l2));
    ggml_backend_free(backend);
    return 0;
}
