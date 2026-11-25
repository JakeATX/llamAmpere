// Standalone real-data, L2-flushed benchmark & GPU accuracy verifier for i8qk unblocking:
//   - Compares real ggml_cuda_flash_attn_ext (i8qk path via GGML_CUDA_FA_I8QK_F16=1 / GGML_CUDA_FA_I8QK_EXT=1)
//     directly against real baseline flash_attn_ext_f16 (ggml_cuda_flash_attn_ext_mma_f16_case) in the same binary.
//   - Realistic data with Outl100x high-bias outlier channels in K, per-head relative biases, sinks, and LSE output.
//   - Cold-L2 cache flush (>32 MiB on 6 MiB L2) before every timed GPU iteration.

#include "../ggml/src/ggml-cuda/fattn.cuh"
#include "../ggml/src/ggml-cuda/fattn-i8qk.cuh"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool preserve_cand = false);

namespace {

#define CUDA_CK(call) do {                                                                 \
    const cudaError_t err__ = (call);                                                      \
    if (err__ != cudaSuccess) {                                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                     cudaGetErrorString(err__));                                           \
        std::exit(1);                                                                      \
    }                                                                                      \
} while (0)

__global__ void k_flush_l2(uint4 * __restrict__ buf, size_t n_vec4, uint32_t seed) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    uint32_t acc = seed;
    for (size_t i = idx; i < n_vec4; i += stride) {
        const uint4 v = buf[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
        buf[i] = make_uint4(v.x + 1u, v.y ^ seed, v.z + acc, v.w);
    }
}

struct L2Flusher {
    uint4 * d_buf = nullptr;
    size_t  n_vec4 = 0;
    uint32_t tick = 1;

    void init(size_t bytes = 32ull << 20) {
        n_vec4 = bytes / sizeof(uint4);
        CUDA_CK(cudaMalloc(&d_buf, n_vec4 * sizeof(uint4)));
        CUDA_CK(cudaMemset(d_buf, 0x5A, n_vec4 * sizeof(uint4)));
    }
    void free_mem() {
        if (d_buf) {
            CUDA_CK(cudaFree(d_buf));
            d_buf = nullptr;
        }
    }
    void flush(cudaStream_t stream) {
        k_flush_l2<<<256, 256, 0, stream>>>(d_buf, n_vec4, tick++);
        CUDA_CK(cudaGetLastError());
    }
};

struct BenchCfg {
    std::string label;
    int D;
    int n_q;
    int n_kv;
    int h_q;
    int h_kv;
    int mask_ne2;
    bool use_sinks;
    bool use_lse;
    ggml_type kv_type; // GGML_TYPE_F16 or GGML_TYPE_Q8_0
    int swa_window;    // 0 = full causal; >0 = causal sliding window
};

struct AccuracyStats {
    double max_abs_out = 0.0;
    double rmse_out    = 0.0;
    double cosine_out  = 0.0;
    double max_abs_lse = 0.0;
};

static AccuracyStats compare_results(
        const std::vector<float> & got_out, const std::vector<float> & ref_out,
        const std::vector<float> & got_lse, const std::vector<float> & ref_lse) {
    AccuracyStats st;
    double sq_sum = 0.0, dot = 0.0, ng = 0.0, nr = 0.0;
    for (size_t i = 0; i < got_out.size(); ++i) {
        const double g = static_cast<double>(got_out[i]);
        const double r = static_cast<double>(ref_out[i]);
        const double diff = std::abs(g - r);
        st.max_abs_out = std::max(st.max_abs_out, diff);
        sq_sum += diff * diff;
        dot += g * r;
        ng  += g * g;
        nr  += r * r;
    }
    st.rmse_out   = got_out.empty() ? 0.0 : std::sqrt(sq_sum / static_cast<double>(got_out.size()));
    st.cosine_out = (ng > 0.0 && nr > 0.0) ? dot / (std::sqrt(ng) * std::sqrt(nr)) : 1.0;

    for (size_t i = 0; i < got_lse.size(); ++i) {
        if (std::isinf(got_lse[i]) && std::isinf(ref_lse[i]) && ((got_lse[i] < 0) == (ref_lse[i] < 0))) {
            continue;
        }
        st.max_abs_lse = std::max(st.max_abs_lse, std::abs(static_cast<double>(got_lse[i]) - static_cast<double>(ref_lse[i])));
    }
    return st;
}

static void launch_f16_baseline(ggml_backend_cuda_context & cuda_ctx, ggml_tensor * dst, int cc) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const int64_t n_kv = K->ne[1];
    const int gqa = static_cast<int>(Q->ne[2] / K->ne[2]);

    // Mirror fattn.cu ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2 exactly for SM86
    constexpr int64_t kFattnKqStride = 256;
    bool use_gqa_opt = mask && mask->ne[2] == 1 && n_kv % kFattnKqStride == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }
    GGML_UNUSED(cc);
    if (Q->ne[0] == 128) {
        if (use_gqa_opt && gqa > 4) {
            ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 8>(cuda_ctx, dst, false);
        } else if (use_gqa_opt && gqa > 2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 16, 4>(cuda_ctx, dst, false);
        } else if (use_gqa_opt && gqa > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 32, 2>(cuda_ctx, dst, false);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 64, 1>(cuda_ctx, dst, false);
        }
    } else if (Q->ne[0] == 256) {
        if (use_gqa_opt && gqa > 4) {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8>(cuda_ctx, dst, false);
        } else if (use_gqa_opt && gqa > 2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 16, 4>(cuda_ctx, dst, false);
        } else if (use_gqa_opt && gqa > 1) {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 32, 2>(cuda_ctx, dst, false);
        } else {
            ggml_cuda_flash_attn_ext_mma_f16_case<256, 256, 64, 1>(cuda_ctx, dst, false);
        }
    } else {
        GGML_ABORT("unsupported D in bench");
    }
}

static bool run_case(
        ggml_backend_t backend,
        ggml_backend_cuda_context & cuda_ctx,
        int cc,
        L2Flusher & l2,
        const BenchCfg & cfg,
        int warmup_iters,
        int timed_iters,
        bool check_only) {
    const int64_t D = cfg.D;
    const int64_t n_q = cfg.n_q;
    const int64_t n_kv = cfg.n_kv;
    const int64_t h_q = cfg.h_q;
    const int64_t h_kv = cfg.h_kv;
    const int64_t mask_ne2 = cfg.mask_ne2;

    ggml_init_params gparams = {
        /*.mem_size   =*/ 64 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(gparams);

    // ggml FA shapes: Q [D, n_q, h_q, 1], K [D, n_kv, h_kv, 1], V [D, n_kv, h_kv, 1]
    ggml_tensor * Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, n_q, h_q, 1);
    ggml_tensor * K = ggml_new_tensor_4d(ctx, cfg.kv_type, D, n_kv, h_kv, 1);
    ggml_tensor * V = ggml_new_tensor_4d(ctx, cfg.kv_type, D, n_kv, h_kv, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, n_q, mask_ne2, 1);
    ggml_tensor * sinks = cfg.use_sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, h_q) : nullptr;
    ggml_tensor * lse = cfg.use_lse ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, h_q, n_q, 1, 1) : nullptr;

    ggml_set_input(Q);
    ggml_set_input(K);
    ggml_set_input(V);
    ggml_set_input(mask);
    if (sinks) ggml_set_input(sinks);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    ggml_tensor * dst = ggml_flash_attn_ext(ctx, Q, K, V, mask, scale, 0.0f, 0.0f);
    if (sinks) {
        dst->src[4] = sinks;
    }
    if (lse) {
        ggml_flash_attn_ext_set_lse(dst, lse);
        ggml_set_output(lse);
    }
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "gallocr failed on %s\n", cfg.label.c_str());
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // Verify i8qk route will actually fire on this node
    if (!ggml_cuda_fattn_i8qk_applies(cc, dst)) {
        std::fprintf(stderr, "ERROR: ggml_cuda_fattn_i8qk_applies returned false for %s!\n", cfg.label.c_str());
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    // Populate realistic tensors with Outl100x channels in K
    std::mt19937 rng(4242 + static_cast<uint32_t>(n_kv) + static_cast<uint32_t>(n_q));
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> h_Q(static_cast<size_t>(h_q) * n_q * D);
    for (float & x : h_Q) x = nd(rng) * 0.7f;
    ggml_backend_tensor_set(Q, h_Q.data(), 0, ggml_nbytes(Q));

    // Generate F32 K/V first (with Outl100x channels in K), then convert to F16 or Q8_0
    {
        std::vector<float> h_K_f32(static_cast<size_t>(h_kv) * n_kv * D);
        std::vector<float> h_V_f32(static_cast<size_t>(h_kv) * n_kv * D);
        for (int64_t kvh = 0; kvh < h_kv; ++kvh) {
            for (int64_t k = 0; k < n_kv; ++k) {
                for (int64_t c = 0; c < D; ++c) {
                    const float bias = (c < 4) ? ((c % 2 == 0) ? 28.0f : -28.0f) : 0.0f;
                    const float stdv = (c < 4) ? 0.5f : 0.9f;
                    h_K_f32[(static_cast<size_t>(kvh) * n_kv + k) * D + c] = bias + nd(rng) * stdv;
                    h_V_f32[(static_cast<size_t>(kvh) * n_kv + k) * D + c] = nd(rng) * 0.9f;
                }
            }
        }
        if (cfg.kv_type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> h_K16(h_K_f32.size()), h_V16(h_V_f32.size());
            ggml_fp32_to_fp16_row(h_K_f32.data(), h_K16.data(), h_K_f32.size());
            ggml_fp32_to_fp16_row(h_V_f32.data(), h_V16.data(), h_V_f32.size());
            ggml_backend_tensor_set(K, h_K16.data(), 0, ggml_nbytes(K));
            ggml_backend_tensor_set(V, h_V16.data(), 0, ggml_nbytes(V));
        } else {
            std::vector<uint8_t> h_Kq(ggml_nbytes(K)), h_Vq(ggml_nbytes(V));
            const int64_t nrows = h_kv * n_kv;
            ggml_quantize_chunk(cfg.kv_type, h_K_f32.data(), h_Kq.data(), 0, nrows, D, nullptr);
            ggml_quantize_chunk(cfg.kv_type, h_V_f32.data(), h_Vq.data(), 0, nrows, D, nullptr);
            ggml_backend_tensor_set(K, h_Kq.data(), 0, ggml_nbytes(K));
            ggml_backend_tensor_set(V, h_Vq.data(), 0, ggml_nbytes(V));
        }
    }

    // Populate F16 mask [n_kv, n_q, mask_ne2, 1]: causal (plus optional SWA window + per-head relative bias)
    {
        std::vector<ggml_fp16_t> h_mask(static_cast<size_t>(mask_ne2) * n_q * n_kv);
        const ggml_fp16_t ninf_h = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_h = ggml_fp32_to_fp16(0.0f);
        const int64_t q_off = n_kv - n_q;
        for (int64_t mh = 0; mh < mask_ne2; ++mh) {
            const float head_slope = mask_ne2 > 1 ? 0.0025f * static_cast<float>((mh % 12) + 1) : 0.0f;
            for (int64_t q = 0; q < n_q; ++q) {
                const int64_t qpos = q_off + q;
                ggml_fp16_t * row = h_mask.data() + (static_cast<size_t>(mh) * n_q + q) * n_kv;
                for (int64_t k = 0; k < n_kv; ++k) {
                    if (k > qpos || (cfg.swa_window > 0 && k <= qpos - cfg.swa_window)) {
                        row[k] = ninf_h;
                    } else if (mask_ne2 > 1) {
                        const int64_t dist = qpos - k;
                        const float bias = -head_slope * static_cast<float>(dist & 31);
                        row[k] = ggml_fp32_to_fp16(bias);
                    } else {
                        row[k] = zero_h;
                    }
                }
            }
        }
        ggml_backend_tensor_set(mask, h_mask.data(), 0, ggml_nbytes(mask));
    }

    if (sinks) {
        std::vector<float> h_sinks(h_q);
        for (int64_t h = 0; h < h_q; ++h) {
            h_sinks[h] = -1.5f + 0.25f * static_cast<float>(h % 16);
        }
        ggml_backend_tensor_set(sinks, h_sinks.data(), 0, ggml_nbytes(sinks));
    }

    cudaStream_t stream = cuda_ctx.stream();

    // 1. Accuracy check: Baseline F16 MMA vs i8qk dispatch on identical buffers
    std::vector<float> ref_out(ggml_nelements(dst));
    std::vector<float> got_out(ggml_nelements(dst));
    std::vector<float> ref_lse(lse ? ggml_nelements(lse) : 0);
    std::vector<float> got_lse(lse ? ggml_nelements(lse) : 0);

    launch_f16_baseline(cuda_ctx, dst, cc);
    CUDA_CK(cudaStreamSynchronize(stream));
    ggml_backend_tensor_get(dst, ref_out.data(), 0, ggml_nbytes(dst));
    if (lse) ggml_backend_tensor_get(lse, ref_lse.data(), 0, ggml_nbytes(lse));

    // Zero outputs before running i8qk so accuracy check cannot pick up stale baseline outputs
    CUDA_CK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    if (lse) CUDA_CK(cudaMemsetAsync(lse->data, 0, ggml_nbytes(lse), stream));
    ggml_cuda_flash_attn_ext(cuda_ctx, dst);
    CUDA_CK(cudaStreamSynchronize(stream));
    ggml_backend_tensor_get(dst, got_out.data(), 0, ggml_nbytes(dst));
    if (lse) ggml_backend_tensor_get(lse, got_lse.data(), 0, ggml_nbytes(lse));

    const AccuracyStats acc = compare_results(got_out, ref_out, got_lse, ref_lse);

    auto measure_fn = [&](auto && fn) -> double {
        cudaEvent_t ev0, ev1;
        CUDA_CK(cudaEventCreate(&ev0));
        CUDA_CK(cudaEventCreate(&ev1));
        for (int i = 0; i < warmup_iters; ++i) {
            l2.flush(stream);
            fn();
        }
        CUDA_CK(cudaStreamSynchronize(stream));
        std::vector<float> times(timed_iters);
        for (int i = 0; i < timed_iters; ++i) {
            l2.flush(stream);
            CUDA_CK(cudaEventRecord(ev0, stream));
            fn();
            CUDA_CK(cudaEventRecord(ev1, stream));
            CUDA_CK(cudaEventSynchronize(ev1));
            CUDA_CK(cudaEventElapsedTime(&times[i], ev0, ev1));
        }
        CUDA_CK(cudaEventDestroy(ev0));
        CUDA_CK(cudaEventDestroy(ev1));
        std::sort(times.begin(), times.end());
        return static_cast<double>(times[times.size() / 2]);
    };

    double base_ms = 0.0, i8_ms = 0.0, speedup = 0.0, tflops_i8 = 0.0;
    if (!check_only) {
        base_ms = measure_fn([&] { launch_f16_baseline(cuda_ctx, dst, cc); });
        i8_ms   = measure_fn([&] { ggml_cuda_flash_attn_ext(cuda_ctx, dst); });
        speedup = base_ms / i8_ms;
        // Effective full-matrix attention FLOPs (4 * H_q * N_q * N_kv * D); causal skips upper triangle when N_q <= N_kv
        const double active_pairs = static_cast<double>(n_q) * static_cast<double>(n_kv) -
                                    0.5 * static_cast<double>(n_q) * static_cast<double>(n_q - 1);
        const double eff_flops = 4.0 * static_cast<double>(h_q) * active_pairs * static_cast<double>(D);
        tflops_i8 = eff_flops / ((i8_ms * 1e-3) * 1e12);
    }

    std::printf("%-42s | cos=%.6f maxE=%.4f rmse=%.4f lseE=%.4f | f16=%.3fms i8qk=%.3fms (%5.2fx, %6.1fT)\n",
                cfg.label.c_str(), acc.cosine_out, acc.max_abs_out, acc.rmse_out, acc.max_abs_lse,
                base_ms, i8_ms, speedup, tflops_i8);
    std::fflush(stdout);

    const bool ok = acc.cosine_out >= 0.995 && !std::isnan(acc.max_abs_out) && (!lse || acc.max_abs_lse < 0.25);
    if (!ok) {
        std::fprintf(stderr, "ACCURACY FAIL on %s: cos=%.6f max_abs=%.4f lse_err=%.4f\n",
                     cfg.label.c_str(), acc.cosine_out, acc.max_abs_out, acc.max_abs_lse);
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    int warmup = 3;
    int iters = 12;
    bool check_only = false;
    bool include_d256 = false;
    int swa_window = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--check-only") == 0) {
            check_only = true;
        } else if (std::strcmp(argv[i], "--d256") == 0) {
            include_d256 = true;
        } else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--swa") == 0 && i + 1 < argc) {
            swa_window = std::atoi(argv[++i]);
        }
    }

    // Default ON inside this comparison binary unless caller explicitly set =0
    setenv("GGML_CUDA_FA_I8QK",     "1", 0);
    setenv("GGML_CUDA_FA_I8QK_F16", "1", 0);
    setenv("GGML_CUDA_FA_I8QK_EXT", "1", 0);

    if (!ggml_cuda_fattn_i8qk_f16_enabled() || !ggml_cuda_fattn_i8qk_ext_enabled()) {
        std::fprintf(stderr, "WARNING: GGML_CUDA_FA_I8QK_F16=%d GGML_CUDA_FA_I8QK_EXT=%d\n",
                     (int) ggml_cuda_fattn_i8qk_f16_enabled(), (int) ggml_cuda_fattn_i8qk_ext_enabled());
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "Failed to initialize CUDA backend 0\n");
        return 1;
    }
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    L2Flusher l2;
    l2.init(32ull << 20);

    std::vector<BenchCfg> cases;
    // Suite 1: Inkling SWA / SJ-KVaRN Unblockers (D=128, GQA=6, H_q=24, H_kv=4, per-head mask ne2=24, sinks, LSE)
    for (int n_q : {512, 2048}) {
        for (int n_kv : {4096, 32768, 100000}) {
            cases.push_back({
                "InkSWA F16 Nq=" + std::to_string(n_q) + " Nkv=" + std::to_string(n_kv),
                128, n_q, n_kv, 24, 4, 24, true, true, GGML_TYPE_F16, swa_window
            });
        }
    }
    // Inkling SWA real default Q8_0 KV at 100K & 32K
    for (int n_q : {512, 2048}) {
        for (int n_kv : {4096, 32768, 100000}) {
            cases.push_back({
                "InkSWA Q8_0 Nq=" + std::to_string(n_q) + " Nkv=" + std::to_string(n_kv),
                128, n_q, n_kv, 24, 4, 24, true, true, GGML_TYPE_Q8_0, swa_window
            });
        }
    }
    // Suite 2: Standard F16 KV Prefill (D=128, GQA=8, H_q=32, H_kv=4, shared mask ne2=1)
    for (int n_q : {512, 2048}) {
        for (int n_kv : {4096, 32768, 100000, 100096}) {
            cases.push_back({
                "Std F16 GQA8 Nq=" + std::to_string(n_q) + " Nkv=" + std::to_string(n_kv),
                128, n_q, n_kv, 32, 4, 1, false, false, GGML_TYPE_F16, 0
            });
        }
    }
    if (include_d256) {
        for (int n_q : {512, 2048}) {
            for (int n_kv : {4096, 32768}) {
                cases.push_back({
                    "Std F16 D256 GQA8 Nq=" + std::to_string(n_q) + " Nkv=" + std::to_string(n_kv),
                    256, n_q, n_kv, 32, 4, 1, true, true, GGML_TYPE_F16, 0
                });
            }
        }
    }

    std::printf("=== bench-i8qk-unblock (cc=%d, F16=%d, EXT=%d, L2-flushed 32MiB, warmup=%d, iters=%d) ===\n",
                cc, (int) ggml_cuda_fattn_i8qk_f16_enabled(), (int) ggml_cuda_fattn_i8qk_ext_enabled(), warmup, iters);
    bool all_ok = true;
    for (const BenchCfg & c : cases) {
        if (!run_case(backend, *cuda_ctx, cc, l2, c, warmup, iters, check_only)) {
            all_ok = false;
        }
    }

    l2.free_mem();
    ggml_backend_free(backend);
    return all_ok ? 0 : 1;
}
