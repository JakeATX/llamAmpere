// L2-flushed SM 8.6 benchmark + GPU bit-exactness gate for the prefill glue kernels (fused GLU / RMS_NORM -> f32 output
// + quantized activation). Needs a free GPU; not part of ctest.

#include "ggml.h"
#include "ggml-cuda.h"
#include "prefill-glue.cuh"
#include "rq4.cuh"
#include "ai4-kern.cuh"
#include "nvfp4pf-kern.cuh"
#include "quantize.cuh"
#include "norm.cuh"
#include "unary.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CK(call) do {                                                                 \
    cudaError_t err__ = (call);                                                            \
    if (err__ != cudaSuccess) {                                                            \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                                                      \
    }                                                                                      \
} while (0)

// L2 cache scrub (> 32 MiB writes to evict 6 MiB GA102 L2 cleanly between measured launches)
static __global__ void k_flush_l2(uint4 * buf, size_t n_vec) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = idx; i < n_vec; i += stride) {
        buf[i] = make_uint4(0xdeadbeefu, (uint32_t) i, 0x12345678u, 0x9abcdef0u);
    }
}

// Reference baseline kernels matching llamAmpere unfused ops for same-binary comparison
static __global__ void k_ref_situ(const float * g, const float * u, float * out, int K, int64_t rows, ink_situ_params p) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        out[r * K + c] = ink_situ_elem(g[r * K + c], u[r * K + c], p);
    }
}

static __global__ void k_ref_swiglu(const float * g, const float * u, float * out, int K, int64_t rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        out[r * K + c] = ggml_cuda_op_silu_single(g[r * K + c]) * u[r * K + c];
    }
}

// stock GLU element functions (unary.cuh / unary.cu), one per kind
static __global__ void k_ref_glu(int glu_kind, const float * g, const float * u, float * out, int K, int64_t rows,
        ink_situ_params p, float limit, float oai_alpha, float oai_limit) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const float x = g[r * K + c], y = u[r * K + c];
        float v;
        switch (glu_kind) {
            case PF_GLU_SITU:         v = ink_situ_elem(x, y, p); break;
            case PF_GLU_REGLU:        v = fmaxf(x, 0) * y; break;
            case PF_GLU_GEGLU:        v = ggml_cuda_op_gelu_single(x) * y; break;
            case PF_GLU_SWIGLU:       v = ggml_cuda_op_silu_single(x) * y; break;
            case PF_GLU_SWIGLU_OAI:   v = ggml_cuda_op_swiglu_oai_single(x, y, oai_alpha, oai_limit); break;
            case PF_GLU_SWIGLU_CLAMP: v = ggml_cuda_op_swiglu_clamp_single(x, y, limit); break;
            case PF_GLU_GEGLU_ERF:    v = (0.5f*x*(1.0f + erff(x*0.70710678118654752440084436210484f))) * y; break;
            case PF_GLU_GEGLU_QUICK:  v = (x * (1.0f / (1.0f + expf(-1.702f * x)))) * y; break;
            default:                  v = y; break;
        }
        out[r * K + c] = v;
    }
}

template <int BS>
static __global__ void k_ref_rmsnorm_mul(const float * x, const float * w, float * dst, int K, int64_t rows, float eps) {
    __shared__ float s_red[32];
    const int64_t row = blockIdx.x;
    if (row >= rows) return;
    const float * xr = x + row * K;
    float * dr = dst + row * K;
    float tmp = 0.0f;
    for (int col = threadIdx.x; col < K; col += BS) {
        const float xi = xr[col];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_red);
    const float sc = rsqrtf(tmp / (float) K + eps);
    __syncthreads();
    for (int col = threadIdx.x; col < K; col += BS) {
        dr[col] = (sc * xr[col]) * w[col];
    }
}

template <typename FN>
static float measure_median_us(FN && fn, uint4 * d_flush, size_t flush_vecs, int warmup = 5, int iters = 25) {
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(iters);
    cudaEvent_t ev0, ev1;
    CUDA_CK(cudaEventCreate(&ev0));
    CUDA_CK(cudaEventCreate(&ev1));
    for (int i = 0; i < iters; ++i) {
        k_flush_l2<<<256, 256>>>(d_flush, flush_vecs);
        CUDA_CK(cudaEventRecord(ev0));
        fn();
        CUDA_CK(cudaEventRecord(ev1));
        CUDA_CK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CK(cudaEventElapsedTime(&ms, ev0, ev1));
        times.push_back(ms * 1000.0f);
    }
    CUDA_CK(cudaEventDestroy(ev0));
    CUDA_CK(cudaEventDestroy(ev1));
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static void fill_realistic(std::vector<float> & buf, std::mt19937 & rng, bool outliers = true) {
    std::normal_distribution<float> dist(0.0f, 0.75f);
    for (size_t i = 0; i < buf.size(); ++i) {
        float scale = (outliers && (i % 100 == 7)) ? 100.0f : 1.0f;
        buf[i] = dist(rng) * scale;
    }
}

static bool same_f32(const float * d_a, const float * d_b, size_t n) {
    std::vector<float> a(n), b(n);
    CUDA_CK(cudaMemcpy(a.data(), d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CK(cudaMemcpy(b.data(), d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
    return std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0;
}

static bool run_gpu_correctness_checks(std::mt19937 & rng) {
    bool ok = true;
    const ink_situ_params situ = { 0.85f, 0.05f, 1.1f, -0.02f, 0.75f, 0.01f, 0.9f, 0.1f, 1, 1 };

    auto verify_glu = [&](int rows, int K, int glu_kind, const char * tag) {
        std::vector<float> hg((size_t) rows * K), hu((size_t) rows * K);
        fill_realistic(hg, rng, true);
        fill_realistic(hu, rng, true);

        float * dg = nullptr, * du = nullptr, * df32 = nullptr, * dfus = nullptr, * d_ref_s = nullptr, * d_fus_s = nullptr;
        int8_t * d_ref_q = nullptr, * d_fus_q = nullptr;
        const int xsh = (K % 256 == 0) ? 1 : 0;
        const int ns  = K >> (7 + xsh);
        CUDA_CK(cudaMalloc(&dg, hg.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&du, hu.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&df32, hg.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&dfus, hg.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&d_ref_q, hg.size()));
        CUDA_CK(cudaMalloc(&d_fus_q, hg.size()));
        CUDA_CK(cudaMalloc(&d_ref_s, (size_t) rows * ns * sizeof(float)));
        CUDA_CK(cudaMalloc(&d_fus_s, (size_t) rows * ns * sizeof(float)));
        CUDA_CK(cudaMemcpy(dg, hg.data(), hg.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CK(cudaMemcpy(du, hu.data(), hu.size() * sizeof(float), cudaMemcpyHostToDevice));

        const dim3 grid_e((unsigned) ((K + 255) / 256), (unsigned) rows);
        k_ref_glu<<<grid_e, 256>>>(glu_kind, dg, du, df32, K, rows, situ, 7.0f, 1.702f, 7.0f);

        if (xsh) rq4::launch_quantize_act<false, false, false>(df32, K, d_ref_q, d_ref_s, K, rows, 0);
        else     ai4k::launch_quantize_act128<false, false, false>(df32, K, d_ref_q, d_ref_s, K, rows, 0);

        ggml_cuda_prefill_launch_glu_ai4(
            glu_kind, dg, du, dfus, d_fus_q, d_fus_s, K, rows, rows,
            (int64_t) K * sizeof(float), (int64_t) rows * K * sizeof(float),
            (int64_t) K * sizeof(float), (int64_t) rows * K * sizeof(float),
            xsh, 0, situ, 7.0f, 1.702f, 7.0f, 0);
        CUDA_CK(cudaDeviceSynchronize());

        std::vector<int8_t> hq0(hg.size()), hq1(hg.size());
        std::vector<float>  hs0((size_t) rows * ns), hs1((size_t) rows * ns);
        CUDA_CK(cudaMemcpy(hq0.data(), d_ref_q, hq0.size(), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hq1.data(), d_fus_q, hq1.size(), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hs0.data(), d_ref_s, hs0.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hs1.data(), d_fus_s, hs1.size() * sizeof(float), cudaMemcpyDeviceToHost));

        const bool eq = (std::memcmp(hq0.data(), hq1.data(), hq0.size()) == 0) &&
                        (std::memcmp(hs0.data(), hs1.data(), hs0.size() * sizeof(float)) == 0) &&
                        same_f32(df32, dfus, hg.size());
        std::printf("[GPU GATE] %-24s rows=%-4d K=%-5d -> f32 + AI4 %s (0-ULP)\n", tag, rows, K, eq ? "PASS" : "FAIL");
        ok &= eq;

        CUDA_CK(cudaFree(dg)); CUDA_CK(cudaFree(du)); CUDA_CK(cudaFree(df32)); CUDA_CK(cudaFree(dfus));
        CUDA_CK(cudaFree(d_ref_q)); CUDA_CK(cudaFree(d_fus_q));
        CUDA_CK(cudaFree(d_ref_s)); CUDA_CK(cudaFree(d_fus_s));
    };

    auto verify_norm = [&](int rows, int K, const char * tag) {
        std::vector<float> hx((size_t) rows * K), hw(K);
        fill_realistic(hx, rng, true);
        fill_realistic(hw, rng, false);

        float * dx = nullptr, * dw = nullptr, * df32 = nullptr, * dfus = nullptr, * d_ref_s = nullptr, * d_fus_s = nullptr;
        int8_t * d_ref_q = nullptr, * d_fus_q = nullptr;
        const int xsh = (K % 256 == 0) ? 1 : 0;
        const int ns  = K >> (7 + xsh);
        CUDA_CK(cudaMalloc(&dx, hx.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&dw, hw.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&df32, hx.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&dfus, hx.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&d_ref_q, hx.size()));
        CUDA_CK(cudaMalloc(&d_fus_q, hx.size()));
        CUDA_CK(cudaMalloc(&d_ref_s, (size_t) rows * ns * sizeof(float)));
        CUDA_CK(cudaMalloc(&d_fus_s, (size_t) rows * ns * sizeof(float)));
        CUDA_CK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

        if (K < 1024) k_ref_rmsnorm_mul<256><<<rows, 256>>>(dx, dw, df32, K, rows, 1e-6f);
        else          k_ref_rmsnorm_mul<1024><<<rows, 1024>>>(dx, dw, df32, K, rows, 1e-6f);

        if (xsh) rq4::launch_quantize_act<false, false, false>(df32, K, d_ref_q, d_ref_s, K, rows, 0);
        else     ai4k::launch_quantize_act128<false, false, false>(df32, K, d_ref_q, d_ref_s, K, rows, 0);

        ggml_cuda_prefill_launch_norm_ai4(dx, nullptr, nullptr, dw, dfus, d_fus_q, d_fus_s, rows, K, 1e-6f, xsh, 0, 0);
        CUDA_CK(cudaDeviceSynchronize());

        std::vector<int8_t> hq0(hx.size()), hq1(hx.size());
        std::vector<float>  hs0((size_t) rows * ns), hs1((size_t) rows * ns);
        CUDA_CK(cudaMemcpy(hq0.data(), d_ref_q, hq0.size(), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hq1.data(), d_fus_q, hq1.size(), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hs0.data(), d_ref_s, hs0.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hs1.data(), d_fus_s, hs1.size() * sizeof(float), cudaMemcpyDeviceToHost));

        const bool eq = (std::memcmp(hq0.data(), hq1.data(), hq0.size()) == 0) &&
                        (std::memcmp(hs0.data(), hs1.data(), hs0.size() * sizeof(float)) == 0) &&
                        same_f32(df32, dfus, hx.size());
        std::printf("[GPU GATE] %-24s rows=%-4d K=%-5d -> f32 + AI4 %s (0-ULP)\n", tag, rows, K, eq ? "PASS" : "FAIL");
        ok &= eq;

        CUDA_CK(cudaFree(dx)); CUDA_CK(cudaFree(dw)); CUDA_CK(cudaFree(df32)); CUDA_CK(cudaFree(dfus));
        CUDA_CK(cudaFree(d_ref_q)); CUDA_CK(cudaFree(d_fus_q));
        CUDA_CK(cudaFree(d_ref_s)); CUDA_CK(cudaFree(d_fus_s));
    };

    verify_glu(64, 1152,  PF_GLU_SITU,   "SITU Inkling H128");
    verify_glu(64, 2304,  PF_GLU_SITU,   "SITU Inkling H256");
    verify_glu(32, 28672, PF_GLU_SWIGLU, "SwiGLU Dense H256");
    verify_glu(64, 2304,  PF_GLU_REGLU,        "ReGLU H256");
    verify_glu(64, 1152,  PF_GLU_GEGLU,        "GeGLU H128");
    verify_glu(64, 2304,  PF_GLU_GEGLU_ERF,    "GeGLU-erf H256");
    verify_glu(64, 2304,  PF_GLU_GEGLU_QUICK,  "GeGLU-quick H256");
    verify_glu(64, 2304,  PF_GLU_SWIGLU_OAI,   "SwiGLU-OAI H256");
    verify_glu(64, 2304,  PF_GLU_SWIGLU_CLAMP, "SwiGLU-clamp H256");

    // NVFP4PF KF=64 0-ULP GPU verification
    {
        const int rows = 64, K = 1152, kf = 64;
        std::vector<float> hg((size_t) rows * K), hu((size_t) rows * K);
        fill_realistic(hg, rng, true);
        fill_realistic(hu, rng, true);
        float * dg = nullptr, * du = nullptr, * df32 = nullptr, * dfus = nullptr, * d_ref_s = nullptr, * d_fus_s = nullptr;
        int8_t * d_ref_q = nullptr, * d_fus_q = nullptr;
        CUDA_CK(cudaMalloc(&dg, hg.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&du, hu.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&df32, hg.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&dfus, hg.size() * sizeof(float)));
        CUDA_CK(cudaMalloc(&d_ref_q, hg.size()));
        CUDA_CK(cudaMalloc(&d_fus_q, hg.size()));
        CUDA_CK(cudaMalloc(&d_ref_s, (size_t) rows * (K / kf) * sizeof(float)));
        CUDA_CK(cudaMalloc(&d_fus_s, (size_t) rows * (K / kf) * sizeof(float)));
        CUDA_CK(cudaMemcpy(dg, hg.data(), hg.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CK(cudaMemcpy(du, hu.data(), hu.size() * sizeof(float), cudaMemcpyHostToDevice));

        const dim3 grid_e((unsigned) ((K + 255) / 256), (unsigned) rows);
        k_ref_situ<<<grid_e, 256>>>(dg, du, df32, K, rows, situ);
        const dim3 gq((unsigned) ((K / kf + 3) / 4), (unsigned) rows);
        k_nvfp4pf_xq<64><<<gq, 128>>>(df32, K, d_ref_q, d_ref_s, K, rows);
        ggml_cuda_prefill_launch_glu_nvfp4pf(
            PF_GLU_SITU, dg, du, dfus, d_fus_q, d_fus_s, K, rows, rows,
            (int64_t) K * sizeof(float), (int64_t) rows * K * sizeof(float),
            (int64_t) K * sizeof(float), (int64_t) rows * K * sizeof(float),
            kf, situ, 7.0f, 1.702f, 7.0f, 0);
        CUDA_CK(cudaDeviceSynchronize());

        std::vector<int8_t> hq0(hg.size()), hq1(hg.size());
        std::vector<float>  hs0((size_t) rows * (K / kf)), hs1((size_t) rows * (K / kf));
        CUDA_CK(cudaMemcpy(hq0.data(), d_ref_q, hq0.size(), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hq1.data(), d_fus_q, hq1.size(), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hs0.data(), d_ref_s, hs0.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(hs1.data(), d_fus_s, hs1.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const bool eq = (std::memcmp(hq0.data(), hq1.data(), hq0.size()) == 0) &&
                        (std::memcmp(hs0.data(), hs1.data(), hs0.size() * sizeof(float)) == 0) &&
                        same_f32(df32, dfus, hg.size());
        std::printf("[GPU GATE] %-24s rows=%-4d K=%-5d -> f32 + NVFP4PF %s (0-ULP)\n", "SITU Inkling NVFP4PF", rows, K, eq ? "PASS" : "FAIL");
        ok &= eq;
        CUDA_CK(cudaFree(dg)); CUDA_CK(cudaFree(du)); CUDA_CK(cudaFree(df32)); CUDA_CK(cudaFree(dfus));
        CUDA_CK(cudaFree(d_ref_q)); CUDA_CK(cudaFree(d_fus_q));
        CUDA_CK(cudaFree(d_ref_s)); CUDA_CK(cudaFree(d_fus_s));
    }

    verify_norm(64, 2304, "RMSNorm Inkling 2304");
    verify_norm(32, 8192, "RMSNorm Dense 8192");

    return ok;
}

static void bench_mlp_glue(std::mt19937 & rng, uint4 * d_flush, size_t flush_vecs,
                           int rows, int K, int glu_kind, const char * label) {
    std::vector<float> hg((size_t) rows * K), hu((size_t) rows * K);
    fill_realistic(hg, rng, true);
    fill_realistic(hu, rng, true);

    float * dg = nullptr, * du = nullptr, * df32 = nullptr, * d_s = nullptr;
    int8_t * d_q = nullptr;
    const int xsh = (K % 256 == 0) ? 1 : 0;
    const int ns  = K >> (7 + xsh);
    const ink_situ_params situ = { 0.85f, 0.05f, 1.1f, -0.02f, 0.75f, 0.01f, 0.9f, 0.1f, 1, 1 };

    CUDA_CK(cudaMalloc(&dg, hg.size() * sizeof(float)));
    CUDA_CK(cudaMalloc(&du, hu.size() * sizeof(float)));
    CUDA_CK(cudaMalloc(&df32, hg.size() * sizeof(float)));
    CUDA_CK(cudaMalloc(&d_q, hg.size()));
    CUDA_CK(cudaMalloc(&d_s, (size_t) rows * ns * sizeof(float)));
    CUDA_CK(cudaMemcpy(dg, hg.data(), hg.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CK(cudaMemcpy(du, hu.data(), hu.size() * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 grid_e((unsigned) ((K + 255) / 256), (unsigned) std::min<int>(rows, 65535));
    const float base_us = measure_median_us([&]() {
        if (glu_kind == PF_GLU_SITU) k_ref_situ<<<grid_e, 256>>>(dg, du, df32, K, rows, situ);
        else                         k_ref_swiglu<<<grid_e, 256>>>(dg, du, df32, K, rows);
        if (xsh) rq4::launch_quantize_act<false, false, false>(df32, K, d_q, d_s, K, std::min(rows, 65535), 0);
        else     ai4k::launch_quantize_act128<false, false, false>(df32, K, d_q, d_s, K, std::min(rows, 65535), 0);
    }, d_flush, flush_vecs);

    // fused: one kernel writes the f32 GLU output and the quantized activation
    const float fused_us = measure_median_us([&]() {
        ggml_cuda_prefill_launch_glu_ai4(
            glu_kind, dg, du, df32, d_q, d_s, K, rows, rows,
            (int64_t) K * sizeof(float), (int64_t) rows * K * sizeof(float),
            (int64_t) K * sizeof(float), (int64_t) rows * K * sizeof(float),
            xsh, 0, situ, 7.0f, 1.702f, 7.0f, 0);
    }, d_flush, flush_vecs);

    std::printf("[BENCH L2-FLUSH] %-22s rows=%-5d K=%-5d | Base=%7.1f us | fused=%7.1f us (%.2fx)\n",
                label, rows, K, base_us, fused_us, base_us / fused_us);

    CUDA_CK(cudaFree(dg)); CUDA_CK(cudaFree(du)); CUDA_CK(cudaFree(df32));
    CUDA_CK(cudaFree(d_q)); CUDA_CK(cudaFree(d_s));
}

static void bench_norm_glue(std::mt19937 & rng, uint4 * d_flush, size_t flush_vecs,
                            int rows, int K, int n_consumers, const char * label) {
    std::vector<float> hx((size_t) rows * K), hw(K);
    fill_realistic(hx, rng, true);
    fill_realistic(hw, rng, false);

    float * dx = nullptr, * dw = nullptr, * df32 = nullptr, * d_s = nullptr;
    int8_t * d_q = nullptr;
    const int xsh = (K % 256 == 0) ? 1 : 0;
    const int ns  = K >> (7 + xsh);
    CUDA_CK(cudaMalloc(&dx, hx.size() * sizeof(float)));
    CUDA_CK(cudaMalloc(&dw, hw.size() * sizeof(float)));
    CUDA_CK(cudaMalloc(&df32, hx.size() * sizeof(float)));
    CUDA_CK(cudaMalloc(&d_q, hx.size()));
    CUDA_CK(cudaMalloc(&d_s, (size_t) rows * ns * sizeof(float)));
    CUDA_CK(cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CK(cudaMemcpy(dw, hw.data(), hw.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Unfused: RMSNorm + MUL + `n_consumers` quantize launches (mirrors Q/K/V, gate/up, or SWA 6-way before cache)
    const float base_us = measure_median_us([&]() {
        if (K < 1024) k_ref_rmsnorm_mul<256><<<rows, 256>>>(dx, dw, df32, K, rows, 1e-6f);
        else          k_ref_rmsnorm_mul<1024><<<rows, 1024>>>(dx, dw, df32, K, rows, 1e-6f);
        for (int c = 0; c < n_consumers; ++c) {
            if (xsh) rq4::launch_quantize_act<false, false, false>(df32, K, d_q, d_s, K, rows, 0);
            else     ai4k::launch_quantize_act128<false, false, false>(df32, K, d_q, d_s, K, rows, 0);
        }
    }, d_flush, flush_vecs);

    // fused: one kernel writes the f32 norm*w output and the quantized activation (consumers then take the cache)
    const float p2_us = measure_median_us([&]() {
        ggml_cuda_prefill_launch_norm_ai4(dx, nullptr, nullptr, dw, df32, d_q, d_s, rows, K, 1e-6f, xsh, 0, 0);
    }, d_flush, flush_vecs);

    std::printf("[BENCH L2-FLUSH] %-22s rows=%-5d K=%-5d (consumers=%d) | Base=%7.1f us | fused=%7.1f us (%.2fx)\n",
                label, rows, K, n_consumers, base_us, p2_us, base_us / p2_us);

    CUDA_CK(cudaFree(dx)); CUDA_CK(cudaFree(dw)); CUDA_CK(cudaFree(df32));
    CUDA_CK(cudaFree(d_q)); CUDA_CK(cudaFree(d_s));
}

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) {
        std::fprintf(stderr, "No CUDA GPU available\n");
        return 77;
    }
    CUDA_CK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s (SM %d.%d, L2 %.1f MiB)\n", prop.name, prop.major, prop.minor, prop.l2CacheSize / (1024.0 * 1024.0));

    std::mt19937 rng(12345);
    if (!run_gpu_correctness_checks(rng)) {
        std::fprintf(stderr, "GPU correctness check FAILED!\n");
        return 1;
    }

    const size_t flush_bytes = 32u * 1024u * 1024u;
    const size_t flush_vecs  = flush_bytes / sizeof(uint4);
    uint4 * d_flush = nullptr;
    CUDA_CK(cudaMalloc(&d_flush, flush_bytes));

    std::printf("\n--- L2-Flushed Prefill Benchmarks (N = 512, 2048) ---\n");
    // Inkling2 MoE shapes (F=1152 down * k=8 -> 4096, 16384 rows; dense/shared 2304 & 18432 MoE wide)
    bench_mlp_glue(rng, d_flush, flush_vecs, 4096,  1152,  PF_GLU_SITU,   "Inkling MoE down N=512");
    bench_mlp_glue(rng, d_flush, flush_vecs, 16384, 1152,  PF_GLU_SITU,   "Inkling MoE down N=2048");
    bench_mlp_glue(rng, d_flush, flush_vecs, 512,   18432, PF_GLU_SITU,   "Inkling 18432 N=512");
    bench_mlp_glue(rng, d_flush, flush_vecs, 2048,  18432, PF_GLU_SITU,   "Inkling 18432 N=2048");
    // Dense LLM shapes (8192 hidden, 28672 MLP intermediate)
    bench_mlp_glue(rng, d_flush, flush_vecs, 512,   28672, PF_GLU_SWIGLU, "Dense 28672 N=512");
    bench_mlp_glue(rng, d_flush, flush_vecs, 2048,  28672, PF_GLU_SWIGLU, "Dense 28672 N=2048");

    bench_norm_glue(rng, d_flush, flush_vecs, 512,  2304, 6, "Inkling Norm6x N=512");
    bench_norm_glue(rng, d_flush, flush_vecs, 2048, 2304, 6, "Inkling Norm6x N=2048");
    bench_norm_glue(rng, d_flush, flush_vecs, 512,  8192, 3, "Dense Norm3x N=512");
    bench_norm_glue(rng, d_flush, flush_vecs, 2048, 8192, 3, "Dense Norm3x N=2048");

    CUDA_CK(cudaFree(d_flush));
    return 0;
}
