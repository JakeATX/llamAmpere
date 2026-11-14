#include "dense-tc.cuh"

#include <atomic>
#include <cstdlib>
#include <mutex>

// -1 not detected yet, 0 off, 1 on
static std::atomic<int> g_dense_tc[GGML_CUDA_MAX_DEVICES] = {};
static std::once_flag   g_dense_tc_once[GGML_CUDA_MAX_DEVICES];

static int dense_tc_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_CUDA_DENSE_TC");
        // Different prompt chunk sizes can change results after a cache restore. Keep opt-in until stable.
        return e ? atoi(e) : 0; // -1 = auto, 0 = off
    }();
    return mode;
}

static int64_t dense_tc_min() {
    static const int64_t v = [] {
        const char * e = getenv("GGML_CUDA_DENSE_TC_MIN");
        const long long v = e && *e ? atoll(e) : 4096; // empty or <= 0 keeps the 4096 floor (0 would send decode to bf16 cuBLAS)
        return (int64_t) (v > 0 ? v : 4096);
    }();
    return v;
}

// time n_iter GEMMs of n x n x n with the given input type and compute type; < 0 on failure
static float dense_tc_time(cublasHandle_t h, cudaStream_t s, void * a, void * b, void * c, int n,
        cudaDataType_t in_t, cudaDataType_t out_t, cublasComputeType_t ct, const void * alpha, const void * beta) {
    cudaEvent_t e0, e1;
    if (cudaEventCreate(&e0) != cudaSuccess || cudaEventCreate(&e1) != cudaSuccess) {
        return -1.0f;
    }
    float ms = -1.0f;
    bool ok = true;
    for (int it = 0; it < 13 && ok; ++it) { // 3 warm-up + 10 timed
        if (it == 3) {
            cudaEventRecord(e0, s);
        }
        ok = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, alpha, a, in_t, n, b, in_t, n, beta, c, out_t, n, ct,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
    }
    if (ok) {
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ok ? ms : -1.0f;
}

static void dense_tc_detect_impl(int device) {
    const int mode = dense_tc_mode();
    const int cc   = ggml_cuda_info().devices[device].cc;
    if (mode == 0 || !GGML_CUDA_CC_IS_NVIDIA(cc) || !fast_bf16_hardware_available(cc)) {
        g_dense_tc[device] = 0;
        return;
    }
    if (mode > 0) {
        g_dense_tc[device] = 1;
        GGML_LOG_INFO("%s: [DENSE_TC] device %d: bf16 dense prompt matmuls forced on (>= %lld tokens)\n", __func__,
                device, (long long) dense_tc_min());
        return;
    }
    // auto: compare bf16 in / FP32 accumulate against fp16 in / fp16 accumulate at 2048^3 (32 MiB of scratch).
    // The probe runs in a throwaway context so the library code it loads is unloaded with it; otherwise that code
    // stays resident and shrinks the memory left for other buffers planned after backend start.
#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    CUcontext prev_ctx = nullptr, probe_ctx = nullptr;
    CUdevice  cu_dev;
    cuCtxGetCurrent(&prev_ctx);
    bool own_ctx = cuDeviceGet(&cu_dev, device) == CUDA_SUCCESS;
#if CUDA_VERSION >= 13000
    own_ctx = own_ctx && cuCtxCreate(&probe_ctx, nullptr, 0, cu_dev) == CUDA_SUCCESS;
#else
    own_ctx = own_ctx && cuCtxCreate(&probe_ctx, 0, cu_dev) == CUDA_SUCCESS;
#endif
    if (!own_ctx) {
        probe_ctx = nullptr;
        cuCtxSetCurrent(prev_ctx);
        ggml_cuda_set_device(device);
    }
#else
    ggml_cuda_set_device(device);
#endif
    const int n = 2048;
    void * a = nullptr, * b = nullptr, * c = nullptr;
    cublasHandle_t h = nullptr;
    cudaStream_t   s = nullptr;
    float t32 = -1.0f, t16 = -1.0f;
    if (cudaMalloc(&a, (size_t) n*n*2) == cudaSuccess && cudaMalloc(&b, (size_t) n*n*2) == cudaSuccess &&
            cudaMalloc(&c, (size_t) n*n*4) == cudaSuccess && cudaMemset(a, 0, (size_t) n*n*2) == cudaSuccess &&
            cudaMemset(b, 0, (size_t) n*n*2) == cudaSuccess &&
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess &&
            cublasCreate(&h) == CUBLAS_STATUS_SUCCESS && cublasSetStream(h, s) == CUBLAS_STATUS_SUCCESS) {
        const float  one_f = 1.0f, zero_f = 0.0f;
        const half   one_h = 1.0f, zero_h = 0.0f;
        t32 = dense_tc_time(h, s, a, b, c, n, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, &one_f, &zero_f);
        t16 = dense_tc_time(h, s, a, b, c, n, CUDA_R_16F,  CUDA_R_16F, CUBLAS_COMPUTE_16F, &one_h, &zero_h);
    }
    if (h) { cublasDestroy(h); }
    if (s) { cudaStreamDestroy(s); }
    if (a) { cudaFree(a); }
    if (b) { cudaFree(b); }
    if (c) { cudaFree(c); }
    cudaGetLastError(); // a failed allocation leaves the path off, not an error
#if defined(GGML_USE_VMM) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (probe_ctx) {
        cuCtxDestroy(probe_ctx);
        cuCtxSetCurrent(prev_ctx);
        if (!prev_ctx) {
            cudaSetDevice(device);
        }
    }
#endif
    const float ratio = (t32 > 0.0f && t16 > 0.0f) ? t32/t16 : -1.0f;
    const bool  on    = ratio > 0.0f && ratio < 1.4f;
    g_dense_tc[device] = on ? 1 : 0;
    GGML_LOG_INFO("%s: [DENSE_TC] device %d: bf16/FP32-acc vs fp16/fp16-acc time ratio %.2f (%.3f / %.3f ms) -> %s\n",
            __func__, device, ratio, t32, t16, on ? "bf16 dense prompt matmuls on" : "off (int8 MMQ)");
}

void ggml_cuda_dense_tc_detect(int device) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        return;
    }
    std::call_once(g_dense_tc_once[device], [device] { dense_tc_detect_impl(device); });
}

bool ggml_cuda_dense_tc_wants(ggml_type type, int64_t ne11) {
    if (ne11 < dense_tc_min()) {
        return false;
    }
    switch (type) { // measured on Flash-Next shapes (test-backend-ops GGML_TEST_FN_DENSE_PERF)
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ4_XS:
            break;
        default:
            return false;
    }
    const int device = ggml_cuda_get_device();
    return device >= 0 && device < GGML_CUDA_MAX_DEVICES && g_dense_tc[device].load(std::memory_order_relaxed) == 1;
}
