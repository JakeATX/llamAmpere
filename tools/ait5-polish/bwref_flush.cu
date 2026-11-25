#include "bwref_flush.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BWREF_CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(1); \
    } \
} while (0)

// Read-only 128-bit streaming reduction: reads every 16B vector via __ldg (evicts L2 cleanly, 0 dirty writebacks).
static __global__ void k_ro_stream_ldg128(const uint4 * __restrict__ src, size_t n_vec4, uint32_t * __restrict__ sink) {
    const size_t tid    = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x * blockDim.x;
    uint32_t acc = 0;
    for (size_t i = tid; i < n_vec4; i += stride) {
        const uint4 v = __ldg(src + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    for (int o = 16; o > 0; o >>= 1) {
        acc ^= __shfl_xor_sync(0xffffffffu, acc, o, 32);
    }
    if (threadIdx.x == 0 && blockIdx.x == 0 && acc == 0xdeadbeefu) {
        *sink = acc;
    }
}

struct bwref_ctx {
    int         device   = 0;
    uint4     * d_buf    = nullptr;
    uint32_t  * d_sink   = nullptr;
    size_t      bwref_n4 = 0; // 256 MiB / 16
    size_t      flush_n4 = 0; //  64 MiB / 16
    int         sm_count = 82;
    cudaEvent_t ev0      = nullptr;
    cudaEvent_t ev1      = nullptr;
};

bwref_ctx * bwref_init(int device) {
    BWREF_CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    BWREF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    auto * ctx = new bwref_ctx();
    ctx->device   = device;
    ctx->sm_count = prop.multiProcessorCount;
    ctx->bwref_n4 = (size_t) 256 * 1024 * 1024 / sizeof(uint4);
    ctx->flush_n4 = (size_t)  64 * 1024 * 1024 / sizeof(uint4);

    BWREF_CUDA_CHECK(cudaMalloc(&ctx->d_buf,  ctx->bwref_n4 * sizeof(uint4)));
    BWREF_CUDA_CHECK(cudaMalloc(&ctx->d_sink, sizeof(uint32_t)));
    BWREF_CUDA_CHECK(cudaMemset(ctx->d_buf, 0x5A, ctx->bwref_n4 * sizeof(uint4)));
    BWREF_CUDA_CHECK(cudaMemset(ctx->d_sink, 0, sizeof(uint32_t)));
    BWREF_CUDA_CHECK(cudaEventCreate(&ctx->ev0));
    BWREF_CUDA_CHECK(cudaEventCreate(&ctx->ev1));
    BWREF_CUDA_CHECK(cudaDeviceSynchronize());
    return ctx;
}

void bwref_free(bwref_ctx * ctx) {
    if (!ctx) return;
    if (ctx->ev0)    cudaEventDestroy(ctx->ev0);
    if (ctx->ev1)    cudaEventDestroy(ctx->ev1);
    if (ctx->d_buf)  cudaFree(ctx->d_buf);
    if (ctx->d_sink) cudaFree(ctx->d_sink);
    delete ctx;
}

void bwref_flush_l2(bwref_ctx * ctx) {
    const int grid = ctx->sm_count * 8;
    k_ro_stream_ldg128<<<grid, 256>>>(ctx->d_buf, ctx->flush_n4, ctx->d_sink);
    BWREF_CUDA_CHECK(cudaGetLastError());
    BWREF_CUDA_CHECK(cudaDeviceSynchronize());
}

double bwref_measure_gbps(bwref_ctx * ctx, int iters) {
    const int grid = ctx->sm_count * 16;
    for (int i = 0; i < 5; ++i) {
        k_ro_stream_ldg128<<<grid, 256>>>(ctx->d_buf, ctx->bwref_n4, ctx->d_sink);
    }
    BWREF_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> ms_list;
    ms_list.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        BWREF_CUDA_CHECK(cudaEventRecord(ctx->ev0));
        k_ro_stream_ldg128<<<grid, 256>>>(ctx->d_buf, ctx->bwref_n4, ctx->d_sink);
        BWREF_CUDA_CHECK(cudaEventRecord(ctx->ev1));
        BWREF_CUDA_CHECK(cudaEventSynchronize(ctx->ev1));
        float ms = 0.0f;
        BWREF_CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev0, ctx->ev1));
        ms_list.push_back((double) ms);
    }
    std::sort(ms_list.begin(), ms_list.end());
    const double med_ms = ms_list[ms_list.size() / 2];
    const double bytes  = (double) ctx->bwref_n4 * sizeof(uint4);
    return (bytes / (med_ms * 1e-3)) / 1e9;
}

void bwref_print_device(int device) {
    cudaDeviceProp prop{};
    BWREF_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device %d: %s (SM %d.%d, %d SMs, L2 %.1f MiB, VRAM %.2f GiB)\n",
           device, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize / (1024.0 * 1024.0),
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
}

void bwref_timer_start(bwref_ctx * ctx, void * cuda_stream) {
    cudaStream_t st = static_cast<cudaStream_t>(cuda_stream);
    BWREF_CUDA_CHECK(cudaEventRecord(ctx->ev0, st));
}

double bwref_timer_stop_us(bwref_ctx * ctx, void * cuda_stream) {
    cudaStream_t st = static_cast<cudaStream_t>(cuda_stream);
    BWREF_CUDA_CHECK(cudaEventRecord(ctx->ev1, st));
    BWREF_CUDA_CHECK(cudaEventSynchronize(ctx->ev1));
    float ms = 0.0f;
    BWREF_CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev0, ctx->ev1));
    return (double) ms * 1000.0;
}
