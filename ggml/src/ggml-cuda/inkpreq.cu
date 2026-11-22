#include "inkpreq.cuh"

#include <mutex>
#include <unordered_map>

namespace {
struct ink_preq {
    const ggml_tensor * consumer = nullptr;
    const void *   x    = nullptr;
    int64_t        rows = 0;
    int            K    = 0;
    int            xsh  = 0;
    const int8_t * q    = nullptr;
    const float *  s    = nullptr;
};
thread_local ink_preq g_preq;

// one fixed-size device buffer per backend context, allocated on first use and freed with the context: a captured CUDA
// graph may replay kernels that use it (as ai_prep_cache / q8_cache), and it never touches the LIFO pool
constexpr size_t INK_PREQ_Q_BYTES = 8192;
constexpr size_t INK_PREQ_S_BYTES = 8192;
struct ink_preq_buf {
    int    dev = -1;
    char * p   = nullptr;
};
std::mutex g_preq_mu;
std::unordered_map<const ggml_backend_cuda_context *, ink_preq_buf> g_preq_bufs;
}

void ggml_cuda_ink_ai4_preq_set(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t * q,
        const float * s) {
    g_preq = { consumer, x, rows, K, xsh, q, s };
}

bool ggml_cuda_ink_ai4_preq_alloc(ggml_backend_cuda_context & ctx, int64_t rows, int K, int xsh, int8_t ** q, float ** s) {
    ggml_cuda_ink_ai4_preq_release();
    if (rows <= 0 || (size_t) (rows*K) > INK_PREQ_Q_BYTES || (size_t) (rows*(K >> (7 + xsh)))*sizeof(float) > INK_PREQ_S_BYTES) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_preq_mu);
    ink_preq_buf & b = g_preq_bufs[&ctx];
    if (b.p == nullptr || b.dev != ctx.device) {
        if (b.p != nullptr) {
            ggml_cuda_set_device(b.dev);
            CUDA_CHECK(cudaFree(b.p));
            b.p = nullptr;
        }
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(ggml_cuda_device_malloc((void **) &b.p, INK_PREQ_Q_BYTES + INK_PREQ_S_BYTES, ctx.device));
        b.dev = ctx.device;
    }
    *q = (int8_t *) b.p;
    *s = (float *) (b.p + INK_PREQ_Q_BYTES);
    return true;
}

void ggml_cuda_ink_ai4_preq_release() {
    g_preq = ink_preq{};
}

void ggml_cuda_ink_ai4_preq_free(const ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(g_preq_mu);
    auto it = g_preq_bufs.find(ctx);
    if (it == g_preq_bufs.end()) {
        return;
    }
    if (it->second.p != nullptr) {
        ggml_cuda_set_device(it->second.dev);
        cudaFree(it->second.p);
    }
    g_preq_bufs.erase(it);
}

bool ggml_cuda_ink_ai4_preq_take(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t ** q,
        const float ** s) {
    const ink_preq p = g_preq;
    if (!p.consumer || p.consumer != consumer) {
        return false; // not for this node: keep it for its consumer
    }
    g_preq.consumer = nullptr; // once
    if (p.x != x || p.rows != rows || p.K != K || p.xsh != xsh) {
        return false;
    }
    *q = p.q;
    *s = p.s;
    return true;
}
