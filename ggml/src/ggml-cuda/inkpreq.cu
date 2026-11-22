#include "inkpreq.cuh"

#include <memory>

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
thread_local std::unique_ptr<ggml_cuda_pool_alloc<int8_t>> g_preq_q;
thread_local std::unique_ptr<ggml_cuda_pool_alloc<float>>  g_preq_s;
}

void ggml_cuda_ink_ai4_preq_set(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t * q,
        const float * s) {
    g_preq = { consumer, x, rows, K, xsh, q, s };
}

void ggml_cuda_ink_ai4_preq_alloc(ggml_backend_cuda_context & ctx, int64_t rows, int K, int xsh, int8_t ** q, float ** s) {
    ggml_cuda_ink_ai4_preq_release();
    g_preq_q = std::make_unique<ggml_cuda_pool_alloc<int8_t>>(ctx.pool(), (size_t) (rows*K));
    g_preq_s = std::make_unique<ggml_cuda_pool_alloc<float>>(ctx.pool(), (size_t) (rows*(K >> (7 + xsh))));
    *q = g_preq_q->get();
    *s = g_preq_s->get();
}

void ggml_cuda_ink_ai4_preq_release() {
    g_preq = ink_preq{};
    g_preq_s.reset();
    g_preq_q.reset();
}

bool ggml_cuda_ink_ai4_preq_take(const ggml_tensor * consumer, const void * x, int64_t rows, int K, int xsh, const int8_t ** q,
        const float ** s) {
    const ink_preq p = g_preq;
    if (!p.consumer || p.consumer != consumer) {
        return false; // not for this node: keep it for its consumer
    }
    g_preq.consumer = nullptr; // once (the buffers stay until the next set or release)
    if (p.x != x || p.rows != rows || p.K != K || p.xsh != xsh) {
        return false;
    }
    *q = p.q;
    *s = p.s;
    return true;
}
