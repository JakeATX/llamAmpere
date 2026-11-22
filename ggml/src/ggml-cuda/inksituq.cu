// [INKSITUQ] Inkling 2 MoE decode: k_ink_situ + the down MUL_MAT_ID's quantize_act128 in one launch (one launch fewer per
// MoE visit). Bit-identical: each lane computes the 8 SITU values quantize_act128_body would load (same ink_situ_elem as
// k_ink_situ, so the f32 values match the stored ones), stores them as the f32 output, and runs quantize_act128_regs (the
// body's own rotate / quantize / store code) on them. Grid, block and lane-to-element map are quantize_act128's.
// The graph matcher is in ggml-cuda.cu (ggml_cuda_try_fuse_ink_situ); the consumer hook in ggml_cuda_mul_mat_id_ai4.

#include "inksituq.cuh"
#include "ai4-kern.cuh"
#include "ai4-mm.cuh"
#include "ggml-backend-impl.h"

#include <atomic>

static std::atomic<int> g_situq_count[2];

int ggml_cuda_ink_situq_count(int which) {
    return g_situq_count[which & 1].load();
}

static __global__ void __launch_bounds__(128) k_ink_situq(const char * __restrict__ g, const char * __restrict__ u,
        float * __restrict__ dst, int8_t * __restrict__ q, float * __restrict__ s, const int K, const int ne1, const int N,
        const int64_t g_nb1, const int64_t g_nb2, const int64_t u_nb1, const int64_t u_nb2, const ink_situ_params p) {
    const int tok  = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int l    = lane & 15;
    const int cb   = (blockIdx.x*4 + (threadIdx.x >> 5))*2 + (lane >> 4);
    const bool live = cb*128 < K && tok < N;   // as quantize_act128_body: dead half-warps still run the shuffles
    float v[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    if (live) {
        const int64_t i1 = tok % ne1;
        const int64_t i2 = tok / ne1;
        const float * gr = (const float *) (g + i2*g_nb2 + i1*g_nb1) + cb*128 + l*8;
        const float * ur = (const float *) (u + i2*u_nb2 + i1*u_nb1) + cb*128 + l*8;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            v[e] = ink_situ_elem(gr[e], ur[e], p); // __fmul_rn result: the value k_ink_situ stores, never contracted
        }
        float * yr = dst + (int64_t) tok*K + cb*128 + l*8;
        *(float4 *) yr       = make_float4(v[0], v[1], v[2], v[3]);
        *(float4 *) (yr + 4) = make_float4(v[4], v[5], v[6], v[7]);
    }
    ai4k::quantize_act128_regs(v, live, l, cb, q, s, K, tok);
}

static bool situq_cuda_buf(const ggml_tensor * t, int device) {
    return t->buffer && t->buffer->buft == ggml_backend_cuda_buffer_type(device);
}

bool ggml_cuda_ink_situq_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u, ggml_tensor * out,
        const ink_situ_params & p, const ggml_tensor * down) {
    const ggml_tensor * w = down->src[0];
    const int K = (int) out->ne[0];
    const int64_t rows = out->ne[1]*out->ne[2];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    // main stream, no borrowed scratch, the consumer on the device-routed fast path (not the host-synced reference),
    // every tensor in this device's plain CUDA buffer (split / host / foreign buffers keep the unfused path)
    if (ctx.curr_stream_no != 0 || !ctx.stream_context().concurrent_events.empty() || ctx.scratch_borrowed ||
            ggml_cuda_ai4_id_needs_sync(cc, w->type) || !situq_cuda_buf(w, ctx.device) || !situq_cuda_buf(out, ctx.device) ||
            !situq_cuda_buf(g, ctx.device) || !situq_cuda_buf(u, ctx.device) || ((uintptr_t) out->data & 15) != 0 ||
            g->nb[0] != sizeof(float) || u->nb[0] != sizeof(float) || (g->nb[1] | g->nb[2] | u->nb[1] | u->nb[2]) % sizeof(float) != 0) {
        return false;
    }
    const size_t q_bytes = (size_t) rows*K;
    const size_t bytes   = q_bytes + (size_t) rows*(K/128)*sizeof(float);
    auto & e = ctx.ink_situq;
    if (e.dev != ctx.device || e.cap < bytes) {
        // never freed here: a captured graph may still replay kernels that point at the old buffer
        if (e.ptr != nullptr) {
            e.retired.push_back({ e.ptr, e.cap, e.dev });
            e.ptr = nullptr;
            e.cap = 0;
        }
        CUDA_CHECK(ggml_cuda_device_malloc((void **) &e.ptr, bytes, ctx.device));
        e.cap = bytes;
        e.dev = ctx.device;
    }
    e.src1     = out;
    e.consumer = down;
    e.data     = out->data;
    e.epoch    = ctx.graph_epoch;
    e.rows     = rows;
    e.K        = K;
    e.stream   = ctx.stream();
    const dim3 grid((K/128 + 7)/8, (unsigned) rows);
    k_ink_situq<<<grid, 128, 0, ctx.stream()>>>((const char *) g->data, (const char *) u->data, (float *) out->data,
        (int8_t *) e.ptr, (float *) (e.ptr + q_bytes), K, (int) out->ne[1], (int) rows, g->nb[1], g->nb[2], u->nb[1], u->nb[2], p);
    CUDA_CHECK(cudaGetLastError());
    g_situq_count[0]++;
    return true;
}

bool ggml_cuda_ink_situq_take(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, int64_t rows, int K,
        const int8_t ** q, const float ** s) {
    auto & e = ctx.ink_situq;
    const ggml_tensor * src1 = dst->src[1];
    if (e.epoch != ctx.graph_epoch || e.consumer != dst || e.src1 != src1 || e.data != src1->data || e.dev != ctx.device ||
            e.rows != rows || e.K != K || e.stream != ctx.stream() || ctx.curr_stream_no != 0) {
        return false;
    }
    e.epoch = 0;   // one-shot
    *q = (const int8_t *) e.ptr;
    *s = (const float *) (e.ptr + (size_t) rows*K);
    g_situq_count[1]++;
    return true;
}
