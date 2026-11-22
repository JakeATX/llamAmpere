// INKNVFP4K Q8A16: host side of the Q8_0 f32-activation decode matvec (see q8a16-kern.cuh, lanes/INKNVFP4K.md)
#include "q8a16.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "q8a16-kern.cuh"
#endif

#include <climits>
#include <cstdlib>

// CTA shape per batch width T: RW rows per warp (they share each activation load), NW warps per CTA (tuned on a 3090)
#define Q8A16_RW1 2
#define Q8A16_NW1 1
#define Q8A16_RWN 3
#define Q8A16_NWN 1

bool ggml_cuda_q8a16_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_Q8A16"); return !e || atoi(e) != 0; }();
    return on;
}

[[maybe_unused]] static int ggml_cuda_q8a16_max_t() {
    static const int m = [] {
        const char * e = getenv("GGML_CUDA_Q8A16_MAXT");
        const int v = e ? atoi(e) : 8;
        return v < 0 ? 0 : v > 8 ? 8 : v;
    }();
    return m;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static bool q8a16_device_ok(const ggml_backend_cuda_context & ctx) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_info().devices[ctx.device].warp_size == 32;
}

// weights: a plain (non-compute) Q8_0 matrix with contiguous 34-byte blocks
static bool q8a16_weight_ok(const ggml_tensor * w) {
    return w->type == GGML_TYPE_Q8_0 && w->buffer && ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        w->nb[0] == ggml_type_size(GGML_TYPE_Q8_0) && w->ne[0] % QK8_0 == 0 && w->ne[0] > 0 && w->ne[1] > 0 &&
        w->ne[1] <= INT_MAX && w->ne[2] == 1 && w->ne[3] == 1 &&
        w->nb[1] == (size_t) (w->ne[0]/QK8_0)*sizeof(block_q8_0) && (uintptr_t) w->data % 2 == 0;
}

template <int T, int RW, int NW>
static void q8a16_launch(q8a16_args & a, int nb, cudaStream_t stream) {
    uint32_t blk = 0;
    for (int k = 0; k < a.n; ++k) {
        blk += (uint32_t) ((a.nrows[k] + RW*NW - 1) / (RW*NW));
        a.blk_end[k] = blk;
    }
    k_q8a16<T, RW, NW><<<blk, 32*NW, 0, stream>>>(a, nb);
}

static void q8a16_dispatch(q8a16_args & a, int T, int nb, cudaStream_t stream) {
    switch (T) {
        case 1: q8a16_launch<1, Q8A16_RW1, Q8A16_NW1>(a, nb, stream); break;
        case 2: q8a16_launch<2, Q8A16_RWN, Q8A16_NWN>(a, nb, stream); break;
        case 3: q8a16_launch<3, Q8A16_RWN, Q8A16_NWN>(a, nb, stream); break;
        case 4: q8a16_launch<4, Q8A16_RWN, Q8A16_NWN>(a, nb, stream); break;
        case 5: q8a16_launch<5, Q8A16_RWN, Q8A16_NWN>(a, nb, stream); break;
        case 6: q8a16_launch<6, Q8A16_RWN, Q8A16_NWN>(a, nb, stream); break;
        case 7: q8a16_launch<7, Q8A16_RWN, Q8A16_NWN>(a, nb, stream); break;
        case 8: q8a16_launch<8, Q8A16_RWN, Q8A16_NWN>(a, nb, stream); break;
        default: GGML_ABORT("q8a16: batch width");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_q8a16_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (!ggml_cuda_q8a16_enabled() || !q8a16_device_ok(ctx) || !q8a16_weight_ok(src0)) {
        return false;
    }
    const int64_t T = src1->ne[1];
    if (T < 1 || T > ggml_cuda_q8a16_max_t() || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[0] != src0->ne[1] || dst->ne[1] != T ||
            dst->ne[2] != 1 || dst->ne[3] != 1 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) ||
            src1->nb[1] % 16 != 0 || dst->nb[1] % sizeof(float) != 0 || (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    q8a16_args a{};
    a.n = 1;
    a.vx[0]    = (const char *) src0->data;
    a.x[0]     = (const float *) src1->data;
    a.dst[0]   = (float *) dst->data;
    a.srow[0]  = (int64_t) src0->nb[1];
    a.nrows[0] = (int) src0->ne[1];
    a.sx[0] = (int64_t) (src1->nb[1] / sizeof(float));
    a.sd[0] = (int64_t) (dst->nb[1] / sizeof(float));
    q8a16_dispatch(a, (int) T, (int) (src0->ne[0] / QK8_0), ctx.stream());
    return true;
}

bool ggml_cuda_q8a16_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    // one token: dst[:, k] = expert ids[k] of src0 x src1 column (k % ne11), k < n_used
    if (!ggml_cuda_q8a16_enabled() || ggml_cuda_q8a16_max_t() < 1 || !q8a16_device_ok(ctx) || src0->type != GGML_TYPE_Q8_0 ||
            !src0->buffer || ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            src0->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || src0->ne[0] % QK8_0 != 0 || src0->ne[1] <= 0 || src0->ne[1] > INT_MAX ||
            src0->ne[3] != 1 || src0->nb[1] != (size_t) (src0->ne[0]/QK8_0)*sizeof(block_q8_0) || src0->nb[2] % 2 != 0 ||
            (uintptr_t) src0->data % 2 != 0 || ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[0] < 1 ||
            ids->ne[0] > Q8A16_MAX_GROUP || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src1->ne[0] != src0->ne[0] || src1->ne[2] != 1 || src1->ne[3] != 1 || (src1->ne[1] != 1 && src1->ne[1] != ids->ne[0]) ||
            src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || (uintptr_t) src1->data % 16 != 0 ||
            dst->ne[0] != src0->ne[1] || dst->ne[1] != ids->ne[0] || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0) {
        return false;
    }
    q8a16_args a{};
    a.n    = (int) ids->ne[0];
    a.ids  = (const int32_t *) ids->data;
    a.sexp = (int64_t) src0->nb[2];
    for (int k = 0; k < a.n; ++k) {
        a.vx[k]    = (const char *) src0->data;
        a.x[k]     = (const float *) ((const char *) src1->data + (src1->ne[1] == 1 ? 0 : k)*src1->nb[1]);
        a.dst[k]   = (float *) ((char *) dst->data + k*dst->nb[1]);
        a.srow[k]  = (int64_t) src0->nb[1];
        a.nrows[k] = (int) src0->ne[1];
    }
    q8a16_dispatch(a, 1, (int) (src0->ne[0] / QK8_0), ctx.stream());
    return true;
}

bool ggml_cuda_q8a16_group(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n, bool check_only) {
    if (!ggml_cuda_q8a16_enabled() || ggml_cuda_q8a16_max_t() < 1 || n < 1 || n > Q8A16_MAX_GROUP || !q8a16_device_ok(ctx)) {
        return false;
    }
    const int64_t ne10 = dsts[0]->src[0]->ne[0];
    const int64_t T    = dsts[0]->src[1]->ne[1];  // 1 (decode) or a verify width; every member the same
    if (T < 1 || T > ggml_cuda_q8a16_max_t()) {
        return false;
    }
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * d  = dsts[k];
        const ggml_tensor * w  = d->src[0];
        const ggml_tensor * x  = d->src[1];
        if (!q8a16_weight_ok(w) || w->ne[0] != ne10 || x->type != GGML_TYPE_F32 || x->ne[0] != ne10 || x->ne[1] != T ||
                x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 || (uintptr_t) x->data % 16 != 0 ||
                d->type != GGML_TYPE_F32 || d->ne[0] != w->ne[1] || d->ne[1] != T || d->ne[2] != 1 || d->ne[3] != 1 ||
                d->nb[0] != sizeof(float) || d->nb[1] % sizeof(float) != 0) {
            return false;
        }
    }
    if (check_only) {
        return true;
    }
    q8a16_args a{};
    a.n = n;
    for (int k = 0; k < n; ++k) {
        a.vx[k]    = (const char *) dsts[k]->src[0]->data;
        a.x[k]     = (const float *) dsts[k]->src[1]->data;
        a.dst[k]   = (float *) dsts[k]->data;
        a.srow[k]  = (int64_t) dsts[k]->src[0]->nb[1];
        a.nrows[k] = (int) dsts[k]->src[0]->ne[1];
        a.sx[k]    = (int64_t) (dsts[k]->src[1]->nb[1] / sizeof(float));
        a.sd[k]    = (int64_t) (dsts[k]->nb[1] / sizeof(float));
    }
    q8a16_dispatch(a, (int) T, (int) (ne10 / QK8_0), ctx.stream());
    return true;
}

#else

bool ggml_cuda_q8a16_mul_mat(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }
bool ggml_cuda_q8a16_group(ggml_backend_cuda_context &, ggml_tensor * const *, int, bool) { return false; }
bool ggml_cuda_q8a16_mul_mat_id(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return false; }

#endif
