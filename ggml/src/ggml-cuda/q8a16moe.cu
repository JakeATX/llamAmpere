// INKNVFP4K Q8A16 MoE: host side of the T = 2..8 Q8_0 MUL_MAT_ID on f32 activations (kernel in q8a16moe-kern.cuh).
// Opt-in GGML_CUDA_Q8A16_MOE=1 (default off: NEONKERN NK_Q8_0_TC won the same-job h2h on this slice, 2026-11-24); GGML_CUDA_Q8A16=0 turns it off too.
#include "q8a16moe.cuh"
#include "q8a16.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "q8a16moe-kern.cuh"
#endif

#include <climits>
#include <cstdlib>

#define Q8A16M_RW 3
#define Q8A16M_NW 1

bool ggml_cuda_q8a16_moe_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_Q8A16_MOE"); return e && atoi(e) != 0; }();
    return on && ggml_cuda_q8a16_enabled();
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

template <int T, int RW>
static void q8a16m_launch(const q8a16m_args & a, int nb, int64_t n_used, cudaStream_t st) {
    constexpr int rows = RW*Q8A16M_NW;
    const dim3 grid((unsigned) ((a.nrows + rows - 1) / rows), (unsigned) (n_used*T));
    k_q8a16_moe<T, RW, Q8A16M_NW><<<grid, 32*Q8A16M_NW, 0, st>>>(a, nb);
}

bool ggml_cuda_q8a16_moe_mul_mat_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        const ggml_tensor * ids, ggml_tensor * dst) {
    // src0 Q8_0 [K, N, E]; ids I32 [n_used, T]; src1 f32 [K, 1 or n_used, T]; dst f32 [N, n_used, T]
    if (!ggml_cuda_q8a16_moe_enabled()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_info().devices[ctx.device].warp_size != 32) {
        return false;
    }
    const int64_t n_used = ids->ne[0];
    const int64_t T      = ids->ne[1];
    if (src0->type != GGML_TYPE_Q8_0 || !src0->buffer || ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            src0->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || src0->ne[0] % QK8_0 != 0 || src0->ne[1] <= 0 || src0->ne[1] > INT_MAX ||
            src0->ne[3] != 1 || src0->nb[1] != (size_t) (src0->ne[0]/QK8_0)*sizeof(block_q8_0) || src0->nb[1] % 2 != 0 ||
            src0->nb[2] % 2 != 0 || (uintptr_t) src0->data % 2 != 0) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 || ids->ne[2] != 1 ||
            ids->ne[3] != 1 || n_used < 1 || n_used > Q8A16_MAX_GROUP || T < 2 || T > 8 || T > ggml_cuda_q8a16_max_t() ||
            n_used*T > Q8A16M_MAX_SLOTS) {
        // n_used and the MAXT knob as the T = 1 Q8A16 MUL_MAT_ID: decode and verify take the same arithmetic
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || src1->ne[0] != src0->ne[0] || (src1->ne[1] != 1 && src1->ne[1] != n_used) ||
            src1->ne[2] != T || src1->ne[3] != 1 || src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || src1->nb[2] % 16 != 0 ||
            (uintptr_t) src1->data % 16 != 0) {
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != src0->ne[1] || dst->ne[1] != n_used || dst->ne[2] != T || dst->ne[3] != 1 ||
            dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[2] % sizeof(float) != 0) {
        return false;
    }
    q8a16m_args a{};
    a.vx     = (const char *) src0->data;
    a.sexp   = (int64_t) src0->nb[2];
    a.srow   = (int64_t) src0->nb[1];
    a.x      = (const float *) src1->data;
    a.sxk    = src1->ne[1] == 1 ? 0 : (int64_t) (src1->nb[1] / sizeof(float));
    a.sxt    = (int64_t) (src1->nb[2] / sizeof(float));
    a.dst    = (float *) dst->data;
    a.sdk    = (int64_t) (dst->nb[1] / sizeof(float));
    a.sdt    = (int64_t) (dst->nb[2] / sizeof(float));
    a.ids    = (const int32_t *) ids->data;
    a.sid    = (int64_t) (ids->nb[1] / sizeof(int32_t));
    a.n_used = (int) n_used;
    a.nrows  = (int) src0->ne[1];

    const int nb = (int) (src0->ne[0] / QK8_0);
    cudaStream_t st = ctx.stream();
    switch (T) { // the widest member count the kernel compiles is T: registers sized for the actual batch
        case 2: q8a16m_launch<2, 6>(a, nb, n_used, st); break; // 6 rows per CTA fit up to T = 5 (RW*T <= 32), ~5% over 3
        case 3: q8a16m_launch<3, 6>(a, nb, n_used, st); break;
        case 4: q8a16m_launch<4, 6>(a, nb, n_used, st); break;
        case 5: q8a16m_launch<5, 6>(a, nb, n_used, st); break;
        case 6: q8a16m_launch<6, Q8A16M_RW>(a, nb, n_used, st); break;
        case 7: q8a16m_launch<7, Q8A16M_RW>(a, nb, n_used, st); break;
        case 8: q8a16m_launch<8, Q8A16M_RW>(a, nb, n_used, st); break;
        default: GGML_ABORT("q8a16 moe: batch width");
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_q8a16_moe_mul_mat_id(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    return false;
}

#endif
