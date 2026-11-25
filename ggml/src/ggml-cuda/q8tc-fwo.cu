// SURGE3 FUSE-WO: Q8TC attn_output matvec + the Inkling attn-out short-conv ring epilogue in one launch (q8tc.cuh)
#include "q8tc.cuh"
#include "q8a16.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "q8tc-kern.cuh"
#endif

#include <cstdlib>

bool ggml_cuda_q8tc_fusewo_on() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_Q8TC_FUSEWO"); return e && atoi(e) == 1; }();
    return on;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

template <int NW, bool HILO, int MINB>
static void q8tc_fwo_launch(const q8a16_args & a, int T, int nb, const q8tc_epi & e, cudaStream_t st) {
    static_assert(NW*16*68 >= NW*128 + 16*8, "epilogue tile past the reduction buffer");
    if constexpr (HILO) {
        if (ggml_cuda_q8tc_ia()) {
            k_q8tc3e<NW, HILO, MINB, true><<<a.blk_end[0], 32*NW, NW*16*68*sizeof(uint32_t), st>>>(a, nb, T, e);
            return;
        }
    }
    k_q8tc3e<NW, HILO, MINB><<<a.blk_end[0], 32*NW, NW*16*68*sizeof(uint32_t), st>>>(a, nb, T, e);
}

bool ggml_cuda_q8tc_fwo(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const q8tc_epi & e) {
    if (!ggml_cuda_q8tc_fusewo_on() || !ggml_cuda_q8a16_enabled()) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || ggml_cuda_info().devices[ctx.device].warp_size != 32) {
        return false;
    }
    // the Q8A16 plain-product conditions (q8a16_weight_ok + ggml_cuda_q8a16_mul_mat)
    const int64_t T = src1->ne[1];
    if (src0->type != GGML_TYPE_Q8_0 || !src0->buffer || ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            src0->nb[0] != ggml_type_size(GGML_TYPE_Q8_0) || src0->ne[0] % QK8_0 != 0 || src0->ne[1] < 1 || src0->ne[2] != 1 ||
            src0->ne[3] != 1 || src0->nb[1] != (size_t) (src0->ne[0]/QK8_0)*sizeof(block_q8_0) ||
            T < 1 || T > 8 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src1->ne[0] != src0->ne[0] ||
            src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[0] != src0->ne[1] || dst->ne[1] != T || dst->ne[2] != 1 ||
            dst->ne[3] != 1 || src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 ||
            dst->nb[1] % sizeof(float) != 0 || (uintptr_t) src1->data % 16 != 0) {
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
    const int nb = (int) (src0->ne[0] / QK8_0);
    const int mode = ggml_cuda_q8tc_prep(ctx, a, (int) T, nb, src0, false);
    if (mode == 0) {
        return false;
    }
    cudaStream_t st = ctx.stream();
    const bool few = nb/8 <= 9;
    if (mode == 1) {
        few ? q8tc_fwo_launch<3, false, 4>(a, (int) T, nb, e, st) : q8tc_fwo_launch<6, false, 2>(a, (int) T, nb, e, st);
    } else {
        few ? q8tc_fwo_launch<3, true, 4>(a, (int) T, nb, e, st) : q8tc_fwo_launch<6, true, 2>(a, (int) T, nb, e, st);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_q8tc_fwo(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *, const q8tc_epi &) {
    return false;
}

#endif
