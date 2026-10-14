#include "rq4-mm.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// the kernels are NVIDIA PTX (cp.async, IMMA m16n8k32): no RQ4 on HIP/MUSA
bool ggml_cuda_rq4_supported(const ggml_tensor * op, int cc) {
    GGML_UNUSED(op); GGML_UNUSED(cc);
    return false;
}
void ggml_cuda_mul_mat_rq4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(src0); GGML_UNUSED(src1); GGML_UNUSED(dst);
    GGML_ABORT("RQ4 needs an NVIDIA SM80+ device");
}
#else
#include "rq4.cuh"

#include <cstdlib>

bool ggml_cuda_rq4_supported(const ggml_tensor * op, int cc) {
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    return cc >= GGML_CUDA_CC_AMPERE && op->op == GGML_OP_MUL_MAT && ggml_cuda_is_rq4(a->type) &&
        b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
        a->ne[2] == 1 && a->ne[3] == 1 &&
        a->ne[0] % RQ4_SB == 0 && a->ne[1] % rq4::GEMM_BM == 0 &&
        a->nb[1] == ggml_row_size(a->type, a->ne[0]) &&
        ggml_is_contiguous(b) && ggml_is_contiguous(op) &&
        ggml_nrows(b) <= 65535;   // tokens on grid.y of the activation kernel
}

// N above which the GEMM uses 128-wide token tiles (GGML_CUDA_RQ4_BN128_MIN overrides, for tuning)
static int rq4_bn128_min() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_RQ4_BN128_MIN");
        return e ? atoi(e) : 512;
    }();
    return v;
}

// Variant switches for A/B runs (read once per process):
//   GGML_CUDA_RQ4_MMV=1   the original per-chunk N <= 8 kernel (mmv); 3 = mmv_pair without the 4-blocks/SM
//                         register cap; unset or any other value = mmv_pair capped at 128 registers (MINB 4)
//   GGML_CUDA_RQ4_GEMM_WM=4  GEMM warp layout 4 (M) x 2 (N) instead of 2 x 4: each A fragment / RQ4NL table
//                            lookup feeds twice as many IMMAs
static int rq4_env_int(const char * name, int def) {
    const char * e = getenv(name);
    return e ? atoi(e) : def;
}
static int rq4_mmv_variant() {
    static const int v = rq4_env_int("GGML_CUDA_RQ4_MMV", 2);
    return v;
}
static bool rq4_gemm_wm4() {
    static const bool v = rq4_env_int("GGML_CUDA_RQ4_GEMM_WM", 2) == 4;
    return v;
}

template <class Codec, int WMW>
static void rq4_gemm(const uint8_t * W, int64_t rb, const int8_t * q, const float * s, float * Y,
                     int M, int N, int K, cudaStream_t stream) {
    if (N >= rq4_bn128_min()) {
        rq4::launch_gemm<Codec, 128, 3, WMW>(W, rb, q, s, Y, M, M, N, K, stream);
    } else {
        rq4::launch_gemm<Codec, 64, 3, WMW>(W, rb, q, s, Y, M, M, N, K, stream);
    }
}

template <class Codec>
static void rq4_mm(const uint8_t * W, int64_t rb, const int8_t * q, const float * s, float * Y,
                   int M, int N, int K, cudaStream_t stream) {
    if (N <= 8) {
        switch (rq4_mmv_variant()) {
            case 1:  rq4::launch_mmv<Codec>        (W, rb, q, s, Y, M, M, N, K, stream); break;
            case 3:  rq4::launch_mmv_pair<Codec, 1>(W, rb, q, s, Y, M, M, N, K, stream); break;
            default: rq4::launch_mmv_pair<Codec, 4>(W, rb, q, s, Y, M, M, N, K, stream); break;
        }
    } else if (rq4_gemm_wm4()) {
        rq4_gemm<Codec, 4>(W, rb, q, s, Y, M, N, K, stream);
    } else {
        rq4_gemm<Codec, 2>(W, rb, q, s, Y, M, N, K, stream);
    }
}

void ggml_cuda_mul_mat_rq4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_rq4_supported(dst, ggml_cuda_info().devices[ctx.device].cc));
    const int K = (int) src0->ne[0];
    const int M = (int) src0->ne[1];
    const int64_t N64 = ggml_nrows(src1);
    GGML_ASSERT(N64 <= 65535);   // tokens on grid.y of the activation kernel
    const int N = (int) N64;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int8_t> q(ctx.pool(), (size_t) N*K);
    ggml_cuda_pool_alloc<float>  s(ctx.pool(), (size_t) N*(K/RQ4_HB));
    rq4::launch_quantize_act((const float *) src1->data, K, q.get(), s.get(), K, N, stream);

    const uint8_t * W  = (const uint8_t *) src0->data;
    const int64_t   rb = (int64_t) src0->nb[1];
    float * Y = (float *) dst->data;
    switch (src0->type) {
        case GGML_TYPE_RQ4_G128:   rq4_mm<rq4::rq4_codec_nibble<128>>(W, rb, q.get(), s.get(), Y, M, N, K, stream); break;
        case GGML_TYPE_RQ4_G64:    rq4_mm<rq4::rq4_codec_nibble<64>> (W, rb, q.get(), s.get(), Y, M, N, K, stream); break;
        case GGML_TYPE_RQ4NL_G128: rq4_mm<rq4::rq4_codec_nl<128>>    (W, rb, q.get(), s.get(), Y, M, N, K, stream); break;
        case GGML_TYPE_RQ4NL_G64:  rq4_mm<rq4::rq4_codec_nl<64>>     (W, rb, q.get(), s.get(), Y, M, N, K, stream); break;
        default: GGML_ABORT("not an RQ4 type");
    }
    CUDA_CHECK(cudaGetLastError());
}
#endif // !GGML_USE_HIP && !GGML_USE_MUSA
