// AI+ cooperative 2-warp-per-slab BN=128 prefill GEMM (own TU). Opt-in: GGML_CUDA_AI4_COOP128=1 (MoE expert tiles),
// GGML_CUDA_AI4_COOP128_DENSE=1 (dense AIT4_U MUL_MAT).
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "ai4-coop128.cuh"

#include <cstdlib>

namespace ai4_coop {

bool env_coop128_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AI4_COOP128");
        return e && atoi(e) != 0;
    }();
    return on;
}

// dense AIT4_U MUL_MAT (own switch; GGML_CUDA_AI4_COOP128 alone stays MoE-only)
bool env_coop128_dense_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_AI4_COOP128_DENSE");
        return e && atoi(e) != 0;
    }();
    return on;
}

void dispatch_coop128_ait4(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, int xsh, bool is_id,
        const ai4k::idmap & idm, int max_tiles, cudaStream_t stream) {
    if (is_id) {
        if (xsh) {
            launch_gemm_s8_coop128<ai4k::LAY_AIT4_U, 1, true, 2, 2>(W, Xq, Xs, Y, ldy, M, N, K, stream, idm, max_tiles);
        } else {
            launch_gemm_s8_coop128<ai4k::LAY_AIT4_U, 0, true, 2, 2>(W, Xq, Xs, Y, ldy, M, N, K, stream, idm, max_tiles);
        }
    } else {
        if (xsh) {
            launch_gemm_s8_coop128<ai4k::LAY_AIT4_U, 1, false, 2, 2>(W, Xq, Xs, Y, ldy, M, N, K, stream);
        } else {
            launch_gemm_s8_coop128<ai4k::LAY_AIT4_U, 0, false, 2, 2>(W, Xq, Xs, Y, ldy, M, N, K, stream);
        }
    }
}

} // namespace ai4_coop
#else
#include "ai4-kern.cuh"
namespace ai4_coop {
bool env_coop128_enabled() { return false; }
bool env_coop128_dense_enabled() { return false; }
void dispatch_coop128_ait4(const uint8_t *, const int8_t *, const float *, float *, int64_t, int, int, int, int, bool, const ai4k::idmap &, int, cudaStream_t) {}
} // namespace ai4_coop
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
