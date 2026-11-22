// INKNVFP4K: NVFP4 MoE decode launches (kernels in nvfp4moe-kern.cuh, graph matcher in ggml-cuda.cu)
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "nvfp4moe-kern.cuh"
#else
#include "nvfp4moe.cuh"
#endif

#include <cstdlib>
#include <cstring>

bool ggml_cuda_nvfp4moe_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_NVFP4MOE"); return !e || atoi(e) != 0; }();
    return on;
}

int ggml_cuda_nvfp4moe_f32act() {
    static const int m = [] {
        const char * e = getenv("GGML_CUDA_NVFP4MOE_ACT");
        return !e ? 0 : strcmp(e, "f32") == 0 ? 1 : strcmp(e, "f16") == 0 ? 2 : 0;
    }();
    return m;
}

int ggml_cuda_nvfp4moe_max_tok() {
    static const int n = [] {
        const char * e = getenv("GGML_CUDA_NVFP4MOE_MAXT");
        const int v = e ? atoi(e) : 4;
        return v < 1 ? 1 : v > NVFP4MOE_MAX_TOK ? NVFP4MOE_MAX_TOK : v;
    }();
    return n;
}

bool ggml_cuda_nvfp4moe_supported(int cc, int64_t K, int64_t F, int64_t n_used) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(cc); GGML_UNUSED(K); GGML_UNUSED(F); GGML_UNUSED(n_used);
    return false;  // NVIDIA only (__byte_perm, 32-lane warps)
#else
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_HOPPER &&
           K == 2304 && F == 1152 && n_used == 8;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <int MAXM>
static void nvfp4moe_launch_m(const nvfp4moe_params & P, cudaStream_t stream) {
    const dim3 grid_gu(P.F/32, P.T*P.n_used);
    if constexpr (MAXM == 1) {
        k_nvfp4moe_gu1w<8><<<dim3(P.F/64, P.T*P.n_used), 256, 0, stream>>>(P);
    } else {
        k_nvfp4moe_gu<MAXM, 9><<<grid_gu, 256, MAXM*(P.K/QK8_1)*sizeof(block_q8_1), stream>>>(P);
    }
    CUDA_CHECK(cudaGetLastError());
    if constexpr (MAXM == 1) {
        k_nvfp4moe_dn1<<<P.K/8, 256, 0, stream>>>(P);
    } else {
        const dim3 grid_dn(P.K/64, P.T*P.n_used);
        k_nvfp4moe_dnm<MAXM><<<grid_dn, 256, MAXM*(P.F/QK8_1)*sizeof(block_q8_1), stream>>>(P);
    }
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_nvfp4moe_launch(const nvfp4moe_params & P, cudaStream_t stream) {
    GGML_ASSERT(P.T >= 1 && P.T <= NVFP4MOE_MAX_TOK && P.n_used == 8 && P.K == 2304 && P.F == 1152);
    if (P.f32act == 2) {
        GGML_ASSERT(P.T == 1 && P.acth);
        k_nvfp4moe_gu1h<8><<<dim3(P.F/64, P.n_used), 256, 0, stream>>>(P);
        CUDA_CHECK(cudaGetLastError());
        k_nvfp4moe_dn1h<<<P.K/8, 256, 0, stream>>>(P);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (P.f32act) {
        GGML_ASSERT(P.T == 1 && P.actf);
        k_nvfp4moe_gu1f<8><<<dim3(P.F/64, P.n_used), 256, 0, stream>>>(P);
        CUDA_CHECK(cudaGetLastError());
        k_nvfp4moe_dn1f<<<P.K/8, 256, 0, stream>>>(P);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (P.T == 1) {
        nvfp4moe_launch_m<1>(P, stream);
    } else if (P.T <= 2) {
        nvfp4moe_launch_m<2>(P, stream);
    } else if (P.T <= 4) {
        nvfp4moe_launch_m<4>(P, stream);
    } else if (P.T <= 8) {
        nvfp4moe_launch_m<8>(P, stream);
    } else {
        nvfp4moe_launch_m<16>(P, stream);
    }
}
#else
void ggml_cuda_nvfp4moe_launch(const nvfp4moe_params &, cudaStream_t) {
    GGML_ABORT("NVFP4MOE: NVIDIA only");
}
#endif
