// INKNVFP4K NVFP4MT: 1..8 token NVFP4 MoE decode launches (kernels in nvfp4mt-kern.cuh), called by ggml_cuda_nvfp4moe_launch
#include "nvfp4mt.cuh"

#include <cstdlib>

bool ggml_cuda_nvfp4mt_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_NVFP4MT"); return !e || atoi(e) != 0; }();
    return on;
}

bool ggml_cuda_nvfp4mt_t1() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_NVFP4MT_T1"); return !e || atoi(e) != 0; }();
    return ggml_cuda_nvfp4mt_enabled() && on;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "nvfp4mt-kern.cuh"

template <int T, int NW, int FPW, int D, int DNW, int SPW, int DD>
static void nvfp4mt_launch_t(const nvfp4moe_params & P, cudaStream_t stream) {
    constexpr size_t sg = nvfp4mt_gu_smem<T, NW, FPW, D>(), sd = nvfp4mt_dn_smem<T, DNW, SPW, DD>();
    static_assert(sg <= 99*1024 && sd <= 99*1024, "NVFP4MT shared memory");
    static bool init[GGML_CUDA_MAX_DEVICES] = {};  // the attribute is per device
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (!init[dev]) {
        CUDA_CHECK(cudaFuncSetAttribute(k_nvfp4mt_gu<T, NW, FPW, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sg));
        CUDA_CHECK(cudaFuncSetAttribute(k_nvfp4mt_dn<T, DNW, SPW, DD>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sd));
        init[dev] = true;
    }
    k_nvfp4moe_xq<<<(T*(P.K/QK8_1) + 7)/8, 256, 0, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());
    k_nvfp4mt_gu<T, NW, FPW, D><<<dim3(P.F/(NW*FPW), T*P.n_used), NW*32, sg, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());
    k_nvfp4mt_dn<T, DNW, SPW, DD><<<dim3(P.K/(DNW*SPW*4), T*P.n_used), DNW*32, sd, stream>>>(P);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_nvfp4mt_accepts(const nvfp4moe_params & P) {
    // GGML_CUDA_NVFP4MOE_GORD=0 / _EXACT=0 ask for the older T > 1 kernels: leave those to nvfp4moe_launch
    return ggml_cuda_nvfp4mt_enabled() && ggml_cuda_nvfp4moe_gord() && ggml_cuda_nvfp4moe_exact() &&
        (P.T >= 2 || (P.T == 1 && !P.write_sum && ggml_cuda_nvfp4mt_t1())) && P.T <= 8 && P.xq && !P.f32act && P.n_used == 8 &&
        P.K == 2304 && P.F == 1152 && P.gu_nb1 == NVFP4MT_ROW_GU && P.dn_nb1 == NVFP4MT_ROW_DN && P.gu_nb2 % 16 == 0 &&
        P.dn_nb2 % 16 == 0 && (uintptr_t) P.w_gu % 16 == 0 && (uintptr_t) P.w_dn % 16 == 0 && (uintptr_t) P.act % 4 == 0 &&
        (uintptr_t) P.xq % 4 == 0;
}

bool ggml_cuda_nvfp4mt_launch(const nvfp4moe_params & P, cudaStream_t stream) {
    if (!ggml_cuda_nvfp4mt_accepts(P)) {
        return false;
    }
    switch (P.T) {
        case 1:  nvfp4mt_launch_t<1, 4, 8, 2, 4, 8, 2>(P, stream); break;
        case 2:  nvfp4mt_launch_t<2, 4, 8, 3, 4, 8, 3>(P, stream); break;
        case 3:  nvfp4mt_launch_t<3, 4, 8, 3, 4, 8, 3>(P, stream); break;
        case 4:  nvfp4mt_launch_t<4, 4, 8, 3, 4, 8, 3>(P, stream); break;
        case 5:  nvfp4mt_launch_t<5, 4, 8, 3, 4, 8, 3>(P, stream); break;
        case 6:  nvfp4mt_launch_t<6, 4, 8, 2, 4, 8, 3>(P, stream); break;
        case 7:  nvfp4mt_launch_t<7, 4, 8, 2, 4, 8, 3>(P, stream); break;
        default: nvfp4mt_launch_t<8, 4, 8, 2, 4, 8, 3>(P, stream); break;
    }
    return true;
}
#else
bool ggml_cuda_nvfp4mt_accepts(const nvfp4moe_params &) {
    return false;
}
bool ggml_cuda_nvfp4mt_launch(const nvfp4moe_params &, cudaStream_t) {
    return false;
}
#endif
