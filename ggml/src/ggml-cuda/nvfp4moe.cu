// INKNVFP4K: NVFP4 MoE decode launches (kernels in nvfp4moe-kern.cuh, graph matcher in ggml-cuda.cu)
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "nvfp4moe-kern.cuh"
#else
#include "nvfp4moe.cuh"
#endif

#include "nvfp4mt.cuh"
#include "nk-nvfp4-tc.cuh"

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
        const int v = e ? atoi(e) : 8;
        return v < 1 ? 1 : v > NVFP4MOE_MAX_TOK ? NVFP4MOE_MAX_TOK : v;
    }();
    return n;
}

bool ggml_cuda_nvfp4moe_preq() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_NVFP4MOE_PREQ"); return !e || atoi(e) != 0; }();
    return on;
}

bool ggml_cuda_nvfp4moe_gord() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_NVFP4MOE_GORD"); return !e || atoi(e) != 0; }();
    return on;
}

bool ggml_cuda_nvfp4moe_exact() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_NVFP4MOE_EXACT"); return !e || atoi(e) != 0; }();
    return on;
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
template <int MAXM, bool GO>
static void nvfp4moe_launch_mg(const nvfp4moe_params & P, cudaStream_t stream) {
    const dim3 grid_gu(P.F/32, P.T*P.n_used);
    const size_t smem_gu = MAXM*(P.K/QK8_1)*sizeof(block_q8_1);
    if constexpr (MAXM == 1) {
        k_nvfp4moe_gu1w<8><<<dim3(P.F/64, P.T*P.n_used), 256, 0, stream>>>(P);
    } else if (P.xq) {
        k_nvfp4moe_xq<<<(P.T*(P.K/QK8_1) + 7)/8, 256, 0, stream>>>(P);
        CUDA_CHECK(cudaGetLastError());
        k_nvfp4moe_gu<MAXM, 9, true, GO><<<grid_gu, 256, smem_gu, stream>>>(P);
    } else {
        k_nvfp4moe_gu<MAXM, 9, false, GO><<<grid_gu, 256, smem_gu, stream>>>(P);
    }
    CUDA_CHECK(cudaGetLastError());
    if constexpr (MAXM == 1) {
        k_nvfp4moe_dn1<<<P.K/8, 256, 0, stream>>>(P);
    } else {
        const dim3 grid_dn(P.K/64, P.T*P.n_used);
        k_nvfp4moe_dnm<MAXM, GO><<<grid_dn, 256, MAXM*(P.F/QK8_1)*sizeof(block_q8_1), stream>>>(P);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int MAXM>
static void nvfp4moe_launch_m(const nvfp4moe_params & P, cudaStream_t stream) {
    if (MAXM > 1 && ggml_cuda_nvfp4moe_gord()) {
        nvfp4moe_launch_mg<MAXM, true>(P, stream);
    } else {
        nvfp4moe_launch_mg<MAXM, false>(P, stream);
    }
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
    if (ggml_cuda_nk_nvfp4_tc_launch(P, stream)) {
        return;  // NEONKERN: sm_86 int8 tensor-core NVFP4 MoE for T = 1..8 (GGML_CUDA_NK_NVFP4_TC=1)
    }
    if (P.T == 1) {
        if (!ggml_cuda_nvfp4mt_launch(P, stream)) {  // NVFP4MT at one token: only without write_sum (e_out, caller sums)
            nvfp4moe_launch_m<1>(P, stream);
        }
    } else if (ggml_cuda_nvfp4mt_launch(P, stream)) {
        return;  // NVFP4MT: 2..8 tokens, weights streamed once per visit
    } else if (ggml_cuda_nvfp4moe_exact() && P.T <= 8) {
        switch (P.T) {
            case 2: nvfp4moe_launch_m<2>(P, stream); break;
            case 3: nvfp4moe_launch_m<3>(P, stream); break;
            case 4: nvfp4moe_launch_m<4>(P, stream); break;
            case 5: nvfp4moe_launch_m<5>(P, stream); break;
            case 6: nvfp4moe_launch_m<6>(P, stream); break;
            case 7: nvfp4moe_launch_m<7>(P, stream); break;
            default: nvfp4moe_launch_m<8>(P, stream); break;
        }
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
