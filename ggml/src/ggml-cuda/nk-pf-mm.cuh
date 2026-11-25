#pragma once
// NEONKERN NK_PF_MM: Ampere (sm_80 / sm_86) NVFP4 MoE prefill int8 tensor-core GEMM.
// Opt-in via GGML_CUDA_NK_PF_MM=1 (default OFF; unset = exact base nvfp4pf).
// Bit-exact with k_nvfp4pf_mm for both KF=64 and KF=32.

#include "common.cuh"

#include <cstdint>

struct nk_pf_map {
    const int * hdr;
    const int * exps;
    const int * off;
    const int * rows;
    const int * cols;
    const int * tiles;
    int64_t     nb02;
};

bool    ggml_cuda_nk_pf_mm_enabled();
int64_t ggml_cuda_nk_pf_mm_count();

void ggml_cuda_nk_pf_mm_launch(
        int KF, int bn,
        const char * w, int64_t stride_blk,
        const int8_t * xq, const float * xs,
        float * y, int M, int K,
        const nk_pf_map & map, int max_tiles,
        const float * s_exp, int device, cudaStream_t stream);

// [GFFN-P9] M = 2F gate_up rows; stores SITU(gate*s, up*s) into y [F] per column (the bytes of the plain launch above
// followed by the SITU kernel)
struct ink_situ_params;
void ggml_cuda_nk_pf_mm_launch_situ(
        int KF, int bn,
        const char * w, int64_t stride_blk,
        const int8_t * xq, const float * xs,
        float * y, int M, int K,
        const nk_pf_map & map, int max_tiles,
        const float * s_exp, const ink_situ_params & situ, int device, cudaStream_t stream);
