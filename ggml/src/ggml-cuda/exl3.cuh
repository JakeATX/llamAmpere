#pragma once
// EXL3 format and trellis GEMV design: Turboderp, exllamav3 (https://github.com/turboderp-org/exllamav3), MIT License,
// Copyright (c) 2025 Turboderp; see licenses/LICENSE-exllamav3. Independent ggml/CUDA reimplementation.
#include "common.cuh"

// 128-point Walsh-Hadamard in Sylvester order (== the file's exl3_had128.weight up to the 1/sqrt(128) scale),
// one warp per 128-chunk, lane holds indices j*32 + lane (j = 0..3): bits 5,6 in registers, bits 0..4 via shfl.
static __device__ __forceinline__ void exl3_wht128(float (&v)[4]) {
    const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[1] = a1 + a3; v[2] = a0 - a2; v[3] = a1 - a3;
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? (o - v[j]) : (v[j] + o);
        }
    }
}
#define EXL3_WHT128_SCALE 0.08838834764831845f   // 1/sqrt(128)


// EXL3 (exllamav3 trellis) weights. M1 reference path: reconstruct the whole [K, N] weight to f16 in pool memory
// (codebook values only) and run a cuBLAS GEMM. The .suh/.svh vectors and the 128-block Hadamard are graph ops
// around the mul_mat (llama build_lora_mm). Fused decode GEMV/GEMM come in later milestones.

// dst: f16 [N][K] row-major (row n = output feature), codebook values, no suh/svh/Hadamard
void ggml_cuda_exl3_reconstruct_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream);

// M2: trellis-direct GEMV/GEMM for T <= 8 activation columns (env GGML_CUDA_EXL3_GEMV=0 forces the M1 path)
bool ggml_cuda_exl3_gemv_supported(const int64_t T);
int  ggml_cuda_exl3_gemv_ksplit(const int kt, const int nt, const int T);
// suh/svh may be null (no Hadamard/scale glue)
void ggml_cuda_exl3_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int64_t T,
                         const float * suh, const float * svh);

// out[T][K] (f16) = scale * H128(suh * x)  (suh null: scale * x)
void ggml_cuda_exl3_glue_in_f16(const float * x, const float * suh, half * out, const int K, const int64_t T, const float scale, cudaStream_t stream);
// y[T][N] = scale * svh * H128( sum_ks part[ks][T][N] )  (svh null: scale * sum)
void ggml_cuda_exl3_glue_out(const float * part, const float * svh, float * y, const int N, const int64_t T, const int ksplit, const float scale, cudaStream_t stream);

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
