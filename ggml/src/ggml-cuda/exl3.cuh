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

// M2: trellis-direct GEMV/GEMM for T <= EXL3_GEMV_MAX_T (16) activation columns (env GGML_CUDA_EXL3_GEMV=0 forces the
// M1 reconstruct + cuBLAS path for those widths; the #13 GEMM only takes T > 16)
bool ggml_cuda_exl3_gemv_supported(const int64_t T);
int  ggml_cuda_exl3_gemv_ksplit(const int kt, const int nt, const int T);
// suh/svh may be null (no Hadamard/scale glue)
void ggml_cuda_exl3_gemv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const int64_t T,
                         const float * suh, const float * svh);

// out[T][K] (f16) = scale * H128(suh * x)  (suh null: scale * x)
void ggml_cuda_exl3_glue_in_f16(const float * x, const float * suh, half * out, const int K, const int64_t T, const float scale, cudaStream_t stream);
// y[T][N] = scale * svh * H128( sum_ks part[ks][T][N] )  (svh null: scale * sum); N % 128 == 0 when svh is set
void ggml_cuda_exl3_glue_out(const float * part, const float * svh, float * y, const int N, const int64_t T, const int ksplit, const float scale, cudaStream_t stream);
// [FIT12 overhead] ksplit == 1 with svh, in place on y (bit-identical to glue_out(y, svh, y2, ..., 1, ...))
void ggml_cuda_exl3_glue_out_inplace(float * y, const float * svh, const int N, const int64_t T, const float scale, cudaStream_t stream);

// [#74] out[T][N] (f16) = x_scale * H128(suh_down * silu(svh_gate * H128(sum part_gate)) * svh_up * H128(sum part_up)):
// glue_out of the FFN gate and up projections, the SwiGLU and glue_in of the down projection in one kernel
void ggml_cuda_exl3_ffn_bridge_f16(const float * part_gate, const int ksplit_gate, const float * svh_gate,
                                   const float * part_up, const int ksplit_up, const float * svh_up,
                                   const float * suh_down, half * out, const int N, const int64_t T, const float x_scale,
                                   cudaStream_t stream, const int ldp = 0);
// [#74] gate/up/down EXL3 projections with glu = swiglu_split(gate, up) between them, through the bridge above; the
// gate and up outputs are never materialized. Returns false (nothing launched) when a precondition fails.
// [#136] xh: the gate's glued fp16 input when the caller already has it (RMS_NORM fold), else null; res/y: the down
// projection's residual ADD folded into its glue_out (y = down + res), or null/null for y = mm_down->data.
bool ggml_cuda_exl3_ffn_bridge(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_gate, const ggml_tensor * mm_up,
                               ggml_tensor * mm_down, const half * xh = nullptr, const float * res = nullptr, float * y = nullptr);

// ---- [#136] EXL3 glue folding (GGML_CUDA_EXL3_GLUE_FOLD=1, bit-identical to the unfolded nodes) ----
// y[T][N] = svh * H128(sum part) + res[T][N]  (glue_out with the consumer's residual ADD)
void ggml_cuda_exl3_glue_out_res(const float * part, const float * svh, const float * res, float * y, const int N, const int64_t T,
                                 const int ksplit, cudaStream_t stream);
// dst = rms_norm(x) * gamma (f32, rows of ncols) and xh = x_scale * H128(suh * dst) (fp16)
void ggml_cuda_exl3_rms_norm_mul_glue(const float * x, const float * gamma, float * dst, const float * suh, half * xh,
                                      const int ncols, const int nrows, const float eps, const float x_scale, cudaStream_t stream);
bool  ggml_cuda_exl3_glue_fold_enabled();
float ggml_cuda_exl3_x_scale();
// mm (EXL3 MUL_MAT with suh/svh) can take the folded split-glue GEMV path at its width
bool ggml_cuda_exl3_fold_ok(const ggml_tensor * mm);
// one EXL3 linear on the split-glue GEMV path: xh (prepared glued input) or null (glue_in here), y = mm + res (res may
// be null), y defaults to mm->data
void ggml_cuda_exl3_linear_fold(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const half * xh, const float * res, float * y);
// fused [gate | up] projection mm_gu (a --fuse GGUF), glu = swiglu(mm_gu) (single source, swapped or not), down reads glu:
// gate/up glue_out + SwiGLU + down glue_in in the bridge kernel. xh/res/y as above. Returns false (nothing launched)
// when a precondition fails.
bool ggml_cuda_exl3_ffn_bridge_gu(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_gu, const ggml_tensor * glu,
                                  ggml_tensor * mm_down, const half * xh, const float * res, float * y);

// [#13] trellis-direct tensor-core GEMM for T > EXL3_GEMV_MAX_T (prefill); GGML_CUDA_EXL3_GEMM=0 forces the M1
// reconstruct + cuBLAS path. suh/svh may be null.
bool ggml_cuda_exl3_gemm_enabled();
bool ggml_cuda_exl3_gemm_supported(const ggml_tensor * src0, const int64_t T);
void ggml_cuda_exl3_gemm(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                         const int64_t T, const float * suh, const float * svh);

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
