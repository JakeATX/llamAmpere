#pragma once
// INKNVFP4K: NVFP4 MoE decode (1..NVFP4MOE_MAX_TOK tokens) in two launches, see nvfp4moe-kern.cuh.
// GGML_CUDA_NVFP4MOE=0 turns it off; GGML_CUDA_NVFP4MOE_MAXT=<n> (1..16, default 8) is the largest token count it takes.

#include "common.cuh"
#include "inkglue.cuh"

#define NVFP4MOE_MAX_TOK 16

struct nvfp4moe_params {
    const char    * w_gu;  int64_t gu_nb1, gu_nb2;   // gate_up [K, 2F, E]: row and expert strides in bytes
    const char    * w_dn;  int64_t dn_nb1, dn_nb2;   // down    [F, K, E]
    const float   * s_gu;                             // per-expert scales [E] or nullptr
    const float   * s_dn;
    const float   * x;     int64_t x_s2;              // expert input [K, 1, T], token stride in floats
    const int32_t * ids;   int64_t ids_s1;            // [n_used, T], token stride in int32
    const float   * rw;    int64_t rw_s1, rw_s2;      // routing weights [1, n_used, T] (float strides), dn1 only
    block_q8_1    * act;                              // [T][n_used][F/32] SITU output as q8_1 (scratch)
    block_q8_1    * xq;                               // [T][K/32] x as q8_1 (T > 1 with PREQ; scratch, 16-byte aligned)
    float         * actf;                             // [T][n_used][F] SITU output as f32 (f32-activation arm)
    half2         * acth;                             // [T][n_used][F/2] SITU output as f16 pairs (f16 arm)
    float         * e_out;                            // [K, n_used, T] scaled down outputs (dnm)
    float         * out;                              // [K, T] expert sum (dn1)
    ink_situ_params situ;
    int K, F, n_used, T;
    int f32act;                                       // 1: f32-activation arm, 2: f16 arm (T == 1 only)
    int write_sum;                                    // 1: T == 1 writes out (expert sum), T > 1 e_out; 0: e_out only
};

bool ggml_cuda_nvfp4moe_enabled();
// GGML_CUDA_NVFP4MOE_ACT=f32 | f16: f32 / f16 activations (no q8_1 of x and SITU output) at one token; default q8 (dp4a)
// returns 0 (q8), 1 (f32), 2 (f16)
int ggml_cuda_nvfp4moe_f32act();
int  ggml_cuda_nvfp4moe_max_tok();
// GGML_CUDA_NVFP4MOE_PREQ=0: T > 1 gate_up CTAs quantize x themselves (default: one k_nvfp4moe_xq launch per visit)
bool ggml_cuda_nvfp4moe_preq();
// GGML_CUDA_NVFP4MOE_GORD=0: T > 1 sums each token sequentially (default: the generic MMVQ order, bit-identical per token)
bool ggml_cuda_nvfp4moe_gord();
// GGML_CUDA_NVFP4MOE_EXACT=0: T > 1 runs the 2 / 4 / 8 / 16-token kernels (default: the kernel of exactly T tokens, T <= 8)
bool ggml_cuda_nvfp4moe_exact();
// shapes the kernels take (K = 2304, F = 1152, n_used = 8, NVIDIA Ampere/Ada)
bool ggml_cuda_nvfp4moe_supported(int cc, int64_t K, int64_t F, int64_t n_used);
// write_sum && T == 1: writes P.out (expert sum); else P.e_out (scaled per-slot down outputs; the caller sums)
void ggml_cuda_nvfp4moe_launch(const nvfp4moe_params & P, cudaStream_t stream);
