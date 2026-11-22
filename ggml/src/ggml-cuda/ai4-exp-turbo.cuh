#pragma once

#include "common.cuh"

#include <cstdint>

// [INK200 T1] SM 8.0+ (Ampere RTX 3090/Ti) SpecDecode (M = 2..32) grouped MoE expert kernels
// for AI+ unit layouts (AIT4_U, AIT3_U, AIT2_U).
//
// Two-regime dispatch (N == 1 is intentionally untouched and stays on rc's single-token mmvr):
//   1. M = 2..8 (n_pairs <= 64, e.g. Inkling-2-Mini DSpark d4 verify M=5, 40 routes):
//      - exp_turbo_direct_s8: ZERO ids_helper launch, ZERO static SMEM, ZERO __syncthreads() for routing.
//        Warp-ballot + warp-private 32B scatter into dead ring[0..31]; dead CTAs exit in ~15 insns
//        with 0 BAR.SYNC; Split-K WARPS=3 (100% K-balance on K=2304=18*128 & K=1152=9*128, 64 regs,
//        10 CTAs/SM = 30 warps/SM) or WARPS=4 (64 regs, 8 CTAs/SM = 32 warps/SM); native a_regs_smem
//        (levels4 PRMT) for AIT4_U; skips empty LDG.128 on inactive columns (g >= N).
//   2. M = 9..32 (or M = 2..8 with n_pairs > 64):
//      - exp_turbo_grouped_s8: zero-static-SMEM A_UPFRONT 2x LDG.128 grouped Split-K (BN=8/16/32)
//        paired with a 1-entry per-context CUDA-graph-safe ids_helper cache so gate_up and down
//        share ONE routing launch per MoE visit (-50% routing overhead).
//
// Env knobs:
//   GGML_CUDA_AI4_EXP_TURBO=0|1           kill switch (default 1 = on)
//   GGML_CUDA_AI4_EXP_TURBO_CFG=auto|3x2|3x3|4x2  direct kernel (warps x stages), default auto
//   GGML_CUDA_AI4_EXP_TURBO_MAX_N=32      max verify token width routed here (default 32, min 2)
//   GGML_CUDA_AI4_EXP_TURBO_IDCACHE=0|1   cache ids_helper across gate_up & down for M=9..32 (default 1)
bool ggml_cuda_ai4_exp_turbo_try(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * ids,
        const int8_t * xq,
        const float * xs,
        float * y_out,
        int M,
        int K,
        int n_tok,
        int n_used,
        int ne11,
        int n_expert,
        int xsh,
        bool has_y_into,
        bool skip_zero);
