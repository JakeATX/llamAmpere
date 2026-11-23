#pragma once

#include "common.cuh"
#include "inkglue.cuh"

// [T5-GLUE] G1: Vectorized SM 8.6 loop-gate kernels (down + apply) for both G=1 (k_loop_gate_*) and G=2..4 (k_ink_lg_*).
// Uses 128-bit uint4 BF16/F16 loads (8 weights/load), quarter-warp (8-lane) 4-rows/warp apply with 3-step sub-warp reduction,
// and SMEM-staged 100% coalesced reads/writes of x, m, dst.
// Opt-in switch: GGML_CUDA_INK_LG_VEC=1 (default 0 = off until KLD gate clears; FP32 reduction order differs ~1e-7/gate).

bool ggml_cuda_ink_lg_vec_enabled();

// Multi-gate (G = 1..4) vectorized loop gate; returns false if unsupported (falls back to scalar loop-gate.cu / inkglue.cu).
bool ggml_cuda_ink_loop_gate_vec(ggml_backend_cuda_context & ctx, ggml_tensor * const * gates, int G);
