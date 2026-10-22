#pragma once
#include "common.cuh"

// [HCGP] gate GEMM (BF16 w_up x f32 lo) fused with the gated DSV4_HC_PRE that consumes it (see hc-gate-pre.cu)
bool ggml_cuda_hc_gate_pre_supported(const ggml_tensor * mm, const ggml_tensor * hc_pre, int cc);
bool ggml_cuda_hc_gate_pre_xn_supported(const ggml_tensor * xr, const ggml_tensor * gamma, const ggml_tensor * hc_pre);
// xr/rs/gamma non-null: hc_pre->src[0] was never written (GGML_CUDA_HC_PRE_XN); xn is rebuilt as
// (rs[(t*hc + c)*n_embd] * xr) * gamma, rs being the per-row scales the norm left in xn's buffer
void ggml_cuda_op_hc_gate_pre(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * hc_pre,
                              const ggml_tensor * xr, const float * rs, const ggml_tensor * gamma);
