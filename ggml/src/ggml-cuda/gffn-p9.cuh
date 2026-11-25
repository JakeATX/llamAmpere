#pragma once

#include "common.cuh"
#include "inkglue.cuh"

// [GFFN-P9] (GGML_CUDA_GFFN_P9=1, default off): an NVFP4 MoE prefill FFN's gate_up MUL_MAT_ID with its per-expert scale
// and the SITU run on its gate / up views as one NVFP4PF GEMM (k_nvfp4pf_mm<KF, BN, true>) whose epilogue stores
// SITU(gate, up) into the SITU output, so the [2F] f32 gate_up rows are never written or read back. Same dot products,
// scale and SITU expression as the unfused run (NVFP4PF GEMM, then the SITU kernel), so the output is its bytes. The down
// MUL_MAT_ID quantizes that output itself (k_nvfp4pf_xq, the bytes the fused SITU + quantize producer would cache).
bool ggml_cuda_gffn_p9_on();

// shape / type / mode checks only (no data pointers: also used in plan mode). gu = MUL_MAT_ID(w [K, 2F, E] NVFP4,
// x [K, ne11, T] f32, ids [n_used, T]), s_gu = its per-expert scale [E], act = the SITU output [F, n_used, T]
bool ggml_cuda_gffn_p9_supported(int cc, const ggml_tensor * gu, const ggml_tensor * s_gu, const ggml_tensor * act,
        const ink_situ_params & p);

// route, x int8 (or the prefill producer's cached copy), fused GEMM into act. false = nothing launched
bool ggml_cuda_gffn_p9_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * gu, const ggml_tensor * s_gu, ggml_tensor * act,
        const ink_situ_params & p);

// launches so far (test-backend-ops required_fusion "gffn_p9")
int64_t ggml_cuda_gffn_p9_count();
