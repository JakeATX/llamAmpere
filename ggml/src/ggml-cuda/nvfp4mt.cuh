#pragma once
// INKNVFP4K NVFP4MT: NVFP4 MoE decode for 1..8 tokens with each routed expert's weights streamed once per visit
// (cp.async rings, per member count bodies), bit-identical to the generic path. See nvfp4mt-kern.cuh.
// GGML_CUDA_NVFP4MT=0 turns it off (T > 1 then runs k_nvfp4moe_gu / k_nvfp4moe_dnm, T = 1 k_nvfp4moe_gu1w / dn1).
// GGML_CUDA_NVFP4MT_T1=0 keeps T = 1 on k_nvfp4moe_gu1w / dn1 only.

#include "nvfp4moe.cuh"

bool ggml_cuda_nvfp4mt_enabled();
// T = 1 may run NVFP4MT (the caller then reserves P.xq at one token too)
bool ggml_cuda_nvfp4mt_t1();
// true if ggml_cuda_nvfp4mt_launch takes P: T = 1..8 (1 only with ggml_cuda_nvfp4mt_t1 and !P.write_sum), P.xq set (PREQ), q8 activations,
// the Inkling shapes and contiguous 16-byte aligned weights
bool ggml_cuda_nvfp4mt_accepts(const nvfp4moe_params & P);
// if ggml_cuda_nvfp4mt_accepts(P): launches x -> q8_1, gate_up + SITU, down, writes P.e_out (never P.out: the caller adds
// the weighted expert sum, also at T = 1) and returns true; otherwise false (nothing launched)
bool ggml_cuda_nvfp4mt_launch(const nvfp4moe_params & P, cudaStream_t stream);
