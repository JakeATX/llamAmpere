#pragma once
// INKNVFP4K NVFP4MT: NVFP4 MoE decode for 2..8 tokens with each routed expert's weights streamed once per visit
// (cp.async rings, per member count bodies), bit-identical to the generic path. See nvfp4mt-kern.cuh.
// GGML_CUDA_NVFP4MT=0 turns it off (T > 1 then runs k_nvfp4moe_gu / k_nvfp4moe_dnm).

#include "nvfp4moe.cuh"

bool ggml_cuda_nvfp4mt_enabled();
// T = 2..8 with P.xq set (PREQ) and the Inkling shapes / contiguous 16-byte aligned weights: launches x -> q8_1, gate_up
// + SITU, down (writes P.e_out; the caller sums) and returns true; otherwise false (nothing launched)
bool ggml_cuda_nvfp4mt_launch(const nvfp4moe_params & P, cudaStream_t stream);
