#pragma once

#include "common.cuh"

// INKATTN-FAST (fork): SM86 occupancy & dead-tile-skipping decode kernel for Inkling 2 SWA/near windows.
// Opt-in (GGML_CUDA_INKATTN_FAST=1) until gated; unset or 0 = the baseline inkattn.cu kernel.
bool ggml_cuda_inkattn_fast_enabled();

void ggml_cuda_inkattn_fast_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * Q,
        const ggml_tensor * K,
        const ggml_tensor * V,
        const ggml_tensor * mask,
        const ggml_tensor * snk,
        ggml_tensor * dst,
        ggml_tensor * lse,
        const ggml_tensor * tab,
        const ggml_tensor * rp,
        const ggml_tensor * gate,
        float scale,
        bool inkrel,
        bool q8);
