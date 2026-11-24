#pragma once

#include "common.cuh"

// fork (ONEGRAPH, backlog #101): GGML_OP_MOE_OG_SLOTS, the GPU-anchored MoE cache lookup (protocol: ggml-moe-onegraph.h).
// Kernel 1 (one block) publishes the ids to the host ring and, on a miss, waits for the host worker's answer;
// kernel 2 does the answer's copies, applies its slot map changes and writes the slots.
void ggml_cuda_moe_og_slots(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_moe_og_supported(const ggml_tensor * op);

// page-lock a host range read-only and mapped for the copies of kernel 2 (exported as ggml_backend_cuda_moe_og_pin);
// false if the device cannot read it at the same address
bool ggml_backend_cuda_moe_og_pin(void * ptr, size_t size);
void ggml_backend_cuda_moe_og_unpin(void * ptr);
