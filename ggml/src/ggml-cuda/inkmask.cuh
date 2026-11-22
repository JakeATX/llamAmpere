#pragma once

#include "common.cuh"

// INKPREFILL: the per-head relative-bias mask of Inkling 2 in one kernel.
// The graph builds the flash-attention mask of a visit as
//   GET_ROWS(table [H, R, T] F32, rel [n_kv, T] I32) -> [RESHAPE] -> PERMUTE(2,0,1,3) -> CONT -> [CPY F16->F32 mask]
//   -> ADD(mask, broadcast) -> CPY F16
// i.e. out[k, t, h] = f16(table[h, rel[k, t], t] + mask[k, t]). Matched at the GET_ROWS node, the chain runs as one
// launch that writes only the F16 result (same F32 add, same round-to-nearest cast: bit-identical).
// GGML_CUDA_INK_MASK=0 turns it off.

// Returns the offset n of the chain's last node (the F16 CPY) from node i, or 0 when the chain does not match.
// The caller checks the memory ranges of nodes i..i+n and then calls ggml_cuda_ink_relmask_launch.
int ggml_cuda_ink_relmask_match(const ggml_cgraph * cgraph, int i);

void ggml_cuda_ink_relmask_launch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int n);
