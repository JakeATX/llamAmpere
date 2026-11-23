#pragma once

#include "common.cuh"
#include "ggml-inkrel.h"

// INKPREFILL inkfa: tensor-core prefill attention for INKREL nodes (ggml-inkrel.h: the Inkling 2 relative-position bias
// table + rel_pos + output gate inside FLASH_ATTN_EXT) over an F16 K/V cache, D = 128. INKATTN serves the decode widths
// (q heads per KV head x tokens <= 16); inkfa takes the wider ones, so a prefill ubatch no longer materialises the
// [n_kv, T, n_head] mask. Layout: one block = 16*NW query rows of
// one q head (NW = 8 warps x 16 rows; 4 when the 8-warp blocks would not cover the SMs), 64-key K/V tiles double-buffered in shared memory by cp.async, mma m16n8k16, online
// softmax in log2 units, sinks and the gate in the epilogue. Key tiles no row of the block sees (all -inf in the base
// mask) are skipped: a pre-pass lists the live tiles of each row block.
// Per logit: scale*q.k + f16(mask[c, t] + table[h, rel_pos[c, t], t]) (inkrel-bias.cuh), the INKREL contract.
// GGML_CUDA_INKFA=0 (or LLAMA_INKFA=0) disables it: the graph keeps the dense prefill path. The graph route also needs
// INKATTN on (GGML_CUDA_INKATTN=0 / LLAMA_INKATTN_REL=0 turn both off).

bool ggml_cuda_inkfa_enabled();
// can inkfa take dst on this device (an INKREL node INKATTN does not take: g*T > 16)?
bool ggml_cuda_inkfa_supported(int device, const ggml_tensor * dst);
// runs dst and returns true when supported
bool ggml_cuda_inkfa_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
