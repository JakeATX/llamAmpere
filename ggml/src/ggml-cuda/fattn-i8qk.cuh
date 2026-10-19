#pragma once

// [I8QK] Default-on int8-QK prefill flash attention (SageAttention-style), SM80+ (tuned for SM86).
//
// Unless GGML_CUDA_FA_I8QK=0, routes wide-query FLASH_ATTN_EXT ops (>= GGML_CUDA_FA_I8QK_MIN_Q queries, default 64) that would
// otherwise take the f16 MMA kernel (generic or bounded prefill route) onto this path:
//   - K: converted to f16 (same converter as the f16 route), smoothed by subtracting the per-channel mean over all
//     n_kv tokens of each KV head (softmax is invariant to the per-query constant q.mean), then quantized to int8 with
//     one f32 scale per token (default KCB decodes tq5_0/tq6_0 directly, with one scale per 128-value block);
//   - Q: quantized to int8 per (query, head) row inside the kernel, softmax scale folded into the row scale;
//   - QK^T on IMMA m16n8k32 s8 with int32 accumulation, dequantized by the outer product of the row/token scales;
//   - online softmax in f32, P in f16, PV on f16 MMA m16n8k16 with f16 accumulation by default (V stays f16).
// GGML_CUDA_FA_I8QK_PV16 and GGML_CUDA_FA_I8QK_KCB default to 1; =0 disables each (PV16=0 uses f32 P.V accumulation).
// Opt-in variants (fattn-i8qk.cu, i8qk_modes): GGML_CUDA_FA_I8QK_QH=1 (Q scale per 128-dim half),
// GGML_CUDA_FA_I8QK_Q2L=1 (two-limb Q), GGML_CUDA_FA_I8QK_KBLK=0|n (no K smoothing / block-frozen K offsets).
// Quantized K and V caches only (q8_0, turbo2/3/4, tq5_0, tq6_0). No new
// VRAM: the int8 K and the f16 V reuse the f16 K/V reservation the allocator already made for the f16 route behind dst
// (bounded prefill workspace or the generic f16 copies); without that reservation the op keeps the f16 route.

#include "common.cuh"

bool ggml_cuda_fattn_i8qk_enabled();

// Shape/type gate of the int8-QK route (the caller also checks that the op would take the f16 MMA route).
bool ggml_cuda_fattn_i8qk_applies(int cc, const ggml_tensor * dst);

// [I8SJ] the same gate for f16 K and V read in place (no f16 K/V reservation behind dst): only for callers that pass
// their own int8 K workspace, i.e. the SJ-KVaRN prefill (fattn-sjkvarn-prefill.cu) on its expanded f16 K/V scratch.
bool ggml_cuda_fattn_i8qk_applies_f16kv(int cc, const ggml_tensor * dst);

// [I8SJ] planning predicate on the original SJ-KVaRN node (switch on, SM80+, D 256, wide query, no sinks / bias /
// softcap, one mask plane): the SJ-KVaRN prefill plan then reserves n_kv*256 bytes of int8 K per KV head of a group.
bool ggml_cuda_fattn_i8qk_sj_kvarn_wanted(int cc, const ggml_tensor * dst);

// Run KV heads [first, first + heads) of dst. ws_k: >= n_kv*D*heads bytes for the int8 K; ws_v: >= n_kv*D*heads halves
// for the f16 K staging and then the f16 V (both required).
void ggml_cuda_flash_attn_ext_i8qk(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int first, int heads,
                                   void * ws_k, half * ws_v);
