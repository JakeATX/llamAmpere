#pragma once

#include "common.cuh"

// INKGQA (surge, all flags default OFF): one tensor-core prefill attention kernel whose block covers 16*RW query rows
// of all G = 3 q heads of one KV head, so every 32-key K/V tile is loaded into shared memory once for the 3 heads
// (the generic D 128 MMA tile runs ncols2 = 1 per q head, or ncols2 = 4 with a padded 4th head; inkfa launches one
// block per q head). D = 128, F16 K/V, f32 accumulate, online softmax in log2 units, optional key split with a combine
// pass, row log-sum-exp output. Key tiles no row of a block sees are skipped; tiles every row sees with a zero mask
// skip the mask read.
//
//   GGML_CUDA_INKGQA_FAR=1  plain F16 FA nodes with GQA 3 at D 128 (broadcast F16 mask, optional LSE / sinks): the
//                           packed D128 SJ-KVaRN prefill (inksjd128.cu), the INKKV pair-packed route
//                           (fattn-sjkvarn-prefill.cu) and generic F16 nodes of that shape.
//   GGML_CUDA_INKGQA_SWA=1  INKREL nodes (ggml-inkrel.h, SWA prefill visits) in place of inkfa.
//   INKREL factor nodes (ggml_flash_attn_ext_set_inkrel_factor, built by the graph under LLAMA_INKFA_RELF=1) always
//                           run here: the relative bias r . E[d] is computed inside the kernel from its rank-16
//                           factors (one m16n8k16 k-step per 8 distances), no [H, E+1, T] table.
//   GGML_CUDA_INKGQA_DIRECT=1 (with _FAR) the packed D128 SJ-KVaRN route decodes 4/4 K/V tiles straight into
//                           shared memory (no f16 expand scratch).
//   GGML_CUDA_INKGQA_SPLIT=N forces N key splits; GGML_CUDA_INKGQA_RW=2/4 forces the row warps per head.

bool ggml_cuda_inkgqa_far_enabled();
bool ggml_cuda_inkgqa_swa_enabled();
bool ggml_cuda_inkgqa_direct_enabled();

// plain FA node (no SJ-KVaRN / INKREL attachments): runs it and returns true when the shape is covered
bool ggml_cuda_inkgqa_fa_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// INKREL node (table or factor form)
bool ggml_cuda_inkgqa_inkrel_supported(int device, const ggml_tensor * dst);
bool ggml_cuda_inkgqa_inkrel_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
