#pragma once

#include "common.cuh"
#include "ggml-inkrel.h"

// INKATTN (fork, Inkling 2 decode attention): GQA-grouped tensor-core FLASH_ATTN_EXT for small query widths.
// One warp streams a run of 16-key tiles of one KV head straight from the F16 cache into mma fragments (no shared
// memory) for all g*T query rows of that KV head at once (g = q heads per KV head, T = query tokens, g*T <= 16), so
// each K/V byte is read once per KV head instead of once per q head. Partial (max, sum, O) per warp, then a combine
// kernel (sinks, optional LSE output). D = 128, F16 K/V, F16 mask (broadcast or per head), no ALiBi / softcap.
// Default scope: per-head masks (Inkling rel bias). GGML_CUDA_INKATTN=0 disables it (old routes), =2 adds LSE nodes with
// a broadcast mask (INKREL far), =3 adds every broadcast-mask node.
// Returns true when it handled dst.
bool ggml_cuda_inkattn_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// INKREL nodes (ggml-inkrel.h: relative bias table + gate inside the node) run only here: can this device take dst?
bool ggml_cuda_inkattn_supported(int device, const ggml_tensor * dst);
// INKATTN live tiles (inkattn-live.cu): compacted ids of the 16-key tiles with any visible mask entry per (seq, KV head)
// into live[(seq*Hkv + kvh)*NTmax + i], their count into nlive[seq*Hkv + kvh]
void ggml_cuda_inkattn_live(const char * M, int64_t nbm1, int64_t nbm2, int64_t nbm3, int ne32, int ne33, int n_kv, int T, int g,
        int Hkv, int S, bool al16, int NTmax, int * live, int * nlive, cudaStream_t stream);
