#pragma once
// AIT5_U Cooperative 2-Warp-per-Slab BN=128 MoE Prefill GEMM (SM 8.0 / SM 8.6):
//   gemm_s8_coop128_ait5<XSH, STAGES=2, MINB=2> (BM = 64 [M_SLABS = 4], BN = 128 [N_WARPS = 2])
//
// Solves the 128-route MoE prefill bottleneck for GGML_TYPE_AIT5_U (e.g., N = 1024..2048):
//   - 1x weight HBM read per 128-route expert tile (vs 2x in BN=64 lowreg),
//   - Cooperative 2-warp slab decode via 64-thread named barrier (`bar.sync (mw+1), 64`):
//     each warp decodes ONLY 2 of the 4 j-steps -> HALVES k=5 trellis unpack INT32 ALU!
//   - Compact 32B scale slots: 48,384 B dynamic + 1,024 B static (+ 1,024 B reserved) = 50,432 B/CTA
//     -> 2 CTAs/SM (16 warps/SM) inside RTX 3090's 100 KiB SMEM!
//   - 100% BITWISE IDENTICAL to base gemm<128, 3, 4, LAY_AIT5_U, XSH, true> (honors idm.tail_skip).
//
// Independent runtime kill switch: GGML_CUDA_AI5_COOP128=0|1 (default 0 = opt-in).

#include "common.cuh"
#include "ai4-kern.cuh"

#include <cstdint>

namespace ai5_coop {

bool env_coop128_enabled();

void dispatch_coop128_ait5_id(
        const uint8_t * W, const int8_t * Xq, const float * Xs,
        float * Y, int64_t ldy, int M, int N, int K, int xsh,
        const ai4k::idmap & idm, int max_tiles, cudaStream_t stream);

} // namespace ai5_coop

extern "C" {
GGML_BACKEND_API uint64_t ggml_cuda_ai5_coop128_fire_count(void);
GGML_BACKEND_API void     ggml_cuda_ai5_coop128_reset_fire_count(void);
}
