#pragma once

// [HQ4] Prefill tile probe for the D256 MMA f16 flash-attention path (HyperQwen audit patch 0002, fixed).
// GGML_CUDA_FA_PREFILL_NCOLS2=1|2|8 picks the GQA packing (ncols2) of wide-query D256 launches on SM86; unset
// (or any other value) = today's rule, which for Qwen3.8 (24 query heads / 4 KV heads, GQA 6) is ncols2 = 8.
//   8: today's tile (6 real heads padded to 8), explicit arm for bracketing; the same launch as unset.
//   1: the audit's head-major arm (64 queries x 1 head). No cp.async pipeline: nstages = 0 for ncols2 < 2
//      (ggml_cuda_fattn_mma_get_nstages, fattn-mma-f16.cuh).
//   2: 32 queries x 2 heads, exact for GQA 6 (no padded heads) and keeps the cp.async pipeline. Separates the
//      geometry effect from the pipeline effect of arm 1.
// Applies only to prefill (>= 128 queries), one sequence, D 256/256, and only where the arm divides the GQA
// ratio (ncols2 > 1 also needs the GQA-packing preconditions, use_gqa_opt). Plain C++ for the host test.

#include <cstdint>
#include <cstdlib>

static inline int ggml_cuda_fa_prefill_ncols2_env() {
    static const int v = [] {
        const char * e = std::getenv("GGML_CUDA_FA_PREFILL_NCOLS2");
        if (e == nullptr || e[0] == '\0' || e[1] != '\0') {
            return 0;
        }
        return (e[0] == '1' || e[0] == '2' || e[0] == '8') ? e[0] - '0' : 0;
    }();
    return v;
}

// The ncols2 to force, or 0 = today's rule.
static constexpr int ggml_cuda_fa_prefill_ncols2_pick(
        int arm, int cc, int dk, int dv, int64_t queries, int64_t gqa_ratio, bool one_sequence, bool gqa_opt) {
    return (arm == 1 || arm == 2 || arm == 8) && cc == 860 && dk == 256 && dv == 256 && queries >= 128 &&
           one_sequence && gqa_ratio >= 1 && (arm == 1 || (gqa_opt && gqa_ratio % arm == 0) ||
                                               (arm == 8 && gqa_opt && gqa_ratio > 4)) ? arm : 0;
}
