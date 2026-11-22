#pragma once

#include "common.cuh"

// INKGLUE REL table (INKATTN visits): PAD(bias, 0, 1) -> MUL_MAT(., r) -> SUB(., sinks) -> CONT(PERMUTE(1, 0, 2, 3)) in one
// launch. bias [n_rel, E] (n_rel <= 32), r [n_rel, H, T], sinks [1, H, T] -> dst [H, E + 1, T] contiguous F32; the row E is
// the pad row (bias 0). Same fmaf chain as k_ink_rel_mask's full path, so the table equals the one the dense fused mask uses.
void ggml_cuda_ink_rel_table(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, ggml_tensor * dst);
