#pragma once

#include "common.cuh"
#include "ggml.h"

// Inkling-2 DSpark CUDA fused kernels (new .cu file: dspark-head.cu).
//
// 1. Fused folded Markov head + fold_bias + base + softcap + top-K (or argmax) + Gumbel-max sampling:
//    Replaces 18+ kernels per Markov walk step (mul_mat_vec_f / mul_mat_id + add + add + scale + tanh + scale
//    + unpad cpy + topk_tile_select + argsort + topk_unmap + get_rows + scale + add_noise + argmax + get_rows)
//    with 2 capture-safe, zero-sync kernels (tile-level warp-per-row LDG.E.128 GEMV + 1-CTA merge/sample).
//    Supports BOTH full vocabulary (ids == nullptr) AND dynamic draft-vocabulary shortlist (ids != nullptr)
//    with 100% coalesced warp-per-row gather and zero duplicate weight copies.
//    Kill switch: GGML_CUDA_DSPARK_HEAD=0 (default on).
//
// 2. Fused Markov GRU state update:
//    Replaces 9 tiny kernels per walk step (bias add + 3x chunk cont + sigmoid + tanh + sub + mul + add + tanh)
//    with 1 capture-safe kernel.
//    Kill switch: GGML_CUDA_DSPARK_GRU=0 (default on).
//
// 3. Capture-safe indexed F16 1-row-expert GEMV (fallback when GGML_CUDA_DSPARK_HEAD=0 with shortlist):
//    Allows F16 MUL_MAT_ID over shortlisted rows on NVIDIA without stream synchronization / CUDA graph breaks.

bool ggml_cuda_dspark_head_enabled();
bool ggml_cuda_dspark_gru_enabled();

// Capture-safe F16 1-row-expert MUL_MAT_ID (src0: [K, 1, N] F16, src1: [K, 1, nb] F32, ids: [n_sel, nb])
bool ggml_cuda_dspark_f16_rows_supported(const ggml_tensor * dst);
void ggml_cuda_dspark_mul_mat_id_f16_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fused Markov GRU update:
//   joint_raw: [3*R, nb] F32 (output of dspark_joint_proj @ joint)
//   joint_b:   [3*R]     F32 (dspark_joint_proj_b)
//   state_in:  [R, nb]   F32
//   state_out: [R, nb]   F32
//   latent_out:[R, nb]   F32
void ggml_cuda_dspark_gru_update(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * joint_raw,
        const ggml_tensor * joint_b,
        const ggml_tensor * state_in,
        ggml_tensor       * state_out,
        ggml_tensor       * latent_out);

// Fused folded head + bias + base + softcap + top-K (sampled) + optional shortlist remap + Gumbel-max draw:
//   fold_w:     [256, N_full] F16 (dspark_head_fold; or [256, 1, N_full] in shortlist mode)
//   latent:     [256, nb]     F32
//   fold_bias:  [N_eval]      F32 (full fold_bias or gathered shortlist bias, indexed by local_idx 0..N_eval-1)
//   base:       [N_base, nb]  F32 (N_base >= N_eval; indexed by local_idx 0..N_eval-1)
//   ids:        [N_eval]      I32 or nullptr (shortlist -> real token id map; when non-null, fold_w row is ids[local_idx])
//   noise:      [sk, nb]      F32 or nullptr (Gumbel noise; nullptr => greedy argmax)
//   cand_out:   [sk, nb]      I32 or nullptr (real token ids when sampled; nullptr when greedy)
//   vals_out:   [sk, nb]      F32 or nullptr (softcapped top-K logits when sampled; nullptr when greedy)
//   prev_out:   [nb]          I32 (selected real token id per sequence)
//   n_eval:     number of rows evaluated (n_vocab_unpadded for full vocab, n_sel for shortlist)
//   sk:         top-K size (1..64 for sampled, 1 for greedy)
//   softcap:    final logit softcap (> 0, e.g. 30.0f)
//   inv_temp:   1.0f / dspark_sample_t (unused when greedy)
bool ggml_cuda_dspark_head_launch(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * fold_w,
        const ggml_tensor * latent,
        const ggml_tensor * fold_bias,
        const ggml_tensor * base,
        const ggml_tensor * ids,
        const ggml_tensor * noise,
        ggml_tensor       * cand_out,
        ggml_tensor       * vals_out,
        ggml_tensor       * prev_out,
        int64_t             n_eval,
        int                 sk,
        float               softcap,
        float               inv_temp);
