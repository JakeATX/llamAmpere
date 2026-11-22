#pragma once

#include "common.cuh"
#include "topk-moe.cuh"

// [INKROUTER] decode MoE router in one launch: MUL_MAT(BF16 ffn_gate_inp [ncols, n_expert], x [ncols, 1]) followed by the
// topk-moe chain (gating .. weights SCALE) that ggml_cuda_op_topk_moe runs. One warp per expert row replays the
// mul_mat_vec_f reduction that ggml_cuda_mul_mat picks for this product (BF16, one column, block of 256), so the logits are
// bit-identical to MMVF; the last block to finish runs topk_moe_cuda's per-row device code (topk-moe-device.cuh) on them,
// so ids and weights are bit-identical to the topk-moe kernel.

// Shape gate: n_expert is an instantiated count, the reduction replay matches the block size MMVF picks for ncols.
bool ggml_cuda_inkrouter_supported(const ggml_tensor * w, const ggml_tensor * y, int n_expert_used, int cc, int warp_size);

// logits = the MUL_MAT node (written), weights / ids / clamp / scale / bias as for ggml_cuda_op_topk_moe.
// Returns false (and launches nothing) when no completion-counter slot is free; the caller then runs the unfused nodes.
bool ggml_cuda_op_inkrouter(ggml_backend_cuda_context &     ctx,
                            const ggml_tensor *             w,
                            const ggml_tensor *             y,
                            ggml_tensor *                   logits,
                            ggml_tensor *                   weights,
                            ggml_tensor *                   ids,
                            const ggml_tensor *             clamp,
                            const ggml_tensor *             scale,
                            const ggml_tensor *             bias,
                            const ggml_cuda_topk_moe_args & args);
