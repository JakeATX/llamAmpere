#pragma once

// Device code of the fused top-k MoE kernel (topk-moe.cu), shared with the INKROUTER router fusion (inkrouter.cu).
// Moved here verbatim from topk-moe.cu: the per-row selection after the logits are loaded is topk_moe_row(), so both
// kernels run the same instructions on the same values.

#include "common.cuh"

#include <cfloat>
#include <cmath>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool norm_eps;   // INKGLUE: w / (1*sum + eps), sum in reduce_rows order (bit-identical to SUM_ROWS -> SCALE -> DIV)
    bool delayed_softmax;
    int  n_groups;
    int  n_group_used;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : log1pf(expf(vals[i]))) : -INFINITY;
    }
}

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
// Set -inf for experts outside the top n_group_used groups. Group score is the sum of its top-2 experts (DeepSeek V3).
template <int n_experts, int experts_per_thread>
__device__ void group_mask_warp_inplace(float (&vals)[experts_per_thread], const int n_groups, const int n_group_used, const int lane) {
    const int n_per_group = n_experts / n_groups;

    float score = -INFINITY; // lane g holds the score of group g
    for (int g = 0; g < n_groups; g++) {
        float top1 = -INFINITY;
        float top2 = -INFINITY;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if ((i * WARP_SIZE + lane) / n_per_group == g) {
                top2 = fmaxf(top2, fminf(top1, vals[i]));
                top1 = fmaxf(top1, vals[i]);
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            const float o1 = __shfl_xor_sync(0xFFFFFFFF, top1, offset, WARP_SIZE);
            const float o2 = __shfl_xor_sync(0xFFFFFFFF, top2, offset, WARP_SIZE);
            top2 = fmaxf(fminf(top1, o1), fmaxf(top2, o2));
            top1 = fmaxf(top1, o1);
        }
        if (lane == g) {
            score = top1 + top2;
        }
    }

    // rank of each group, lower index wins ties
    int rank = 0;
    for (int g = 0; g < n_groups; g++) {
        const float other = __shfl_sync(0xFFFFFFFF, score, g, WARP_SIZE);
        rank += other > score || (other == score && g < lane);
    }
    const auto keep = __ballot_sync(0xFFFFFFFFULL, lane < n_groups && rank < n_group_used);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int g = (i * WARP_SIZE + lane) / n_per_group;
        if (g >= n_groups || !((keep >> g) & 1)) {
            vals[i] = -INFINITY;
        }
    }
}
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

// Everything topk_moe_cuda does for one row after its logits sit in wt (lane threadIdx.x holds experts
// threadIdx.x + i*WARP_SIZE): gating, bias, group mask, the top-k argmax rounds, norm and scale. weights and ids point
// at the row. Called by one full warp.
template <int n_experts, bool has_bias>
static __device__ void topk_moe_row(float (&wt)[(n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1],
                                   float *               weights,
                                   int32_t *             ids,
                                   const float *         bias,
                                   const int             n_expert_used,
                                   const float           clamp_val,
                                   const float           scale_val,
                                   const topk_moe_config config) {
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
            if (__isnanf(selection_wt[i / WARP_SIZE])) {
                selection_wt[i / WARP_SIZE] = -FLT_MAX;
            }
        }
    }

#if !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000
    // without bias, masking wt is safe: the matcher ensures the kept groups hold all used experts
    if (config.n_groups > 0) {
        if constexpr (has_bias) {
            group_mask_warp_inplace<n_experts>(selection_wt, config.n_groups, config.n_group_used, threadIdx.x);
        } else {
            group_mask_warp_inplace<n_experts>(wt, config.n_groups, config.n_group_used, threadIdx.x);
        }
    }
#endif // !defined(GGML_USE_HIP) || HIP_VERSION >= 60300000

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.norm_eps) {
        // INKGLUE: weight k sits in lane k % 32 (slot k / 32): for n_expert_used <= 32 the warp butterfly adds the
        // weights in the same order as reduce_rows_f32 does with one column per thread, and the epilogue repeats the
        // SCALE (1*sum + eps) and DIV kernels' arithmetic, so the weights match the unfused nodes bit for bit
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            s += output_weights[i];
        }
        s = warp_reduce_sum(s);
        s = 1.0f * s + clamp_val;
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] = output_weights[i] / s;
        }
    } else if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}
