#include "moe-weighted-reduction.cuh"

static __global__ void moe_mixed_reduction_f32(const float * experts, const float * weights,
        const int32_t * bank_ids, size_t ids_stride, float * dst, int n_embd, int n_used, bool tree) {
    const int lane = threadIdx.x & 31;
    const int col = blockIdx.x*4 + (threadIdx.x >> 5);
    if (col >= n_embd) { return; }
    const int token = blockIdx.y;
    const int64_t first = int64_t(token)*n_used;
    float hot = 0.0f, cold = 0.0f;
    if (tree) {
        if (lane < n_used) {
            const float value = __fmul_rn(experts[(first + lane)*n_embd + col], weights[first + lane]);
            if (bank_ids[token*ids_stride + lane] >= 0) { hot = value; }
            else { cold = value; }
        }
        hot = warp_reduce_sum(hot);
        cold = warp_reduce_sum(cold);
    } else if (lane == 0) {
        for (int e = 0; e < n_used; ++e) {
            const float value = __fmul_rn(experts[(first + e)*n_embd + col], weights[first + e]);
            if (bank_ids[token*ids_stride + e] >= 0) { hot = __fadd_rn(hot, value); }
            else { cold = __fadd_rn(cold, value); }
        }
    }
    if (lane == 0) { dst[int64_t(token)*n_embd + col] = __fadd_rn(hot, cold); }
}

void ggml_cuda_op_moe_mixed_reduction(ggml_backend_cuda_context & ctx,
        const ggml_tensor * experts, const ggml_tensor * weights,
        const ggml_tensor * bank_ids, ggml_tensor * dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32);
    GGML_ASSERT(bank_ids->type == GGML_TYPE_I32 && bank_ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(experts->ne[1] <= 32 && ggml_is_contiguous(experts) && ggml_is_contiguous(weights));
    const dim3 blocks((experts->ne[0] + 3)/4, experts->ne[2]);
    moe_mixed_reduction_f32<<<blocks, 128, 0, ctx.stream()>>>(
            (const float *) experts->data, (const float *) weights->data, (const int32_t *) bank_ids->data,
            bank_ids->nb[1]/sizeof(int32_t), (float *) dst->data, experts->ne[0], experts->ne[1],
            ggml_cuda_moe_mixed_reduction_tree(experts));
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_moe_mixed_reduction_tree(const ggml_tensor * experts) {
    static const bool small = getenv("LLAMA_MOE_SMALL_REDUCE") && atoi(getenv("LLAMA_MOE_SMALL_REDUCE"));
    return small || experts->ne[2] == 1;
}

// The routed sum, shared by the plain and the combine kernel so both evaluate the same expression.
static __device__ __forceinline__ float moe_weighted_sum(const float * __restrict__ experts,
                                                         const float * __restrict__ expert_scale,
                                                         const float * __restrict__ weights,
                                                         const int64_t n_embd,
                                                         const int     n_expert_used,
                                                         const int64_t token,
                                                         const int64_t col) {
    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    return sum;
}

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    dst[token * n_embd + col] = moe_weighted_sum(experts, expert_scale, weights, n_embd, n_expert_used, token, col);
}

// dst = routed_sum + shexp * sigmoid(gate_logit), one value of gate_logit per token. The sigmoid is written as in
// unary.cu (op_sigmoid) and the mul and the add round separately (no contraction), as the unfused graph does.
// No __restrict__ on shexp/dst: dst may be the shexp buffer itself (same index, read before the write).
static __global__ void moe_weighted_combine_f32(const float * __restrict__ experts,
                                                const float * __restrict__ expert_scale,
                                                const float * __restrict__ weights,
                                                const float * __restrict__ gate_logit,
                                                const int64_t s_gate,
                                                const float * shexp,
                                                const int64_t s_shexp,
                                                float *       dst,
                                                const int64_t s_dst,
                                                const int64_t n_embd,
                                                const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    const float sum  = moe_weighted_sum(experts, expert_scale, weights, n_embd, n_expert_used, token, col);
    const float sg   = 1.0f / (1.0f + expf(-gate_logit[token * s_gate]));
    const float gated = __fmul_rn(shexp[token * s_shexp + col], sg);
    dst[token * s_dst + col] = __fadd_rn(sum, gated);
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_moe_weighted_combine(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor *         experts,
                                       const ggml_tensor *         expert_scale,
                                       const ggml_tensor *         weights,
                                       const ggml_tensor *         gate_logit,
                                       const ggml_tensor *         shexp,
                                       ggml_tensor *               dst) {
    GGML_ASSERT(ggml_is_contiguous(experts) && ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(shexp->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];

    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_combine_f32<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *) experts->data, expert_scale ? (const float *) expert_scale->data : nullptr,
        (const float *) weights->data, (const float *) gate_logit->data, (int64_t) (gate_logit->nb[1] / sizeof(float)),
        (const float *) shexp->data, (int64_t) (shexp->nb[1] / sizeof(float)),
        (float *) dst->data, (int64_t) (dst->nb[1] / sizeof(float)), n_embd, (int) n_expert_used);
    CUDA_CHECK(cudaGetLastError());
}

static __global__ void moe_bank_joint_reduction_f32(const float * hot, const float * cold,
        const float * hot_weights, const float * cold_weights, float * dst, int64_t n_embd, int n_used) {
    const int64_t col = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }
    const int64_t first = int64_t(blockIdx.x) * n_used;
    float sum = __fadd_rn(__fmul_rn(hot[first*n_embd + col], hot_weights[first]),
                         __fmul_rn(cold[first*n_embd + col], cold_weights[first]));
    for (int e = 1; e < n_used; ++e) {
        const int64_t row = first + e;
        const float value = __fadd_rn(__fmul_rn(hot[row*n_embd + col], hot_weights[row]),
                                     __fmul_rn(cold[row*n_embd + col], cold_weights[row]));
        sum = __fadd_rn(sum, value);
    }
    dst[int64_t(blockIdx.x)*n_embd + col] = sum;
}

void ggml_cuda_op_moe_bank_joint_reduction(ggml_backend_cuda_context & ctx,
        const ggml_tensor * hot, const ggml_tensor * cold,
        const ggml_tensor * hot_weights, const ggml_tensor * cold_weights, ggml_tensor * dst) {
    GGML_ASSERT(hot->type == GGML_TYPE_F32 && cold->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(hot_weights->type == GGML_TYPE_F32 && cold_weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(hot) && ggml_is_contiguous(cold) && ggml_are_same_shape(hot, cold));
    GGML_ASSERT(ggml_is_contiguous(hot_weights) && ggml_is_contiguous(cold_weights) && ggml_is_contiguous(dst));
    GGML_ASSERT(hot->ne[1] >= 1 && hot->ne[1] <= 15 && hot->ne[2] >= 1 && hot->ne[2] <= 8 && hot->ne[3] == 1);
    GGML_ASSERT(hot_weights->ne[0] == 1 && hot_weights->ne[1] == hot->ne[1] && hot_weights->ne[2] == hot->ne[2] && hot_weights->ne[3] == 1);
    GGML_ASSERT(ggml_are_same_shape(hot_weights, cold_weights));
    GGML_ASSERT(dst->ne[0] == hot->ne[0] && dst->ne[1] == hot->ne[2] && dst->ne[2] == 1 && dst->ne[3] == 1);
    constexpr int threads = 256;
    const dim3 blocks(hot->ne[2], (hot->ne[0] + threads - 1)/threads, 1);
    moe_bank_joint_reduction_f32<<<blocks, threads, 0, ctx.stream()>>>(
        (const float *) hot->data, (const float *) cold->data, (const float *) hot_weights->data,
        (const float *) cold_weights->data, (float *) dst->data, hot->ne[0], (int) hot->ne[1]);
    CUDA_CHECK(cudaGetLastError());
    static const bool trace = [] {
        const char * value = getenv("GGML_CUDA_MOE_BANK_JOINT_TRACE");
        return value && atoi(value) != 0;
    }();
    if (trace) {
        GGML_LOG_INFO("[moe-bank-joint] launch n_embd=%lld n_used=%lld n_tokens=%lld\n",
                (long long) hot->ne[0], (long long) hot->ne[1], (long long) hot->ne[2]);
    }
}
