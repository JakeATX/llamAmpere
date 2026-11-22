#include "inkrouter.cuh"
#include "topk-moe-device.cuh"

#include <climits>
#include <mutex>
#include <unordered_map>

// The mul_mat_vec_f block size whose launch the logit blocks repeat (mmvf picks it for ncols = 2304;
// ggml_cuda_inkrouter_supported checks the pick for the actual ncols).
#define INKROUTER_MMVF_BLOCK     256
// Completion counters: one per backend context, so contexts on different streams never share one.
#define INKROUTER_MAX_SLOTS      64

// Zero at module load; the last block of every launch resets its slot, so a CUDA graph replays with no host work.
static __device__ unsigned int g_inkrouter_done[INKROUTER_MAX_SLOTS];

// Grid = n_experts blocks of INKROUTER_MMVF_BLOCK threads: block `row` is mul_mat_vec_f<nv_bfloat16, float, 1, 256,
// false, false>'s block `row` (mmvf.cu: one dst column, no ids, no fusion, CUDA nv_bfloat162 branch), line for line.
template <int n_experts, bool has_bias>
static __global__ void __launch_bounds__(INKROUTER_MMVF_BLOCK)
k_inkrouter(const nv_bfloat16 * __restrict__ w, const float * __restrict__ y, float * logits, float * weights, int32_t * ids,
            const float * bias, const int ncols2, const int64_t stride_row, const int n_expert_used, const float clamp_val,
            const float scale_val, const topk_moe_config config, const int slot) {
    constexpr int block_size = INKROUTER_MMVF_BLOCK;
    constexpr int warp_size  = ggml_cuda_get_physical_warp_size();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();

    const nv_bfloat162 * x2 = (const nv_bfloat162 *) (w + row*stride_row);
    const float2       * y2 = (const float2 *) y;

    __shared__ float buf_iw[warp_size];
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf = 0.0f;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const nv_bfloat162 tmpx = x2[col2];
        const float2       tmpy = y2[col2];
        ggml_cuda_mad(sumf, tmpx.x, tmpy.x);
        ggml_cuda_mad(sumf, tmpx.y, tmpy.y);
    }

    ggml_cuda_pdl_lc();

    sumf = warp_reduce_sum<warp_size>(sumf);
    buf_iw[tid/warp_size] = sumf;
    __syncthreads();
    if (tid >= warp_size) {
        return; // mmvf's other warps are done here too (only thread 0 writes)
    }
    sumf = buf_iw[tid];
    sumf = warp_reduce_sum<warp_size>(sumf);

    // completion count: thread 0 of every block; the warp learns by shuffle whether its block came last (no block barrier)
    unsigned int prev = 0;
    if (tid == 0) {
        logits[row] = sumf;
        __threadfence();
        prev = atomicAdd(&g_inkrouter_done[slot], 1u);
        __threadfence(); // acquire side for the observing lane (review note: shfl alone is not a sync op)
    }
    __syncwarp();        // orders lane 0's observation before the other lanes' logit loads
    prev = __shfl_sync(0xFFFFFFFF, prev, 0, WARP_SIZE);
    if (prev != gridDim.x - 1) {
        return;
    }
    if (tid == 0) {
        g_inkrouter_done[slot] = 0; // every block has counted: ready for the next launch
    }
    __threadfence();

    // topk_moe_cuda's load of its row (topk-moe.cu), through L2 since other blocks wrote the logits
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? __ldcg(logits + expert) : -INFINITY;
    }

    // one warp: its shuffles order every lane's logit load before any ids / weights store, which may alias the logits
    topk_moe_row<n_experts, has_bias>(wt, weights, ids, bias, n_expert_used, clamp_val, scale_val, config);
}

// mmvf.cu launch_mul_mat_vec_f_cuda: the block size it picks for ncols
static int64_t inkrouter_mmvf_block_size(const int64_t ncols, const int warp_size, const int cc) {
    int64_t block_size_best = warp_size;
    int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
    int64_t max_block_size  = 256;
    if (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) {
        max_block_size = 128;
    }
    for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
        const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
        if (niter < niter_best) {
            niter_best      = niter;
            block_size_best = block_size;
        }
    }
    return block_size_best;
}

bool ggml_cuda_inkrouter_supported(const ggml_tensor * w, const ggml_tensor * y, const int n_expert_used, const int cc,
                                   const int warp_size) {
    const int64_t ncols    = w->ne[0];
    const int64_t n_expert = w->ne[1];
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || warp_size != WARP_SIZE) {
        return false;
    }
    if (n_expert != 32 && n_expert != 64 && n_expert != 128) {
        return false;
    }
    if (n_expert_used < 1 || n_expert_used > n_expert || n_expert_used > WARP_SIZE) {
        return false;
    }
    if (ncols % 2 != 0 || ncols > INT_MAX || y->ne[0] != ncols) {
        return false;
    }
    return inkrouter_mmvf_block_size(ncols, warp_size, cc) == INKROUTER_MMVF_BLOCK;
}

static int inkrouter_slot(const ggml_backend_cuda_context * ctx) {
    static std::mutex                                   mtx;
    static std::unordered_map<const void *, int>        slots;
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = slots.find(ctx);
    if (it != slots.end()) {
        return it->second;
    }
    if (slots.size() >= INKROUTER_MAX_SLOTS) {
        return -1;
    }
    const int slot = (int) slots.size();
    slots.emplace(ctx, slot);
    return slot;
}

template <int n_experts>
static void inkrouter_launch(const bool has_bias, const ggml_cuda_kernel_launch_params & lp, const nv_bfloat16 * w,
        const float * y, float * logits, float * weights, int32_t * ids, const float * bias, const int ncols2,
        const int64_t stride_row, const int n_expert_used, const float clamp_val, const float scale_val,
        const topk_moe_config config, const int slot) {
    if (has_bias) {
        ggml_cuda_kernel_launch(k_inkrouter<n_experts, true>, lp, w, y, logits, weights, ids, bias, ncols2, stride_row,
            n_expert_used, clamp_val, scale_val, config, slot);
    } else {
        ggml_cuda_kernel_launch(k_inkrouter<n_experts, false>, lp, w, y, logits, weights, ids, bias, ncols2, stride_row,
            n_expert_used, clamp_val, scale_val, config, slot);
    }
}

bool ggml_cuda_op_inkrouter(ggml_backend_cuda_context &     ctx,
                            const ggml_tensor *             w,
                            const ggml_tensor *             y,
                            ggml_tensor *                   logits,
                            ggml_tensor *                   weights,
                            ggml_tensor *                   ids,
                            const ggml_tensor *             clamp,
                            const ggml_tensor *             scale,
                            const ggml_tensor *             bias,
                            const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(w->type == GGML_TYPE_BF16 && y->type == GGML_TYPE_F32);
    GGML_ASSERT(logits->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(!args.delayed_softmax && args.n_groups == 0);

    const int slot = inkrouter_slot(&ctx);
    if (slot < 0) {
        return false;
    }

    const int n_experts     = logits->ne[0];
    const int n_expert_used = weights->ne[1];
    GGML_ASSERT(ggml_nrows(logits) == 1);
    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    // ggml_cuda_op_topk_moe's arguments (topk-moe.cu)
    const float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;
    const bool  with_norm = clamp != nullptr;
    const bool  norm_eps  = clamp != nullptr && clamp->op == GGML_OP_SCALE;
    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, norm_eps ? 1 : 0); // SCALE: its bias (eps)
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.norm_eps          = norm_eps;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    const ggml_cuda_kernel_launch_params lp(dim3(n_experts, 1, 1), dim3(INKROUTER_MMVF_BLOCK, 1, 1), 0, ctx.stream());

    const nv_bfloat16 * w_d       = (const nv_bfloat16 *) w->data;
    const float *       y_d       = (const float *) y->data;
    float *             logits_d  = (float *) logits->data;
    float *             weights_d = (float *) weights->data;
    int32_t *           ids_d     = (int32_t *) ids->data;
    const float *       bias_d    = bias ? (const float *) bias->data : nullptr;
    const int           ncols2    = (int) (w->ne[0] / 2);
    const int64_t       s01       = w->nb[1] / ggml_type_size(w->type);

    switch (n_experts) {
        case 32:
            inkrouter_launch<32>(bias != nullptr, lp, w_d, y_d, logits_d, weights_d, ids_d, bias_d, ncols2, s01,
                n_expert_used, clamp_val, scale_val, config, slot);
            break;
        case 64:
            inkrouter_launch<64>(bias != nullptr, lp, w_d, y_d, logits_d, weights_d, ids_d, bias_d, ncols2, s01,
                n_expert_used, clamp_val, scale_val, config, slot);
            break;
        case 128:
            inkrouter_launch<128>(bias != nullptr, lp, w_d, y_d, logits_d, weights_d, ids_d, bias_d, ncols2, s01,
                n_expert_used, clamp_val, scale_val, config, slot);
            break;
        default:
            GGML_ABORT("inkrouter: unsupported n_expert %d", n_experts);
    }
    return true;
}
