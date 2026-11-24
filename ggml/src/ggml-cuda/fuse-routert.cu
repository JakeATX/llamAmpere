#include "fuse-routert.cuh"
#include "topk-moe-device.cuh"

#include <mutex>
#include <unordered_map>

#define ROUTERT_BLOCK     256
#define ROUTERT_MAX_SLOTS 64

// Zero at module load; the last block of every launch resets its slot, so a CUDA graph replays with no host work.
static __device__ unsigned int g_routert_done[ROUTERT_MAX_SLOTS];

// Grid = n_experts blocks of (32, 8) threads (linear id = mul_mat_vec_f's tid for its block of 256): block `row` computes
// expert row `row` against every column the way mul_mat_vec_f<nv_bfloat16, float, ncols, 256> does (CUDA
// nv_bfloat162 branch, per-column warp / block reductions). The last block to finish selects: warp r takes row r.
template <int n_experts, bool has_bias, int ncols>
static __global__ void __launch_bounds__(ROUTERT_BLOCK)
k_fuse_routert(const nv_bfloat16 * GGML_CUDA_RESTRICT w, const float * GGML_CUDA_RESTRICT y, float * logits, float * weights,
               int32_t * ids, const float * bias, const int ncols2, const int64_t stride_row, const int64_t stride_col_y2,
               const int n_expert_used, const float clamp_val, const float scale_val, const topk_moe_config config,
               const int slot) {
    constexpr int block_size = ROUTERT_BLOCK;
    constexpr int warp_size  = WARP_SIZE;
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    NO_DEVICE_CODE;
#else
    const int row = blockIdx.x;
    const int tid = threadIdx.y*warp_size + threadIdx.x;

    ggml_cuda_pdl_sync();

    const nv_bfloat162 * x2 = (const nv_bfloat162 *) (w + row*stride_row);
    const float2       * y2 = (const float2 *) y;

    __shared__ float        buf_iw[warp_size];
    __shared__ unsigned int s_last;
    if (tid < warp_size) {
        buf_iw[tid] = 0.0f;
    }
    __syncthreads();

    float sumf[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sumf[j] = 0.0f;
    }
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
        const nv_bfloat162 tmpx = x2[col2];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const float2 tmpy = y2[j*stride_col_y2 + col2];
            ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
            ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
        }
    }
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        buf_iw[tid/warp_size] = sumf[j];
        __syncthreads();
        if (tid < warp_size) {
            sumf[j] = buf_iw[tid];
            sumf[j] = warp_reduce_sum<warp_size>(sumf[j]);
        }
        __syncthreads();
    }

    if (tid == 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            logits[j*n_experts + row] = sumf[j];
        }
        __threadfence();
        const unsigned int prev = atomicAdd(&g_routert_done[slot], 1u);
        s_last = prev == gridDim.x - 1;
        if (s_last) {
            g_routert_done[slot] = 0; // every block has counted: ready for the next launch
        }
        __threadfence();
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();

    // topk_moe_cuda's load of a row (through L2: other blocks wrote the logits); warp r < ncols takes row r
    const int r = threadIdx.y;
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;
    float wt[experts_per_thread];
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }
    if (r < ncols) {
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert  = i + threadIdx.x;
            wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? __ldcg(logits + r*n_experts + expert) : -INFINITY;
        }
    }
    // ids / weights may alias any row's logits: every row is loaded before any row stores (as topk_moe_cuda's block)
    __syncthreads();
    if (r >= ncols) {
        return;
    }
    topk_moe_row<n_experts, has_bias>(wt, weights + r*n_expert_used, ids + r*n_experts, bias, n_expert_used, clamp_val,
        scale_val, config);
#endif // defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
}

static int routert_slot(const ggml_backend_cuda_context * ctx) {
    static std::mutex                            mtx;
    static std::unordered_map<const void *, int> slots;
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = slots.find(ctx);
    if (it != slots.end()) {
        return it->second;
    }
    if (slots.size() >= ROUTERT_MAX_SLOTS) {
        return -1;
    }
    const int slot = (int) slots.size();
    slots.emplace(ctx, slot);
    return slot;
}

struct routert_launch_args {
    const nv_bfloat16 * w; const float * y; float * logits; float * weights; int32_t * ids; const float * bias;
    int ncols2; int64_t stride_row; int64_t stride_col_y2; int n_expert_used; float clamp_val; float scale_val;
    topk_moe_config config; int slot;
};

template <int n_experts, int ncols>
static void routert_launch(const ggml_cuda_kernel_launch_params & lp, const routert_launch_args & a) {
    if (a.bias) {
        ggml_cuda_kernel_launch(k_fuse_routert<n_experts, true, ncols>, lp, a.w, a.y, a.logits, a.weights, a.ids, a.bias,
            a.ncols2, a.stride_row, a.stride_col_y2, a.n_expert_used, a.clamp_val, a.scale_val, a.config, a.slot);
    } else {
        ggml_cuda_kernel_launch(k_fuse_routert<n_experts, false, ncols>, lp, a.w, a.y, a.logits, a.weights, a.ids, a.bias,
            a.ncols2, a.stride_row, a.stride_col_y2, a.n_expert_used, a.clamp_val, a.scale_val, a.config, a.slot);
    }
}

template <int n_experts>
static void routert_launch_rows(const ggml_cuda_kernel_launch_params & lp, const routert_launch_args & a, const int n_rows) {
    switch (n_rows) {
        case 2: routert_launch<n_experts, 2>(lp, a); break;
        case 3: routert_launch<n_experts, 3>(lp, a); break;
        case 4: routert_launch<n_experts, 4>(lp, a); break;
        case 5: routert_launch<n_experts, 5>(lp, a); break;
        case 6: routert_launch<n_experts, 6>(lp, a); break;
        case 7: routert_launch<n_experts, 7>(lp, a); break;
        case 8: routert_launch<n_experts, 8>(lp, a); break;
        default: GGML_ABORT("routert: unsupported row count %d", n_rows);
    }
}

bool ggml_cuda_op_router_t(ggml_backend_cuda_context &     ctx,
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

    const int n_experts = (int) logits->ne[0];
    const int n_rows    = (int) logits->ne[1];
    if (ctx.curr_stream_no != 0 || n_rows < 2 || n_rows > GGML_CUDA_ROUTERT_MAX_ROWS ||
            (n_experts != 32 && n_experts != 64 && n_experts != 128)) {
        return false;
    }
    const int slot = routert_slot(&ctx);
    if (slot < 0) {
        return false;
    }
    GGML_ASSERT(ggml_nrows(logits) == n_rows && y->ne[1] == n_rows && ggml_is_contiguous(y));
    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    // ggml_cuda_op_topk_moe's arguments (topk-moe.cu)
    const bool norm_eps  = clamp != nullptr && clamp->op == GGML_OP_SCALE;
    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, norm_eps ? 1 : 0); // SCALE: its bias (eps)
    }
    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = clamp != nullptr;
    config.norm_eps          = norm_eps;
    config.delayed_softmax   = args.delayed_softmax;
    config.n_groups          = args.n_groups;
    config.n_group_used      = args.n_group_used;

    routert_launch_args a;
    a.w             = (const nv_bfloat16 *) w->data;
    a.y             = (const float *) y->data;
    a.logits        = (float *) logits->data;
    a.weights       = (float *) weights->data;
    a.ids           = (int32_t *) ids->data;
    a.bias          = bias ? (const float *) bias->data : nullptr;
    a.ncols2        = (int) (w->ne[0] / 2);
    a.stride_row    = w->nb[1] / ggml_type_size(w->type);
    a.stride_col_y2 = (int64_t) (y->nb[1] / sizeof(float) / 2);
    a.n_expert_used = (int) weights->ne[1];
    a.clamp_val     = clamp_val;
    a.scale_val     = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;
    a.config        = config;
    a.slot          = slot;

    const ggml_cuda_kernel_launch_params lp(dim3(n_experts, 1, 1), dim3(WARP_SIZE, ROUTERT_BLOCK/WARP_SIZE, 1), 0, ctx.stream());
    switch (n_experts) {
        case 32:  routert_launch_rows<32>(lp, a, n_rows);  break;
        case 64:  routert_launch_rows<64>(lp, a, n_rows);  break;
        case 128: routert_launch_rows<128>(lp, a, n_rows); break;
        default: GGML_ABORT("routert: unsupported n_expert %d", n_experts);
    }
    return true;
}
