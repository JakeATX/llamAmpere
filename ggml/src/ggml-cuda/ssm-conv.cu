#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

// [#110 SF fold 2] ssm_conv_f32 with the conv input read from the two CONCAT operands instead of the CONCAT
// result, plus the conv-state snapshot windows the CPY / CONT+SET_ROWS nodes would have copied out of it. The
// conv arithmetic is ssm_conv_f32's, term for term (same products, same order, same bias add and SILU), so the
// output bytes match; the windows are bit copies. Each thread owns one channel of one sequence: it reads every
// value it needs before it writes anything, and it is the only writer of that channel's window columns.
template <bool apply_silu, int d_conv, int n_t>
static __global__ void ssm_conv_ring_f32(const ggml_cuda_conv_ring_args args, const float * __restrict__ w_ptr,
                                         const int w_nb1, const float * __restrict__ bias,
                                         float * __restrict__ dst, const int dst_nb1, const int dst_nb2) {
    ggml_cuda_pdl_lc();
    constexpr int n_keep = d_conv - 1;
    constexpr int n_col  = n_keep + n_t;

    const int     tid = threadIdx.x;
    const int     s   = blockIdx.x;
    const int64_t c   = (int64_t) blockIdx.y * blockDim.x + tid;

    float x[n_col];
    float w[d_conv];

    ggml_cuda_pdl_sync();
    const float * xs = (const float *) ((const char *) args.state + s * args.state_nb2 + c * args.state_nb1);
#pragma unroll
    for (int j = 0; j < n_keep; ++j) {
        x[j] = xs[j];
    }
    const char * xn = (const char *) args.x + s * args.x_nb2 + c * args.x_nb1;
#pragma unroll
    for (int i = 0; i < n_t; ++i) {
        x[n_keep + i] = *(const float *) (xn + i * args.x_nb0);
    }
    const int stride_w = w_nb1 / sizeof(float);
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        w[j] = w_ptr[c * stride_w + j];
    }
    const float b = bias != nullptr ? bias[c] : 0.0f;

    // windows in graph order; s_idx picks the columns through an unrolled compare so x stays in registers
#pragma unroll
    for (int k = 0; k < GGML_CUDA_CONV_RING_MAX_WIN; ++k) {
        if (k < args.n_win) {
            const ggml_cuda_conv_ring_window & win = args.win[k];
            const int64_t row = win.rows != nullptr ? (int64_t) win.rows[s] : (int64_t) s;
            float * out = win.dst + row * win.row_stride + c * n_keep;
#pragma unroll
            for (int q = 0; q <= n_t; ++q) {
                if (q == win.s_idx) {
#pragma unroll
                    for (int j = 0; j < n_keep; ++j) {
                        out[j] = x[q + j];
                    }
                }
            }
        }
    }

    float * y = (float *) ((char *) dst + s * dst_nb2) + c;
    const int stride_y = dst_nb1 / sizeof(float);
#pragma unroll
    for (int i = 0; i < n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; j++) {
            sumf += x[i + j] * w[j];
        }
        sumf += b;
        y[i * stride_y] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_ring_f32_cuda(const ggml_cuda_conv_ring_args & args, const float * w, const int w_nb1,
                                   const float * bias, float * dst, const int dst_nb1, const int dst_nb2,
                                   const int64_t nr, const int64_t n_t, const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);
    const dim3 blocks(n_s, nr / threads, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);

    auto launch = [&](auto NT) {
        constexpr int kNT = decltype(NT)::value;
        ggml_cuda_kernel_launch(ssm_conv_ring_f32<apply_silu, 4, kNT>, launch_params, args, w, w_nb1, bias,
                                dst, dst_nb1, dst_nb2);
    };
    // the plan (ggml_cuda_conv_ring_match) only accepts d_conv 4 and 1 <= n_t <= GGML_CUDA_CONV_RING_MAX_T
    switch (n_t) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: GGML_ABORT("conv ring: n_t %lld outside 1..%d", (long long) n_t, GGML_CUDA_CONV_RING_MAX_T);
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    // [#110] planned conv ring: src0 (the CONCAT) was not computed; read its operands instead
    for (auto & e : ctx.conv_rings) {
        if (e.conv != dst) {
            continue;
        }
        GGML_ASSERT(nc == 4 && n_t >= 1 && n_t <= GGML_CUDA_CONV_RING_MAX_T && out->nb[0] == sizeof(float));
        if (fuse_silu) {
            ssm_conv_ring_f32_cuda<true>(e.args, src1_d, src1->nb[1], bias_d, dst_d, out->nb[1], out->nb[2], nr, n_t, n_s, stream);
        } else {
            ssm_conv_ring_f32_cuda<false>(e.args, src1_d, src1->nb[1], bias_d, dst_d, out->nb[1], out->nb[2], nr, n_t, n_s, stream);
        }
        e.used = true;
        ctx.fusion_stats.conv_ring++;
        return;
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
