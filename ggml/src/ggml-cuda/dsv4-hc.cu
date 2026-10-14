#include "common.cuh"
#include "convert.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;
static constexpr int DSV4_HC_SQ = DSV4_HC * DSV4_HC; // 16
static constexpr int ELEMS_PER_MATRIX = DSV4_HC_SQ;
static constexpr int TOKENS_PER_WARP = 32 / ELEMS_PER_MATRIX; // 2
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32; // 128
static constexpr int TOKENS_PER_BLOCK = WARPS_PER_BLOCK * TOKENS_PER_WARP; // 8
static constexpr unsigned MASK_FULL = 0xFFFFFFFF;

// Warp-level 4x4 sinkhorn normalization.
// 16 threads = one 4x4 matrix.  2 matrices per warp (lanes 0-15, 16-31).
// Each thread holds and operates on exactly one matrix element.
// Row reductions use XOR mask 1,2 and column reductions use XOR mask 4,8.
static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2 * DSV4_HC;
    constexpr int ROW_REDUCE_MASK_1 = 1;  // adjacent pairs in row
    constexpr int ROW_REDUCE_MASK_2 = 2;  // quartets = full row
    constexpr int COL_REDUCE_MASK_1 = 4;  // pairs across rows (stride 4)
    constexpr int COL_REDUCE_MASK_2 = 8;  // all rows (stride 4, then 8)

    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x % 32;
    const int token_off = lane / ELEMS_PER_MATRIX; // 0 or 1
    const int elem_id   = lane % ELEMS_PER_MATRIX; // 0..15
    const int row = elem_id / DSV4_HC;   // 0..3 (isrc in old kernel)
    const int col = elem_id % DSV4_HC;    // 0..3 (idst in old kernel)

    ggml_cuda_pdl_lc();

    for (int64_t it = (int64_t) blockIdx.x * TOKENS_PER_BLOCK + warp_id * TOKENS_PER_WARP + token_off;
         it < n_tokens;
         it += gridDim.x * TOKENS_PER_BLOCK) {

        const float scale_comb = scale[2 * ss0];
        const int mix_idx = comb_offset + elem_id;

        // Load this element = mixes[combined_offset + elem_id] * scale + base
        float v = mixes[mix_idx * sm0 + it * sm1] * scale_comb + base[mix_idx * sb0];

        // ---- Row softmax (over columns within each row) ----
        // max over the 4 columns of this row
        float max_v = v;
        max_v = fmaxf(max_v, __shfl_xor_sync(MASK_FULL, max_v, ROW_REDUCE_MASK_1));
        max_v = fmaxf(max_v, __shfl_xor_sync(MASK_FULL, max_v, ROW_REDUCE_MASK_2));

        // exp and sum over the 4 columns
        float ex = expf(v - max_v);
        float sum_ex = ex;
        sum_ex += __shfl_xor_sync(MASK_FULL, sum_ex, ROW_REDUCE_MASK_1);
        sum_ex += __shfl_xor_sync(MASK_FULL, sum_ex, ROW_REDUCE_MASK_2);

        // normalize: exp(v - max) / sum + eps
        float comb_val = ex * __frcp_rn(sum_ex) + eps;

        // ---- Column normalization (over rows within each column) ----
        float col_sum = comb_val;
        col_sum += __shfl_xor_sync(MASK_FULL, col_sum, COL_REDUCE_MASK_1);
        col_sum += __shfl_xor_sync(MASK_FULL, col_sum, COL_REDUCE_MASK_2);
        col_sum += eps; // match CPU: sum = eps + elements
        comb_val *= __frcp_rn(col_sum);

        // ---- Sinkhorn iterations ----
        // unroll=1 prevents compiler from massively unrolling the sinkhorn loop
        #pragma unroll 1
        for (int32_t i = 1; i < n_iter; ++i) {
            // Row norm
            float rsum = comb_val;
            rsum += __shfl_xor_sync(MASK_FULL, rsum, ROW_REDUCE_MASK_1);
            rsum += __shfl_xor_sync(MASK_FULL, rsum, ROW_REDUCE_MASK_2);
            rsum += eps; // match CPU: sum = eps + elements
            comb_val *= __frcp_rn(rsum);

            // Col norm
            float csum = comb_val;
            csum += __shfl_xor_sync(MASK_FULL, csum, COL_REDUCE_MASK_1);
            csum += __shfl_xor_sync(MASK_FULL, csum, COL_REDUCE_MASK_2);
            csum += eps; // match CPU: sum = eps + elements
            comb_val *= __frcp_rn(csum);
        }

        // ---- Write ----
        dst[col * sd0 + row * sd1 + it * sd2] = comb_val;
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = BLOCK_SIZE; // 128
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

// [HCPN] GGML_CUDA_HC_POST_NORM=1 / GGML_CUDA_HC_PRE_XN=1, see ggml_cuda_fuse_hc_norm in ggml-cuda.cu.
//
// dsv4_hc_post_norm_f32 runs one row of the hyper-connection norm RMS_NORM -> MUL(gamma) [-> CPY to BF16], and with
// do_post first produces that row with the identity-comb hc_post (dsv4_hc_post_f32<false>'s expression), writing it
// to the hc_post output for the residual stream and keeping it in registers for the norm. The norm arithmetic is
// rms_norm_mul_cast_bf16_f32<old_bs>'s (norm.cu) in the same order: old thread o sums x*x over the columns
// o, o + old_bs, ... in that order, block_reduce<SUM, old_bs> adds the partial sums, and each output is
// (scale * x) * gamma. Here old_bs/R threads each play R old threads (thread t is old thread t + r*old_bs/R, so it
// sits on the same lane of old warp t/32 + r*old_bs/(32*R)); every old partial sum stays in its own register and goes
// through the same warp shuffles, the same shared-memory slot and the same final warp sum, so every value and every
// output byte is the unfused path's. write_xn == false (HC_PRE_XN) writes the per-row scale instead of the f32 output,
// into the first float of that row of the output buffer.
//
// The f32 output (or its scales) may sit exactly on the hc_post residual (same layout, same address): the graph
// allocator gives the norm output the residual's buffer once the hc_post was its last reader. That is race-free here:
// block srow reads only row srow of the residual, all of it before the __syncthreads, and writes only row srow of the
// output after it. Any other overlap between what one block writes and what another reads is refused by the caller.
static constexpr int DSV4_HC_NORM_MAX_ITER = 4; // columns per old thread kept in registers: ncols <= 4*old_bs
static constexpr int DSV4_HC_NORM_R        = 4; // old threads per thread in the 1024-thread layout (256 threads), the fastest
                                                // of 1/2/4 at 12288 tokens

template <int old_bs, int R, bool do_post, bool write_xn>
static __global__ void dsv4_hc_post_norm_f32(
        const float * hx,           // do_post: hc_post x (the block output), [n_embd, n_tokens]
        const float * residual,     // do_post: hc_post residual, [n_embd, hc, n_tokens]
        const float * post,         // do_post: hc_post weights, [hc, n_tokens]
        float *       h,            // do_post: hc_post output, the norm input
        const float * x,            // !do_post: the norm input
        const int64_t sx0,
        const int64_t sx1,
        const int64_t sr0,
        const int64_t sr1,
        const int64_t sr2,
        const int64_t sp0,
        const int64_t sp1,
        const int64_t sh0,
        const int64_t sh1,
        const int64_t sh2,
        const int64_t stride_row,
        const int64_t stride_channel,
        const int64_t stride_sample,
        const int     ncols,
        const float   eps,
        const float * mul,
        const int64_t mul_stride_row,
        const int64_t mul_stride_channel,
        const int64_t mul_stride_sample,
        const uint3   mul_nrows_packed,
        const uint3   mul_nchannels_packed,
        const uint3   mul_nsamples_packed,
        float *       dst,
        nv_bfloat16 * dst_bf16,
        float *       row_scale) {
    constexpr int NB = old_bs / R;          // threads per block
    constexpr int NW = NB / WARP_SIZE;      // warps per block
    constexpr int K  = DSV4_HC_NORM_MAX_ITER;
    static_assert(old_bs % R == 0 && NB % WARP_SIZE == 0 && old_bs <= 1024, "bad block layout");

    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row     = blockIdx.x;
    const int channel = blockIdx.y;
    const int sample  = blockIdx.z;
    const int tid     = threadIdx.x;

    const int64_t srow    = (int64_t(sample)*nchannels + channel)*nrows + row;
    const int64_t dst_off = srow*ncols;

    const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
    const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
    const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
    mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

    if constexpr (!do_post) {
        x += sample*stride_sample + channel*stride_channel + row*stride_row;
    }

    float v[R][K];
    float tmp[R]; // the partial sums of old threads tid + r*NB

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        tmp[r] = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int col = tid + r*NB + k*old_bs;
            v[r][k] = 0.0f;
            if (col < ncols) {
                float xi;
                if constexpr (do_post) {
                    // hc_post element (i0, idst, it) = (col, row, channel); the norm reads hc_post's output row by row
                    float sum = hx[col*sx0 + channel*sx1] * post[row*sp0 + channel*sp1];
                    sum += residual[col*sr0 + row*sr1 + channel*sr2];
                    h[col*sh0 + row*sh1 + channel*sh2] = sum;
                    xi = sum;
                } else {
                    xi = x[col];
                }
                v[r][k] = xi;
                tmp[r] += xi * xi;
            }
        }
    }

    // block_reduce<SUM, old_bs>: each old warp's shuffle sum, one shared slot per old warp, then one warp sum
    __shared__ float s_sum[32];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        tmp[r] = warp_reduce_sum(tmp[r]);
    }
    if (lane_id == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            s_sum[warp_id + r*NW] = tmp[r];
        }
    }
    __syncthreads();
    float total = 0.0f;
    if (lane_id < old_bs / WARP_SIZE) {
        total = s_sum[lane_id];
    }
    total = warp_reduce_sum(total);

    const float mean  = total / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (!write_xn) {
        if (tid == 0) {
            row_scale[dst_off] = scale; // the first float of this row of the f32 output, which only this block touches
        }
    }

#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int col = tid + r*NB + k*old_bs;
            if (col < ncols) {
                const float o = scale * v[r][k] * mul[col];
                if constexpr (write_xn) {
                    dst[dst_off + col] = o;
                }
                if (dst_bf16 != nullptr) {
                    dst_bf16[dst_off + col] = ggml_cuda_cast<nv_bfloat16>(o);
                }
            }
        }
    }
}

// dsv4_hc_pre_f32<true> with each xn value rebuilt from the norm input: xn = (scale * x) * gamma, the norm's two
// products in its order, so the value, and so the sum and the output, are bit-identical. x and xn share the
// [n_embd, hc, n_tokens] element order (hc_pre's xn is a contiguous view of the norm output), the scale of
// (stream, token) row it*hc + ih is the first float of that row of xn's buffer, and gamma is broadcast over rows and
// tokens as the MUL did.
static __global__ void dsv4_hc_pre_xn_f32(
        const float * x,
        const float * gamma,
        const float * row_scale,
        const float * weights,
        float *       dst,
        int64_t       n_embd,
        int64_t       hc,
        int64_t       n_tokens,
        int64_t       sx1,
        int64_t       sx2,
        int64_t       sg1,
        int64_t       sg2,
        int64_t       g_ne1,
        int64_t       g_ne2,
        int64_t       sw0,
        int64_t       sw1,
        int64_t       sw2,
        int64_t       sd0,
        int64_t       sd1,
        float         scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    const float * g_it = gamma + (it % g_ne2)*sg2 + i0;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = row_scale[(it*hc + ih)*n_embd] * x[i0 + ih*sx1 + it*sx2] * g_it[(ih % g_ne1)*sg1];
        const float wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <int old_bs, int R, bool do_post, bool write_xn>
static void dsv4_hc_post_norm_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * post, const ggml_tensor * norm,
                                     const ggml_tensor * mul, nv_bfloat16 * dst_bf16, float * row_scale) {
    const ggml_tensor * src   = norm->src[0];
    const ggml_tensor * gamma = mul->src[0] == norm ? mul->src[1] : mul->src[0];

    float eps = 0.0f;
    memcpy(&eps, norm->op_params, sizeof(float));

    const int64_t ne00 = src->ne[0];
    const dim3 blocks_num(src->ne[1], src->ne[2], src->ne[3]);
    const dim3 block_dims(old_bs / R, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks_num, block_dims, 0, ctx.stream());

    const float * hx  = nullptr;
    const float * res = nullptr;
    const float * pw  = nullptr;
    float * h = nullptr;
    int64_t sx0 = 0, sx1 = 0, sr0 = 0, sr1 = 0, sr2 = 0, sp0 = 0, sp1 = 0, sh0 = 0, sh1 = 0, sh2 = 0;
    if (do_post) {
        const ggml_tensor * x        = post->src[0];
        const ggml_tensor * residual = post->src[1];
        const ggml_tensor * weights  = post->src[2];
        hx  = (const float *) x->data;
        res = (const float *) residual->data;
        pw  = (const float *) weights->data;
        h   = (float *) post->data;
        sx0 = x->nb[0] / sizeof(float);         sx1 = x->nb[1] / sizeof(float);
        sr0 = residual->nb[0] / sizeof(float);  sr1 = residual->nb[1] / sizeof(float); sr2 = residual->nb[2] / sizeof(float);
        sp0 = weights->nb[0] / sizeof(float);   sp1 = weights->nb[1] / sizeof(float);
        sh0 = post->nb[0] / sizeof(float);      sh1 = post->nb[1] / sizeof(float);     sh2 = post->nb[2] / sizeof(float);
    }

    ggml_cuda_kernel_launch(dsv4_hc_post_norm_f32<old_bs, R, do_post, write_xn>, launch_params,
            hx, res, pw, h, (const float *) src->data,
            sx0, sx1, sr0, sr1, sr2, sp0, sp1, sh0, sh1, sh2,
            (int64_t) (src->nb[1] / sizeof(float)), (int64_t) (src->nb[2] / sizeof(float)), (int64_t) (src->nb[3] / sizeof(float)),
            (int) ne00, eps,
            (const float *) gamma->data,
            (int64_t) (gamma->nb[1] / sizeof(float)), (int64_t) (gamma->nb[2] / sizeof(float)), (int64_t) (gamma->nb[3] / sizeof(float)),
            init_fastdiv_values((uint32_t) gamma->ne[1]), init_fastdiv_values((uint32_t) gamma->ne[2]),
            init_fastdiv_values((uint32_t) gamma->ne[3]),
            (float *) mul->data, dst_bf16, row_scale);
}

void ggml_cuda_op_dsv4_hc_post_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * post, const ggml_tensor * norm,
                                    const ggml_tensor * mul, nv_bfloat16 * dst_bf16, float * row_scale) {
    const ggml_tensor * src = norm->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && src->nb[0] == sizeof(float));
    GGML_ASSERT(post == nullptr || src == post);
    GGML_ASSERT(row_scale != nullptr || post != nullptr); // the norm alone with the f32 output is NORM_CAST's kernel
    GGML_ASSERT(row_scale == nullptr || (dst_bf16 != nullptr && row_scale == (const float *) mul->data));
    // same block size choice as rms_norm_f32_cuda: the reduction order, and so the result, depends on it
    const int64_t ne00 = src->ne[0];
    if (ne00 < 1024) {
        if (post != nullptr && row_scale == nullptr) {
            dsv4_hc_post_norm_launch<256, 1, true,  true >(ctx, post, norm, mul, dst_bf16, row_scale);
        } else if (post != nullptr) {
            dsv4_hc_post_norm_launch<256, 1, true,  false>(ctx, post, norm, mul, dst_bf16, row_scale);
        } else {
            dsv4_hc_post_norm_launch<256, 1, false, false>(ctx, post, norm, mul, dst_bf16, row_scale);
        }
    } else {
        GGML_ASSERT(ne00 <= DSV4_HC_NORM_MAX_ITER*1024);
        if (post != nullptr && row_scale == nullptr) {
            dsv4_hc_post_norm_launch<1024, DSV4_HC_NORM_R, true,  true >(ctx, post, norm, mul, dst_bf16, row_scale);
        } else if (post != nullptr) {
            dsv4_hc_post_norm_launch<1024, DSV4_HC_NORM_R, true,  false>(ctx, post, norm, mul, dst_bf16, row_scale);
        } else {
            dsv4_hc_post_norm_launch<1024, DSV4_HC_NORM_R, false, false>(ctx, post, norm, mul, dst_bf16, row_scale);
        }
    }
}

void ggml_cuda_op_dsv4_hc_pre_xn(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * x,
                                 const ggml_tensor * gamma, const float * row_scale) {
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(gamma->type == GGML_TYPE_F32 && gamma->nb[0] == sizeof(float));
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_get_op_params_i32(dst, 1) != 0);
    GGML_ASSERT(ggml_are_same_shape(x, dst->src[0]) && x->ne[3] == 1);

    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_xn_f32, launch_params,
            (const float *) x->data, (const float *) gamma->data, row_scale, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)),
            (int64_t) (gamma->nb[1] / sizeof(float)), (int64_t) (gamma->nb[2] / sizeof(float)), gamma->ne[1], gamma->ne[2],
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}
