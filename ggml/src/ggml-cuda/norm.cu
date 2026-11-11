#include "norm.cuh"
#include "convert.cuh"
#include "mmq.cuh"
#include <cstdint>
#include <climits>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float2 s_sum2[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float2 mean_var = make_float2(0.0f, 0.0f);

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                mean_var.x += xi;
                mean_var.y += xi * xi;
            }

            // sum up partial sums
            mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

            const float mean = mean_var.x / ncols;
            const float var = mean_var.y / ncols - mean * mean;
            const float inv_std = rsqrtf(var + eps);

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = (xc[col] - mean) * inv_std;
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum2 across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

template <int block_size, int vpt, bool do_vec, bool do_multiply = false, bool do_add = false, bool do_scale = false>
static __global__ void rms_norm_f32(const float * x,
                                    float *       dst,
                                    const int     ncols,
                                    const int     nchannels,
                                    const int     nsamples,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const float * mul                  = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float * add                  = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const float   scale_out            = 1.0f) {
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            [[maybe_unused]] const float * mulc = nullptr;
            if constexpr (do_multiply) {
                const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
                const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
                const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
                mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
            }

            [[maybe_unused]] const float * addc = nullptr;
            if constexpr (do_add) {
                const int add_row     = fastmodulo(row, add_nrows_packed);
                const int add_channel = fastmodulo(channel, add_nchannels_packed);
                const int add_sample  = fastmodulo(sample, add_nsamples_packed);
                addc = add + add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
            }

            if constexpr (do_vec) {
                const int      ncols4 = ncols / 4;
                const float4 * x4     = (const float4 *) xc;
                const float4 * mul4   = (const float4 *) mulc;
                const float4 * add4   = (const float4 *) addc;
                float4 *       dst4   = (float4 *) dstc;

                float4 xs[vpt];
                float  tmp = 0.0f;

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        xs[k] = x4[i];
                        tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
                    }
                }
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
                const float scale = rsqrtf(tmp / ncols + eps);

#pragma unroll
                for (int k = 0; k < vpt; k++) {
                    const int i = tid + k*block_size;
                    if (i < ncols4) {
                        float4 r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
                        if constexpr (do_multiply) {
                            const float4 m = mul4[i];
                            r.x *= m.x;
                            r.y *= m.y;
                            r.z *= m.z;
                            r.w *= m.w;
                        }
                        if constexpr (do_add) {
                            const float4 a = add4[i];
                            r.x += a.x;
                            r.y += a.y;
                            r.z += a.z;
                            r.w += a.w;
                        }
                        if constexpr (do_scale) {
                            r.x *= scale_out;
                            r.y *= scale_out;
                            r.z *= scale_out;
                            r.w *= scale_out;
                        }
                        dst4[i] = r;
                    }
                }
            } else {
                float tmp = 0.0f; // partial sum for thread in warp

                for (int col = tid; col < ncols; col += block_size) {
                    const float xi = xc[col];
                    tmp += xi * xi;
                }

                // sum up partial sums
                tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

                const float mean = tmp / ncols;
                const float scale = rsqrtf(mean + eps);

                for (int col = tid; col < ncols; col += block_size) {
                    if constexpr (do_multiply && do_add) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        const int add_col = fastmodulo(col, add_ncols_packed);
                        dstc[col]         = scale * xc[col] * mulc[mul_col] + addc[add_col];
                    } else if constexpr (do_multiply) {
                        const int mul_col = fastmodulo(col, mul_ncols_packed);
                        dstc[col]         = scale * xc[col] * mulc[mul_col];
                    } else if constexpr (do_scale) {
                        dstc[col] = scale_out * (scale * xc[col]);
                    } else {
                        dstc[col] = scale * xc[col];
                    }
                }
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

// [CAST] rms_norm_f32<block_size, true> that also writes a BF16 copy of dst (GGML_CUDA_NORM_CAST=1, see ggml-cuda.cu).
// The sum, its reduction and the f32 dst are rms_norm_f32's, in the same order; the copy is ggml_cuda_cast<nv_bfloat16>
// of the same f32 value (round to nearest even), what the separate CPY kernel would write. dst_bf16 has dst's layout.
template <int block_size>
static __global__ void rms_norm_mul_cast_bf16_f32(const float * x,
                                                  float *       dst,
                                                  nv_bfloat16 * dst_bf16,
                                                  const int     ncols,
                                                  const int     nchannels,
                                                  const int     nsamples,
                                                  const int64_t stride_row,
                                                  const int64_t stride_channel,
                                                  const int64_t stride_sample,
                                                  const float   eps,
                                                  const float * mul,
                                                  const int64_t mul_stride_row,
                                                  const int64_t mul_stride_channel,
                                                  const int64_t mul_stride_sample,
                                                  const uint3   mul_ncols_packed,
                                                  const uint3   mul_nrows_packed,
                                                  const uint3   mul_nchannels_packed,
                                                  const uint3   mul_nsamples_packed) {
    ggml_cuda_pdl_lc();
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit (as upstream's rms_norm_f32), iterate over the excess
    // channels/samples; per row the arithmetic is unchanged
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc = x + sample*stride_sample + channel*stride_channel + row*stride_row;
            const int64_t dst_off = ((int64_t(sample)*nchannels + channel)*nrows + row)*ncols;
            float       * dstc      = dst      + dst_off;
            nv_bfloat16 * dstc_bf16 = dst_bf16 + dst_off;

            const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
            const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
            const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
            const float * mulc = mul + mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            // sum up partial sums
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            const float mean = tmp / ncols;
            const float scale = rsqrtf(mean + eps);

            for (int col = tid; col < ncols; col += block_size) {
                const int   mul_col = fastmodulo(col, mul_ncols_packed);
                const float v       = scale * xc[col] * mulc[mul_col];
                dstc[col]      = v;
                dstc_bf16[col] = ggml_cuda_cast<nv_bfloat16>(v);
            }

            if constexpr (block_size > WARP_SIZE) {
                __syncthreads(); // s_sum is reused by the next block_reduce
            }
        }
    }
}

// [#46] ADD -> RMS_NORM -> MUL(gamma) in one pass: sum = a + b (kept, it is the residual), dst = rms_norm(sum) * gamma.
// The arithmetic is rms_norm_f32<block_size, true>'s, in the same order, so both outputs are bit-identical to the
// unfused ADD + fused RMS_NORM/MUL. With q8_mode != 0 the pass also writes dst as q8_1 rows of ncols_padded values,
// exactly as quantize_q8_1 would (q8_mode 2: the IQ4_XS-swizzled layout), which prefills the MMVQ shared-quantize cache
// for the consumer. Every warp owns whole q8_1 blocks: col = tid + k*block_size and block_size % QK8_1 == 0.
// sum may alias a or b exactly (each element is read before its own thread writes it), and dst may alias a or b
// exactly (a row is written only after the block's reduction barrier, and only this block reads that row).
template <int block_size, int q8_mode>
static __global__ void add_rms_norm_mul_f32(const float * a,
                                            const float * b,
                                            float *       sum,
                                            const float * gamma,
                                            float *       dst,
                                            block_q8_1 *  q8,
                                            const int     ncols,
                                            const int     ncols_padded,
                                            const float   eps) {
    ggml_cuda_pdl_lc();
    const int64_t row = ((int64_t) blockIdx.z*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x;
    const int     tid = threadIdx.x;

    static_assert(block_size % QK8_1 == 0, "q8_1 blocks must not straddle warps");

    a   += row*ncols;
    b   += row*ncols;
    sum += row*ncols;
    dst += row*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a[col] + b[col];
        sum[col] = xi;
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if constexpr (q8_mode == 0) {
        for (int col = tid; col < ncols; col += block_size) {
            dst[col] = scale * sum[col] * gamma[col];
        }
    } else {
        const int    lane = tid % QK8_1;
        block_q8_1 * y    = q8 + row*(ncols_padded/QK8_1);
        // the loop condition is warp-uniform (ncols_padded % QK8_1 == 0): the shuffles below see full warps
        for (int col = tid; col - lane < ncols_padded; col += block_size) {
            float xi = 0.0f;
            if (col < ncols) {
                xi = scale * sum[col] * gamma[col];
                dst[col] = xi;
            }
            float amax = fabsf(xi);
            amax = warp_reduce_max<QK8_1>(amax);

            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

            const int ib = col / QK8_1;
            if constexpr (q8_mode == 2) {
                y[ib].qs[2*(lane % 16) + lane/16] = q;
            } else {
                y[ib].qs[lane] = q;
            }
            const float qsum = warp_reduce_sum<QK8_1>((float) q);

            if (lane == 0) {
                y[ib].ds = make_half2(d, d * qsum);
            }
        }
    }
}

// fork: keep the RMS reduction order and emit the swizzled MMQ D4 activation beside the float output.
template <int block_size>
static __global__ void rms_norm_mul_mmq_f32(const float * x, const float * gamma, float * dst,
                                            block_q8_1_mmq * q8, const int ncols, const int ncols_padded,
                                            const float eps) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % QK8_1;
    x   += (int64_t) row*ncols;
    dst += (int64_t) row*ncols;

    float tmp = 0.0f;
    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col - lane < ncols_padded; col += block_size) {
        float xi = 0.0f;
        if (col < ncols) {
            xi = scale * x[col] * gamma[col];
            dst[col] = xi;
        }
        const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
        const float d_inv = 127.0f / amax;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi*d_inv);
        const int64_t ib = (int64_t) (col/QK8_1_MMQ)*gridDim.x + row;
        const int is = col % QK8_1_MMQ / QK8_1;
        q8[ib].qs[is*QK8_1 + 2*(lane % 16) + lane/16] = q;
        if (lane == 0) {
            q8[ib].d4[is] = 1.0f / d_inv;
        }
    }
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

template <int block_size>
static __global__ void l2_norm_f32(
        const float * x, float * dst, const int ncols, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps) {
    const int nrows = gridDim.x;
    const int row   = blockIdx.x;
    const int tid   = threadIdx.x;

    extern __shared__ float s_sum[];

    ggml_cuda_pdl_sync();

    // grid.y and grid.z are clamped to the CUDA limit, iterate over the excess channels/samples
    for (int sample = blockIdx.z; sample < nsamples; sample += gridDim.z) {
        for (int channel = blockIdx.y; channel < nchannels; channel += gridDim.y) {
            const float * xc   = x   + sample*stride_sample + channel*stride_channel + row*stride_row;
            float       * dstc = dst + ((sample*nchannels + channel)*nrows + row)*ncols;

            float tmp = 0.0f; // partial sum for thread in warp

            for (int col = tid; col < ncols; col += block_size) {
                const float xi = xc[col];
                tmp += xi * xi;
            }

            // sum up partial sums
            tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

            // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));

            for (int col = tid; col < ncols; col += block_size) {
                dstc[col] = scale * xc[col];
            }

            if constexpr (block_size > WARP_SIZE) {
                // sync is needed as we reuse s_sum across block_reduce invocations, see #26385
                __syncthreads();
            }
        }
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

// [NORMROWS] RMS norm of rows with at most 256 columns: one warp per row, RMS_ROWS_PER_BLOCK rows per block.
// rms_norm_f32<256, ...> runs one 256-thread block per row there; with 128-column rows half the threads idle and the
// block scheduler, not memory bandwidth, sets the speed (the GDN head norms of Flash-Next prompts).
// Bit-identical to rms_norm_f32<256, ...>: lane l keeps the values that threads l, 32+l, ..., 224+l of the 256-thread
// block hold (one column each at most), reduces them per virtual warp with the same butterfly, and combines the 8
// virtual-warp sums with the butterfly block_reduce runs over lanes 0..7 (other lanes 0). mean, scale and the
// epilogues are the same expressions; MUL_GATE multiplies the MUL result by the sigmoid of op_sigmoid / the chain.
// GGML_CUDA_NO_NORM_ROWS=1 restores the 256-thread kernel; GGML_CUDA_NORM_ROWS_CHECK=1 runs both and traps on any bit
// difference.
enum rms_rows_mode { RMS_ROWS_PLAIN = 0, RMS_ROWS_SCALE = 1, RMS_ROWS_MUL = 2, RMS_ROWS_MUL_GATE = 3 };
#define RMS_ROWS_PER_BLOCK 8
#define RMS_ROWS_MAX_COLS  256

template <int mode>
static __global__ void __launch_bounds__(WARP_SIZE*RMS_ROWS_PER_BLOCK) rms_norm_rows_f32(
        const float * x, float * dst, const int ncols, const uint32_t nrows_total, const uint3 nrows_packed,
        const uint3 nchannels_packed, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const float eps, const float scale_out, const float * mul, const float * gate, const int64_t gate_stride_row,
        const int64_t gate_stride_channel, const int64_t gate_stride_sample) {
    constexpr int nw = RMS_ROWS_MAX_COLS/WARP_SIZE;
    ggml_cuda_pdl_lc();
    const uint32_t r    = blockIdx.x*RMS_ROWS_PER_BLOCK + threadIdx.y;
    const int      lane = threadIdx.x;
    if (r >= nrows_total) {
        return;
    }
    const uint2   rc      = fast_div_modulo(r, nrows_packed);      // <r / nrows, row>
    const uint2   cs      = fast_div_modulo(rc.x, nchannels_packed); // <sample, channel>
    const int64_t row     = rc.y;
    const int64_t channel = cs.y;
    const int64_t sample  = cs.x;
    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += (int64_t) r*ncols;
    if constexpr (mode == RMS_ROWS_MUL_GATE) {
        gate += sample*gate_stride_sample + channel*gate_stride_channel + row*gate_stride_row;
    }

    ggml_cuda_pdl_sync();
    float xv[nw];
    float part[nw];
#pragma unroll
    for (int w = 0; w < nw; ++w) {
        const int col = w*WARP_SIZE + lane;
        xv[w]   = 0.0f;
        part[w] = 0.0f;
        if (w*WARP_SIZE < ncols) { // warp-uniform
            float tmp = 0.0f;
            if (col < ncols) {
                xv[w] = x[col];
                tmp += xv[w] * xv[w];
            }
            part[w] = warp_reduce_sum(tmp);
        }
    }
    float tmp = 0.0f;
#pragma unroll
    for (int w = 0; w < nw; ++w) {
        if (lane == w) {
            tmp = part[w];
        }
    }
    tmp = warp_reduce_sum(tmp);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

#pragma unroll
    for (int w = 0; w < nw; ++w) {
        const int col = w*WARP_SIZE + lane;
        if (w*WARP_SIZE < ncols && col < ncols) {
            if constexpr (mode == RMS_ROWS_MUL_GATE) {
                const float g = gate[col];
                const float n = scale * xv[w] * mul[col];
                dst[col] = n * (1.0f / (1.0f + expf(-g)));
            } else if constexpr (mode == RMS_ROWS_MUL) {
                dst[col] = scale * xv[w] * mul[col];
            } else if constexpr (mode == RMS_ROWS_SCALE) {
                dst[col] = scale_out * (scale * xv[w]);
            } else {
                dst[col] = scale * xv[w];
            }
        }
    }
}

static __global__ void rms_norm_rows_check(const float * actual, const float * reference, int64_t n) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(actual[i]) != __float_as_uint(reference[i])) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        asm volatile("trap;");
#endif
    }
}

static bool rms_norm_rows_applies(const int ncols, const int64_t nrows_total) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    GGML_UNUSED(ncols); GGML_UNUSED(nrows_total);
    return false; // the emulated reduction assumes 32-wide warps
#else
    static const bool off = [] { const char * e = getenv("GGML_CUDA_NO_NORM_ROWS"); return e && atoi(e) != 0; }();
    return !off && ncols >= 1 && ncols <= RMS_ROWS_MAX_COLS && nrows_total >= 1 && nrows_total <= UINT32_MAX &&
        (nrows_total + RMS_ROWS_PER_BLOCK - 1)/RMS_ROWS_PER_BLOCK <= INT_MAX;
#endif
}

template <int mode>
static void rms_norm_rows_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        cudaStream_t stream, const float scale_out = 1.0f, const float * mul = nullptr, const float * gate = nullptr,
        const int64_t gate_stride_row = 0, const int64_t gate_stride_channel = 0, const int64_t gate_stride_sample = 0) {
    const int64_t nrows_total = (int64_t) nrows*nchannels*nsamples;
    const dim3 blocks_num((nrows_total + RMS_ROWS_PER_BLOCK - 1)/RMS_ROWS_PER_BLOCK, 1, 1);
    const dim3 block_dims(WARP_SIZE, RMS_ROWS_PER_BLOCK, 1);
    const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, 0, stream};
    ggml_cuda_kernel_launch(rms_norm_rows_f32<mode>, launch_params, x, dst, ncols, (uint32_t) nrows_total,
        init_fastdiv_values(nrows), init_fastdiv_values(nchannels), stride_row, stride_channel, stride_sample, eps,
        scale_out, mul, gate, gate_stride_row, gate_stride_channel, gate_stride_sample);
}

// launches the rows kernel instead of legacy() when it applies; with GGML_CUDA_NORM_ROWS_CHECK=1 runs both and compares
template <int mode, typename F>
static void rms_norm_rows_or(F && legacy, const float * x, float * dst, const int ncols, const int nrows,
        const int nchannels, const int nsamples, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, cudaStream_t stream, const float scale_out = 1.0f,
        const float * mul = nullptr) {
    const int64_t nrows_total = (int64_t) nrows*nchannels*nsamples;
    if (!rms_norm_rows_applies(ncols, nrows_total)) {
        legacy();
        return;
    }
    static const bool check = [] { const char * e = getenv("GGML_CUDA_NORM_ROWS_CHECK"); return e && atoi(e) != 0; }();
    if (!check) {
        rms_norm_rows_f32_cuda<mode>(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel,
            stride_sample, eps, stream, scale_out, mul);
        return;
    }
    // the rows kernel first: legacy() may write over x in place
    const int64_t n = nrows_total*ncols;
    float * ref = nullptr;
    CUDA_CHECK(cudaMallocAsync((void **) &ref, n*sizeof(float), stream));
    rms_norm_rows_f32_cuda<mode>(x, ref, ncols, nrows, nchannels, nsamples, stride_row, stride_channel,
        stride_sample, eps, stream, scale_out, mul);
    legacy();
    rms_norm_rows_check<<<(n + 255)/256, 256, 0, stream>>>(dst, ref, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFreeAsync(ref, stream));
}

#define RMS_NORM_LAUNCH(BLOCK_SIZE, VPT, VEC) \
    do { \
        const dim3 block_dims(BLOCK_SIZE, 1, 1); \
        const ggml_cuda_kernel_launch_params launch_params{ blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float) : 0, stream }; \
        ggml_cuda_kernel_launch(rms_norm_f32<BLOCK_SIZE, VPT, VEC, do_multiply, do_add, do_scale>, launch_params, \
            x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, \
            mul, mul_stride_row, mul_stride_channel, mul_stride_sample, \
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, \
            add, add_stride_row, add_stride_channel, add_stride_sample, \
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, scale_out); \
    } while (0)

template <bool do_multiply, bool do_add, bool do_scale>
static void rms_norm_f32_dispatch(
        const dim3     blocks_num,
        const ggml_tensor * x_t,   float * dst,   const int ncols, const int nchannels, const int nsamples,
        const int64_t  stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps,
        const ggml_tensor * mul_t, const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const uint3    mul_ncols_packed, const uint3 mul_nrows_packed, const uint3 mul_nchannels_packed, const uint3 mul_nsamples_packed,
        const ggml_tensor * add_t, const int64_t add_stride_row, const int64_t add_stride_channel, const int64_t add_stride_sample,
        const uint3    add_ncols_packed, const uint3 add_nrows_packed, const uint3 add_nchannels_packed, const uint3 add_nsamples_packed,
        cudaStream_t   stream, const float scale_out = 1.0f) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    static_assert(!do_scale || !do_multiply, "fusing scale is not supported with multiplying");

    const float * x   = (const float *) x_t->data;
    const float * mul = mul_t ? (const float *) mul_t->data : nullptr;
    const float * add = add_t ? (const float *) add_t->data : nullptr;

    // [NORMROWS] rows of at most RMS_ROWS_MAX_COLS columns (plain, scaled, or times a one-row weight vector): one warp
    // per row. legacy is the scalar 256-thread rms_norm_f32 instance the rows kernel is bit-identical to.
    if constexpr (!do_add) {
        bool rows_ok = true;
        if constexpr (do_multiply) {
            rows_ok = mul_ncols_packed.z == (uint32_t) ncols && mul_nrows_packed.z == 1 &&
                mul_nchannels_packed.z == 1 && mul_nsamples_packed.z == 1;
        }
        if (rows_ok && rms_norm_rows_applies(ncols, (int64_t) blocks_num.x*nchannels*nsamples)) {
            const auto legacy = [&] { RMS_NORM_LAUNCH(256, 1, false); };
            rms_norm_rows_or<do_multiply ? RMS_ROWS_MUL : (do_scale ? RMS_ROWS_SCALE : RMS_ROWS_PLAIN)>(legacy, x, dst,
                ncols, (int) blocks_num.x, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream,
                scale_out, mul);
            return;
        }
    }

    // fork: upstream's float4 path is opt-in (GGML_CUDA_RMS_NORM_VEC=1). Its per-thread summation order differs, and
    // our fused norm kernels ([CAST], [#46], [HCPN], rms_norm_mul_mmq) are bit-identical to the scalar 256/1024-thread
    // kernels, which stay the default (the release gate is bit-identical to v0.6.3).
    static const bool no_vec = [] { const char * e = getenv("GGML_CUDA_RMS_NORM_VEC"); return !(e && atoi(e) != 0); }();
    // dst is written at row * ncols, i.e. always at a multiple of ncols, so only its base needs checking here
    bool vec = !no_vec && ncols % 4 == 0 && ncols > 0 && ((uintptr_t) dst & 15) == 0;
    vec = vec && ggml_cuda_is_aligned(x_t, 16);
    if constexpr (do_multiply) {
        vec = vec && mul_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(mul_t, 16);
    }
    if constexpr (do_add) {
        vec = vec && add_ncols_packed.z == (uint32_t) ncols
            && ggml_cuda_is_aligned(add_t, 16);
    }

    const int n4 = ncols / 4;
    if (vec) {
        if (n4 <= 32) {
            RMS_NORM_LAUNCH(32,   1, true);
        } else if (n4 <= 128) {
            RMS_NORM_LAUNCH(128,  1, true);
        } else if (n4 <= 256) {
            RMS_NORM_LAUNCH(256,  1, true);
        } else if (n4 <= 512) {
            RMS_NORM_LAUNCH(512,  1, true);
        } else if (n4 <= 1024) {
            RMS_NORM_LAUNCH(1024, 1, true);
        } else if (n4 <= 4096) {
            RMS_NORM_LAUNCH(1024, 4, true);
        } else {
            RMS_NORM_LAUNCH(1024, 1, false);
        }
    } else if (ncols < 1024) {
        RMS_NORM_LAUNCH(256,  1, false);
    } else {
        RMS_NORM_LAUNCH(1024, 1, false);
    }
}
#undef RMS_NORM_LAUNCH

template <bool do_scale = false>
static void rms_norm_f32_cuda(
        const ggml_tensor * x_t, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream,
        const float scale_out = 1.0f) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    rms_norm_f32_dispatch</*do_multiply*/false, /*do_add*/false, do_scale>(
        blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        stream, scale_out);
}

static void rms_norm_mul_f32_cuda(const ggml_tensor * x_t,
                                  const ggml_tensor * mul_t,
                                  const ggml_tensor * add_t,
                                  float             * dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (mul_t == nullptr) {
        rms_norm_f32_cuda(x_t, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
    const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
    const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
    const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
    const uint3 zero_packed          = make_uint3(0, 0, 0);

    if (add_t == nullptr) {
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/false, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed, stream);
    } else {
        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        rms_norm_f32_dispatch</*do_multiply*/true, /*do_add*/true, /*do_scale*/false>(
            blocks_num, x_t, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps,
            mul_t, mul_stride_row, mul_stride_channel, mul_stride_sample,
            mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
            add_t, add_stride_row, add_stride_channel, add_stride_sample,
            add_ncols_packed, add_nrows_packed, add_nchannels_packed, add_nsamples_packed, stream);
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

static void l2_norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, MIN(nchannels, UINT16_MAX), MIN(nsamples, UINT16_MAX));
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024>, launch_params, x, dst, ncols, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor) {
    const ggml_tensor * src0   = dst->src[0];
    float *             dst_d  = (float *) scale_tensor->data;
    cudaStream_t        stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    float scale;
    memcpy(&scale, (const float *) scale_tensor->op_params + 0, sizeof(float));

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    rms_norm_f32_cuda<true>(src0, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream, scale);
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor, float * dst_override,
                                 nv_bfloat16 * dst_bf16) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float * dst_d = dst_override ? dst_override : (float *) mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    if (dst_bf16 != nullptr) {
        // [CAST] the launch of rms_norm_mul_f32_cuda's mul-only branch, with the BF16 copy added
        const float * src0_d = (const float *) rms_norm_src->data;
        const float * mul_d  = (const float *) mul_src->data;
        const dim3  blocks_num(ne01, MIN(ne02, UINT16_MAX), MIN(ne03, UINT16_MAX));
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (ne00 < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_mul_cast_bf16_f32<256>, launch_params, src0_d, dst_d, dst_bf16, (int) ne00, (int) ne02, (int) ne03, s01, s02, s03, eps,
                mul_d, mul_s01, mul_s02, mul_s03, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_mul_cast_bf16_f32<1024>, launch_params, src0_d, dst_d, dst_bf16, (int) ne00, (int) ne02, (int) ne03, s01, s02, s03, eps,
                mul_d, mul_s01, mul_s02, mul_s03, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed);
        }
        return;
    }

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src, nullptr, dst_d,
                          ne00, ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ 0, 0, 0,
                          0, 0, 0, 0,
                          eps, stream);
}

void ggml_cuda_op_rms_norm_mmq_fused(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul,
                                     void * q8_dst, int64_t ncols_padded) {
    const ggml_tensor * gamma = mul->src[0] == norm ? mul->src[1] : mul->src[0];
    float eps;
    memcpy(&eps, norm->op_params, sizeof(eps));
    GGML_ASSERT(eps >= 0.0f);
    const int ncols = mul->ne[0];
    const dim3 blocks_num(mul->ne[1], 1, 1);
    const dim3 block_dims(ncols < 1024 ? 256 : 1024, 1, 1);
    const ggml_cuda_kernel_launch_params params = { blocks_num, block_dims, 32*sizeof(float), ctx.stream() };
    if (ncols < 1024) {
        ggml_cuda_kernel_launch(rms_norm_mul_mmq_f32<256>, params, (const float *) norm->src[0]->data,
            (const float *) gamma->data, (float *) mul->data, (block_q8_1_mmq *) q8_dst, ncols, (int) ncols_padded, eps);
    } else {
        ggml_cuda_kernel_launch(rms_norm_mul_mmq_f32<1024>, params, (const float *) norm->src[0]->data,
            (const float *) gamma->data, (float *) mul->data, (block_q8_1_mmq *) q8_dst, ncols, (int) ncols_padded, eps);
    }
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    float *      dst_d  = (float *) add_tensor->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    rms_norm_mul_f32_cuda(rms_norm_src, mul_src,add_src,dst_d,
                          ne00,ne01, ne02, ne03,
                          /*s00*/ s01, s02, s03,
                          /*mul_s00*/ mul_s01, mul_s02, mul_s03,
                          mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
                          /*add_s00*/ add_s01, add_s02, add_s03,
                          add_ncols, add_nrows, add_nchannels, add_nsamples,
                          eps, stream);
}

template <int block_size, int q8_mode>
static void add_rms_norm_mul_f32_launch(const float * a, const float * b, float * sum, const float * gamma, float * dst,
                                        block_q8_1 * q8, const int ncols, const int ncols_padded, const dim3 blocks_num,
                                        const float eps, cudaStream_t stream) {
    const dim3 block_dims(block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_size > WARP_SIZE ? 32 * sizeof(float) : 0, stream};
    ggml_cuda_kernel_launch(add_rms_norm_mul_f32<block_size, q8_mode>, launch_params,
                            a, b, sum, gamma, dst, q8, ncols, ncols_padded, eps);
}

template <int q8_mode>
static void add_rms_norm_mul_f32_cuda(const float * a, const float * b, float * sum, const float * gamma, float * dst,
                                      block_q8_1 * q8, const int ncols, const int ncols_padded, const dim3 blocks_num,
                                      const float eps, cudaStream_t stream) {
    // same block size choice as rms_norm_f32_cuda: the reduction order, and so the result, depends on it
    if (ncols < 1024) {
        add_rms_norm_mul_f32_launch<256, q8_mode>(a, b, sum, gamma, dst, q8, ncols, ncols_padded, blocks_num, eps, stream);
    } else {
        add_rms_norm_mul_f32_launch<1024, q8_mode>(a, b, sum, gamma, dst, q8, ncols, ncols_padded, blocks_num, eps, stream);
    }
}

void ggml_cuda_op_add_rms_norm_mul(ggml_backend_cuda_context & ctx,
                                   ggml_tensor *               add,
                                   ggml_tensor *               rms_norm,
                                   ggml_tensor *               mul,
                                   void *                      q8_dst,
                                   ggml_type                   q8_type) {
    const ggml_tensor * gamma = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    GGML_ASSERT(rms_norm->src[0] == add && (mul->src[0] == rms_norm || mul->src[1] == rms_norm));
    GGML_ASSERT(add->type == GGML_TYPE_F32 && add->src[0]->type == GGML_TYPE_F32 && add->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(mul->type == GGML_TYPE_F32 && gamma->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(add->src[0], add) && ggml_are_same_shape(add->src[1], add) && ggml_are_same_shape(mul, add));
    GGML_ASSERT(ggml_is_contiguous(add->src[0]) && ggml_is_contiguous(add->src[1]) && ggml_is_contiguous(add) &&
                ggml_is_contiguous(mul) && ggml_is_contiguous(gamma));
    GGML_ASSERT(gamma->ne[0] == add->ne[0] && ggml_nrows(gamma) == 1);
    GGML_ASSERT(add->ne[0] <= INT_MAX);

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const int  ncols        = (int) add->ne[0];
    const int  ncols_padded = (int) GGML_PAD(add->ne[0], MATRIX_ROW_PADDING);
    // rows are contiguous and indexed linearly, so a 1-D grid gives the same row per block without the 65535 y/z limit
    GGML_ASSERT(ggml_nrows(add) <= INT_MAX);
    const dim3 blocks_num((unsigned int) ggml_nrows(add), 1, 1);

    const float * a     = (const float *) add->src[0]->data;
    const float * b     = (const float *) add->src[1]->data;
    float *       sum   = (float *) add->data;
    const float * g     = (const float *) gamma->data;
    float *       dst   = (float *) mul->data;
    block_q8_1 *  q8    = (block_q8_1 *) q8_dst;
    cudaStream_t stream = ctx.stream();

    if (q8 == nullptr) {
        add_rms_norm_mul_f32_cuda<0>(a, b, sum, g, dst, nullptr, ncols, ncols_padded, blocks_num, eps, stream);
    } else if (q8_type == GGML_TYPE_IQ4_XS) {
        add_rms_norm_mul_f32_cuda<2>(a, b, sum, g, dst, q8, ncols, ncols_padded, blocks_num, eps, stream);
    } else {
        add_rms_norm_mul_f32_cuda<1>(a, b, sum, g, dst, q8, ncols, ncols_padded, blocks_num, eps, stream);
    }
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    l2_norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

// [GATEDNORM] see norm.cuh
bool ggml_cuda_rms_norm_mul_gate_supported(const ggml_tensor * norm) {
    return rms_norm_rows_applies((int) norm->ne[0], ggml_nrows(norm));
}

void ggml_cuda_op_rms_norm_mul_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * norm, const ggml_tensor * weight,
                                    const ggml_tensor * gate, float * dst) {
    const ggml_tensor * x = norm->src[0];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 && gate->type == GGML_TYPE_F32);
    GGML_ASSERT(x->nb[0] == sizeof(float) && gate->nb[0] == sizeof(float) && ggml_is_contiguous(weight));
    GGML_ASSERT(weight->ne[0] == x->ne[0] && ggml_nrows(weight) == 1 && ggml_are_same_shape(x, gate));
    GGML_ASSERT(ggml_cuda_rms_norm_mul_gate_supported(norm));
    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    const size_t ts = sizeof(float);
    rms_norm_rows_f32_cuda<RMS_ROWS_MUL_GATE>((const float *) x->data, dst, (int) x->ne[0], (int) x->ne[1],
        (int) x->ne[2], (int) x->ne[3], x->nb[1]/ts, x->nb[2]/ts, x->nb[3]/ts, eps, ctx.stream(), 1.0f,
        (const float *) weight->data, (const float *) gate->data, gate->nb[1]/ts, gate->nb[2]/ts, gate->nb[3]/ts);
}
