#include "reduce_rows.cuh"
#include "sumrows.cuh"

// fork: rows of at most 32 columns, one thread per row. The 32-thread block per row below leaves most lanes idle and is
// bound by block scheduling there (147,456 rows of 4: ~100 us on a 3090). Adds in the order of that block's
// one-column-per-lane butterfly (lane value 0 + x, then xor offsets 16..1), so the sums are bit-identical.
// Opt-in (port, default off): GGML_CUDA_SUM_ROWS_SHORT=1; unset or 0 = the block per row.
#define SUM_ROWS_SHORT_MAX 32

static __global__ void sum_rows_f32_short(const float * __restrict__ x, float * __restrict__ dst, const int ncols,
        const int nrows) {
    ggml_cuda_pdl_sync();
    const int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    const float * xr = x + (int64_t) row*ncols;
    float s[SUM_ROWS_SHORT_MAX];
#pragma unroll
    for (int l = 0; l < SUM_ROWS_SHORT_MAX; ++l) {
        s[l] = l < ncols ? 0.0f + xr[l] : 0.0f;
    }
#pragma unroll
    for (int off = SUM_ROWS_SHORT_MAX/2; off > 0; off >>= 1) {
#pragma unroll
        for (int l = 0; l < off; ++l) {
            s[l] = s[l] + s[l + off];
        }
    }
    dst[row] = s[0];
}

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    static const bool short_on = [] { const char * e = getenv("GGML_CUDA_SUM_ROWS_SHORT"); return e && atoi(e) != 0; }();
    if (short_on && ncols <= SUM_ROWS_SHORT_MAX && (nrows / nsm) >= 2) {
        // same condition as the 32-thread branch below, whose order the kernel reproduces
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
                dim3((nrows + 255) / 256, 1, 1), dim3(256, 1, 1), 0, stream);
        ggml_cuda_kernel_launch(sum_rows_f32_short, launch_params, x, dst, ncols, nrows);
        return;
    }
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
        ggml_cuda_kernel_launch(reduce_rows_f32</*norm=*/false>, launch_params, x, dst, ncols);
    }
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_rows(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    if (ggml_is_contiguous(src0)) {
        sum_rows_f32_cuda(src0_d, dst_d, ncols, nrows, stream);
        return;
    }

    const dim3 block_nums(nrows, 1, 1);

    const int id  = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[id].nsm;
    dim3 block_dims;
    if ((nrows / nsm) < 2) {
        // Increase num threads to 512 for small nrows to better hide the latency
        block_dims = dim3(512, 1, 1);
    } else {
        // Enough active SMs to hide latency, use smaller blocks to allow better scheduling
        block_dims = dim3(ncols < 1024 ? 32 : 128, 1, 1);
    }
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const char * src0_d_bytes = (const char *) src0->data;
    ggml_cuda_kernel_launch(reduce_rows_f32_strided</*norm=*/false>, launch_params, src0_d_bytes, dst_d, ncols,
            src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3]);
}
