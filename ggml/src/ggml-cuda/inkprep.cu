// INKGLUE PREP (see inkprep.cuh). The graph matcher is ggml_cuda_try_fuse_ink_prep in ggml-cuda.cu.
#include "inkprep.cuh"
#include "convert.cuh"

#define INK_PREP_ROWS_MAX_COLS 256 // rms_norm_rows_f32's RMS_ROWS_MAX_COLS

static __global__ void k_ink_prep(const ink_prep_args a) {
    const int warp = blockIdx.x*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    int j = 0;
    while (j + 1 < a.n && warp >= a.job[j + 1].row0) {
        ++j;
    }
    const ink_prep_job & J = a.job[j];
    const int r = warp - J.row0;
    if (r < 0 || r >= J.nrows*J.nch) {
        return;
    }
    const int row = r % J.nrows;
    const int ch  = r / J.nrows;
    const int ncols = J.ncols;
    const float * xr = J.x + ch*J.sx2 + row*J.sx1;
    float * dr = J.dst + (int64_t) r*ncols;

    if (J.kind == 0) {
        // k_ink_clip: [x*m ->] sum of squares over col = lane + k*32, warp_reduce_sum, rsqrtf(fmaxf(sum, eps*eps)), s*v + b
        const float * mr = J.m ? J.m + (int64_t) row*ncols : nullptr;
        float tmp = 0.0f;
        for (int col = lane; col < ncols; col += WARP_SIZE) {
            const float xi = mr ? xr[col]*mr[col] : xr[col];
            tmp += xi * xi;
        }
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(fmaxf(tmp, J.eps * J.eps));
        const float t = J.tau ? J.tau[ch] : 0.0f;
        float * d2 = J.tau ? J.dst2 + (int64_t) r*ncols : nullptr;
        for (int col = lane; col < ncols; col += WARP_SIZE) {
            const float xi = mr ? xr[col]*mr[col] : xr[col];
            const float v  = scale * xi;
            const float o  = J.s*v + J.b;
            dr[col] = o;
            float out = o;
            if (d2) {
                out = o * t; // binbcast's op_mul (the tau MUL)
                d2[col] = out;
            }
            if (J.cache) {
                char * cr = J.cache + J.idx[ch]*J.cache_nb1;
                if (J.cache_f16) {
                    ((half *) cr)[row*ncols + col] = ggml_cuda_cast<half>(out);
                } else {
                    ((float *) cr)[row*ncols + col] = out;
                }
            }
        }
        return;
    }
    // rms_norm_rows_f32 PLAIN: per 32-column slice a warp sum, then the slices' sums gathered on lanes and summed again
    constexpr int nw = INK_PREP_ROWS_MAX_COLS/WARP_SIZE;
    float xv[nw];
    float part[nw];
#pragma unroll
    for (int w = 0; w < nw; ++w) {
        const int col = w*WARP_SIZE + lane;
        xv[w]   = 0.0f;
        part[w] = 0.0f;
        if (w*WARP_SIZE < ncols) {
            float tmp = 0.0f;
            if (col < ncols) {
                xv[w] = xr[col];
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
    const float scale = rsqrtf(mean + J.eps);
#pragma unroll
    for (int w = 0; w < nw; ++w) {
        const int col = w*WARP_SIZE + lane;
        if (w*WARP_SIZE < ncols && col < ncols) {
            const float o = scale * xv[w];
            dr[col] = o;
            if (J.cache) {
                char * cr = J.cache + J.idx[ch]*J.cache_nb1;
                if (J.cache_f16) {
                    ((half *) cr)[row*ncols + col] = ggml_cuda_cast<half>(o);
                } else {
                    ((float *) cr)[row*ncols + col] = o;
                }
            }
        }
    }
}

void ggml_cuda_ink_prep(ggml_backend_cuda_context & ctx, const ink_prep_args & a, int total_rows) {
    constexpr int warps = 4;
    k_ink_prep<<<(total_rows + warps - 1)/warps, warps*WARP_SIZE, 0, ctx.stream()>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
