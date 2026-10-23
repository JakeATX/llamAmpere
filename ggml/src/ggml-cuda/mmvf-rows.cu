// [FROWS] float-weight row kernel for narrow weights at 2..8 columns (see mmvf-rows.cuh).
#include "mmvf-rows.cuh"

#include <cstdlib>

int64_t ggml_cuda_frows_count = 0;

namespace {

// 16-byte weight vector of each type and its exact widening to f32
template <typename T> struct frows_io;

template <> struct frows_io<float> {
    static constexpr int V = 4;
    static __device__ __forceinline__ void unpack(const uint4 & u, float (&f)[V]) {
        f[0] = __uint_as_float(u.x); f[1] = __uint_as_float(u.y);
        f[2] = __uint_as_float(u.z); f[3] = __uint_as_float(u.w);
    }
};

template <> struct frows_io<nv_bfloat16> {
    static constexpr int V = 8;
    static __device__ __forceinline__ void unpack(const uint4 & u, float (&f)[V]) {
        const uint32_t w[4] = { u.x, u.y, u.z, u.w };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            f[2*i + 0] = __uint_as_float(w[i] << 16);         // low half = element 2i
            f[2*i + 1] = __uint_as_float(w[i] & 0xFFFF0000u); // high half = element 2i+1
        }
    }
};

template <> struct frows_io<half> {
    static constexpr int V = 8;
    static __device__ __forceinline__ void unpack(const uint4 & u, float (&f)[V]) {
        const uint32_t w[4] = { u.x, u.y, u.z, u.w };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            half2 h;
            memcpy(&h, &w[i], sizeof(h));
            const float2 t = __half22float2(h);
            f[2*i + 0] = t.x;
            f[2*i + 1] = t.y;
        }
    }
};

// x: nrows rows of nchunks*V weights, row stride stride_row elements; y: ncols columns of nchunks*V floats, column
// stride stride_col_y floats; dst: column stride stride_col_dst floats. One block = R consecutive rows.
template <typename T, int ncols, int R, int NT>
static __global__ void __launch_bounds__(NT) frows_kernel(
        const T * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
        const int nchunks, const int nrows, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst) {
    using IO = frows_io<T>;
    constexpr int V  = IO::V;
    constexpr int NW = NT / WARP_SIZE;

    ggml_cuda_pdl_sync();

    const int row0 = blockIdx.x * R;
    const int tid  = threadIdx.x;

    // rows past the end read the last row (in bounds) and are never written
    const uint4 * xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = min(row0 + r, nrows - 1);
        xr[r] = (const uint4 *) (x + int64_t(row)*stride_row);
    }
    const float4 * y4 = (const float4 *) y;
    const int64_t stride_col_y4 = stride_col_y / 4;

    float acc[R][ncols];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            acc[r][j] = 0.0f;
        }
    }

#pragma unroll 2
    for (int c = tid; c < nchunks; c += NT) {
        // every load of this K step first ...
        uint4 w[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            w[r] = __ldg(xr[r] + c);
        }
        float4 yv[ncols][V/4];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int q = 0; q < V/4; ++q) {
                yv[j][q] = __ldg(y4 + j*stride_col_y4 + int64_t(c)*(V/4) + q);
            }
        }
        // ... then the multiply-adds, element order 0..V-1
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float wf[V];
            IO::unpack(w[r], wf);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int q = 0; q < V/4; ++q) {
                    acc[r][j] = fmaf(wf[4*q + 0], yv[j][q].x, acc[r][j]);
                    acc[r][j] = fmaf(wf[4*q + 1], yv[j][q].y, acc[r][j]);
                    acc[r][j] = fmaf(wf[4*q + 2], yv[j][q].z, acc[r][j]);
                    acc[r][j] = fmaf(wf[4*q + 3], yv[j][q].w, acc[r][j]);
                }
            }
        }
    }

    // butterfly inside each warp (fixed order, every lane ends with the same total)
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
                acc[r][j] += __shfl_xor_sync(0xFFFFFFFF, acc[r][j], off, WARP_SIZE);
            }
        }
    }

    __shared__ float part[NW][R*ncols];
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                part[warp][r*ncols + j] = acc[r][j];
            }
        }
    }
    __syncthreads();

    if (tid < R*ncols) {
        const int r = tid / ncols;
        const int j = tid % ncols;
        float sum = part[0][tid];
#pragma unroll
        for (int k = 1; k < NW; ++k) {
            sum += part[k][tid];
        }
        if (row0 + r < nrows) {
            dst[j*stride_col_dst + row0 + r] = sum;
        }
    }
}

struct frows_cfg {
    bool    on       = false;
    int64_t max_rows = 1024;
    int     rows     = 0; // 0 = by shape
    int     nt       = 0; // 0 = 128
    int     min_n    = 2;
    int     max_n    = 8;
};

static const frows_cfg & frows_get_cfg() {
    static const frows_cfg cfg = [] {
        frows_cfg c;
        const char * on = getenv("GGML_CUDA_FROWS");
        c.on = on != nullptr && atoi(on) != 0;
        if (const char * m = getenv("GGML_CUDA_FROWS_MAX_ROWS")) {
            c.max_rows = atoll(m);
        }
        if (const char * r = getenv("GGML_CUDA_FROWS_R")) {
            const int v = atoi(r);
            c.rows = (v == 1 || v == 2 || v == 4 || v == 8) ? v : 0;
        }
        if (const char * t = getenv("GGML_CUDA_FROWS_NT")) {
            const int v = atoi(t);
            c.nt = (v == 128 || v == 256) ? v : 0;
        }
        if (const char * m = getenv("GGML_CUDA_FROWS_MIN_N")) {
            const int v = atoi(m);
            c.min_n = v < 2 ? 2 : (v > 8 ? 8 : v);
        }
        if (const char * m = getenv("GGML_CUDA_FROWS_MAX_N")) {
            const int v = atoi(m);
            c.max_n = v < 2 ? 2 : (v > 8 ? 8 : v);
        }
        if (c.on) {
            GGML_LOG_INFO("frows: on, weights up to %lld rows, widths %d..%d, rows per block %d (0 = by shape), threads %d (0 = 128)\n",
                (long long) c.max_rows, c.min_n, c.max_n, c.rows, c.nt);
        }
        return c;
    }();
    return cfg;
}

static bool frows_aligned(const void * p, size_t a) {
    return (reinterpret_cast<uintptr_t>(p) % a) == 0;
}

static int frows_pick_rows(const frows_cfg & cfg, int64_t nrows, int nsm) {
    if (cfg.rows > 0) {
        return cfg.rows;
    }
    // most rows per block that still leaves at least one block per SM (more rows = fewer activation re-reads)
    for (int r : { 4, 2 }) {
        if ((nrows + r - 1) / r >= nsm) {
            return r;
        }
    }
    return 1;
}

template <typename T, int ncols, int R, int NT>
static void frows_launch(const T * x, const float * y, float * dst, int nchunks, int nrows, int64_t stride_row,
                         int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    const dim3 grid((nrows + R - 1) / R, 1, 1);
    const dim3 block(NT, 1, 1);
    frows_kernel<T, ncols, R, NT><<<grid, block, 0, stream>>>(x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst);
}

template <typename T, int ncols, int R>
static void frows_switch_nt(int nt, const T * x, const float * y, float * dst, int nchunks, int nrows, int64_t stride_row,
                            int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    if (nt == 256) {
        frows_launch<T, ncols, R, 256>(x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream);
    } else {
        frows_launch<T, ncols, R, 128>(x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream);
    }
}

template <typename T, int ncols>
static void frows_switch_rows(int rows, int nt, const T * x, const float * y, float * dst, int nchunks, int nrows,
                              int64_t stride_row, int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    switch (rows) {
        case 1:  frows_switch_nt<T, ncols, 1>(nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 2:  frows_switch_nt<T, ncols, 2>(nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 4:  frows_switch_nt<T, ncols, 4>(nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        default: frows_switch_nt<T, ncols, 8>(nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
    }
}

template <typename T>
static void frows_switch_ncols(int ncols, int rows, int nt, const T * x, const float * y, float * dst, int nchunks, int nrows,
                               int64_t stride_row, int64_t stride_col_y, int64_t stride_col_dst, cudaStream_t stream) {
    switch (ncols) {
        case 2: frows_switch_rows<T, 2>(rows, nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 3: frows_switch_rows<T, 3>(rows, nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 4: frows_switch_rows<T, 4>(rows, nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 5: frows_switch_rows<T, 5>(rows, nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 6: frows_switch_rows<T, 6>(rows, nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 7: frows_switch_rows<T, 7>(rows, nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        case 8: frows_switch_rows<T, 8>(rows, nt, x, y, dst, nchunks, nrows, stride_row, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("frows: ncols %d", ncols);
    }
}

} // namespace

bool ggml_cuda_frows_wants(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    const frows_cfg & cfg = frows_get_cfg();
    if (!cfg.on || !GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return false;
    }
    const ggml_type type = src0->type;
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t ne11 = src1->ne[1];
    if (ne11 < cfg.min_n || ne11 > cfg.max_n || src0->ne[1] > cfg.max_rows || src0->ne[1] < 1) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0] || dst->ne[0] != src0->ne[1] || dst->ne[1] != ne11) {
        return false;
    }
    const size_t ts = ggml_type_size(type);
    const int    V  = type == GGML_TYPE_F32 ? 4 : 8; // elements per 16-byte load
    if (src0->ne[0] % V != 0 || src0->ne[0] / V > INT32_MAX) {
        return false;
    }
    // 16-byte loads of weights and activations
    if (src0->nb[0] != ts || src0->nb[1] % 16 != 0 || !frows_aligned(src0->data, 16)) {
        return false;
    }
    if (src1->nb[0] != sizeof(float) || src1->nb[1] % 16 != 0 || !frows_aligned(src1->data, 16)) {
        return false;
    }
    if (dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_frows_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const frows_cfg & cfg = frows_get_cfg();
    const int     nsm   = ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t nrows = src0->ne[1];
    const int     ncols = (int) src1->ne[1];
    const int     rows  = frows_pick_rows(cfg, nrows, nsm);
    const int     nt    = cfg.nt > 0 ? cfg.nt : 128;
    const int64_t stride_row     = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t stride_col_y   = src1->nb[1] / sizeof(float);
    const int64_t stride_col_dst = dst->nb[1]  / sizeof(float);
    const float * y = (const float *) src1->data;
    float * d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_F32:
            frows_switch_ncols<float>(ncols, rows, nt, (const float *) src0->data, y, d, (int) (src0->ne[0] / 4), (int) nrows,
                stride_row, stride_col_y, stride_col_dst, stream);
            break;
        case GGML_TYPE_F16:
            frows_switch_ncols<half>(ncols, rows, nt, (const half *) src0->data, y, d, (int) (src0->ne[0] / 8), (int) nrows,
                stride_row, stride_col_y, stride_col_dst, stream);
            break;
        case GGML_TYPE_BF16:
            frows_switch_ncols<nv_bfloat16>(ncols, rows, nt, (const nv_bfloat16 *) src0->data, y, d, (int) (src0->ne[0] / 8),
                (int) nrows, stride_row, stride_col_y, stride_col_dst, stream);
            break;
        default:
            GGML_ABORT("frows: type %s", ggml_type_name(src0->type));
    }
    CUDA_CHECK(cudaGetLastError());
    ++ggml_cuda_frows_count;
}
