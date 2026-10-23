#pragma once

#include "mmvq-kqrows.cuh"
#include "vecdotq.cuh"

// [KQROWS] kernel body, included by the three per-type launch files. See mmvq-kqrows.cuh for the contract.
//
// Thread mapping is the MMVQ kernel's: kbx = tid / (qi/vdr) + i*blocks_per_iter, kqs = vdr * (tid % (qi/vdr)), with
// blocks_per_iter = vdr*nwarps*32/qi. Each per-format trait splits vec_dot_<type>_q8_1 into an activation part (xload,
// once per column per block) and a weight part (wload, once per row per block); dot() is the rest of
// vec_dot_<type>_q8_1_impl, the same float expression in the same order.

template <ggml_type type> struct kqrows_traits;

template <> struct kqrows_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk  = QK_K;
    static constexpr int qi  = QI6_K;
    static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    using block = block_q6_K;
    struct X { int u[QR6_K]; float d8[QR6_K]; };
    struct W { int vi[QR6_K]; int sc[QR6_K]; float d; };

    static __device__ __forceinline__ X xload(const block_q8_1 * __restrict__ bq8_1, const int iqs) {
        X r;
        const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            r.u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
            r.d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
        }
        return r;
    }
    static __device__ __forceinline__ W wload(const block * __restrict__ bq6_K, const int iqs) {
        W r;
        const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
        const int vh_shift     = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
        const int vl = get_int_b2(bq6_K->ql, iqs);
        const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
        const int8_t * scales = bq6_K->scales + scale_offset;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            r.sc[i] = scales[4*i];
            const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
            const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
            r.vi[i] = __vsub4((vil | vih), 0x20202020); // vi = (vil | vih) - 32
        }
        r.d = bq6_K->d;
        return r;
    }
    // vec_dot_q6_K_q8_1_impl_mmvq
    static __device__ __forceinline__ float dot(const W & w, const X & x) {
        float sumf = 0.0f;
#pragma unroll
        for (int i = 0; i < QR6_K; ++i) {
            sumf += x.d8[i] * (ggml_cuda_dp4a(w.vi[i], x.u[i], 0) * w.sc[i]);
        }
        return w.d*sumf;
    }
};

// Q4_K and Q5_K share the activation fragment and the scale/min unpacking.
struct kqrows_x_k45 {
    int   u[2*QR4_K];
    float d8[QR4_K];
    int   su[QR4_K]; // sum of the activation bytes of fragment i (the "dot2" of the impl); depends on the column only
};

static __device__ __forceinline__ kqrows_x_k45 kqrows_xload_k45(const block_q8_1 * __restrict__ bq8_1, const int iqs) {
    kqrows_x_k45 r;
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        r.d8[i] = __low2float(bq8i->ds);
        const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);
        r.u[2*i+0] = q8[0];
        r.u[2*i+1] = q8[4];
        r.su[i] = ggml_cuda_dp4a(0x01010101, r.u[2*i+1], ggml_cuda_dp4a(0x01010101, r.u[2*i+0], 0));
    }
    return r;
}

// the branchless scale/min unpack of vec_dot_q4_K_q8_1 / vec_dot_q5_K_q8_1
static __device__ __forceinline__ void kqrows_scales_k45(const uint8_t * scales8, const int bq8_offset, int (&sc)[2], int (&m)[2]) {
    const uint16_t * scales = (const uint16_t *) scales8;
    const int j  = bq8_offset/2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * a = (const uint8_t *) aux;
    sc[0] = a[0];
    sc[1] = a[1];
    m[0]  = a[2];
    m[1]  = a[3];
}

template <> struct kqrows_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk  = QK_K;
    static constexpr int qi  = QI4_K;
    static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;
    using block = block_q4_K;
    using X = kqrows_x_k45;
    struct W { int v0[QR4_K]; int v1[QR4_K]; int sc[2]; int m[2]; float2 dm; };

    static __device__ __forceinline__ X xload(const block_q8_1 * __restrict__ bq8_1, const int iqs) {
        return kqrows_xload_k45(bq8_1, iqs);
    }
    static __device__ __forceinline__ W wload(const block * __restrict__ bq4_K, const int iqs) {
        W r;
        const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
        const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
        const int v0 = q4[0];
        const int v1 = q4[4];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            r.v0[i] = (v0 >> (4*i)) & 0x0F0F0F0F;
            r.v1[i] = (v1 >> (4*i)) & 0x0F0F0F0F;
        }
        kqrows_scales_k45(bq4_K->scales, bq8_offset, r.sc, r.m);
        r.dm = __half22float2(bq4_K->dm);
        return r;
    }
    // vec_dot_q4_K_q8_1_impl_vmmq
    static __device__ __forceinline__ float dot(const W & w, const X & x) {
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const int dot1 = ggml_cuda_dp4a(w.v1[i], x.u[2*i+1], ggml_cuda_dp4a(w.v0[i], x.u[2*i+0], 0));
            sumf_d += x.d8[i] * (dot1 * w.sc[i]);
            sumf_m += x.d8[i] * (x.su[i] * w.m[i]);
        }
        return w.dm.x*sumf_d - w.dm.y*sumf_m;
    }
};

template <> struct kqrows_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk  = QK_K;
    static constexpr int qi  = QI5_K;
    static constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    using block = block_q5_K;
    using X = kqrows_x_k45;
    struct W { int v0[QR5_K]; int v1[QR5_K]; int sc[2]; int m[2]; float2 dm; };

    static __device__ __forceinline__ X xload(const block_q8_1 * __restrict__ bq8_1, const int iqs) {
        return kqrows_xload_k45(bq8_1, iqs);
    }
    static __device__ __forceinline__ W wload(const block * __restrict__ bq5_K, const int iqs) {
        W r;
        const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
        const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
        const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));
        const int vl0 = ql[0];
        const int vl1 = ql[4];
        const int vh0 = qh[0] >> bq8_offset;
        const int vh1 = qh[4] >> bq8_offset;
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            r.v0[i] = ((vl0 >> (4*i)) & 0x0F0F0F0F) | (((vh0 >> i) << 4) & 0x10101010);
            r.v1[i] = ((vl1 >> (4*i)) & 0x0F0F0F0F) | (((vh1 >> i) << 4) & 0x10101010);
        }
        kqrows_scales_k45(bq5_K->scales, bq8_offset, r.sc, r.m);
        r.dm = __half22float2(bq5_K->dm);
        return r;
    }
    // vec_dot_q5_K_q8_1_impl_vmmq
    static __device__ __forceinline__ float dot(const W & w, const X & x) {
        float sumf_d = 0.0f;
        float sumf_m = 0.0f;
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const int dot1 = ggml_cuda_dp4a(w.v0[i], x.u[2*i+0], ggml_cuda_dp4a(w.v1[i], x.u[2*i+1], 0));
            sumf_d += x.d8[i] * (dot1 * w.sc[i]);
            sumf_m += x.d8[i] * (x.su[i] * w.m[i]);
        }
        return w.dm.x*sumf_d - w.dm.y*sumf_m;
    }
};

template <ggml_type type, int ncols_dst, int rows, int nwarps>
__launch_bounds__(nwarps*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q_kqrows(
        const void * __restrict__ vx, const block_q8_1 * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst) {
    using T = kqrows_traits<type>;
    constexpr int qk  = T::qk;
    constexpr int qi  = T::qi;
    constexpr int vdr = T::vdr;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

    const int tid  = WARP_SIZE*threadIdx.y + threadIdx.x;
    const int row0 = rows*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int kqs = vdr * (tid % (qi/vdr));

    const typename T::block * x = (const typename T::block *) vx + (int64_t) row0*stride_row_x;

    // Rows past the end of the matrix (last CTA only) re-read the last row instead of branching around it: a branch
    // between the per-row loads made nvcc issue them one row at a time (load, dot, load, dot), which ran 15-55% slower
    // than MMVQ on 2560- and 10240-row matrices; MMVQ issues every load of a K step first. The extra rows' sums are
    // never written, so the output is unchanged.
    int row_off[rows];
#pragma unroll
    for (int i = 0; i < rows; ++i) {
        row_off[i] = min(i, nrows_x - 1 - row0) * stride_row_x;
    }

    float tmp[ncols_dst][rows] = {{0.0f}};

    ggml_cuda_pdl_sync();

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);

        typename T::X xv[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            xv[j] = T::xload(y + j*stride_col_y + kby, kqs);
        }
        typename T::W w[rows];
#pragma unroll
        for (int i = 0; i < rows; ++i) {
            w[i] = T::wload(x + row_off[i] + kbx, kqs);
        }
#pragma unroll
        for (int i = 0; i < rows; ++i) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j][i] += T::dot(w[i], xv[j]);
            }
        }
    }

    // the MMVQ kernel's reduction: warps 1.. park their partials, warp 0 adds them in warp order, then a warp sum
    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<WARP_SIZE>(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < nrows_x) {
                dst[j*stride_col_dst + row0 + i] = tmp[j][i];
            }
        }
    }
}

template <ggml_type type, int ncols_dst, int rows>
static void kqrows_launch_nw(const void * vx, const block_q8_1 * y, float * dst,
        int ncols_x, int nrows_x, int stride_row_x, int stride_col_y, int stride_col_dst, int nwarps, cudaStream_t stream) {
    const dim3 grid((nrows_x + rows - 1) / rows, 1, 1);
    switch (nwarps) {
        case 2: {
            const dim3 block(WARP_SIZE, 2, 1);
            mul_mat_vec_q_kqrows<type, ncols_dst, rows, 2><<<grid, block, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
        } break;
        case 4: {
            const dim3 block(WARP_SIZE, 4, 1);
            mul_mat_vec_q_kqrows<type, ncols_dst, rows, 4><<<grid, block, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
        } break;
        case 8: {
            const dim3 block(WARP_SIZE, 8, 1);
            mul_mat_vec_q_kqrows<type, ncols_dst, rows, 8><<<grid, block, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
        } break;
        default:
            GGML_ABORT("kqrows: nwarps must be 2, 4 or 8");
    }
}

template <ggml_type type, int ncols_dst>
static void kqrows_launch_rows(const void * vx, const block_q8_1 * y, float * dst,
        int ncols_x, int nrows_x, int stride_row_x, int stride_col_y, int stride_col_dst, int rows, int nwarps, cudaStream_t stream) {
    switch (rows) {
        case 1: kqrows_launch_nw<type, ncols_dst, 1>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, nwarps, stream); break;
        case 2: kqrows_launch_nw<type, ncols_dst, 2>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, nwarps, stream); break;
        case 4: kqrows_launch_nw<type, ncols_dst, 4>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, nwarps, stream); break;
        case 8: kqrows_launch_nw<type, ncols_dst, 8>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, nwarps, stream); break;
        default: GGML_ABORT("kqrows: rows must be 1, 2, 4 or 8");
    }
}

template <ggml_type type>
void ggml_cuda_kqrows_launch(const void * vx, const void * vy_q8_1, float * dst,
        int ncols_x, int nrows_x, int ncols_dst, int stride_row_x, int stride_col_y, int stride_col_dst,
        int rows, int nwarps, cudaStream_t stream) {
    const block_q8_1 * y = (const block_q8_1 *) vy_q8_1;
    switch (ncols_dst) {
        case 2: kqrows_launch_rows<type, 2>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream); break;
        case 3: kqrows_launch_rows<type, 3>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream); break;
        case 4: kqrows_launch_rows<type, 4>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream); break;
        case 5: kqrows_launch_rows<type, 5>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream); break;
        case 6: kqrows_launch_rows<type, 6>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream); break;
        case 7: kqrows_launch_rows<type, 7>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream); break;
        case 8: kqrows_launch_rows<type, 8>(vx, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, rows, nwarps, stream); break;
        default: GGML_ABORT("kqrows: ncols_dst must be 2..8");
    }
    CUDA_CHECK(cudaGetLastError());
}
