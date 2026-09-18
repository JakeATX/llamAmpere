// EXL3 format, "mul1" codebook arithmetic, tile layout and trellis GEMV design: Turboderp, exllamav3
// (https://github.com/turboderp-org/exllamav3), MIT License, Copyright (c) 2025 Turboderp; see
// licenses/LICENSE-exllamav3. EXL3 is Turboderp's streamlined variant of QTIP (Tseng, Sun, Hou, De Sa,
// "QTIP: Quantization with Trellises and Incoherence Processing", NeurIPS 2024, arXiv:2406.11235).
// This file is an independent ggml/CUDA reimplementation written with the exllamav3 source as reference;
// no QTIP code was used.
#include "exl3.cuh"
#include "convert.cuh"

// exllamav3 "mul1" codebook (cb2, exl3_dq.cuh): x = code * 0x83DCD12D; s = bytesum(x) + 0x6400 as an fp16 bit
// pattern; value = hfma(s, fp16(0x1eee), fp16(0xc931)) -- one rounding.
static __device__ __forceinline__ half exl3_mul1_decode(uint32_t code) {
    const uint32_t x = code * 0x83DCD12Du;
    const uint32_t s = (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + ((x >> 24) & 0xffu) + 0x6400u;
    const half h = __ushort_as_half((unsigned short) s);
    return __hfma(h, __ushort_as_half((unsigned short) 0x1eee), __ushort_as_half((unsigned short) 0xc931));
}

// one block (256 threads) per 16x16 tile, one thread per weight; tile order is the native trellis order
// [K/16][N/16][8*bits words] (k-tile major).
static __global__ void k_exl3_reconstruct_f16(const uint32_t * __restrict__ trellis, half * __restrict__ W,
                                              const int kt, const int nt, const int bits) {
    __shared__ uint32_t words[64];
    const int tile = blockIdx.x;
    const int ki = tile / nt;
    const int ni = tile - ki * nt;
    const int nw = 8 * bits;
    const uint32_t * src = trellis + (size_t) tile * nw;
    if (threadIdx.x < (unsigned) nw) {
        words[threadIdx.x] = src[threadIdx.x];
    }
    __syncthreads();
    const int t  = threadIdx.x;
    const int b0 = ((t + 257) * bits - 16) % (256 * bits);
    const int i0 = b0 >> 5;
    const int sh = b0 & 31;
    const uint32_t a = words[i0];
    const uint32_t b = words[(i0 + 1) % nw];
    const uint32_t code = (uint32_t) (((((uint64_t) a) << 32) | (uint64_t) b) >> (48 - sh)) & 0xffffu;
    const int lane = t >> 3;
    const int j    = t & 7;
    const int row  = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);      // k within tile
    const int col  = 2 * (lane >> 3) + ((lane >> 2) & 1) + 8 * (j >> 2);  // n within tile
    const size_t K = (size_t) kt * 16;
    W[((size_t) ni * 16 + col) * K + (size_t) ki * 16 + row] = exl3_mul1_decode(code);
}

void ggml_cuda_exl3_reconstruct_f16(const ggml_tensor * src0, half * dst, cudaStream_t stream) {
    const int bits = ggml_exl3_bits(src0->type);
    GGML_ASSERT(bits != 0);
    GGML_ASSERT(src0->ne[0] % 16 == 0 && src0->ne[1] % 16 == 0 && src0->ne[2] == 1 && src0->ne[3] == 1);
    const int kt = (int) (src0->ne[0] / 16);
    const int nt = (int) (src0->ne[1] / 16);
    k_exl3_reconstruct_f16<<<kt * nt, 256, 0, stream>>>((const uint32_t *) src0->data, dst, kt, nt, bits);
}

// ---- fused glue: y = svh * H128( W . H128( suh * x ) )  (exllamav3 exl3.py forward, Hadamard block 128) ----
static __device__ __forceinline__ void exl3_store(float * p, float v) { *p = v; }
static __device__ __forceinline__ void exl3_store(half  * p, float v) { *p = __float2half_rn(v); }

// out[T][K] = scale * H128(suh * x)   (HAD)  |  scale * x   (!HAD);  x, out contiguous [T][K], K % 128 == 0
template <typename OUT, bool HAD>
static __global__ void k_exl3_glue_in(const float * __restrict__ x, const float * __restrict__ suh, OUT * __restrict__ out,
                                      const int K, const size_t n_chunks, const float scale) {
    const size_t c = (size_t) blockIdx.x * 4 + (threadIdx.x >> 5);
    if (c >= n_chunks) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const size_t base = c * 128;
    const int kc = (int) (base % (size_t) K);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = x[base + j * 32 + lane];
        if (HAD) {
            v[j] *= suh[kc + j * 32 + lane];
        }
    }
    float s = scale;
    if (HAD) {
        exl3_wht128(v);
        s *= EXL3_WHT128_SCALE;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        exl3_store(out + base + j * 32 + lane, v[j] * s);
    }
}

// y[T][N] = scale * svh * H128( sum_ks part[ks][T][N] )   (HAD)  |  scale * sum_ks part   (!HAD);  N % 128 == 0
template <bool HAD>
static __global__ void k_exl3_glue_out(const float * __restrict__ part, const float * __restrict__ svh, float * __restrict__ y,
                                       const int N, const size_t n_chunks, const int ksplit, const size_t ks_stride, const float scale) {
    const size_t c = (size_t) blockIdx.x * 4 + (threadIdx.x >> 5);
    if (c >= n_chunks) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const size_t base = c * 128;
    const int nc = (int) (base % (size_t) N);
    // sum the split-K partials with four rows in flight (the serial chain was latency-bound at width > 1);
    // summation order is fixed => deterministic
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int ks = 0;
    for (; ks + 4 <= ksplit; ks += 4) {
        const float * p = part + (size_t) ks * ks_stride + base + lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float a0 = p[j * 32];
            const float a1 = p[ks_stride + j * 32];
            const float a2 = p[2 * ks_stride + j * 32];
            const float a3 = p[3 * ks_stride + j * 32];
            v[j] += (a0 + a1) + (a2 + a3);
        }
    }
    for (; ks < ksplit; ++ks) {
        const float * p = part + (size_t) ks * ks_stride + base + lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[j] += p[j * 32];
        }
    }
    float s = scale;
    if (HAD) {
        exl3_wht128(v);
        s *= EXL3_WHT128_SCALE;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float m = HAD ? svh[nc + j * 32 + lane] : 1.0f;
        y[base + j * 32 + lane] = v[j] * s * m;
    }
}

template <typename OUT>
static void exl3_glue_in(const float * x, const float * suh, OUT * out, const int K, const int64_t T, const float scale, cudaStream_t stream) {
    GGML_ASSERT(K % 128 == 0);
    const size_t n_chunks = (size_t) T * K / 128;
    const int nb = (int) ((n_chunks + 3) / 4);
    if (suh) {
        k_exl3_glue_in<OUT, true ><<<nb, 128, 0, stream>>>(x, suh, out, K, n_chunks, scale);
    } else {
        k_exl3_glue_in<OUT, false><<<nb, 128, 0, stream>>>(x, suh, out, K, n_chunks, scale);
    }
}

void ggml_cuda_exl3_glue_in_f16(const float * x, const float * suh, half * out, const int K, const int64_t T, const float scale, cudaStream_t stream) {
    exl3_glue_in<half>(x, suh, out, K, T, scale, stream);
}

void ggml_cuda_exl3_glue_out(const float * part, const float * svh, float * y, const int N, const int64_t T, const int ksplit, const float scale, cudaStream_t stream) {
    GGML_ASSERT(N % 128 == 0);
    const size_t n_chunks = (size_t) T * N / 128;
    const size_t ks_stride = (size_t) T * N;
    const int nb = (int) ((n_chunks + 3) / 4);
    if (svh) {
        k_exl3_glue_out<true ><<<nb, 128, 0, stream>>>(part, svh, y, N, n_chunks, ksplit, ks_stride, scale);
    } else {
        k_exl3_glue_out<false><<<nb, 128, 0, stream>>>(part, svh, y, N, n_chunks, ksplit, ks_stride, scale);
    }
}

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_exl3_bits(src0->type) != 0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    GGML_ASSERT(src1->ne[0] == K && dst->ne[0] == N);
    const int64_t T = src1->ne[1] * src1->ne[2] * src1->ne[3];   // all activation columns (src0 is 2-D, broadcast)

    // fused glue (llama build_lora_mm): src[2] = suh [K], src[3] = svh [N]; absent -> codebook values only
    const ggml_tensor * suh_t = dst->src[2];
    const ggml_tensor * svh_t = dst->src[3];
    const float * suh = nullptr;
    const float * svh = nullptr;
    if (suh_t != nullptr) {
        GGML_ASSERT(svh_t != nullptr && suh_t->type == GGML_TYPE_F32 && svh_t->type == GGML_TYPE_F32);
        GGML_ASSERT(suh_t->ne[0] == K && svh_t->ne[0] == N && K % 128 == 0 && N % 128 == 0);
        GGML_ASSERT(ggml_is_contiguous(suh_t) && ggml_is_contiguous(svh_t));
        suh = (const float *) suh_t->data;
        svh = (const float *) svh_t->data;
    }

    if (ggml_cuda_exl3_gemv_supported(T)) {
        ggml_cuda_exl3_gemv(ctx, src0, src1, dst, T, suh, svh);
        return;
    }

    const int id = ggml_cuda_get_device();
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> w_f16(ctx.pool(id), K * N);
    ggml_cuda_exl3_reconstruct_f16(src0, w_f16.get(), stream);

    ggml_cuda_pool_alloc<half> x_f16(ctx.pool(id), K * T);
    ggml_cuda_exl3_glue_in_f16((const float *) src1->data, suh, x_f16.get(), (int) K, T, 1.0f, stream);

    ggml_cuda_pool_alloc<float> y_raw(ctx.pool(id));
    float * y = (float *) dst->data;
    if (svh != nullptr) {
        y = y_raw.alloc(N * T);
    }

    // y[N x T] = W[N x K] . X[K x T]  (fp16 inputs, fp32 accumulate/output)
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
    CUBLAS_CHECK(
        cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                (int) N, (int) T, (int) K,
                &alpha, w_f16.get(), CUDA_R_16F, (int) K,
                        x_f16.get(), CUDA_R_16F, (int) K,
                &beta,  y, CUDA_R_32F, (int) N,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (svh != nullptr) {
        ggml_cuda_exl3_glue_out(y, svh, (float *) dst->data, (int) N, T, 1, 1.0f, stream);
    }
}
