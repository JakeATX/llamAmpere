#include "mmq-prefill.cuh"
#include "mmq.cuh"

#include <climits>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int tile_m = 128;
constexpr int nthreads = 256;

static int prefill_stages() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_MMQ_PREFILL_STAGES");
        return env && strcmp(env, "2") == 0 ? 2 : env && strcmp(env, "3") == 0 ? 3 : 1;
    }();
    return value;
}

static int prefill_tile_n() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_MMQ_PREFILL_TILE");
        return env && strcmp(env, "32") == 0 ? 32 : env && strcmp(env, "128") == 0 ? 128 : 64;
    }();
    return value;
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ int prmt(const int a, const int b, const uint32_t selector) {
    int v;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(selector));
    return v;
}

static __device__ __forceinline__ int2 lookup_iq4(const uint32_t q) {
    const int * table = (const int *) kvalues_iq4nl;
    const uint32_t indices = q & 0x77777777u;
    const uint32_t select = 0x32103210u | ((q & 0x88888888u) >> 1);
    const int lo0 = prmt(table[0], table[1], indices);
    const int hi0 = prmt(table[2], table[3], indices);
    const int lo1 = prmt(table[0], table[1], indices >> 16);
    const int hi1 = prmt(table[2], table[3], indices >> 16);
    return make_int2(prmt(lo0, hi0, select), prmt(lo1, hi1, select >> 16));
}

template <int bytes>
static __device__ __forceinline__ void copy_async(void * dst, const void * src, const bool valid) {
#if __CUDA_ARCH__ >= 800
    const uint32_t addr = __cvta_generic_to_shared(dst);
    if constexpr (bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(addr), "l"(src), "r"(valid ? 16 : 0) : "memory");
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(addr), "l"(src), "r"(valid ? 8 : 0) : "memory");
    }
#else
    GGML_UNUSED_VARS(dst, src, valid);
#endif
}

template <ggml_type type>
static constexpr __host__ __device__ int weight_stride() {
    return type == GGML_TYPE_IQ4_XS ? 72 : 4*sizeof(block_q5_0);
}

template <ggml_type type, int tile_n>
static __device__ __forceinline__ void load_stage(
        char * stage, const char * x, const block_q8_1_mmq * y,
        const int k, const int m0, const int n0, const int m, const int n, const int stride_x) {
    constexpr int x_stride = weight_stride<type>();
    constexpr int chunks = x_stride/8;
#pragma unroll 1
    for (int t = threadIdx.x; t < tile_m*chunks; t += nthreads) {
        const int row = t / chunks;
        const int chunk = t % chunks;
        const char * src;
        if constexpr (type == GGML_TYPE_IQ4_XS) {
            src = x + (size_t(min(m0 + row, m - 1))*stride_x + k/2)*sizeof(block_iq4_xs);
            src += chunk == 0 ? 0 : 8 + (k%2)*64 + (chunk - 1)*8;
        } else {
            src = x + (size_t(min(m0 + row, m - 1))*stride_x + k*4)*sizeof(block_q5_0) + chunk*8;
        }
        copy_async<8>(stage + row*x_stride + chunk*8, src, m0 + row < m);
    }
    char * ys = stage + tile_m*x_stride;
#pragma unroll 1
    for (int t = threadIdx.x; t < tile_n*9; t += nthreads) {
        const int row = t / 9;
        const int chunk = t % 9;
        const char * src = (const char *) (y + size_t(k)*n + min(n0 + row, n - 1));
        copy_async<16>(ys + row*sizeof(block_q8_1_mmq) + chunk*16, src + chunk*16, n0 + row < n);
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <ggml_type type>
static __device__ __forceinline__ int2 weight_fragment(const char * row, const int sub, const int lane) {
    if constexpr (type == GGML_TYPE_IQ4_XS) {
        const uint16_t * qs = (const uint16_t *) (row + 8 + sub*16);
        const uint32_t q = uint32_t(qs[lane%4]) | (uint32_t(qs[4 + lane%4]) << 16);
        return lookup_iq4(q);
    } else {
        const block_q5_0 * b = (const block_q5_0 *) row + sub;
        uint32_t high;
        memcpy(&high, b->qh, sizeof(high));
        uint32_t words[2] = {};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int i = 2*(lane%4) + (p/2)*8 + p%2;
            const int q = b->qs[i];
            const uint8_t lo = ((q & 15) | (((high >> i) & 1) << 4)) - 16;
            const uint8_t hi = ((q >> 4) | (((high >> (i + 16)) & 1) << 4)) - 16;
            words[p/2] |= (uint32_t(lo) | (uint32_t(hi) << 8)) << (16*(p%2));
        }
        return make_int2(words[0], words[1]);
    }
}

template <ggml_type type>
static __device__ __forceinline__ float weight_scale(const char * row, const int sub) {
    if constexpr (type == GGML_TYPE_IQ4_XS) {
        const int high = *(const uint16_t *) (row + 2);
        const uint8_t * low = (const uint8_t *) (row + 4);
        const int scale = ((low[sub/2] >> (4*(sub%2))) & 15) | (((high >> (2*sub)) & 3) << 4);
        return __half2float(*(const half *) row) * (scale - 32);
    } else {
        return __half2float(((const block_q5_0 *) row)[sub%4].d);
    }
}

template <ggml_type type, int tile_n, int nstages>
__launch_bounds__(nthreads, tile_n == 128 ? 1 : 2)
static __global__ void mul_mat_q_prefill(
        const char * x, const block_q8_1_mmq * y, float * dst,
        const int m, const int n, const int k_steps, const int stride_x, const int stride_dst) {
#if __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 900
    using namespace ggml_cuda_mma;
    constexpr int x_stride = weight_stride<type>();
    constexpr int stage_bytes = tile_m*x_stride + tile_n*sizeof(block_q8_1_mmq);
    extern __shared__ __align__(16) char stages[];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int m0 = blockIdx.x*tile_m;
    const int n0 = blockIdx.y*tile_n;
    const int row0 = warp*16 + lane/4;
    float sum[tile_n/8][4] = {};

    // fork: Each stage owns one async group, including empty groups at the K tail.
    if constexpr (nstages > 1) {
#pragma unroll
        for (int p = 0; p < nstages - 1; ++p) {
            if (p < k_steps) {
                load_stage<type, tile_n>(stages + p*stage_bytes, x, y, p, m0, n0, m, n, stride_x);
            } else {
                asm volatile("cp.async.commit_group;" ::: "memory");
            }
        }
    }
    for (int k = 0; k < k_steps; ++k) {
        const int slot = k%nstages;
        if constexpr (nstages == 1) {
            load_stage<type, tile_n>(stages, x, y, k, m0, n0, m, n, stride_x);
        }
        asm volatile("cp.async.wait_group %0;" :: "n"(nstages > 1 ? nstages - 2 : 0) : "memory");
        __syncthreads();
        if constexpr (nstages > 1) {
            const int next = k + nstages - 1;
            if (next < k_steps) {
                load_stage<type, tile_n>(stages + (next%nstages)*stage_bytes, x, y, next, m0, n0, m, n, stride_x);
            } else {
                asm volatile("cp.async.commit_group;" ::: "memory");
            }
        }

        const char * xr0 = stages + slot*stage_bytes + row0*x_stride;
        const char * xr1 = xr0 + 8*x_stride;
        const block_q8_1_mmq * ys = (const block_q8_1_mmq *) (stages + slot*stage_bytes + tile_m*x_stride);
#pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            tile<16, 8, int> a;
            const int2 v0 = weight_fragment<type>(xr0, sub, lane);
            const int2 v1 = weight_fragment<type>(xr1, sub, lane);
            a.x[0] = v0.x;
            a.x[1] = v1.x;
            a.x[2] = v0.y;
            a.x[3] = v1.y;
            const float da0 = weight_scale<type>(xr0, (k%2)*4 + sub);
            const float da1 = weight_scale<type>(xr1, (k%2)*4 + sub);
#pragma unroll
            for (int j = 0; j < tile_n/8; ++j) {
                const int * q = (const int *) (ys[j*8 + lane/4].qs + sub*32);
                tile<8, 8, int> b;
                b.x[0] = q[lane%4];
                b.x[1] = q[4 + lane%4];
                tile<16, 8, int> c;
                mma(c, a, b);
                const float db0 = ys[j*8 + 2*(lane%4) + 0].d4[sub];
                const float db1 = ys[j*8 + 2*(lane%4) + 1].d4[sub];
                sum[j][0] += c.x[0]*da0*db0;
                sum[j][1] += c.x[1]*da0*db1;
                sum[j][2] += c.x[2]*da1*db0;
                sum[j][3] += c.x[3]*da1*db1;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < tile_n/8; ++j) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int row = m0 + row0 + (l/2)*8;
            const int col = n0 + j*8 + 2*(lane%4) + l%2;
            if (row < m && col < n) {
                dst[size_t(col)*stride_dst + row] = sum[j][l];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, m, n, k_steps, stride_x, stride_dst);
#endif
}

template <ggml_type type, int tile_n, int nstages>
static void launch_prefill(const mmq_args & args, cudaStream_t stream) {
    constexpr int shared = nstages*(tile_m*weight_stride<type>() + tile_n*sizeof(block_q8_1_mmq));
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_prefill<type, tile_n, nstages>), shared);
    const dim3 grid((args.nrows_x + tile_m - 1)/tile_m, (args.ncols_dst + tile_n - 1)/tile_n);
    mul_mat_q_prefill<type, tile_n, nstages><<<grid, nthreads, shared, stream>>>(
        args.x, (const block_q8_1_mmq *) args.y, args.dst,
        args.nrows_x, args.ncols_dst, args.ncols_x/128, args.stride_row_x, args.nrows_dst);
}

template <ggml_type type, int tile_n>
static void launch_prefill_stages(const mmq_args & args, cudaStream_t stream) {
    switch (prefill_stages()) {
        case 2: launch_prefill<type, tile_n, 2>(args, stream); break;
        case 3: launch_prefill<type, tile_n, 3>(args, stream); break;
        default: launch_prefill<type, tile_n, 1>(args, stream); break;
    }
}

template <ggml_type type>
static void launch_prefill_tile(const mmq_args & args, cudaStream_t stream) {
    switch (prefill_tile_n()) {
        case 32: launch_prefill_stages<type, 32>(args, stream); break;
        case 128: launch_prefill_stages<type, 128>(args, stream); break;
        default: launch_prefill_stages<type, 64>(args, stream); break;
    }
}

#endif // !GGML_USE_HIP && !GGML_USE_MUSA

} // namespace

bool ggml_cuda_should_use_mmq_prefill(const mmq_args & args) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_MMQ_PREFILL");
        return env && strcmp(env, "1") == 0;
    }();
    if (!enabled || (args.type_x != GGML_TYPE_IQ4_XS && args.type_x != GGML_TYPE_Q5_0) ||
            args.ids_dst || args.expert_bounds || args.y_scale ||
            args.nchannels_x != 1 || args.nchannels_y != 1 || args.nsamples_x != 1 || args.nsamples_y != 1 ||
            args.ncols_dst < 256 || args.ncols_dst != args.ncols_y || args.ncols_max != args.ncols_dst ||
            args.ncols_x <= 0 || args.ncols_x%256 != 0 || args.nrows_x <= 0 ||
            args.ncols_x > INT_MAX || args.nrows_x > INT_MAX - tile_m || args.ncols_dst > INT_MAX ||
            args.stride_row_x <= 0 || args.stride_row_x > INT_MAX ||
            args.nrows_dst < args.nrows_x || args.nrows_dst > INT_MAX || args.ncols_dst > 65535*128 ||
            uintptr_t(args.x)%8 != 0 || uintptr_t(args.y)%16 != 0 ||
            (args.type_x == GGML_TYPE_Q5_0 && args.stride_row_x%4 != 0)) {
        return false;
    }
    if (size_t(args.stride_row_x) > SIZE_MAX/ggml_type_size(args.type_x)/size_t(args.nrows_x)) {
        return false;
    }
    const auto & device = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const int cc = device.cc;
    const int stride = args.type_x == GGML_TYPE_IQ4_XS ? weight_stride<GGML_TYPE_IQ4_XS>() : weight_stride<GGML_TYPE_Q5_0>();
    const size_t shared = prefill_stages()*(tile_m*stride + prefill_tile_n()*sizeof(block_q8_1_mmq));
    return ampere_mma_available(cc) && (cc == 800 || cc == 860) && shared <= device.smpbo &&
           (args.ncols_dst + prefill_tile_n() - 1)/prefill_tile_n() <= 65535;
#else
    GGML_UNUSED(args);
    return false;
#endif
}

void ggml_cuda_mul_mat_q_prefill(const mmq_args & args, cudaStream_t stream) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (args.type_x == GGML_TYPE_IQ4_XS) {
        launch_prefill_tile<GGML_TYPE_IQ4_XS>(args, stream);
    } else {
        launch_prefill_tile<GGML_TYPE_Q5_0>(args, stream);
    }
    CUDA_CHECK(cudaGetLastError());
#else
    GGML_UNUSED_VARS(args, stream);
    GGML_ABORT("MMQ prefill requires Ampere CUDA");
#endif
}
