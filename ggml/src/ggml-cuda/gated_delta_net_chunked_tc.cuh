#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "mma.cuh"

namespace {

constexpr int GDN_TC_CHUNK = 32;
constexpr int GDN_TC_COLS = 32;
constexpr int GDN_TC_LD = GDN_TC_COLS;
constexpr int GDN_TC_THREADS = 128;
constexpr int GDN_TC_PACK = 32 * 128 / 2;
constexpr int GDN_TC_FLOATS = 4 * GDN_TC_CHUNK * GDN_TC_COLS + 2 * GDN_TC_CHUNK;
constexpr int GDN_TC_SHARED = 4 * GDN_TC_PACK * sizeof(nv_bfloat162) +
                             GDN_TC_FLOATS * sizeof(float) + GDN_TC_CHUNK * sizeof(double);
static_assert(GDN_TC_SHARED <= 49 * 1024, "GDN TC needs two CTAs per sm_86 SM");

static __device__ __forceinline__ void gdn_tc_pack_pair(
        nv_bfloat162 * hi, nv_bfloat162 * lo, int r, int c, int ld, float x, float y) {
    const nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    const int index = ggml_cuda_mma::swizzle_bytes<true, nv_bfloat162>(r, c, ld) / sizeof(nv_bfloat162);
    hi[index] = h;
    lo[index] = __floats2bfloat162_rn(x - __low2float(h), y - __high2float(h));
}

template <int R, int C>
static __device__ __forceinline__ void gdn_tc_pack(
        nv_bfloat162 * hi, nv_bfloat162 * lo, const float * src, int64_t sr, int64_t sc, int count = R) {
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    for (int e = tid; e < R * C / 2; e += GDN_TC_THREADS) {
        const int r = e / (C / 2);
        const int c = e % (C / 2);
        gdn_tc_pack_pair(hi, lo, r, c, C / 2,
            r < count ? src[r * sr + (2 * c + 0) * sc] : 0.0f,
            r < count ? src[r * sr + (2 * c + 1) * sc] : 0.0f);
    }
    __syncthreads();
}

static __device__ __forceinline__ void gdn_tc_load_b(
        ggml_cuda_mma::tile<8, 8, nv_bfloat162> & t, const nv_bfloat162 * src, int col, int p, int ld) {
#if defined(AMPERE_MMA_AVAILABLE)
    int * x = (int *) t.x;
    const void * ptr = ggml_cuda_mma::swizzle<true>(src, col + threadIdx.x % 8, p / 2 + (threadIdx.x / 8 % 2) * 4, ld);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(x[0]), "=r"(x[1]) : "l"(ptr));
#else
    GGML_UNUSED_VARS(t, src, col, p, ld);
    NO_DEVICE_CODE;
#endif
}

template <int M, int N, int K>
static __device__ __forceinline__ void gdn_tc_mm(
        const nv_bfloat162 * ahs, const nv_bfloat162 * als,
        const nv_bfloat162 * bhs, const nv_bfloat162 * bls, float * c, int cr) {
#if defined(AMPERE_MMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    const int warp = threadIdx.y;
    for (int tile_id = warp; tile_id < (M / 16) * (N / 16); tile_id += GDN_TC_THREADS / WARP_SIZE) {
        const int row = (tile_id / (N / 16)) * 16;
        const int col = (tile_id % (N / 16)) * 16;
        tile<16, 8, float> sum[2];
        for (int p = 0; p < K; p += 16) {
            tile<16, 8, nv_bfloat162> ah, al;
            load_ldmatrix<true>(ah, ahs, row, p / 2, K / 2);
            load_ldmatrix<true>(al, als, row, p / 2, K / 2);
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                tile<8, 8, nv_bfloat162> bh, bl;
                gdn_tc_load_b(bh, bhs, col + n * 8, p, K / 2);
                gdn_tc_load_b(bl, bls, col + n * 8, p, K / 2);
                mma(sum[n], al, bh);
                mma(sum[n], ah, bl);
                mma(sum[n], ah, bh);
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int i = 0; i < sum[n].ne; ++i) {
                c[(row + sum[n].get_i(i)) * cr + col + n * 8 + sum[n].get_j(i)] = sum[n].x[i];
            }
        }
    }
    __syncthreads();
#else
    GGML_UNUSED_VARS(ahs, als, bhs, bls, c, cr);
    NO_DEVICE_CODE;
#endif
}

__global__ void __launch_bounds__(GDN_TC_THREADS) gated_delta_net_chunked_tc(
        const float * q, const float * k, const float * v, const float * g, const float * beta,
        const float * input_state, float * output, float * state,
        int64_t H, int64_t T, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, uint3 neqk1_magic, uint3 rq3_magic,
        float scale, const int32_t * slot_rows, ggml_cuda_gated_delta_net_state_src state_src) {
#if defined(AMPERE_MMA_AVAILABLE)
    extern __shared__ float smem[];
    __shared__ int fallback;
    nv_bfloat162 * ah = reinterpret_cast<nv_bfloat162 *>(smem);
    nv_bfloat162 * al = ah + GDN_TC_PACK;
    nv_bfloat162 * bh = al + GDN_TC_PACK;
    nv_bfloat162 * bl = bh + GDN_TC_PACK;
    float * lower = reinterpret_cast<float *>(bl + GDN_TC_PACK);
    float * attn = lower + GDN_TC_CHUNK * GDN_TC_CHUNK;
    float * delta = attn + GDN_TC_CHUNK * GDN_TC_CHUNK;
    float * base = delta + GDN_TC_CHUNK * GDN_TC_LD;
    float * betas = base + GDN_TC_CHUNK * GDN_TC_LD;
    float * decay = betas + GDN_TC_CHUNK;
    double * prefix = reinterpret_cast<double *>(decay + GDN_TC_CHUNK);

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int h = blockIdx.x;
    const int seq = blockIdx.y;
    const int col0 = blockIdx.z * GDN_TC_COLS;
    const int iq1 = fastmodulo(h, neqk1_magic);
    const int iq3 = fastdiv(seq, rq3_magic);
    const int64_t D = H * 128 * 128;
    ggml_cuda_pdl_sync();
    const int64_t src_row = state_src.rows ? (int64_t) state_src.rows[seq] * state_src.row_stride : seq * D;
    const float * src = input_state + src_row + h * 128 * 128 + col0 * 128;
    float s[128 * GDN_TC_COLS / GDN_TC_THREADS];
#pragma unroll
    for (int j = 0; j < 128 * GDN_TC_COLS / GDN_TC_THREADS; ++j) {
        const int col = threadIdx.y * 8 + j / 4;
        const int row = threadIdx.x + (j % 4) * WARP_SIZE;
        s[j] = src[col * 128 + row];
    }

    for (int64_t t0 = 0; t0 < T; t0 += GDN_TC_CHUNK) {
        const int count = (int) min((int64_t) GDN_TC_CHUNK, T - t0);
        bool unsafe_inputs = false;
        for (int e = tid; e < GDN_TC_CHUNK * 128; e += GDN_TC_THREADS) {
            const int r = e / 128;
            const int d = e % 128;
            const int64_t off = iq3 * sq3 + (t0 + r) * sq2 + iq1 * sq1 + d;
            const float qv = r < count ? q[off] : 0.0f;
            const float kv = r < count ? k[off] : 0.0f;
            unsafe_inputs = unsafe_inputs || !(fabsf(qv) <= 2.0f) || !(fabsf(kv) <= 2.0f);
        }
        for (int e = tid; e < count * GDN_TC_COLS; e += GDN_TC_THREADS) {
            const float value = v[seq * sv3 + (t0 + e / GDN_TC_COLS) * sv2 + h * sv1 + col0 + e % GDN_TC_COLS];
            unsafe_inputs = unsafe_inputs || !(fabsf(value) <= 1e30f);
        }
        if (tid < GDN_TC_CHUNK) {
            const int64_t off = seq * sb3 + (t0 + tid) * sb2 + h * sb1;
            prefix[tid] = tid < count ? (double) g[off] : 0.0;
            betas[tid] = tid < count ? beta[off] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < 128 * GDN_TC_COLS / GDN_TC_THREADS; ++j) {
            unsafe_inputs = unsafe_inputs || !(fabsf(s[j]) <= 1e30f);
        }
        const bool unsafe_block = __syncthreads_or(unsafe_inputs);
        if (tid == 0) {
            double total = 0.0;
            bool unsafe = unsafe_block;
            for (int i = 0; i < GDN_TC_CHUNK; ++i) {
                unsafe = unsafe || !(prefix[i] <= 0.0);
                unsafe = unsafe || !(betas[i] >= 0.0f && betas[i] <= 1.0f);
                total += prefix[i];
                prefix[i] = total;
            }
            fallback = unsafe || total < -80.0;
        }
        __syncthreads();
        // Avoid factoring a decay that can underflow before it multiplies a large state.
        // The flag stays live until every warp reaches the end-of-chunk barrier.
        if (fallback) {
            for (int t = 0; t < count; ++t) {
                const int64_t off = iq3 * sq3 + (t0 + t) * sq2 + iq1 * sq1;
                const int64_t gb = seq * sb3 + (t0 + t) * sb2 + h * sb1;
                const float d = expf(g[gb]);
                const float b = beta[gb];
                float qr[4], kr[4];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    qr[r] = q[off + threadIdx.x + r * WARP_SIZE];
                    kr[r] = k[off + threadIdx.x + r * WARP_SIZE];
                }
#pragma unroll
                for (int c = 0; c < 8; ++c) {
                    const int col = col0 + threadIdx.y * 8 + c;
                    float dot = 0.0f;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        dot += s[c * 4 + r] * kr[r];
                    }
                    dot = warp_reduce_sum<WARP_SIZE>(dot);
                    const float dv = b * (v[seq * sv3 + (t0 + t) * sv2 + h * sv1 + col] - d * dot);
                    float out = 0.0f;
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        s[c * 4 + r] = d * s[c * 4 + r] + kr[r] * dv;
                        out += s[c * 4 + r] * qr[r];
                    }
                    out = warp_reduce_sum<WARP_SIZE>(out);
                    if (threadIdx.x == 0) {
                        output[((seq * T + t0 + t) * H + h) * 128 + col] = out * scale;
                    }
                }
            }
            __syncthreads();
            continue;
        }
        if (tid < GDN_TC_CHUNK) {
            decay[tid] = expf((float) prefix[tid]);
        }

        const float * kc = k + iq3 * sq3 + t0 * sq2 + iq1 * sq1;
        const float * qc = q + iq3 * sq3 + t0 * sq2 + iq1 * sq1;
        gdn_tc_pack<GDN_TC_CHUNK, 128>(ah, al, kc, sq2, 1, count);
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_CHUNK, 128>(ah, al, ah, al, lower, GDN_TC_CHUNK);
#pragma unroll
        for (int j = 0; j < 128 * GDN_TC_COLS / GDN_TC_THREADS; ++j) {
            const int col = threadIdx.y * 8 + j / 4;
            const int row = threadIdx.x + (j % 4) * WARP_SIZE;
            const float next = __shfl_down_sync(0xffffffff, s[j], 1);
            if (threadIdx.x % 2 == 0) {
                gdn_tc_pack_pair(bh, bl, col, row / 2, 64, s[j], next);
            }
        }
        __syncthreads();
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_COLS, 128>(ah, al, bh, bl, delta, GDN_TC_LD);
        gdn_tc_pack<GDN_TC_CHUNK, 128>(ah, al, qc, sq2, 1, count);
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_COLS, 128>(ah, al, bh, bl, base, GDN_TC_LD);
        gdn_tc_pack<GDN_TC_CHUNK, 128>(bh, bl, kc, sq2, 1, count);
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_CHUNK, 128>(ah, al, bh, bl, attn, GDN_TC_CHUNK);
        float * lower_t = reinterpret_cast<float *>(ah);
        float * rhs_t = reinterpret_cast<float *>(al);
        for (int e = tid; e < GDN_TC_CHUNK * GDN_TC_CHUNK; e += GDN_TC_THREADS) {
            const int i = e / GDN_TC_CHUNK;
            const int j = e % GDN_TC_CHUNK;
            // Mask before exp: the upper triangle can overflow under strong decay.
            const float d = i >= j ? expf((float) (prefix[i] - prefix[j])) : 0.0f;
            lower_t[j * GDN_TC_CHUNK + i] = i > j ? lower[e] * betas[i] * d : 0.0f;
            attn[e] = i >= j ? attn[e] * d : 0.0f;
        }
        for (int e = tid; e < GDN_TC_CHUNK * GDN_TC_COLS; e += GDN_TC_THREADS) {
            const int r = e / GDN_TC_COLS;
            const int c = e % GDN_TC_COLS;
            const float value = r < count ? v[seq * sv3 + (t0 + r) * sv2 + h * sv1 + col0 + c] : 0.0f;
            rhs_t[c * GDN_TC_CHUNK + r] = betas[r] * (value - decay[r] * delta[r * GDN_TC_LD + c]);
        }
        __syncthreads();
        // Each warp owns eight RHS columns; lanes own token rows.
        float solved[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            solved[c] = rhs_t[(threadIdx.y * 8 + c) * GDN_TC_CHUNK + threadIdx.x];
        }
        for (int j = 0; j < count; ++j) {
            const float coefficient = lower_t[j * GDN_TC_CHUNK + threadIdx.x];
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float pivot = __shfl_sync(0xffffffff, solved[c], j);
                if (threadIdx.x > j) {
                    solved[c] -= coefficient * pivot;
                }
            }
        }
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float next = __shfl_down_sync(0xffffffff, solved[c], 1);
            if (threadIdx.x % 2 == 0) {
                gdn_tc_pack_pair(bh, bl, threadIdx.y * 8 + c, threadIdx.x / 2, GDN_TC_CHUNK / 2, solved[c], next);
            }
        }
        // All warps must release the coefficient/RHS workspace before packing attention.
        __syncthreads();
        gdn_tc_pack<GDN_TC_CHUNK, GDN_TC_CHUNK>(ah, al, attn, GDN_TC_CHUNK, 1);
        // The solve coefficients are dead; reuse them for the intra-chunk output.
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_COLS, GDN_TC_CHUNK>(ah, al, bh, bl, lower, GDN_TC_LD);
        for (int e = tid; e < count * GDN_TC_COLS; e += GDN_TC_THREADS) {
            const int r = e / GDN_TC_COLS;
            const int c = e % GDN_TC_COLS;
            output[((seq * T + t0 + r) * H + h) * 128 + col0 + c] =
                (decay[r] * base[r * GDN_TC_LD + c] + lower[r * GDN_TC_LD + c]) * scale;
        }
        if (tid < GDN_TC_CHUNK) {
            betas[tid] = expf((float) (prefix[count - 1] - prefix[tid]));
        }
        __syncthreads();
        for (int e = tid; e < 128 * GDN_TC_CHUNK / 2; e += GDN_TC_THREADS) {
            const int row = e / (GDN_TC_CHUNK / 2);
            const int t = 2 * (e % (GDN_TC_CHUNK / 2));
            gdn_tc_pack_pair(ah, al, row, t / 2, GDN_TC_CHUNK / 2,
                t + 0 < count ? kc[(t + 0) * sq2 + row] * betas[t + 0] : 0.0f,
                t + 1 < count ? kc[(t + 1) * sq2 + row] * betas[t + 1] : 0.0f);
        }
        __syncthreads();
        // All four FP32 matrices are dead once delta is packed in bh/bl.
        gdn_tc_mm<128, GDN_TC_COLS, GDN_TC_CHUNK>(ah, al, bh, bl, lower, GDN_TC_LD);
        const float carry = decay[count - 1];
#pragma unroll
        for (int j = 0; j < 128 * GDN_TC_COLS / GDN_TC_THREADS; ++j) {
            const int col = threadIdx.y * 8 + j / 4;
            const int row = threadIdx.x + (j % 4) * WARP_SIZE;
            s[j] = carry * s[j] + lower[row * GDN_TC_LD + col];
        }
        // All warps must finish shared reads before the next chunk reuses the workspace.
        __syncthreads();
    }
    const int64_t dst_row = slot_rows ? (int64_t) slot_rows[seq] * D : seq * D;
    float * dst = state + dst_row + h * 128 * 128 + col0 * 128;
#pragma unroll
    for (int j = 0; j < 128 * GDN_TC_COLS / GDN_TC_THREADS; ++j) {
        const int col = threadIdx.y * 8 + j / 4;
        const int row = threadIdx.x + (j % 4) * WARP_SIZE;
        dst[col * 128 + row] = s[j];
    }
    GGML_UNUSED(n_seqs);
#else
    GGML_UNUSED_VARS(q, k, v, g, beta, input_state, output, state, H, T, n_seqs,
                    sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, slot_rows, state_src);
    NO_DEVICE_CODE;
#endif
}

static bool ggml_cuda_gdn_tc_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_GDN_TC");
        return value != nullptr && atoi(value) == 1;
    }();
    return enabled;
}

} // namespace
#endif
