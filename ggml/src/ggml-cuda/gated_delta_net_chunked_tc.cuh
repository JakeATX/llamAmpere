#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "mma.cuh"

namespace {

constexpr int GDN_TC_CHUNK = 32;
constexpr int GDN_TC_COLS = 32;
constexpr int GDN_TC_LD = GDN_TC_COLS + 1;
constexpr int GDN_TC_QK_LD = 132;
constexpr int GDN_TC_THREADS = 128;
constexpr int GDN_TC_FLOATS = 2 * GDN_TC_CHUNK * GDN_TC_QK_LD + 128 * GDN_TC_LD +
                              2 * GDN_TC_CHUNK * GDN_TC_CHUNK + 2 * GDN_TC_CHUNK * GDN_TC_LD +
                              2 * GDN_TC_CHUNK;
constexpr int GDN_TC_SHARED = GDN_TC_FLOATS * sizeof(float) + GDN_TC_CHUNK * sizeof(double);

// The residual products retain about 16 mantissa bits without the FP16 range limit.
static __device__ __forceinline__ void gdn_tc_split(float x, float y, nv_bfloat162 & hi, nv_bfloat162 & lo) {
    hi = __floats2bfloat162_rn(x, y);
    lo = __floats2bfloat162_rn(x - __low2float(hi), y - __high2float(hi));
}

template <int M, int N, int K>
static __device__ __forceinline__ void gdn_tc_mm(
        const float * a, int ar, int ac, const float * b, int br, int bc, float * c, int cr) {
#if defined(AMPERE_MMA_AVAILABLE)
    using namespace ggml_cuda_mma;
    const int warp = threadIdx.y;
    for (int tile_id = warp; tile_id < (M / 16) * (N / 8); tile_id += GDN_TC_THREADS / WARP_SIZE) {
        const int row = (tile_id / (N / 8)) * 16;
        const int col = (tile_id % (N / 8)) * 8;
        tile<16, 8, float> sum;
        for (int p = 0; p < K; p += 16) {
            tile<16, 8, nv_bfloat162> ah, al;
            tile<8, 8, nv_bfloat162> bh, bl;
#pragma unroll
            for (int i = 0; i < ah.ne; ++i) {
                const int r = row + ah.get_i(i);
                const int k = p + 2 * ah.get_j(i);
                gdn_tc_split(a[r * ar + k * ac], a[r * ar + (k + 1) * ac], ah.x[i], al.x[i]);
            }
#pragma unroll
            for (int i = 0; i < bh.ne; ++i) {
                const int n = col + bh.get_i(i);
                const int k = p + 2 * bh.get_j(i);
                gdn_tc_split(b[k * br + n * bc], b[(k + 1) * br + n * bc], bh.x[i], bl.x[i]);
            }
            mma(sum, al, bh);
            mma(sum, ah, bl);
            mma(sum, ah, bh);
        }
#pragma unroll
        for (int i = 0; i < sum.ne; ++i) {
            c[(row + sum.get_i(i)) * cr + col + sum.get_j(i)] = sum.x[i];
        }
    }
    __syncthreads();
#else
    GGML_UNUSED_VARS(a, ar, ac, b, br, bc, c, cr);
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
    float * qs = smem;
    float * ks = qs + GDN_TC_CHUNK * GDN_TC_QK_LD;
    float * ss = ks + GDN_TC_CHUNK * GDN_TC_QK_LD;
    float * lower = ss + 128 * GDN_TC_LD;
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
            qs[r * GDN_TC_QK_LD + d] = qv;
            ks[r * GDN_TC_QK_LD + d] = kv;
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
            const int col = threadIdx.y * 8 + j / 4;
            const int row = threadIdx.x + (j % 4) * WARP_SIZE;
            ss[row * GDN_TC_LD + col] = s[j];
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
            decay[0] = unsafe || total < -80.0 ? 1.0f : 0.0f;
        }
        __syncthreads();
        // Avoid factoring a decay that can underflow before it multiplies a large state.
        if (decay[0] != 0.0f) {
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

        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_CHUNK, 128>(ks, GDN_TC_QK_LD, 1, ks, 1, GDN_TC_QK_LD, lower, GDN_TC_CHUNK);
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_CHUNK, 128>(qs, GDN_TC_QK_LD, 1, ks, 1, GDN_TC_QK_LD, attn, GDN_TC_CHUNK);
        for (int e = tid; e < GDN_TC_CHUNK * GDN_TC_CHUNK; e += GDN_TC_THREADS) {
            const int i = e / GDN_TC_CHUNK;
            const int j = e % GDN_TC_CHUNK;
            // Mask before exp: the upper triangle can overflow under strong decay.
            const float d = i >= j ? expf((float) (prefix[i] - prefix[j])) : 0.0f;
            lower[e] = i > j ? lower[e] * betas[i] * d : 0.0f;
            attn[e] = i >= j ? attn[e] * d : 0.0f;
        }
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_COLS, 128>(ks, GDN_TC_QK_LD, 1, ss, GDN_TC_LD, 1, delta, GDN_TC_LD);
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_COLS, 128>(qs, GDN_TC_QK_LD, 1, ss, GDN_TC_LD, 1, base, GDN_TC_LD);
        for (int e = tid; e < GDN_TC_CHUNK * GDN_TC_COLS; e += GDN_TC_THREADS) {
            const int r = e / GDN_TC_COLS;
            const int c = e % GDN_TC_COLS;
            const float value = r < count ? v[seq * sv3 + (t0 + r) * sv2 + h * sv1 + col0 + c] : 0.0f;
            delta[r * GDN_TC_LD + c] = betas[r] * (value - decay[r] * delta[r * GDN_TC_LD + c]);
        }
        __syncthreads();
        // Forward substitution of (I + lower) delta = beta (v - decay K S).
        for (int j = 0; j < count; ++j) {
            for (int e = tid; e < GDN_TC_CHUNK * GDN_TC_COLS; e += GDN_TC_THREADS) {
                const int i = e / GDN_TC_COLS;
                const int c = e % GDN_TC_COLS;
                if (i > j) {
                    delta[i * GDN_TC_LD + c] -= lower[i * GDN_TC_CHUNK + j] * delta[j * GDN_TC_LD + c];
                }
            }
            __syncthreads();
        }
        // ss is no longer needed as input; reuse it for the intra-chunk output.
        gdn_tc_mm<GDN_TC_CHUNK, GDN_TC_COLS, GDN_TC_CHUNK>(attn, GDN_TC_CHUNK, 1, delta, GDN_TC_LD, 1, ss, GDN_TC_LD);
        for (int e = tid; e < count * GDN_TC_COLS; e += GDN_TC_THREADS) {
            const int r = e / GDN_TC_COLS;
            const int c = e % GDN_TC_COLS;
            output[((seq * T + t0 + r) * H + h) * 128 + col0 + c] =
                (decay[r] * base[r * GDN_TC_LD + c] + ss[r * GDN_TC_LD + c]) * scale;
        }
        if (tid < GDN_TC_CHUNK) {
            betas[tid] = expf((float) (prefix[count - 1] - prefix[tid]));
        }
        __syncthreads();
        for (int e = tid; e < GDN_TC_CHUNK * 128; e += GDN_TC_THREADS) {
            ks[(e / 128) * GDN_TC_QK_LD + e % 128] *= betas[e / 128];
        }
        __syncthreads();
        gdn_tc_mm<128, GDN_TC_COLS, GDN_TC_CHUNK>(ks, 1, GDN_TC_QK_LD, delta, GDN_TC_LD, 1, ss, GDN_TC_LD);
        const float carry = decay[count - 1];
#pragma unroll
        for (int j = 0; j < 128 * GDN_TC_COLS / GDN_TC_THREADS; ++j) {
            const int col = threadIdx.y * 8 + j / 4;
            const int row = threadIdx.x + (j % 4) * WARP_SIZE;
            s[j] = carry * s[j] + ss[row * GDN_TC_LD + col];
        }
        // All warps must finish reading ss before the next chunk stages its state.
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
