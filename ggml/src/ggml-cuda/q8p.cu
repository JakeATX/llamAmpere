// [INKDENSE PORTMAP row 1] planar Q8_0 decode matvec (see q8p.cuh): warp per row, 128 int8 values per load, F32
// activations, grouped launches and an in-place repack.
#include "q8p.cuh"

#include <algorithm>
#include <atomic>
#include <cstdlib>

static std::atomic<uint64_t> g_q8p_generation{0};

uint64_t ggml_cuda_q8p_generation() {
    return g_q8p_generation.load(std::memory_order_acquire);
}

bool ggml_cuda_q8p_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_Q8P"); return e != nullptr && atoi(e) != 0; }();
    return on;
}

bool ggml_cuda_q8p_shape_ok(const ggml_tensor * w) {
    return w->type == GGML_TYPE_Q8_0 && w->op == GGML_OP_NONE && w->view_src == nullptr && w->ne[0] % 128 == 0 &&
           w->ne[1] >= 1 && w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) && ((uintptr_t) w->data % 16) == 0;
}

// ---- layout: block b (row r, block j of the row, b = r*K/32 + j) keeps its 32 values at q[32*b] and its scale at d[b];
//      q = data, d = data + M*K (2-byte halves)

static __global__ void q8p_pack_kernel(const block_q8_0 * __restrict__ src, int8_t * __restrict__ q, half * __restrict__ d,
                                       const int64_t nb) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) {
        return;
    }
    const uint16_t * s = (const uint16_t *) (src + b);   // 34-byte blocks: 2-byte aligned
    uint32_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = (uint32_t) s[1 + 2*i] | ((uint32_t) s[2 + 2*i] << 16);
    }
    uint4 * o = (uint4 *) (q + 32*b);
    o[0] = make_uint4(v[0], v[1], v[2], v[3]);
    o[1] = make_uint4(v[4], v[5], v[6], v[7]);
    d[b] = src[b].d;
}

static __global__ void q8p_unpack_kernel(const int8_t * __restrict__ q, const half * __restrict__ d, block_q8_0 * __restrict__ dst,
                                         const int64_t nb) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) {
        return;
    }
    const uint4 * in = (const uint4 *) (q + 32*b);
    const uint4 a0 = in[0], a1 = in[1];
    const uint32_t v[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w };
    uint16_t * o = (uint16_t *) (dst + b);
    o[0] = __half_as_ushort(d[b]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        o[1 + 2*i] = (uint16_t) (v[i] & 0xFFFF);
        o[2 + 2*i] = (uint16_t) (v[i] >> 16);
    }
}

static int64_t q8p_nblocks(const ggml_tensor * w) {
    return w->ne[0] * w->ne[1] / QK8_0;
}

static void q8p_planes(const ggml_tensor * w, int8_t *& q, half *& d) {
    q = (int8_t *) w->data;
    d = (half *) ((char *) w->data + w->ne[0] * w->ne[1]);
}

// rows per chunk of the in-place conversions: about 16 MiB of blocks
static int64_t q8p_chunk_rows(const ggml_tensor * w) {
    return std::max<int64_t>(1, (16 << 20) / (int64_t) ggml_row_size(GGML_TYPE_Q8_0, w->ne[0]));
}

static __global__ void q8p_scales_kernel(const block_q8_0 * __restrict__ src, half * __restrict__ d, const int64_t nb) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) {
        d[b] = src[b].d;
    }
}

// blocks -> planes in row order: chunk c writes values below K*r1 and reads blocks from K*r0*34/32 up, so no later
// chunk's blocks are overwritten before they are copied out; the scales go in last (they cover the last rows' blocks)
void ggml_cuda_q8p_pack_inplace(ggml_tensor * w, cudaStream_t stream) {
    const int64_t K = w->ne[0], M = w->ne[1], nb = q8p_nblocks(w);
    const int64_t R = std::min(M, q8p_chunk_rows(w));
    const size_t  row_bytes = ggml_row_size(GGML_TYPE_Q8_0, K);
    half * d_tmp = nullptr;
    block_q8_0 * b_tmp = nullptr;
    CUDA_CHECK(cudaMallocAsync((void **) &d_tmp, nb * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync((void **) &b_tmp, R * row_bytes, stream));
    q8p_scales_kernel<<<(nb + 255) / 256, 256, 0, stream>>>((const block_q8_0 *) w->data, d_tmp, nb);
    CUDA_CHECK(cudaGetLastError());
    int8_t * q; half * d;
    q8p_planes(w, q, d);
    half * d_junk = d_tmp;   // the pack kernel also writes scales: into the scratch copy (same values)
    for (int64_t r0 = 0; r0 < M; r0 += R) {
        const int64_t rn = std::min(R, M - r0), cb = rn * K / QK8_0;
        CUDA_CHECK(cudaMemcpyAsync(b_tmp, (const char *) w->data + r0 * row_bytes, rn * row_bytes, cudaMemcpyDeviceToDevice, stream));
        q8p_pack_kernel<<<(cb + 255) / 256, 256, 0, stream>>>(b_tmp, q + r0 * K, d_junk + r0 * K / QK8_0, cb);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpyAsync(d, d_tmp, nb * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaFreeAsync(b_tmp, stream));
    CUDA_CHECK(cudaFreeAsync(d_tmp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    g_q8p_generation.fetch_add(1, std::memory_order_acq_rel);
}

void ggml_cuda_q8p_unpack_to(const ggml_tensor * w, void * dst, cudaStream_t stream) {
    int8_t * q; half * d;
    q8p_planes(w, q, d);
    const int64_t nb = q8p_nblocks(w);
    q8p_unpack_kernel<<<(nb + 255) / 256, 256, 0, stream>>>(q, d, (block_q8_0 *) dst, nb);
    CUDA_CHECK(cudaGetLastError());
}

// planes -> blocks in reverse row order: chunk c writes blocks from K*r0*34/32 up, over values of rows >= r0 (its own,
// copied out first, and later rows', already done); the scales are copied out before anything is written
void ggml_cuda_q8p_unpack_inplace(ggml_tensor * w, cudaStream_t stream) {
    const int64_t K = w->ne[0], M = w->ne[1], nb = q8p_nblocks(w);
    const int64_t R = std::min(M, q8p_chunk_rows(w));
    const size_t  row_bytes = ggml_row_size(GGML_TYPE_Q8_0, K);
    int8_t * q; half * d;
    q8p_planes(w, q, d);
    half * d_tmp = nullptr;
    int8_t * q_tmp = nullptr;
    CUDA_CHECK(cudaMallocAsync((void **) &d_tmp, nb * sizeof(half), stream));
    CUDA_CHECK(cudaMallocAsync((void **) &q_tmp, R * K, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tmp, d, nb * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    const int64_t n_chunks = (M + R - 1) / R;
    for (int64_t c = n_chunks - 1; c >= 0; --c) {
        const int64_t r0 = c * R, rn = std::min(R, M - r0), cb = rn * K / QK8_0;
        CUDA_CHECK(cudaMemcpyAsync(q_tmp, q + r0 * K, rn * K, cudaMemcpyDeviceToDevice, stream));
        q8p_unpack_kernel<<<(cb + 255) / 256, 256, 0, stream>>>(q_tmp, d_tmp + r0 * K / QK8_0,
                (block_q8_0 *) ((char *) w->data + r0 * row_bytes), cb);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaFreeAsync(q_tmp, stream));
    CUDA_CHECK(cudaFreeAsync(d_tmp, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    g_q8p_generation.fetch_add(1, std::memory_order_acq_rel);
}

// ---- decode matvec: a warp per row; block b of the grid belongs to member s with s0[s] <= b < s0[s + 1]

struct q8p_group {
    const int8_t * q[GGML_CUDA_Q8P_GRP_MAX];
    const half   * d[GGML_CUDA_Q8P_GRP_MAX];
    const float  * x[GGML_CUDA_Q8P_GRP_MAX];
    float        * y[GGML_CUDA_Q8P_GRP_MAX];
    int M[GGML_CUDA_Q8P_GRP_MAX];
    int ldx[GGML_CUDA_Q8P_GRP_MAX];    // floats between activation columns
    int ldy[GGML_CUDA_Q8P_GRP_MAX];    // floats between output columns
    int s0[GGML_CUDA_Q8P_GRP_MAX + 1];
    int n;
    int K;
};

template <int NT, int WARPS>
static __global__ void __launch_bounds__(WARPS * WARP_SIZE) q8p_mmv_kernel(const q8p_group g) {
    const int b = blockIdx.x;
    int s = 0;
#pragma unroll
    for (int j = 1; j < GGML_CUDA_Q8P_GRP_MAX; ++j) {
        if (j < g.n && b >= g.s0[j]) {
            s = j;
        }
    }
    const int lane = threadIdx.x % WARP_SIZE;
    const int row  = (b - g.s0[s]) * WARPS + threadIdx.x / WARP_SIZE;
    if (row >= g.M[s]) {
        return;
    }
    const int K = g.K;
    const uint32_t * qr = (const uint32_t *) (g.q[s] + (size_t) row * K);
    const half     * dr = g.d[s] + (size_t) row * (K / QK8_0);
    const float4   * x4 = (const float4 *) g.x[s];
    const int ldx4 = g.ldx[s] / 4;

    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        acc[t] = 0.0f;
    }
    const int n_it = K / 128;
#pragma unroll 4
    for (int it = 0; it < n_it; ++it) {
        const uint32_t w  = __ldg(qr + it * WARP_SIZE + lane);
        const float    dv = __half2float(dr[it * 4 + lane / 8]);
        const float w0 = (float) (int8_t) (w & 0xFF);
        const float w1 = (float) (int8_t) ((w >> 8) & 0xFF);
        const float w2 = (float) (int8_t) ((w >> 16) & 0xFF);
        const float w3 = (float) (int8_t) (w >> 24);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 xv = x4[(size_t) t * ldx4 + it * WARP_SIZE + lane];
            acc[t] += dv * (w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float a = warp_reduce_sum(acc[t]);
        if (lane == 0) {
            g.y[s][(size_t) t * g.ldy[s] + row] = a;
        }
    }
}

template <int NT>
static void q8p_launch(const q8p_group & g, cudaStream_t stream) {
    constexpr int WARPS = 4;
    q8p_mmv_kernel<NT, WARPS><<<g.s0[g.n], WARPS * WARP_SIZE, 0, stream>>>(g);
}

bool ggml_cuda_q8p_mmv(ggml_backend_cuda_context & ctx, ggml_tensor * const * dsts, int n) {
    if (n < 1 || n > GGML_CUDA_Q8P_GRP_MAX) {
        return false;
    }
    constexpr int WARPS = 4;
    q8p_group g = {};
    g.n = n;
    g.K = (int) dsts[0]->src[0]->ne[0];
    const int64_t N = dsts[0]->src[1]->ne[1];
    if (N < 1 || N > 8) {
        return false;
    }
    int blocks = 0;
    for (int j = 0; j < n; ++j) {
        const ggml_tensor * dst = dsts[j];
        const ggml_tensor * w   = dst->src[0];
        const ggml_tensor * x   = dst->src[1];
        if (!ggml_cuda_q8p_is(w) || w->ne[0] != g.K || x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
                x->ne[0] != g.K || x->ne[1] != N || x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float) ||
                x->nb[1] % 16 != 0 || ((uintptr_t) x->data % 16) != 0 ||
                dst->ne[0] != w->ne[1] || dst->ne[1] != N || dst->nb[0] != sizeof(float) || w->ne[1] > INT_MAX / 2) {
            return false;
        }
        int8_t * q; half * d;
        q8p_planes(w, q, d);
        g.q[j]   = q;
        g.d[j]   = d;
        g.x[j]   = (const float *) x->data;
        g.y[j]   = (float *) dst->data;
        g.M[j]   = (int) w->ne[1];
        g.ldx[j] = (int) (x->nb[1] / sizeof(float));
        g.ldy[j] = (int) (dst->nb[1] / sizeof(float));
        g.s0[j]  = blocks;
        blocks  += (g.M[j] + WARPS - 1) / WARPS;
    }
    g.s0[n] = blocks;
    cudaStream_t stream = ctx.stream();
    switch (N) {
        case 1: q8p_launch<1>(g, stream); break;
        case 2: q8p_launch<2>(g, stream); break;
        case 3: q8p_launch<3>(g, stream); break;
        case 4: q8p_launch<4>(g, stream); break;
        case 5: q8p_launch<5>(g, stream); break;
        case 6: q8p_launch<6>(g, stream); break;
        case 7: q8p_launch<7>(g, stream); break;
        case 8: q8p_launch<8>(g, stream); break;
        default: return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
