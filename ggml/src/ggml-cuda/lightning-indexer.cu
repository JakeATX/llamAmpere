#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)

static __global__ void lightning_indexer_kernel_prefill(
        const float * Q, const float * K, const float * W, const half * M, float * dst,
        int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw3,
        size_t nbm1, size_t nbm3, int64_t nem3) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int N_QUERY = 8;
    constexpr int N_HEAD = 4;
    constexpr int N_KEY = 64;
    constexpr int N_EMBD = 128;
    constexpr int N_EMBD_PAD = N_EMBD + 8;
    constexpr int N_WARP = 8;
    constexpr int N_THREAD = N_WARP * WARP_SIZE;

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int q_start = blockIdx.y * N_QUERY;
    const int k_start = blockIdx.x * N_KEY;
    const int stream = blockIdx.z;

    __shared__ __align__(32) half q_shared[N_QUERY * N_HEAD][N_EMBD_PAD];
    __shared__ __align__(32) half k_shared[N_KEY][N_EMBD_PAD];
    __shared__ __align__(32) float qk_shared[N_QUERY * N_HEAD][N_KEY];
    __shared__ float w_shared[N_QUERY][N_HEAD];

    bool out_of_range = false;
    for (int i = tid; i < N_QUERY * N_HEAD * N_EMBD / 4; i += N_THREAD) {
        const int row = i / (N_EMBD / 4);
        const int embd = i % (N_EMBD / 4) * 4;
        const int query = q_start + row / N_HEAD;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (query < n_batch) {
            v = *(const float4 *) ((const char *) Q + stream * nbq3 + query * nbq2 + (row % N_HEAD) * nbq1 + embd * sizeof(float));
        }
        out_of_range |= !(fabsf(v.x) <= 65504.0f && fabsf(v.y) <= 65504.0f && fabsf(v.z) <= 65504.0f && fabsf(v.w) <= 65504.0f);
        half2 * out = (half2 *) &q_shared[row][embd];
        out[0] = __float22half2_rn(make_float2(v.x, v.y));
        out[1] = __float22half2_rn(make_float2(v.z, v.w));
    }

    for (int i = tid; i < N_KEY * N_EMBD / 4; i += N_THREAD) {
        const int key = i / (N_EMBD / 4);
        const int embd = i % (N_EMBD / 4) * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (k_start + key < n_kv) {
            v = *(const float4 *) ((const char *) K + stream * nbk3 + (k_start + key) * nbk2 + embd * sizeof(float));
        }
        out_of_range |= !(fabsf(v.x) <= 65504.0f && fabsf(v.y) <= 65504.0f && fabsf(v.z) <= 65504.0f && fabsf(v.w) <= 65504.0f);
        half2 * out = (half2 *) &k_shared[key][embd];
        out[0] = __float22half2_rn(make_float2(v.x, v.y));
        out[1] = __float22half2_rn(make_float2(v.z, v.w));
    }

    if (tid < N_QUERY * N_HEAD) {
        const int query = q_start + tid / N_HEAD;
        float weight = 0.0f;
        if (query < n_batch) {
            const float * weights = (const float *) ((const char *) W + stream * nbw3 + query * nbw1);
            weight = weights[tid % N_HEAD];
        }
        w_shared[tid / N_HEAD][tid % N_HEAD] = weight;
    }

    // Keep the float32 path's arithmetic when conversion would overflow or receive nonfinite inputs.
    if (__syncthreads_or(out_of_range)) {
        for (int i = threadIdx.y; i < N_QUERY * N_KEY; i += N_WARP) {
            const int query = q_start + i / N_KEY;
            const int key = k_start + i % N_KEY;
            if (query >= n_batch || key >= n_kv) {
                continue;
            }
            const float4 k = *(const float4 *) ((const char *) K + stream * nbk3 + key * nbk2 + threadIdx.x * sizeof(float4));
            float score = 0.0f;
            for (int head = 0; head < N_HEAD; ++head) {
                const float4 q = *(const float4 *) ((const char *) Q + stream * nbq3 + query * nbq2 + head * nbq1 + threadIdx.x * sizeof(float4));
                float dot = 0.0f;
                ggml_cuda_mad(dot, q.x, k.x);
                ggml_cuda_mad(dot, q.y, k.y);
                ggml_cuda_mad(dot, q.z, k.z);
                ggml_cuda_mad(dot, q.w, k.w);
                dot = warp_reduce_sum(dot);
                if (threadIdx.x == 0) {
                    score += (dot > 0.0f ? dot : 0.0f) * w_shared[i / N_KEY][head];
                }
            }
            if (threadIdx.x == 0) {
                const half * mask = (const half *) ((const char *) M + (stream % nem3) * nbm3 + query * nbm1);
                float * out = (float *) ((char *) dst + stream * nb3 + query * nb1);
                out[key] = score + __half2float(mask[key]);
            }
        }
        return;
    }

    const int row = (threadIdx.y / 4) * 16;
    const int key = (threadIdx.y % 4) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

#pragma unroll
    for (int embd = 0; embd < N_EMBD; embd += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_q;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_k;
        wmma::load_matrix_sync(frag_q, &q_shared[row][embd], N_EMBD_PAD);
        wmma::load_matrix_sync(frag_k, &k_shared[key][embd], N_EMBD_PAD);
        wmma::mma_sync(acc, frag_q, frag_k, acc);
    }
    wmma::store_matrix_sync(&qk_shared[row][key], acc, N_KEY, wmma::mem_row_major);

    __syncthreads();

    for (int i = tid; i < N_QUERY * N_KEY; i += N_THREAD) {
        const int query = i / N_KEY;
        const int key_out = k_start + i % N_KEY;
        if (q_start + query < n_batch && key_out < n_kv) {
            float score = 0.0f;
#pragma unroll
            for (int head = 0; head < N_HEAD; ++head) {
                const float qk = qk_shared[query * N_HEAD + head][i % N_KEY];
                score += (qk > 0.0f ? qk : 0.0f) * w_shared[query][head];
            }
            const half * mask = (const half *) ((const char *) M + (stream % nem3) * nbm3 + (q_start + query) * nbm1);
            float * out = (float *) ((char *) dst + stream * nb3 + (q_start + query) * nb1);
            out[key_out] = score + __half2float(mask[key_out]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, W, M, dst, n_batch, n_kv, nb1, nb3,
        nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
    NO_DEVICE_CODE;
#endif
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// tokens scored per block by the tile kernel
#define LIGHTNING_INDEXER_TILE_TOKENS 8

// heads whose queries the tile kernel stages per pass, MUSA arch 21 caps static shared memory
// at 28 KB and the queries of four heads do not fit there next to the key tile
#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
#else
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
#endif

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// one block scores a tile of K_VECS_PER_BLOCK keys against TOKENS_PER_BLOCK tokens: the keys are
// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD
// keys for one token, a warp shares its token so the query reads are broadcasts, and every key
// element is widened once for all heads, so no dot product needs a cross thread reduction
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_tile(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TOKENS_PER_BLOCK  = LIGHTNING_INDEXER_TILE_TOKENS;
    constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
    constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;
    constexpr int N_EMBD_H2         = N_EMBD / 2;
    constexpr int HEADS_PER_PASS    = N_HEAD < LIGHTNING_INDEXER_TILE_HEADS_PER_PASS ? N_HEAD : LIGHTNING_INDEXER_TILE_HEADS_PER_PASS;

    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");
    static_assert(N_HEAD % HEADS_PER_PASS == 0, "head passes must cover the heads");

    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;
    const int i_stream    = blockIdx.z;

    // the row padding keeps the keys of consecutive threads in distinct banks
    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
    __shared__ float2 q_shared[HEADS_PER_PASS][TOKENS_PER_BLOCK][N_EMBD_H2];
    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];

    // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero

#pragma unroll
    for (int i = tid; i < K_VECS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int r  = i / (N_EMBD / 4);
        const int c4 = i % (N_EMBD / 4);

        half2 lo = make_half2(0.0f, 0.0f);
        half2 hi = lo;
        if (start_kv + r < n_kv) {
            const char * k_row = K + (start_kv + r)*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F16) {
                lo = ((const half2 *) k_row)[2*c4 + 0];
                hi = ((const half2 *) k_row)[2*c4 + 1];
            } else {
                float4 v;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    v = ((const float4 *) k_row)[c4];
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(k_row, &v, c4 * 4);
                }
                lo = make_half2(v.x, v.y);
                hi = make_half2(v.z, v.w);
            }
        }

        k_shared[r][2*c4 + 0] = lo;
        k_shared[r][2*c4 + 1] = hi;
    }

    // phase 2 - stage the weights of every head, tokens past n_batch are zero

    if (tid < N_HEAD * TOKENS_PER_BLOCK) {
        const int h = tid / TOKENS_PER_BLOCK;
        const int r = tid % TOKENS_PER_BLOCK;
        w_shared[h][r] = start_batch + r < n_batch ?
            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    const int kl = tid % KEY_LANES;
    const int tl = tid / KEY_LANES;

    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };

#pragma unroll
    for (int h0 = 0; h0 < N_HEAD; h0 += HEADS_PER_PASS) {
        // the previous pass is fully consumed before its queries are replaced
        if (h0 > 0) {
            __syncthreads();
        }

        // phase 3 - stage the queries of the heads of this pass, tokens past n_batch are zero

#pragma unroll
        for (int i = tid; i < HEADS_PER_PASS * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
            const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
            const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
            const int c4 = i % (N_EMBD / 4);

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (start_batch + r < n_batch) {
                v = *(const float4 *) ((const char *) Q + (h0 + h)*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
            }

            q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
            q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
        }

        __syncthreads();

        // phase 4 - float products of the widened keys for the heads of this pass

#pragma unroll 8
        for (int c = 0; c < N_EMBD_H2; ++c) {
            float2 k_val[KEYS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
            }
#pragma unroll
            for (int h = 0; h < HEADS_PER_PASS; ++h) {
                const float2 q_val = q_shared[h][tl][c];
#pragma unroll
                for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                    qk[h0 + h][j] = fmaf(k_val[j].x, q_val.x, qk[h0 + h][j]);
                    qk[h0 + h][j] = fmaf(k_val[j].y, q_val.y, qk[h0 + h][j]);
                }
            }
        }
    }

    // phase 5 - ReLU, weight, add the mask and write, consecutive threads write consecutive keys

    float score[KEYS_PER_THREAD] = { 0.0f };

#pragma unroll
    for (int h = 0; h < N_HEAD; ++h) {
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            score[j] += fmaxf(qk[h][j], 0.0f) * w_shared[h][tl];
        }
    }

    const int i_batch = start_batch + tl;
    if (i_batch >= n_batch) {
        return;
    }

    const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);

#pragma unroll
    for (int j = 0; j < KEYS_PER_THREAD; ++j) {
        const int i_kv = start_kv + kl + j*KEY_LANES;
        if (i_kv < n_kv) {
            dst_base[i_kv] = score[j] + __half2float(m_base[i_kv]);
        }
    }
}

// fork (opt-in GGML_CUDA_LID_EXACT=1): the tile kernel's arithmetic for float32 keys in a register-blocked layout.
// Same numbers bit for bit: each key element is rounded to half and widened back, each query-head-key product is
// one fmaf chain over the 128 elements in order, and the ReLU-weight sum and mask add are the same expressions.
// A block scores 32 tokens x 128 keys and each thread 2 tokens x 4 heads x 8 keys, so one shared load feeds
// 16 products (the tile kernel: 2-4) and each key tile is read once per 32 tokens (the tile kernel: per 8).
#define LIX_TOKENS 32
#define LIX_KEYS   128
#define LIX_CHUNK  32
#define LIX_LD     (128 + 4)

static __global__ void __launch_bounds__(256, 2) lightning_indexer_kernel_exact(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_batch, int64_t n_kv, size_t nb1, size_t nb3, size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk2, size_t nbk3, size_t nbw1, size_t nbw3, size_t nbm1, size_t nbm3, int64_t nem3) {
    constexpr int N_HEAD = 4;
    constexpr int N_EMBD = 128;

    // rows of q_s are query rows (token-major, head-minor), rows of k_s are keys; both stored element-major
    __shared__ __align__(16) float q_s[LIX_CHUNK][LIX_LD];
    __shared__ __align__(16) float k_s[LIX_CHUNK][LIX_LD];
    __shared__ float w_s[LIX_TOKENS][N_HEAD];

    const int tid = threadIdx.x;
    const int tx  = tid % 16; // keys 4*tx..4*tx+3 and 64+4*tx..64+4*tx+3
    const int ty  = tid / 16; // tokens 2*ty and 2*ty+1, query rows 8*ty..8*ty+7

    const int64_t start_kv    = (int64_t) blockIdx.x * LIX_KEYS;
    const int64_t start_batch = (int64_t) blockIdx.y * LIX_TOKENS;
    const int     stream      = blockIdx.z;

    if (tid < LIX_TOKENS * N_HEAD) {
        const int64_t t = start_batch + tid / N_HEAD;
        w_s[tid / N_HEAD][tid % N_HEAD] = t < n_batch ? ((const float *) ((const char *) W + t*nbw1 + stream*nbw3))[tid % N_HEAD] : 0.0f;
    }

    float acc[8][8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc[r][j] = 0.0f;
        }
    }

    for (int e0 = 0; e0 < N_EMBD; e0 += LIX_CHUNK) {
        if (e0 > 0) {
            __syncthreads();
        }
        // a warp stages 32 consecutive rows at one 4-element column, so the transposed stores hit distinct banks
#pragma unroll
        for (int it = 0; it < LIX_CHUNK / 8; ++it) {
            const int r  = tid % 128;
            const int c  = 4 * (tid / 128 + 2 * it);
            const int64_t t = start_batch + r / N_HEAD;
            float4 qv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t < n_batch) {
                qv = *(const float4 *) ((const char *) Q + stream*nbq3 + t*nbq2 + (r % N_HEAD)*nbq1 + (e0 + c)*sizeof(float));
            }
            q_s[c + 0][r] = qv.x;
            q_s[c + 1][r] = qv.y;
            q_s[c + 2][r] = qv.z;
            q_s[c + 3][r] = qv.w;
            float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (start_kv + r < n_kv) {
                kv = *(const float4 *) (K + stream*nbk3 + (start_kv + r)*nbk2 + (e0 + c)*sizeof(float));
            }
            k_s[c + 0][r] = __half2float(__float2half(kv.x));
            k_s[c + 1][r] = __half2float(__float2half(kv.y));
            k_s[c + 2][r] = __half2float(__float2half(kv.z));
            k_s[c + 3][r] = __half2float(__float2half(kv.w));
        }
        __syncthreads();

#pragma unroll 4
        for (int e = 0; e < LIX_CHUNK; ++e) {
            const float4 qa = *(const float4 *) &q_s[e][8*ty];
            const float4 qb = *(const float4 *) &q_s[e][8*ty + 4];
            const float4 ka = *(const float4 *) &k_s[e][4*tx];
            const float4 kb = *(const float4 *) &k_s[e][64 + 4*tx];
            const float q[8] = { qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w };
            const float k[8] = { ka.x, ka.y, ka.z, ka.w, kb.x, kb.y, kb.z, kb.w };
#pragma unroll
            for (int r = 0; r < 8; ++r) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[r][j] = fmaf(k[j], q[r], acc[r][j]);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const int64_t i_batch = start_batch + 2*ty + t;
        if (i_batch >= n_batch) {
            continue;
        }
        const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (stream % nem3)*nbm3);
        float * dst_base = (float *) ((char *) dst + i_batch*nb1 + stream*nb3);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int64_t i_kv = start_kv + (j < 4 ? 4*tx + j : 64 + 4*tx + j - 4);
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < N_HEAD; ++h) {
                score += fmaxf(acc[4*t + h][j], 0.0f) * w_s[2*ty + t][h];
            }
            if (i_kv < n_kv) {
                dst_base[i_kv] = score + __half2float(m_base[i_kv]);
            }
        }
    }
}

// GGML_CUDA_LID_EXACT=2: counts outputs whose bits differ between the exact kernel and the tile kernel
static __global__ void lightning_indexer_compare(const char * a, const char * b, int64_t n_kv, int64_t n_batch, int64_t n_stream,
        size_t nb1, size_t nb3, unsigned long long * n_diff) {
    const int64_t n = n_kv * n_batch * n_stream;
    unsigned long long d = 0;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t kv = i % n_kv, t = i / n_kv % n_batch, s = i / (n_kv * n_batch);
        const size_t off = s*nb3 + t*nb1 + kv*sizeof(float);
        d += *(const uint32_t *) (a + off) != *(const uint32_t *) (b + off);
    }
    if (d) {
        atomicAdd(n_diff, d);
    }
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // fork: keep reduced-precision prefill opt-in until model quality is measured.
        static const bool use_prefill = [] {
            const char * value = getenv("GGML_CUDA_LID_WMMA");
            return value != nullptr && value[0] == '1';
        }();
        if (use_prefill && n_batch >= 8 && k->type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc)) {
            const dim3 block(WARP_SIZE, 8);
            const dim3 grid(n_kv / 64 + (n_kv % 64 != 0), n_batch / 8 + (n_batch % 8 != 0), n_stream);
            lightning_indexer_kernel_prefill<<<grid, block, 0, ctx.stream()>>>(
                q_d, (const float *) k_d, w_d, m_d, dst_d, n_batch, n_kv, nb1, nb3,
                nbq1, nbq2, nbq3, nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
            return;
        }
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;
        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);

        // fork: opt-in register-blocked float32-key path with the tile kernel's exact arithmetic (2 = also check bits)
        static const int exact_mode = [] {
            const char * value = getenv("GGML_CUDA_LID_EXACT");
            return value != nullptr ? atoi(value) : 0;
        }();
        if (exact_mode > 0 && k->type == GGML_TYPE_F32 && GGML_CUDA_CC_IS_NVIDIA(cc)) {
            ggml_cuda_pool_alloc<char> ref(ctx.pool());
            if (exact_mode == 2) {
                ref.alloc(ggml_nbytes(dst));
                float * dst_tile = (float *) ref.get();
                lightning_indexer_kernel_tile<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, 128, 4, GGML_TYPE_F32><<<grid, block, 0, ctx.stream()>>>(
                    q_d, k_d, w_d, m_d, dst_tile, n_stream, n_batch, n_kv, nb1, nb2, nb3, nbq1, nbq2, nbq3,
                    nbk1, nbk2, nbk3, nbw1, nbw2, nbw3, nbm1, nbm2, nbm3, nem3);
            }
            const dim3 grid_x((n_kv + LIX_KEYS - 1) / LIX_KEYS, (n_batch + LIX_TOKENS - 1) / LIX_TOKENS, n_stream);
            lightning_indexer_kernel_exact<<<grid_x, 256, 0, ctx.stream()>>>(
                q_d, k_d, w_d, m_d, dst_d, n_batch, n_kv, nb1, nb3, nbq1, nbq2, nbq3,
                nbk2, nbk3, nbw1, nbw3, nbm1, nbm3, nem3);
            if (exact_mode == 2) {
                static unsigned long long n_calls = 0, n_diff_total = 0, n_out_total = 0;
                ggml_cuda_pool_alloc<unsigned long long> n_diff(ctx.pool(), 1);
                CUDA_CHECK(cudaMemsetAsync(n_diff.get(), 0, sizeof(unsigned long long), ctx.stream()));
                lightning_indexer_compare<<<512, 256, 0, ctx.stream()>>>(ref.get(), (const char *) dst_d, n_kv, n_batch, n_stream, nb1, nb3, n_diff.get());
                unsigned long long h_diff = 0;
                CUDA_CHECK(cudaMemcpyAsync(&h_diff, n_diff.get(), sizeof(h_diff), cudaMemcpyDeviceToHost, ctx.stream()));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
                n_calls++; n_diff_total += h_diff; n_out_total += (unsigned long long) n_kv * n_batch * n_stream;
                if (h_diff != 0 || (n_calls & (n_calls - 1)) == 0) {
                    ggml_log_internal(h_diff != 0 ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_INFO,
                        "%s: LID_EXACT check call %llu (n_batch %d n_kv %d): %llu differing outputs, total %llu of %llu\n",
                        __func__, n_calls, n_batch, n_kv, h_diff, n_diff_total, n_out_total);
                }
            }
            return;
        }

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else if (n_embd == 128 && n_head == 4) {
        // a batch smaller than a token tile, use vector kernel
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        dim3 grid(num_kv_blocks, n_batch, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32 && neq1 != 4) {
        return false;
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}
