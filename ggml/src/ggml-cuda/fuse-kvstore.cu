#include "fuse-kvstore.cuh"
#include "set-rows-sjkvarn-rot.cuh"
#include "turbo-quant.cuh"

#define KVS_GROUP 256                   // rotation group of the store, threads per block
#define KVS_WARPS (KVS_GROUP/WARP_SIZE) // Q rotation groups per block (one per warp)

struct kvs_store_args {
    const float * src0;
    const void  * src1;
    char        * dst;
    char        * sink;
    int           sink_rows;
    int64_t       sink_rows_per_stream;
    size_t        sink_stream_bytes;
    int64_t       ne00, ne01, ne02, ne11, ne12;
    int64_t       s01, s02, s03, s10, s11, s12;
    int64_t       s1, s2, s3;
};

struct kvs_wht_args {
    const float * src;
    float       * dst;
    int64_t       n_groups;
    int64_t       head_dim;
    int64_t       groups_per_head;
};

// k_set_rows_tq6_sj_kvarn_rot<idx_t, 256> (set-rows-sjkvarn-rot.cu) for slice g
template <typename idx_t>
static __device__ __forceinline__ void kvs_store(const kvs_store_args & a, const int64_t g) {
    constexpr int GROUP = KVS_GROUP;
    constexpr int NB    = GROUP / QK_TQ6;

    const int t    = threadIdx.x;
    const int lane = t & 31;
    const int b    = t / QK_TQ6;
    const int j    = t % QK_TQ6;

    const int64_t n_grp_per_row = a.ne00 / GROUP;
    const int64_t i_grp = g % n_grp_per_row;
    int64_t       tmp   = g / n_grp_per_row;
    const int64_t i01   = tmp % a.ne01;
    tmp                 = tmp / a.ne01;
    const int64_t i02   = tmp % a.ne02;
    const int64_t i03   = tmp / a.ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % a.ne11;
    const int64_t i12 = i03 % a.ne12;

    const idx_t * src1 = (const idx_t *) a.src1;
    const int64_t dst_row = *(src1 + i10*a.s10 + i11*a.s11 + i12*a.s12);
    const float * src_row = a.src0 + i01*a.s01 + i02*a.s02 + i03*a.s03 + i_grp*GROUP;
    block_tq6_0 * blk = (block_tq6_0 *) (a.dst + dst_row*a.s1 + i02*a.s2 + i03*a.s3) + i_grp*NB + b;

    __shared__ float x[GROUP];
    float val = src_row[t];

#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }
    x[t] = val;
    __syncthreads();

#pragma unroll
    for (int h = 32; h < GROUP; h <<= 1) {
        if (t % (2*h) < h) {
            const float p = x[t], c = x[t + h];
            x[t]     = p + c;
            x[t + h] = p - c;
        }
        __syncthreads();
    }

    constexpr float inv_sqrt_256 = 0.0625f;
    const float r = x[t]*inv_sqrt_256;

    if (a.sink != nullptr && dst_row >= 0) {
        int64_t srow = dst_row;
        char * sbase = a.sink;
        if (a.sink_rows_per_stream > 0) {
            const int64_t s = srow/a.sink_rows_per_stream;
            srow  -= s*a.sink_rows_per_stream;
            sbase += s*a.sink_stream_bytes;
        }
        if (srow < a.sink_rows) {
            ((half *) sbase)[srow*a.ne00 + i_grp*GROUP + t] = __float2half(r);
        }
    }

    constexpr int n_warps = QK_TQ6 / WARP_SIZE;
    __shared__ float warp_accum[NB*n_warps];
    __shared__ float warp_accum_rc[NB*n_warps];
    __shared__ float s_norm_sq[NB];
    __shared__ float s_recon_sq[NB];

    float v2 = r * r;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[t / WARP_SIZE] = v2;
    __syncthreads();

    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[b*n_warps + w];
        s_norm_sq[b] = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq[b]);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    const float rv = r * inv_norm;
    const uint8_t idx = tq6_nearest_centroid(rv);

    const uint8_t my_nibble = idx & 0xF;
    const uint8_t my_high   = (idx >> 4) & 0x3;
    const uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1, WARP_SIZE);
    if (j % 2 == 0) {
        blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
    }

    const int quad = lane & ~3;
    const uint8_t h0 = __shfl_sync(0xffffffff, my_high, quad + 0, WARP_SIZE);
    const uint8_t h1 = __shfl_sync(0xffffffff, my_high, quad + 1, WARP_SIZE);
    const uint8_t h2 = __shfl_sync(0xffffffff, my_high, quad + 2, WARP_SIZE);
    const uint8_t h3 = __shfl_sync(0xffffffff, my_high, quad + 3, WARP_SIZE);
    if (j % 4 == 0) {
        blk->qh[j / 4] = h0 | (h1 << 2) | (h2 << 4) | (h3 << 6);
    }

    const float c = TQ6_CENTROIDS[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum_rc[t / WARP_SIZE] = rc;
    __syncthreads();

    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum_rc[b*n_warps + w];
        s_recon_sq[b] = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq[b]);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
}

// k_turbo_wht_f32_plain256 (turbo-wht.cu) for group g, on one warp
static __device__ __forceinline__ void kvs_wht(const kvs_wht_args & a, const int64_t g) {
    const int lane = threadIdx.x & 31;
    if (g >= a.n_groups) return;

    const int64_t head_idx    = g / a.groups_per_head;
    const int64_t grp_in_head = g % a.groups_per_head;
    const int64_t base        = head_idx * a.head_dim + grp_in_head * 256;

    float x[8];
    {
        const float4 p = *((const float4 *) (a.src + base) + 2*lane);
        const float4 q = *((const float4 *) (a.src + base) + 2*lane + 1);
        x[0] = p.x; x[1] = p.y; x[2] = p.z; x[3] = p.w;
        x[4] = q.x; x[5] = q.y; x[6] = q.z; x[7] = q.w;
    }

#pragma unroll
    for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
        for (int i = 0; i < 8; i += 2*h) {
#pragma unroll
            for (int j = i; j < i + h; ++j) {
                const float p = x[j], q = x[j + h];
                x[j]     = p + q;
                x[j + h] = p - q;
            }
        }
    }

#pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p = __shfl_xor_sync(0xffffffff, x[j], m);
            x[j] = hi ? p - x[j] : x[j] + p;
        }
    }

    constexpr float inv_sqrt = 0.0625f;
    float4 p, q;
    p.x = x[0]*inv_sqrt; p.y = x[1]*inv_sqrt; p.z = x[2]*inv_sqrt; p.w = x[3]*inv_sqrt;
    q.x = x[4]*inv_sqrt; q.y = x[5]*inv_sqrt; q.z = x[6]*inv_sqrt; q.w = x[7]*inv_sqrt;
    *((float4 *) (a.dst + base) + 2*lane)     = p;
    *((float4 *) (a.dst + base) + 2*lane + 1) = q;
}

// blocks [0, nk): K slices, [nk, nk + nv): V slices, then KVS_WARPS Q groups per block (the branch is block-uniform)
template <typename idx_t>
__launch_bounds__(KVS_GROUP)
static __global__ void k_kvstore(const kvs_store_args k, const kvs_store_args v, const kvs_wht_args q, const int nk,
        const int nv) {
    const int blk = blockIdx.x;
    if (blk < nk) {
        kvs_store<idx_t>(k, blk);
    } else if (blk < nk + nv) {
        kvs_store<idx_t>(v, blk - nk);
    } else {
        kvs_wht(q, (int64_t) (blk - nk - nv)*KVS_WARPS + threadIdx.x/WARP_SIZE);
    }
}

bool ggml_cuda_kvstore_wht_ok(const ggml_tensor * wht) {
    const ggml_tensor * src = wht->src[0];
    int group_size;
    memcpy(&group_size, wht->op_params + sizeof(int), sizeof(int));
    return wht->op == GGML_OP_TURBO_WHT && wht->src[1] == nullptr && group_size == 256 &&
        src->type == GGML_TYPE_F32 && wht->type == GGML_TYPE_F32 && ggml_is_contiguous(src) &&
        ggml_is_contiguous(wht) && src->ne[0] % 256 == 0 && ggml_nelements(src) > 0 &&
        (!src->data || ((uintptr_t) src->data | (uintptr_t) wht->data) % 16 == 0);
}

bool ggml_cuda_kvstore_store_ok(const ggml_tensor * st) {
    const ggml_tensor * src0 = st->src[0];
    const ggml_tensor * src1 = st->src[1];
    if (st->op != GGML_OP_SET_ROWS || st->type != GGML_TYPE_TQ6_0 ||
            ggml_get_op_params_i32(st, 1) != GGML_SET_ROWS_SJKVARN_ROT256 || src0->type != GGML_TYPE_F32 ||
            (src1->type != GGML_TYPE_I64 && src1->type != GGML_TYPE_I32) || src0->nb[0] != sizeof(float) ||
            src0->ne[0] % KVS_GROUP != 0 || ggml_nelements(src0) == 0) {
        return false;
    }
    const int sink_rows = ggml_get_op_params_i32(st, 2);
    const int64_t sink_rows_per_stream = ggml_get_op_params_i32(st, 4);
    if (sink_rows > 0) {
        // the unfused launcher asserts these
        const size_t offset = (size_t) ggml_get_op_params_i32(st, 3)*st->nb[1];
        const size_t bytes  = offset + (size_t) sink_rows*src0->ne[0]*sizeof(half);
        if (src0->ne[2] != 1 || src0->ne[3] != 1 || bytes > ggml_nbytes(st) || (sink_rows_per_stream != 0 &&
                (bytes > (size_t) sink_rows_per_stream*st->nb[1] || (sink_rows_per_stream*st->nb[1]) % sizeof(half) != 0))) {
            return false;
        }
    }
    return true;
}

static kvs_store_args kvs_args(const ggml_tensor * st, const size_t idx_size) {
    const ggml_tensor * src0 = st->src[0];
    const ggml_tensor * src1 = st->src[1];
    kvs_store_args a;
    a.src0 = (const float *) src0->data;
    a.src1 = src1->data;
    a.dst  = (char *) st->data;
    a.sink_rows            = ggml_get_op_params_i32(st, 2);
    a.sink_rows_per_stream = ggml_get_op_params_i32(st, 4);
    a.sink = a.sink_rows > 0 ? (char *) st->data + (size_t) ggml_get_op_params_i32(st, 3)*st->nb[1] : nullptr;
    a.sink_stream_bytes = (size_t) a.sink_rows_per_stream*st->nb[1];
    a.ne00 = src0->ne[0]; a.ne01 = src0->ne[1]; a.ne02 = src0->ne[2];
    a.ne11 = src1->ne[1]; a.ne12 = src1->ne[2];
    a.s01 = src0->nb[1]/sizeof(float); a.s02 = src0->nb[2]/sizeof(float); a.s03 = src0->nb[3]/sizeof(float);
    a.s10 = src1->nb[0]/idx_size; a.s11 = src1->nb[1]/idx_size; a.s12 = src1->nb[2]/idx_size;
    a.s1 = st->nb[1]; a.s2 = st->nb[2]; a.s3 = st->nb[3];
    return a;
}

bool ggml_cuda_kvstore_launch(ggml_backend_cuda_context & ctx, ggml_tensor * wht, ggml_tensor * sk, ggml_tensor * sv) {
    const bool i64 = sk->src[1]->type == GGML_TYPE_I64;
    GGML_ASSERT(sv->src[1]->type == sk->src[1]->type);
    const size_t idx_size = i64 ? sizeof(int64_t) : sizeof(int32_t);

    const int64_t nk = ggml_nelements(sk->src[0])/KVS_GROUP;
    const int64_t nv = ggml_nelements(sv->src[0])/KVS_GROUP;

    kvs_wht_args q;
    q.src      = (const float *) wht->src[0]->data;
    q.dst      = (float *) wht->data;
    q.head_dim = wht->src[0]->ne[0];
    q.groups_per_head = q.head_dim/256;
    q.n_groups = q.groups_per_head*(ggml_nelements(wht->src[0])/q.head_dim);
    const int64_t nq = (q.n_groups + KVS_WARPS - 1)/KVS_WARPS;
    if (nk + nv + nq > INT_MAX) {
        return false;
    }

    const kvs_store_args ka = kvs_args(sk, idx_size);
    const kvs_store_args va = kvs_args(sv, idx_size);
    const int n_blocks = (int) (nk + nv + nq);
    cudaStream_t stream = ctx.stream();
    if (i64) {
        k_kvstore<int64_t><<<n_blocks, KVS_GROUP, 0, stream>>>(ka, va, q, (int) nk, (int) nv);
    } else {
        k_kvstore<int32_t><<<n_blocks, KVS_GROUP, 0, stream>>>(ka, va, q, (int) nk, (int) nv);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
