// KVarN fused write rotation (llamAmpere #139, GGML_KVARN_FUSED_ROT).
//
// One launch replaces GGML_OP_TURBO_WHT (forward, group 256 plain or group 128 signed) followed by the
// TQ6_0 SET_ROWS in the input basis (ggml_set_rows_tq6_rotated) and its f16 sink mirror. The bytes are
// identical to that unfused pair: the butterfly network, stage order, operand order and scaling match
// k_turbo_wht_f32_plain256 / k_turbo_wht_f32_fast (turbo-wht.cu), and the TQ6 pack below is the
// already_rotated branch of k_set_rows_tq6 (set-rows.cu) unchanged, one element per thread.
//
// Layout: one CUDA block per GROUP-element slice of a source row, GROUP threads, element t on thread t.
// The slice covers GROUP/128 TQ6 blocks; threads [128b, 128b+128) pack TQ6 block b of the slice.

#include "set-rows-kvarn-rot.cuh"
#include "turbo-quant.cuh"

template <typename idx_t, int GROUP>
__launch_bounds__(GROUP)
static __global__ void k_set_rows_tq6_kvarn_rot(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        char        * __restrict__ dst,
        half        * __restrict__ sink,
        const int     sink_rows,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {
    static_assert(GROUP == 128 || GROUP == 256, "KVarN rotation group must be 128 or 256");
    constexpr int NB = GROUP / QK_TQ6;  // TQ6 blocks per slice

    const int t    = threadIdx.x;
    const int lane = t & 31;
    const int b    = t / QK_TQ6;        // TQ6 block of the slice
    const int j    = t % QK_TQ6;        // element within that block

    const int64_t n_grp_per_row = ne00 / GROUP;
    const int64_t g     = blockIdx.x;
    const int64_t i_grp = g % n_grp_per_row;
    int64_t       tmp   = g / n_grp_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne02;
    const int64_t i03   = tmp / ne02;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03 + i_grp*GROUP;
    block_tq6_0 * blk = (block_tq6_0 *) (dst + dst_row*s1 + i02*s2 + i03*s3) + i_grp*NB + b;

    // ---- KVarN rotation (forward) ----
    __shared__ float x[GROUP];
    float val = src_row[t];
    if constexpr (GROUP == 128) {
        val *= TURBO_WHT_SIGNS1[t];  // exact: a multiply by +-1 equals the sign flip of the fast kernel
    }

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
            const float a = x[t], c = x[t + h];
            x[t]     = a + c;
            x[t + h] = a - c;
        }
        __syncthreads();
    }

    float r;
    if constexpr (GROUP == 256) {
        constexpr float inv_sqrt_256 = 0.0625f;
        r = x[t]*inv_sqrt_256;
    } else {
        constexpr float inv_sqrt_128 = 0.08838834764831845f;
        r = x[t]*inv_sqrt_128*TURBO_WHT_SIGNS2[t];
    }

    // ---- f16 sink mirror of the rotated row (k_set_rows_f16_sink on the unfused path) ----
    if (sink != nullptr && dst_row >= 0 && dst_row < sink_rows) {
        sink[dst_row*ne00 + i_grp*GROUP + t] = __float2half(r);
    }

    // ---- TQ6 pack in the rotated basis (k_set_rows_tq6<idx_t, true>, per 128-element block) ----
    constexpr int n_warps = QK_TQ6 / WARP_SIZE;  // = 4 per TQ6 block
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

template <typename idx_t>
static void set_rows_cuda_tq6_kvarn_rot(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
        const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int group = ggml_get_op_params_i32(dst, 1) == GGML_SET_ROWS_KVARN_ROT256 ? 256 : 128;
    GGML_ASSERT(ne00 % group == 0);
    GGML_ASSERT(nb00 == sizeof(float));

    half * sink = nullptr;
    const int sink_rows = ggml_get_op_params_i32(dst, 2);
    if (sink_rows > 0) {
        GGML_ASSERT(ne02 == 1 && ne03 == 1);
        const size_t offset = (size_t) ggml_get_op_params_i32(dst, 3)*nb1;
        GGML_ASSERT(offset + (size_t) sink_rows*ne00*sizeof(half) <= ggml_nbytes(dst));
        sink = (half *) ((char *) dst->data + offset);
    }

    const int64_t n_slices = (ne00/group)*ne01*ne02*ne03;
    if (n_slices == 0) {
        return;
    }

    const float * src0_d = (const float *) src0->data;
    const idx_t * src1_d = (const idx_t *) src1->data;
    cudaStream_t stream = ctx.stream();

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    if (group == 256) {
        k_set_rows_tq6_kvarn_rot<idx_t, 256><<<(int) n_slices, 256, 0, stream>>>(
            src0_d, src1_d, (char *) dst->data, sink, sink_rows, ne00, ne01, ne02, ne11, ne12,
            s01, s02, s03, s10, s11, s12, nb1, nb2, nb3);
    } else {
        k_set_rows_tq6_kvarn_rot<idx_t, 128><<<(int) n_slices, 128, 0, stream>>>(
            src0_d, src1_d, (char *) dst->data, sink, sink_rows, ne00, ne01, ne02, ne11, ne12,
            s01, s02, s03, s10, s11, s12, nb1, nb2, nb3);
    }
}

void ggml_cuda_set_rows_kvarn_rot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_TQ6_0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda_tq6_kvarn_rot<int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda_tq6_kvarn_rot<int32_t>(ctx, src0, src1, dst);
    }
}
