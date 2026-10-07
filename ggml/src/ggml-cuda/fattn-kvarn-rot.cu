// KVarN fused rotation (fork-only, GGML_KVARN_FUSED_ROT): separate-pass fallback for the KVarN paths that do not
// rotate in-kernel (4/4 prefill expansion, 4/4 MMA tile kernel, fragment-direct kernel, unaligned Q, non-Ampere MMA
// tables). Same values as the graph-level GGML_OP_TURBO_WHT pair; it only removes the graph nodes.

#include "fattn-kvarn-rot.cuh"

#include <cstring>

static __global__ void k_kvarn_rot256_rows(
        const char * __restrict__ src, float * __restrict__ dst, const int64_t nrows, const int ne1, const int ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int lane = threadIdx.x;
    const int64_t r = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (r >= nrows) {
        return; // a row is a whole warp
    }
    const int64_t i1 = r % ne1, i2 = (r / ne1) % ne2, i3 = r / ((int64_t) ne1*ne2);
    const float * s = (const float *) (src + i1*nb1 + i2*nb2 + i3*nb3) + 8*lane;
    float x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        x[j] = s[j];
    }
    kvarn_rot256_row8(x, lane);
    float * d = dst + r*256 + 8*lane;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        d[j] = x[j];
    }
}

static void kvarn_rot256_launch(const char * src, float * dst, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const size_t nb1, const size_t nb2, const size_t nb3, cudaStream_t stream) {
    const int64_t nrows = ne1*ne2*ne3;
    if (nrows == 0) {
        return;
    }
    constexpr int warps = 4;
    const dim3 block(WARP_SIZE, warps, 1);
    const dim3 grid((unsigned) ((nrows + warps - 1)/warps), 1, 1);
    k_kvarn_rot256_rows<<<grid, block, 0, stream>>>(src, dst, nrows, (int) ne1, (int) ne2, nb1, nb2, nb3);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_kvarn_rot256_q_pass(ggml_backend_cuda_context & ctx, const ggml_tensor * Q, float * q_rot, ggml_tensor * q2) {
    GGML_ASSERT(Q->type == GGML_TYPE_F32 && Q->ne[0] == 256 && Q->nb[0] == sizeof(float));
    kvarn_rot256_launch((const char *) Q->data, q_rot, Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3], ctx.stream());

    *q2 = *Q;
    q2->data  = q_rot;
    q2->nb[0] = sizeof(float);
    q2->nb[1] = q2->nb[0]*q2->ne[0];
    q2->nb[2] = q2->nb[1]*q2->ne[1];
    q2->nb[3] = q2->nb[2]*q2->ne[2];
    q2->view_src  = nullptr;
    q2->view_offs = 0;
}

void ggml_cuda_flash_attn_ext_kvarn_rot_unfused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        void (*run)(ggml_backend_cuda_context & ctx, ggml_tensor * dst)) {
    const ggml_tensor * Q = dst->src[0];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && dst->ne[0] == 256 && ggml_is_contiguous(dst));
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> q_rot(ctx.pool(), ggml_nelements(Q));
    ggml_tensor q2;
    ggml_cuda_kvarn_rot256_q_pass(ctx, Q, q_rot.ptr, &q2);

    ggml_tensor d2 = *dst;
    d2.src[0] = &q2;
    ggml_set_op_params_i32(&d2, GGML_KVARN_ROT_PARAM, 0);
    run(ctx, &d2);

    // in place: each warp reads its whole row before writing it
    kvarn_rot256_launch((const char *) dst->data, (float *) dst->data, dst->ne[1], dst->ne[2], dst->ne[3],
            dst->nb[1], dst->nb[2], dst->nb[3], stream);
}
