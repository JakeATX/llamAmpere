#include "inkreltab.cuh"
#include "inkreltab_vec.cuh"

static __global__ void k_ink_rel_table(const float * __restrict__ bias, const float * __restrict__ r, const char * __restrict__ sinks,
        float * __restrict__ dst, const int n_rel, const int E, const int H, const int64_t r_s1, const int64_t r_s2,
        const int64_t sk_nb1, const int64_t sk_nb2) {
    extern __shared__ float sm[];
    float * rs = sm;            // [H][n_rel]
    float * ss = sm + H*n_rel;  // [H]
    const int t = blockIdx.y;
    for (int k = threadIdx.x; k < H*n_rel; k += blockDim.x) {
        rs[k] = r[t*r_s2 + (k / n_rel)*r_s1 + k % n_rel];
    }
    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        ss[h] = *(const float *) (sinks + t*sk_nb2 + h*sk_nb1);
    }
    __syncthreads();
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= (E + 1)*H) {
        return;
    }
    const int e = j / H;
    const int h = j % H;
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < 32; ++c) {
        if (c < n_rel) {
            const float b = e < E ? bias[(int64_t) e*n_rel + c] : 0.0f;
            acc = fmaf(b, rs[h*n_rel + c], acc);
        }
    }
    dst[((int64_t) t*(E + 1) + e)*H + h] = acc - ss[h];
}

void ggml_cuda_ink_rel_table(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, ggml_tensor * dst) {
    if (ggml_cuda_ink_rel_table_vec(ctx, bias, r, sinks, dst)) {
        return;
    }
    const int n_rel = (int) bias->ne[0];
    const int E     = (int) bias->ne[1];
    const int H     = (int) r->ne[1];
    const int T     = (int) r->ne[2];
    GGML_ASSERT(n_rel <= 32 && bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias));
    GGML_ASSERT(r->type == GGML_TYPE_F32 && r->nb[0] == sizeof(float) && r->ne[0] == n_rel && sinks->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->ne[0] == H && dst->ne[1] == E + 1 && dst->ne[2] == T);
    const int n = (E + 1)*H;
    const dim3 grid((unsigned) ((n + 255)/256), (unsigned) T, 1);
    const size_t smem = (size_t) (H*n_rel + H)*sizeof(float);
    k_ink_rel_table<<<grid, 256, smem, ctx.stream()>>>((const float *) bias->data, (const float *) r->data, (const char *) sinks->data,
        (float *) dst->data, n_rel, E, H, r->nb[1]/sizeof(float), r->nb[2]/sizeof(float), sinks->nb[1], sinks->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
