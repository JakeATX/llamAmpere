#include "inkreltab2.cuh"

// INKPREFILL: k_ink_rel_table (inkreltab.cu) with the same per-output fmaf chain (c ascending from 0, then - sink), so the
// table is bit-identical. Changes: TT tokens per block (r and sinks staged once per block, bias row in registers) and the
// r rows padded to n_rel + 1 floats (the [H][n_rel = 16] layout put the 24 heads on 2 banks: 12-way conflicts).
template <int TT>
static __global__ void k_ink_rel_table2(const float * __restrict__ bias, const float * __restrict__ r, const char * __restrict__ sinks,
        float * __restrict__ dst, const int n_rel, const int E, const int H, const int T, const int64_t r_s1, const int64_t r_s2,
        const int64_t sk_nb1, const int64_t sk_nb2) {
    extern __shared__ float sm[];
    const int ld = n_rel + 1;
    float * rs = sm;                // [TT][H][ld]
    float * ss = sm + TT*H*ld;      // [TT][H]
    const int t0 = blockIdx.y*TT;
    const int nt = min(TT, T - t0);
    for (int k = threadIdx.x; k < nt*H*n_rel; k += blockDim.x) {
        const int tt = k / (H*n_rel);
        const int hc = k % (H*n_rel);
        rs[(tt*H + hc / n_rel)*ld + hc % n_rel] = r[(int64_t) (t0 + tt)*r_s2 + (hc / n_rel)*r_s1 + hc % n_rel];
    }
    for (int k = threadIdx.x; k < nt*H; k += blockDim.x) {
        ss[k] = *(const float *) (sinks + (int64_t) (t0 + k / H)*sk_nb2 + (k % H)*sk_nb1);
    }
    __syncthreads();
    const int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= (E + 1)*H) {
        return;
    }
    const int e = j / H;
    const int h = j % H;
    float b[32];
#pragma unroll
    for (int c = 0; c < 32; ++c) {
        b[c] = c < n_rel && e < E ? bias[(int64_t) e*n_rel + c] : 0.0f;
    }
    for (int tt = 0; tt < nt; ++tt) {
        const float * rr = rs + (tt*H + h)*ld;
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < 32; ++c) {
            if (c < n_rel) {
                acc = fmaf(b[c], rr[c], acc);
            }
        }
        dst[((int64_t) (t0 + tt)*(E + 1) + e)*H + h] = acc - ss[tt*H + h];
    }
}

bool ggml_cuda_ink_rel_table2_enabled() {
    static const bool on = [] {
        const char * v = getenv("GGML_CUDA_INK_RELTAB2");
        return !(v && *v && atoi(v) == 0);
    }();
    return on;
}

void ggml_cuda_ink_rel_table2(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, ggml_tensor * dst) {
    constexpr int TT = 8;
    const int n_rel = (int) bias->ne[0];
    const int E     = (int) bias->ne[1];
    const int H     = (int) r->ne[1];
    const int T     = (int) r->ne[2];
    GGML_ASSERT(n_rel <= 32 && bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias));
    GGML_ASSERT(r->type == GGML_TYPE_F32 && r->nb[0] == sizeof(float) && r->ne[0] == n_rel && sinks->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->ne[0] == H && dst->ne[1] == E + 1 && dst->ne[2] == T);
    const int n = (E + 1)*H;
    const dim3 grid((unsigned) ((n + 255)/256), (unsigned) ((T + TT - 1)/TT), 1);
    const size_t smem = (size_t) TT*(H*(n_rel + 1) + H)*sizeof(float);
    GGML_ASSERT(smem <= 48*1024);
    k_ink_rel_table2<TT><<<grid, 256, smem, ctx.stream()>>>((const float *) bias->data, (const float *) r->data, (const char *) sinks->data,
        (float *) dst->data, n_rel, E, H, T, r->nb[1]/sizeof(float), r->nb[2]/sizeof(float), sinks->nb[1], sinks->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
