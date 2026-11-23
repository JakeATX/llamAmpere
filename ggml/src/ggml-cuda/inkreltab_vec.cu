// [T5-GLUE] G3: Vectorized T-reusing k_ink_rel_table_vec16 (n_rel=16, H<=32, T<=8).
// Bit-identical to inkreltab.cu's k_ink_rel_table (same ascending c=0..15 fmaf(b, rs, acc) chain).
// Kill switch: GGML_CUDA_INK_RELTAB_VEC=0 (default 1 = on).

#include "inkreltab_vec.cuh"

#include <cstdlib>

#define RELTAB_VEC_WARPS 8
#define RELTAB_VEC_MAX_T 8

// 1 warp (32 lanes) owns 1 relative-distance row e across all H <= 32 heads and all T <= T_CAP tokens:
// - Lanes 0..15 perform a single coalesced 64-byte global read of bias[e, 0..15] (once for all H heads and T tokens!).
// - __shfl_sync broadcasts b_c to all H active lanes; each lane h accumulates acc[t] = fmaf(b_c, rs[(t*H + h)*17 + c], acc[t]).
// - Stride 17 on rs eliminates all SMEM bank conflicts across lane = 0..H-1.
// - Coalesced store dst[(t*(E+1) + e)*H + h] = acc[t] - ss[t*H + h].
template <int T_CAP>
static __global__ void __launch_bounds__(RELTAB_VEC_WARPS * WARP_SIZE) k_ink_rel_table_vec16(
        const float * __restrict__ bias, const float * __restrict__ r, const char * __restrict__ sinks,
        float * __restrict__ dst, const int E, const int H, const int T,
        const int64_t r_s1, const int64_t r_s2, const int64_t sk_nb1, const int64_t sk_nb2) {
    extern __shared__ float sm[];
    // rs layout: [T][H][17] (padded +1 to avoid SMEM bank conflicts), ss layout: [T][H]
    float * rs = sm;
    float * ss = sm + T * H * 17;

    const int tid = threadIdx.x;
    const int n_rs = T * H * 16;
    for (int idx = tid; idx < n_rs; idx += blockDim.x) {
        const int c  = idx & 15;
        const int th = idx >> 4;
        const int t  = th / H;
        const int h  = th % H;
        rs[th * 17 + c] = r[t * r_s2 + h * r_s1 + c];
    }
    const int n_ss = T * H;
    for (int idx = tid; idx < n_ss; idx += blockDim.x) {
        const int t = idx / H;
        const int h = idx % H;
        ss[idx] = *(const float *) (sinks + t * sk_nb2 + h * sk_nb1);
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int e    = blockIdx.x * RELTAB_VEC_WARPS + warp;
    if (e > E) {
        return;
    }

    // Single coalesced 64-byte load of bias[e, 0..15] for all H heads and all T tokens
    const float b_lane = (lane < 16 && e < E) ? bias[(int64_t) e * 16 + lane] : 0.0f;

    float acc[T_CAP];
#pragma unroll
    for (int t = 0; t < T_CAP; ++t) {
        acc[t] = 0.0f;
    }

    const int h = lane;
    // Uniform across all 32 lanes: __shfl_sync(0xffffffff, ...) must be executed at the same program point by all lanes.
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        const float bc = __shfl_sync(0xffffffff, b_lane, c);
        if (h < H) {
#pragma unroll
            for (int t = 0; t < T_CAP; ++t) {
                if (t < T) {
                    acc[t] = fmaf(bc, rs[(t * H + h) * 17 + c], acc[t]);
                }
            }
        }
    }
    if (h < H) {
#pragma unroll
        for (int t = 0; t < T_CAP; ++t) {
            if (t < T) {
                dst[((int64_t) t * (E + 1) + e) * H + h] = acc[t] - ss[t * H + h];
            }
        }
    }
}

bool ggml_cuda_ink_rel_table_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * bias, const ggml_tensor * r,
        const ggml_tensor * sinks, ggml_tensor * dst) {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_INK_RELTAB_VEC");
        return e == nullptr || atoi(e) != 0;
    }();
    if (!on) {
        return false;
    }
    const int n_rel = (int) bias->ne[0];
    const int E     = (int) bias->ne[1];
    const int H     = (int) r->ne[1];
    const int T     = (int) r->ne[2];
    if (n_rel != 16 || H < 1 || H > WARP_SIZE || T < 1 || T > RELTAB_VEC_MAX_T) {
        return false;
    }
    if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            r->type != GGML_TYPE_F32 || r->nb[0] != sizeof(float) || r->ne[0] != n_rel ||
            sinks->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            dst->ne[0] != H || dst->ne[1] != E + 1 || dst->ne[2] != T) {
        return false;
    }
    const dim3 grid((unsigned) ((E + 1 + RELTAB_VEC_WARPS - 1) / RELTAB_VEC_WARPS), 1, 1);
    const size_t smem = (size_t) (T * H * 17 + T * H) * sizeof(float);
    const int64_t r_s1 = r->nb[1] / sizeof(float);
    const int64_t r_s2 = r->nb[2] / sizeof(float);
    if (T == 1) {
        k_ink_rel_table_vec16<1><<<grid, RELTAB_VEC_WARPS * WARP_SIZE, smem, ctx.stream()>>>(
            (const float *) bias->data, (const float *) r->data, (const char *) sinks->data, (float *) dst->data,
            E, H, T, r_s1, r_s2, sinks->nb[1], sinks->nb[2]);
    } else if (T <= 5) {
        k_ink_rel_table_vec16<5><<<grid, RELTAB_VEC_WARPS * WARP_SIZE, smem, ctx.stream()>>>(
            (const float *) bias->data, (const float *) r->data, (const char *) sinks->data, (float *) dst->data,
            E, H, T, r_s1, r_s2, sinks->nb[1], sinks->nb[2]);
    } else {
        k_ink_rel_table_vec16<RELTAB_VEC_MAX_T><<<grid, RELTAB_VEC_WARPS * WARP_SIZE, smem, ctx.stream()>>>(
            (const float *) bias->data, (const float *) r->data, (const char *) sinks->data, (float *) dst->data,
            E, H, T, r_s1, r_s2, sinks->nb[1], sinks->nb[2]);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
