#include <atomic>
#include <cstring>
#include <vector>
#include "mmvq.cuh"
#include "mmvq-kqrows.cuh"
#include "convrot.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#include "mmvq-iq4xs-k4.cuh"
#include "mmvq-iq4xs-wide.cuh"
#endif

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

// Experimental SM86 narrow-N path. The generic MMVQ kernel invokes the full
// quantized-weight decoder once for every destination column. During MTP
// verification those columns share the same weights, so Q4_K/Q5_K weight bits
// and scales can be decoded once and applied to all columns. These helpers keep
// each column's Q8_1 activation decode and dot-product implementation unchanged.
template<int ncols_dst>
static __device__ __forceinline__ void vec_dot_q4_K_q8_1_multi(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ y,
        const int stride_col_y, const int kby, const int kbx, const int iqs,
        float (&dots)[ncols_dst]) {
    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int v[2];
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));
    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    const uint16_t * scales = (const uint16_t *) bq4_K->scales;
    uint16_t aux[2];
    const int sj = bq8_offset/2;
    if (sj < 2) {
        aux[0] = scales[sj+0] & 0x3f3f;
        aux[1] = scales[sj+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[sj+2] >> 0) & 0x0f0f) | ((scales[sj-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[sj+2] >> 4) & 0x0f0f) | ((scales[sj-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        int u[2*QR4_K];
        float d8[QR4_K];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1 * bq8i = y + col*stride_col_y + kby + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);
            u[2*i+0] = q8[0];
            u[2*i+1] = q8[4];
        }
        dots[col] = vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
    }
}

template<int ncols_dst>
static __device__ __forceinline__ void vec_dot_q5_K_q8_1_multi(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ y,
        const int stride_col_y, const int kby, const int kbx, const int iqs,
        float (&dots)[ncols_dst]) {
    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int vl[2];
    int vh[2];
    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *) bq5_K->scales;
    uint16_t aux[2];
    const int sj = bq8_offset/2;
    if (sj < 2) {
        aux[0] = scales[sj+0] & 0x3f3f;
        aux[1] = scales[sj+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[sj+2] >> 0) & 0x0f0f) | ((scales[sj-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[sj+2] >> 4) & 0x0f0f) | ((scales[sj-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *) aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        int u[2*QR5_K];
        float d8[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1 * bq8i = y + col*stride_col_y + kby + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4);
            u[2*i+0] = q8[0];
            u[2*i+1] = q8[4];
        }
        dots[col] = vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
    }
}

// IQ4_XS: decode the eight nibbles of each 32-bit weight group once (interleaved 4-PRMT lookup)
// and dp4a them into one integer accumulator per destination column. The per-column integer
// sum, scale multiply and FP32 ops are exactly those of vec_dot_iq4_xs_q8_1, so each column's
// dot is bit-identical to the single-column path.
template<int ncols_dst>
static __device__ __forceinline__ void vec_dot_iq4_xs_q8_1_multi(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ y,
        const int stride_col_y, const int kby, const int kbx, const int iqs,
        float (&dots)[ncols_dst]) {
    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi[ncols_dst];
#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        sumi[col] = 0;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16_interleaved(aux_q4, kvalues_iq4nl);
#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            const block_q8_1 * bq8 = y + col*stride_col_y + kby + iqs/4;
            sumi[col] = ggml_cuda_dp4a(v.x, get_int_b4(bq8->qs, 2*j + 0), sumi[col]);
            sumi[col] = ggml_cuda_dp4a(v.y, get_int_b4(bq8->qs, 2*j + 1), sumi[col]);
        }
    }

    const int ls = vec_dot_iq4_xs_q8_1_scale(bq4, iqs);
#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        int s = sumi[col];
        s *= ls - 32;
        const float d = __half2float(bq4->d) * __low2float((y + col*stride_col_y + kby + iqs/4)->ds);
        dots[col] = d * s;
    }
}

// QC2 — Q5_0: the 5th bit of each weight is assembled from `qh` with four shift+mask pairs per
// 32-bit group, ten integer ops per group that the generic path repeats for every destination column.
// Assemble the group once and dp4a it into one integer accumulator per column. The accumulation order
// within a column, the scale multiply and the -16 offset term are exactly those of vec_dot_q5_0_q8_1,
// so each column's dot is bit-identical to the single-column path.
template<int ncols_dst>
static __device__ __forceinline__ void vec_dot_q5_0_q8_1_multi(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ y,
        const int stride_col_y, const int kby, const int kbx, const int iqs,
        float (&dots)[ncols_dst]) {
    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq + kbx;

    int sumi[ncols_dst];
#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        sumi[col] = 0;
    }

    const int qh = get_int_b2(bq5_0->qh, 0);
#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = qh >> (4 * (iqs + i));

        int vi0 = (vl >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh << 25) & 0x10000000; // 3 -> 28

        int vi1 = (vl >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh <<  9) & 0x10000000; // 19 -> 28

#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            const block_q8_1 * bq8 = y + col*stride_col_y + kby;
            sumi[col] = ggml_cuda_dp4a(vi0, get_int_b4(bq8->qs, iqs + i),          sumi[col]);
            sumi[col] = ggml_cuda_dp4a(vi1, get_int_b4(bq8->qs, iqs + i + QI5_0), sumi[col]);
        }
    }

    const float d5 = bq5_0->d;
#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        const float2 ds8f = __half22float2((y + col*stride_col_y + kby)->ds);
        // second part effectively subtracts 16 from each quant value
        dots[col] = d5 * (sumi[col] * ds8f.x - (16*VDR_Q5_0_Q8_1_MMVQ/QI5_0) * ds8f.y);
    }
}

// IQ4_XS cross-column reuse is exact and on by default for SM86; GGML_CUDA_SM86_IQ4_REUSE=0 disables it.
static bool ggml_cuda_sm86_iq4_reuse() {
    static const bool value = [] {
        const char * env = getenv("GGML_CUDA_SM86_IQ4_REUSE");
        return env == nullptr || env[0] != '0';
    }();
    return value;
}

// QC2 Q5_0 cross-column reuse is exact and on by default for SM86; GGML_CUDA_SM86_Q5_0_REUSE=0 disables it.
static bool ggml_cuda_sm86_q5_0_reuse() {
    static const bool value = [] {
        const char * env = getenv("GGML_CUDA_SM86_Q5_0_REUSE");
        return env == nullptr || env[0] != '0';
    }();
    return value;
}

// QC5: nwarps=1 for the speculative-verify widths (ncols_dst 2..4) on the GENERIC/Ampere
// table. NOT BIT-EXACT vs nwarps=4: FP addition is non-associative, and nwarps sets both the
// stride each warp takes through a row (blocks_per_iter) and the number of cross-warp partial
// sums added at the end (3 at nwarps=4, 0 at nwarps=1). Output is a different -- not worse --
// rounding of the same dot product. On by default; GGML_CUDA_QC4_NW1=0 restores nwarps=4 and
// with it bit-exactness against builds before this change.
// nwarps=1 launch shape for ncols_dst 2..4. OFF by default: the acceptance-free batched-bench
// arbitration (QC5) measured it at -1.00%/-0.59% at ncols_dst 4 -- the width a default
// --spec-draft-n-max 3 actually runs -- and +4.54%/+4.12% only at ncols_dst 3.
static bool ggml_cuda_qc4_nw1() {
    static const bool value = [] {
        const char * env = getenv("GGML_CUDA_QC4_NW1");
        return env != nullptr && env[0] == '1';
    }();
    return value;
}

// IQ3_XXS / IQ3_S: stage the codebook grid (1 KB / 2 KB, a static const __device__ table in
// global memory) into shared memory once per block. Same values, same accumulation order, so
// bit-identical to the global-table path. ON by default for SM86 at ncols_dst 1..4 (3090 Ti gate
// 2026-09-16, m=4096 k=14336: iq3_s +2.3/+9.1/+6.9/+10.1%, iq3_xxs +1.2/+5.3/+14.1/+5.5% at widths
// 1/2/3/4; at widths 5..8 rows_per_block drops 8 -> 2 and the win vanishes: iq3_xxs -4.1/-2.8%,
// iq3_s +0.9/+0.2%). GGML_CUDA_SM86_IQ3_SMEM_GRID=0 disables it, =1 forces it at every width.
// Returns the largest ncols_dst that uses the staged grid.
// [VBEAT fix 6] IQ3_S defaults to widths 1..5 (Flash-Next verify width 5 runs its dense IQ3_S projections here);
// =N (2..8) stages at widths 1..N for both types.
static int ggml_cuda_sm86_iq3_smem_grid_max_ncols(ggml_type type) {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_SM86_IQ3_SMEM_GRID");
        if (env == nullptr) {
            return -1;
        }
        const int n = atoi(env);
        return n == 1 ? MMVQ_MAX_BATCH_SIZE : (n <= 0 ? 0 : (n > MMVQ_MAX_BATCH_SIZE ? MMVQ_MAX_BATCH_SIZE : n));
    }();
    GGML_UNUSED(type);
    return value >= 0 ? value : 4; // VBEAT fix 6 (IQ3_S width 5) is opt-in via =5 until its MUL_MAT check runs
}

// IQ2_XXS / IQ2_XS / IQ2_S: same staged-codebook trick (2 KB / 4 KB / 8 KB uint64 tables). 2026-09-20 gate
// (m=4096 k=14336, us/run): iq2_xxs +1.1..+7.2% at every width 1..8, iq2_xs -0.5(noise)/+6.8/+7.5/+4.9/+9.7/+2.2%,
// iq2_s -18.1% at width 1 and -9.3% at width 8 (the 8 KB per-block copy dominates there) but +5.4..+8.9% at 2..5.
// SM86 default: iq2_xxs and iq2_xs at every width, iq2_s at widths 2..5. GGML_CUDA_SM86_IQ2_SMEM_GRID=0 turns it
// off, =1 stages at every width, =N at widths 1..N (overrides the per-type default).
static int ggml_cuda_sm86_iq2_smem_grid_env() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_SM86_IQ2_SMEM_GRID");
        if (env == nullptr) {
            return -1;
        }
        const int n = atoi(env);
        return n == 1 ? MMVQ_MAX_BATCH_SIZE : (n < 0 ? 0 : (n > MMVQ_MAX_BATCH_SIZE ? MMVQ_MAX_BATCH_SIZE : n));
    }();
    return value;
}

static bool ggml_cuda_sm86_iq2_smem_grid_use(ggml_type type, int ncols_dst) {
    const int env = ggml_cuda_sm86_iq2_smem_grid_env();
    if (env >= 0) {
        return ncols_dst <= env;
    }
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:  return true;
        case GGML_TYPE_IQ2_S:   return ncols_dst >= 2 && ncols_dst <= 5;
        default:                return false;
    }
}

static constexpr __host__ __device__ bool ggml_cuda_mmvq_smem_grid_type(ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S ||
           type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S;
}

// PTQ1_0 cross-column reuse (widths 2-8) is exact and on by default for SM86; GGML_CUDA_SM86_PTQ1_REUSE=0 disables it.
// Without it the generic path re-unpacks the 128-trit block once per column per row (4 cols x 8 rows unrolled) and
// spills 540-1260 B per thread at widths 2-4 (ptxas, 2026-09-17), which made a width-4 MTP verify cost 4.4 T=1 steps.
static bool ggml_cuda_sm86_ptq1_reuse() {
    static const bool value = [] {
        const char * env = getenv("GGML_CUDA_SM86_PTQ1_REUSE");
        return env == nullptr || env[0] != '0';
    }();
    return value;
}

static bool ggml_cuda_sm86_exact_reuse() {
    static const bool value = [] {
        const char * env = getenv("GGML_CUDA_SM86_EXACT_REUSE");
        return env != nullptr && env[0] == '1';
    }();
    return value;
}

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_PQ2_0: return vec_dot_pq2_0_q8_1;
        case GGML_TYPE_PTQ1_0: return vec_dot_ptq1_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_PQ2_0: return VDR_PQ2_0_Q8_1_MMVQ;
        case GGML_TYPE_PTQ1_0: return VDR_PTQ1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
#if !defined(GGML_USE_HIP)
    if (type == GGML_TYPE_PTQ1_0 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING) {
        return ne11 <= 8; // the V6 reuse lane kernel beats the MMQ tile at width 8 too
    }
#endif
    // Per-type MMVQ width ceiling override for kernel tuning sweeps, e.g.
    // GGML_MMVQ_NMAX="q6_K=5,iq4_xs=8" (batches above the ceiling go to MMQ when it
    // supports the type). Read once; unset = no override.
    {
        static const std::string spec = [] { const char * e = getenv("GGML_MMVQ_NMAX"); return std::string(e ? e : ""); }();
        if (!spec.empty()) {
            const char * tn = ggml_type_name(type);
            const size_t tl = strlen(tn);
            size_t pos = 0;
            while (pos < spec.size()) {
                size_t end = spec.find(',', pos); if (end == std::string::npos) end = spec.size();
                size_t eq = spec.find('=', pos);
                if (eq != std::string::npos && eq < end && eq - pos == tl && spec.compare(pos, tl, tn) == 0) {
                    return ne11 <= atoll(spec.c_str() + eq + 1);
                }
                pos = end + 1;
            }
        }
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 3090 Ti (m=4096 k=14336 sweep, W58 width-5..8 MMVQ cells vs MMQ):
                        // MMQ at widths 5..8 costs the same as its width-9 tile, and for the K-quants that is
                        // already below the width-5 MMVQ cell (q5_K +8%, q6_K +7%, q4_K +2%); iq4_xs/q5_0 keep MMVQ
                        // through width 8 (MMQ 10-20% slower there).
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 4; // [KQROWS] widths 5..8 can still reach MMVQ through ggml_cuda_kqrows_wants (route)
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

// QC4 -- MMVQ launch parameters for SM86. This build is CMAKE_CUDA_ARCHITECTURES=86 only and
// get_device_table_id() gates MMVQ_PARAMETERS_TURING on cc < AMPERE, so SM86 falls through to
// GENERIC: GENERIC *is* the Ampere table here. There is no tuned Ampere table in this fork or
// upstream. These macros exist so a sweep can retune the cells the decode path actually uses
// (ncols_dst 1..4 -- production runs --spec-draft-n-max 3, so verify width never exceeds 4)
// by editing one file, not one compile flag that rebuilds every template instance.
#ifndef QC4_NWARPS_1
#define QC4_NWARPS_1 4
#define QC4_NWARPS_2 4
#define QC4_NWARPS_3 4
#define QC4_NWARPS_4 4
#ifndef PTQ1_REUSE_ROWS_2_4
#define PTQ1_REUSE_ROWS_2_4 8
#endif
#ifndef PTQ1_REUSE_ROWS_5_8
#define PTQ1_REUSE_ROWS_5_8 8
#endif
#define QC4_ROWS_1   1
#define QC4_ROWS_2   8
#define QC4_ROWS_3   8
#define QC4_ROWS_4   8
#endif
// Verify widths 5..8 (W58, KDEV 2026-09-17, microbench m=4096 k=14336): rows 8 is the whole win, nwarps 2 beats 4
// for iq4_xs/q5_0/q6_K (width 5 = 1.03x width 4 for iq4_xs, was 1.33x; width 8 = 1.25x, was 1.94x); the Q4_K/Q5_K
// reuse kernel wants nwarps 4. PTQ1_0 keeps rows 2 (ALU-bound, see calc_rows_per_block).
#ifndef QC4_NWARPS_5
#define QC4_NWARPS_5 2
#define QC4_NWARPS_6 2
#define QC4_NWARPS_7 2
#define QC4_NWARPS_8 2
#define QC4_ROWS_5   8
#define QC4_ROWS_6   8
#define QC4_ROWS_7   8
#define QC4_ROWS_8   8
#endif
#ifndef QC4_NWARPS_58_K
#define QC4_NWARPS_58_K 4
#endif

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id,
        bool small_k = false, bool halve_iters = false, bool nw1 = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1: return QC4_NWARPS_1;
            case 2: return nw1 ? 1 : QC4_NWARPS_2;
            case 3: return nw1 ? 1 : QC4_NWARPS_3;
            case 4: return nw1 ? 1 : QC4_NWARPS_4;
            case 5: return (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) ? QC4_NWARPS_58_K : QC4_NWARPS_5;
            case 6: return (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) ? QC4_NWARPS_58_K : QC4_NWARPS_6;
            case 7: return (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) ? QC4_NWARPS_58_K : QC4_NWARPS_7;
            case 8: return (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) ? QC4_NWARPS_58_K : QC4_NWARPS_8;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1,
        int rows_override = 0) {
    if (rows_override > 0) {
        return rows_override; // [SMALLM] see ggml_cuda_mmvq_smallm
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        if (type == GGML_TYPE_PTQ1_0 && ncols_dst >= 2 && ncols_dst <= 4) {
            return PTQ1_REUSE_ROWS_2_4; // y loads are shared across the rows of a block; the reuse lane kernel needs enough rows to amortize them
        }
        if (type == GGML_TYPE_PTQ1_0 && ncols_dst >= 5 && ncols_dst <= 8) {
            return PTQ1_REUSE_ROWS_5_8;
        }
        switch (ncols_dst) {
            case 1: return small_k ? nwarps : QC4_ROWS_1;
            case 2: return QC4_ROWS_2;
            case 3: return QC4_ROWS_3;
            case 4: return QC4_ROWS_4;
            case 5: return QC4_ROWS_5;
            case 6: return QC4_ROWS_6;
            case 7: return QC4_ROWS_7;
            case 8: return QC4_ROWS_8;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false,
          bool reuse_weights = false, bool nw1 = false, bool smem_grid = false, bool check_rows = true, int rows_override = 0>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters, nw1)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t nrows_x) {
    // Indexed output columns span all routed experts. Their stride is not
    // the logical row count: tail blocks must not read or overwrite the next expert.
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    // PTQ1_0 verify widths (reuse path, ncols_dst >= 2) split each 128-weight block over 4 lanes
    // (VDR 1) so a lane unpacks a quarter block once and dots every column; width 1 keeps the
    // whole-block VDR 4 so K=5120 still takes the small-K launch shape (measured faster at T=1).
    constexpr int vdr = (type == GGML_TYPE_PTQ1_0 && reuse_weights) ? 1 : get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters, nw1);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps, rows_override);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    // GGML_CUDA_SM86_IQ3_SMEM_GRID: cooperative copy of the IQ3 codebook into shared memory.
    // One __syncthreads() per block, amortized over the whole K loop.
    // Table sizes in 32-bit words: iq3_xxs 256 (1 KB), iq3_s 512 (2 KB), iq2_xxs 512 (2 KB), iq2_xs 1024 (4 KB),
    // iq2_s 2048 (8 KB). All are static const __device__ tables in global memory (ggml-common.h).
    constexpr bool use_smem_grid = smem_grid && ggml_cuda_mmvq_smem_grid_type(type);
    constexpr int  smem_grid_n   = type == GGML_TYPE_IQ3_XXS ? 256 : type == GGML_TYPE_IQ3_S ? 512 :
                                   type == GGML_TYPE_IQ2_XXS ? 512 : type == GGML_TYPE_IQ2_XS ? 1024 : 2048;
    [[maybe_unused]] __shared__ uint32_t grid_s[use_smem_grid ? smem_grid_n : 1];
    if constexpr (use_smem_grid) {
        const uint32_t * grid_g = type == GGML_TYPE_IQ3_XXS ? iq3xxs_grid : type == GGML_TYPE_IQ3_S ? iq3s_grid :
                                  type == GGML_TYPE_IQ2_XXS ? (const uint32_t *) iq2xxs_grid :
                                  type == GGML_TYPE_IQ2_XS  ? (const uint32_t *) iq2xs_grid : (const uint32_t *) iq2s_grid;
#pragma unroll
        for (int i = tid; i < smem_grid_n; i += nwarps*warp_size) {
            grid_s[i] = grid_g[i];
        }
        __syncthreads();
    }
    auto vec_dot = [&](const void * vbq, const block_q8_1 * bq8, const int kbx_, const int iqs_) -> float {
        if constexpr (use_smem_grid && type == GGML_TYPE_IQ3_XXS) {
            return vec_dot_iq3_xxs_q8_1_impl(grid_s, vbq, bq8, kbx_, iqs_);
        } else if constexpr (use_smem_grid && type == GGML_TYPE_IQ3_S) {
            return vec_dot_iq3_s_q8_1_impl(grid_s, vbq, bq8, kbx_, iqs_);
        } else if constexpr (use_smem_grid && type == GGML_TYPE_IQ2_XXS) {
            return vec_dot_iq2_xxs_q8_1_impl((const uint2 *) grid_s, vbq, bq8, kbx_, iqs_);
        } else if constexpr (use_smem_grid && type == GGML_TYPE_IQ2_XS) {
            return vec_dot_iq2_xs_q8_1_impl((const uint2 *) grid_s, vbq, bq8, kbx_, iqs_);
        } else if constexpr (use_smem_grid && type == GGML_TYPE_IQ2_S) {
            return vec_dot_iq2_s_q8_1_impl((const uint2 *) grid_s, vbq, bq8, kbx_, iqs_);
        } else {
            return vec_dot_q_cuda(vbq, bq8, kbx_, iqs_);
        }
    };

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if constexpr (check_rows) {
                        if (uint32_t(row0 + i) >= nrows_x) { continue; }
                    }
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

        if constexpr (reuse_weights && type == GGML_TYPE_PTQ1_0) {
            // 4 lanes per 128-weight block (kqs = lane, VDR 1): the lane's y ints, scales and y sums are loaded
            // once per k step and shared by every row below; each lane unpacks its share once and dots every column.
            // Kept in its own branch: declaring the lane struct for the other reuse types put its 160-byte y array
            // on the stack of the iq4_xs width-5 kernel (REG 179 -> 109 + 160 B spill, 3x slower).
            ptq1_lane_y<ncols_dst> ptq1_y;
            ptq1_0_lane_load_y<ncols_dst>(y + kby, kqs, stride_col_y, ptq1_y);
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if constexpr (check_rows) {
                    if (uint32_t(row0 + i) >= nrows_x) { continue; }
                }
                float dots[ncols_dst];
                vec_dot_ptq1_0_q8_1_lane<ncols_dst>(
                    vx, kbx_offset + i*stride_row_x + kbx, kqs, ptq1_y, dots);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmp[j][i] += dots[j];
                }
            }
        } else if constexpr (reuse_weights && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q5_0)) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if constexpr (check_rows) {
                    if (uint32_t(row0 + i) >= nrows_x) { continue; }
                }
                float dots[ncols_dst];
                if constexpr (type == GGML_TYPE_IQ4_XS) {
                    vec_dot_iq4_xs_q8_1_multi<ncols_dst>(
                        vx, y, stride_col_y, kby, kbx_offset + i*stride_row_x + kbx, kqs, dots);
                } else if constexpr (type == GGML_TYPE_Q5_0) {
                    vec_dot_q5_0_q8_1_multi<ncols_dst>(
                        vx, y, stride_col_y, kby, kbx_offset + i*stride_row_x + kbx, kqs, dots);
                } else if constexpr (type == GGML_TYPE_Q4_K) {
                    vec_dot_q4_K_q8_1_multi<ncols_dst>(
                        vx, y, stride_col_y, kby, kbx_offset + i*stride_row_x + kbx, kqs, dots);
                } else {
                    vec_dot_q5_K_q8_1_multi<ncols_dst>(
                        vx, y, stride_col_y, kby, kbx_offset + i*stride_row_x + kbx, kqs, dots);
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    tmp[j][i] += dots[j];
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        float gate_dots[ncols_dst];
                        if constexpr (type == GGML_TYPE_IQ4_XS) {
                            vec_dot_iq4_xs_q8_1_multi<ncols_dst>(
                                vgate, y, stride_col_y, kby, kbx_offset + i*stride_row_x + kbx, kqs, gate_dots);
                        } else {
                            static_assert(type == GGML_TYPE_Q4_K, "unsupported fused weight reuse type");
                            vec_dot_q4_K_q8_1_multi<ncols_dst>(
                                vgate, y, stride_col_y, kby, kbx_offset + i*stride_row_x + kbx, kqs, gate_dots);
                        }
#pragma unroll
                        for (int j = 0; j < ncols_dst; ++j) {
                            tmp_gate[j][i] += gate_dots[j];
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    if constexpr (check_rows) {
                        if (uint32_t(row0 + i) >= nrows_x) { continue; }
                    }
                    tmp[j][i] += vec_dot(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// fork: keep each route's DP4A sequence while sharing the weight decoder across routes.
template<ggml_type type, int cols>
static __device__ __forceinline__ void moe_grouped_dot(
        const void * weights, const block_q8_1 * const (&y)[cols], const int kbx, const int iqs,
        float (&dots)[cols]) {
    int sumi[cols] = {};
    if constexpr (type == GGML_TYPE_IQ3_S) {
        const block_iq3_s * w = (const block_iq3_s *) weights + kbx;
        const int2 packed = make_int2(get_int_b2(w->qs, iqs), get_int_b2(w->qs, iqs + 1));
        const uint8_t * qs = (const uint8_t *) &packed;
        const int qh = w->qh[iqs/2];
        const int packed_signs = get_int_b2(w->signs, iqs/2);
        const uint8_t * signs = (const uint8_t *) &packed_signs;
#pragma unroll
        for (int l = 0; l < 8; l += 2) {
            const int lo = iq3s_grid[qs[l]     | ((qh << (8 - l)) & 0x100)];
            const int hi = iq3s_grid[qs[l + 1] | ((qh << (7 - l)) & 0x100)];
            const int s0 = __vcmpne4(((signs[l/2] & 0x03) << 7) | ((signs[l/2] & 0x0C) << 21), 0);
            const int s1 = __vcmpne4(((signs[l/2] & 0x30) << 3) | ((signs[l/2] & 0xC0) << 17), 0);
            const int v0 = __vsub4(lo ^ s0, s0);
            const int v1 = __vsub4(hi ^ s1, s1);
#pragma unroll
            for (int col = 0; col < cols; ++col) {
                sumi[col] = ggml_cuda_dp4a(v0, get_int_b4(y[col][iqs/2].qs, l), sumi[col]);
                sumi[col] = ggml_cuda_dp4a(v1, get_int_b4(y[col][iqs/2].qs, l + 1), sumi[col]);
            }
        }
        const int scale = 1 + 2*((w->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
#pragma unroll
        for (int col = 0; col < cols; ++col) {
            sumi[col] *= scale;
            const float d = __half2float(w->d) * __low2float(y[col][iqs/2].ds);
            dots[col] = d * sumi[col];
        }
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl * w = (const block_iq4_nl *) weights + kbx;
#pragma unroll
        for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
            const int2 v = get_int_from_table_16(get_int_b2(w->qs, iqs + l), kvalues_iq4nl);
#pragma unroll
            for (int col = 0; col < cols; ++col) {
                const int * q8 = (const int *) y[col]->qs + iqs;
                sumi[col] = ggml_cuda_dp4a(v.x, q8[l], sumi[col]);
                sumi[col] = ggml_cuda_dp4a(v.y, q8[l + 4], sumi[col]);
            }
        }
#pragma unroll
        for (int col = 0; col < cols; ++col) {
            const float d = __half2float(w->d) * __low2float(y[col]->ds);
            dots[col] = d * sumi[col];
        }
    } else {
        static_assert(type == GGML_TYPE_Q2_0, "unsupported grouped MoE type");
        const block_q2_0 * w = (const block_q2_0 *) weights + kbx;
        const int16_t * qs = (const int16_t *) w->qs + iqs*4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            const int qx = __byte_perm(qe, qo, 0x5140);
            const int qy = __byte_perm(qe, qo, 0x7362);
#pragma unroll
            for (int col = 0; col < cols; ++col) {
                sumi[col] = ggml_cuda_dp4a(get_int_b4(y[col][iqs].qs, 2*j), qx, sumi[col]);
                sumi[col] = ggml_cuda_dp4a(get_int_b4(y[col][iqs].qs, 2*j + 1), qy, sumi[col]);
            }
        }
        const float d2 = w->d;
#pragma unroll
        for (int col = 0; col < cols; ++col) {
            const float d8 = __low2float(y[col][iqs].ds);
            dots[col] = d2 * d8 * sumi[col];
        }
    }
}

template<ggml_type type, int cols, bool fused>
static __device__ __forceinline__ void moe_grouped_rows(
        const void * weights, const void * gate, const block_q8_1 * act, const int * members,
        float * dst, const int n_in, const int n_out, const int n_ids, const int channels) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    const int row0 = 8*blockIdx.x + 2*threadIdx.y;
    const int stride_y = GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1;
    const block_q8_1 * y[cols];
#pragma unroll
    for (int col = 0; col < cols; ++col) {
        const int route = members[col];
        y[col] = act + (int64_t)((route / n_ids)*channels + (route % n_ids) % channels)*stride_y;
    }
    float sums[2][cols] = {};
    float gates[2][cols] = {};
    for (int k = threadIdx.x / (qi/vdr); k < n_in/qk; k += vdr*32/qi) {
        const block_q8_1 * yk[cols];
#pragma unroll
        for (int col = 0; col < cols; ++col) {
            yk[col] = y[col] + k*(qk/QK8_1);
        }
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            if (row0 + row < n_out) {
                float dots[cols];
                const int kbx = (row0 + row)*(n_in/qk) + k;
                const int iqs = vdr*(threadIdx.x % (qi/vdr));
                moe_grouped_dot<type, cols>(weights, yk, kbx, iqs, dots);
#pragma unroll
                for (int col = 0; col < cols; ++col) {
                    sums[row][col] += dots[col];
                }
                if constexpr (fused) {
                    moe_grouped_dot<type, cols>(gate, yk, kbx, iqs, dots);
#pragma unroll
                    for (int col = 0; col < cols; ++col) {
                        gates[row][col] += dots[col];
                    }
                }
            }
        }
    }
#pragma unroll
    for (int row = 0; row < 2; ++row) {
#pragma unroll
        for (int col = 0; col < cols; ++col) {
            sums[row][col] = warp_reduce_sum<32>(sums[row][col]);
            if constexpr (fused) {
                gates[row][col] = warp_reduce_sum<32>(gates[row][col]);
            }
        }
    }
    if (threadIdx.x < 2 && row0 + (int)threadIdx.x < n_out) {
#pragma unroll
        for (int col = 0; col < cols; ++col) {
            float result = sums[threadIdx.x][col];
            if constexpr (fused) {
                result *= ggml_cuda_op_silu_single(gates[threadIdx.x][col]);
            }
            dst[members[col]*n_out + row0 + threadIdx.x] = result;
        }
    }
}

template<ggml_type type, bool fused>
__launch_bounds__(128, 1)
static __global__ void mul_mat_vec_q_moe_grouped(
        const void * const * table, const void * const * gate_table, const block_q8_1 * act,
        const int32_t * ids, float * dst, const int n_in, const int n_out,
        const int n_ids, const int tokens, const int channels, const int ids_stride) {
    __shared__ int selected[320];
    __shared__ int members[320];
    __shared__ int count;
    ggml_cuda_pdl_sync();
    const int tid = threadIdx.y*32 + threadIdx.x;
    const int routes = n_ids*tokens;
    for (int r = tid; r < routes; r += 128) {
        selected[r] = ids[(int64_t)(r / n_ids)*ids_stride + r % n_ids];
    }
    __syncthreads();
    if (tid == 0) {
        count = 0;
        bool leader = true;
        for (int r = 0; r < (int)blockIdx.y; ++r) {
            leader &= selected[r] != selected[blockIdx.y];
        }
        if (leader) {
            for (int r = blockIdx.y; r < routes; ++r) {
                if (selected[r] == selected[blockIdx.y]) {
                    members[count++] = r;
                }
            }
        }
    }
    __syncthreads();
    if (count == 0) {
        ggml_cuda_pdl_lc();
        return;
    }
    const void * weights = table[selected[blockIdx.y]];
    const void * gate = fused ? gate_table[selected[blockIdx.y]] : nullptr;
    for (int first = 0; first < count; first += 4) {
        const int left = count - first;
        if (left >= 4) {
            moe_grouped_rows<type, 4, fused>(weights, gate, act, members + first, dst, n_in, n_out, n_ids, channels);
        } else if (left == 3) {
            moe_grouped_rows<type, 3, fused>(weights, gate, act, members + first, dst, n_in, n_out, n_ids, channels);
        } else if (left == 2) {
            moe_grouped_rows<type, 2, fused>(weights, gate, act, members + first, dst, n_in, n_out, n_ids, channels);
        } else {
            moe_grouped_rows<type, 1, fused>(weights, gate, act, members + first, dst, n_in, n_out, n_ids, channels);
        }
    }
    ggml_cuda_pdl_lc();
}

static __global__ void moe_grouped_check(const uint32_t * reference, const uint32_t * actual, const int count) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count && reference[i] != actual[i]) {
        printf("grouped MoE bit mismatch: index=%d reference=%08x actual=%08x\n", i, reference[i], actual[i]);
        asm volatile("trap;");
    }
}
#endif

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false, bool has_clamp = false, bool route_filter = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr,
        const int32_t * ids_ptr, const int32_t * act_ids_ptr, const int32_t * gate_ids_ptr,
        const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const float up_min, const float up_max, const float gate_min, const float gate_max,
        const void * const * expert_table, const void * const * gate_table, const bool rank_major,
        const uint8_t * route_hot, const int route_want_hot) {
    // [VFLOOR] rank_major (GGML_MOE_MMV_RANK_MAJOR=1): the grid is (ranks, row blocks) instead of (row blocks, ranks),
    // so the blocks of every rank for one row block run together and tokens that share an expert read its rows from L2.
    // Every output keeps the same arithmetic (bit-identical); only the block order changes.
    const uint32_t blk_row  = rank_major ? blockIdx.y : blockIdx.x;
    const uint32_t blk_rank = rank_major ? blockIdx.x : blockIdx.y;
    const uint32_t n_rank   = rank_major ? gridDim.x  : gridDim.y;
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    const int32_t * GGML_CUDA_RESTRICT act_ids = act_ids_ptr;
    const int32_t * GGML_CUDA_RESTRICT gate_ids = gate_ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blk_rank == n_rank - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blk_row;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blk_rank;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    if constexpr (route_filter) {
        // fork (missjoin hit-first): a pass computes only the routes whose expert residency matches
        // route_want_hot; the complementary pass computes the rest. Every route runs exactly once, in
        // the same code with the same inputs, so the results are bit-identical.
        if (!shared_expert && (route_hot[channel_x] != 0) != (route_want_hot != 0)) {
            return;
        }
    }
    // the fork MoE slot cache keeps the gate weights in a separate pool with their own slot ids;
    // upstream fusion keeps the gate in the same expert layout, so fall back to channel_x there.
    const uint32_t channel_gate = shared_expert ? 0 : gate_ids ? gate_ids[channel_dst + token_idx * ids_stride] : channel_x;
    const uint32_t channel_y = act_ids && !shared_expert
        ? act_ids[channel_dst + token_idx * ids_stride]
        : fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    if (expert_table && !shared_expert) {
        vx = expert_table[channel_x];
    }
    const int kbx_offset  = (expert_table && !shared_expert ? 0 : channel_x*stride_channel_x) + row0*stride_row_x;
    if constexpr (has_fusion) {
        if (gate_table && !shared_expert) {
            vgate = gate_table[channel_gate];
        }
    }
    const int gate_kbx_offset = (gate_table && !shared_expert ? 0 : channel_gate*stride_channel_x) + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            // the fork MoE slot cache hands out a partially filled last row block, those rows
            // are not backed by memory, so they must not be read.
            if (uint32_t(row0 + i) < nrows_x) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], gate_kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                // fork: the fused MoE slot cache carries the model own up/gate clamp bounds
                if constexpr (has_clamp) {
                    result     = fmaxf(fminf(result,     up_max),   up_min);
                    gate_value = fmaxf(fminf(gate_value, gate_max), gate_min);
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_clamp) {
        GGML_UNUSED_VARS(up_min, up_max, gate_min, gate_max);
    }
    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale, gate_kbx_offset);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id,
        const bool small_k = false, const bool halve_iters = false, const bool nw1 = false, const int rows_override = 0) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters, nw1);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps, rows_override);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, bool nw1 = false, int rows_override = 0>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t nrows_x, cudaStream_t stream) {

    auto launch = [&](auto check_rows_tag) {
        constexpr bool check_rows = decltype(check_rows_tag)::value;
        const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

        [[maybe_unused]] bool iq3_smem = false; // staged codebook grid (IQ3 and IQ2 types)
        if constexpr (ggml_cuda_mmvq_smem_grid_type(type)) {
            const int device = ggml_cuda_get_device();
            const int cc = ggml_cuda_info().devices[device].cc;
            const bool use = (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S) ?
                c_ncols_dst <= ggml_cuda_sm86_iq3_smem_grid_max_ncols(type) : ggml_cuda_sm86_iq2_smem_grid_use(type, c_ncols_dst);
            iq3_smem = cc == 860 && use;
        }

        // [SMALLM] rows_override launches are only made without fusion (see MMVQ_SMALLM_TRY), so skip the fused kernels.
        if constexpr (rows_override == 0 && (c_ncols_dst == 1 || (c_ncols_dst <= 8 &&
                (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q4_K)))) {
            if (has_fusion) {
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
                if constexpr (ggml_cuda_mmvq_smem_grid_type(type)) {
                    if (iq3_smem) {
                        ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, false, nw1, true, check_rows, rows_override>, launch_params,
                             vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                             channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                             sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
                        return;
                    }
                }
                if constexpr ((type == GGML_TYPE_IQ4_XS && c_ncols_dst >= 2 && c_ncols_dst <= 5) ||
                        (type == GGML_TYPE_Q4_K && c_ncols_dst >= 3 && c_ncols_dst <= 5)) {
                    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
                    const bool reuse = type == GGML_TYPE_IQ4_XS ? ggml_cuda_sm86_iq4_reuse() : ggml_cuda_sm86_exact_reuse();
                    if (cc == 860 && reuse) {
                        ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, true, nw1, false, check_rows, rows_override>, launch_params,
                            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
                        return;
                    }
                }
                ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, false, nw1, false, check_rows, rows_override>, launch_params,
                     vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                     channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                     sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
                return;
            }
        }

        GGML_ASSERT(!has_fusion && "unsupported batch fusion type");

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
        if constexpr (ggml_cuda_mmvq_smem_grid_type(type)) {
            if (iq3_smem) {
                ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, false, nw1, true, check_rows, rows_override>, launch_params,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
                return;
            }
        }
        if constexpr (((type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) && c_ncols_dst >= 3 && c_ncols_dst <= 5) ||
                      ((type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q5_0) && c_ncols_dst >= 2 && c_ncols_dst <= 5) ||
                      (type == GGML_TYPE_PTQ1_0 && c_ncols_dst >= 2 && c_ncols_dst <= 8)) {
            const int device = ggml_cuda_get_device();
            const int cc = ggml_cuda_info().devices[device].cc;
            const bool reuse = type == GGML_TYPE_PTQ1_0 ? ggml_cuda_sm86_ptq1_reuse() :
                               type == GGML_TYPE_IQ4_XS ? ggml_cuda_sm86_iq4_reuse() :
                               type == GGML_TYPE_Q5_0   ? ggml_cuda_sm86_q5_0_reuse() : ggml_cuda_sm86_exact_reuse();
            if (cc == 860 && reuse) {
                ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, true, nw1, false, check_rows, rows_override>, launch_params,
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
                return;
            }
        }
        ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, false, nw1, false, check_rows, rows_override>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
            channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
            sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, nrows_x);
    };

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int rows_per_block = calc_rows_per_block(type, c_ncols_dst, get_device_table_id(cc), small_k, block_dims.y, rows_override);
    if constexpr (rows_override > 0) {
        // [SMALLM] the caller guarantees nrows_x % rows_override == 0, so only the unguarded kernel is built.
        GGML_ASSERT(nrows_x % rows_per_block == 0);
        launch(std::false_type{});
        return;
    }
    // Keep partial-row guards; specialize complete row blocks before launch.
    if (nrows_x % rows_per_block == 0) {
        launch(std::false_type{});
    } else {
        launch(std::true_type{});
    }
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
template <int ncols>
static void mul_mat_vec_q5_K_batch_head_launch(
        const void * vx, const block_q8_1 * y, float * dst, int nrows, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const auto dims = calc_launch_params<GGML_TYPE_Q5_K>(ncols, nrows, 1, 1, warp_size, MMVQ_PARAMETERS_GENERIC);
    const ggml_cuda_kernel_launch_params launch(dims.first, dims.second, 0, stream);
    const uint3 one = init_fastdiv_values(1);
    const ggml_cuda_mm_fusion_args_device fusion{};
    auto launch_rows = [&](auto check_rows_tag) {
        constexpr bool check_rows = decltype(check_rows_tag)::value;
        ggml_cuda_kernel_launch(mul_mat_vec_q<GGML_TYPE_Q5_K, ncols, false, false, false, true, false, false, check_rows>, launch,
                vx, y, nullptr, fusion, dst, 2560, make_uint3(0, 0, 0), 10, 80, nrows,
                one, 0, 0, 0, one, 0, 0, 0, nrows);
    };
    const int rows_per_block = calc_rows_per_block(GGML_TYPE_Q5_K, ncols, MMVQ_PARAMETERS_GENERIC, false, dims.second.y);
    if (nrows % rows_per_block == 0) {
        launch_rows(std::false_type{});
    } else {
        launch_rows(std::true_type{});
    }
}
#endif

template <ggml_type type, bool has_fusion = false, bool has_clamp = false, bool route_filter = false>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids,
        const int32_t * act_ids, const int32_t * gate_ids,
        const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst,
        const float up_min, const float up_max, const float gate_min,
        const float gate_max, cudaStream_t stream, const void * const * expert_table = nullptr, const void * const * gate_table = nullptr,
        const uint8_t * route_hot = nullptr, const int route_want_hot = 0) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    static const bool rank_major_env = [] { const char * e = getenv("GGML_MOE_MMV_RANK_MAJOR"); return e && atoi(e) != 0; }();
    const int64_t nranks = nchannels_dst + (fusion.shared_up != nullptr);
    const bool rank_major = rank_major_env && nblocks_rows <= 65535;
    const dim3 block_nums = rank_major ? dim3(nranks, nblocks_rows) : dim3(nblocks_rows, nranks);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, has_fusion, has_clamp, route_filter>, launch_params,
        vx, vy, ids, act_ids, gate_ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride, up_min, up_max, gate_min, gate_max, expert_table, gate_table, rank_major, route_hot, route_want_hot);
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// [VBEAT D1] Decode-once expert kernel for the device expert table path (GGML_MOE_MMV_D1).
// One block covers c_warps*c_rows rows of ONE distinct expert. Each warp owns c_rows rows and computes every
// token routed to that expert from the same weight loads: the per-token dot calls are unrolled over a compile-time
// member count with no stores between them, so the compiler loads and decodes each weight chunk once.
// Per (token, row) the lane's k blocks, their order, the dot function and the shuffle reduction are those of
// mul_mat_vec_q_moe, so every output is bitwise the same (check mode GGML_MOE_MMV_D1=check proves it in place).
// c_smem: the IQ2/IQ3 codebook is read from the block's shared-memory copy (same values and the same _impl dot
// function the global-table path calls, so the result does not change).
template <ggml_type type, int c_rows, int cols, bool c_smem>
static __device__ __forceinline__ float moe_d1_dot(const uint32_t * grid_s, const void * vbq, const block_q8_1 * bq8,
        const int kbx, const int iqs) {
    if constexpr (c_smem && type == GGML_TYPE_IQ3_XXS) {
        return vec_dot_iq3_xxs_q8_1_impl(grid_s, vbq, bq8, kbx, iqs);
    } else if constexpr (c_smem && type == GGML_TYPE_IQ3_S) {
        return vec_dot_iq3_s_q8_1_impl(grid_s, vbq, bq8, kbx, iqs);
    } else if constexpr (c_smem && type == GGML_TYPE_IQ2_XXS) {
        return vec_dot_iq2_xxs_q8_1_impl((const uint2 *) grid_s, vbq, bq8, kbx, iqs);
    } else if constexpr (c_smem && type == GGML_TYPE_IQ2_XS) {
        return vec_dot_iq2_xs_q8_1_impl((const uint2 *) grid_s, vbq, bq8, kbx, iqs);
    } else if constexpr (c_smem && type == GGML_TYPE_IQ2_S) {
        return vec_dot_iq2_s_q8_1_impl((const uint2 *) grid_s, vbq, bq8, kbx, iqs);
    } else {
        GGML_UNUSED(grid_s);
        constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
        return vec_dot_q_cuda(vbq, bq8, kbx, iqs);
    }
}

template <ggml_type type, int c_rows, int cols, bool c_smem>
static __device__ __forceinline__ void moe_d1_rows(
        const void * vx, const block_q8_1 * const (&y)[4], float * const (&out)[4],
        const int row0, const int nrows_x, const int blocks_per_row_x, const int stride_row_x, const uint32_t * grid_s) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const int kbx_offset = row0*stride_row_x;
    float tmp[cols][c_rows];
#pragma unroll
    for (int c = 0; c < cols; ++c) {
#pragma unroll
        for (int i = 0; i < c_rows; ++i) {
            tmp[c][i] = 0.0f;
        }
    }
    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));
#pragma unroll
        for (int i = 0; i < c_rows; ++i) {
            if (row0 + i < nrows_x) {
#pragma unroll
                for (int c = 0; c < cols; ++c) {
                    tmp[c][i] += moe_d1_dot<type, c_rows, cols, c_smem>(grid_s, vx, &y[c][kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < cols; ++c) {
#pragma unroll
        for (int i = 0; i < c_rows; ++i) {
            tmp[c][i] = warp_reduce_sum<warp_size>(tmp[c][i]);
        }
    }
#pragma unroll
    for (int i = 0; i < c_rows; ++i) {
        if (threadIdx.x == i && row0 + i < nrows_x) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
                out[c][row0 + i] = tmp[c][i];
            }
        }
    }
}

// Grid: (row tiles, n_ids*n_tokens routes in token-major order). Every warp reads the route list itself (at most 64
// routes, two per lane) and ballots for the routes that share this block's expert; only the block of the expert's
// first route works, so no shared memory and no block barrier are needed. Members run in token order, c_group at a
// time. The register budget is set by the widest member count compiled in, so c_group 2 (most experts have one or
// two routes at width 5) keeps the whole kernel inside the 80-register cap without the spills c_group 4 shows.
template <ggml_type type, int c_warps, int c_rows, int c_group, bool c_smem>
__launch_bounds__(c_warps*32, 768/(c_warps*32))
static __global__ void mul_mat_vec_q_moe_d1(
        const void * const * table, const block_q8_1 * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int act_channels,
        const int stride_col_y, const int stride_channel_y, const int stride_col_dst, const int stride_channel_dst,
        const int n_ids, const int n_tokens, const int ids_stride,
        const uint8_t * route_hot, const int route_want_hot) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const int routes = n_ids*n_tokens;
    const int q = blockIdx.y;
    const int r0 = threadIdx.x;
    const int r1 = threadIdx.x + 32;
    ggml_cuda_pdl_sync();
    const int e0 = r0 < routes ? ids[(r0 / n_ids)*ids_stride + r0 % n_ids] : -1;
    const int e1 = r1 < routes ? ids[(r1 / n_ids)*ids_stride + r1 % n_ids] : -1;
    const int me = __shfl_sync(0xffffffff, q < 32 ? e0 : e1, q & 31);
    const uint64_t mask = (uint64_t) __ballot_sync(0xffffffff, e0 == me) |
                          ((uint64_t) __ballot_sync(0xffffffff, e1 == me) << 32);
    if ((mask & ((1ull << q) - 1)) != 0) {
        return; // not the expert's first route: that block computes this route
    }
    if (route_hot != nullptr) {
        // the residency flag belongs to the expert, so the plain kernel filters all of its routes the same way
        if ((route_hot[me] != 0) != (route_want_hot != 0)) {
            return;
        }
    }
    // every early return above is the same for the whole block (all warps read the same route list)
    constexpr bool use_smem = c_smem && ggml_cuda_mmvq_smem_grid_type(type);
    constexpr int  smem_n   = type == GGML_TYPE_IQ3_XXS ? 256 : type == GGML_TYPE_IQ3_S ? 512 :
                              type == GGML_TYPE_IQ2_XXS ? 512 : type == GGML_TYPE_IQ2_XS ? 1024 : 2048;
    [[maybe_unused]] __shared__ uint32_t grid_s[use_smem ? smem_n : 1];
    if constexpr (use_smem) {
        const uint32_t * grid_g = type == GGML_TYPE_IQ3_XXS ? iq3xxs_grid : type == GGML_TYPE_IQ3_S ? iq3s_grid :
                                  type == GGML_TYPE_IQ2_XXS ? (const uint32_t *) iq2xxs_grid :
                                  type == GGML_TYPE_IQ2_XS  ? (const uint32_t *) iq2xs_grid : (const uint32_t *) iq2s_grid;
        for (int i = threadIdx.y*32 + threadIdx.x; i < smem_n; i += c_warps*32) {
            grid_s[i] = grid_g[i];
        }
        __syncthreads();
    }
    const int row0 = (blockIdx.x*c_warps + threadIdx.y)*c_rows;
    if (row0 >= nrows_x) {
        return;
    }
    const void * vx = table[me];
    const int blocks_per_row_x = ncols_x / qk;
    uint64_t left = mask;
    while (left != 0) {
        const block_q8_1 * y[4];
        float * out[4];
        int n = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < c_group && left != 0) {
                const int r = __ffsll((long long) left) - 1;
                left &= left - 1;
                const int t = r / n_ids;
                const int k = r - t*n_ids;
                y[j]   = vy + (k % act_channels)*stride_channel_y + t*stride_col_y;
                out[j] = dst + k*stride_channel_dst + t*stride_col_dst;
                ++n;
            } else {
                y[j]   = vy;
                out[j] = dst;
            }
        }
        if (c_group >= 4 && n == 4) {
            moe_d1_rows<type, c_rows, c_group >= 4 ? 4 : 1, use_smem>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        } else if (c_group >= 3 && n == 3) {
            moe_d1_rows<type, c_rows, c_group >= 3 ? 3 : 1, use_smem>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        } else if (c_group >= 2 && n == 2) {
            moe_d1_rows<type, c_rows, c_group >= 2 ? 2 : 1, use_smem>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        } else {
            moe_d1_rows<type, c_rows, 1, use_smem>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        }
    }
    ggml_cuda_pdl_lc();
}

template <ggml_type type, int c_warps, int c_rows, int c_group, bool c_smem = false>
static void moe_d1_launch(const void * const * table, const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels, int64_t ids_stride,
        const uint8_t * route_hot, int route_want_hot, cudaStream_t stream) {
    const int s11 = (int) (GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1);
    const int stride_row_x = (int) (ggml_row_size(type, n_in) / ggml_type_size(type));
    constexpr int rows_per_block = c_warps*c_rows;
    const ggml_cuda_kernel_launch_params params(dim3((n_out + rows_per_block - 1)/rows_per_block, n_ids*n_tokens),
            dim3(32, c_warps), 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_moe_d1<type, c_warps, c_rows, c_group, c_smem>, params, table, (const block_q8_1 *) act_q8,
            ids, dst, (int) n_in, (int) n_out, stride_row_x, (int) channels, (int) channels*s11, s11,
            (int) (n_ids*n_out), (int) n_out, (int) n_ids, (int) n_tokens, (int) ids_stride, route_hot, route_want_hot);
}

// [MOEGAP DOWNSW] Sub-warp rows for the expert down projection (GGML_MOE_DOWN_SW). Its rows are short (the
// expert's hidden width), so one warp per row leaves lanes idle: IQ4_NL at 640 columns is 2 passes with 32 then 8 lanes
// busy. Instead of one warp per row, each row gets a few lanes; our implementation keeps our arithmetic bit for
// bit: each of the c_lanes lanes of a group stands in for 32/c_lanes lanes of the plain kernel ("virtual lanes"),
// accumulates their partial sums in the same order, then runs the same xor butterfly (the steps that cross virtual
// lanes are done in registers, the rest by shuffles within the group). Float addition is commutative, so every
// partial sum, and therefore the output, has the same bits as moe_d1_rows / the plain kernel.
template <ggml_type type, int c_lanes, int c_rows, int cols>
static __device__ __forceinline__ void moe_sw_rows(
        const void * vx, const block_q8_1 * const (&y)[4], float * const (&out)[4],
        const int row0, const int nrows_x, const int blocks_per_row_x, const int stride_row_x) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr int VL = warp_size / c_lanes;
    static_assert(c_lanes >= 2 && c_lanes <= 32 && warp_size % c_lanes == 0 && c_rows <= c_lanes, "bad sub-warp shape");
    const int p = threadIdx.x % c_lanes;
    const int kbx_offset = row0*stride_row_x;
    float tmp[VL][cols][c_rows];
#pragma unroll
    for (int j = 0; j < VL; ++j) {
#pragma unroll
        for (int c = 0; c < cols; ++c) {
#pragma unroll
            for (int i = 0; i < c_rows; ++i) {
                tmp[j][c][i] = 0.0f;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < VL; ++j) {
        const int v = p + c_lanes*j;   // the plain kernel's lane this register set stands for
        for (int kbx = v / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
            const int kqs = vdr * (v % (qi/vdr));
#pragma unroll
            for (int i = 0; i < c_rows; ++i) {
                if (row0 + i < nrows_x) {
#pragma unroll
                    for (int c = 0; c < cols; ++c) {
                        tmp[j][c][i] += moe_d1_dot<type, c_rows, cols, false>(nullptr, vx, &y[c][kby],
                                kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }
    // butterfly steps 16 .. c_lanes: partner lane v ^ o is virtual set j ^ (o / c_lanes) of this lane
#pragma unroll
    for (int o = warp_size/2; o >= c_lanes; o >>= 1) {
        float nt[VL][cols][c_rows];
#pragma unroll
        for (int j = 0; j < VL; ++j) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
#pragma unroll
                for (int i = 0; i < c_rows; ++i) {
                    nt[j][c][i] = tmp[j][c][i] + tmp[j ^ (o / c_lanes)][c][i];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < VL; ++j) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
#pragma unroll
                for (int i = 0; i < c_rows; ++i) {
                    tmp[j][c][i] = nt[j][c][i];
                }
            }
        }
    }
    // steps below c_lanes stay inside the aligned group
#pragma unroll
    for (int o = c_lanes/2; o > 0; o >>= 1) {
#pragma unroll
        for (int c = 0; c < cols; ++c) {
#pragma unroll
            for (int i = 0; i < c_rows; ++i) {
                tmp[0][c][i] += __shfl_xor_sync(0xffffffff, tmp[0][c][i], o, warp_size);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < c_rows; ++i) {
        if (p == i && row0 + i < nrows_x) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
                out[c][row0 + i] = tmp[0][c][i];
            }
        }
    }
}

// Grid and route grouping as mul_mat_vec_q_moe_d1; a warp covers (32/c_lanes)*c_rows rows.
template <ggml_type type, int c_warps, int c_lanes, int c_rows, int c_group>
__launch_bounds__(c_warps*32, 768/(c_warps*32))
static __global__ void mul_mat_vec_q_moe_sw(
        const void * const * table, const block_q8_1 * vy, const int32_t * ids, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int act_channels,
        const int stride_col_y, const int stride_channel_y, const int stride_col_dst, const int stride_channel_dst,
        const int n_ids, const int n_tokens, const int ids_stride,
        const uint8_t * route_hot, const int route_want_hot) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int rows_per_warp = (32/c_lanes)*c_rows;
    const int routes = n_ids*n_tokens;
    const int q = blockIdx.y;
    const int r0 = threadIdx.x;
    const int r1 = threadIdx.x + 32;
    ggml_cuda_pdl_sync();
    const int e0 = r0 < routes ? ids[(r0 / n_ids)*ids_stride + r0 % n_ids] : -1;
    const int e1 = r1 < routes ? ids[(r1 / n_ids)*ids_stride + r1 % n_ids] : -1;
    const int me = __shfl_sync(0xffffffff, q < 32 ? e0 : e1, q & 31);
    const uint64_t mask = (uint64_t) __ballot_sync(0xffffffff, e0 == me) |
                          ((uint64_t) __ballot_sync(0xffffffff, e1 == me) << 32);
    if ((mask & ((1ull << q) - 1)) != 0) {
        return;
    }
    if (route_hot != nullptr && (route_hot[me] != 0) != (route_want_hot != 0)) {
        return;
    }
    const int wrow0 = (blockIdx.x*c_warps + threadIdx.y)*rows_per_warp;
    if (wrow0 >= nrows_x) {
        return;
    }
    const int row0 = wrow0 + (threadIdx.x / c_lanes)*c_rows;
    const void * vx = table[me];
    const int blocks_per_row_x = ncols_x / qk;
    uint64_t left = mask;
    while (left != 0) {
        const block_q8_1 * y[4];
        float * out[4];
        int n = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < c_group && left != 0) {
                const int r = __ffsll((long long) left) - 1;
                left &= left - 1;
                const int t = r / n_ids;
                const int k = r - t*n_ids;
                y[j]   = vy + (k % act_channels)*stride_channel_y + t*stride_col_y;
                out[j] = dst + k*stride_channel_dst + t*stride_col_dst;
                ++n;
            } else {
                y[j]   = vy;
                out[j] = dst;
            }
        }
        if (c_group >= 4 && n == 4) {
            moe_sw_rows<type, c_lanes, c_rows, c_group >= 4 ? 4 : 1>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x);
        } else if (c_group >= 3 && n == 3) {
            moe_sw_rows<type, c_lanes, c_rows, c_group >= 3 ? 3 : 1>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x);
        } else if (c_group >= 2 && n == 2) {
            moe_sw_rows<type, c_lanes, c_rows, c_group >= 2 ? 2 : 1>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x);
        } else {
            moe_sw_rows<type, c_lanes, c_rows, 1>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x);
        }
    }
    ggml_cuda_pdl_lc();
}

template <ggml_type type, int c_warps, int c_lanes, int c_rows, int c_group>
static void moe_sw_launch(const void * const * table, const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels, int64_t ids_stride,
        const uint8_t * route_hot, int route_want_hot, cudaStream_t stream) {
    const int s11 = (int) (GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1);
    const int stride_row_x = (int) (ggml_row_size(type, n_in) / ggml_type_size(type));
    constexpr int rows_per_block = c_warps*(32/c_lanes)*c_rows;
    const ggml_cuda_kernel_launch_params params(dim3((n_out + rows_per_block - 1)/rows_per_block, n_ids*n_tokens),
            dim3(32, c_warps), 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_moe_sw<type, c_warps, c_lanes, c_rows, c_group>, params, table,
            (const block_q8_1 *) act_q8, ids, dst, (int) n_in, (int) n_out, stride_row_x, (int) channels,
            (int) channels*s11, s11, (int) (n_ids*n_out), (int) n_out, (int) n_ids, (int) n_tokens, (int) ids_stride,
            route_hot, route_want_hot);
}

// [MOEGAP HITG] Fused gate + up + SwiGLU form of the decode-once kernel (GGML_MOE_HIT_FUSED). Both
// projections and the SwiGLU of an expert are done in one pass, with our own arithmetic. Each warp computes the up
// and the gate rows of its expert for every member token from one read of the activation chunk; the up and gate sums are those of mul_mat_vec_q_moe (same lanes, same k
// order, same reduction), and the output is up * silu(gate), the bits of the separate GLU kernel (silu(gate) * up).
// One launch replaces the up and gate launches and the GLU launch of a pass.
// Route filter: a route works in a pass when (m0[e] && m1[e]) == want, so with the residency flags of both
// projections the hit pass takes the experts resident in both, and with the miss-pass skip flags (want 0) the miss
// pass takes every other expert the card computes. Every route of the layer is written once.
template <ggml_type type, int c_rows, int cols, bool c_smem>
static __device__ __forceinline__ void moe_hitg_rows(
        const void * vu, const void * vg, const block_q8_1 * const (&y)[4], float * const (&out)[4],
        const int row0, const int nrows_x, const int blocks_per_row_x, const int stride_row_x, const uint32_t * grid_s) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const int kbx_offset = row0*stride_row_x;
    float tu[cols][c_rows];
    float tg[cols][c_rows];
#pragma unroll
    for (int c = 0; c < cols; ++c) {
#pragma unroll
        for (int i = 0; i < c_rows; ++i) {
            tu[c][i] = 0.0f;
            tg[c][i] = 0.0f;
        }
    }
    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));
#pragma unroll
        for (int i = 0; i < c_rows; ++i) {
            if (row0 + i < nrows_x) {
#pragma unroll
                for (int c = 0; c < cols; ++c) {
                    tu[c][i] += moe_d1_dot<type, c_rows, cols, c_smem>(grid_s, vu, &y[c][kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                    tg[c][i] += moe_d1_dot<type, c_rows, cols, c_smem>(grid_s, vg, &y[c][kby],
                            kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < cols; ++c) {
#pragma unroll
        for (int i = 0; i < c_rows; ++i) {
            tu[c][i] = warp_reduce_sum<warp_size>(tu[c][i]);
            tg[c][i] = warp_reduce_sum<warp_size>(tg[c][i]);
        }
    }
#pragma unroll
    for (int i = 0; i < c_rows; ++i) {
        if (threadIdx.x == i && row0 + i < nrows_x) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
                out[c][row0 + i] = ggml_cuda_op_silu_single(tg[c][i]) * tu[c][i];
            }
        }
    }
}

// Grid and route handling as mul_mat_vec_q_moe_d1: (row tiles, routes in token-major order); the block of an
// expert's first route computes all of its routes, the others return.
template <ggml_type type, int c_warps, int c_rows, int c_group, bool c_smem>
__launch_bounds__(c_warps*32, 768/(c_warps*32))
static __global__ void mul_mat_vec_q_moe_hitg(
        const void * const * up_table, const void * const * gate_table, const block_q8_1 * vy, const int32_t * ids,
        float * dst, const int ncols_x, const int nrows_x, const int stride_row_x, const int act_channels,
        const int stride_col_y, const int stride_channel_y, const int stride_col_dst, const int stride_channel_dst,
        const int n_ids, const int n_tokens, const int ids_stride,
        const uint8_t * m0, const uint8_t * m1, const int want) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const int routes = n_ids*n_tokens;
    const int q = blockIdx.y;
    const int r0 = threadIdx.x;
    const int r1 = threadIdx.x + 32;
    ggml_cuda_pdl_sync();
    const int e0 = r0 < routes ? ids[(r0 / n_ids)*ids_stride + r0 % n_ids] : -1;
    const int e1 = r1 < routes ? ids[(r1 / n_ids)*ids_stride + r1 % n_ids] : -1;
    const int me = __shfl_sync(0xffffffff, q < 32 ? e0 : e1, q & 31);
    const uint64_t mask = (uint64_t) __ballot_sync(0xffffffff, e0 == me) |
                          ((uint64_t) __ballot_sync(0xffffffff, e1 == me) << 32);
    if ((mask & ((1ull << q) - 1)) != 0) {
        return;
    }
    if (((m0[me] != 0) && (m1[me] != 0)) != (want != 0)) {
        return;
    }
    constexpr bool use_smem = c_smem && ggml_cuda_mmvq_smem_grid_type(type);
    constexpr int  smem_n   = type == GGML_TYPE_IQ3_XXS ? 256 : type == GGML_TYPE_IQ3_S ? 512 :
                              type == GGML_TYPE_IQ2_XXS ? 512 : type == GGML_TYPE_IQ2_XS ? 1024 : 2048;
    [[maybe_unused]] __shared__ uint32_t grid_s[use_smem ? smem_n : 1];
    if constexpr (use_smem) {
        const uint32_t * grid_g = type == GGML_TYPE_IQ3_XXS ? iq3xxs_grid : type == GGML_TYPE_IQ3_S ? iq3s_grid :
                                  type == GGML_TYPE_IQ2_XXS ? (const uint32_t *) iq2xxs_grid :
                                  type == GGML_TYPE_IQ2_XS  ? (const uint32_t *) iq2xs_grid : (const uint32_t *) iq2s_grid;
        for (int i = threadIdx.y*32 + threadIdx.x; i < smem_n; i += c_warps*32) {
            grid_s[i] = grid_g[i];
        }
        __syncthreads();
    }
    const int row0 = (blockIdx.x*c_warps + threadIdx.y)*c_rows;
    if (row0 >= nrows_x) {
        return;
    }
    const void * vu = up_table[me];
    const void * vg = gate_table[me];
    const int blocks_per_row_x = ncols_x / qk;
    uint64_t left = mask;
    while (left != 0) {
        const block_q8_1 * y[4];
        float * out[4];
        int n = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < c_group && left != 0) {
                const int r = __ffsll((long long) left) - 1;
                left &= left - 1;
                const int t = r / n_ids;
                const int k = r - t*n_ids;
                y[j]   = vy + (k % act_channels)*stride_channel_y + t*stride_col_y;
                out[j] = dst + k*stride_channel_dst + t*stride_col_dst;
                ++n;
            } else {
                y[j]   = vy;
                out[j] = dst;
            }
        }
        if (c_group >= 2 && n == 2) {
            moe_hitg_rows<type, c_rows, c_group >= 2 ? 2 : 1, use_smem>(vu, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        } else {
            moe_hitg_rows<type, c_rows, 1, use_smem>(vu, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        }
    }
    ggml_cuda_pdl_lc();
}

// [MOEGAP HITGSW] Sub-warp rows for the fused gate/up/SwiGLU hit pass (GGML_MOE_HIT_FUSED_SHAPE=sw...). 16-lane gate/up rows keep 5 dots per
// lane busy at K = 2560. Each of the c_lanes lanes of a group stands in for 32/c_lanes lanes of the plain kernel
// ("virtual lanes", as moe_sw_rows), keeps their partial sums in the same k order and runs the same xor butterfly, so
// the up and gate sums, and up * silu(gate), have the bits of moe_hitg_rows / the plain kernels. c_kb > 0 is the
// row's block count known at compile time: the k loop then has a fixed trip count (predicated), so the loads of all
// its steps can be in flight together; the order of the additions does not change.
template <ggml_type type, int c_lanes, int c_rows, int cols, int c_kb, bool c_smem = false>
static __device__ __forceinline__ void moe_hitg_sw_rows(
        const void * vu, const void * vg, const block_q8_1 * const (&y)[4], float * const (&out)[4],
        const int row0, const int nrows_x, const int blocks_per_row_x, const int stride_row_x,
        const uint32_t * grid_s = nullptr) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = 32;
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr int VL = warp_size / c_lanes;
    static_assert(c_lanes >= 2 && c_lanes <= 16 && warp_size % c_lanes == 0 && c_rows <= c_lanes, "bad sub-warp shape");
    const int p = threadIdx.x % c_lanes;
    const int kbx_offset = row0*stride_row_x;
    const int kb = c_kb > 0 ? c_kb : blocks_per_row_x;
    float tu[VL][cols][c_rows];
    float tg[VL][cols][c_rows];
#pragma unroll
    for (int j = 0; j < VL; ++j) {
#pragma unroll
        for (int c = 0; c < cols; ++c) {
#pragma unroll
            for (int i = 0; i < c_rows; ++i) {
                tu[j][c][i] = 0.0f;
                tg[j][c][i] = 0.0f;
            }
        }
    }
    constexpr int n_it = c_kb > 0 ? (c_kb + blocks_per_iter - 1) / blocks_per_iter : 1;
#pragma unroll
    for (int j = 0; j < VL; ++j) {
        const int v = p + c_lanes*j;   // the plain kernel's lane this register set stands for
        const int kqs = vdr * (v % (qi/vdr));
        if constexpr (c_kb > 0) {
#pragma unroll
            for (int it = 0; it < n_it; ++it) {
                const int kbx = v / (qi/vdr) + it*blocks_per_iter;
                if (kbx < kb) {
                    const int kby = kbx * (qk/QK8_1);
#pragma unroll
                    for (int i = 0; i < c_rows; ++i) {
                        if (row0 + i < nrows_x) {
#pragma unroll
                            for (int c = 0; c < cols; ++c) {
                                tu[j][c][i] += moe_d1_dot<type, c_rows, cols, c_smem>(grid_s, vu, &y[c][kby],
                                        kbx_offset + i*stride_row_x + kbx, kqs);
                                tg[j][c][i] += moe_d1_dot<type, c_rows, cols, c_smem>(grid_s, vg, &y[c][kby],
                                        kbx_offset + i*stride_row_x + kbx, kqs);
                            }
                        }
                    }
                }
            }
        } else {
            for (int kbx = v / (qi/vdr); kbx < kb; kbx += blocks_per_iter) {
                const int kby = kbx * (qk/QK8_1);
#pragma unroll
                for (int i = 0; i < c_rows; ++i) {
                    if (row0 + i < nrows_x) {
#pragma unroll
                        for (int c = 0; c < cols; ++c) {
                            tu[j][c][i] += moe_d1_dot<type, c_rows, cols, c_smem>(grid_s, vu, &y[c][kby],
                                    kbx_offset + i*stride_row_x + kbx, kqs);
                            tg[j][c][i] += moe_d1_dot<type, c_rows, cols, c_smem>(grid_s, vg, &y[c][kby],
                                    kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }
    // butterfly steps 16 .. c_lanes: partner lane v ^ o is virtual set j ^ (o / c_lanes) of this lane
#pragma unroll
    for (int o = warp_size/2; o >= c_lanes; o >>= 1) {
        float nu[VL][cols][c_rows];
        float ng[VL][cols][c_rows];
#pragma unroll
        for (int j = 0; j < VL; ++j) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
#pragma unroll
                for (int i = 0; i < c_rows; ++i) {
                    nu[j][c][i] = tu[j][c][i] + tu[j ^ (o / c_lanes)][c][i];
                    ng[j][c][i] = tg[j][c][i] + tg[j ^ (o / c_lanes)][c][i];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < VL; ++j) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
#pragma unroll
                for (int i = 0; i < c_rows; ++i) {
                    tu[j][c][i] = nu[j][c][i];
                    tg[j][c][i] = ng[j][c][i];
                }
            }
        }
    }
    // steps below c_lanes stay inside the aligned group
#pragma unroll
    for (int o = c_lanes/2; o > 0; o >>= 1) {
#pragma unroll
        for (int c = 0; c < cols; ++c) {
#pragma unroll
            for (int i = 0; i < c_rows; ++i) {
                tu[0][c][i] += __shfl_xor_sync(0xffffffff, tu[0][c][i], o, warp_size);
                tg[0][c][i] += __shfl_xor_sync(0xffffffff, tg[0][c][i], o, warp_size);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < c_rows; ++i) {
        if (p == i && row0 + i < nrows_x) {
#pragma unroll
            for (int c = 0; c < cols; ++c) {
                out[c][row0 + i] = ggml_cuda_op_silu_single(tg[0][c][i]) * tu[0][c][i];
            }
        }
    }
}

// Grid and route handling as mul_mat_vec_q_moe_hitg; a warp covers (32/c_lanes)*c_rows rows, members are taken
// c_group (up to 4) tokens at a time for one read of the expert's rows.
template <ggml_type type, int c_warps, int c_lanes, int c_rows, int c_group, int c_kb, bool c_smem = false>
__launch_bounds__(c_warps*32, 768/(c_warps*32))
static __global__ void mul_mat_vec_q_moe_hitg_sw(
        const void * const * up_table, const void * const * gate_table, const block_q8_1 * vy, const int32_t * ids,
        float * dst, const int ncols_x, const int nrows_x, const int stride_row_x, const int act_channels,
        const int stride_col_y, const int stride_channel_y, const int stride_col_dst, const int stride_channel_dst,
        const int n_ids, const int n_tokens, const int ids_stride,
        const uint8_t * m0, const uint8_t * m1, const int want) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int rows_per_warp = (32/c_lanes)*c_rows;
    const int routes = n_ids*n_tokens;
    const int q = blockIdx.y;
    const int r0 = threadIdx.x;
    const int r1 = threadIdx.x + 32;
    ggml_cuda_pdl_sync();
    const int e0 = r0 < routes ? ids[(r0 / n_ids)*ids_stride + r0 % n_ids] : -1;
    const int e1 = r1 < routes ? ids[(r1 / n_ids)*ids_stride + r1 % n_ids] : -1;
    const int me = __shfl_sync(0xffffffff, q < 32 ? e0 : e1, q & 31);
    const uint64_t mask = (uint64_t) __ballot_sync(0xffffffff, e0 == me) |
                          ((uint64_t) __ballot_sync(0xffffffff, e1 == me) << 32);
    if ((mask & ((1ull << q) - 1)) != 0) {
        return;
    }
    if (((m0[me] != 0) && (m1[me] != 0)) != (want != 0)) {
        return;
    }
    // [MOEGAP HITGSW s] codebook copy in shared memory, as mul_mat_vec_q_moe_hitg (block-uniform point: every
    // return above depends only on blockIdx.y)
    constexpr bool use_smem = c_smem && ggml_cuda_mmvq_smem_grid_type(type);
    constexpr int  smem_n   = type == GGML_TYPE_IQ3_XXS ? 256 : type == GGML_TYPE_IQ3_S ? 512 :
                              type == GGML_TYPE_IQ2_XXS ? 512 : type == GGML_TYPE_IQ2_XS ? 1024 : 2048;
    [[maybe_unused]] __shared__ uint32_t grid_s[use_smem ? smem_n : 1];
    if constexpr (use_smem) {
        const uint32_t * grid_g = type == GGML_TYPE_IQ3_XXS ? iq3xxs_grid : type == GGML_TYPE_IQ3_S ? iq3s_grid :
                                  type == GGML_TYPE_IQ2_XXS ? (const uint32_t *) iq2xxs_grid :
                                  type == GGML_TYPE_IQ2_XS  ? (const uint32_t *) iq2xs_grid : (const uint32_t *) iq2s_grid;
        for (int i = threadIdx.y*32 + threadIdx.x; i < smem_n; i += c_warps*32) {
            grid_s[i] = grid_g[i];
        }
        __syncthreads();
    }
    const int wrow0 = (blockIdx.x*c_warps + threadIdx.y)*rows_per_warp;
    if (wrow0 >= nrows_x) {
        return;
    }
    const int row0 = wrow0 + (threadIdx.x / c_lanes)*c_rows;
    const void * vu = up_table[me];
    const void * vg = gate_table[me];
    const int blocks_per_row_x = ncols_x / qk;
    uint64_t left = mask;
    while (left != 0) {
        const block_q8_1 * y[4];
        float * out[4];
        int n = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j < c_group && left != 0) {
                const int r = __ffsll((long long) left) - 1;
                left &= left - 1;
                const int t = r / n_ids;
                const int k = r - t*n_ids;
                y[j]   = vy + (k % act_channels)*stride_channel_y + t*stride_col_y;
                out[j] = dst + k*stride_channel_dst + t*stride_col_dst;
                ++n;
            } else {
                y[j]   = vy;
                out[j] = dst;
            }
        }
        if (c_group >= 4 && n == 4) {
            moe_hitg_sw_rows<type, c_lanes, c_rows, c_group >= 4 ? 4 : 1, c_kb, use_smem>(vu, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        } else if (c_group >= 3 && n == 3) {
            moe_hitg_sw_rows<type, c_lanes, c_rows, c_group >= 3 ? 3 : 1, c_kb, use_smem>(vu, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        } else if (c_group >= 2 && n == 2) {
            moe_hitg_sw_rows<type, c_lanes, c_rows, c_group >= 2 ? 2 : 1, c_kb, use_smem>(vu, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        } else {
            moe_hitg_sw_rows<type, c_lanes, c_rows, 1, c_kb, use_smem>(vu, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, grid_s);
        }
    }
    ggml_cuda_pdl_lc();
}

template <ggml_type type, int c_warps, int c_lanes, int c_rows, int c_group, bool c_smem = false>
static void moe_hitg_sw_launch(const void * const * up_table, const void * const * gate_table, const char * act_q8,
        const int32_t * ids, float * dst, int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels,
        int64_t ids_stride, const uint8_t * m0, const uint8_t * m1, int want, cudaStream_t stream) {
    const int s11 = (int) (GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1);
    const int stride_row_x = (int) (ggml_row_size(type, n_in) / ggml_type_size(type));
    constexpr int rows_per_block = c_warps*(32/c_lanes)*c_rows;
    const ggml_cuda_kernel_launch_params params(dim3((n_out + rows_per_block - 1)/rows_per_block, n_ids*n_tokens),
            dim3(32, c_warps), 0, stream);
#define MOE_HITG_SW_KARGS up_table, gate_table, (const block_q8_1 *) act_q8, ids, dst, (int) n_in, (int) n_out, \
        stride_row_x, (int) channels, (int) channels*s11, s11, (int) (n_ids*n_out), (int) n_out, (int) n_ids, \
        (int) n_tokens, (int) ids_stride, m0, m1, want
    if (n_in == 10*ggml_cuda_type_traits<type>::qk) {   // Flash-Next gate/up: K = 2560, 10 blocks per row
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe_hitg_sw<type, c_warps, c_lanes, c_rows, c_group, 10, c_smem>, params, MOE_HITG_SW_KARGS);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe_hitg_sw<type, c_warps, c_lanes, c_rows, c_group, 0, c_smem>, params, MOE_HITG_SW_KARGS);
    }
#undef MOE_HITG_SW_KARGS
}

template <ggml_type type, int c_warps, int c_rows, int c_group, bool c_smem = false>
static void moe_hitg_launch(const void * const * up_table, const void * const * gate_table, const char * act_q8,
        const int32_t * ids, float * dst, int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels,
        int64_t ids_stride, const uint8_t * m0, const uint8_t * m1, int want, cudaStream_t stream) {
    const int s11 = (int) (GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1);
    const int stride_row_x = (int) (ggml_row_size(type, n_in) / ggml_type_size(type));
    constexpr int rows_per_block = c_warps*c_rows;
    const ggml_cuda_kernel_launch_params params(dim3((n_out + rows_per_block - 1)/rows_per_block, n_ids*n_tokens),
            dim3(32, c_warps), 0, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q_moe_hitg<type, c_warps, c_rows, c_group, c_smem>, params, up_table, gate_table,
            (const block_q8_1 *) act_q8, ids, dst, (int) n_in, (int) n_out, stride_row_x, (int) channels,
            (int) channels*s11, s11, (int) (n_ids*n_out), (int) n_out, (int) n_ids, (int) n_tokens, (int) ids_stride,
            m0, m1, want);
}

// check mode: dst must equal silu(gate) * up of the plain kernel on every route of the pass
static __global__ void moe_hitg_check_kernel(const float * up, const float * gate, const float * dst,
        const int32_t * ids, const uint8_t * m0, const uint8_t * m1, int want, int n_out, int n_ids, int n_tokens,
        int ids_stride) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_out*n_ids*n_tokens) {
        return;
    }
    const int route = i / n_out;
    const int t = route / n_ids, k = route % n_ids;
    const int e = ids[t*ids_stride + k];
    if (((m0[e] != 0) && (m1[e] != 0)) != (want != 0)) {
        return;
    }
    const float ref = ggml_cuda_op_silu_single(gate[i]) * up[i];
    if (__float_as_uint(ref) != __float_as_uint(dst[i])) {
        printf("[moe-hitg] check FAILED: route %d (token %d rank %d expert %d) row %d: %.9g vs %.9g\n",
                route, t, k, e, i % n_out, ref, dst[i]);
        __trap();
    }
}
#endif

// Indexed vocabulary rows share one activation vector and never read padded rows.
// [PARW] ncols > 1: several activation columns (one per drafting slot) share one id list (ids broadcast, stride 0),
// so each weight row is read once for all columns. Each column's sum keeps the single-column order (bit-identical).
template <ggml_type type, int ncols>
static __global__ void mul_mat_vec_q_indexed_rows(
        const void * vx, const block_q8_1 * y, const int32_t * ids, float * dst,
        const int ncols_x, const int count, const int64_t row_bytes, const int dst_stride,
        const int stride_col_y, const int stride_col_dst) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr auto vec_dot = get_vec_dot_q_cuda(type);
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    ggml_cuda_pdl_sync();
    if (row >= count) { return; }
    const void * weights = (const char *) vx + (int64_t) ids[row] * row_bytes;
    const int lane = threadIdx.x;
    float sum[ncols] = {0.0f};
    for (int k = lane / (qi / vdr); k < ncols_x / qk; k += vdr * warp_size / qi) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            sum[j] += vec_dot(weights, y + j * stride_col_y + k * (qk / QK8_1), k, vdr * (lane % (qi / vdr)));
        }
    }
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sum[j] = warp_reduce_sum<warp_size>(sum[j]);
    }
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            dst[j * stride_col_dst + row * dst_stride] = sum[j];
        }
    }
}

// [SMALLM] Matrices with few rows at verify widths 4..6 (Flash-Next shared expert gate/up 640 x 2560 and down
// 2560 x 640, GDN/QSA small projections): the table packs 8 rows per CTA, so a 640-row matrix fills only 80 CTAs
// with 2 warps each and runs latency-bound (7.3 us for 0.87 MB, ~120 GB/s, 100K trace 2026-10-21). This launch
// keeps nwarps and the K split of the table launch and only packs fewer rows per CTA. Each row's partial sums and
// their cross-warp/warp reduction order are unchanged, so the result is bit-identical to the table launch.
// Opt-in: GGML_CUDA_MMVQ_SMALLM=<max rows>[:<rows per CTA 1|2>] (default rows per CTA 2), off when unset. Only plain
// (unfused) launches at 4 and 5 columns with nrows % rows-per-CTA == 0 take it, to keep the template count small.
static int ggml_cuda_mmvq_smallm_max_rows() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_SMALLM");
        const int n = env == nullptr ? 0 : atoi(env);
        return n < 0 ? 0 : n;
    }();
    return value;
}

static int ggml_cuda_mmvq_smallm_rpb() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_SMALLM");
        const char * c = env ? strchr(env, ':') : nullptr;
        const int r = c ? atoi(c + 1) : 2;
        return r == 1 ? 1 : 2;
    }();
    return value;
}

static constexpr bool ggml_cuda_mmvq_smallm_type(ggml_type type) {
    return type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K ||
           type == GGML_TYPE_IQ3_S  || type == GGML_TYPE_IQ4_NL;
}

// [#45] Thin launch for matrices with few rows at verify widths 2..8. The GDN gate projections ssm_alpha and
// ssm_beta are 48 rows x 5120 (Q8_0 in the ATX/RVN/Swift IQ4_XS quants, PTQ1_0 in the Swift ternary quant); the
// table launch packs 8 rows per CTA at widths >= 2, so each one fills 6 CTAs on an 84-SM card and runs
// latency-bound (9.0 us per call, 96 calls per verify round). Here each CTA owns one row and its nwarps warps split
// K, giving nrows CTAs. The cross-warp sum has a fixed order, so the result is deterministic, but it groups partial
// sums differently from the table launch: not bit-identical to it. Opt-in: GGML_CUDA_MMVQ_THIN=<max rows> (off when
// unset or 0), GGML_CUDA_MMVQ_THIN_NWARPS=4|8 (default 4).
static int ggml_cuda_mmvq_thin_max_rows() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_THIN");
        const int n = env == nullptr ? 0 : atoi(env);
        return n < 0 ? 0 : n;
    }();
    return value;
}

static int ggml_cuda_mmvq_thin_nwarps() {
    static const int value = [] {
        const char * env = getenv("GGML_CUDA_MMVQ_THIN_NWARPS");
        return env != nullptr && atoi(env) == 8 ? 8 : 4;
    }();
    return value;
}

static constexpr bool ggml_cuda_mmvq_thin_type(ggml_type type) {
    return type == GGML_TYPE_Q8_0 || type == GGML_TYPE_PTQ1_0;
}

template <ggml_type type, int ncols_dst, int nwarps>
__launch_bounds__(nwarps*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_thin(
        const void * vx_ptr, const block_q8_1 * vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint3 channel_ratio, const uint32_t stride_channel_x, const uint32_t stride_channel_y,
        const uint32_t stride_channel_dst, const uint3 sample_ratio, const uint32_t stride_sample_x,
        const uint32_t stride_sample_y, const uint32_t stride_sample_dst) {
    const void       * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const block_q8_1 * GGML_CUDA_RESTRICT vy  = vy_ptr;
    float            * GGML_CUDA_RESTRICT dst = dst_ptr;
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr vec_dot_q_cuda_t vec_dot = get_vec_dot_q_cuda(type);
    constexpr int blocks_per_iter = vdr*nwarps*warp_size/qi;

    const int      tid         = warp_size*threadIdx.y + threadIdx.x;
    const uint32_t row         = blockIdx.x;
    const uint32_t channel_dst = blockIdx.y;
    const uint32_t sample_dst  = blockIdx.z;
    const uint32_t channel_x   = fastdiv(channel_dst, channel_ratio);
    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const int      blocks_per_row_x = ncols_x / qk;

    ggml_cuda_pdl_sync();
    const block_q8_1 * y = vy + sample_dst*stride_sample_y + channel_dst*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row*stride_row_x;

    float tmp[ncols_dst] = {0.0f};
    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            tmp[j] += vec_dot(vx, &y[j*stride_col_y + kby], kbx_offset + kbx, kqs);
        }
    }

    __shared__ float tmp_shared[nwarps][ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        tmp[j] = warp_reduce_sum<warp_size>(tmp[j]);
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            tmp_shared[threadIdx.y][j] = tmp[j];
        }
    }
    __syncthreads();
    if (tid < ncols_dst) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            sum += tmp_shared[w][tid];
        }
        dst[sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + tid*stride_col_dst + row] = sum;
    }
}

template <ggml_type type, int nwarps>
static void mul_mat_vec_q_thin_launch(
        const void * vx, const void * vy, float * dst, const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, const uint3 channel_ratio,
        const int nchannels_dst, const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const uint3 sample_ratio, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y,
        const int stride_sample_dst, const int warp_size, cudaStream_t stream) {
    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
    const dim3 block_dims(warp_size, nwarps, 1);
    const ggml_cuda_kernel_launch_params launch_params(block_nums, block_dims, 0, stream);
    const block_q8_1 * y = (const block_q8_1 *) vy;
#define MMVQ_THIN_CASE(n) \
        case n: ggml_cuda_kernel_launch(mul_mat_vec_q_thin<type, n, nwarps>, launch_params, vx, y, dst, \
            ncols_x, stride_row_x, stride_col_y, stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y, \
            stride_channel_dst, sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst); break;
    switch (ncols_dst) {
        MMVQ_THIN_CASE(2)
        MMVQ_THIN_CASE(3)
        MMVQ_THIN_CASE(4)
        MMVQ_THIN_CASE(5)
        MMVQ_THIN_CASE(6)
        MMVQ_THIN_CASE(7)
        MMVQ_THIN_CASE(8)
        default: GGML_ABORT("fatal error");
    }
#undef MMVQ_THIN_CASE
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// fork: retain the four-warp sum tree while each physical warp computes one head row.
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q5_K_head(
        const void * vx, const block_q8_1 * y, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qi = ggml_cuda_type_traits<GGML_TYPE_Q5_K>::qi;
    constexpr int vdr = VDR_Q5_K_Q8_1_MMVQ;
    constexpr int blocks_per_iter = vdr * 4*warp_size / qi;
    const uint32_t row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    if (row >= (uint32_t) nrows_x) {
        return;
    }

    const block_q5_K * weights = (const block_q5_K *) vx + (int64_t) row * stride_row_x;
    const int kqs = vdr * (lane % (qi / vdr));
    float sums[4] = { 0.0f };
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        for (int k = (w*warp_size + lane) / (qi / vdr); k < ncols_x / QK_K; k += blocks_per_iter) {
            sums[w] += vec_dot_q5_K_q8_1(weights, y + k * (QK_K / QK8_1), k, kqs);
        }
    }

    // The reference stores these partials in shared memory before the additions.
    float sum = __fadd_rn(sums[0], sums[1]);
    sum = __fadd_rn(sum, sums[2]);
    sum = __fadd_rn(sum, sums[3]);
    sum = warp_reduce_sum<warp_size>(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}
#endif

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, bool allow_small_k) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    if (has_ids && nrows_x == 1 && (ncols_dst == 1 || ids_stride == 0) && nchannels_y == 1 &&
            nsamples_x == 1 && nsamples_dst == 1 && !fusion.gate && !fusion.x_bias &&
            !fusion.gate_bias && !fusion.x_scale && !fusion.gate_scale) {
        const dim3 blocks((nchannels_dst + 3) / 4);
        const dim3 threads(warp_size, 4);
        const ggml_cuda_kernel_launch_params launch(blocks, threads, 0, stream);
        const int64_t row_bytes = (int64_t) stride_channel_x * ggml_type_size(type);
#define MMVQ_INDEXED_CASE(n) \
        case n: ggml_cuda_kernel_launch(mul_mat_vec_q_indexed_rows<type, n>, launch, \
                vx, (const block_q8_1 *) vy, ids, dst, ncols_x, nchannels_dst, row_bytes, stride_channel_dst, \
                stride_col_y, stride_col_dst); return;
        switch (ncols_dst) {
            MMVQ_INDEXED_CASE(1)
            MMVQ_INDEXED_CASE(2)
            MMVQ_INDEXED_CASE(3)
            MMVQ_INDEXED_CASE(4)
            MMVQ_INDEXED_CASE(5)
            MMVQ_INDEXED_CASE(6)
            MMVQ_INDEXED_CASE(7)
            MMVQ_INDEXED_CASE(8)
            default: GGML_ABORT("fatal error");
        }
#undef MMVQ_INDEXED_CASE
    }

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        const bool moe_has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                    fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
        if (moe_has_fusion) {
            mul_mat_vec_q_moe_launch<type, true, false>(
                vx, vy, ids, nullptr, nullptr, fusion, dst,
                ncols_x, nchannels_y_fd, nrows_x,
                stride_row_x, stride_col_y, stride_col_dst,
                stride_channel_x, stride_channel_y, stride_channel_dst,
                ncols_dst, ids_stride, warp_size, nchannels_dst,
                0.0f, 0.0f, 0.0f, 0.0f, stream);
            return;
        }
        mul_mat_vec_q_moe_launch<type, false, false>(
            vx, vy, ids, nullptr, nullptr, fusion, dst,
            ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst,
            0.0f, 0.0f, 0.0f, 0.0f, stream);
        return;
    }

    if constexpr (ggml_cuda_mmvq_thin_type(type)) {
        const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                                fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
        if (!has_ids && !has_fusion && ncols_dst >= 2 && nrows_x <= ggml_cuda_mmvq_thin_max_rows() &&
                table_id == MMVQ_PARAMETERS_GENERIC) {
            if (ggml_cuda_mmvq_thin_nwarps() == 8) {
                mul_mat_vec_q_thin_launch<type, 8>(vx, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,
                    stride_col_dst, channel_ratio_fd, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio_fd, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, warp_size, stream);
            } else {
                mul_mat_vec_q_thin_launch<type, 4>(vx, vy, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y,
                    stride_col_dst, channel_ratio_fd, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                    sample_ratio_fd, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, warp_size, stream);
            }
            return;
        }
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if constexpr (type == GGML_TYPE_Q5_K &&
                  calc_nwarps(type, 1, MMVQ_PARAMETERS_GENERIC) == 4 &&
                  calc_rows_per_block(type, 1, MMVQ_PARAMETERS_GENERIC) == 1) {
        const char * env = getenv("GGML_CUDA_MTP_Q5_HEAD");
        if (env && strcmp(env, "1") == 0 && cc == 860 && !has_ids && ncols_x == 2560 &&
                ncols_dst == 1 && nrows_x >= 65536 && stride_row_x == ncols_x / QK_K &&
                stride_col_y == ncols_x / QK8_1 && stride_col_dst == nrows_x &&
                nchannels_x == 1 && nchannels_y == 1 && nchannels_dst == 1 &&
                nsamples_x == 1 && nsamples_dst == 1 &&
                !fusion.gate && !fusion.x_bias && !fusion.gate_bias && !fusion.x_scale && !fusion.gate_scale &&
                !fusion.shared_up && !fusion.shared_gate && !fusion.shared_dst) {
            const char * audit = getenv("GGML_CUDA_MTP_Q5_HEAD_AUDIT");
            if (audit && strcmp(audit, "1") == 0) {
                GGML_LOG_INFO("mtp_q5_head_dispatch rows=%d k=%d cols=%d stride=%d\n",
                        nrows_x, ncols_x, ncols_dst, stride_row_x);
            }
            const dim3 blocks(((uint32_t) nrows_x + 3) / 4);
            const dim3 threads(warp_size, 4);
            const ggml_cuda_kernel_launch_params launch(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q5_K_head, launch,
                    vx, (const block_q8_1 *) vy, dst, ncols_x, nrows_x, stride_row_x);
            return;
        }
    }
#endif


// [SMALLM] inside the ncols switch: launch with fewer rows per CTA (bit-identical, see ggml_cuda_mmvq_smallm_max_rows)
#define MMVQ_SMALLM_TRY(NC)                                                                                              \
    if constexpr (ggml_cuda_mmvq_smallm_type(type)) {                                                                   \
        const int smallm_r = ggml_cuda_mmvq_smallm_rpb();                                                              \
        const bool smallm_fused = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||   \
                                  fusion.x_scale != nullptr || fusion.gate_scale != nullptr;                             \
        if (table_id == MMVQ_PARAMETERS_GENERIC && nrows_x <= ggml_cuda_mmvq_smallm_max_rows() && !smallm_fused &&     \
                nrows_x % smallm_r == 0) {                                                                             \
            auto smallm_launch = [&](auto r_tag) {                                                                     \
                constexpr int c_r = decltype(r_tag)::value;                                                            \
                std::pair<dim3, dim3> dims = calc_launch_params<type>(NC, nrows_x, nchannels_dst, nsamples_dst,          \
                    warp_size, table_id, false, false, false, c_r);                                                    \
                mul_mat_vec_q_switch_fusion<type, NC, false, false, false, c_r>(vx, vy, ids, fusion, dst, ncols_x,       \
                    nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst, channel_ratio_fd, stride_channel_x,     \
                    stride_channel_y, stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y,            \
                    stride_sample_dst, dims.first, dims.second, 0, nrows_x, stream);                                   \
            };                                                                                                         \
            if (smallm_r == 1) { smallm_launch(std::integral_constant<int, 1>{}); }                                    \
            else { smallm_launch(std::integral_constant<int, 2>{}); }                                                  \
            break;                                                                                                     \
        }                                                                                                              \
    }
    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, nrows_x,
                    stream);
            };

            if (allow_small_k && should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            // QC5: nwarps=1 is the measured-faster launch shape for the speculative-verify
            // widths. Default on, NOT bit-exact vs nwarps=4; GGML_CUDA_QC4_NW1=0 opts out.
            if (table_id == MMVQ_PARAMETERS_GENERIC && ggml_cuda_qc4_nw1()) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, false, false, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, true>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            // QC5: nwarps=1 is the measured-faster launch shape for the speculative-verify
            // widths. Default on, NOT bit-exact vs nwarps=4; GGML_CUDA_QC4_NW1=0 opts out.
            if (table_id == MMVQ_PARAMETERS_GENERIC && ggml_cuda_qc4_nw1()) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, false, false, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, true>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
                break;
            }
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            // QC5: nwarps=1 is the measured-faster launch shape for the speculative-verify
            // widths. Default on, NOT bit-exact vs nwarps=4; GGML_CUDA_QC4_NW1=0 opts out.
            if (table_id == MMVQ_PARAMETERS_GENERIC && ggml_cuda_qc4_nw1()) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id, false, false, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, true>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
                break;
            }
            MMVQ_SMALLM_TRY(4)
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            MMVQ_SMALLM_TRY(5)
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, nrows_x, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
#undef MMVQ_SMALLM_TRY
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, bool allow_small_k = true) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_PQ2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_PQ2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_PTQ1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_PTQ1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

bool ggml_cuda_q8_cacheable(const ggml_backend_cuda_context & ctx, size_t q8_bytes) {
    static const bool q8_cache_disabled = getenv("GGML_CUDA_Q8CACHE") != nullptr && atoi(getenv("GGML_CUDA_Q8CACHE")) == 0;
    // Main stream only: a sibling stream could consume the buffer with no cross-stream ordering.
    return !ctx.scratch_borrowed && !q8_cache_disabled && q8_bytes <= (1u << 20) && ctx.curr_stream_no == 0;
}

// The q8_1 layout depends on the weight type only through the IQ4_XS swizzle (quantize_row_q8_1_cuda), so the
// cache keys on that layout, not on the type (#89a).
static bool ggml_cuda_q8_cache_swizzle(ggml_type type_src0) {
    return type_src0 == GGML_TYPE_IQ4_XS;
}

static ggml_backend_cuda_context::q8_cache_entry * ggml_cuda_q8_cache_find(ggml_backend_cuda_context & ctx, const ggml_tensor * src1,
                                                                           ggml_type type_src0, size_t q8_bytes, int64_t ne10_padded) {
    const bool swizzle = ggml_cuda_q8_cache_swizzle(type_src0);
    for (auto & e : ctx.q8_cache.entries) {
        if (e.epoch == ctx.graph_epoch && e.src1 == src1 && e.data == src1->data && e.size == q8_bytes &&
            e.ne10_padded == ne10_padded && e.swizzle_iq4 == swizzle && e.dev == ctx.device) {
            return &e;
        }
    }
    return nullptr;
}

// [MOEGAP Q8SHARE] lookup only: the cached buffer of (src1, layout) in this graph eval, or nullptr
char * ggml_cuda_q8_cache_lookup(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_type type_src0,
                                 size_t q8_bytes, int64_t ne10_padded) {
    ggml_backend_cuda_context::q8_cache_entry * qe = ggml_cuda_q8_cache_find(ctx, src1, type_src0, q8_bytes, ne10_padded);
    if (qe == nullptr) {
        return nullptr;
    }
    qe->last_use = ++ctx.q8_cache.tick;
    ctx.fusion_stats.q8_cache_hits++;
    return qe->ptr;
}

char * ggml_cuda_q8_cache_claim(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_type type_src0,
                                size_t q8_bytes, int64_t ne10_padded) {
    auto & qc = ctx.q8_cache;
    ggml_backend_cuda_context::q8_cache_entry * qe = ggml_cuda_q8_cache_find(ctx, src1, type_src0, q8_bytes, ne10_padded);
    if (qe == nullptr) {
        // replace an entry from an older graph eval first, else the least recently used one
        qe = &qc.entries[0];
        for (auto & e : qc.entries) {
            if (e.epoch != ctx.graph_epoch) {
                qe = &e;
                break;
            }
            if (e.last_use < qe->last_use) {
                qe = &e;
            }
        }
    }
    qe->last_use = ++qc.tick;
    if (qe->dev != ctx.device || qe->cap < q8_bytes) {
        // Never free a buffer here: a CUDA graph captured earlier may still replay
        // kernels that point at it (several graphs per context with --n-cpu-moe splits).
        // Retire it and release everything at context teardown instead.
        if (qe->ptr != nullptr) {
            qc.retired.push_back({ qe->ptr, qe->cap, qe->dev });
        }
        // Plain device memory, not pool memory: the pool frees strict LIFO, and this
        // buffer is taken while transient pool allocations sit below it. CUDA graph
        // capture runs in relaxed mode, which allows cudaMalloc during capture.
        CUDA_CHECK(ggml_cuda_device_malloc((void **) &qe->ptr, q8_bytes, ctx.device));
        qe->cap = q8_bytes;
        qe->dev = ctx.device;
    }
    qe->src1        = src1;
    qe->data        = src1->data;
    qe->epoch       = ctx.graph_epoch;
    qe->size        = q8_bytes;
    qe->ne10_padded = ne10_padded;
    qe->swizzle_iq4 = ggml_cuda_q8_cache_swizzle(type_src0);
    return qe->ptr;
}

bool ggml_cuda_mul_mat_iq4xs_wide_try(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    if (src0->type != GGML_TYPE_IQ4_XS || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            ggml_cuda_iq4xs_wide_mr() == 0) {
        return false;
    }
    const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne11 = src1->ne[1];
    const int max_n = ggml_cuda_iq4xs_wide_48_enabled() ? IQ4XS_WIDE_MAX_N_EXT : IQ4XS_WIDE_MAX_N;
    if (ne11 < IQ4XS_WIDE_MIN_N || ne11 > max_n ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            ne00 <= 0 || ne00 % QK_K != 0 || ne00 > std::numeric_limits<int>::max() || src1->ne[0] != ne00 ||
            ne01 <= 0 || ne01 % 16 != 0 || ne01 > std::numeric_limits<int>::max() ||
            src0->nb[1] % ggml_type_size(src0->type) != 0 || src1->nb[0] != sizeof(float) || src1->nb[1] % sizeof(float) != 0 ||
            dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 ||
            (uintptr_t) src0->data % alignof(uint2) != 0 || src0->nb[1] % alignof(uint2) != 0) {
        return false;
    }
    // 21-32 columns with a long K (the FFN down projection): MMQ measured equal or faster (102 vs 108 us at 5120x17408, w24)
    if (ne11 > 20 && ne11 <= IQ4XS_WIDE_MAX_N && ne00 >= 16384) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    ggml_cuda_mul_mat_iq4xs_wide(ctx, src0->data, (const float *) src1->data, (float *) dst->data,
                                 (int) ne00, (int) ne01, (int) ne11,
                                 src0->nb[1] / ggml_type_size(src0->type), src1->nb[1] / sizeof(float), dst->nb[1] / sizeof(float));
    return true;
#else
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
#endif
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, bool convrot) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);
    GGML_ASSERT(!convrot || (!ids && !fusion));

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(ids || dst->ne[1] == 1 || (dst->ne[1] <= 8 && fusion->gate &&
                !fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale &&
                (src0->type == GGML_TYPE_IQ3_S || src0->type == GGML_TYPE_IQ4_XS || src0->type == GGML_TYPE_Q4_K)));
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // Other types, fused/batched ops and unsupported layouts keep the existing MMVQ route.
    if (src0->type == GGML_TYPE_IQ4_XS && ggml_cuda_iq4xs_k4_variant() != IQ4XS_K4_OFF &&
            !ids && !fusion && !convrot && ne1 >= ggml_cuda_iq4xs_k4_min_n() && ne1 <= 8 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
            ne00 > 0 && ne00 % QK_K == 0 && ne00 <= std::numeric_limits<int>::max() &&
            ne01 > 0 && ne01 % IQ4XS_TC_ROWS == 0 && ne01 <= std::numeric_limits<int>::max() &&
            nb01 % ts_src0 == 0 && nb11 % ts_src1 == 0 && nb1 % ts_dst == 0 &&
            (uintptr_t) src0->data % alignof(uint2) == 0) {
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) {
            if (ggml_cuda_iq4xs_k4_variant() == IQ4XS_K4A) {
                ggml_cuda_mul_mat_iq4xs_k4<true>(ctx, src0->data, src1_d, dst_d, ne00, ne01, ne1,
                                               nb01 / ts_src0, nb11 / ts_src1, nb1 / ts_dst);
            } else {
                ggml_cuda_mul_mat_iq4xs_k4<false>(ctx, src0->data, src1_d, dst_d, ne00, ne01, ne1,
                                                nb01 / ts_src0, nb11 / ts_src1, nb1 / ts_dst);
            }
            return;
        }
    }
#endif

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_bytes = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;

    // Shared-quantize cache: reuse an earlier quantization when the same src1 tensor is
    // consumed again in this graph eval with the same layout (see q8_cache in common.cuh).
    // The layout depends on src0->type only through the IQ4_XS swizzle (quantize_row_q8_1_cuda).
    // ConvRot types and MUL_MAT_ID stay uncached; oversized batches fall back too.
    auto & qc = ctx.q8_cache;
    const bool q8_cacheable = !convrot && ids == nullptr && ggml_cuda_q8_cacheable(ctx, q8_bytes);
    ggml_backend_cuda_context::q8_cache_entry * qe =
        q8_cacheable ? ggml_cuda_q8_cache_find(ctx, src1, src0->type, q8_bytes, ne10_padded) : nullptr;
    const bool q8_hit = qe != nullptr;
    if (q8_hit) {
        qe->last_use = ++qc.tick;
    }

    ggml_cuda_pool_alloc<char> src1_q8_1_local(ctx.pool());
    char * src1_q8_1 = nullptr;

    if (q8_hit) {
        ctx.fusion_stats.q8_cache_hits++;
        src1_q8_1 = qe->ptr;
    } else {
        if (q8_cacheable) {
            src1_q8_1 = ggml_cuda_q8_cache_claim(ctx, src1, src0->type, q8_bytes, ne10_padded);
        } else {
            src1_q8_1 = src1_q8_1_local.alloc(q8_bytes);
        }

        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        if (convrot) {
            ggml_cuda_convrot_quantize_q8_1(
                src1_d, src1_q8_1, ne10, s11, s12, s13,
                ne10_padded, ne11, ne12, ne13, stream);
        } else {
            quantize_row_q8_1_cuda(
                src1_d, nullptr, src1_q8_1, src0->type, ne10, s11, s12, s13,
                ne10_padded, ne11, ne12, ne13, stream);
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    // fork: restrict full-head column reuse to the helper graph's explicit output name.
    constexpr bool batch_head_warps = calc_nwarps(GGML_TYPE_Q5_K, 1, MMVQ_PARAMETERS_GENERIC) == 4 &&
                                     calc_nwarps(GGML_TYPE_Q5_K, 3, MMVQ_PARAMETERS_GENERIC) == 4 &&
                                     calc_nwarps(GGML_TYPE_Q5_K, 4, MMVQ_PARAMETERS_GENERIC) == 4;
    constexpr char batch_head_prefix[] = "mtp_chain_full_logits";
    if (batch_head_warps && src0->type == GGML_TYPE_Q5_K && !ids && !fusion && !convrot &&
            ne00 == 2560 && ne10 == 2560 && ne01 >= 65536 && ne01 <= std::numeric_limits<int>::max() / 10 &&
            (ne11 == 3 || ne11 == 4) && ne1 == ne11 &&
            ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 && ne2 == 1 && ne3 == 1 &&
            ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
            strncmp(dst->name, batch_head_prefix, sizeof(batch_head_prefix) - 1) == 0 &&
            ggml_cuda_info().devices[ctx.device].cc == 860) {
        const char * env = getenv("GGML_CUDA_MTP_Q5_BATCH_HEAD");
        if (env && strcmp(env, "1") == 0) {
            const char * audit = getenv("GGML_CUDA_MTP_Q5_BATCH_HEAD_AUDIT");
            if (audit && strcmp(audit, "1") == 0) {
                GGML_LOG_INFO("mtp_q5_batch_head_dispatch rows=%lld k=%lld cols=%lld name=%s\n",
                        (long long) ne01, (long long) ne00, (long long) ne11, dst->name);
            }
            if (ne11 == 3) {
                mul_mat_vec_q5_K_batch_head_launch<3>(src0->data, (const block_q8_1 *) src1_q8_1, dst_d, ne01, stream);
            } else {
                mul_mat_vec_q5_K_batch_head_launch<4>(src0->data, (const block_q8_1 *) src1_q8_1, dst_d, ne01, stream);
            }
            return;
        }
    }
#endif

    // [KQROWS] plain 2D K-quant products at 2..8 columns on the row kernel (GGML_CUDA_KQ_ROWS=1, default off)
    {
        const bool any_fusion = fusion_local.gate || fusion_local.x_bias || fusion_local.gate_bias ||
                                fusion_local.x_scale || fusion_local.gate_scale || fusion_local.shared_up;
        if (!ids && !any_fusion && !convrot && ne02 == 1 && ne03 == 1 && ne12 == 1 && ne13 == 1 &&
                ggml_cuda_kqrows_try(src0->type, src0->data, src1_q8_1, dst_d, ne00, ne01, ncols_dst, s01,
                    stride_col_y, stride_col_dst, ggml_cuda_info().devices[ctx.device].cc, stream)) {
            return;
        }
    }

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

template <ggml_type type>
static void ggml_cuda_moe_cache_mmv_t(
        const void * pool, const char * act_q8,
        const int32_t * ids_dev, const int32_t * act_ids_dev,
        float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
        int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
        cudaStream_t stream) {
    const int64_t ts0 = ggml_type_size(type);
    const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
    const int64_t s01 = ggml_row_size(type, n_in) / ts0;
    const int64_t s02 = slot_stride_bytes / ts0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s12 = act_rows * s11;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_moe_launch<type>(
        pool, act_q8, ids_dev, act_ids_dev, nullptr, fusion_local, dst_dev, n_in,
        init_fastdiv_values(act_rows), n_out,
        s01, s12, n_out, s02, s11, n_out,
        1, n_hits, warp_size, n_hits,
        0.0f, 0.0f, 0.0f, 0.0f, stream);

    GGML_UNUSED(n_slots);
}

void ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8,
    const int32_t * ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows, cudaStream_t stream) {

    const int64_t ts0 = ggml_type_size(type0);
    GGML_ASSERT(slot_stride_bytes % ts0 == 0);

    if (!act_ids_dev) {
        const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
        const int64_t s01 = ggml_row_size(type0, n_in) / ts0;
        const int64_t s02 = slot_stride_bytes / ts0;
        const int64_t s11 = ne10_padded / QK8_1;
        const int64_t s12 = act_rows * s11;
        ggml_cuda_mm_fusion_args_device fusion_local{};
        mul_mat_vec_q_switch_type(
            pool, type0, act_q8, ids_dev, fusion_local, dst_dev, n_in,
            n_out, 1, s01, s12, n_out,
            n_slots, act_rows, n_hits, s02, s11, n_out,
            1, 1, s02*n_slots, s12, n_out*n_hits, n_hits, stream,
            false);
        return;
    }

#define MOE_CACHE_MMV_CASE(type_name) \
        case type_name: \
            ggml_cuda_moe_cache_mmv_t<type_name>( \
                pool, act_q8, ids_dev, act_ids_dev, dst_dev, n_in, n_out, \
                n_slots, slot_stride_bytes, n_hits, act_rows, stream); \
            break
    switch (type0) {
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q1_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q2_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q4_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q4_1);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q5_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q5_1);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q8_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_MXFP4);
        MOE_CACHE_MMV_CASE(GGML_TYPE_NVFP4);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q2_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q3_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q4_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q5_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q6_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ2_XXS);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ2_XS);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ2_S);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ3_XXS);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ3_S);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ1_S);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ1_M);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ4_NL);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ4_XS);
        default:
            GGML_ABORT("unsupported MoE cache type");
    }
#undef MOE_CACHE_MMV_CASE
}

template <ggml_type type>
static void ggml_cuda_moe_table_mmv_t(const void * const * table, const char * act_q8,
        const int32_t * ids, float * dst, int64_t n_in, int64_t n_out,
        int64_t n_ids, int64_t n_tokens, int64_t act_channels, int64_t ids_stride, cudaStream_t stream) {
    ggml_cuda_mm_fusion_args_device fusion{};
    const int64_t s11 = GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1;
    mul_mat_vec_q_moe_launch<type>(
        nullptr, act_q8, ids, nullptr, nullptr, fusion, dst, n_in,
        init_fastdiv_values(act_channels), n_out,
        ggml_row_size(type, n_in) / ggml_type_size(type), act_channels*s11, n_ids*n_out,
        0, s11, n_out, n_tokens, ids_stride,
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, n_ids,
        0.0f, 0.0f, 0.0f, 0.0f, stream, table);
}

// fork (missjoin hit-first): the same launch restricted to routes whose expert residency flag matches.
template <ggml_type type>
static void ggml_cuda_moe_table_mmv_filtered_t(const void * const * table, const char * act_q8,
        const int32_t * ids, float * dst, int64_t n_in, int64_t n_out,
        int64_t n_ids, int64_t n_tokens, int64_t act_channels, int64_t ids_stride,
        const uint8_t * route_hot, int route_want_hot, cudaStream_t stream) {
    ggml_cuda_mm_fusion_args_device fusion{};
    const int64_t s11 = GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1;
    mul_mat_vec_q_moe_launch<type, false, false, true>(
        nullptr, act_q8, ids, nullptr, nullptr, fusion, dst, n_in,
        init_fastdiv_values(act_channels), n_out,
        ggml_row_size(type, n_in) / ggml_type_size(type), act_channels*s11, n_ids*n_out,
        0, s11, n_out, n_tokens, ids_stride,
        ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, n_ids,
        0.0f, 0.0f, 0.0f, 0.0f, stream, table, nullptr, route_hot, route_want_hot);
}

// Types with a filtered instantiation (the measured model inventory plus the test fixture type).
bool ggml_cuda_moe_table_mmv_filtered_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_Q4_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_Q4_K:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_moe_table_mmv_filtered(const void * const * table, ggml_type type,
        const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens,
        int64_t act_channels, int64_t ids_stride, const uint8_t * route_hot, int route_want_hot, cudaStream_t stream) {
#define MOE_TABLE_FILTERED_CASE(T) case T: ggml_cuda_moe_table_mmv_filtered_t<T>(table, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, act_channels, ids_stride, route_hot, route_want_hot, stream); return true
    switch (type) {
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_Q2_0);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_Q4_0);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_IQ2_XXS);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_IQ2_XS);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_IQ2_S);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_IQ3_XXS);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_IQ3_S);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_IQ4_NL);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_IQ4_XS);
        MOE_TABLE_FILTERED_CASE(GGML_TYPE_Q4_K);
        default: return false;
    }
#undef MOE_TABLE_FILTERED_CASE
}

void ggml_cuda_moe_table_mmv(const void * const * table, ggml_type type,
        const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens,
        int64_t act_channels, int64_t ids_stride, cudaStream_t stream) {
#define MOE_TABLE_CASE(T) case T: ggml_cuda_moe_table_mmv_t<T>(table, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, act_channels, ids_stride, stream); break
    switch (type) {
        MOE_TABLE_CASE(GGML_TYPE_Q1_0);
        MOE_TABLE_CASE(GGML_TYPE_Q2_0);
        MOE_TABLE_CASE(GGML_TYPE_Q4_0);
        MOE_TABLE_CASE(GGML_TYPE_Q4_1);
        MOE_TABLE_CASE(GGML_TYPE_Q5_0);
        MOE_TABLE_CASE(GGML_TYPE_Q5_1);
        MOE_TABLE_CASE(GGML_TYPE_Q8_0);
        MOE_TABLE_CASE(GGML_TYPE_MXFP4);
        MOE_TABLE_CASE(GGML_TYPE_NVFP4);
        MOE_TABLE_CASE(GGML_TYPE_Q2_K);
        MOE_TABLE_CASE(GGML_TYPE_Q3_K);
        MOE_TABLE_CASE(GGML_TYPE_Q4_K);
        MOE_TABLE_CASE(GGML_TYPE_Q5_K);
        MOE_TABLE_CASE(GGML_TYPE_Q6_K);
        MOE_TABLE_CASE(GGML_TYPE_IQ2_XXS);
        MOE_TABLE_CASE(GGML_TYPE_IQ2_XS);
        MOE_TABLE_CASE(GGML_TYPE_IQ2_S);
        MOE_TABLE_CASE(GGML_TYPE_IQ3_XXS);
        MOE_TABLE_CASE(GGML_TYPE_IQ3_S);
        MOE_TABLE_CASE(GGML_TYPE_IQ1_S);
        MOE_TABLE_CASE(GGML_TYPE_IQ1_M);
        MOE_TABLE_CASE(GGML_TYPE_IQ4_NL);
        MOE_TABLE_CASE(GGML_TYPE_IQ4_XS);
        default: GGML_ABORT("unsupported mapped expert type");
    }
#undef MOE_TABLE_CASE
}

// fork: gate and up keep independent resident/mapped expert address tables.
template <ggml_type type>
static void ggml_cuda_moe_table_mmv_fused_t(const void * const * up_table, const void * const * gate_table,
        const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens,
        int64_t act_channels, int64_t ids_stride, cudaStream_t stream) {
    ggml_cuda_mm_fusion_args_device fusion{};
    fusion.gate = gate_table;
    fusion.glu_op = GGML_GLU_OP_SWIGLU;
    const int64_t s11 = GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1;
    mul_mat_vec_q_moe_launch<type, true>(nullptr, act_q8, ids, nullptr, nullptr, fusion, dst,
            n_in, init_fastdiv_values(act_channels), n_out,
            ggml_row_size(type, n_in) / ggml_type_size(type), act_channels*s11, n_ids*n_out,
            0, s11, n_out, n_tokens, ids_stride,
            ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size, n_ids,
            0.0f, 0.0f, 0.0f, 0.0f, stream, up_table, gate_table);
}

void ggml_cuda_moe_table_mmv_fused(const void * const * up_table, const void * const * gate_table, ggml_type type,
        const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens,
        int64_t act_channels, int64_t ids_stride, cudaStream_t stream) {
#define MOE_TABLE_FUSED_CASE(T) case T: ggml_cuda_moe_table_mmv_fused_t<T>(up_table, gate_table, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, act_channels, ids_stride, stream); break
    switch (type) {
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q1_0);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q2_0);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q4_0);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q4_1);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q5_0);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q5_1);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q8_0);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_MXFP4);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_NVFP4);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q2_K);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q3_K);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q4_K);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q5_K);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_Q6_K);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ2_XXS);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ2_XS);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ2_S);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ3_XXS);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ3_S);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ1_S);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ1_M);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ4_NL);
        MOE_TABLE_FUSED_CASE(GGML_TYPE_IQ4_XS);
        default: GGML_ABORT("unsupported fused mapped expert type");
    }
#undef MOE_TABLE_FUSED_CASE
}

// fork: off by default; check mode compares the actual independent-route kernel on the same stream.
bool ggml_cuda_moe_table_mmv_grouped_try(ggml_backend_cuda_context & ctx,
        const void * const * table, const void * const * gate_table, ggml_type type,
        const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels, int64_t ids_stride) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const char * env = getenv("GGML_MOE_MMV_GROUPED");
    const bool check = env && (strcmp(env, "check") == 0 || strcmp(env, "2") == 0);
    // [VBEAT 1b] a type list limits grouping to those types: "iq3_s,q2_0", or "fused:iq3_s" for the fused
    // gate/up call only (the measured win: IQ3_S fused 104.3 vs 129.4 us at width 5; unfused grouped is slower).
    bool listed = false;
    if (env && !check && strcmp(env, "1") != 0) {
        const bool fused_only = strncmp(env, "fused:", 6) == 0;
        const std::string names = std::string(",") + (fused_only ? env + 6 : env) + ",";
        listed = (!fused_only || gate_table) &&
                 names.find(std::string(",") + ggml_type_name(type) + ",") != std::string::npos;
    }
    if (!env || (!check && !listed && strcmp(env, "1") != 0) || ggml_cuda_info().devices[ctx.device].cc != 860 ||
            (type != GGML_TYPE_IQ3_S && type != GGML_TYPE_IQ4_NL && type != GGML_TYPE_Q2_0) ||
            n_tokens < 2 || n_tokens > 5 || n_ids < 1 || n_ids > 64 || channels < 1 || channels > n_ids ||
            n_in < 1 || n_in > INT_MAX - MATRIX_ROW_PADDING || n_in % ggml_blck_size(type) != 0 ||
            n_out < 1 || n_out > INT_MAX / (n_ids*n_tokens) || n_out > INT_MAX / (n_in / ggml_blck_size(type)) ||
            ids_stride < n_ids || ids_stride > INT_MAX / n_tokens) {
        return false;
    }
    ggml_cuda_pool_alloc<float> reference(ctx.pool());
    if (check) {
        float * ref = reference.alloc((size_t)(n_out*n_ids*n_tokens));
        if (gate_table) {
            ggml_cuda_moe_table_mmv_fused(table, gate_table, type, act_q8, ids, ref,
                    n_in, n_out, n_ids, n_tokens, channels, ids_stride, ctx.stream());
        } else {
            ggml_cuda_moe_table_mmv(table, type, act_q8, ids, ref,
                    n_in, n_out, n_ids, n_tokens, channels, ids_stride, ctx.stream());
        }
        GGML_LOG_INFO("[moe-cache] grouped MMV check: type=%s tokens=%lld channels=%lld fused=%d\n",
                ggml_type_name(type), (long long)n_tokens, (long long)channels, gate_table != nullptr);
    }
    const ggml_cuda_kernel_launch_params params(dim3((n_out + 7)/8, n_ids*n_tokens), dim3(32, 4), 0, ctx.stream());
#define MOE_GROUPED_CASE(T) \
    case T: \
        if (gate_table) { \
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_grouped<T, true>, params, table, gate_table, \
                    (const block_q8_1 *)act_q8, ids, dst, (int)n_in, (int)n_out, (int)n_ids, (int)n_tokens, (int)channels, (int)ids_stride); \
        } else { \
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_grouped<T, false>, params, table, gate_table, \
                    (const block_q8_1 *)act_q8, ids, dst, (int)n_in, (int)n_out, (int)n_ids, (int)n_tokens, (int)channels, (int)ids_stride); \
        } \
        break
    switch (type) {
        MOE_GROUPED_CASE(GGML_TYPE_IQ3_S);
        MOE_GROUPED_CASE(GGML_TYPE_IQ4_NL);
        MOE_GROUPED_CASE(GGML_TYPE_Q2_0);
        default: GGML_ABORT("unsupported grouped MoE type");
    }
#undef MOE_GROUPED_CASE
    if (check) {
        const int count = n_out*n_ids*n_tokens;
        moe_grouped_check<<<(count + 255LL)/256, 256, 0, ctx.stream()>>>(
                (const uint32_t *)reference.get(), (const uint32_t *)dst, count);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED_VARS(ctx, table, gate_table, type, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, channels, ids_stride);
    return false;
#endif
}

// [VBEAT D1] decode-once expert kernel; see mul_mat_vec_q_moe_d1. GGML_MOE_MMV_D1: 0 off, 1 on, check = run the
// plain kernel too and trap on any bit difference. GGML_MOE_MMV_D1_SHAPE: warps x rows per warp (4x2, 8x2, 8x1, 4x4),
// with a gN suffix for at most N routes per weight pass (4x2g2, 8x2g2, 8x1g2, 4x2g1; no suffix = 4) and an s suffix
// for the shared-memory IQ2/IQ3 codebook (4x2s, 4x2g2s, 8x2g2s; other types ignore it). Built for the Flash-Next
// expert types only (IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S gate-up, IQ4_NL/Q2_0 down); other types keep the plain kernel.
// GGML_MOE_MMV_D1_MIN_TOKENS: narrowest batch that uses it (default 1).
static int moe_d1_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_MOE_MMV_D1");
        if (!e) {
            return 0;
        }
        return strcmp(e, "check") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}

// [MOEGAP DOWNSW] GGML_MOE_DOWN_SW: 0 off, 1 on, check (trap on any bit difference to the plain kernel); the
// Q2_0 / IQ4_NL expert projections (Flash-Next's down) then take mul_mat_vec_q_moe_sw, whatever GGML_MOE_MMV_D1 says.
// GGML_MOE_DOWN_SW_SHAPE: lanes per row x rows per lane group, gN = routes per weight pass, wN = warps per block:
// 8x2 (default: 8x2g2w4), 16x2, 8x1, 8x2g1, 8x2w8, 16x1, 8x2g4, 4x2.
static int moe_sw_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_MOE_DOWN_SW");
        if (!e) {
            return 0;
        }
        return strcmp(e, "check") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}

static std::atomic<unsigned long long> g_moe_sw_count{0};
unsigned long long ggml_cuda_moe_down_sw_count() { return g_moe_sw_count.load(std::memory_order_relaxed); }

bool ggml_cuda_moe_down_sw_on(ggml_type type) {
    return moe_sw_mode() != 0 && (type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ4_NL);
}

bool ggml_cuda_moe_table_mmv_d1_try(ggml_backend_cuda_context & ctx,
        const void * const * table, ggml_type type, const char * act_q8, const int32_t * ids, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels, int64_t ids_stride,
        const uint8_t * route_hot, int route_want_hot) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const bool sw = ggml_cuda_moe_down_sw_on(type);
    const int mode = sw ? moe_sw_mode() : moe_d1_mode();
    static const int shape = [] {
        const char * e = getenv("GGML_MOE_MMV_D1_SHAPE");
        static const char * const names[] = { "4x2", "8x2", "4x2g2", "8x2g2", "8x1g2", "4x2g1", "4x2s", "4x2g2s", "8x2g2s" };
        for (int i = 0; e && i < 9; ++i) {
            if (strcmp(e, names[i]) == 0) {
                return i;
            }
        }
        return 0;
    }();
    static const int min_tokens = [] {
        const char * e = getenv("GGML_MOE_MMV_D1_MIN_TOKENS");
        return e ? std::max(1, atoi(e)) : 1;
    }();
    const bool d1_type = type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
                         type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_NL;
    if (mode == 0 || !d1_type || ggml_cuda_info().devices[ctx.device].cc != 860 || !ggml_cuda_moe_table_mmv_filtered_supported(type) ||
            n_tokens < min_tokens || n_tokens > 8 || n_ids < 1 || n_ids*n_tokens > 64 || channels < 1 || channels > n_ids ||
            n_in < 1 || n_in > INT_MAX / 2 || n_in % ggml_blck_size(type) != 0 ||
            n_out < 1 || n_out > INT_MAX / (n_ids*n_tokens) || ids_stride < n_ids || ids_stride > INT_MAX / n_tokens) {
        return false;
    }
    if (sw) { g_moe_sw_count.fetch_add(1, std::memory_order_relaxed); }
    static const int sw_shape = [] {
        const char * e = getenv("GGML_MOE_DOWN_SW_SHAPE");
        static const char * const names[] = { "8x2", "16x2", "8x1", "8x2g1", "8x2w8", "16x1", "8x2g4", "4x2" };
        for (int i = 0; e && i < 8; ++i) {
            if (strcmp(e, names[i]) == 0) {
                return i;
            }
        }
        return 0;
    }();
    const int64_t count = n_out*n_ids*n_tokens;
    ggml_cuda_pool_alloc<float> reference(ctx.pool());
    cudaStream_t stream = ctx.stream();
    if (mode == 2) {
        float * ref = reference.alloc((size_t) count);
        if (route_hot) {
            // a filtered pass writes only its routes: start the reference from the same buffer contents
            CUDA_CHECK(cudaMemcpyAsync(ref, dst, (size_t) count*sizeof(float), cudaMemcpyDeviceToDevice, stream));
            GGML_ASSERT(ggml_cuda_moe_table_mmv_filtered(table, type, act_q8, ids, ref, n_in, n_out, n_ids, n_tokens,
                    channels, ids_stride, route_hot, route_want_hot, stream));
        } else {
            ggml_cuda_moe_table_mmv(table, type, act_q8, ids, ref, n_in, n_out, n_ids, n_tokens, channels, ids_stride, stream);
        }
    }
#define MOE_D1_ARGS table, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, channels, ids_stride, route_hot, route_want_hot, stream
#define MOE_D1_SHAPES(T) \
    case T: \
        switch (shape) { \
            case 1:  moe_d1_launch<T, 8, 2, 4>(MOE_D1_ARGS); break; \
            case 2:  moe_d1_launch<T, 4, 2, 2>(MOE_D1_ARGS); break; \
            case 3:  moe_d1_launch<T, 8, 2, 2>(MOE_D1_ARGS); break; \
            case 4:  moe_d1_launch<T, 8, 1, 2>(MOE_D1_ARGS); break; \
            case 5:  moe_d1_launch<T, 4, 2, 1>(MOE_D1_ARGS); break; \
            case 6:  moe_d1_launch<T, 4, 2, 4, ggml_cuda_mmvq_smem_grid_type(T)>(MOE_D1_ARGS); break; \
            case 7:  moe_d1_launch<T, 4, 2, 2, ggml_cuda_mmvq_smem_grid_type(T)>(MOE_D1_ARGS); break; \
            case 8:  moe_d1_launch<T, 8, 2, 2, ggml_cuda_mmvq_smem_grid_type(T)>(MOE_D1_ARGS); break; \
            default: moe_d1_launch<T, 4, 2, 4>(MOE_D1_ARGS); break; \
        } \
        break
#define MOE_SW_SHAPES(T) \
    case T: \
        switch (sw_shape) { \
            case 1:  moe_sw_launch<T, 4, 16, 2, 2>(MOE_D1_ARGS); break; \
            case 2:  moe_sw_launch<T, 4,  8, 1, 2>(MOE_D1_ARGS); break; \
            case 3:  moe_sw_launch<T, 4,  8, 2, 1>(MOE_D1_ARGS); break; \
            case 4:  moe_sw_launch<T, 8,  8, 2, 2>(MOE_D1_ARGS); break; \
            case 5:  moe_sw_launch<T, 4, 16, 1, 2>(MOE_D1_ARGS); break; \
            case 6:  moe_sw_launch<T, 4,  8, 2, 4>(MOE_D1_ARGS); break; \
            case 7:  moe_sw_launch<T, 4,  4, 2, 2>(MOE_D1_ARGS); break; \
            default: moe_sw_launch<T, 4,  8, 2, 2>(MOE_D1_ARGS); break; \
        } \
        break
    if (sw) {
        switch (type) {
            MOE_SW_SHAPES(GGML_TYPE_Q2_0);
            MOE_SW_SHAPES(GGML_TYPE_IQ4_NL);
            default: GGML_ABORT("unsupported sub-warp MoE type");
        }
    } else switch (type) {
        MOE_D1_SHAPES(GGML_TYPE_Q2_0);
        MOE_D1_SHAPES(GGML_TYPE_IQ2_XXS);
        MOE_D1_SHAPES(GGML_TYPE_IQ2_XS);
        MOE_D1_SHAPES(GGML_TYPE_IQ2_S);
        MOE_D1_SHAPES(GGML_TYPE_IQ3_XXS);
        MOE_D1_SHAPES(GGML_TYPE_IQ3_S);
        MOE_D1_SHAPES(GGML_TYPE_IQ4_NL);
        default: GGML_ABORT("unsupported decode-once MoE type");
    }
#undef MOE_SW_SHAPES
#undef MOE_D1_SHAPES
#undef MOE_D1_ARGS
    if (mode == 2) {
        moe_grouped_check<<<(count + 255)/256, 256, 0, stream>>>(
                (const uint32_t *) reference.get(), (const uint32_t *) dst, (int) count);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED_VARS(ctx, table, type, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, channels, ids_stride, route_hot, route_want_hot);
    return false;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// [MOEGAP MISSLIST] miss pass over the plan's compact list: grid (row tiles, slices); slice y takes list entries
// y, y + slices, ... so a pass with nothing to do is one load per block. Per route the dot code, lane split and
// reduction are those of the plain kernel (moe_d1_rows / moe_hitg_rows), so every output keeps its bits.
template <ggml_type type, int c_warps, int c_rows, bool c_fused, int c_lanes = 32>
__launch_bounds__(c_warps*32, 768/(c_warps*32))
static __global__ void mul_mat_vec_q_moe_list(const moe_miss_list * list,
        const void * const * table, const void * const * gate_table, const block_q8_1 * vy, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int act_channels,
        const int stride_col_y, const int stride_channel_y, const int stride_col_dst, const int stride_channel_dst,
        const int n_ids) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    ggml_cuda_pdl_sync();
    const int n = list->n;
    const int wrow0 = (blockIdx.x*c_warps + threadIdx.y)*((32/c_lanes)*c_rows);
    if ((int) blockIdx.y >= n || wrow0 >= nrows_x) {
        return;
    }
    const int row0 = wrow0 + (c_lanes < 32 ? (int) (threadIdx.x / c_lanes)*c_rows : 0);
    const int blocks_per_row_x = ncols_x / qk;
    for (int j = blockIdx.y; j < n; j += gridDim.y) {
        const int e = list->expert[j];
        unsigned long long left = list->routes[j];
        const void * vx = table[e];
        const void * vg = c_fused ? gate_table[e] : nullptr;
        while (left != 0) {
            const block_q8_1 * y[4];
            float * out[4];
            int m = 0;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (q < 2 && left != 0) {
                    const int r = __ffsll((long long) left) - 1;
                    left &= left - 1;
                    const int t = r / n_ids;
                    const int k = r - t*n_ids;
                    y[q]   = vy + (k % act_channels)*stride_channel_y + t*stride_col_y;
                    out[q] = dst + k*stride_channel_dst + t*stride_col_dst;
                    ++m;
                } else {
                    y[q]   = vy;
                    out[q] = dst;
                }
            }
            if constexpr (c_fused) {
                if (m == 2) {
                    moe_hitg_rows<type, c_rows, 2, false>(vx, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, nullptr);
                } else {
                    moe_hitg_rows<type, c_rows, 1, false>(vx, vg, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, nullptr);
                }
            } else if constexpr (c_lanes < 32) {
                if (m == 2) {
                    moe_sw_rows<type, c_lanes, c_rows, 2>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x);
                } else {
                    moe_sw_rows<type, c_lanes, c_rows, 1>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x);
                }
            } else {
                if (m == 2) {
                    moe_d1_rows<type, c_rows, 2, false>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, nullptr);
                } else {
                    moe_d1_rows<type, c_rows, 1, false>(vx, y, out, row0, nrows_x, blocks_per_row_x, stride_row_x, nullptr);
                }
            }
        }
    }
    ggml_cuda_pdl_lc();
}
#endif

void ggml_cuda_moe_bits_check(const float * reference, const float * actual, int64_t count, cudaStream_t stream) {
    moe_grouped_check<<<(count + 255)/256, 256, 0, stream>>>((const uint32_t *) reference, (const uint32_t *) actual, (int) count);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_moe_table_mmv_list_supported(ggml_type type, bool fused, int64_t n_in, int64_t n_ids, int64_t n_tokens,
        int64_t channels) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const bool gu = type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
                    type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
    const bool t = gu || (!fused && (type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ4_NL));
    return t && n_tokens >= 1 && n_ids >= 1 && n_ids*n_tokens <= 64 && channels >= 1 && channels <= n_ids &&
           n_in >= 1 && n_in <= INT_MAX / 2 && n_in % ggml_blck_size(type) == 0;
#else
    GGML_UNUSED_VARS(type, fused, n_in, n_ids, n_tokens, channels);
    return false;
#endif
}

void ggml_cuda_moe_table_mmv_list(ggml_backend_cuda_context & ctx, const moe_miss_list * list,
        const void * const * table, const void * const * gate_table, ggml_type type, const char * act_q8, float * dst,
        int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens, int64_t channels) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_ASSERT(ggml_cuda_moe_table_mmv_list_supported(type, gate_table != nullptr, n_in, n_ids, n_tokens, channels));
    static const int slices = [] {
        const char * e = getenv("GGML_MOE_MISS_LIST_SLICES");
        return e ? std::max(1, std::min(MOE_MISS_LIST_MAX, atoi(e))) : 4;
    }();
    constexpr int c_warps = 4, c_rows = 2;
    const int s11 = (int) (GGML_PAD(n_in, MATRIX_ROW_PADDING) / QK8_1);
    const int stride_row_x = (int) (ggml_row_size(type, n_in) / ggml_type_size(type));
    const ggml_cuda_kernel_launch_params params(dim3((n_out + c_warps*c_rows - 1)/(c_warps*c_rows), slices),
            dim3(32, c_warps), 0, ctx.stream());
    const ggml_cuda_kernel_launch_params params_sw(dim3((n_out + c_warps*4*c_rows - 1)/(c_warps*4*c_rows), slices),
            dim3(32, c_warps), 0, ctx.stream());
#define MOE_LIST_ARGS list, table, gate_table, (const block_q8_1 *) act_q8, dst, (int) n_in, (int) n_out, stride_row_x, \
        (int) channels, (int) channels*s11, s11, (int) (n_ids*n_out), (int) n_out, (int) n_ids
#define MOE_LIST_CASE(T) \
    case T: \
        if (gate_table) { \
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_list<T, c_warps, c_rows, true>, params, MOE_LIST_ARGS); \
        } else { \
            ggml_cuda_kernel_launch(mul_mat_vec_q_moe_list<T, c_warps, c_rows, false>, params, MOE_LIST_ARGS); \
        } \
        break
    switch (type) {
        MOE_LIST_CASE(GGML_TYPE_IQ2_XXS);
        MOE_LIST_CASE(GGML_TYPE_IQ2_XS);
        MOE_LIST_CASE(GGML_TYPE_IQ2_S);
        MOE_LIST_CASE(GGML_TYPE_IQ3_XXS);
        MOE_LIST_CASE(GGML_TYPE_IQ3_S);
        // down types: the sub-warp rows (8 lanes x 2 rows, bit-identical) when GGML_MOE_DOWN_SW is on
        case GGML_TYPE_Q2_0:
            GGML_ASSERT(!gate_table);
            if (ggml_cuda_moe_down_sw_on(type)) {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe_list<GGML_TYPE_Q2_0, c_warps, c_rows, false, 8>, params_sw, MOE_LIST_ARGS);
            } else {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe_list<GGML_TYPE_Q2_0, c_warps, c_rows, false>, params, MOE_LIST_ARGS);
            }
            break;
        case GGML_TYPE_IQ4_NL:
            GGML_ASSERT(!gate_table);
            if (ggml_cuda_moe_down_sw_on(type)) {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe_list<GGML_TYPE_IQ4_NL, c_warps, c_rows, false, 8>, params_sw, MOE_LIST_ARGS);
            } else {
                ggml_cuda_kernel_launch(mul_mat_vec_q_moe_list<GGML_TYPE_IQ4_NL, c_warps, c_rows, false>, params, MOE_LIST_ARGS);
            }
            break;
        default: GGML_ABORT("unsupported miss-list type");
    }
#undef MOE_LIST_CASE
#undef MOE_LIST_ARGS
    CUDA_CHECK(cudaGetLastError());
    GGML_UNUSED(n_tokens);
#else
    GGML_UNUSED_VARS(ctx, list, table, gate_table, type, act_q8, dst, n_in, n_out, n_ids, n_tokens, channels);
    GGML_ABORT("miss-list kernel not built");
#endif
}

// [MOEGAP HITG] GGML_MOE_HIT_FUSED: 0 off, 1 on, check = also run the plain up and gate kernels and trap on any
// bit difference. GGML_MOE_HIT_FUSED_SHAPE: warps x rows per warp, s = shared-memory codebook (4x2, 8x2, 4x1, 8x1,
// 4x2s, 8x2s, 8x1s; default 4x2). Members are taken two tokens at a time. Gate/up types of Flash-Next only.
// Sub-warp shapes (HITGSW): swLxR[gG][wW] = L lanes per row x R rows per lane group, G tokens per weight read
// (default 2), W warps per block (default 4): sw16x2, sw16x2g4, sw8x2, sw8x2g4, sw16x4, sw16x1g4, sw16x2w8,
// sw16x1g4w16 (32 rows per 512-thread block), sw16x2g4w8 (32 rows per 256-thread block), sw16x1g4s and sw16x2s
// (shared-memory codebook); auto = per type and width.
int ggml_cuda_moe_hit_fused_mode() {
    static const int mode = [] {
        const char * e = getenv("GGML_MOE_HIT_FUSED");
        if (!e) {
            return 0;
        }
        return strcmp(e, "check") == 0 ? 2 : atoi(e) != 0 ? 1 : 0;
    }();
    return mode;
}

bool ggml_cuda_moe_table_mmv_hitg_supported(ggml_type type, int64_t n_in, int64_t n_out, int64_t n_ids,
        int64_t n_tokens, int64_t channels, int64_t ids_stride) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const bool t = type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
                   type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S;
    return t && ggml_cuda_moe_hit_fused_mode() != 0 && n_tokens >= 1 && n_tokens <= 8 && n_ids >= 1 &&
           n_ids*n_tokens <= 64 && channels >= 1 && channels <= n_ids && n_in >= 1 && n_in <= INT_MAX / 2 &&
           n_in % ggml_blck_size(type) == 0 && n_out >= 1 && n_out <= INT_MAX / (n_ids*n_tokens) &&
           ids_stride >= n_ids && ids_stride <= INT_MAX / n_tokens;
#else
    GGML_UNUSED_VARS(type, n_in, n_out, n_ids, n_tokens, channels, ids_stride);
    return false;
#endif
}

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
// shape index as GGML_MOE_HIT_FUSED_SHAPE's name list (0 = default 4x2); shared by the op and the bench below
static void moe_hitg_dispatch(int shape, const void * const * up_table, const void * const * gate_table, ggml_type type,
        const char * act_q8, const int32_t * ids, float * dst, int64_t n_in, int64_t n_out, int64_t n_ids,
        int64_t n_tokens, int64_t channels, int64_t ids_stride, const uint8_t * m0, const uint8_t * m1, int want,
        cudaStream_t stream) {
#define MOE_HITG_ARGS up_table, gate_table, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, channels, ids_stride, m0, m1, want, stream
#define MOE_HITG_SHAPES(T) \
    case T: \
        switch (shape) { \
            case 1:  moe_hitg_launch<T, 8, 2, 2>(MOE_HITG_ARGS); break; \
            case 2:  moe_hitg_launch<T, 4, 1, 2>(MOE_HITG_ARGS); break; \
            case 3:  moe_hitg_launch<T, 8, 1, 2>(MOE_HITG_ARGS); break; \
            case 4:  moe_hitg_launch<T, 4, 2, 2, true>(MOE_HITG_ARGS); break; \
            case 5:  moe_hitg_launch<T, 8, 2, 2, true>(MOE_HITG_ARGS); break; \
            case 6:  moe_hitg_launch<T, 8, 1, 2, true>(MOE_HITG_ARGS); break; \
            case 7:  moe_hitg_sw_launch<T, 4, 16, 2, 2>(MOE_HITG_ARGS); break; \
            case 8:  moe_hitg_sw_launch<T, 4, 16, 2, 4>(MOE_HITG_ARGS); break; \
            case 9:  moe_hitg_sw_launch<T, 4, 8, 2, 2>(MOE_HITG_ARGS); break; \
            case 10: moe_hitg_sw_launch<T, 4, 8, 2, 4>(MOE_HITG_ARGS); break; \
            case 11: moe_hitg_sw_launch<T, 4, 16, 4, 2>(MOE_HITG_ARGS); break; \
            case 12: moe_hitg_sw_launch<T, 4, 16, 1, 4>(MOE_HITG_ARGS); break; \
            case 13: moe_hitg_sw_launch<T, 8, 16, 2, 2>(MOE_HITG_ARGS); break; \
            case 14: moe_hitg_sw_launch<T, 16, 16, 1, 4>(MOE_HITG_ARGS); break; \
            case 15: moe_hitg_sw_launch<T, 8, 16, 2, 4>(MOE_HITG_ARGS); break; \
            case 16: moe_hitg_sw_launch<T, 4, 16, 1, 4, true>(MOE_HITG_ARGS); break; \
            case 17: moe_hitg_sw_launch<T, 4, 16, 2, 2, true>(MOE_HITG_ARGS); break; \
            default: moe_hitg_launch<T, 4, 2, 2>(MOE_HITG_ARGS); break; \
        } \
        break
    switch (type) {
        MOE_HITG_SHAPES(GGML_TYPE_IQ2_XXS);
        MOE_HITG_SHAPES(GGML_TYPE_IQ2_XS);
        MOE_HITG_SHAPES(GGML_TYPE_IQ2_S);
        MOE_HITG_SHAPES(GGML_TYPE_IQ3_XXS);
        MOE_HITG_SHAPES(GGML_TYPE_IQ3_S);
        default: GGML_ABORT("unsupported fused hit type");
    }
#undef MOE_HITG_SHAPES
#undef MOE_HITG_ARGS
}

#endif

// [MOEGAP HITGBENCH] Kernel-only timing of the fused hit pass for tests/test-moe-hitg-bench (not used by the
// server). Flash-Next shape: K = 2560, 640 rows, 10 experts per token. n_pool experts are drawn from per token (the
// union is what one launch reads), n_sets copies of the weights are cycled so each launch reads cold memory as a
// decode layer does. Weights and activations are random with small scales. Returns the mean us per launch and the MB
// one launch reads; *same = 1 when this shape's output has the bits of shape 0.
extern "C" bool ggml_cuda_moe_hitg_bench(int type_i, int n_tokens, int n_pool, int n_sets, int reps, int shape,
        unsigned seed, double * us, double * mb, int * n_union, int * same) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    const ggml_type type = (ggml_type) type_i;
    const int64_t n_in = 2560, n_out = 640, n_ids = 10;
    if (!ggml_cuda_moe_table_mmv_hitg_supported(type, n_in, n_out, n_ids, n_tokens, 1, n_ids) || n_pool < n_ids ||
            n_pool > 512 || n_sets < 1 || reps < 1) {
        return false;
    }
    const size_t mat = (size_t) n_out * ggml_row_size(type, n_in);
    const size_t blk = ggml_type_size(type);
    uint64_t st = seed*2654435761ull + 12345;
    auto rnd = [&st]() { st = st*6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (st >> 33); };
    std::vector<uint8_t> h(mat);
    std::vector<char *> w((size_t) n_sets*n_pool*2);
    for (auto & p : w) {
        CUDA_CHECK(cudaMalloc((void **) &p, mat));
        for (size_t b = 0; b < mat; ++b) { h[b] = (uint8_t) rnd(); }
        for (size_t b = 0; b + blk <= mat; b += blk) {   // every IQ block starts with its f16 scale d
            const ggml_half d = __float2half(0.002f + 0.002f*(rnd() % 8)/8.0f);
            memcpy(&h[b], &d, sizeof(d));
        }
        CUDA_CHECK(cudaMemcpy(p, h.data(), mat, cudaMemcpyHostToDevice));
    }
    std::vector<const void *> tab((size_t) n_sets*2*512, nullptr);
    for (int s_ = 0; s_ < n_sets; ++s_) {
        for (int e = 0; e < n_pool; ++e) {
            tab[((size_t) s_*2 + 0)*512 + e] = w[((size_t) s_*n_pool + e)*2 + 0];
            tab[((size_t) s_*2 + 1)*512 + e] = w[((size_t) s_*n_pool + e)*2 + 1];
        }
    }
    const void ** d_tab;
    CUDA_CHECK(cudaMalloc((void **) &d_tab, tab.size()*sizeof(void *)));
    CUDA_CHECK(cudaMemcpy(d_tab, tab.data(), tab.size()*sizeof(void *), cudaMemcpyHostToDevice));
    const int s11 = (int) (n_in / QK8_1);
    std::vector<block_q8_1> y((size_t) n_tokens*s11);
    for (auto & b : y) {
        for (int j = 0; j < QK8_1; ++j) { b.qs[j] = (int8_t) (rnd() % 255 - 127); }
        b.ds = make_half2(__float2half(0.02f), __float2half(0.0f));
    }
    std::vector<int32_t> ids((size_t) n_tokens*n_ids);
    std::vector<int> used(n_pool, 0);
    for (int t = 0; t < n_tokens; ++t) {
        for (int k = 0; k < n_ids; ++k) {
            int e;
            bool dup;
            do {
                e = (int) (rnd() % n_pool);
                dup = false;
                for (int j = 0; j < k; ++j) { dup = dup || ids[t*n_ids + j] == e; }
            } while (dup);
            ids[t*n_ids + k] = e;
            used[e] = 1;
        }
    }
    *n_union = 0;
    for (int e = 0; e < n_pool; ++e) { *n_union += used[e]; }
    *mb = 2.0 * mat * *n_union / 1e6;
    block_q8_1 * d_y; int32_t * d_ids; uint8_t * d_m; float * d_dst; float * d_ref;
    const size_t n_dst = (size_t) n_tokens*n_ids*n_out;
    CUDA_CHECK(cudaMalloc((void **) &d_y, y.size()*sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc((void **) &d_ids, ids.size()*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void **) &d_m, 512));
    CUDA_CHECK(cudaMalloc((void **) &d_dst, n_dst*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &d_ref, n_dst*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), y.size()*sizeof(block_q8_1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), ids.size()*sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_m, 1, 512));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto run = [&](int sh, int set, float * out) {
        moe_hitg_dispatch(sh, d_tab + (size_t) set*2*512 + 0, d_tab + (size_t) set*2*512 + 512, type,
                (const char *) d_y, d_ids, out, n_in, n_out, n_ids, n_tokens, 1, n_ids, d_m, d_m, 1, stream);
    };
    CUDA_CHECK(cudaMemsetAsync(d_ref, 0, n_dst*sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_dst, 0, n_dst*sizeof(float), stream));
    run(0, 0, d_ref);
    run(shape, 0, d_dst);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> a(n_dst), b(n_dst);
    CUDA_CHECK(cudaMemcpy(a.data(), d_ref, n_dst*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_dst, n_dst*sizeof(float), cudaMemcpyDeviceToHost));
    *same = memcmp(a.data(), b.data(), n_dst*sizeof(float)) == 0;
    for (int i = 0; i < 3*n_sets; ++i) { run(shape, i % n_sets, d_dst); }
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < reps; ++i) { run(shape, i % n_sets, d_dst); }
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    *us = 1000.0 * ms / reps;
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaStreamDestroy(stream);
    cudaFree(d_y); cudaFree(d_ids); cudaFree(d_m); cudaFree(d_dst); cudaFree(d_ref); cudaFree(d_tab);
    for (auto p : w) { cudaFree(p); }
    return true;
#else
    GGML_UNUSED_VARS(type_i, n_tokens, n_pool, n_sets, reps, shape, seed, us, mb, n_union, same);
    return false;
#endif
}

void ggml_cuda_moe_table_mmv_hitg(ggml_backend_cuda_context & ctx,
        const void * const * up_table, const void * const * gate_table, ggml_type type, const char * act_q8,
        const int32_t * ids, float * dst, int64_t n_in, int64_t n_out, int64_t n_ids, int64_t n_tokens,
        int64_t channels, int64_t ids_stride, const uint8_t * m0, const uint8_t * m1, int want) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
    GGML_ASSERT(ggml_cuda_moe_table_mmv_hitg_supported(type, n_in, n_out, n_ids, n_tokens, channels, ids_stride));
    static const int shape = [] {
        const char * e = getenv("GGML_MOE_HIT_FUSED_SHAPE");
        static const char * const names[] = { "4x2", "8x2", "4x1", "8x1", "4x2s", "8x2s", "8x1s",
                                              "sw16x2", "sw16x2g4", "sw8x2", "sw8x2g4", "sw16x4", "sw16x1g4", "sw16x2w8",
                                              "sw16x1g4w16", "sw16x2g4w8", "sw16x1g4s", "sw16x2s" };
        for (int i = 0; e && i < 18; ++i) {
            if (strcmp(e, names[i]) == 0) {
                return i;
            }
        }
        return e && strcmp(e, "auto") == 0 ? -1 : 0;
    }();
    // "auto": the fastest shape per weight type and width in test-moe-hitg-bench (Taichi 3090, 2026-10-23T08:00Z):
    // sw16x1g4 everywhere except IQ3_S at 3 or fewer words, where 4x2s is. Every shape has the same bits.
    const int use = shape >= 0 ? shape : (type == GGML_TYPE_IQ3_S && n_tokens <= 3) ? 4 : 12;
    cudaStream_t stream = ctx.stream();
    moe_hitg_dispatch(use, up_table, gate_table, type, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, channels,
            ids_stride, m0, m1, want, stream);
    CUDA_CHECK(cudaGetLastError());
    if (ggml_cuda_moe_hit_fused_mode() == 2) {
        // the plain kernel on every route of both tables, then compare the routes of this pass
        const int64_t count = n_out*n_ids*n_tokens;
        ggml_cuda_pool_alloc<float> ref_up(ctx.pool(), (size_t) count);
        ggml_cuda_pool_alloc<float> ref_gate(ctx.pool(), (size_t) count);
        ggml_cuda_moe_table_mmv(up_table, type, act_q8, ids, ref_up.get(), n_in, n_out, n_ids, n_tokens, channels, ids_stride, stream);
        ggml_cuda_moe_table_mmv(gate_table, type, act_q8, ids, ref_gate.get(), n_in, n_out, n_ids, n_tokens, channels, ids_stride, stream);
        moe_hitg_check_kernel<<<(count + 255)/256, 256, 0, stream>>>(ref_up.get(), ref_gate.get(), dst, ids, m0, m1, want,
                (int) n_out, (int) n_ids, (int) n_tokens, (int) ids_stride);
        CUDA_CHECK(cudaGetLastError());
    }
#else
    GGML_UNUSED_VARS(ctx, up_table, gate_table, type, act_q8, ids, dst, n_in, n_out, n_ids, n_tokens, channels, ids_stride, m0, m1, want);
    GGML_ABORT("fused hit kernel not built");
#endif
}

template <ggml_type type>
static void ggml_cuda_moe_cache_mmv_fused_t(
        const void * up_pool, const void * gate_pool, const char * act_q8,
        const int32_t * up_ids_dev, const int32_t * gate_ids_dev,
        const int32_t * act_ids_dev, float * dst_dev,
        int64_t n_in, int64_t n_out, int64_t slot_stride_bytes,
        int64_t n_hits, int64_t act_rows, float up_min, float up_max,
        float gate_min, float gate_max, cudaStream_t stream) {
    const int64_t ts0 = ggml_type_size(type);
    const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
    const int64_t s01 = ggml_row_size(type, n_in) / ts0;
    const int64_t s02 = slot_stride_bytes / ts0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s12 = act_rows * s11;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const float inf = std::numeric_limits<float>::infinity();
    ggml_cuda_mm_fusion_args_device fusion_local{};
    fusion_local.gate   = gate_pool;
    fusion_local.glu_op = GGML_GLU_OP_SWIGLU;
    if (up_min == -inf && up_max == inf &&
        gate_min == -inf && gate_max == inf) {
        mul_mat_vec_q_moe_launch<type, true, false>(
            up_pool, act_q8, up_ids_dev, act_ids_dev,
            gate_ids_dev, fusion_local, dst_dev, n_in,
            init_fastdiv_values(act_rows), n_out,
            s01, s12, n_out, s02, s11, n_out,
            1, n_hits, warp_size, n_hits,
            up_min, up_max, gate_min, gate_max, stream);
    } else {
        mul_mat_vec_q_moe_launch<type, true, true>(
            up_pool, act_q8, up_ids_dev, act_ids_dev,
            gate_ids_dev, fusion_local, dst_dev, n_in,
            init_fastdiv_values(act_rows), n_out,
            s01, s12, n_out, s02, s11, n_out,
            1, n_hits, warp_size, n_hits,
            up_min, up_max, gate_min, gate_max, stream);
    }
}

void ggml_cuda_moe_cache_mmv_fused(
        const void * up_pool, const void * gate_pool, ggml_type type0,
        const char * act_q8, const int32_t * up_ids_dev,
        const int32_t * gate_ids_dev, const int32_t * act_ids_dev,
        float * dst_dev, int64_t n_in, int64_t n_out,
        int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
        float up_min, float up_max, float gate_min, float gate_max,
        cudaStream_t stream) {
    const int64_t ts0 = ggml_type_size(type0);
    GGML_ASSERT(slot_stride_bytes % ts0 == 0);

#define MOE_CACHE_MMV_FUSED_CASE(type_name) \
        case type_name: \
            ggml_cuda_moe_cache_mmv_fused_t<type_name>( \
                up_pool, gate_pool, act_q8, up_ids_dev, gate_ids_dev, \
                act_ids_dev, dst_dev, n_in, n_out, slot_stride_bytes, \
                n_hits, act_rows, up_min, up_max, gate_min, gate_max, \
                stream); \
            break
    switch (type0) {
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q1_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q2_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q4_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q4_1);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q5_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q5_1);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q8_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_MXFP4);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q2_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q3_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q4_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q5_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q6_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ2_XXS);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ2_XS);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ2_S);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ3_XXS);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ3_S);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ1_S);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ1_M);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ4_NL);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ4_XS);
        default:
            GGML_ABORT("unsupported fused MoE cache type");
    }
#undef MOE_CACHE_MMV_FUSED_CASE
}

// fork: keep the ordinary MMVQ lane partition and reductions when the expert outputs stay in shared memory.
template <ggml_type type>
static __global__ void mul_mat_vec_q_moe_sum(
        const void * matrix, const block_q8_1 * activation, const int32_t * ids,
        const float * weights, const float * scales, float * dst,
        const int ncols, const int nrows, const int nexperts, const int nselected,
        const int64_t stride_row, const int64_t stride_expert, const int stride_activation) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = calc_nwarps(type, 1, MMVQ_PARAMETERS_GENERIC);
    constexpr int rows = 4;
    constexpr int parallel = 2;
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
    constexpr auto vec_dot = get_vec_dot_q_cuda(type);
    const int slot_local = threadIdx.y / nwarps;
    const int warp = threadIdx.y % nwarps;
    const int tid = warp * warp_size + threadIdx.x;
    const int row0 = blockIdx.x * rows;
    __shared__ float partial[parallel][nwarps - 1 > 0 ? nwarps - 1 : 1][rows][warp_size];
    __shared__ float weighted[32][rows];

    for (int slot0 = 0; slot0 < nselected; slot0 += parallel) {
        const int slot = slot0 + slot_local;
        const int expert = slot < nselected ? ids[slot] : -1;
        float sums[rows] = {};
        if (expert >= 0 && expert < nexperts) {
            const block_q8_1 * y = activation + slot * stride_activation;
            for (int kbx = tid / (qi / vdr); kbx < ncols / qk; kbx += blocks_per_iter) {
                const int kby = kbx * (qk / QK8_1);
                const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
                for (int row = 0; row < rows; ++row) {
                    if (row0 + row < nrows) {
                        sums[row] += vec_dot(matrix, y + kby,
                                expert * stride_expert + (row0 + row) * stride_row + kbx, kqs);
                    }
                }
            }
        }
        if (warp > 0) {
#pragma unroll
            for (int row = 0; row < rows; ++row) {
                partial[slot_local][warp - 1][row][threadIdx.x] = sums[row];
            }
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int row = 0; row < rows; ++row) {
#pragma unroll
                for (int w = 0; w < nwarps - 1; ++w) {
                    sums[row] += partial[slot_local][w][row][threadIdx.x];
                }
                const float value = warp_reduce_sum<warp_size>(sums[row]);
                if (threadIdx.x == 0 && slot < nselected) {
                    const float scaled = scales ? __fmul_rn(value, scales[slot]) : value;
                    weighted[slot][row] = __fmul_rn(scaled, weights[slot]);
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.y == 0) {
#pragma unroll
        for (int row = 0; row < rows; ++row) {
            // SUM_ROWS assigns one selected expert to each lane for top-k <= 32.
            float value = threadIdx.x < nselected ? __fadd_rn(0.0f, weighted[threadIdx.x][row]) : 0.0f;
            value = warp_reduce_sum<warp_size>(value);
            if (threadIdx.x == 0 && row0 + row < nrows) {
                dst[row0 + row] = value;
            }
        }
    }
}

bool ggml_cuda_moe_sum_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_PQ2_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_mul_mat_vec_q_moe_sum(ggml_backend_cuda_context & ctx,
        const ggml_tensor * matrix, const ggml_tensor * activation, const ggml_tensor * ids,
        const ggml_tensor * weights, const ggml_tensor * scales, ggml_tensor * dst) {
    const int64_t padded = GGML_PAD(activation->ne[0], MATRIX_ROW_PADDING);
    const size_t q8_bytes = padded * activation->ne[1] * sizeof(block_q8_1) / QK8_1;
    ggml_cuda_pool_alloc<char> q8(ctx.pool(), q8_bytes);
    quantize_row_q8_1_cuda((const float *) activation->data, nullptr, q8.get(), matrix->type,
            activation->ne[0], activation->nb[1] / sizeof(float), activation->nb[2] / sizeof(float),
            activation->nb[3] / sizeof(float), padded, activation->ne[1], 1, 1, ctx.stream());
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    GGML_ASSERT(cc == 800 || cc == 860);
    const dim3 blocks((matrix->ne[1] + 3) / 4);
#define MOE_SUM_CASE(type_name) \
    case type_name: { \
        const dim3 threads(ggml_cuda_info().devices[ctx.device].warp_size, 2 * calc_nwarps(type_name, 1, MMVQ_PARAMETERS_GENERIC)); \
        mul_mat_vec_q_moe_sum<type_name><<<blocks, threads, 0, ctx.stream()>>>( \
            matrix->data, (const block_q8_1 *) q8.get(), (const int32_t *) ids->data, \
            (const float *) weights->data, scales ? (const float *) scales->data : nullptr, (float *) dst->data, \
            matrix->ne[0], matrix->ne[1], matrix->ne[2], ids->ne[0], \
            matrix->nb[1] / ggml_type_size(type_name), matrix->nb[2] / ggml_type_size(type_name), padded / QK8_1); \
        break; \
    }
    switch (matrix->type) {
        MOE_SUM_CASE(GGML_TYPE_Q2_0)
        MOE_SUM_CASE(GGML_TYPE_PQ2_0)
        MOE_SUM_CASE(GGML_TYPE_Q2_K)
        MOE_SUM_CASE(GGML_TYPE_IQ2_XXS)
        MOE_SUM_CASE(GGML_TYPE_IQ2_XS)
        MOE_SUM_CASE(GGML_TYPE_IQ2_S)
        default: GGML_ABORT("unsupported MoE reduction type");
    }
#undef MOE_SUM_CASE
}
