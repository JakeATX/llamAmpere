// FATTN-SJKVARN-FAST (fork): SM 8.6 optimized SJ-KVaRN 4/4 streaming decode kernel (D = 256, 4/4 scalar body).
//
// Isolated in its own translation unit so fattn.cu compile time and legacy turbo4/LSA/strip_map paths are untouched.
// See fattn-sjkvarn-fast.cuh for mode switches (GGML_SJKVARN_FAST=0/1/2/3, GGML_SJKVARN_FAST_NSTAGE=2/4).

#include "fattn-sjkvarn-fast.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-sjkvarn-rot.cuh"
#include "mma.cuh"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace fattn_sj_kvarn_fast {

using namespace ggml_cuda_mma;

constexpr int D           = 256;
constexpr int NTC         = D / 16;          // 16 channel tiles per row
constexpr int NCOLS2      = 8;               // 8 columns per warp
constexpr int UNIT        = 128;             // 128 positions per record
constexpr int STRIP_BYTES = NTC*WARP_SIZE*4; // 2048 B per K or V strip
constexpr int META_BYTES  = 96;              // ktok, vscale, vzero: 16 halves each
constexpr int META_KTOK   = 2*STRIP_BYTES;
constexpr int META_VS     = META_KTOK + 32;
constexpr int META_VZ     = META_VS   + 32;
constexpr int STAGE_BYTES = 2*STRIP_BYTES + META_BYTES; // 4192 B
constexpr int EXACT_ROW_BYTES = D*2;                    // 512 B
constexpr int EXACT_BYTES = 16*EXACT_ROW_BYTES;         // 8192 B
constexpr int RMETA_KZ    = 2*D;
constexpr int RMETA_VCH   = 4*D;
constexpr int RMETA_BYTES = 6*D;                        // 1536 B
constexpr int COL_STRIDE  = D + 4;
constexpr int SLOT_FLOATS = NCOLS2*COL_STRIDE + 2*NCOLS2;
constexpr size_t SLOT_BYTES = SLOT_FLOATS*sizeof(float);

typedef tile<16, 8, half2> T_A;
typedef tile< 8, 8, half2> T_B;
typedef tile<16, 8, float> T_C;
typedef tile<16, 4, half2> T_CH; // FP16-accum m16n8k16 C/D tile (142 TFLOPS on SM 8.6 GA102 = 2x FP32-accum!)

static __host__ __device__ __forceinline__ size_t split_bytes(const int nstage) {
    return (size_t) nstage*STAGE_BYTES + 2*RMETA_BYTES;
}
static __host__ __device__ __forceinline__ size_t ring_bytes(const int nsplit, const int nstage) {
    return (size_t) nsplit * split_bytes(nstage);
}
static __host__ __forceinline__ size_t shared_bytes(const int NT, const int KW, const int nstage) {
    return std::max(ring_bytes(KW, nstage), (size_t) NT*(KW/2)*SLOT_BYTES);
}

static __device__ __forceinline__ void split_sync(const int id, const int nthreads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(nthreads) : "memory");
}
static __device__ __forceinline__ float * slot_acc (float * base, const int slot) { return base + slot*SLOT_FLOATS; }
static __device__ __forceinline__ float * slot_meta(float * base, const int slot) { return base + slot*SLOT_FLOATS + NCOLS2*COL_STRIDE; }

static __device__ __forceinline__ uint32_t lop3_and_or(const uint32_t x, const uint32_t mask, const uint32_t magic) {
    uint32_t r;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(r) : "r"(x), "r"(mask), "r"(magic));
    return r;
}

static __device__ __forceinline__ void raw_pairs(const uint32_t x, const uint32_t magic, half2 * const __restrict__ r) {
    const uint32_t y = x >> 8;
    r[0] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(x, 0x000F000FU, magic));
    r[1] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(x, 0x00F000F0U, magic));
    r[2] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(y, 0x000F000FU, magic));
    r[3] = fattn_sj_kvarn_u32_as_half2(lop3_and_or(y, 0x00F000F0U, magic));
}

static __device__ __forceinline__ void decode_pairs(const uint32_t x, const uint32_t magic, half2 * const __restrict__ q) {
    const half2 m1024 = make_half2(1024.0f, 1024.0f);
    const half2 c16   = make_half2(0.0625f, 0.0625f);
    const half2 m64   = make_half2(-64.0f, -64.0f);
    half2 r[4];
    raw_pairs(x, magic, r);
    q[0] = __hsub2(r[0], m1024);
    q[1] = __hfma2(r[1], c16, m64);
    q[2] = __hsub2(r[2], m1024);
    q[3] = __hfma2(r[3], c16, m64);
}

static __device__ __forceinline__ void cp_async_commit() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
    NO_DEVICE_CODE;
#endif
}
template <int n>
static __device__ __forceinline__ void cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" :: "n"(n) : "memory");
#else
    NO_DEVICE_CODE;
#endif
}

// fast_mode:
//   1 = FP32 HMMA on K (raw_pairs K, zero HADD2/HFMA2!) + FP32 HMMA on V (raw_pairs V) [NMSE ~3.9e-10]
//   2 = FP32 HMMA on K (raw_pairs K) + per-strip FP16 HMMA.16816.F16 on decoded V -> FP32 acc [NMSE ~2.1e-7, LSE err = 0]
//   3 = FP16 HMMA.16816.F16 on decoded K + per-strip FP16 HMMA.16816.F16 on decoded V -> FP32 acc [NMSE ~3.2e-5]
template <int max_warps, int NSTAGE, bool rot, bool compact, int fast_mode>
__launch_bounds__(WARP_SIZE*max_warps, max_warps <= 6 ? 2 : 1)
static __global__ void flash_attn_ext_sj_kvarn_fast(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        const int  * __restrict__ KV_rows,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
        const char    * __restrict__ sj_kvarn_body,
        const int32_t * __restrict__ sj_kvarn_desc) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE) && defined(CP_ASYNC_AVAILABLE)
    GGML_UNUSED_VARS(sinks, KV_max, KV_rows, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb03, ne10, ne13,
                     nb13, nb23, ne31, ne32, ne33, nb32, nb33);

    constexpr bool use_raw_k = (fast_mode == 1 || fast_mode == 2);
    constexpr bool use_f16_k = (fast_mode == 3);
    constexpr bool use_f16_v = (fast_mode == 2 || fast_mode == 3);

    extern __shared__ __align__(16) char smem[];

    const int lane   = threadIdx.x;
    const int warp   = threadIdx.y;
    const int nwarps = blockDim.y;
    const int n_q    = int(ne01.z);
    constexpr bool fixed_nt = compact || max_warps == 5 || max_warps == 6;
    const int NT     = fixed_nt ? max_warps : n_q;
    const int KW     = fixed_nt ? 1 : nwarps / NT;
    const int nt     = fixed_nt ? warp : warp % NT;
    const int kw     = fixed_nt ? 0 : warp / NT;

    const int z_KV  = blockIdx.z;
    const int gqa   = ne02 / ne12;
    const int head0 = z_KV*gqa;
    const int hcol  = lane >> 2;
    const int npair = 6*n_q;
    auto col_query = [&](const int col) -> int  { return compact ? (8*nt + col < npair ? (8*nt + col)/6 : 0) : nt; };
    auto col_head  = [&](const int col) -> int  { return compact ? (8*nt + col) % 6 : col; };
    auto col_valid = [&](const int col) -> bool { return compact ? 8*nt + col < npair : col < gqa; };
    const bool hvalid = col_valid(hcol);

    const fattn_sj_kvarn_ctx kv = fattn_sj_kvarn_make_ctx(sj_kvarn_body, sj_kvarn_desc, z_KV, D, 4, 4, nb12, nb22);
    const int n_kv_pad = ne11;
    const int G16 = kv.G/16;

    const int nsplit  = gridDim.y * KW;
    const int split   = blockIdx.y * KW + kw;
    const int sb0     = kv.S/16;
    const int sb1     = kv.B/16;
    const int nbody   = sb1 - sb0;
    const int s_lo    = sb0 + (int) (((int64_t) nbody *  split     ) / nsplit);
    const int s_hi    = sb0 + (int) (((int64_t) nbody * (split + 1)) / nsplit);
    const int n_ex_sink = kv.S/16;
    const int n_ex    = n_ex_sink + (n_kv_pad - kv.B)/16; // padded positions are masked (as the stream kernel)

    const float * Qf = (const float *) (Q + (size_t) col_query(hcol)*nb01 + (size_t) (head0 + (hvalid ? col_head(hcol) : 0))*nb02);
    const half  * mrow_h = (const half *) (mask + (size_t) col_query(2*(lane & 3))*nb31);
    const char  * Kh = K + (size_t) z_KV*nb12;
    const char  * Vh = V + (size_t) z_KV*nb22;

    char * ring  = smem + (size_t) kw*split_bytes(NSTAGE);
    char * rmeta = ring + NSTAGE*STAGE_BYTES;
    const uint32_t ring_s  = ggml_cuda_cvta_generic_to_shared(ring);
    const uint32_t rmeta_s = ring_s + NSTAGE*STAGE_BYTES;

    auto sync_split = [&]() {
        if (NT == 1) {
            __syncwarp();
        } else {
            split_sync(1 + kw, NT*WARP_SIZE);
        }
    };

    const uint32_t magic = 0x64006400U;

    T_B   qp[NTC];
    float c_r[2]    = {0.0f, 0.0f};
    float c_r_hi[2] = {0.0f, 0.0f}; // used only when use_raw_k == true: 16*c_r - 1024*q_sum
    T_C   acc[NTC];
    float kq_max[2] = {-FLT_MAX/2, -FLT_MAX/2};
    float kq_sum[2] = {0.0f, 0.0f};
    float ps[2]     = {0.0f, 0.0f};
    float pz[2]     = {0.0f, 0.0f};

    auto load_q = [&](const int c, float2 & q0, float2 & q1) {
        if (hvalid) {
            q0 = *(const float2 *) (Qf + 16*c + 2*(lane & 3));
            q1 = *(const float2 *) (Qf + 16*c + 2*(lane & 3) + 8);
            q0.x *= scale; q0.y *= scale; q1.x *= scale; q1.y *= scale;
        } else {
            q0 = make_float2(0.0f, 0.0f);
            q1 = make_float2(0.0f, 0.0f);
        }
    };

    auto setup_body = [&](const char * rm) {
        const uint2 * ks2 = (const uint2 *) (rm            + 2*(lane & 3)*(D/4));
        const uint2 * kz2 = (const uint2 *) (rm + RMETA_KZ + 2*(lane & 3)*(D/4));
        float cr = 0.0f;
        float qs = 0.0f;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const uint2 ks = ks2[c];
            const uint2 kz = kz2[c];
            float2 q0, q1;
            load_q(c, q0, q1);
            const half2 ks0 = fattn_sj_kvarn_u32_as_half2(ks.x), ks1 = fattn_sj_kvarn_u32_as_half2(ks.y);
            const float2 kz0 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.x));
            const float2 kz1 = __half22float2(fattn_sj_kvarn_u32_as_half2(kz.y));
            const half2 qp0 = __hmul2(__floats2half2_rn(q0.x, q0.y), ks0);
            const half2 qp1 = __hmul2(__floats2half2_rn(q1.x, q1.y), ks1);
            qp[c].x[0] = qp0;
            qp[c].x[1] = qp1;
            cr += q0.x*kz0.x + q0.y*kz0.y + q1.x*kz1.x + q1.y*kz1.y;
            if constexpr (use_raw_k) {
                // Exact sum of the fp16 Q' values actually fed to the K MMA, so 1024*qs cancels the raw_pairs bias!
                const float2 fq0 = __half22float2(qp0);
                const float2 fq1 = __half22float2(qp1);
                qs += fq0.x + fq0.y + fq1.x + fq1.y;
            }
        }
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 1, WARP_SIZE);
        cr += __shfl_xor_sync(0xFFFFFFFF, cr, 2, WARP_SIZE);
        if constexpr (use_raw_k) {
            qs += __shfl_xor_sync(0xFFFFFFFF, qs, 1, WARP_SIZE);
            qs += __shfl_xor_sync(0xFFFFFFFF, qs, 2, WARP_SIZE);
            const float cr_lo = cr - 1024.0f*qs;
            const float cr_hi = 16.0f*cr - 1024.0f*qs;
            c_r[0]    = __shfl_sync(0xFFFFFFFF, cr_lo, 8*(lane & 3),     WARP_SIZE);
            c_r[1]    = __shfl_sync(0xFFFFFFFF, cr_lo, 8*(lane & 3) + 4, WARP_SIZE);
            c_r_hi[0] = __shfl_sync(0xFFFFFFFF, cr_hi, 8*(lane & 3),     WARP_SIZE);
            c_r_hi[1] = __shfl_sync(0xFFFFFFFF, cr_hi, 8*(lane & 3) + 4, WARP_SIZE);
        } else {
            c_r[0] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3),     WARP_SIZE);
            c_r[1] = __shfl_sync(0xFFFFFFFF, cr, 8*(lane & 3) + 4, WARP_SIZE);
        }
    };

    auto setup_exact = [&]() {
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            float2 q0, q1;
            load_q(c, q0, q1);
            qp[c].x[0] = __floats2half2_rn(q0.x, q0.y);
            qp[c].x[1] = __floats2half2_rn(q1.x, q1.y);
        }
        c_r[0] = 0.0f;
        c_r[1] = 0.0f;
    };

    // Templated record change: zero branches inside the 16-tile loop; 0.0625f cancels identically when has_new == true!
    auto record_change_mid = [&](const char * rec_old, const char * rec_new) {
        float z0s = pz[0] + __shfl_xor_sync(0xFFFFFFFF, pz[0], 4, WARP_SIZE);
        float z1s = pz[1] + __shfl_xor_sync(0xFFFFFFFF, pz[1], 4, WARP_SIZE);
#pragma unroll
        for (int off = 8; off < WARP_SIZE; off <<= 1) {
            z0s += __shfl_xor_sync(0xFFFFFFFF, z0s, off, WARP_SIZE);
            z1s += __shfl_xor_sync(0xFFFFFFFF, z1s, off, WARP_SIZE);
        }
        if constexpr (!use_f16_v) {
            float p0s = ps[0] + __shfl_xor_sync(0xFFFFFFFF, ps[0], 4, WARP_SIZE);
            float p1s = ps[1] + __shfl_xor_sync(0xFFFFFFFF, ps[1], 4, WARP_SIZE);
#pragma unroll
            for (int off = 8; off < WARP_SIZE; off <<= 1) {
                p0s += __shfl_xor_sync(0xFFFFFFFF, p0s, off, WARP_SIZE);
                p1s += __shfl_xor_sync(0xFFFFFFFF, p1s, off, WARP_SIZE);
            }
            const float lo0 = z0s - 1024.0f*p0s,       lo1 = z1s - 1024.0f*p1s;
            const float hi0 = 16.0f*z0s - 1024.0f*p0s, hi1 = 16.0f*z1s - 1024.0f*p1s;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += lo0; acc[c].x[1] += lo1; acc[c].x[2] += hi0; acc[c].x[3] += hi1;
            }
            ps[0] = 0.0f; ps[1] = 0.0f;
        } else {
            // Decoded V (q in 0..15): no 1024*ps bias and no s=16 factor!
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += z0s; acc[c].x[1] += z1s; acc[c].x[2] += z0s; acc[c].x[3] += z1s;
            }
        }
        pz[0] = 0.0f; pz[1] = 0.0f;

        const uint32_t * vo = (const uint32_t *) (rec_old + RMETA_VCH + 2*((lane >> 2)*(D/8)));
        const uint32_t * vn = (const uint32_t *) (rec_new + RMETA_VCH + 2*((lane >> 2)*(D/8)));
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const float2 o = __half22float2(fattn_sj_kvarn_u32_as_half2(vo[c]));
            const float2 n = __half22float2(fattn_sj_kvarn_u32_as_half2(vn[c]));
            // Note: 0.0625f cancels identically in (0.0625 * o.y) / (0.0625 * n.y) == o.y / n.y!
            const float r_lo = __fdividef(o.x, n.x);
            const float r_hi = __fdividef(o.y, n.y);
            acc[c].x[0] *= r_lo; acc[c].x[1] *= r_lo; acc[c].x[2] *= r_hi; acc[c].x[3] *= r_hi;
        }
    };

    auto record_change_end = [&](const char * rec_old) {
        float z0s = pz[0] + __shfl_xor_sync(0xFFFFFFFF, pz[0], 4, WARP_SIZE);
        float z1s = pz[1] + __shfl_xor_sync(0xFFFFFFFF, pz[1], 4, WARP_SIZE);
#pragma unroll
        for (int off = 8; off < WARP_SIZE; off <<= 1) {
            z0s += __shfl_xor_sync(0xFFFFFFFF, z0s, off, WARP_SIZE);
            z1s += __shfl_xor_sync(0xFFFFFFFF, z1s, off, WARP_SIZE);
        }
        if constexpr (!use_f16_v) {
            float p0s = ps[0] + __shfl_xor_sync(0xFFFFFFFF, ps[0], 4, WARP_SIZE);
            float p1s = ps[1] + __shfl_xor_sync(0xFFFFFFFF, ps[1], 4, WARP_SIZE);
#pragma unroll
            for (int off = 8; off < WARP_SIZE; off <<= 1) {
                p0s += __shfl_xor_sync(0xFFFFFFFF, p0s, off, WARP_SIZE);
                p1s += __shfl_xor_sync(0xFFFFFFFF, p1s, off, WARP_SIZE);
            }
            const float lo0 = z0s - 1024.0f*p0s,       lo1 = z1s - 1024.0f*p1s;
            const float hi0 = 16.0f*z0s - 1024.0f*p0s, hi1 = 16.0f*z1s - 1024.0f*p1s;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += lo0; acc[c].x[1] += lo1; acc[c].x[2] += hi0; acc[c].x[3] += hi1;
            }
            ps[0] = 0.0f; ps[1] = 0.0f;
        } else {
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] += z0s; acc[c].x[1] += z1s; acc[c].x[2] += z0s; acc[c].x[3] += z1s;
            }
        }
        pz[0] = 0.0f; pz[1] = 0.0f;

        const uint32_t * vo = (const uint32_t *) (rec_old + RMETA_VCH + 2*((lane >> 2)*(D/8)));
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            const float2 o = __half22float2(fattn_sj_kvarn_u32_as_half2(vo[c]));
            const float r_lo = o.x;
            const float r_hi = use_f16_v ? o.y : 0.0625f*o.y;
            acc[c].x[0] *= r_lo; acc[c].x[1] *= r_lo; acc[c].x[2] *= r_hi; acc[c].x[3] *= r_hi;
        }
    };

    // Fast online softmax using __expf (2 SASS insns vs 9 for IEEE expf).
    auto softmax = [&](T_C & kq) -> T_B {
        float mx0 = fmaxf(kq.x[0], kq.x[2]);
        float mx1 = fmaxf(kq.x[1], kq.x[3]);
#pragma unroll
        for (int off = 4; off < WARP_SIZE; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFF, mx0, off, WARP_SIZE));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFF, mx1, off, WARP_SIZE));
        }
        const float nm0 = fmaxf(kq_max[0], mx0);
        const float nm1 = fmaxf(kq_max[1], mx1);
        const bool changed = nm0 != kq_max[0] || nm1 != kq_max[1];
        if (__any_sync(0xFFFFFFFF, changed)) {
            const float sc0 = __expf(kq_max[0] - nm0);
            const float sc1 = __expf(kq_max[1] - nm1);
            kq_max[0] = nm0;
            kq_max[1] = nm1;
            kq_sum[0] *= sc0;
            kq_sum[1] *= sc1;
            if constexpr (!use_f16_v) {
                ps[0] *= sc0; ps[1] *= sc1;
            }
            pz[0] *= sc0; pz[1] *= sc1;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                acc[c].x[0] *= sc0; acc[c].x[1] *= sc1; acc[c].x[2] *= sc0; acc[c].x[3] *= sc1;
            }
        }
        kq.x[0] = __expf(kq.x[0] - nm0);
        kq.x[1] = __expf(kq.x[1] - nm1);
        kq.x[2] = __expf(kq.x[2] - nm0);
        kq.x[3] = __expf(kq.x[3] - nm1);
        kq_sum[0] += kq.x[0] + kq.x[2];
        kq_sum[1] += kq.x[1] + kq.x[3];
        return get_transposed(get_half2(kq));
    };

    // Branchless fast cp.async issue (kv.strip_map == nullptr guaranteed):
    // When fixed_nt (compact W=3..6), compile-time nthr_s collapses the K/V/RMETA loops into straight-line LDGSTS!
    const int tid_s = nt*WARP_SIZE + lane;

    auto issue_fast = [&](const int g, const int sl, const int stage, const bool with_meta) {
        const char * rec = fattn_sj_kvarn_rec(kv, g);
        const char * ks  = rec + sl*STRIP_BYTES;
        const char * vs  = ks + kv.v_payload;
        const uint32_t dst_s = ring_s + stage*STAGE_BYTES;

        if constexpr (fixed_nt) {
            constexpr int NTHR = max_warps*WARP_SIZE;
            if (with_meta) {
                const uint32_t dst_m = rmeta_s + (g & 1)*RMETA_BYTES;
                // RMETA_VCH/16 = 64 chunks, (RMETA_BYTES - RMETA_VCH)/16 = 32 chunks (NTHR >= 96 always!)
                if (tid_s < RMETA_VCH/16) {
                    cp_async_cg_16<0>(dst_m + 16*tid_s, rec + kv.k_scale + 16*tid_s);
                } else if (tid_s < RMETA_BYTES/16) {
                    const int q = tid_s - RMETA_VCH/16;
                    cp_async_cg_16<0>(dst_m + RMETA_VCH + 16*q, rec + kv.v_ch + 16*q);
                }
            }
            // K strip (128 chunks = 2048 B) and V strip (128 chunks = 2048 B)
            if constexpr (NTHR == 128) {
                // M=5 compact (W=4): exact 1:1 thread-to-chunk mapping -> ZERO loop, ZERO branches!
                cp_async_cg_16<0>(dst_s               + 16*tid_s, ks + 16*tid_s);
                cp_async_cg_16<0>(dst_s + STRIP_BYTES + 16*tid_s, vs + 16*tid_s);
            } else if constexpr (NTHR > 128) {
                // W=5 (160 thr) or W=6 (192 thr): single predicated LDGSTS for K and V
                if (tid_s < STRIP_BYTES/16) {
                    cp_async_cg_16<0>(dst_s               + 16*tid_s, ks + 16*tid_s);
                    cp_async_cg_16<0>(dst_s + STRIP_BYTES + 16*tid_s, vs + 16*tid_s);
                }
            } else {
                // W=3 (96 thr): 256 chunks / 96 = 2 full + 64 tail
#pragma unroll
                for (int q = tid_s; q < STRIP_BYTES/16; q += NTHR) {
                    cp_async_cg_16<0>(dst_s + 16*q, ks + 16*q);
                }
#pragma unroll
                for (int q = tid_s; q < STRIP_BYTES/16; q += NTHR) {
                    cp_async_cg_16<0>(dst_s + STRIP_BYTES + 16*q, vs + 16*q);
                }
            }
            if (tid_s < 6) {
                const int half_off = 16*(tid_s & 1);
                const char * src = tid_s < 2 ? rec + kv.k_tok   + sl*32 + half_off
                                 : tid_s < 4 ? rec + kv.v_scale + sl*32 + half_off
                                 :             rec + kv.v_zero  + sl*32 + half_off;
                cp_async_cg_16<0>(dst_s + 2*STRIP_BYTES + 16*tid_s, src);
            }
        } else {
            const int nthr_s = NT*WARP_SIZE;
            if (with_meta) {
                const uint32_t dst_m = rmeta_s + (g & 1)*RMETA_BYTES;
                for (int q = tid_s; q < RMETA_VCH/16; q += nthr_s) {
                    cp_async_cg_16<0>(dst_m + 16*q, rec + kv.k_scale + 16*q);
                }
                for (int q = tid_s; q < (RMETA_BYTES - RMETA_VCH)/16; q += nthr_s) {
                    cp_async_cg_16<0>(dst_m + RMETA_VCH + 16*q, rec + kv.v_ch + 16*q);
                }
            }
            for (int q = tid_s; q < STRIP_BYTES/16; q += nthr_s) {
                cp_async_cg_16<0>(dst_s + 16*q, ks + 16*q);
            }
            for (int q = tid_s; q < STRIP_BYTES/16; q += nthr_s) {
                cp_async_cg_16<0>(dst_s + STRIP_BYTES + 16*q, vs + 16*q);
            }
            if (tid_s < 6) {
                const int half_off = 16*(tid_s & 1);
                const char * src = tid_s < 2 ? rec + kv.k_tok   + sl*32 + half_off
                                 : tid_s < 4 ? rec + kv.v_scale + sl*32 + half_off
                                 :             rec + kv.v_zero  + sl*32 + half_off;
                cp_async_cg_16<0>(dst_s + 2*STRIP_BYTES + 16*tid_s, src);
            }
        }
        cp_async_commit();
    };

    const int nthr_s = NT*WARP_SIZE;

    auto issue_exact = [&](const char * rows, const size_t nb, const int type) {
        const float tq6_mag = type == GGML_TYPE_TQ6_0 ? TQ6_CENTROIDS[32 + lane] : 0.0f;
        for (int q = tid_s; q < EXACT_BYTES/16; q += nthr_s) {
            const int r = q / (EXACT_ROW_BYTES/16);
            const int j = q % (EXACT_ROW_BYTES/16);
            if (type != GGML_TYPE_F16) {
                half2 * out = (half2 *) (ring + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)));
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    out[i] = fattn_sj_kvarn_ring_pair_warp(rows + (size_t) r*nb, 4*j + i, type, tq6_mag);
                }
            } else {
                cp_async_cg_16<0>(ring_s + r*EXACT_ROW_BYTES + 16*(j ^ (r & 7)), rows + (size_t) r*nb + 16*j);
            }
        }
        cp_async_commit();
    };
    auto exact_word = [&](const int r, const int cp) -> uint32_t {
        return *(const uint32_t *) (ring + r*EXACT_ROW_BYTES + 16*((cp >> 2) ^ (r & 7)) + 4*(cp & 3));
    };

    // Initialize dense body cursors (p0 = 16*s_lo, strictly inside [kv.S, kv.B) when s_lo < s_hi).
    int gi = 0, sli = 0;
    if (s_lo < s_hi) {
        const int off0 = 16*s_lo - kv.S;
        gi  = off0 / kv.G;
        sli = (off0 - gi*kv.G) / 16;
    }
    int gc = gi, slc = sli, p0c = 16*s_lo;
    int stage_i = 0, stage_c = 0;

#pragma unroll
    for (int i = 0; i < NSTAGE - 1; ++i) {
        if (s_lo + i < s_hi) {
            issue_fast(gi, sli, stage_i, i == 0 || sli == 0);
            if (++sli == G16) { sli = 0; ++gi; }
            stage_i = stage_i + 1 == NSTAGE ? 0 : stage_i + 1;
        } else {
            cp_async_commit();
        }
    }

    int cur_id = INT_MIN;
    const char * cur_rm = nullptr;

    for (int s = s_lo; s < s_hi; ++s) {
        if (s + NSTAGE - 1 < s_hi) {
            issue_fast(gi, sli, stage_i, sli == 0);
            if (++sli == G16) { sli = 0; ++gi; }
            stage_i = stage_i + 1 == NSTAGE ? 0 : stage_i + 1;
        } else {
            cp_async_commit();
        }
        cp_async_wait_group<NSTAGE - 1>();
        sync_split();

        const int id = gc;
        const char * rm = rmeta + (gc & 1)*RMETA_BYTES;
        if (id != cur_id) {
            if (cur_id != INT_MIN) {
                record_change_mid(cur_rm, rm);
            }
            setup_body(rm);
            cur_id = id;
            cur_rm = rm;
        }

        const char * st = ring + stage_c*STAGE_BYTES;
        const uint32_t * kwp = (const uint32_t *) st + lane;
        const uint32_t * vwp = (const uint32_t *) (st + STRIP_BYTES) + lane;

        T_C kq;
        if constexpr (use_f16_k) {
            // 2x SM 8.6 TC throughput: HMMA.16816.F16 on decoded K codes [0..15]
            T_CH kqh;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                decode_pairs(kwp[c*WARP_SIZE], magic, a.x);
                mma(kqh, a, qp[c]);
            }
            const float2 f0 = __half22float2(kqh.x[0]);
            const float2 f1 = __half22float2(kqh.x[1]);
            kq.x[0] = f0.x; kq.x[1] = f0.y; kq.x[2] = f1.x; kq.x[3] = f1.y;
        } else {
            // Exact FP32 HMMA with raw_pairs K (ZERO HADD2 / HFMA2 decode insns!)
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                raw_pairs(kwp[c*WARP_SIZE], magic, a.x);
                mma(kq, a, qp[c]);
            }
        }

        const half * ktok = (const half *) (st + META_KTOK);
        const half * vsh  = (const half *) (st + META_VS);
        const half * vzh  = (const half *) (st + META_VZ);
        const float tok0  = __half2float(ktok[lane >> 2]);
        const float tok1  = __half2float(ktok[(lane >> 2) + 8]);
        const float mk0   = __half2float(mrow_h[p0c + (lane >> 2)]);
        const float mk1   = __half2float(mrow_h[p0c + (lane >> 2) + 8]);

        if constexpr (use_raw_k) {
            // Token 0..7: raw = 1024 + q, so c_r[e] = c_r - 1024*q_sum.
            // Token 8..15: raw = 1024 + 16*q, so (0.0625*tok1)*(raw + 16*c_r - 1024*q_sum).
            const float tok1_s = 0.0625f*tok1;
            kq.x[0] = tok0  *(kq.x[0] + c_r[0])    + mk0;
            kq.x[1] = tok0  *(kq.x[1] + c_r[1])    + mk0;
            kq.x[2] = tok1_s*(kq.x[2] + c_r_hi[0]) + mk1;
            kq.x[3] = tok1_s*(kq.x[3] + c_r_hi[1]) + mk1;
        } else {
            kq.x[0] = tok0*(kq.x[0] + c_r[0]) + mk0;
            kq.x[1] = tok0*(kq.x[1] + c_r[1]) + mk0;
            kq.x[2] = tok1*(kq.x[2] + c_r[0]) + mk1;
            kq.x[3] = tok1*(kq.x[3] + c_r[1]) + mk1;
        }

        T_B P = softmax(kq);

        const half2 vs_b0 = *(const half2 *) (vsh + 2*(lane & 3));
        const half2 vs_b1 = *(const half2 *) (vsh + 2*(lane & 3) + 8);
        P.x[0] = __hmul2(P.x[0], vs_b0);
        P.x[1] = __hmul2(P.x[1], vs_b1);
        const float2 vz_c = make_float2(__half2float(vzh[lane >> 2]), __half2float(vzh[(lane >> 2) + 8]));
        pz[0] += kq.x[0]*vz_c.x + kq.x[2]*vz_c.y;
        pz[1] += kq.x[1]*vz_c.x + kq.x[3]*vz_c.y;

        if constexpr (!use_f16_v) {
            const half2 vs_c = __halves2half2(vsh[lane >> 2], vsh[(lane >> 2) + 8]);
            const float2 pe0 = __half22float2(__hmul2(make_half2(kq.x[0], kq.x[2]), vs_c));
            const float2 pe1 = __half22float2(__hmul2(make_half2(kq.x[1], kq.x[3]), vs_c));
            ps[0] += pe0.x + pe0.y;
            ps[1] += pe1.x + pe1.y;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                raw_pairs(vwp[c*WARP_SIZE], magic, a.x);
                mma(acc[c], a, P);
            }
        } else {
            // Per-strip FP16 HMMA.16816.F16 on decoded V [0..15] accumulated into FP32 acc[c] (2x V TC rate!)
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                T_A a;
                decode_pairs(vwp[c*WARP_SIZE], magic, a.x);
                T_CH dv;
                mma(dv, a, P);
                const float2 d0 = __half22float2(dv.x[0]);
                const float2 d1 = __half22float2(dv.x[1]);
                acc[c].x[0] += d0.x; acc[c].x[1] += d0.y; acc[c].x[2] += d1.x; acc[c].x[3] += d1.y;
            }
        }

        sync_split();
        p0c += 16;
        if (++slc == G16) { slc = 0; ++gc; }
        stage_c = stage_c + 1 == NSTAGE ? 0 : stage_c + 1;
    }

    if (cur_id != INT_MIN) {
        record_change_end(cur_rm);
    }
    cp_async_wait_all();
    sync_split();

    // Exact f16/ring strips (sink + recent ring), always full FP32 HMMA.
    if (split < n_ex) {
        setup_exact();
    }
    for (int e = split; e < n_ex; e += nsplit) {
        const int p0   = e < n_ex_sink ? 16*e : kv.B + 16*(e - n_ex_sink);
        const int row0 = p0 < kv.S ? p0 : kv.S + (p0 - kv.S) % kv.cap;
        const bool f16_sink = p0 < kv.S && kv.sink_type == GGML_TYPE_F16;
        const char * Kr = f16_sink ? fattn_sj_kvarn_sink_row(Kh, kv, nb11, row0, false) : Kh + (size_t) row0*nb11;
        const char * Vr = f16_sink ? fattn_sj_kvarn_sink_row(Vh, kv, nb21, row0, true)  : Vh + (size_t) row0*nb21;

        issue_exact(Kr, f16_sink ? kv.sink_stride : nb11, f16_sink ? GGML_TYPE_F16 : kv.type_k);
        cp_async_wait_all();
        sync_split();

        T_C kq;
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            T_A a;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int row = (lane >> 2) + 8*(l & 1);
                const int cp  = 8*c + 4*(l >> 1) + (lane & 3);
                a.x[l] = fattn_sj_kvarn_u32_as_half2(exact_word(row, cp));
            }
            mma(kq, a, qp[c]);
        }
        const float mk0 = __half2float(mrow_h[p0 + (lane >> 2)]);
        const float mk1 = __half2float(mrow_h[p0 + (lane >> 2) + 8]);
        kq.x[0] += mk0; kq.x[1] += mk0; kq.x[2] += mk1; kq.x[3] += mk1;
        const T_B P = softmax(kq);

        sync_split();
        issue_exact(Vr, f16_sink ? kv.sink_stride : nb21, f16_sink ? GGML_TYPE_F16 : kv.type_v);
        cp_async_wait_all();
        sync_split();
#pragma unroll
        for (int c = 0; c < NTC; ++c) {
            T_A a;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int row = 8*(l >> 1) + (lane >> 2);
                const int cp  = 8*c + 4*(l & 1) + (lane & 3);
                a.x[l] = ggml_cuda_movmatrix(fattn_sj_kvarn_u32_as_half2(exact_word(row, cp)));
            }
            mma(acc[c], a, P);
        }
        sync_split();
    }

    // Finish lane-partial sums.
#pragma unroll
    for (int off = 4; off < WARP_SIZE; off <<= 1) {
        kq_sum[0] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[0], off, WARP_SIZE);
        kq_sum[1] += __shfl_xor_sync(0xFFFFFFFF, kq_sum[1], off, WARP_SIZE);
    }

    // In-block KV split combine (only when KW > 1).
    float * smem_combine = (float *) smem;
    __syncthreads();
    for (int hf = KW/2; hf >= 1; hf >>= 1) {
        if (kw >= hf && kw < 2*hf) {
            const int slot = nt*hf + (kw - hf);
            float * a = slot_acc (smem_combine, slot);
            float * m = slot_meta(smem_combine, slot);
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int col = 2*(lane & 3) + (l & 1);
                    const int d   = 16*c + (lane >> 2) + 8*(l >> 1);
                    a[col*COL_STRIDE + d] = acc[c].x[l];
                }
            }
            if (lane < 4) {
                m[2*(2*lane + 0) + 0] = kq_max[0]; m[2*(2*lane + 0) + 1] = kq_sum[0];
                m[2*(2*lane + 1) + 0] = kq_max[1]; m[2*(2*lane + 1) + 1] = kq_sum[1];
            }
        }
        __syncthreads();
        if (kw < hf) {
            const int slot = nt*hf + kw;
            const float * a = slot_acc (smem_combine, slot);
            const float * m = slot_meta(smem_combine, slot);
            const float om0 = m[2*(2*(lane & 3) + 0) + 0], os0 = m[2*(2*(lane & 3) + 0) + 1];
            const float om1 = m[2*(2*(lane & 3) + 1) + 0], os1 = m[2*(2*(lane & 3) + 1) + 1];
            const float nm0 = fmaxf(kq_max[0], om0), nm1 = fmaxf(kq_max[1], om1);
            const float sa0 = __expf(kq_max[0] - nm0), sb0 = __expf(om0 - nm0);
            const float sa1 = __expf(kq_max[1] - nm1), sb1 = __expf(om1 - nm1);
            kq_max[0] = nm0; kq_max[1] = nm1;
            kq_sum[0] = sa0*kq_sum[0] + sb0*os0;
            kq_sum[1] = sa1*kq_sum[1] + sb1*os1;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    const int col = 2*(lane & 3) + (l & 1);
                    const int d   = 16*c + (lane >> 2) + 8*(l >> 1);
                    const float sa = (l & 1) ? sa1 : sa0;
                    const float sb = (l & 1) ? sb1 : sb0;
                    acc[c].x[l] = sa*acc[c].x[l] + sb*a[col*COL_STRIDE + d];
                }
            }
        }
        __syncthreads();
    }

    if (kw != 0) {
        return;
    }

    const bool single = gridDim.y == 1;
    if constexpr (rot) {
        if (single) {
            const float inv0 = 1.0f/kq_sum[0], inv1 = 1.0f/kq_sum[1];
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    acc[c].x[l] *= (l & 1) ? inv1 : inv0;
                }
            }
            sj_kvarn_rot256_frag<T_C, NTC>(acc, lane);
            kq_sum[0] = 1.0f;
            kq_sum[1] = 1.0f;
        }
    }
#pragma unroll
    for (int l = 0; l < 4; l += 2) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*(lane & 3) + e;
            if (!col_valid(col)) {
                continue;
            }
            const int head = head0 + col_head(col);
            const float inv = single ? 1.0f/kq_sum[e] : 1.0f;
            float * out = dst + ((size_t) (col_query(col)*ne02 + head)*gridDim.y + blockIdx.y)*D;
#pragma unroll
            for (int c = 0; c < NTC; ++c) {
                const int d = 16*c + (lane >> 2) + 8*(l >> 1);
                out[d] = acc[c].x[l + e]*inv;
            }
        }
    }
    if (!single && lane < 4) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int col = 2*lane + e;
            if (col_valid(col)) {
                dst_meta[(size_t) (col_query(col)*ne02 + head0 + col_head(col))*gridDim.y + blockIdx.y] = make_float2(kq_max[e], kq_sum[e]);
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, KV_rows, dst, dst_meta, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                     ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10, ne11, ne12, ne13, nb11, nb12, nb13, nb21, nb22, nb23,
                     ne31, ne32, ne33, nb31, nb32, nb33, sj_kvarn_body, sj_kvarn_desc);
    NO_DEVICE_CODE;
#endif
}

template <int max_warps, int NSTAGE, bool rot, bool compact, int fast_mode>
static fattn_kernel_t get_fast_kernel() {
    fattn_kernel_t kernel = flash_attn_ext_sj_kvarn_fast<max_warps, NSTAGE, rot, compact, fast_mode>;
    const int id = ggml_cuda_get_device();
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {false};
    if (!initialized[id]) {
        const size_t limit = ggml_cuda_info().devices[id].smpbo;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int) limit));
        initialized[id] = true;
    }
    return kernel;
}

template <int max_warps, bool rot, bool compact, int fast_mode>
static void launch_fast_nstage(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int NT, const int KW, const int nstage_req) {
    const bool use4 = nstage_req >= 4;
    const int nstage = use4 ? 4 : 2;
    fattn_kernel_t kernel = use4 ? get_fast_kernel<max_warps, 4, rot, compact, fast_mode>()
                                 : get_fast_kernel<max_warps, 2, rot, compact, fast_mode>();
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, reinterpret_cast<const void *>(kernel)));
        int ctas = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ctas, kernel, WARP_SIZE*NT*KW, shared_bytes(NT, KW, nstage)));
        GGML_LOG_INFO("SJKVARN FAST: mode=%d warps=%d compact=%d rot=%d nstage=%d -> %d CTA/SM (regs %d, local %zu B)\n",
            fast_mode, NT*KW, (int) compact, (int) rot, nstage, ctas, attr.numRegs, attr.localSizeBytes);
    }
    launch_fattn<D, NCOLS2, NCOLS2>(
        ctx, dst, kernel, NT*KW, shared_bytes(NT, KW, nstage), UNIT,
        /*need_f16_K=*/false, /*need_f16_V=*/false, /*stream_k=*/false, /*use_sparse=*/false,
        WARP_SIZE, nullptr, nullptr, /*parallel_blocks_override=*/0,
        /*async_kv_preload=*/false, /*sj_fill_kv_splits=*/KW);
}

template <int max_warps, bool rot, bool compact>
static void dispatch_mode(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int NT, const int KW, const int mode, const int nstage) {
    if (mode == 3) {
        launch_fast_nstage<max_warps, rot, compact, 3>(ctx, dst, NT, KW, nstage);
    } else if (mode == 2) {
        launch_fast_nstage<max_warps, rot, compact, 2>(ctx, dst, NT, KW, nstage);
    } else {
        launch_fast_nstage<max_warps, rot, compact, 1>(ctx, dst, NT, KW, nstage);
    }
}

template <bool rot>
static bool dispatch_shape(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int mode, const int nstage) {
    const int n_q = (int) dst->src[0]->ne[1];
    const int gqa = (int) (dst->src[0]->ne[2] / dst->src[1]->ne[2]);
    if (n_q >= 4 && n_q <= 8 && gqa == 6) {
        const int W = (6*n_q + 7)/8; // 4->3, 5->4, 6->5, 7/8->6
        switch (W) {
            case 3:  dispatch_mode<3, rot, true>(ctx, dst, 3, 1, mode, nstage); return true;
            case 4:  dispatch_mode<4, rot, true>(ctx, dst, 4, 1, mode, nstage); return true;
            case 5:  dispatch_mode<5, rot, true>(ctx, dst, 5, 1, mode, nstage); return true;
            default: dispatch_mode<6, rot, true>(ctx, dst, 6, 1, mode, nstage); return true;
        }
    }
    if (n_q >= 1 && n_q <= 4) {
        const int KW = 4 / n_q;
        dispatch_mode<4, rot, false>(ctx, dst, n_q, KW, mode, nstage);
        return true;
    }
    return false;
}

} // namespace fattn_sj_kvarn_fast

bool ggml_cuda_fattn_sj_kvarn_fast_try(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const bool rot) {
    // INKATTN review: the old guard here used device-pass macros (__CUDA_ARCH__), so the host function always returned
    // false; decide on the device's compute capability at run time instead (kernel bodies keep their device guards)
    {
        const int cc = ggml_cuda_info().devices[ctx.device].cc;
        if (!GGML_CUDA_CC_IS_NVIDIA(cc) || !cp_async_available(cc) || !turing_mma_available(cc)) {
            return false;
        }
    }
    static const int mode = [] {
        const char * e = getenv("GGML_SJKVARN_FAST");
        if (!e || !*e) return 0; // default OFF until gated on the card (INKATTN review: never launched before; raw K is rounding-level, not exact)
        const int m = atoi(e);
        return (m >= 0 && m <= 3) ? m : 1;
    }();
    if (mode == 0) {
        return false;
    }
    static const bool stable = getenv("GGML_SJKVARN_DIRECT_STABLE") != nullptr && atoi(getenv("GGML_SJKVARN_DIRECT_STABLE")) != 0;
    if (stable) {
        return false;
    }
    const ggml_tensor * mask = dst->src[3];
    if (!mask || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 || dst->src[0]->ne[3] != 1) {
        return false;
    }
    if (ggml_cuda_fattn_sj_live_records != nullptr || ggml_cuda_qsa_strip_body(dst->src[5])) {
        return false;
    }
    const int n_q = (int) dst->src[0]->ne[1];
    if (n_q < 1 || n_q > 8) {
        return false;
    }
    // Default NSTAGE: 4 for mode >= 2 (higher HBM demand), 2 for mode 1 (override with GGML_SJKVARN_FAST_NSTAGE).
    static const int nstage_env = [] {
        const char * e = getenv("GGML_SJKVARN_FAST_NSTAGE");
        return e && *e ? atoi(e) : 0;
    }();
    const int nstage = nstage_env != 0 ? nstage_env : (mode >= 2 ? 4 : 2);
    return rot ? fattn_sj_kvarn_fast::dispatch_shape<true>(ctx, dst, mode, nstage)
               : fattn_sj_kvarn_fast::dispatch_shape<false>(ctx, dst, mode, nstage);
}
