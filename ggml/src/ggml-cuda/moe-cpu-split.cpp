// fork STRATAGAP: host side of the CPU expert split (see moe-cpu-split.h).
//
// Thread 0 of the pool watches the mapped job; when the card raises `seq`, every pool thread works through the
// job's four phases (quantize the input rows, up + gate + SwiGLU, quantize the hidden rows, down) taking small
// items from a shared counter, then thread 0 raises `done`. Threads spin between jobs for GGML_MOE_CPU_SPLIT_SPIN_MS
// after the last one (a decode keeps them hot) and sleep after that, so a long prompt or an idle server costs
// no CPU. The dot products are ggml's own CPU kernels (their AVX2 versions on the Zen 3 build), except Q2_0, whose
// CPU kernel is scalar: it gets its own AVX2 loop below.
#if defined(__linux__)
#include <sys/prctl.h>
#include <dirent.h>
#endif
#include "moe-cpu-split.h"
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP   // the IQ grids, for the AVX-512 dot loops below
#include "../ggml-common.h"
#include "moe-iq-rows.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <time.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

using clk = std::chrono::steady_clock;

int env_int(const char * name, int def, int lo, int hi) {
    const char * v = getenv(name);
    if (!v || !*v) { return def; }
    return std::max(lo, std::min(hi, atoi(v)));
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

// Q2_0 (64 weights: an fp16 scale and 2-bit codes 0..3 meaning -1..2) against two Q8_0 blocks.
// BMI2 pdep spreads each 16-bit group of codes over 8 bytes, so a 32-weight half is four pdeps.
void dot_q2_0_q8_0(int n, float * s, const void * vx, const void * vy) {
    const block_q2_0 * x = (const block_q2_0 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    const int nb = n / QK2_0;
#if defined(__AVX2__) && defined(__BMI2__)
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d0 = ggml_fp16_to_fp32(x[i].d);
        for (int k = 0; k < 2; ++k) {
            const block_q8_0 & yb = y[2 * i + k];
            uint64_t codes;
            memcpy(&codes, x[i].qs + 8 * k, 8);
            const uint64_t m = 0x0303030303030303ull;
            const __m256i w = _mm256_set_epi64x((long long) _pdep_u64(codes >> 48, m), (long long) _pdep_u64((codes >> 32) & 0xffff, m),
                    (long long) _pdep_u64((codes >> 16) & 0xffff, m), (long long) _pdep_u64(codes & 0xffff, m));
            const __m256i q = _mm256_loadu_si256((const __m256i *) yb.qs);
            const __m256i wq = _mm256_madd_epi16(_mm256_maddubs_epi16(w, q), ones16);
            const __m256i sq = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q), ones16);
            const __m256 d = _mm256_set1_ps(d0 * ggml_fp16_to_fp32(yb.d));
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(_mm256_sub_epi32(wq, sq)), acc);
        }
    }
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    *s = _mm_cvtss_f32(h);
#else
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const float d0 = ggml_fp16_to_fp32(x[i].d);
        for (int k = 0; k < 2; ++k) {
            const block_q8_0 & yb = y[2 * i + k];
            int isum = 0;
            for (int j = 0; j < 32; ++j) {
                const int code = (x[i].qs[8 * k + j / 4] >> (2 * (j % 4))) & 3;
                isum += (code - 1) * yb.qs[j];
            }
            sum += d0 * ggml_fp16_to_fp32(yb.d) * (float) isum;
        }
    }
    *s = sum;
#endif
}

// Q2_0 down rows, fast form (GGML_MOE_CPU_SPLIT_Q2_FAST=1). The hidden row is quantized once per route and its
// bytes are reordered inside each 32-weight block (position 8 s + b holds weight 4 b + s), so the 2-bit codes need
// no spreading: one 64-bit shift per lane (0, 2, 4, 6) and a mask line them up with the reordered bytes. The -1 code
// offset is folded out: sum (c - 1) q = sum c q - sum q, and sum q per block is computed once per hidden row (yc).
// Four weight rows share each hidden block load. Same integer products as dot_q2_0_q8_0; the float sums group
// differently (valid math).
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
inline float hsum8(__m256 v) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}
#endif

void q2_reorder_hidden(int n, block_q8_0 * y, float * yd, float * yc) {
    const int nb = n / QK8_0;
    for (int i = 0; i < nb; ++i) {
        int8_t t[QK8_0];
        int sum = 0;
        for (int s = 0; s < 4; ++s) {
            for (int b = 0; b < 8; ++b) { t[8 * s + b] = y[i].qs[4 * b + s]; }
        }
        for (int j = 0; j < QK8_0; ++j) { sum += t[j]; }
        memcpy(y[i].qs, t, QK8_0);
        yd[i] = ggml_fp16_to_fp32(y[i].d);
        if (i & 1) { yc[i / 2] += yd[i] * (float) sum; } else { yc[i / 2] = yd[i] * (float) sum; }
    }
}

void dot_q2_0_rows(int n, int nrows, const char * w, size_t row_bytes, const block_q8_0 * y, const float * yd,
        const float * yc, float * out) {
    const int nb = n / QK2_0;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    const __m256i shifts = _mm256_set_epi64x(6, 4, 2, 0);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    int r = 0;
    for (; r < nrows; r += 4) {
        const int nr = std::min(4, nrows - r);
        __m256 acc[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
        float corr[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int i = 0; i < nb; ++i) {
            const __m256i q0 = _mm256_loadu_si256((const __m256i *) y[2 * i].qs);
            const __m256i q1 = _mm256_loadu_si256((const __m256i *) y[2 * i + 1].qs);
            const __m256 s0 = _mm256_set1_ps(yd[2 * i]);
            const __m256 s1 = _mm256_set1_ps(yd[2 * i + 1]);
            for (int k = 0; k < nr; ++k) {
                const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) (r + k) * row_bytes) + i;
                const float d = _cvtsh_ss(x->d);
                long long c0, c1;
                memcpy(&c0, x->qs, 8);
                memcpy(&c1, x->qs + 8, 8);
                const __m256i w0 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(c0), shifts), m3);
                const __m256i w1 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(c1), shifts), m3);
                const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, q0), ones16);
                const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w1, q1), ones16);
                const __m256 f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), s0, _mm256_mul_ps(_mm256_cvtepi32_ps(p1), s1));
                acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(d), f, acc[k]);
                corr[k] += d * yc[i];
            }
        }
        for (int k = 0; k < nr; ++k) { out[r + k] = hsum8(acc[k]) - corr[k]; }
    }
#else
    for (int r = 0; r < nrows; ++r) {
        const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) r * row_bytes);
        float sum = 0.0f;
        for (int i = 0; i < nb; ++i) {
            const float d = ggml_fp16_to_fp32(x[i].d);
            for (int k = 0; k < 2; ++k) {
                const block_q8_0 & yb = y[2 * i + k];
                int isum = 0;
                for (int s = 0; s < 4; ++s) {
                    for (int b = 0; b < 8; ++b) { isum += ((x[i].qs[8 * k + b] >> (2 * s)) & 3) * yb.qs[8 * s + b]; }
                }
                sum += d * yd[2 * i + k] * (float) isum;
            }
            sum -= d * yc[i];
        }
        out[r] = sum;
    }
#endif
}

// AVX-512 dot loops for the IQ up/gate types (IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS) (GGML_MOE_CPU_SPLIT_AVX512=1, default off; used only when the CPU has
// AVX-512 F/BW/VNNI, checked at pool start). Same block formats and integer sums as ggml-cpu's AVX2 loops: the grid
// lookups become one gather per 64 weights, the signs one masked subtract (the 64 sign bits are a byte mask), and
// the products one VNNI multiply-add. Float sums group differently (valid math). Compiled with function targets, so
// the AVX2 build carries them and never calls them on a CPU without AVX-512.
#if defined(__x86_64__) && defined(__GNUC__)
#define MOE_CS_AVX512 1
#define MOE_CS_T512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,f16c,bmi2")))

MOE_CS_T512 void dot_iq3_s_q8_K_512(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_iq3_s * x = (const block_iq3_s *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    const __m512i lane = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i one = _mm512_set1_epi32(1);
    __m512 accf = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        __m512i sumi = _mm512_setzero_si512();
        for (int ib = 0; ib < QK_K / 64; ++ib) {   // 64 weights: qs 16 indices, qh 2 bytes, signs 8 bytes
            const int ib32 = 2 * ib;
            const uint32_t qh = (uint32_t) x[i].qh[ib32] | ((uint32_t) x[i].qh[ib32 + 1] << 8);
            const __m512i hi = _mm512_slli_epi32(_mm512_and_si512(_mm512_srlv_epi32(_mm512_set1_epi32((int) qh), lane), one), 8);
            const __m512i idx = _mm512_or_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (x[i].qs + 16 * ib))), hi);
            const __m512i g = _mm512_i32gather_epi32(idx, (const void *) iq3s_grid, 4);
            uint64_t sg;
            memcpy(&sg, x[i].signs + 8 * ib, 8);
            const __m512i q8 = _mm512_loadu_si512((const void *) (y[i].qs + 64 * ib));
            const __m512i q8s = _mm512_mask_sub_epi8(q8, (__mmask64) sg, _mm512_setzero_si512(), q8);
            const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), g, q8s);
            const int ls1 = 2 * (x[i].scales[ib] & 0xf) + 1, ls2 = 2 * (x[i].scales[ib] >> 4) + 1;
            const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(ls1)), _mm256_set1_epi32(ls2), 1);
            sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(dp, sc));
        }
        accf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), accf);
    }
    *s = _mm512_reduce_add_ps(accf);
}

// iq2_xxs: per 32 weights two 32-bit words; the first holds four 8-weight grid indices, the second four 7-bit sign
// fields (the eighth sign makes the count of minus signs even) and a 4-bit scale.
MOE_CS_T512 inline uint32_t iq2_signs4(uint32_t w) {
    const uint32_t b = _pdep_u32(w, 0x7f7f7f7fu);
    uint32_t p = b ^ (b >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return b | ((p & 0x01010101u) << 7);
}

MOE_CS_T512 void dot_iq2_xxs_q8_K_512(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    __m512 accf = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        __m512i sumi = _mm512_setzero_si512();
        for (int ib = 0; ib < QK_K / 64; ++ib) {
            uint32_t a[4];
            memcpy(a, x[i].qs + 8 * ib, 16);
            const uint64_t ix = (uint64_t) a[0] | ((uint64_t) a[2] << 32);
            const __m512i idx = _mm512_cvtepu8_epi64(_mm_cvtsi64_si128((long long) ix));
            const __m512i g = _mm512_i64gather_epi64(idx, (const void *) iq2xxs_grid, 8);
            const uint64_t sg = (uint64_t) iq2_signs4(a[1]) | ((uint64_t) iq2_signs4(a[3]) << 32);
            const __m512i q8 = _mm512_loadu_si512((const void *) (y[i].qs + 64 * ib));
            const __m512i q8s = _mm512_mask_sub_epi8(q8, (__mmask64) sg, _mm512_setzero_si512(), q8);
            const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), g, q8s);
            const int ls1 = 2 * (int) (a[1] >> 28) + 1, ls2 = 2 * (int) (a[3] >> 28) + 1;
            const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(ls1)), _mm256_set1_epi32(ls2), 1);
            sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(dp, sc));
        }
        accf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), accf);
    }
    *s = 0.125f * _mm512_reduce_add_ps(accf);
}

// iq3_xxs: 64 index bytes, then one 32-bit sign + scale word per 32 weights (same layout as iq2_xxs's second word).
MOE_CS_T512 void dot_iq3_xxs_q8_K_512(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    __m512 accf = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        const uint8_t * gas = x[i].qs + QK_K / 4;
        __m512i sumi = _mm512_setzero_si512();
        for (int ib = 0; ib < QK_K / 64; ++ib) {
            uint32_t a[2];
            memcpy(a, gas + 8 * ib, 8);
            const __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (x[i].qs + 16 * ib)));
            const __m512i g = _mm512_i32gather_epi32(idx, (const void *) iq3xxs_grid, 4);
            const uint64_t sg = (uint64_t) iq2_signs4(a[0]) | ((uint64_t) iq2_signs4(a[1]) << 32);
            const __m512i q8 = _mm512_loadu_si512((const void *) (y[i].qs + 64 * ib));
            const __m512i q8s = _mm512_mask_sub_epi8(q8, (__mmask64) sg, _mm512_setzero_si512(), q8);
            const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), g, q8s);
            const int ls1 = 2 * (int) (a[0] >> 28) + 1, ls2 = 2 * (int) (a[1] >> 28) + 1;
            const __m512i sc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi32(ls1)), _mm256_set1_epi32(ls2), 1);
            sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(dp, sc));
        }
        accf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), accf);
    }
    *s = 0.25f * _mm512_reduce_add_ps(accf);
}

// Scales of iq2_xs / iq2_s: one nibble per 16 weights; two scale bytes cover 64 weights = 16 VNNI lanes of 4.
MOE_CS_T512 inline __m512i iq2_scales16(const uint8_t * sc) {
    const uint32_t two = (uint32_t) sc[0] | ((uint32_t) sc[1] << 8);
    const __m128i nib = _mm_cvtsi32_si128((int) _pdep_u32(two, 0x0f0f0f0fu));
    const __m128i rep = _mm_shuffle_epi8(nib, _mm_set_epi8(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0));
    return _mm512_or_si512(_mm512_slli_epi32(_mm512_cvtepu8_epi32(rep), 1), _mm512_set1_epi32(1));
}

// iq2_xs: 16-bit codes, 9-bit grid index and 7 sign bits (eighth by parity).
MOE_CS_T512 void dot_iq2_xs_q8_K_512(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_iq2_xs * x = (const block_iq2_xs *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    const __m128i m511 = _mm_set1_epi16(511);
    __m512 accf = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        __m512i sumi = _mm512_setzero_si512();
        for (int ib = 0; ib < QK_K / 64; ++ib) {
            const __m128i v = _mm_loadu_si128((const __m128i *) (x[i].qs + 8 * ib));
            const __m512i idx = _mm512_cvtepu16_epi64(_mm_and_si128(v, m511));
            const __m512i g = _mm512_i64gather_epi64(idx, (const void *) iq2xs_grid, 8);
            const uint64_t b = (uint64_t) _mm_cvtsi128_si64(_mm_packus_epi16(_mm_srli_epi16(v, 9), _mm_setzero_si128()));
            uint64_t p = b ^ (b >> 4);
            p ^= p >> 2;
            p ^= p >> 1;
            const uint64_t sg = b | ((p & 0x0101010101010101ull) << 7);
            const __m512i q8 = _mm512_loadu_si512((const void *) (y[i].qs + 64 * ib));
            const __m512i q8s = _mm512_mask_sub_epi8(q8, (__mmask64) sg, _mm512_setzero_si512(), q8);
            const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), g, q8s);
            sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(dp, iq2_scales16(x[i].scales + 2 * ib)));
        }
        accf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), accf);
    }
    *s = 0.125f * _mm512_reduce_add_ps(accf);
}

// iq2_s: 8-bit index low parts, 2 high bits each in qh, explicit sign bytes.
MOE_CS_T512 void dot_iq2_s_q8_K_512(int n, float * s, size_t, const void * vx, size_t, const void * vy, size_t, int) {
    const block_iq2_s * x = (const block_iq2_s *) vx;
    const block_q8_K * y = (const block_q8_K *) vy;
    const int nb = n / QK_K;
    const __m512i sh = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i m3 = _mm512_set1_epi64(3);
    __m512 accf = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = _cvtsh_ss(x[i].d) * y[i].d;
        __m512i sumi = _mm512_setzero_si512();
        for (int ib = 0; ib < QK_K / 64; ++ib) {
            const uint32_t qh = (uint32_t) x[i].qh[2 * ib] | ((uint32_t) x[i].qh[2 * ib + 1] << 8);
            uint64_t lo, sg;
            memcpy(&lo, x[i].qs + 8 * ib, 8);
            memcpy(&sg, x[i].qs + QK_K / 8 + 8 * ib, 8);
            const __m512i hi = _mm512_slli_epi64(_mm512_and_si512(_mm512_srlv_epi64(_mm512_set1_epi64((long long) qh), sh), m3), 8);
            const __m512i idx = _mm512_or_si512(_mm512_cvtepu8_epi64(_mm_cvtsi64_si128((long long) lo)), hi);
            const __m512i g = _mm512_i64gather_epi64(idx, (const void *) iq2s_grid, 8);
            const __m512i q8 = _mm512_loadu_si512((const void *) (y[i].qs + 64 * ib));
            const __m512i q8s = _mm512_mask_sub_epi8(q8, (__mmask64) sg, _mm512_setzero_si512(), q8);
            const __m512i dp = _mm512_dpbusd_epi32(_mm512_setzero_si512(), g, q8s);
            sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(dp, iq2_scales16(x[i].scales + 2 * ib)));
        }
        accf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), accf);
    }
    *s = 0.125f * _mm512_reduce_add_ps(accf);
}

bool cpu_has_avx512_vnni() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}
#endif

struct role_math {
    moe_cs_role r;
    ggml_vec_dot_t dot = nullptr;
    bool q2 = false;            // dot_q2_0_q8_0 instead of ggml's kernel
    ggml_type vec_type = GGML_TYPE_COUNT;
    ggml_from_float_t quant = nullptr;
    size_t vec_bytes = 0;       // bytes of one quantized input row
#if defined(MOE_IQ_ROWS)
    moe_iq::rows_fn rows[9] = {};   // IQ up/gate kernels for 1..8 tokens
#endif
};

struct layer_math {
    bool ok = false;
    role_math role[3];          // up, gate, down
    bool shared_input = false;  // up and gate quantize their input the same way
};

struct pool_state {
    std::mutex mu;              // registration and start
    bool running = false;
    moe_cs_job * job = nullptr;
    const float * x = nullptr;
    float * out = nullptr;
    size_t x_stride = 0, out_stride = 0;
    int64_t max_ff = 0, max_embd = 0;
    std::vector<layer_math> layers;
    int threads = 1;
    int spin_ms = 250;
    int spin_us = 250000;   // spin window after the last job (GGML_MOE_CPU_SPLIT_SPIN_US, default SPIN_MS x 1000)
    int nap_us = 1000;      // longest leader nap once the window has passed (GGML_MOE_CPU_SPLIT_NAP_US)
    std::vector<std::thread> workers;

    // per-job state, written by thread 0 before `gen` moves
    const layer_math * cur = nullptr;
    int n_exp = 0, n_route = 0;
    int expert[MOE_CS_MAX_EXPERTS] = {};
    int first[MOE_CS_MAX_EXPERTS + 1] = {};
    int input_of[MOE_CS_MAX_ROUTES] = {}, inputs[MOE_CS_MAX_ROUTES] = {}, n_input = 0;
    bool q8_reuse = false;
    int group_tokens = 4;
    std::vector<uint8_t> xq;    // [2][MOE_CS_MAX_ROUTES] quantized input rows
    size_t xq_stride = 0;
    std::vector<float> hidden;  // [MOE_CS_MAX_ROUTES][max_ff]
    std::vector<uint8_t> hq;    // [MOE_CS_MAX_ROUTES] quantized hidden rows
    size_t hq_stride = 0;
    bool avx512 = false;       // GGML_MOE_CPU_SPLIT_AVX512 and the CPU has AVX-512 F/BW/VL/VNNI
    bool q2_fast = false;      // GGML_MOE_CPU_SPLIT_Q2_FAST: reordered hidden rows + dot_q2_0_rows for Q2_0 down
    size_t aux_stride = 0;     // floats per route in hq_aux: n/32 block scales, then n/64 offsets
    std::vector<float> hq_aux;
    int rows_ug = 16, rows_down = 64;
    bool iq_rows = false;      // GGML_MOE_CPU_SPLIT_ROWS: moe-iq-rows.h kernels, each chunk decoded once per expert
    int iq_variant = -1;       // moe_iq::get variant bits (GGML_MOE_CPU_SPLIT_ROWS: 1 = best mix, 2..17 = bits + 2)
    int iq_pf = 0;             // GGML_MOE_CPU_SPLIT_PF: prefetch distance of those kernels in bytes (0 = off)
    bool schedule = false;    // Assign quantization rows directly and claim math tiles in small groups.
    bool host_help = false;
    int host_cpu = -1;
    std::atomic<int> help_phase{-1}; // -1 closed, 1/3 open, 5/7 held by a helper
    std::atomic<unsigned long long> closed[4];

    alignas(64) std::atomic<unsigned long long> gen{0};
    alignas(64) std::atomic<int> next[4];
    alignas(64) std::atomic<unsigned long long> fin[4];
    alignas(64) std::atomic<int> sleepers{0};
    std::mutex sleep_mu;
    std::condition_variable sleep_cv;

    std::atomic<unsigned long long> st_jobs{0}, st_experts{0}, st_routes{0}, st_busy_ns{0}, st_null{0};
    unsigned long long phase_ns[4] = {};   // thread 0's time through each phase (barrier included), summed
    // host-posted jobs (the probe) use this instead of the mapped seq
    std::atomic<unsigned long long> local_seq{0}, local_done{0};
};

pool_state & P() {
    static pool_state * p = new pool_state();   // never destroyed: threads may still spin at exit
    return *p;
}

int items_of(const pool_state & s, int phase) {
    const layer_math & L = *s.cur;
    switch (phase) {
        case 0: return s.n_input * (L.shared_input ? 1 : 2);
        case 1: return s.n_exp * (int) ((L.role[0].r.n_out + s.rows_ug - 1) / s.rows_ug);
        case 2: return s.n_route;
        default: return s.n_exp * (int) ((L.role[2].r.n_out + s.rows_down - 1) / s.rows_down);
    }
}

const char * role_row(pool_state & s, const role_math & m, int expert, int64_t row) {
    const void * base = __atomic_load_n(&m.r.src[expert], __ATOMIC_ACQUIRE);
    if (!base) { return nullptr; }
    return (const char *) base + (size_t) row * m.r.row_bytes;
    (void) s;
}

void run_item(pool_state & s, int phase, int item) {
    const layer_math & L = *s.cur;
    if (phase == 0) {
        const int nr = s.n_input;
        const int k = item / nr, q = s.inputs[item % nr];
        const role_math & m = L.role[k];
        m.quant(s.x + (size_t) q * s.x_stride, s.xq.data() + ((size_t) k * MOE_CS_MAX_ROUTES + q) * s.xq_stride, m.r.n_in);
        return;
    }
    if (phase == 2) {
        const role_math & m = L.role[2];
        m.quant(s.hidden.data() + (size_t) item * s.max_ff, s.hq.data() + (size_t) item * s.hq_stride, m.r.n_in);
        if (s.q2_fast && m.q2) {
            float * aux = s.hq_aux.data() + (size_t) item * s.aux_stride;
            q2_reorder_hidden((int) m.r.n_in, (block_q8_0 *) (s.hq.data() + (size_t) item * s.hq_stride), aux,
                    aux + m.r.n_in / QK8_0);
        }
        return;
    }
    if (phase == 1) {
        const role_math & up = L.role[0];
        const role_math & gate = L.role[1];
        const int blocks = (int) ((up.r.n_out + s.rows_ug - 1) / s.rows_ug);
        const int i = item / blocks;
        const int64_t r0 = (int64_t) (item % blocks) * s.rows_ug, r1 = std::min<int64_t>(r0 + s.rows_ug, up.r.n_out);
        const int e = s.expert[i];
        const size_t xg = L.shared_input ? 0 : (size_t) MOE_CS_MAX_ROUTES;
#if defined(MOE_IQ_ROWS)
        if (up.rows[4] && gate.rows[4]) {
            // every chunk decoded once for up to 4 of the expert's tokens; same bits as the vec_dot loop below
            // (a type whose one-token case is left to ggml, rows[1] null, runs vec_dot row by row for that case)
            auto proj = [&](const role_math & m, const char * w, int nt, int64_t a, int64_t b, int nb,
                                const block_q8_K * const * y, float * const * o) {
                if (m.rows[nt]) { m.rows[nt](w, m.r.row_bytes, a, b, nb, y, o, s.iq_pf); return; }
                for (int t = 0; t < nt; ++t) {
                    for (int64_t row = a; row < b; ++row) {
                        m.dot((int) m.r.n_in, &o[t][row - a], 0, w + (size_t) row * m.r.row_bytes, 0, y[t], 0, 1);
                    }
                }
            };
            const char * wu = role_row(s, up, e, 0);
            const char * wg = role_row(s, gate, e, 0);
            const int nb = (int) (up.r.n_in / QK_K);
            for (int q0 = s.first[i]; q0 < s.first[i + 1]; q0 += s.group_tokens) {
                const int nt = std::min(s.group_tokens, s.first[i + 1] - q0);
                const block_q8_K * yu[8];
                const block_q8_K * yg[8];
                for (int t = 0; t < nt; ++t) {
                    const int q = s.input_of[q0 + t];
                    yu[t] = (const block_q8_K *) (s.xq.data() + (size_t) q * s.xq_stride);
                    yg[t] = (const block_q8_K *) (s.xq.data() + (xg + q) * s.xq_stride);
                }
                for (int64_t a = r0; a < r1; a += 16) {
                    const int64_t b = std::min<int64_t>(a + 16, r1);
                    float ub[8][16], gb[8][16];
                    float * uo[8], * go[8];
                    for (int t = 0; t < nt; ++t) { uo[t] = ub[t]; go[t] = gb[t]; }
                    if (wu && wg) {
                        proj(up, wu, nt, a, b, nb, yu, uo);
                        proj(gate, wg, nt, a, b, nb, yg, go);
                    } else {
                        for (int t = 0; t < nt; ++t) { for (int k = 0; k < 16; ++k) { ub[t][k] = 0.0f; gb[t][k] = 0.0f; } }
                    }
                    for (int t = 0; t < nt; ++t) {
                        for (int64_t row = a; row < b; ++row) {
                            const float u = ub[t][row - a], g = gb[t][row - a];
                            s.hidden[(size_t) (q0 + t) * s.max_ff + row] = g / (1.0f + expf(-g)) * u;
                        }
                    }
                }
            }
            if (!wu || !wg) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
            return;
        }
#endif
        for (int64_t row = r0; row < r1; ++row) {
            const char * wu = role_row(s, up, e, row);
            const char * wg = role_row(s, gate, e, row);
            for (int q = s.first[i]; q < s.first[i + 1]; ++q) {
                float u = 0.0f, g = 0.0f;
                if (wu && wg) {
                    const int input = s.input_of[q];
                    const uint8_t * xu = s.xq.data() + (size_t) input * s.xq_stride;
                    const uint8_t * xgp = s.xq.data() + (xg + input) * s.xq_stride;
                    if (up.q2) { dot_q2_0_q8_0((int) up.r.n_in, &u, wu, xu); } else { up.dot((int) up.r.n_in, &u, 0, wu, 0, xu, 0, 1); }
                    if (gate.q2) { dot_q2_0_q8_0((int) gate.r.n_in, &g, wg, xgp); } else { gate.dot((int) gate.r.n_in, &g, 0, wg, 0, xgp, 0, 1); }
                }
                s.hidden[(size_t) q * s.max_ff + row] = g / (1.0f + expf(-g)) * u;
            }
            if (!wu || !wg) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
        }
        return;
    }
    const role_math & down = L.role[2];
    const int blocks = (int) ((down.r.n_out + s.rows_down - 1) / s.rows_down);
    const int i = item / blocks;
    const int64_t r0 = (int64_t) (item % blocks) * s.rows_down, r1 = std::min<int64_t>(r0 + s.rows_down, down.r.n_out);
    const int e = s.expert[i];
    if (s.q2_fast && down.q2) {
        const char * w = role_row(s, down, e, r0);
        for (int q = s.first[i]; q < s.first[i + 1]; ++q) {
            float * o = s.out + (size_t) q * s.out_stride + r0;
            if (!w) { for (int64_t row = r0; row < r1; ++row) { o[row - r0] = 0.0f; } continue; }
            const float * aux = s.hq_aux.data() + (size_t) q * s.aux_stride;
            dot_q2_0_rows((int) down.r.n_in, (int) (r1 - r0), w, down.r.row_bytes,
                    (const block_q8_0 *) (s.hq.data() + (size_t) q * s.hq_stride), aux, aux + down.r.n_in / QK8_0, o);
        }
        if (!w) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
        return;
    }
    for (int64_t row = r0; row < r1; ++row) {
        const char * w = role_row(s, down, e, row);
        for (int q = s.first[i]; q < s.first[i + 1]; ++q) {
            float v = 0.0f;
            if (w) {
                const uint8_t * h = s.hq.data() + (size_t) q * s.hq_stride;
                if (down.q2) { dot_q2_0_q8_0((int) down.r.n_in, &v, w, h); } else { down.dot((int) down.r.n_in, &v, 0, w, 0, h, 0, 1); }
            }
            s.out[(size_t) q * s.out_stride + row] = v;
        }
        if (!w) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
    }
}

// Every thread runs all four phases of job number `n` (1-based); the fin counters only grow, so a thread that is
// late to notice a finished phase never misses it.
void run_job_phases(pool_state & s, unsigned long long n, int worker, bool lead = false) {
    const unsigned long long target = n * (unsigned long long) s.threads;
    auto t = clk::now();
    for (int p = 0; p < 4; ++p) {
        const int total = items_of(s, p);
        if (s.schedule && (p == 0 || p == 2)) {
            // Each row has one writer. The phase barrier publishes it before any consumer runs.
            for (int it = worker; it < total; it += s.threads) {
                run_item(s, p, it);
            }
        } else {
            // Keep at least three claims per worker, with at most four adjacent tiles per claim.
            const int chunk = s.schedule ? std::max(1, std::min(4, total / (3 * s.threads))) : 1;
            for (int begin = s.next[p].fetch_add(chunk, std::memory_order_relaxed); begin < total;
                    begin = s.next[p].fetch_add(chunk, std::memory_order_relaxed)) {
                for (int it = begin; it < std::min(begin + chunk, total); ++it) {
                    run_item(s, p, it);
                }
            }
        }
        const auto finished = s.fin[p].fetch_add(1, std::memory_order_acq_rel) + 1;
        if (s.host_help) {
            if (finished == target) {
                if (p == 1 || p == 3) {
                    // Close admission and wait for an in-flight helper before publishing this phase.
                    int open = p;
                    while (!s.help_phase.compare_exchange_weak(open, -1, std::memory_order_acq_rel)) {
                        open = p;
                        cpu_relax();
                    }
                } else {
                    s.help_phase.store(p + 1, std::memory_order_release);
                }
                s.closed[p].store(n, std::memory_order_release);
            } else {
                while (s.closed[p].load(std::memory_order_acquire) < n) { cpu_relax(); }
            }
        }
        if (p < 3) {
            while (s.fin[p].load(std::memory_order_acquire) < target) { cpu_relax(); }
        }
        if (lead) {
            if (p == 3) {
                while (s.fin[3].load(std::memory_order_acquire) < target) { cpu_relax(); }
            }
            const auto now = clk::now();
            s.phase_ns[p] += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
            t = now;
        }
    }
}

void pin_thread(int cpu) {
#if defined(__linux__)
    if (cpu < 0) { return; }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void) cpu;
#endif
}

// First logical CPU of each physical core, in core order (Linux sysfs); empty when unknown.
std::vector<int> core_leaders() {
    std::vector<int> out;
#if defined(__linux__)
    std::set<std::pair<int, int>> seen;
    const int n = (int) std::thread::hardware_concurrency();
    for (int c = 0; c < n; ++c) {
        int pkg = 0, core = -1;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        if (FILE * f = fopen((base + "core_id").c_str(), "r")) { if (fscanf(f, "%d", &core) != 1) { core = -1; } fclose(f); }
        if (FILE * f = fopen((base + "physical_package_id").c_str(), "r")) { if (fscanf(f, "%d", &pkg) != 1) { pkg = 0; } fclose(f); }
        if (core < 0) { return {}; }
        if (seen.insert({pkg, core}).second) { out.push_back(c); }
    }
#endif
    return out;
}

void sleep_us(int us) {
#if defined(__linux__)
    timespec ts{0, (long) us * 1000};
    nanosleep(&ts, nullptr);
#else
    std::this_thread::sleep_for(std::chrono::microseconds(us));
#endif
}

void worker_main(pool_state * sp, int idx, int cpu) {
    pool_state & s = *sp;
    pin_thread(cpu);
    unsigned long long seen = 0;
    for (;;) {
        auto spin_until = clk::now() + std::chrono::microseconds(s.spin_us);
        unsigned long long g;
        int polls = 0;
        while ((g = s.gen.load(std::memory_order_acquire)) == seen) {
            cpu_relax();
            if (++polls == 1024) {
                polls = 0;
                if (clk::now() > spin_until) {
                    std::unique_lock<std::mutex> lk(s.sleep_mu);
                    s.sleepers.fetch_add(1);
                    s.sleep_cv.wait(lk, [&] { return s.gen.load() != seen; });
                    s.sleepers.fetch_sub(1);
                }
            }
        }
        seen = g;
        run_job_phases(s, g, idx);
    }
}

// Thread 0: watches the mapped job (and host-posted probe jobs), runs its share, publishes `done`.
void leader_main(pool_state * sp, int cpu) {
    pool_state & s = *sp;
    pin_thread(cpu);
#if defined(__linux__)
    (void) prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);   // naps end within ~1 us of the asked time, not +50 us
#endif
    unsigned long long last_seq = s.job->seq;
    unsigned long long last_local = 0;
    unsigned long long n = 0;
    auto spin_until = clk::now() + std::chrono::microseconds(s.spin_us);
    int idle_us = std::min(20, s.nap_us);
    for (;;) {
        const unsigned long long seq = s.job->seq;
        const unsigned long long local = s.local_seq.load(std::memory_order_acquire);
        const bool card = seq != last_seq;
        if (!card && local == last_local) {
            if (clk::now() < spin_until) {
                for (int i = 0; i < 64; ++i) { cpu_relax(); }
            } else {
                // Idle (prompt reading, between requests): back off to 1 ms naps so an idle pool costs almost no
                // CPU; the first layer after a long idle waits at most that long, then the spin window resumes.
                sleep_us(idle_us);
                idle_us = std::min(s.nap_us, idle_us * 2);
            }
            continue;
        }
        idle_us = std::min(20, s.nap_us);
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto t0 = clk::now();
        const int layer = s.job->layer;
        const int n_exp = std::min<int>(s.job->n_exp, MOE_CS_MAX_EXPERTS);
        s.cur = layer >= 0 && layer < (int) s.layers.size() && s.layers[layer].ok ? &s.layers[layer] : nullptr;
        s.n_exp = s.cur ? std::max(0, n_exp) : 0;
        for (int i = 0; i < s.n_exp; ++i) { s.expert[i] = s.job->expert[i]; }
        bool sane = true;
        for (int i = 0; i <= s.n_exp; ++i) {
            s.first[i] = s.job->first[i];
            sane = sane && s.first[i] >= (i ? s.first[i - 1] : 0) && s.first[i] <= MOE_CS_MAX_ROUTES;
        }
        s.n_route = s.n_exp ? s.first[s.n_exp] : 0;
        for (int i = 0; sane && i < s.n_exp; ++i) {
            sane = s.expert[i] >= 0 && s.expert[i] < s.cur->role[0].r.n_expert;
        }
        if (s.cur && sane && s.n_exp > 0 && s.first[0] == 0) {
            s.n_input = 0;
            for (int q = 0; q < s.n_route; ++q) {
                int input = q;
                if (s.q8_reuse && card) {
                    for (int j = 0; j < s.n_input; ++j) {
                        const int prev = s.inputs[j];
                        if (s.job->input_row[prev] == s.job->input_row[q]) { input = prev; break; }
                    }
                }
                s.input_of[q] = input;
                if (input == q) { s.inputs[s.n_input++] = q; }
            }
            ++n;
            for (int p = 0; p < 4; ++p) { s.next[p].store(0, std::memory_order_relaxed); }
            s.gen.store(n, std::memory_order_seq_cst);
            if (s.sleepers.load(std::memory_order_seq_cst) > 0) {
                std::lock_guard<std::mutex> lk(s.sleep_mu);
                s.sleep_cv.notify_all();
            }
            run_job_phases(s, n, 0, true);
            const unsigned long long target = n * (unsigned long long) s.threads;
            while (s.fin[3].load(std::memory_order_acquire) < target) { cpu_relax(); }
            s.st_experts.fetch_add((unsigned) s.n_exp, std::memory_order_relaxed);
            s.st_routes.fetch_add((unsigned) s.n_route, std::memory_order_relaxed);
        } else {
            s.st_null.fetch_add(1, std::memory_order_relaxed);
        }
        s.st_jobs.fetch_add(1, std::memory_order_relaxed);
        const unsigned long long busy =
            (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
        s.st_busy_ns.fetch_add(busy, std::memory_order_relaxed);
        if (card) { s.job->busy_ns = busy; }   // the card's join reads it to correct the planner's cost (ADAPT)
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (card) {
            last_seq = seq;
            s.job->done = seq;
        } else {
            last_local = local;
            s.local_done.store(local, std::memory_order_release);
        }
        spin_until = clk::now() + std::chrono::microseconds(s.spin_us);
    }
}

} // namespace

bool moe_cs_start(moe_cs_job * job, const float * x, size_t x_stride, float * out, size_t out_stride,
        int64_t max_ff, int64_t max_embd) {
    pool_state & s = P();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.running) {
        return s.job == job && max_ff <= s.max_ff && max_embd <= s.max_embd;
    }
    ggml_cpu_init();
    s.job = job; s.x = x; s.out = out; s.x_stride = x_stride; s.out_stride = out_stride;
    s.max_ff = max_ff; s.max_embd = max_embd;
    // Quantized rows are never larger than the f32 rows they come from.
    s.xq_stride = GGML_PAD((size_t) max_embd * sizeof(float), 64);
    s.hq_stride = GGML_PAD((size_t) max_ff * sizeof(float), 64);
    s.xq.assign(2 * MOE_CS_MAX_ROUTES * s.xq_stride, 0);
    s.hidden.assign((size_t) MOE_CS_MAX_ROUTES * max_ff, 0.0f);
    s.hq.assign(MOE_CS_MAX_ROUTES * s.hq_stride, 0);
    s.q2_fast = env_int("GGML_MOE_CPU_SPLIT_Q2_FAST", 0, 0, 1) != 0;
    s.q8_reuse = env_int("GGML_MOE_CPU_SPLIT_Q8_REUSE", 0, 0, 1) != 0;
    s.group_tokens = env_int("GGML_MOE_CPU_SPLIT_REUSE8", 0, 0, 1) ? 8 : 4;
#if defined(MOE_CS_AVX512)
    s.avx512 = env_int("GGML_MOE_CPU_SPLIT_AVX512", 0, 0, 1) != 0 && cpu_has_avx512_vnni();
#endif
    s.aux_stride = (size_t) max_ff / QK8_0 + (size_t) max_ff / QK2_0 + 16;
    s.hq_aux.assign(MOE_CS_MAX_ROUTES * s.aux_stride, 0.0f);
    s.rows_ug = env_int("GGML_MOE_CPU_SPLIT_ROWS_UG", 16, 1, 4096);
    s.rows_down = env_int("GGML_MOE_CPU_SPLIT_ROWS_DOWN", 64, 1, 8192);
#if defined(MOE_IQ_ROWS)
    {   // 1 = on with the measured best kernel mix; 2..17 = on with moe_iq::get variant bits (value - 2)
        const int v = env_int("GGML_MOE_CPU_SPLIT_ROWS", 0, 0, 17);
        s.iq_variant = v == 0 ? -1 : v == 1 ? moe_iq::variant_best : v - 2;
    }
    s.iq_rows = s.iq_variant >= 0;
    s.iq_pf = env_int("GGML_MOE_CPU_SPLIT_PF", 0, 0, 1 << 16);
#endif
    s.spin_ms = env_int("GGML_MOE_CPU_SPLIT_SPIN_MS", 250, 0, 600000);
    s.spin_us = env_int("GGML_MOE_CPU_SPLIT_SPIN_US", s.spin_ms * 1000, 0, 600000000);
    s.nap_us = env_int("GGML_MOE_CPU_SPLIT_NAP_US", 1000, 1, 100000);
    s.schedule = env_int("GGML_MOE_CPU_SPLIT_SCHEDULE", 0, 0, 1) != 0;
    s.host_help = env_int("GGML_MOE_CPU_SPLIT_HOST", 0, 0, 1) != 0;
    for (int p = 0; p < 4; ++p) { s.next[p].store(0); s.fin[p].store(0); s.closed[p].store(0); }
    // One thread per physical core, never an SMT sibling, never core 0 (the OS and the desktop) and never the
    // core of the thread that drives the card: a spinning pool thread sharing a core with it is time-sliced in
    // scheduler ticks (the first build measured 6 ms per job that way). GGML_CPU_EXPERT_PIN=1 pins llama's main
    // thread to the second physical core, and so does any single-core affinity of the calling thread; both are
    // skipped. Threads take the highest cores first, away from ggml's own pinned workers (cores 1.. upward).
    std::vector<int> cores = core_leaders();
    std::vector<int> avail;
    std::string skipped = "0";
    if (!cores.empty()) {
        // The cooperative host occupies the reserved core instead of competing with a pool worker.
        if (s.host_help) { s.host_cpu = cores[0]; pin_thread(s.host_cpu); }
        std::vector<char> skip(cores.size(), 0);
        skip[0] = 1;
        const char * ep = getenv("GGML_CPU_EXPERT_PIN");
        if (!s.host_help && ep && strcmp(ep, "1") == 0 && cores.size() > 2) { skip[1] = 1; }
#if defined(__linux__)
        cpu_set_t mine;
        if (pthread_getaffinity_np(pthread_self(), sizeof(mine), &mine) == 0 && CPU_COUNT(&mine) <= 2) {
            // pinned caller: skip the physical core of each CPU it may run on
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (!CPU_ISSET(c, &mine)) { continue; }
                int core = -1, pkg = 0;
                const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
                if (FILE * f = fopen((base + "core_id").c_str(), "r")) { if (fscanf(f, "%d", &core) != 1) { core = -1; } fclose(f); }
                if (FILE * f = fopen((base + "physical_package_id").c_str(), "r")) { if (fscanf(f, "%d", &pkg) != 1) { pkg = 0; } fclose(f); }
                for (size_t i = 0; i < cores.size(); ++i) {
                    int ci = -1, pi = 0;
                    const std::string bi = "/sys/devices/system/cpu/cpu" + std::to_string(cores[i]) + "/topology/";
                    if (FILE * f = fopen((bi + "core_id").c_str(), "r")) { if (fscanf(f, "%d", &ci) != 1) { ci = -1; } fclose(f); }
                    if (FILE * f = fopen((bi + "physical_package_id").c_str(), "r")) { if (fscanf(f, "%d", &pi) != 1) { pi = 0; } fclose(f); }
                    if (ci == core && pi == pkg) { skip[i] = 1; }
                }
            }
        }
#endif
        skipped.clear();
        for (size_t i = cores.size(); i-- > 0;) {
            if (!skip[i]) { avail.push_back(cores[i]); }
        }
        for (size_t i = 0; i < cores.size(); ++i) {
            if (skip[i]) { skipped += (skipped.empty() ? "" : ",") + std::to_string(cores[i]); }
        }
    }
    // GGML_MOE_CPU_SPLIT_SMT=1 (default off): also use the second hardware thread of each pool core, listed after
    // all the first threads. The grid lookups of the IQ dot loops stall on load latency, which a sibling can fill.
    const bool smt = env_int("GGML_MOE_CPU_SPLIT_SMT", 0, 0, 1) != 0;
#if defined(__linux__)
    if (smt) {
        const size_t n_lead = avail.size();
        for (size_t i = 0; i < n_lead; ++i) {
            const std::string fn = "/sys/devices/system/cpu/cpu" + std::to_string(avail[i]) + "/topology/thread_siblings_list";
            FILE * f = fopen(fn.c_str(), "r");
            if (!f) { continue; }
            char buf[128] = {};
            if (!fgets(buf, sizeof(buf), f)) { buf[0] = 0; }
            fclose(f);
            for (char * c = buf; *c;) {   // "a,b" or "a-b"
                if (*c < '0' || *c > '9') { ++c; continue; }
                const int lo = (int) strtol(c, &c, 10);
                int hi = lo;
                if (*c == '-') { ++c; hi = (int) strtol(c, &c, 10); }
                for (int v = lo; v <= hi; ++v) { if (v != avail[i]) { avail.push_back(v); } }
            }
        }
    }
#endif
    const int hw_free = !avail.empty() ? (int) avail.size() : (int) std::max(1u, std::thread::hardware_concurrency() / 2 - 1);
    s.threads = env_int("GGML_MOE_CPU_SPLIT_THREADS", hw_free, 1, 64);
    const bool pin = env_int("GGML_MOE_CPU_SPLIT_PIN", 1, 0, 1) != 0 && !avail.empty();
    auto cpu_of = [&](int i) { return pin ? avail[(size_t) i % avail.size()] : -1; };
    s.running = true;
    s.workers.emplace_back(leader_main, &s, cpu_of(0));
    for (int i = 1; i < s.threads; ++i) {
        s.workers.emplace_back(worker_main, &s, i, cpu_of(i));
    }
    for (auto & t : s.workers) { t.detach(); }
    std::string where;
    for (int i = 0; pin && i < s.threads; ++i) { where += (i ? "," : "") + std::to_string(cpu_of(i)); }
    fprintf(stderr, "[moe-cpu-split] pool: %d threads, %s%s (skipped cores at cpus %s), spin %d us, nap %d us, rows up/gate %d down %d%s%s%s\n",
            s.threads, pin ? "pinned to cpus " : "unpinned", where.c_str(), skipped.c_str(), s.spin_us, s.nap_us, s.rows_ug, s.rows_down,
            s.q2_fast ? ", Q2_0 down fast path" : "", smt ? ", SMT siblings" : "", s.avx512 ? ", AVX-512 VNNI IQ dots" : "");
    if (s.iq_rows) { fprintf(stderr, "[moe-cpu-split] IQ up/gate row kernels (each chunk decoded once per expert), variant %d, prefetch %d B\n", s.iq_variant, s.iq_pf); }
    if (s.schedule) { fprintf(stderr, "[moe-cpu-split] static quantization rows, grouped math claims\n"); }
    if (s.host_help) { fprintf(stderr, "[moe-cpu-split] waiting host drains math tiles on cpu %d\n", s.host_cpu); }
    fprintf(stderr, "[moe-cpu-split] input Q8 reuse %s, IQ token group %d\n", s.q8_reuse ? "on" : "off", s.group_tokens);
    return true;
}

bool moe_cs_running() { return P().running; }

bool moe_cs_help_enabled() { return P().running && P().host_help; }

bool moe_cs_help() {
    pool_state & s = P();
    if (!s.running || !s.host_help) { return false; }
#if defined(__linux__)
    if (s.host_cpu >= 0 && sched_getcpu() != s.host_cpu) { pin_thread(s.host_cpu); }
#endif
    int phase = s.help_phase.load(std::memory_order_acquire);
    if ((phase != 1 && phase != 3) ||
        !s.help_phase.compare_exchange_strong(phase, phase + 4, std::memory_order_acq_rel)) { return false; }
    // Ownership prevents the final pool worker from advancing the phase or replacing the job state.
    const int total = items_of(s, phase);
    const int chunk = s.schedule ? std::max(1, std::min(4, total / (3 * s.threads))) : 1;
    const int begin = s.next[phase].fetch_add(chunk, std::memory_order_relaxed);
    if (begin < total) {
        for (int it = begin; it < std::min(begin + chunk, total); ++it) { run_item(s, phase, it); }
    }
    s.help_phase.store(phase, std::memory_order_release);
    return begin < total;
}

int moe_cs_threads() { return P().threads; }

bool moe_cs_register_layer(int layer, const moe_cs_role roles[3]) {
    pool_state & s = P();
    std::lock_guard<std::mutex> lock(s.mu);
    if (layer < 0 || layer >= 4096) { return false; }
    layer_math L;
    bool ok = roles[0].n_in == roles[1].n_in && roles[0].n_out == roles[1].n_out && roles[2].n_in == roles[0].n_out &&
              roles[2].n_out == roles[0].n_in && roles[0].n_expert == roles[1].n_expert && roles[0].n_expert == roles[2].n_expert &&
              roles[0].n_out <= s.max_ff && roles[0].n_in <= s.max_embd;
    for (int k = 0; ok && k < 3; ++k) {
        role_math & m = L.role[k];
        m.r = roles[k];
        ok = m.r.src != nullptr && m.r.type >= 0 && m.r.type < GGML_TYPE_COUNT;
        if (!ok) { break; }
        const ggml_type t = (ggml_type) m.r.type;
        const auto * tr = ggml_get_type_traits_cpu(t);
        ok = tr && tr->vec_dot && m.r.n_in % ggml_blck_size(t) == 0 && m.r.row_bytes == ggml_row_size(t, m.r.n_in);
        if (!ok) { break; }
        m.q2 = t == GGML_TYPE_Q2_0;
        m.dot = tr->vec_dot;
#if defined(MOE_CS_AVX512)
        if (s.avx512 && t == GGML_TYPE_IQ3_S) { m.dot = dot_iq3_s_q8_K_512; }
        if (s.avx512 && t == GGML_TYPE_IQ2_XXS) { m.dot = dot_iq2_xxs_q8_K_512; }
        if (s.avx512 && t == GGML_TYPE_IQ2_XS) { m.dot = dot_iq2_xs_q8_K_512; }
        if (s.avx512 && t == GGML_TYPE_IQ2_S) { m.dot = dot_iq2_s_q8_K_512; }
        if (s.avx512 && t == GGML_TYPE_IQ3_XXS) { m.dot = dot_iq3_xxs_q8_K_512; }
#endif
        m.vec_type = tr->vec_dot_type;
#if defined(MOE_IQ_ROWS)
        for (int nt = 1; nt <= s.group_tokens; ++nt) {
            m.rows[nt] = s.iq_rows && !s.avx512 && k < 2 && m.vec_type == GGML_TYPE_Q8_K ? moe_iq::get(m.r.type, nt, s.iq_variant) : nullptr;
        }
#endif
        const auto * vt = ggml_get_type_traits_cpu(m.vec_type);
        m.quant = vt ? vt->from_float : nullptr;
        ok = m.quant && m.r.n_in % ggml_blck_size(m.vec_type) == 0 &&
             (!m.q2 || m.vec_type == GGML_TYPE_Q8_0);
        m.vec_bytes = ok ? ggml_row_size(m.vec_type, m.r.n_in) : 0;
    }
    L.ok = ok;
    L.shared_input = ok && L.role[0].vec_type == L.role[1].vec_type;
    if ((int) s.layers.size() <= layer) { s.layers.resize(layer + 1); }
    // The pool reads the layer table only while a card job is in flight; registration happens at session
    // preparation, outside any graph, so no job is running.
    s.layers[layer] = L;
    return ok;
}

bool moe_cs_layer_ok(int layer) {
    pool_state & s = P();
    std::lock_guard<std::mutex> lock(s.mu);
    return layer >= 0 && layer < (int) s.layers.size() && s.layers[layer].ok;
}

// Run-queue wait of every thread of this process so far, in ns (/proc/self/task/*/schedstat field 2): time a thread
// was ready to run but another task held its cpu. The probe reads it before and after its timed jobs, so CPU contention
// from other programs on the pool's cpus shows up in every log (0 when schedstat is missing).
static double moe_cs_runq_wait_ns() {
    double sum = 0.0;
    DIR * d = opendir("/proc/self/task");
    if (!d) { return 0.0; }
    while (dirent * e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') { continue; }
        char path[96];
        snprintf(path, sizeof(path), "/proc/self/task/%s/schedstat", e->d_name);
        FILE * f = fopen(path, "r");
        if (!f) { continue; }
        unsigned long long run = 0, wait = 0;
        if (fscanf(f, "%llu %llu", &run, &wait) == 2) { sum += (double) wait; }
        fclose(f);
    }
    closedir(d);
    return sum;
}

bool moe_cs_probe(double * route_ns, double * fixed_ns) {
    pool_state & s = P();
    int layer = -1;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (!s.running) { return false; }
        for (int l = 0; l < (int) s.layers.size(); ++l) {
            if (s.layers[l].ok) { layer = l; break; }
        }
    }
    if (layer < 0) { return false; }
    const layer_math & L = s.layers[layer];
    // Experts with a host source; the probe reads real weights. Enough of them (64 x ~2 MB) that the rotation
    // below reads from memory, not from the last-level cache, as a decode's missed experts do.
    std::vector<int> experts;
    for (int e = 0; e < (int) L.role[0].r.n_expert && experts.size() < 64; ++e) {
        bool all = true;
        for (int k = 0; k < 3; ++k) { all = all && __atomic_load_n(&L.role[k].r.src[e], __ATOMIC_ACQUIRE) != nullptr; }
        if (all) { experts.push_back(e); }
    }
    if (experts.size() < 4) { return false; }
    float * x = const_cast<float *>(s.x);
    for (int q = 0; q < MOE_CS_MAX_ROUTES; ++q) {
        for (int64_t i = 0; i < L.role[0].r.n_in; ++i) {
            x[(size_t) q * s.x_stride + i] = 0.02f * (float) (((q * 131 + i * 7919) % 101) - 50);
        }
    }
    auto post = [&](int e0, int n_exp) {
        s.job->layer = layer;
        s.job->n_exp = n_exp;
        for (int i = 0; i < n_exp; ++i) {
            s.job->expert[i] = experts[((size_t) e0 + i) % experts.size()];
            s.job->first[i] = i;
        }
        s.job->first[n_exp] = n_exp;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const unsigned long long want = s.local_seq.load() + 1;
        s.local_seq.store(want, std::memory_order_release);
        while (s.local_done.load(std::memory_order_acquire) != want) { if (!moe_cs_help()) { cpu_relax(); } }
    };
    // Touch every probe expert once first (page faults on first touch of mapped weights are not the steady cost).
    for (size_t e = 0; e < experts.size(); e += 4) { post((int) e, 4); }
    for (int p = 0; p < 4; ++p) { s.phase_ns[p] = 0; }
    auto run = [&](int n_exp) {
        double best = 1e30;
        for (int rep = 0; rep < 24; ++rep) {
            s.job->layer = layer;
            s.job->n_exp = n_exp;
            for (int i = 0; i < n_exp; ++i) {
                s.job->expert[i] = experts[((size_t) rep * n_exp + i) % experts.size()];
                s.job->first[i] = i;
            }
            s.job->first[n_exp] = n_exp;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const unsigned long long want = s.local_seq.load() + 1;
            const auto t0 = clk::now();
            s.local_seq.store(want, std::memory_order_release);
            while (s.local_done.load(std::memory_order_acquire) != want) { if (!moe_cs_help()) { cpu_relax(); } }
            const double ns = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
            if (rep >= 4) { best = std::min(best, ns); }
        }
        return best;
    };
    const double t0 = run(0);   // empty jobs: the host-side handshake alone
    for (int p = 0; p < 4; ++p) { s.phase_ns[p] = 0; }
    const double q0 = moe_cs_runq_wait_ns();
    const auto w0 = clk::now();
    const double t1 = run(1);
    unsigned long long ph1[4];
    for (int p = 0; p < 4; ++p) { ph1[p] = s.phase_ns[p] / 24; s.phase_ns[p] = 0; }
    const double t4 = run(4);
    const double wall_ns = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - w0).count();
    const double stolen = wall_ns > 0 ? (moe_cs_runq_wait_ns() - q0) / (wall_ns * (double) std::max(1, s.threads)) : 0.0;
    unsigned long long ph4[4];
    for (int p = 0; p < 4; ++p) { ph4[p] = s.phase_ns[p] / 24; s.phase_ns[p] = 0; }
    fprintf(stderr, "[moe-cpu-split] probe phases (mean us, quantize x / up+gate / quantize hidden / down): "
            "1 expert %.1f %.1f %.1f %.1f, 4 experts %.1f %.1f %.1f %.1f\n",
            ph1[0] / 1e3, ph1[1] / 1e3, ph1[2] / 1e3, ph1[3] / 1e3, ph4[0] / 1e3, ph4[1] / 1e3, ph4[2] / 1e3, ph4[3] / 1e3);
    const double per = std::max(1000.0, (t4 - t1) / 3.0);
    *route_ns = per;
    *fixed_ns = std::max(0.0, t1 - per);
    // the probe's jobs count as work in the statistics; clear them
    s.st_jobs = 0; s.st_experts = 0; s.st_routes = 0; s.st_busy_ns = 0; s.st_null = 0;
    fprintf(stderr, "[moe-cpu-split] probe on layer %d: empty job %.1f us, 1 expert %.1f us, 4 experts %.1f us -> %.1f us per (expert, route) + %.1f us per job\n",
            layer, t0 / 1000.0, t1 / 1000.0, t4 / 1000.0, per / 1000.0, *fixed_ns / 1000.0);
    // Contention check (2026-10-23): a home run with git and simulations on the pool's cpus probed 116.9 us per expert
    // against a clean 65 us and wrote at 26.7 tok/s. Two signals, either one warns:
    //  - share of the pool's time spent waiting for a cpu during the timed jobs (over 25% = about 1.33x slower or worse);
    //  - GGML_MOE_CPU_SPLIT_PROBE_EXPECT_US=<clean 1-expert us for this box>: the 1-expert time is over 2x that.
    size_t expert_bytes = 0;
    for (int k = 0; k < 3; ++k) { expert_bytes += L.role[k].r.row_bytes * (size_t) L.role[k].r.n_out; }
    const char * ex = getenv("GGML_MOE_CPU_SPLIT_PROBE_EXPECT_US");
    const double expect_us = ex ? atof(ex) : 0.0;
    fprintf(stderr, "[moe-cpu-split] probe check: %.2f MB per expert, %.1f GB/s for 1 expert, %.1f GB/s for 4; "
            "pool waited for a cpu %.1f%% of the time%s\n", expert_bytes / 1e6, expert_bytes / std::max(1.0, t1),
            4.0 * expert_bytes / std::max(1.0, t4), 100.0 * stolen,
            expect_us > 0 ? "" : " (set GGML_MOE_CPU_SPLIT_PROBE_EXPECT_US for a time check)");
    if (stolen > 0.25 || (expect_us > 0 && t1 / 1000.0 > 2.0 * expect_us)) {
        fprintf(stderr, "[moe-cpu-split] WARN: CPU CONTENTION at load: 1 expert %.1f us (expected %s), pool waited for a cpu "
                "%.1f%% of the time. Other programs are running on the pool's cpus; split timings from this run are not "
                "valid.\n", t1 / 1000.0, expect_us > 0 ? (std::to_string(expect_us) + " us").c_str() : "not set",
                100.0 * stolen);
    }
    return true;
}

bool moe_cs_bench(int layer, const int * experts, int n_list, int n_exp, int reps, double * ns_per_job,
        double phase_ns[4]) {
    pool_state & s = P();
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (!s.running || layer < 0 || layer >= (int) s.layers.size() || !s.layers[layer].ok) { return false; }
    }
    if (n_exp < 0 || n_exp > MOE_CS_MAX_EXPERTS || n_exp > MOE_CS_MAX_ROUTES || n_list < 1 || reps < 1) { return false; }
    for (int p = 0; p < 4; ++p) { s.phase_ns[p] = 0; }
    double total = 0.0;
    for (int rep = 0; rep < reps; ++rep) {
        s.job->layer = layer;
        s.job->n_exp = n_exp;
        for (int i = 0; i < n_exp; ++i) {
            s.job->expert[i] = experts[((size_t) rep * n_exp + i) % (size_t) n_list];
            s.job->first[i] = i;
        }
        s.job->first[n_exp] = n_exp;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        const unsigned long long want = s.local_seq.load() + 1;
        const auto t0 = clk::now();
        s.local_seq.store(want, std::memory_order_release);
        while (s.local_done.load(std::memory_order_acquire) != want) { if (!moe_cs_help()) { cpu_relax(); } }
        total += (double) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
    }
    *ns_per_job = total / reps;
    for (int p = 0; p < 4; ++p) { phase_ns[p] = (double) s.phase_ns[p] / reps; s.phase_ns[p] = 0; }
    s.st_jobs = 0; s.st_experts = 0; s.st_routes = 0; s.st_busy_ns = 0; s.st_null = 0;
    return true;
}

void moe_cs_stats(unsigned long long out[9]) {
    pool_state & s = P();
    for (int p = 0; p < 4; ++p) { out[5 + p] = s.phase_ns[p]; }
    out[0] = s.st_jobs.load();
    out[1] = s.st_experts.load();
    out[2] = s.st_routes.load();
    out[3] = s.st_busy_ns.load();
    out[4] = s.st_null.load();
}

bool moe_cs_iq_rows(int type, int nt, const void * w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_in,
        const void * const * y, float * const * out, int prefetch, int variant) {
#if defined(MOE_IQ_ROWS)
    const moe_iq::rows_fn f = moe_iq::get(type, nt, variant);
    if (!f || n_in % QK_K != 0) { return false; }
    f((const char *) w, row_bytes, r0, r1, (int) (n_in / QK_K), (const block_q8_K * const *) y, out, prefetch);
    return true;
#else
    (void) type; (void) nt; (void) w; (void) row_bytes; (void) r0; (void) r1; (void) n_in; (void) y; (void) out; (void) prefetch; (void) variant;
    return false;
#endif
}
