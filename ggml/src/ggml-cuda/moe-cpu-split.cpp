// fork MOEGAP: host side of the CPU expert split (see moe-cpu-split.h).
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
#include "moe-ai-rows.h"
#include "../ggml-cpu/quants.h"
#include "../ggml-cpu/vec.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// fork AIPFIX: one line the first time the host computes an AI+ unit expert, with the decode path it takes
static void moe_ai_log_first(const char * role) {
    static std::atomic<bool> logged{false};
    if (logged.exchange(true)) { return; }
#if defined(__AVX2__) && defined(__FMA__)
    const char * isa = moe_ai::has_avx512() ? "avx512" : "avx2";
#else
    const char * isa = "scalar";
#endif
#if defined(__AVX2__) && defined(__FMA__)
    if (moe_ai::use_host_q8()) { isa = moe_ai::use_q8_vbmi() ? "avx512-vbmi" : "avx2"; } // q8: AVX2 unless GGML_MOE_AI_HOST_Q8_VBMI=1
#endif
    fprintf(stderr, "[moe-cpu-split] first host AI+ expert compute: role=%s decode=%s act=%s%s\n", role, isa,
            moe_ai::use_host_q8() ? "q8 (default; GGML_MOE_AI_HOST_Q8=0 = f32)" : "f32 (GGML_MOE_AI_HOST_Q8=0)",
            moe_ai::use_host_q8() && moe_ai::use_q8_lv() ? " lv=on (GGML_MOE_AI_HOST_Q8_LV=1)" : "");
}

namespace {

// Acquire load of an expert address the card side publishes. MSVC has no __atomic builtins; a volatile read there is
// an acquire load on x64 (/volatile:ms, the default), the only target CUDA builds for on Windows.
inline const void * moe_cs_load_acquire(const void * const * p) {
#if defined(_MSC_VER) && !defined(__clang__)
    return *(const void * const volatile *) p;
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

// Read prefetch into all cache levels. MSVC has no __builtin_prefetch; _mm_prefetch with T0 is the same hint on x64.
inline void moe_cs_prefetch(const void * p) {
#if defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_X64)
    _mm_prefetch((const char *) p, _MM_HINT_T0);
#else
    (void) p;
#endif
#else
    __builtin_prefetch(p, 0, 3);
#endif
}

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

void q2_reorder_hidden(int n, block_q8_0 * y, float * yd, float * yc, bool shuffle_bytes) {
    const int nb = n / QK8_0;
#if defined(__AVX2__)
    const __m256i shuffle = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
            0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i lanes = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
#else
    (void) shuffle_bytes;
#endif
    for (int i = 0; i < nb; ++i) {
        int sum = 0;
#if defined(__AVX2__)
        if (shuffle_bytes) {
            const __m256i q = _mm256_loadu_si256((const __m256i *) y[i].qs);
            const __m256i reordered = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(q, shuffle), lanes);
            _mm256_storeu_si256((__m256i *) y[i].qs, reordered);
            const __m256i sums = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), q), _mm256_set1_epi16(1));
            __m128i total = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
            total = _mm_hadd_epi32(total, total);
            total = _mm_hadd_epi32(total, total);
            sum = _mm_cvtsi128_si32(total);
        } else
#endif
        {
            int8_t t[QK8_0];
            for (int s = 0; s < 4; ++s) {
                for (int b = 0; b < 8; ++b) {
                    const int8_t q = y[i].qs[4 * b + s];
                    t[8 * s + b] = q;
                    sum += q;
                }
            }
            memcpy(y[i].qs, t, QK8_0);
        }
        yd[i] = ggml_fp16_to_fp32(y[i].d);
        if (i & 1) { yc[i / 2] += yd[i] * (float) sum; } else { yc[i / 2] = yd[i] * (float) sum; }
    }
}

template<int NT>
void dot_q2_0_rows(int n, int nrows, const char * w, size_t row_bytes, const block_q8_0 * y, size_t y_stride,
        const float * yd, const float * yc, size_t aux_stride, float * out, size_t out_stride, int ahead) {
    const int nb = n / QK2_0;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    const __m256i shifts = _mm256_set_epi64x(6, 4, 2, 0);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    constexpr int rows = 4 / NT;
    for (int r = 0; r < nrows; r += rows) {
        const int nr = std::min(rows, nrows - r);
        __m256 acc[NT][rows];
        float corr[NT][rows] = {};
        for (int t = 0; t < NT; ++t) {
            for (int k = 0; k < rows; ++k) { acc[t][k] = _mm256_setzero_ps(); }
        }
        for (int i = 0; i < nb; ++i) {
            __m256i q0[NT], q1[NT];
            __m256 s0[NT], s1[NT];
            for (int t = 0; t < NT; ++t) {
                const auto * yt = (const block_q8_0 *) ((const char *) y + (size_t) t * y_stride);
                q0[t] = _mm256_loadu_si256((const __m256i *) yt[2 * i].qs);
                q1[t] = _mm256_loadu_si256((const __m256i *) yt[2 * i + 1].qs);
                s0[t] = _mm256_set1_ps(yd[(size_t) t * aux_stride + 2 * i]);
                s1[t] = _mm256_set1_ps(yd[(size_t) t * aux_stride + 2 * i + 1]);
            }
            for (int k = 0; k < nr; ++k) {
                const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) (r + k) * row_bytes) + i;
                if (ahead > 0 && i + ahead < nb && (i & 3) == 0) {
                    _mm_prefetch((const char *) (x + ahead), _MM_HINT_T0);
                    _mm_prefetch((const char *) (x + ahead + 1) - 1, _MM_HINT_T0);
                }
                const float d = _cvtsh_ss(x->d);
                long long c0, c1;
                memcpy(&c0, x->qs, 8);
                memcpy(&c1, x->qs + 8, 8);
                const __m256i w0 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(c0), shifts), m3);
                const __m256i w1 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(c1), shifts), m3);
                for (int t = 0; t < NT; ++t) {
                    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, q0[t]), ones16);
                    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w1, q1[t]), ones16);
                    const __m256 f = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), s0[t], _mm256_mul_ps(_mm256_cvtepi32_ps(p1), s1[t]));
                    acc[t][k] = _mm256_fmadd_ps(_mm256_set1_ps(d), f, acc[t][k]);
                    corr[t][k] += d * yc[(size_t) t * aux_stride + i];
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            for (int k = 0; k < nr; ++k) { out[(size_t) t * out_stride + r + k] = hsum8(acc[t][k]) - corr[t][k]; }
        }
    }
#else
    (void) ahead;
    for (int t = 0; t < NT; ++t) {
        const auto * yt = (const block_q8_0 *) ((const char *) y + (size_t) t * y_stride);
        for (int r = 0; r < nrows; ++r) {
            const block_q2_0 * x = (const block_q2_0 *) (w + (size_t) r * row_bytes);
            float sum = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = ggml_fp16_to_fp32(x[i].d);
                for (int k = 0; k < 2; ++k) {
                    const block_q8_0 & yb = yt[2 * i + k];
                    int isum = 0;
                    for (int s = 0; s < 4; ++s) {
                        for (int b = 0; b < 8; ++b) { isum += ((x[i].qs[8 * k + b] >> (2 * s)) & 3) * yb.qs[8 * s + b]; }
                    }
                    sum += d * yd[(size_t) t * aux_stride + 2 * i + k] * (float) isum;
                }
                sum -= d * yc[(size_t) t * aux_stride + i];
            }
            out[(size_t) t * out_stride + r] = sum;
        }
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
    bool ai = false;            // fork AIPFIX D2: AI+ unit expert (moe_ai::slab_rows on rotated f32 inputs)
    bool ai_q8 = false;         // AIPFIX: int8 activations (moe_ai::slab_rows_q8; default on, GGML_MOE_AI_HOST_Q8=0 = off)
    int ai_k = 0;
    uint32_t ai_C = 0;
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

// Scratch rows that several workers write side by side. A std::vector base is only 16-byte aligned, so with the
// 64-byte row strides below every tile boundary would still split a cache line between two workers (the line then
// bounces between their cores, across both CCDs). `align` (GGML_MOE_CPU_SPLIT_ALIGN) starts the data on a line.
template <typename T>
struct scratch_buf {
    std::vector<T> v;
    T * p = nullptr;
    void assign(size_t n, T val, bool align) {
        v.assign(n + (align ? 64 / sizeof(T) : 0), val);
        p = v.data();
        if (align) { while ((uintptr_t) p % 64) { ++p; } }
    }
    T * data() { return p; }
    const T * data() const { return p; }
    T & operator[](size_t i) { return p[i]; }
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
    bool down_reuse = false;
    int group_tokens = 4;
    scratch_buf<uint8_t> xq;    // [2][MOE_CS_MAX_ROUTES] quantized input rows
    size_t xq_stride = 0;
    scratch_buf<float> hidden;  // [MOE_CS_MAX_ROUTES][max_ff]
    scratch_buf<uint8_t> hq;    // [MOE_CS_MAX_ROUTES] quantized hidden rows
    size_t hq_stride = 0;
    bool avx512 = false;       // GGML_MOE_CPU_SPLIT_AVX512 and the CPU has AVX-512 F/BW/VL/VNNI
    bool q2_pair = false;      // Share Q2 down weights between two routes of one expert.
    bool q2_group4 = false;    // Share Q2 down weights between up to four routes of one expert.
    bool vector_silu = false;  // Use the CPU backend's vector activation in grouped up/gate tiles.
    bool q2_fast = false;      // GGML_MOE_CPU_SPLIT_Q2_FAST: reordered hidden rows + dot_q2_0_rows for Q2_0 down
    bool q2_shuffle = false;   // Use AVX2 to reorder hidden bytes, with the same integer sums.
    int q2_pf = 0;             // Q2 down prefetch distance in weight blocks; zero disables it.
    size_t aux_stride = 0;     // floats per route in hq_aux: n/32 block scales, then n/64 offsets
    scratch_buf<float> hq_aux;
    int rows_ug = 16, rows_down = 64;
    bool iq_rows = false;      // GGML_MOE_CPU_SPLIT_ROWS: moe-iq-rows.h kernels, each chunk decoded once per expert
    int iq_variant = -1;       // moe_iq::get variant bits (GGML_MOE_CPU_SPLIT_ROWS: 1 = best mix, 2..17 = bits + 2)
    int iq_pf = 0;             // GGML_MOE_CPU_SPLIT_PF: prefetch distance of those kernels in bytes (0 = off)
    bool schedule = false;    // Assign quantization rows directly and claim math tiles in small groups.
    int waitpf = 0;           // GGML_MOE_CPU_SPLIT_WAITPF: KiB of its first tile a worker prefetches while it waits at a phase barrier (0 = off)
    bool host_help = false;
    bool item_barrier = false; // GGML_MOE_CPU_SPLIT_ITEM_BARRIER: phases end when their items are done, not their threads
    bool own_xq = false;       // GGML_MOE_CPU_SPLIT_OWN_XQ (with ITEM_BARRIER): every thread quantizes the inputs itself
    std::vector<uint8_t> xq_own;   // [threads][2][MOE_CS_MAX_ROUTES] private copies of xq
    bool flow = false;         // GGML_MOE_CPU_SPLIT_FLOW (with ITEM_BARRIER): per-expert dependencies replace barriers 1 and 2
    std::atomic<int> ug_left[MOE_CS_MAX_EXPERTS];   // flow: up + gate tiles of expert i not yet finished
    std::atomic<int> hq_left[MOE_CS_MAX_EXPERTS];   // flow: hidden rows of expert i not yet quantized
    int expert_of[MOE_CS_MAX_ROUTES] = {};          // flow: expert index of each route
    size_t xq_own_stride = 0;
    bool expjob = false;      // GGML_MOE_CPU_SPLIT_EXPJOB=1: one worker runs a whole expert (up/gate, SiLU, down) per claim, no phase barriers inside a job
    int host_cpu = -1;
    std::atomic<int> help_phase{-1}; // -1 closed, 1/3 open, 5/7 held by a helper
    std::atomic<unsigned long long> closed[4];

    alignas(64) std::atomic<unsigned long long> gen{0};
    alignas(64) std::atomic<int> next[4];
    alignas(64) std::atomic<unsigned long long> fin[4];
    // item_barrier: (job number << 32) | next item / items finished, per phase
    alignas(64) std::atomic<unsigned long long> inext[4];
    alignas(64) std::atomic<unsigned long long> idone[4];
    alignas(64) std::atomic<int> sleepers{0};
    std::mutex sleep_mu;
    std::condition_variable sleep_cv;

    std::atomic<unsigned long long> st_jobs{0}, st_experts{0}, st_routes{0}, st_busy_ns{0}, st_null{0};
    // AIPFAST R8 host cache (GGML_MOE_AI_HOST_R8_MB=<MiB>, with GGML_MOE_AI_HOST_Q8=1): int8 level copies of recently
    // CPU-computed AI+ projections, LRU, keyed by (layer, expert, role). A miss decodes and keeps (moe_ai::slab_rows_q8_keep),
    // a hit runs moe_ai::slab_rows_r8; both are bit-identical to slab_rows_q8. Thread 0 plans each card job before `gen`
    // moves; a slot filled by job n turns ready when job n + 1 is planned (job n has finished by then). The load probe
    // posts host jobs, which never use the cache, so the planner's per-route cost stays the decode cost.
    struct r8_state {
        uint8_t * arena = nullptr;
        size_t arena_bytes = 0;                  // mmap length of arena
        size_t slot = 0;
        int n = 0, n_expert = 0, head = -1, tail = -1;
        std::vector<int32_t> map, key, prv, nxt, pending;   // map: (layer*n_expert + e)*3 + role -> slot or -1
        std::vector<uint8_t> ready;
        unsigned long long hit = 0, fill = 0, plain = 0, jobs = 0;
    } r8;
    int r8_mb = 0;
    int r8_every = 16384;                        // card jobs between hit-rate lines (GGML_MOE_AI_HOST_R8_STATS)
    bool r8_reset = false;                       // a layer was (re)registered: rebuild the table at the next card job
    uint8_t * r8p[MOE_CS_MAX_EXPERTS][3] = {};   // this job: slot of (expert i, role)
    uint8_t r8m[MOE_CS_MAX_EXPERTS][3] = {};     // this job: 0 = decode, 1 = hit (R8 kernel), 2 = decode + keep
    unsigned long long phase_ns[4] = {};   // thread 0's time through each phase (barrier included), summed
    // GGML_MOE_CPU_SPLIT_WSTATS=1 (default off): per worker, time spent claiming and running tiles and time spent waiting at
    // the end-of-phase barrier, plus how late the worker joined a job. Tells a slow or late core from a slow kernel.
    struct alignas(64) wstat_t { unsigned long long jobs = 0, lag_ns = 0, work_ns[4] = {}, wait_ns[4] = {}; int cpu = -1; };
    bool wstats = false;
    wstat_t wst[64];
    std::atomic<long long> job_t0_ns{0};   // leader's clock when the job was published
    // host-posted jobs (the probe) use this instead of the mapped seq
    std::atomic<unsigned long long> local_seq{0}, local_done{0};
};

pool_state & P() {
    static pool_state * p = new pool_state();   // never destroyed: threads may still spin at exit
    return *p;
}

// Clears the per-worker counters (the probe's jobs are not production work).
void wst_clear(pool_state & s) {
    for (auto & w : s.wst) {
        const int cpu = w.cpu;
        w = pool_state::wstat_t();
        w.cpu = cpu;
    }
}

inline long long ns_of(clk::time_point t) {
    return (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
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
    const void * base = moe_cs_load_acquire(&m.r.src[expert]);
    if (!base) { return nullptr; }
    return (const char *) base + (size_t) row * m.r.row_bytes;
    (void) s;
}

static void r8_touch(pool_state::r8_state & c, int sl) {   // move slot sl to the front (most recent) of the LRU list
    if (c.head == sl) { return; }
    const int p = c.prv[sl], q = c.nxt[sl];
    if (p >= 0) { c.nxt[p] = q; }
    if (q >= 0) { c.prv[q] = p; }
    if (c.tail == sl) { c.tail = p; }
    c.prv[sl] = -1; c.nxt[sl] = c.head;
    if (c.head >= 0) { c.prv[c.head] = sl; }
    c.head = sl;
    if (c.tail < 0) { c.tail = sl; }
}

static double r8_mem_avail() {   // MemAvailable in bytes (0 if unreadable)
    long long kib = -1;
    if (FILE * fp = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) { if (sscanf(line, "MemAvailable: %lld kB", &kib) == 1) { break; } }
        fclose(fp);
    }
    return kib > 0 ? kib * 1024.0 : 0.0;
}

static bool r8_init(pool_state & s) {
#if !defined(__linux__)
    s.r8_reset = false;
    fprintf(stderr, "[moe-cpu-split] r8 host cache: requires Linux, off\n");
    return false;
#else
    pool_state::r8_state & c = s.r8;
    s.r8_reset = false;
    size_t slot = 0;
    int ne = 0;
    for (const layer_math & L : s.layers) {
        for (int k = 0; L.ok && k < 3; ++k) {
            const role_math & m = L.role[k];
            if (!m.ai_q8 || !moe_ai::r8_ok(m.ai_k, m.ai_C, m.r.n_out, m.r.n_in)) { continue; }
            slot = std::max(slot, moe_ai::r8_slot_bytes(m.r.n_out, m.r.n_in));
            ne = std::max(ne, (int) m.r.n_expert);
        }
    }
    if (!slot) { fprintf(stderr, "[moe-cpu-split] r8 host cache: no AI+ host roles, off\n"); return false; }
    // RAM-aware fit: the cache is sized after load against MemAvailable minus the 5 GiB floor
    // (GGML_MOE_AI_HOST_R8_FLOOR_GIB raises it; R8_MB=-1 = auto, GGML_MOE_AI_HOST_R8_PCT % of the room, default 50),
    // and its pages are committed now (MAP_POPULATE; GGML_MOE_AI_HOST_R8_POPULATE=0 keeps them lazy) so the floor holds.
    if (c.arena) { munmap(c.arena, c.arena_bytes); c.arena = nullptr; c.arena_bytes = 0; }
    const double avail0 = r8_mem_avail();
    const char * fe = getenv("GGML_MOE_AI_HOST_R8_FLOOR_GIB");
    const double floor_gib = fe && fe[0] ? atof(fe) : 5.0;
    const size_t fit = moe_ai::r8_fit_bytes(s.r8_mb, avail0, floor_gib, env_int("GGML_MOE_AI_HOST_R8_PCT", 50, 0, 100));
    const size_t n = std::min<size_t>(fit / slot, 1u << 24);
    if (n < 3 * MOE_CS_MAX_EXPERTS + 1) {   // a job must never evict a slot it uses itself
        fprintf(stderr, "[moe-cpu-split] r8 host cache: %d MiB asked, %zu MiB fits above the %.1f GiB floor (MemAvailable %.2f GiB): "
                "%zu slots, need %d, off\n", s.r8_mb, fit >> 20, std::max(5.0, floor_gib), avail0 / 1073741824.0, n, 3 * MOE_CS_MAX_EXPERTS + 1);
        return false;
    }
    const bool populate = env_int("GGML_MOE_AI_HOST_R8_POPULATE", 1, 0, 1) != 0;
    const auto t0 = std::chrono::steady_clock::now();
    void * mem = mmap(nullptr, n * slot, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (populate ? MAP_POPULATE : 0), -1, 0);
    if (mem == MAP_FAILED) { fprintf(stderr, "[moe-cpu-split] r8 host cache: map of %zu MiB failed, off\n", (n * slot) >> 20); return false; }
    c.arena = (uint8_t *) mem; c.arena_bytes = n * slot;
    fprintf(stderr, "[moe-cpu-split] r8 host cache: RAM fit %d MiB asked, %zu MiB used, MemAvailable %.2f -> %.2f GiB (floor %.1f), "
            "%s in %.0f ms\n", s.r8_mb, (n * slot) >> 20, avail0 / 1073741824.0, r8_mem_avail() / 1073741824.0,
            std::max(5.0, floor_gib), populate ? "populated" : "lazy",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    c.slot = slot; c.n = (int) n; c.n_expert = ne;
    c.map.assign(s.layers.size() * (size_t) ne * 3, -1);
    c.key.assign(n, -1); c.ready.assign(n, 0); c.prv.resize(n); c.nxt.resize(n); c.pending.clear();
    for (int i = 0; i < c.n; ++i) { c.prv[i] = i - 1; c.nxt[i] = i + 1 < c.n ? i + 1 : -1; }
    c.head = 0; c.tail = c.n - 1;
    fprintf(stderr, "[moe-cpu-split] r8 host cache: %d slots x %.2f MiB = %zu MiB (GGML_MOE_AI_HOST_R8_MB, pages fill on first use)\n",
            c.n, slot / 1048576.0, (n * slot) >> 20);
    return true;
#endif
}

// thread 0, before `gen` moves: choose decode / hit / decode + keep for every AI+ q8 (expert, role) of this job
static void r8_plan(pool_state & s, int layer, bool card) {
    for (int i = 0; i < s.n_exp; ++i) { s.r8m[i][0] = s.r8m[i][1] = s.r8m[i][2] = 0; }
    if (!card || s.r8_mb == 0) { return; }
    pool_state::r8_state & c = s.r8;
    if ((c.n == 0 || s.r8_reset) && !r8_init(s)) { s.r8_mb = 0; return; }
    for (const int sl : c.pending) { c.ready[sl] = 1; }
    c.pending.clear();
    for (int i = 0; i < s.n_exp; ++i) {
        for (int k = 0; k < 3; ++k) {
            const role_math & m = s.cur->role[k];
            if (!m.ai_q8) { continue; }
            if (!moe_ai::r8_ok(m.ai_k, m.ai_C, m.r.n_out, m.r.n_in) || !moe_cs_load_acquire(&m.r.src[s.expert[i]])) { ++c.plain; continue; }
            const int32_t idx = (int32_t) (((size_t) layer * c.n_expert + s.expert[i]) * 3 + k);
            int sl = c.map[idx];
            if (sl >= 0 && c.ready[sl]) {
                s.r8m[i][k] = 1; ++c.hit;
            } else {
                if (sl < 0) {
                    sl = c.tail;
                    if (c.key[sl] >= 0) { c.map[c.key[sl]] = -1; }
                    c.key[sl] = idx; c.map[idx] = sl;
                }
                c.ready[sl] = 0; c.pending.push_back(sl);
                s.r8m[i][k] = 2; ++c.fill;
            }
            r8_touch(c, sl);
            s.r8p[i][k] = c.arena + (size_t) sl * c.slot;
        }
    }
    if (++c.jobs % (unsigned long long) s.r8_every == 0) {
        const unsigned long long tot = c.hit + c.fill + c.plain;
        fprintf(stderr, "[moe-cpu-split] r8 host cache: %llu card jobs, AI+ host projections hit %llu fill %llu plain %llu (hit %.1f%%)\n",
                c.jobs, c.hit, c.fill, c.plain, tot ? 100.0 * c.hit / tot : 0.0);
    }
}

// one 16-row slab of an AI+ q8 projection by the job's plan (rows [a, a + 16), needs b == a + 16 for modes 1/2)
static inline void r8_proj(const pool_state & s, const role_math & m, int i, int k, const char * w, int64_t a, int64_t b, int nt,
                           const void * const * q, float * const * o) {
    switch (b - a == AI4_TR ? s.r8m[i][k] : 0) {
        case 1:  moe_ai::slab_rows_r8(s.r8p[i][k], a, m.r.n_out, m.r.n_in, nt, q, o); return;
        case 2:  moe_ai::slab_rows_q8_keep(w, m.r.row_bytes, m.ai_k, m.ai_C, a, m.r.n_out, m.r.n_in, nt, q, o, s.r8p[i][k]); return;
        default: moe_ai::slab_rows_q8(w, m.r.row_bytes, m.ai_k, m.ai_C, a, b, m.r.n_in, nt, q, o);
    }
}

void run_item(pool_state & s, int phase, int item, uint8_t * xq_mine = nullptr) {
    const layer_math & L = *s.cur;
    uint8_t * const xq = xq_mine ? xq_mine : s.xq.data();
    if (phase == 0) {
        const int nr = s.n_input;
        const int k = item / nr, q = s.inputs[item % nr];
        const role_math & m = L.role[k];
        m.quant(s.x + (size_t) q * s.x_stride, xq + ((size_t) k * MOE_CS_MAX_ROUTES + q) * s.xq_stride, m.r.n_in);
        return;
    }
    if (phase == 2) {
        const role_math & m = L.role[2];
        m.quant(s.hidden.data() + (size_t) item * s.max_ff, s.hq.data() + (size_t) item * s.hq_stride, m.r.n_in);
        if (s.q2_fast && m.q2) {
            float * aux = s.hq_aux.data() + (size_t) item * s.aux_stride;
            q2_reorder_hidden((int) m.r.n_in, (block_q8_0 *) (s.hq.data() + (size_t) item * s.hq_stride), aux,
                    aux + m.r.n_in / QK8_0, s.q2_shuffle);
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
        if (up.ai) {
            // fork AIPFIX D2: 16-row AI+ slabs, up to 8 tokens per slab decode
            moe_ai_log_first("gate/up");
            const char * wu = role_row(s, up, e, 0);
            const char * wg = role_row(s, gate, e, 0);
            for (int q0 = s.first[i]; q0 < s.first[i + 1]; q0 += 8) {
                const int nt = std::min(8, s.first[i + 1] - q0);
                const float * yu[8];
                const float * yg[8];
                const void * qu[8];
                const void * qg[8];
                for (int t = 0; t < nt; ++t) {
                    const int q = s.input_of[q0 + t];
                    qu[t] = xq + (size_t) q * s.xq_stride;
                    qg[t] = xq + (xg + q) * s.xq_stride;
                    yu[t] = (const float *) qu[t];
                    yg[t] = (const float *) qg[t];
                }
                for (int64_t a = r0; a < r1; a += AI4_TR) {
                    const int64_t b = std::min<int64_t>(a + AI4_TR, r1);
                    float ub[8][AI4_TR] = {}, gb[8][AI4_TR] = {};
                    float * uo[8], * go[8];
                    for (int t = 0; t < nt; ++t) { uo[t] = ub[t]; go[t] = gb[t]; }
                    if (wu && wg) { GGML_ASSERT(gate.ai_q8 == up.ai_q8); } // gate reads the up activation format
                    if (wu && wg && up.ai_q8) {
                        r8_proj(s, up, i, 0, wu, a, b, nt, qu, uo);
                        r8_proj(s, gate, i, 1, wg, a, b, nt, qg, go);
                    } else if (wu && wg) {
                        moe_ai::slab_rows(wu, up.r.row_bytes, up.ai_k, up.ai_C, a, b, up.r.n_in, nt, yu, uo);
                        moe_ai::slab_rows(wg, gate.r.row_bytes, gate.ai_k, gate.ai_C, a, b, gate.r.n_in, nt, yg, go);
                    }
                    for (int t = 0; t < nt; ++t) {
                        float * h = s.hidden.data() + (size_t) (q0 + t) * s.max_ff + a;
                        for (int64_t row = a; row < b; ++row) {
                            const float u = ub[t][row - a], g = gb[t][row - a];
                            h[row - a] = g / (1.0f + expf(-g)) * u;
                        }
                    }
                }
            }
            if (!wu || !wg) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
            return;
        }
#if defined(MOE_IQ_ROWS)
        if (up.rows[4] && gate.rows[4]) {
            // Decode once for each token group; AVX-512 uses a different valid float reduction order.
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
                    yu[t] = (const block_q8_K *) (xq + (size_t) q * s.xq_stride);
                    yg[t] = (const block_q8_K *) (xq + (xg + q) * s.xq_stride);
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
                        float * h = s.hidden.data() + (size_t) (q0 + t) * s.max_ff + a;
                        if (s.vector_silu) {
                            // fork: same SiLU function as the CPU backend, with its vector rounding.
                            ggml_vec_silu_f32((int) (b - a), h, gb[t]);
                            for (int k = 0; k < b - a; ++k) { h[k] *= ub[t][k]; }
                            continue;
                        }
                        for (int64_t row = a; row < b; ++row) {
                            const float u = ub[t][row - a], g = gb[t][row - a];
                            h[row - a] = g / (1.0f + expf(-g)) * u;
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
                    const uint8_t * xu = xq + (size_t) input * s.xq_stride;
                    const uint8_t * xgp = xq + (xg + input) * s.xq_stride;
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
    if (down.ai) {
        // fork AIPFIX D2: 16-row AI+ slabs of the rotated hidden rows
        moe_ai_log_first("down");
        const char * w = role_row(s, down, e, 0);
        for (int q0 = s.first[i]; q0 < s.first[i + 1]; q0 += 8) {
            const int nt = std::min(8, s.first[i + 1] - q0);
            const float * y[8];
            const void * yq[8];
            float * o[8];
            for (int t = 0; t < nt; ++t) {
                yq[t] = s.hq.data() + (size_t) (q0 + t) * s.hq_stride;
                y[t] = (const float *) yq[t];
                o[t] = s.out + (size_t) (q0 + t) * s.out_stride;
            }
            for (int64_t a = r0; a < r1; a += AI4_TR) {
                const int64_t b = std::min<int64_t>(a + AI4_TR, r1);
                float * oa[8];
                for (int t = 0; t < nt; ++t) { oa[t] = o[t] + a; }
                if (w && down.ai_q8) {
                    r8_proj(s, down, i, 2, w, a, b, nt, yq, oa);
                } else if (w) {
                    moe_ai::slab_rows(w, down.r.row_bytes, down.ai_k, down.ai_C, a, b, down.r.n_in, nt, y, oa);
                } else {
                    for (int t = 0; t < nt; ++t) { for (int64_t row = a; row < b; ++row) { oa[t][row - a] = 0.0f; } }
                }
            }
        }
        if (!w) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
        return;
    }
    if (s.q2_fast && down.q2) {
        const char * w = role_row(s, down, e, r0);
        for (int q = s.first[i]; q < s.first[i + 1];) {
            float * o = s.out + (size_t) q * s.out_stride + r0;
            if (!w) { for (int64_t row = r0; row < r1; ++row) { o[row - r0] = 0.0f; } ++q; continue; }
            const float * aux = s.hq_aux.data() + (size_t) q * s.aux_stride;
            const auto * y = (const block_q8_0 *) (s.hq.data() + (size_t) q * s.hq_stride);
            if (s.q2_group4 && q + 3 < s.first[i + 1]) {
                dot_q2_0_rows<4>((int) down.r.n_in, (int) (r1 - r0), w, down.r.row_bytes, y, s.hq_stride,
                        aux, aux + down.r.n_in / QK8_0, s.aux_stride, o, s.out_stride, s.q2_pf);
                q += 4;
            } else if (s.q2_group4 && q + 2 < s.first[i + 1]) {
                dot_q2_0_rows<3>((int) down.r.n_in, (int) (r1 - r0), w, down.r.row_bytes, y, s.hq_stride,
                        aux, aux + down.r.n_in / QK8_0, s.aux_stride, o, s.out_stride, s.q2_pf);
                q += 3;
            } else if (s.q2_pair && q + 1 < s.first[i + 1]) {
                dot_q2_0_rows<2>((int) down.r.n_in, (int) (r1 - r0), w, down.r.row_bytes, y, s.hq_stride,
                        aux, aux + down.r.n_in / QK8_0, s.aux_stride, o, s.out_stride, s.q2_pf);
                q += 2;
            } else {
                dot_q2_0_rows<1>((int) down.r.n_in, (int) (r1 - r0), w, down.r.row_bytes, y, s.hq_stride,
                        aux, aux + down.r.n_in / QK8_0, s.aux_stride, o, s.out_stride, s.q2_pf);
                ++q;
            }
        }
        if (!w) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
        return;
    }
#if defined(MOE_IQ_ROWS)
    if (s.down_reuse && down.rows[4]) {
        const char * w = role_row(s, down, e, 0);
        for (int q0 = s.first[i]; q0 < s.first[i + 1]; q0 += s.group_tokens) {
            const int nt = std::min(s.group_tokens, s.first[i + 1] - q0);
            const block_q8_K * y[8];
            float * o[8];
            for (int t = 0; t < nt; ++t) {
                y[t] = (const block_q8_K *) (s.hq.data() + (size_t) (q0 + t) * s.hq_stride);
                o[t] = s.out + (size_t) (q0 + t) * s.out_stride + r0;
            }
            if (w && down.rows[nt]) { down.rows[nt](w, down.r.row_bytes, r0, r1, (int) (down.r.n_in / QK_K), y, o, s.iq_pf); }
            else {
                for (int t = 0; t < nt; ++t) {
                    for (int64_t row = r0; row < r1; ++row) {
                        o[t][row - r0] = 0.0f;
                        if (w) { down.dot((int) down.r.n_in, &o[t][row - r0], 0, w + (size_t) row * down.r.row_bytes, 0, y[t], 0, 1); }
                    }
                }
            }
        }
        if (!w) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
        return;
    }
#endif
    if (s.down_reuse && down.r.type == GGML_TYPE_IQ4_NL) {
        for (int q0 = s.first[i]; q0 < s.first[i + 1]; q0 += s.group_tokens) {
            const int nt = std::min(s.group_tokens, s.first[i + 1] - q0);
            const void * y[8];
            for (int t = 0; t < nt; ++t) { y[t] = s.hq.data() + (size_t) (q0 + t) * s.hq_stride; }
            for (int64_t row = r0; row < r1; ++row) {
                const char * w = role_row(s, down, e, row);
                float v[8] = {};
                if (w && !ggml_vec_dot_iq_grouped(GGML_TYPE_IQ4_NL, (int) down.r.n_in, w, y, v, nt)) {
                    for (int t = 0; t < nt; ++t) { down.dot((int) down.r.n_in, &v[t], 0, w, 0, y[t], 0, 1); }
                }
                for (int t = 0; t < nt; ++t) { s.out[(size_t) (q0 + t) * s.out_stride + row] = v[t]; }
                if (!w) { s.st_null.fetch_add(1, std::memory_order_relaxed); }
            }
        }
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

// waitpf: with the first tile of the up/gate and down phases handed out by worker number, a worker that finishes
// early at the previous barrier already knows which weight rows it reads next and pulls them toward its cache
// while the slower workers finish, instead of spinning on a pause instruction.
struct pf_cursor {
    const char * base[2] = {nullptr, nullptr};
    size_t len = 0, at = 0;
};

pf_cursor pf_tile(pool_state & s, int phase, int item) {
    pf_cursor c;
    if (!s.waitpf || (phase != 1 && phase != 3) || item >= items_of(s, phase)) { return c; }
    const layer_math & L = *s.cur;
    const bool ug = phase == 1;
    const role_math & m = L.role[ug ? 0 : 2];
    const int rows = ug ? s.rows_ug : s.rows_down;
    const int blocks = (int) ((m.r.n_out + rows - 1) / rows);
    const int64_t r0 = (int64_t) (item % blocks) * rows;
    const int e = s.expert[item / blocks];
    c.len = std::min<size_t>((size_t) std::min<int64_t>(rows, m.r.n_out - r0) * m.r.row_bytes, (size_t) s.waitpf * 1024);
    c.base[0] = role_row(s, m, e, r0);
    if (ug) { c.base[1] = role_row(s, L.role[1], e, r0); }
    return c;
}

// Issues up to four lines of the tile (alternating up and gate); false once the tile is covered.
bool pf_step(pf_cursor & c) {
    for (int k = 0; k < 4 && c.at < c.len; ++k) {
        for (int j = 0; j < 2; ++j) { if (c.base[j]) { moe_cs_prefetch(c.base[j] + c.at); } }
        c.at += 64;
    }
    return c.at < c.len;
}

// Every thread runs all four phases of job number `n` (1-based); the fin counters only grow, so a thread that is
// late to notice a finished phase never misses it.
void run_job_phases(pool_state & s, unsigned long long n, int worker, bool lead = false) {
    const unsigned long long target = n * (unsigned long long) s.threads;
    auto t = clk::now();
    pool_state::wstat_t * W = s.wstats && worker < 64 ? &s.wst[worker] : nullptr;
    if (W) {
        ++W->jobs;
        if (!lead) { W->lag_ns += (unsigned long long) std::max<long long>(0, ns_of(t) - s.job_t0_ns.load(std::memory_order_relaxed)); }
    }
    auto tw = t;
    pf_cursor pf;
    for (int p = 0; p < 4; ++p) {
        const int total = items_of(s, p);
        if (s.expjob && p >= 1) {
            // Fused phase: claim whole experts. Each expert's tiles are the same items the phased path runs, in the same
            // per-row order, so the output is unchanged; phases 2 and 3 only pass the (already satisfied) barriers.
            if (p == 1) {
                for (int i = s.next[1].fetch_add(1, std::memory_order_relaxed); i < s.n_exp;
                        i = s.next[1].fetch_add(1, std::memory_order_relaxed)) {
                    const int bu = total / s.n_exp;
                    for (int b = 0; b < bu; ++b) { run_item(s, 1, i * bu + b); }
                    for (int q = s.first[i]; q < s.first[i + 1]; ++q) { run_item(s, 2, q); }
                    const int bd = items_of(s, 3) / s.n_exp;
                    for (int b = 0; b < bd; ++b) { run_item(s, 3, i * bd + b); }
                }
            }
        } else if (s.schedule && (p == 0 || p == 2)) {
            // Each row has one writer. The phase barrier publishes it before any consumer runs.
            for (int it = worker; it < total; it += s.threads) {
                run_item(s, p, it);
            }
        } else {
            if (s.waitpf && (p == 1 || p == 3) && worker < total) { run_item(s, p, worker); }
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
        if (W) {
            const auto t1 = clk::now();
            W->work_ns[p] += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - tw).count();
            tw = t1;
        }
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
            if (s.waitpf && p < 2) { pf = pf_tile(s, p == 0 ? 1 : 3, worker); }   // phase 2 keeps the down-tile cursor
            while (s.fin[p].load(std::memory_order_acquire) < target) {
                if (!(s.waitpf && pf_step(pf))) { cpu_relax(); }
            }
        }
        if (lead && p == 3) {
            while (s.fin[3].load(std::memory_order_acquire) < target) { cpu_relax(); }
        }
        if (W) {
            const auto t2 = clk::now();
            W->wait_ns[p] += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - tw).count();
            tw = t2;
        }
        if (lead) {
            const auto now = clk::now();
            s.phase_ns[p] += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
            t = now;
        }
    }
}

// item_barrier: the same items as run_job_phases, but a phase ends when all of its items are finished instead of
// when every thread has passed it, so a pool thread the OS has descheduled while it holds no item (a pool core is
// shared with the desktop and the server's own threads) no longer stalls the job at four barriers. Claims carry the
// job number: a thread that wakes up late in an old job can never claim an item of a newer one. Every item still
// writes its own rows the same way, so the results are bit-identical.
// own_xq: the input rows are a few KB, so instead of sharing phase 0 every thread quantizes all of them into its own
// copy (same function, same bytes) and goes straight to up + gate: no thread waits on another before its first tile.
constexpr unsigned long long item_low = 0xffffffffULL;

// flow: phases 1-3 become one list of items in the same order (all up + gate tiles, then the hidden rows, then the
// down tiles, each expert-major) claimed from inext[1] and counted in idone[1]. Instead of waiting for the whole
// phase before it, a hidden row waits only for its own expert's up + gate tiles and a down tile only for its own
// expert's hidden rows. Those items sit earlier in the list, so they are already claimed and running: nobody waits
// on unclaimed work, and threads that finish early go on with other experts. Same items, same rows: bit-identical.
void run_flow_items(pool_state & s, unsigned long long tag, uint8_t * mine, bool lead, clk::time_point t) {
    const unsigned long long n1 = (unsigned long long) items_of(s, 1), n2 = (unsigned long long) items_of(s, 2);
    const unsigned long long total = n1 + n2 + (unsigned long long) items_of(s, 3);
    const int ug_blocks = (int) (n1 / (unsigned long long) s.n_exp);
    const int down_blocks = (int) ((total - n1 - n2) / (unsigned long long) s.n_exp);
    for (unsigned long long c = s.inext[1].load(std::memory_order_acquire);
            (c & ~item_low) == tag && (c & item_low) < total;) {
        if (!s.inext[1].compare_exchange_weak(c, c + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue;
        }
        const unsigned long long k = c & item_low;
        if (k < n1) {
            const int i = (int) k / ug_blocks;
            run_item(s, 1, (int) k, mine);
            s.ug_left[i].fetch_sub(1, std::memory_order_acq_rel);
        } else if (k < n1 + n2) {
            const int q = (int) (k - n1), i = s.expert_of[q];
            while (s.ug_left[i].load(std::memory_order_acquire) > 0) { cpu_relax(); }
            run_item(s, 2, q);
            s.hq_left[i].fetch_sub(1, std::memory_order_acq_rel);
        } else {
            const int item = (int) (k - n1 - n2), i = item / down_blocks;
            while (s.hq_left[i].load(std::memory_order_acquire) > 0) { cpu_relax(); }
            run_item(s, 3, item);
        }
        s.idone[1].fetch_add(1, std::memory_order_acq_rel);
        c = s.inext[1].load(std::memory_order_acquire);
    }
    if (!lead) { return; }   // thread 0 alone waits for the last items and publishes `done`
    while ((s.idone[1].load(std::memory_order_acquire) & item_low) < total) { cpu_relax(); }
    s.phase_ns[1] += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count();
}

void run_job_items(pool_state & s, unsigned long long n, int worker, bool lead = false) {
    const unsigned long long tag = (n & item_low) << 32;
    uint8_t * const mine = s.own_xq ? s.xq_own.data() + (size_t) worker * s.xq_own_stride : nullptr;
    auto t = clk::now();
    for (int p = 0; p < 4; ++p) {
        if (p == 1 && s.flow) { run_flow_items(s, tag, mine, lead, t); return; }
        const unsigned long long total = (unsigned long long) items_of(s, p);
        if (p == 0 && mine) {
            for (unsigned long long it = 0; it < total; ++it) {
                if ((s.inext[1].load(std::memory_order_acquire) & ~item_low) != tag) { return; }   // job already over
                run_item(s, 0, (int) it, mine);
            }
            if (lead) {
                const auto now = clk::now();
                s.phase_ns[0] += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
                t = now;
            }
            continue;
        }
        for (unsigned long long c = s.inext[p].load(std::memory_order_acquire);
                (c & ~item_low) == tag && (c & item_low) < total;) {
            if (!s.inext[p].compare_exchange_weak(c, c + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                continue;
            }
            run_item(s, p, (int) (c & item_low), p == 1 ? mine : nullptr);
            s.idone[p].fetch_add(1, std::memory_order_acq_rel);
            c = s.inext[p].load(std::memory_order_acquire);
        }
        if (p == 3 && !lead) { return; }   // thread 0 alone waits for the last items and publishes `done`
        for (;;) {
            const unsigned long long d = s.idone[p].load(std::memory_order_acquire);
            if ((d & ~item_low) != tag) { return; }   // this job ended without this thread
            if ((d & item_low) >= total) { break; }
            cpu_relax();
        }
        if (lead) {
            const auto now = clk::now();
            s.phase_ns[p] += (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
            t = now;
        }
    }
}

// item_barrier + host help: the waiting host claims one item of the open phase of the current job, the same
// tagged claim the pool uses, so it needs no admission handshake: thread 0 cannot finish (or replace) the job
// until this item is counted, and a claim against a closed or newer job fails. Same items, same rows.
bool help_items(pool_state & s) {
    const unsigned long long n = s.gen.load(std::memory_order_acquire);
    if (n == 0) { return false; }
    const unsigned long long tag = (n & item_low) << 32;
    // own_xq: the host has no input copy, so it helps hidden + down only; flow: it helps the input rows only
    for (int p = s.own_xq ? 2 : 0; p < (s.flow ? 1 : 4); ++p) {
        if (p > 0) {
            const unsigned long long d = s.idone[p - 1].load(std::memory_order_acquire);
            if ((d & ~item_low) != tag || (d & item_low) < (unsigned long long) items_of(s, p - 1)) { return false; }
        }
        const unsigned long long total = (unsigned long long) items_of(s, p);
        for (unsigned long long c = s.inext[p].load(std::memory_order_acquire);
                (c & ~item_low) == tag && (c & item_low) < total;) {
            if (s.inext[p].compare_exchange_weak(c, c + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                run_item(s, p, (int) (c & item_low));
                s.idone[p].fetch_add(1, std::memory_order_acq_rel);
                return true;
            }
        }
    }
    return false;
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
    if (idx < 64) { s.wst[idx].cpu = cpu; }
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
        if (s.item_barrier) { run_job_items(s, g, idx); } else { run_job_phases(s, g, idx); }
    }
}

// Thread 0: watches the mapped job (and host-posted probe jobs), runs its share, publishes `done`.
void leader_main(pool_state * sp, int cpu) {
    pool_state & s = *sp;
    pin_thread(cpu);
    s.wst[0].cpu = cpu;
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
            r8_plan(s, layer, card);
            ++n;
            const unsigned long long tag = (n & item_low) << 32;
            if (s.flow) {
                const int ug_blocks = items_of(s, 1) / s.n_exp;
                for (int i = 0; i < s.n_exp; ++i) {
                    s.ug_left[i].store(ug_blocks, std::memory_order_relaxed);
                    s.hq_left[i].store(s.first[i + 1] - s.first[i], std::memory_order_relaxed);
                    for (int q = s.first[i]; q < s.first[i + 1]; ++q) { s.expert_of[q] = i; }
                }
            }
            for (int p = 0; p < 4; ++p) {
                s.next[p].store(s.waitpf && (p == 1 || p == 3) ? s.threads : 0, std::memory_order_relaxed);
                s.idone[p].store(tag, std::memory_order_relaxed);
                s.inext[p].store(tag, std::memory_order_relaxed);
            }
            if (s.wstats) { s.job_t0_ns.store(ns_of(clk::now()), std::memory_order_relaxed); }
            s.gen.store(n, std::memory_order_seq_cst);
            if (s.sleepers.load(std::memory_order_seq_cst) > 0) {
                std::lock_guard<std::mutex> lk(s.sleep_mu);
                s.sleep_cv.notify_all();
            }
            if (s.item_barrier) {
                run_job_items(s, n, 0, true);
                // close this job's claims before the next job's state is written
                for (int p = 0; p < 4; ++p) { s.inext[p].store(tag | item_low, std::memory_order_release); }
            } else {
                run_job_phases(s, n, 0, true);
                const unsigned long long target = n * (unsigned long long) s.threads;
                while (s.fin[3].load(std::memory_order_acquire) < target) { cpu_relax(); }
            }
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
    // Quantized rows are never larger than the f32 rows they come from, except the AI+ q8 rows (int16 pairs + scales).
    s.xq_stride = GGML_PAD(std::max((size_t) max_embd * sizeof(float), moe_ai::use_host_q8() ? moe_ai::q8_bytes(max_embd) : 0), 64);
    s.hq_stride = GGML_PAD(std::max((size_t) max_ff * sizeof(float), moe_ai::use_host_q8() ? moe_ai::q8_bytes(max_ff) : 0), 64);
    const bool align = env_int("GGML_MOE_CPU_SPLIT_ALIGN", 0, 0, 1) != 0;
    s.xq.assign(2 * MOE_CS_MAX_ROUTES * s.xq_stride, 0, align);
    s.hidden.assign((size_t) MOE_CS_MAX_ROUTES * max_ff, 0.0f, align);
    s.hq.assign(MOE_CS_MAX_ROUTES * s.hq_stride, 0, align);
    s.q2_pair = env_int("GGML_MOE_CPU_SPLIT_Q2_PAIR", 0, 0, 1) != 0;
    s.q2_group4 = env_int("GGML_MOE_CPU_SPLIT_Q2_GROUP4", 0, 0, 1) != 0;
    s.vector_silu = env_int("GGML_MOE_CPU_SPLIT_VECTOR_SILU", 0, 0, 1) != 0;
    s.q2_fast = env_int("GGML_MOE_CPU_SPLIT_Q2_FAST", 0, 0, 1) != 0;
    s.q2_shuffle = env_int("GGML_MOE_CPU_SPLIT_Q2_SHUFFLE", 0, 0, 1) != 0;
    s.q2_pf = env_int("GGML_MOE_CPU_SPLIT_Q2_PF", 0, 0, 64);
    s.q8_reuse = env_int("GGML_MOE_CPU_SPLIT_Q8_REUSE", 1, 0, 1) != 0;
    s.down_reuse = env_int("GGML_MOE_CPU_SPLIT_DOWN_REUSE", 1, 0, 1) != 0;
    s.group_tokens = env_int("GGML_MOE_CPU_SPLIT_REUSE8", 1, 0, 1) ? 8 : 4;
    s.r8_mb = moe_ai::use_host_q8() ? env_int("GGML_MOE_AI_HOST_R8_MB", 0, -1, 1 << 20) : 0;
    s.r8_every = env_int("GGML_MOE_AI_HOST_R8_STATS", 16384, 1, 1 << 30);
#if defined(MOE_CS_AVX512)
    s.avx512 = env_int("GGML_MOE_CPU_SPLIT_AVX512", 1, 0, 1) != 0 && cpu_has_avx512_vnni();
#endif
    s.aux_stride = (size_t) max_ff / QK8_0 + (size_t) max_ff / QK2_0 + 16;
    s.hq_aux.assign(MOE_CS_MAX_ROUTES * s.aux_stride, 0.0f, align);
    s.rows_ug = env_int("GGML_MOE_CPU_SPLIT_ROWS_UG", 16, 1, 4096);
    s.rows_down = env_int("GGML_MOE_CPU_SPLIT_ROWS_DOWN", 64, 1, 8192);
#if defined(MOE_IQ_ROWS)
    {   // 1 = on with the measured best kernel mix; 2..17 = on with moe_iq::get variant bits (value - 2)
        const int v = env_int("GGML_MOE_CPU_SPLIT_ROWS", 0, 0, 17);
        s.iq_variant = v == 0 ? -1 : v == 1 ? moe_iq::variant_best : v - 2;
    }
    s.iq_rows = s.iq_variant >= 0;
    s.iq_pf = env_int("GGML_MOE_CPU_SPLIT_PF", 2048, 0, 1 << 16);
#endif
    s.spin_ms = env_int("GGML_MOE_CPU_SPLIT_SPIN_MS", 250, 0, 600000);
    s.spin_us = env_int("GGML_MOE_CPU_SPLIT_SPIN_US", s.spin_ms * 1000, 0, 600000000);
    s.nap_us = env_int("GGML_MOE_CPU_SPLIT_NAP_US", 1000, 1, 100000);
    s.schedule = env_int("GGML_MOE_CPU_SPLIT_SCHEDULE", 0, 0, 1) != 0;
    s.host_help = env_int("GGML_MOE_CPU_SPLIT_HOST", 0, 0, 1) != 0;
    s.item_barrier = env_int("GGML_MOE_CPU_SPLIT_ITEM_BARRIER", 0, 0, 1) != 0 && !s.schedule;
    s.wstats = env_int("GGML_MOE_CPU_SPLIT_WSTATS", 0, 0, 1) != 0;
    s.waitpf = env_int("GGML_MOE_CPU_SPLIT_WAITPF", 0, 0, 1024);
    s.expjob = env_int("GGML_MOE_CPU_SPLIT_EXPJOB", 0, 0, 1) != 0;
    // ITEM_BARRIER and WAITPF/EXPJOB were built separately and schedule the same work counters differently: run one at a time.
    if (s.item_barrier) { s.waitpf = 0; s.expjob = false; }
    if (s.expjob) { s.host_help = false; s.waitpf = 0; }
    for (int p = 0; p < 4; ++p) {
        s.next[p].store(0); s.fin[p].store(0); s.closed[p].store(0); s.inext[p].store(item_low); s.idone[p].store(0);
    }
    // Reserve the caller's actual core and keep the card-driving thread on it.
    const bool max_threads = env_int("GGML_MOE_CPU_SPLIT_MAX_THREADS", 1, 0, 1) != 0;
    // GGML_CPU_CARD_FENCE=1 (default): the card-driving thread is not held on one logical cpu. Pinned to one cpu, any
    // other runnable task there (a browser tab, a compile) time-slices it and the card sits idle (home test 11-05,
    // 3 busy loops on that core: 41.8 vs 24.3 ms/round). The pool still skips the card's core; once the pool cpus are
    // known, the card thread may run on every allowed cpu that has no pool thread. 0 = the old single-cpu pin.
    const bool card_fence = env_int("GGML_CPU_CARD_FENCE", 1, 0, 1) != 0;
    (void) card_fence;
#if defined(__linux__)
    cpu_set_t card_allowed;
    const bool have_allowed = pthread_getaffinity_np(pthread_self(), sizeof(card_allowed), &card_allowed) == 0;
    if (max_threads) {
        int home = sched_getcpu();
        if (card_fence) {
            // With the fence on, ggml no longer pins its main thread under GGML_CPU_EXPERT_PIN=1 (ggml-cpu.c). Reserve
            // the core that pin used (the second physical core) so the pool layout stays the same.
            const char * ep0 = getenv("GGML_CPU_EXPERT_PIN");
            const std::vector<int> c0 = core_leaders();
            if (ep0 && strcmp(ep0, "1") == 0 && c0.size() > 2 && have_allowed && CPU_ISSET(c0[1], &card_allowed)) { home = c0[1]; }
        }
        pin_thread(home);
    }
#endif
    // One thread per physical core, never an SMT sibling, never core 0 (the OS and the desktop) and never the
    // core of the thread that drives the card: a spinning pool thread sharing a core with it is time-sliced in
    // scheduler ticks (the first build measured 6 ms per job that way). GGML_CPU_EXPERT_PIN=1 pins llama's main
    // thread to the second physical core, and so does any single-core affinity of the calling thread; both are
    // skipped. Threads take the highest cores first, away from ggml's own pinned workers (cores 1.. upward).
    std::vector<int> cores = core_leaders();
    std::vector<int> avail;
    std::string skipped;
    if (!cores.empty()) {
        // The cooperative host occupies the reserved core instead of competing with a pool worker.
        if (s.host_help) { s.host_cpu = cores[0]; pin_thread(s.host_cpu); }
        std::vector<char> skip(cores.size(), 0);
        if (!max_threads) { skip[0] = 1; }
        // GGML_MOE_CPU_SPLIT_CORE0=0 keeps the pool off core 0 even with MAX_THREADS: the card queue fences every
        // other process onto core 0 during a timed run, and a pool thread there stalls the layer when it is preempted.
        if (env_int("GGML_MOE_CPU_SPLIT_CORE0", 1, 0, 1) == 0 && cores.size() > 2) { skip[0] = 1; }
        const char * ep = getenv("GGML_CPU_EXPERT_PIN");
        if (!s.host_help && !max_threads && ep && strcmp(ep, "1") == 0 && cores.size() > 2) { skip[1] = 1; }
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
    // GGML_MOE_CPU_SPLIT_CPULIST="11,10,9,..." (default unset): use exactly these cpus, in this order, instead of the
    // derived list. Used to test pool placement (which L3 group, whether the desktop core is in the pool).
#if defined(__linux__)
    if (const char * cl = getenv("GGML_MOE_CPU_SPLIT_CPULIST")) {
        std::vector<int> forced;
        for (const char * c = cl; *c;) {
            if (*c < '0' || *c > '9') { ++c; continue; }
            char * end = nullptr;
            const long v = strtol(c, &end, 10);
            c = end;
            if (v >= 0 && v < CPU_SETSIZE) { forced.push_back((int) v); }
        }
        if (!forced.empty()) { avail = forced; skipped = "forced by GGML_MOE_CPU_SPLIT_CPULIST"; }
    }
#endif
    const int hw_free = !avail.empty() ? (int) avail.size() : (int) std::max(1u, std::thread::hardware_concurrency() / 2 - 1);
    s.threads = env_int("GGML_MOE_CPU_SPLIT_THREADS", hw_free, 1, 64);
    s.own_xq = s.item_barrier && env_int("GGML_MOE_CPU_SPLIT_OWN_XQ", 0, 0, 1) != 0;
    s.flow = s.item_barrier && env_int("GGML_MOE_CPU_SPLIT_FLOW", 0, 0, 1) != 0;
    if (s.own_xq) {
        s.xq_own_stride = 2 * MOE_CS_MAX_ROUTES * s.xq_stride;
        s.xq_own.assign((size_t) s.threads * s.xq_own_stride, 0);
    }
    // Experts computed here use CPU int8 dot products, which sum in a different order than the GPU kernel. Which
    // experts land here follows expert-cache timing, so reruns may differ in last-digit rounding (quality unchanged).
    // On the bank path LLAMA_MOE_BANK_DET=1 gives identical reruns.
    const bool pin = env_int("GGML_MOE_CPU_SPLIT_PIN", 1, 0, 1) != 0 && !avail.empty();
    auto cpu_of = [&](int i) { return pin ? avail[(size_t) i % avail.size()] : -1; };
    std::string card_where;
#if defined(__linux__)
    if (max_threads && card_fence && pin && have_allowed && !s.host_help) {
        // widen the card thread from its reserved cpu to every allowed cpu that no pool thread is pinned to
        cpu_set_t m = card_allowed;
        for (int i = 0; i < s.threads; ++i) { CPU_CLR(cpu_of(i), &m); }
        if (CPU_COUNT(&m) > 1 && pthread_setaffinity_np(pthread_self(), sizeof(m), &m) == 0) {
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &m)) { card_where += (card_where.empty() ? "" : ",") + std::to_string(c); }
            }
        }
    }
#endif
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
            s.q2_fast ? ", Q2_0 down fast path" : "", smt ? ", SMT siblings" : "", s.avx512 ? ", grouped AVX-512 VNNI IQ" : "");
    if (!card_where.empty()) { fprintf(stderr, "[moe-cpu-split] card thread may run on cpus %s (GGML_CPU_CARD_FENCE=1)\n", card_where.c_str()); }
    if (align) { fprintf(stderr, "[moe-cpu-split] scratch rows start on 64-byte lines (GGML_MOE_CPU_SPLIT_ALIGN)\n"); }
    if (s.q2_pair && s.q2_fast) { fprintf(stderr, "[moe-cpu-split] Q2_0 down shares weights across route pairs\n"); }
    if (s.q2_group4 && s.q2_fast) { fprintf(stderr, "[moe-cpu-split] Q2_0 down shares weights across three/four-route groups\n"); }
    if (s.vector_silu && s.iq_rows) { fprintf(stderr, "[moe-cpu-split] grouped up/gate uses CPU vector SiLU\n"); }
    if (s.iq_rows) { fprintf(stderr, "[moe-cpu-split] IQ up/gate row kernels (each chunk decoded once per expert), variant %d, prefetch %d B\n", s.iq_variant, s.iq_pf); }
    if (s.expjob) { fprintf(stderr, "[moe-cpu-split] one worker per whole expert (GGML_MOE_CPU_SPLIT_EXPJOB)\n"); }
    if (s.schedule) { fprintf(stderr, "[moe-cpu-split] static quantization rows, grouped math claims\n"); }
    if (s.host_help) { fprintf(stderr, "[moe-cpu-split] waiting host drains math tiles on cpu %d\n", s.host_cpu); }
    if (s.item_barrier) { fprintf(stderr, "[moe-cpu-split] item barriers (a descheduled idle thread does not stall a job)%s%s\n", s.own_xq ? ", private input rows per thread" : "", s.flow ? ", per-expert flow (no phase barriers after the inputs)" : ""); }
    fprintf(stderr, "[moe-cpu-split] input Q8 reuse %s, IQ token group %d, IQ4_NL down reuse %s\n", s.q8_reuse ? "on" : "off", s.group_tokens, s.down_reuse ? "on" : "off");
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
    if (s.item_barrier) { return help_items(s); }
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
        if (ggml_ai4_is_unit(t)) {
            // fork AIPFIX D2: AI+ unit experts; inputs are rotated into the weights' basis instead of quantized
            m.ai = true;
            m.ai_k = ggml_ai4_rate(t);
            m.ai_C = ggml_ai4_trellis_mul(t);
            m.vec_type = GGML_TYPE_F32;
            m.quant = moe_ai::rotate;
            m.vec_bytes = (size_t) m.r.n_in * sizeof(float);
            if (moe_ai::use_host_q8() && m.r.n_in <= 8192) {   // rotate_q8's stack row holds 8192 floats
                m.ai_q8 = true;
                m.quant = moe_ai::rotate_q8;
                m.vec_bytes = moe_ai::q8_bytes(m.r.n_in);
            }
            // U2 row-scale types (K % 640) are read by block (moe_ai::unit_src); the row size must give the type's layout,
            // or the host would walk U2 bytes as U units (wrong outputs, not a crash). AIF25 (rate 25) stays off the host split.
            ok = m.ai_k >= 2 && m.ai_k <= 5 && m.r.n_in % AI4U_UK == 0 && m.r.n_out % AI4_TR == 0 &&
                 m.r.row_bytes == ggml_row_size(t, m.r.n_in) && s.rows_ug % AI4_TR == 0 && s.rows_down % AI4_TR == 0 &&
                 (ggml_ai4v2_rate(t) != 0) == moe_ai::slab_is_v2(m.r.row_bytes, m.ai_k, m.r.n_in);
            continue;
        }
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
            if (s.iq_rows && m.vec_type == GGML_TYPE_Q8_K) {
                m.rows[nt] = moe_iq::get(m.r.type, nt, s.iq_variant);
#if defined(MOE_IQ_ROWS_512)
                if (s.avx512) { m.rows[nt] = moe_iq::get512(m.r.type, nt); }
#endif
            }
        }
#endif
        const auto * vt = ggml_get_type_traits_cpu(m.vec_type);
        m.quant = vt ? vt->from_float : nullptr;
        ok = m.quant && m.r.n_in % ggml_blck_size(m.vec_type) == 0 &&
             (!m.q2 || m.vec_type == GGML_TYPE_Q8_0);
        m.vec_bytes = ok ? ggml_row_size(m.vec_type, m.r.n_in) : 0;
    }
    ok = ok && L.role[0].ai == L.role[1].ai;
    L.ok = ok;
    L.shared_input = ok && L.role[0].vec_type == L.role[1].vec_type;
    if ((int) s.layers.size() <= layer) { s.layers.resize(layer + 1); }
    // The pool reads the layer table only while a card job is in flight; registration happens at session
    // preparation, outside any graph, so no job is running.
    s.layers[layer] = L;
    s.r8_reset = true;
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
#if !defined(__linux__)
    return 0.0;   // no schedstat outside Linux
#else
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
#endif
}

static bool moe_cs_probe_layer(int layer, double * route_ns, double * fixed_ns);

bool moe_cs_probe(double * route_ns, double * fixed_ns, int * timed_layer) {
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
    const bool ok = moe_cs_probe_layer(layer, route_ns, fixed_ns);
    if (timed_layer) { *timed_layer = layer; }
    // AIPFAST: GGML_MOE_CPU_SPLIT_PROBE_LAYERS=20,31 also probes those layers (log only; the planner keeps the first
    // probe), e.g. to time host misses of an AI+ layer next to a 3XS one in the same process.
    if (const char * e = getenv("GGML_MOE_CPU_SPLIT_PROBE_LAYERS")) {
        for (const char * c = e; *c; ) {
            const int l = atoi(c);
            bool usable;
            {
                std::lock_guard<std::mutex> lock(s.mu);
                usable = l >= 0 && l < (int) s.layers.size() && s.layers[l].ok;
            }
            double r = 0.0, f = 0.0;
            if (!usable || !moe_cs_probe_layer(l, &r, &f)) { fprintf(stderr, "[moe-cpu-split] extra probe on layer %d: not available\n", l); }
            while (*c && *c != ',') { ++c; }
            if (*c == ',') { ++c; }
        }
    }
    return ok;
}

static bool moe_cs_probe_layer(int layer, double * route_ns, double * fixed_ns) {
    pool_state & s = P();
    const layer_math & L = s.layers[layer];
    // Experts with a host source; the probe reads real weights. Enough of them (64 x ~2 MB) that the rotation
    // below reads from memory, not from the last-level cache, as a decode's missed experts do.
    std::vector<int> experts;
    for (int e = 0; e < (int) L.role[0].r.n_expert && experts.size() < 64; ++e) {
        bool all = true;
        for (int k = 0; k < 3; ++k) { all = all && moe_cs_load_acquire(&L.role[k].r.src[e]) != nullptr; }
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
    wst_clear(s);
    fprintf(stderr, "[moe-cpu-split] probe on layer %d: empty job %.1f us, 1 expert %.1f us, 4 experts %.1f us -> %.1f us per (expert, route) + %.1f us per job\n",
            layer, t0 / 1000.0, t1 / 1000.0, t4 / 1000.0, per / 1000.0, *fixed_ns / 1000.0);
    // Contention check: warns when the pool spent over 25% of the timed jobs waiting for a cpu (about 1.33x slower or worse).
    size_t expert_bytes = 0;
    for (int k = 0; k < 3; ++k) { expert_bytes += L.role[k].r.row_bytes * (size_t) L.role[k].r.n_out; }
    fprintf(stderr, "[moe-cpu-split] probe check: %.2f MB per expert, %.1f GB/s for 1 expert, %.1f GB/s for 4; "
            "pool waited for a cpu %.1f%% of the time\n", expert_bytes / 1e6, expert_bytes / std::max(1.0, t1),
            4.0 * expert_bytes / std::max(1.0, t4), 100.0 * stolen);
    if (stolen > 0.25) {
        fprintf(stderr, "[moe-cpu-split] WARN: CPU CONTENTION at load: 1 expert %.1f us, pool waited for a cpu "
                "%.1f%% of the time. Other programs are running on the pool's cpus; split timings from this run are not "
                "valid.\n", t1 / 1000.0, 100.0 * stolen);
    }
    return true;
}

// AIPFIX per-type probe (GGML_MOE_CPU_SPLIT_PROBE_PER_TYPE=1): the host types a registered layer computes with, and a
// probe of that layer alone (pool idle, as moe_cs_probe).
bool moe_cs_layer_types(int layer, int types[3]) {
    pool_state & s = P();
    std::lock_guard<std::mutex> lock(s.mu);
    if (layer < 0 || layer >= (int) s.layers.size() || !s.layers[layer].ok) { return false; }
    for (int k = 0; k < 3; ++k) { types[k] = s.layers[layer].role[k].r.type; }
    return true;
}

bool moe_cs_probe_at(int layer, double * route_ns, double * fixed_ns) {
    pool_state & s = P();
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (!s.running || layer < 0 || layer >= (int) s.layers.size() || !s.layers[layer].ok) { return false; }
    }
    return moe_cs_probe_layer(layer, route_ns, fixed_ns);
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
    wst_clear(s);
    return true;
}

void moe_cs_wstats_report(const char * when) {
    pool_state & s = P();
    if (!s.wstats) { return; }
    static const char * const names[4] = {"quantize", "up+gate", "silu", "down"};
    for (int i = 0; i < s.threads && i < 64; ++i) {
        const pool_state::wstat_t & w = s.wst[i];
        const double j = (double) std::max(1ull, w.jobs);
        fprintf(stderr, "[moe-cpu-split] wstats %s worker %d cpu %d jobs %llu join-lag %.2f us/job", when, i, w.cpu, w.jobs, w.lag_ns / j / 1e3);
        for (int p = 0; p < 4; ++p) {
            fprintf(stderr, " | %s work %.1f wait %.1f", names[p], w.work_ns[p] / j / 1e3, w.wait_ns[p] / j / 1e3);
        }
        fprintf(stderr, " us/job\n");
    }
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

bool moe_cs_iq_unpack(int type, const void * block, int c, bool joined, unsigned char * mag, signed char * sgn, int * mul) {
#if defined(MOE_IQ_ROWS)
    return moe_iq::unpack_slice(type, block, c, joined, (uint8_t *) mag, (int8_t *) sgn, mul);
#else
    (void) type; (void) block; (void) c; (void) joined; (void) mag; (void) sgn; (void) mul;
    return false;
#endif
}

bool moe_cs_iq_rows(int type, int nt, const void * w, size_t row_bytes, int64_t r0, int64_t r1, int64_t n_in,
        const void * const * y, float * const * out, int prefetch, int variant, bool wide) {
#if defined(MOE_IQ_ROWS)
    moe_iq::rows_fn f = moe_iq::get(type, nt, variant);
    if (wide) {
#if defined(MOE_IQ_ROWS_512)
        f = cpu_has_avx512_vnni() ? moe_iq::get512(type, nt) : nullptr;
#else
        f = nullptr;
#endif
    }
    if (!f || n_in % QK_K != 0) { return false; }
    f((const char *) w, row_bytes, r0, r1, (int) (n_in / QK_K), (const block_q8_K * const *) y, out, prefetch);
    return true;
#else
    (void) type; (void) nt; (void) w; (void) row_bytes; (void) r0; (void) r1; (void) n_in; (void) y; (void) out; (void) prefetch; (void) variant; (void) wide;
    return false;
#endif
}

// AIPFAST route 2 (see moe-cpu-split.h): host-only copies of routed-expert tensors from other model files.
// With every cache slot in AI+ bytes, a host miss would run the AI+ trellis decode (about 2.5x the IQ row kernels
// on an AVX2 desktop). The base file still holds each expert's 3XS bytes; the pool computes its misses from those
// (same math as the ship stack's host misses) while the card keeps the AI+ bytes. Pages load on first use and stay
// in the page cache; nothing here is pinned.
#include "gguf.h"
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
struct host_alt_file {
    std::string path;
    gguf_context * g = nullptr;
    ggml_context * ctx = nullptr;
    const char * base = nullptr;
    size_t size = 0;
};
struct host_alt_store {
    std::mutex mu;
    bool opened = false;
    std::vector<host_alt_file> files;
    std::vector<std::vector<const void *>> srcs;   // kept alive for the pool's role tables
    std::vector<std::pair<std::string, moe_cs_role>> found;   // one entry per tensor name (sessions re-register)
    std::vector<std::pair<const char *, size_t>> ranges;      // mapped bytes of each kept tensor (warm-up)
    std::map<std::string, std::pair<const char *, size_t>> pending;   // found, not yet kept by moe_cs_host_alt_keep
    std::map<int, bool> layer_kept;                           // per layer: 3XS copy kept (true) or over budget (false)
    double budget = -1.0;                                     // bytes left for kept copies (-1 = not set yet)
    bool warmed = false;
};
// MemAvailable in bytes (0 if unreadable)
double host_alt_mem_avail() {
    long long avail_kib = -1;
    if (FILE * fp = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) { if (sscanf(line, "MemAvailable: %lld kB", &avail_kib) == 1) { break; } }
        fclose(fp);
    }
    return avail_kib > 0 ? avail_kib * 1024.0 : 0.0;
}
double host_alt_floor_gib() {
    const char * fe = getenv("GGML_MOE_CPU_SPLIT_HOST_WARM_FLOOR_GIB");
    return std::max(5.0, fe && fe[0] ? atof(fe) : 5.0);
}
host_alt_store & HA() { static host_alt_store s; return s; }
}

bool moe_cs_host_alt(const char * name, int64_t n_in, int64_t n_out, int64_t n_expert, moe_cs_role * out) {
#if defined(__linux__)
    const char * env = getenv("GGML_MOE_CPU_SPLIT_HOST_GGUF");
    if (!env || !env[0] || !name || !out) { return false; }
    host_alt_store & h = HA();
    std::lock_guard<std::mutex> lock(h.mu);
    if (!h.opened) {
        h.opened = true;
        std::string all(env);
        for (size_t a = 0; a <= all.size();) {
            size_t b = all.find(',', a);
            if (b == std::string::npos) { b = all.size(); }
            const std::string path = all.substr(a, b - a);
            a = b + 1;
            if (path.empty()) { continue; }
            host_alt_file f;
            f.path = path;
            gguf_init_params ip = { /*.no_alloc = */ true, /*.ctx = */ &f.ctx };
            f.g = gguf_init_from_file(path.c_str(), ip);
            const int fd = f.g ? open(path.c_str(), O_RDONLY) : -1;
            struct stat st = {};
            if (fd >= 0 && fstat(fd, &st) == 0 && st.st_size > 0) {
                void * m = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                if (m != MAP_FAILED) { f.base = (const char *) m; f.size = (size_t) st.st_size; }
            }
            if (fd >= 0) { close(fd); }
            fprintf(stderr, "[moe-cpu-split] host 3XS source %s: %s\n", path.c_str(), f.base ? "mapped" : "FAILED (not used)");
            if (f.base) { h.files.push_back(f); }
        }
    }
    for (const auto & fr : h.found) {
        if (fr.first == name && fr.second.n_in == n_in && fr.second.n_out == n_out && fr.second.n_expert == n_expert) {
            *out = fr.second;
            return true;
        }
    }
    // the loader already checked that these files are the base model (llama-model-loader: host 3XS source check)
    for (const host_alt_file & f : h.files) {
        const int64_t id = gguf_find_tensor(f.g, name);
        const ggml_tensor * t = id >= 0 ? ggml_get_tensor(f.ctx, name) : nullptr;
        if (!t) { continue; }
        const ggml_type type = t->type;
        const size_t rb = ggml_row_size(type, t->ne[0]);
        const size_t off = gguf_get_data_offset(f.g) + gguf_get_tensor_offset(f.g, id);
        const size_t eb = rb * (size_t) t->ne[1];
        if (t->ne[0] != n_in || t->ne[1] != n_out || t->ne[2] != n_expert || t->ne[3] != 1 ||
                off + eb * (size_t) n_expert > f.size || ggml_ai4_is_unit(type)) {
            fprintf(stderr, "[moe-cpu-split] host 3XS source: %s in %s has another shape or an AI+ type, trying the next file\n",
                    name, f.path.c_str());
            continue;
        }
        std::vector<const void *> src((size_t) n_expert);
        for (int64_t e = 0; e < n_expert; ++e) { src[(size_t) e] = f.base + off + (size_t) e * eb; }
        h.srcs.push_back(std::move(src));
        out->type = (int) type;
        out->n_in = n_in;
        out->n_out = n_out;
        out->n_expert = n_expert;
        out->row_bytes = rb;
        out->src = h.srcs.back().data();
        h.found.emplace_back(name, *out);
        h.pending[name] = { f.base + off, eb * (size_t) n_expert };
        return true;
    }
    fprintf(stderr, "[moe-cpu-split] host 3XS source: %s not found (with this shape) in GGML_MOE_CPU_SPLIT_HOST_GGUF; "
            "this layer's host misses keep the AI+ decode\n", name);
    return false;
#else
    (void) name; (void) n_in; (void) n_out; (void) n_expert; (void) out;
    return false;
#endif
}

bool moe_cs_host_alt_keep(int layer, const char * const names[3]) {
#if defined(__linux__)
    host_alt_store & h = HA();
    std::lock_guard<std::mutex> lock(h.mu);
    if (const auto it = h.layer_kept.find(layer); it != h.layer_kept.end()) { return it->second; }
    size_t bytes = 0;
    for (int k = 0; k < 3; ++k) {
        const auto it = h.pending.find(names[k]);
        if (it == h.pending.end()) { return false; }
        bytes += it->second.second;
    }
    const double gib = 1024.0*1024.0*1024.0;
    const char * fit = getenv("GGML_MOE_CPU_SPLIT_HOST_FIT");
    const bool fit_on = !(fit && fit[0] == '0');
    if (fit_on && h.budget < 0.0) {
        // budget: GGML_MOE_CPU_SPLIT_HOST_BUDGET_MB, else MemAvailable minus the warm-up floor (5 GiB default, min 5; Jake RAM rule)
        const char * be = getenv("GGML_MOE_CPU_SPLIT_HOST_BUDGET_MB");
        h.budget = be && be[0] ? std::max(0.0, atof(be)) * 1024.0 * 1024.0
                               : std::max(0.0, host_alt_mem_avail() - host_alt_floor_gib() * gib);
        fprintf(stderr, "[moe-cpu-split] host 3XS copies: RAM budget %.2f GiB (%s); layers past it keep the AI+ decode\n",
                h.budget / gib, be && be[0] ? "GGML_MOE_CPU_SPLIT_HOST_BUDGET_MB" : "MemAvailable minus floor");
    }
    if (fit_on && (double) bytes > h.budget) {
        h.layer_kept[layer] = false;
        fprintf(stderr, "[moe-cpu-split] layer %d: host 3XS copy (%.2f GiB) over the RAM budget (%.2f GiB left), "
                "host misses keep the AI+ decode\n", layer, bytes / gib, h.budget / gib);
        return false;
    }
    if (fit_on) { h.budget -= (double) bytes; }
    for (int k = 0; k < 3; ++k) { h.ranges.push_back(h.pending[names[k]]); }
    h.layer_kept[layer] = true;
    return true;
#else
    (void) layer; (void) names;
    return false;
#endif
}

double moe_cs_host_alt_warm() {
#if defined(__linux__)
    host_alt_store & h = HA();
    std::lock_guard<std::mutex> lock(h.mu);
    size_t total = 0;
    for (const auto & r : h.ranges) { total += r.second; }
    if (total == 0 || h.warmed) { return 0.0; }
    h.warmed = true;
    const double gib = 1024.0*1024.0*1024.0;
    const char * e = getenv("GGML_MOE_CPU_SPLIT_HOST_WARM");
    if (e && e[0] == '0') {
        fprintf(stderr, "[moe-cpu-split] host 3XS warm-up off (GGML_MOE_CPU_SPLIT_HOST_WARM=0): %.2f GiB cold, first misses read the disk\n",
                total / gib);
        return 0.0;
    }
    // budget: MemAvailable minus the 5 GiB floor (GGML_MOE_CPU_SPLIT_HOST_WARM_FLOOR_GIB raises it); ranges in layer order
    const double target_gib = host_alt_floor_gib();
    const double avail = host_alt_mem_avail();
    double budget = std::max(0.0, avail - target_gib * gib);
    size_t warmed = 0;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    for (const auto & r : h.ranges) {
        if ((double) (warmed + r.second) > budget) { break; }
        const uintptr_t b = (uintptr_t) r.first & ~(uintptr_t) (page - 1);
        const size_t len = (uintptr_t) r.first + r.second - b;
        if (madvise((void *) b, len, MADV_WILLNEED) == 0) { warmed += r.second; }
    }
    fprintf(stderr, "[moe-cpu-split] host 3XS warm-up: %.2f of %.2f GiB queued for reading (MemAvailable %.2f GiB, keeps %.1f GiB free); "
            "%.2f GiB stay cold (misses there read the disk)%s\n", warmed / gib, total / gib, avail / gib, target_gib,
            (total - warmed) / gib, warmed < total ? " - WARNING: not all host 3XS bytes fit in RAM" : "");
    // WILLNEED only fills the page cache; this mapping's page tables stay empty, so each first touch on the miss path
    // still took a page fault (Taichi h arm, aipfast-r2k: 144 us per host route measured vs 29 us probed). Fault the
    // warmed ranges in now, at load, the same way the model's own map uses MAP_POPULATE.
    // GGML_MOE_CPU_SPLIT_HOST_POPULATE=0 keeps the lazy faults.
#if !defined(MADV_POPULATE_READ)
#define MADV_POPULATE_READ 22
#endif
    const char * pe = getenv("GGML_MOE_CPU_SPLIT_HOST_POPULATE");
    if (warmed > 0 && !(pe && pe[0] == '0')) {
        const auto t0 = std::chrono::steady_clock::now();
        size_t done = 0, left = warmed;
        bool ok = true;
        for (const auto & r : h.ranges) {
            if (r.second > left) { break; }
            left -= r.second;
            const uintptr_t b = (uintptr_t) r.first & ~(uintptr_t) (page - 1);
            const size_t len = (uintptr_t) r.first + r.second - b;
            if (madvise((void *) b, len, MADV_POPULATE_READ) == 0) { done += r.second; } else { ok = false; break; }
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[moe-cpu-split] host 3XS copies mapped in: %.2f of %.2f GiB in %.0f ms%s\n", done / gib, warmed / gib, ms,
                ok ? "" : " (MADV_POPULATE_READ failed: older kernel? the rest fault in on first use)");
    }
    return warmed / gib;
#else
    return 0.0;
#endif
}
