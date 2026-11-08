#pragma once
// Portability helpers for the host-side MoE CPU split (moe-cpu-split.cpp, moe-ai-rows.h, moe-iq-rows.h).
// GCC/Clang: the macros expand to the original attributes/builtins. MSVC: equivalents or runtime CPUID checks.

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MOE_ALWAYS_INLINE __forceinline
// MSVC allows any intrinsic regardless of /arch, so per-function targets are not needed; callers still gate the
// AVX-512 paths on moe_cpu_supports() at run time.
#define MOE_TARGET(features)

#define __builtin_bswap64(x) _byteswap_uint64(x)
#define __builtin_fabsf(x)   fabsf(x)
#define __builtin_roundf(x)  roundf(x)
#define __builtin_fmaf(a, b, c) fmaf((a), (b), (c))

// Runtime feature check matching the __builtin_cpu_supports names the split code asks for.
static inline bool moe_msvc_cpu_supports(const char * f) {
    int r[4];
    __cpuid(r, 0);
    const int max_leaf = r[0];
    __cpuid(r, 1);
    const bool osxsave = (r[2] >> 27) & 1;
    // AVX-512 needs the OS to save opmask + ZMM state (XCR0 bits 1,2,5,6,7)
    const bool os_avx512 = osxsave && ((_xgetbv(0) & 0xE6) == 0xE6);
    if (max_leaf < 7 || !os_avx512) { return false; }
    __cpuidex(r, 7, 0);
    const unsigned ebx = (unsigned) r[1], ecx = (unsigned) r[2];
    if (!strcmp(f, "avx512f"))    { return (ebx >> 16) & 1; }
    if (!strcmp(f, "avx512dq"))   { return (ebx >> 17) & 1; }
    if (!strcmp(f, "avx512bw"))   { return (ebx >> 30) & 1; }
    if (!strcmp(f, "avx512vl"))   { return (ebx >> 31) & 1; }
    if (!strcmp(f, "avx512vbmi")) { return (ecx >> 1) & 1; }
    if (!strcmp(f, "avx512vnni")) { return (ecx >> 11) & 1; }
    return false;
}
#define __builtin_cpu_init() ((void) 0)
#define __builtin_cpu_supports(feature) moe_msvc_cpu_supports(feature)

// GCC/Clang scalar F16C helpers missing from MSVC
#if !defined(_cvtsh_ss)
static inline float moe_msvc_cvtsh_ss(unsigned short h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
#define _cvtsh_ss(h) moe_msvc_cvtsh_ss((unsigned short) (h))
#endif
#if !defined(_cvtss_sh)
static inline unsigned short moe_msvc_cvtss_sh(float f, int) {
    return (unsigned short) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), 0), 0);
}
#define _cvtss_sh(f, r) moe_msvc_cvtss_sh((f), (r))
#endif

#else
#define MOE_ALWAYS_INLINE __attribute__((always_inline))
#define MOE_TARGET(features) __attribute__((target(features)))
#endif
