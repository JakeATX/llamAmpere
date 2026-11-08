#include "moe-cpu-split-dispatch.h"
#include <cstdlib>
#include <cstring>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#elif (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
#include <cpuid.h>
#endif

static bool moe_cs_has_avx2() {
#if defined(MOE_CS_DISPATCH_AVX2) && defined(_MSC_VER) && defined(_M_X64)
    int r[4];
    __cpuidex(r, 1, 0);
    const int need = (1 << 12) | (1 << 27) | (1 << 28) | (1 << 29); // FMA, OSXSAVE, AVX, F16C
    if ((r[2] & need) != need || (_xgetbv(0) & 6) != 6) { return false; }
    __cpuidex(r, 7, 0);
    return (r[1] & ((1 << 5) | (1 << 8))) == ((1 << 5) | (1 << 8)); // AVX2, BMI2
#elif defined(MOE_CS_DISPATCH_AVX2) && (defined(__x86_64__) || defined(__i386__)) && defined(__GNUC__)
    unsigned eax, ebx, ecx, edx;
    // Older Clang accepts AVX2/FMA queries, but not the F16C query string.
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx) || !(ecx & (1u << 29))) { return false; }
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
           __builtin_cpu_supports("bmi2");
#else
    return false;
#endif
}

const moe_cs_kernels & moe_cs_get_kernels() {
    static const moe_cs_kernels * api = [] {
#if defined(MOE_CS_DISPATCH_AVX2)
        const char * e = getenv("GGML_MOE_CPU_SPLIT_ISA");
        if (!(e && strcmp(e, "scalar") == 0) && moe_cs_has_avx2()) { return &moe_cs_avx2_kernels; }
#endif
        return &moe_cs_scalar_kernels;
    }();
    return *api;
}

const char * moe_cs_isa() {
#if defined(MOE_CS_DISPATCH_AVX2)
    if (&moe_cs_get_kernels() == &moe_cs_avx2_kernels) { return "avx2"; }
#endif
    return "scalar";
}

void moe_cs_ai_quant(const float * x, void * q, int64_t K) {
    moe_cs_get_kernels().ai_rotate_q8(x, q, K);
}
void moe_cs_ai_rows(const char * w, size_t row_bytes, int k, uint32_t C, int64_t a, int64_t b, int64_t K, int nt,
        const void * const * xq, float * const * out) {
    moe_cs_get_kernels().ai_q8(w, row_bytes, k, C, a, b, K, nt, xq, out);
}
