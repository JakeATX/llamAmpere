// fork: compile the same host math with AVX2; the baseline dispatcher checks the CPU and OS first.
#define MOE_CS_IMPL moe_cs_avx2_math
#define MOE_CS_BUILD_AVX2 1
#if defined(_MSC_VER) && !defined(__clang__)
#define __FMA__ 1
#define __F16C__ 1
#define __BMI2__ 1
#include "moe-msvc-compat.h"   // _cvtsh_ss / _cvtss_sh, used by moe-iq-rows.h and moe-cpu-split-q2.h
#endif
#include "moe-cpu-split-kernels.cpp"
