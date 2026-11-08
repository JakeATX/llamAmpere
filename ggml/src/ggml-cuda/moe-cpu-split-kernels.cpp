#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../ggml-common.h"
#include "../ggml-impl.h"
#include "../ggml-ai4.h"
#include "moe-cpu-split-dispatch.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif
#ifndef MOE_CS_IMPL
#define MOE_CS_IMPL moe_cs_scalar_math
#endif
namespace MOE_CS_IMPL {
#include "moe-ai-rows.h"
#include "moe-iq-rows.h"
#include "moe-cpu-split-q2.h"

static const char * ai_decode() {
#if defined(__AVX2__) && defined(__FMA__)
    if (moe_ai::use_host_q8()) { return moe_ai::use_q8_vbmi() ? "avx512-vbmi" : "avx2"; }
    return moe_ai::has_avx512() ? "avx512" : "avx2";
#else
    return "scalar";
#endif
}
static void ai_f32(const char * w, size_t rb, int k, uint32_t C, int64_t a, int64_t b, int64_t K, int nt,
        const float * const * x, float * const * o) {
    moe_ai::slab_rows(w, rb, k, C, a, b, K, nt, x, o);
}
static void ai_q8(const char * w, size_t rb, int k, uint32_t C, int64_t a, int64_t b, int64_t K, int nt,
        const void * const * x, float * const * o) {
    moe_ai::slab_rows_q8(w, rb, k, C, a, b, K, nt, x, o);
}
static moe_cs_rows_fn iq_get(int type, int nt, int variant, bool wide) {
#if defined(MOE_IQ_ROWS)
    if (!wide) { return moe_iq::get(type, nt, variant); }
#if defined(MOE_IQ_ROWS_512)
    return moe_iq::get512(type, nt);
#endif
#else
    (void) type; (void) nt; (void) variant; (void) wide;
#endif
    return nullptr;
}
static bool iq_unpack(int type, const void * block, int c, bool joined, unsigned char * mag, signed char * sgn, int * mul) {
#if defined(MOE_IQ_ROWS)
    return moe_iq::unpack_slice(type, block, c, joined, mag, sgn, mul);
#else
    (void) type; (void) block; (void) c; (void) joined; (void) mag; (void) sgn; (void) mul;
    return false;
#endif
}
} // namespace MOE_CS_IMPL

using namespace MOE_CS_IMPL;
#if defined(MOE_CS_BUILD_AVX2)
const moe_cs_kernels moe_cs_avx2_kernels = {
#else
const moe_cs_kernels moe_cs_scalar_kernels = {
#endif
    ai_decode, moe_ai::r8_ok, moe_ai::rotate, moe_ai::rotate_q8, ai_f32, ai_q8,
    moe_ai::slab_rows_q8_keep, moe_ai::slab_rows_r8, dot_q2_0_q8_0, q2_reorder_hidden,
    { nullptr, dot_q2_0_rows<1>, dot_q2_0_rows<2>, dot_q2_0_rows<3>, dot_q2_0_rows<4> }, iq_get, iq_unpack, moe_ai::quant_q8,
};
