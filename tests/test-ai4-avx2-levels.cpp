// test-ai4-avx2 helper TU, built with -mavx2 -mfma on GCC/Clang x86 (tests/CMakeLists.txt) so the AVX2 pair-table decode
// (ggml-ai4-avx2.h) can be checked against the scalar decoder even when the test itself is built for the generic target.
// The caller runs it only on a CPU that reports AVX2 + FMA. No standard-library templates here: an AVX2 copy of a template
// shared with the main TU could be the one the linker keeps.
#include "../ggml/src/ggml-ai4.h"
#include "../ggml/src/ggml-ai4-avx2.h"

#include <cstring>

// 1: bit-identical, 0: mismatch, -1: not built with AVX2/FMA
int ai4_avx2_pair_levels_check(unsigned seed) {
#if GGML_AI4_PAIR_AVX2
    constexpr int64_t K = 2560;
    static uint8_t slab[(K/AI4U_UK)*AI4U_UNIT_BYTES(2)];
    static float a[AI4_TR*K], b[AI4_TR*K];
    uint32_t s = seed ? seed : 1u;
    for (auto & v : slab) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; v = (uint8_t) (s >> 24); }   // xorshift32
    for (int64_t u = 0; u < K/AI4U_UK; ++u) {
        uint16_t * dh = (uint16_t *) (slab + u*AI4U_UNIT_BYTES(2));
        for (int r = 0; r < AI4_TR; ++r) { dh[r] = (uint16_t) (0x2100 + 16*r); }   // fp16 ~0.01 .. 0.016
    }
    aik_dequantize_slab_rotated_c(2, 0, slab, a, K);
    ai4_pair_avx2::dequantize_slab_rotated(slab, b, K);
    return memcmp(a, b, sizeof(a)) == 0 ? 1 : 0;
#else
    (void) seed;
    return -1;
#endif
}
