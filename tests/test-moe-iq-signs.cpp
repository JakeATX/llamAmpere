// The compile-time sign bytes of the host MoE IQ row kernels (moe-iq-signs.h) against ggml's runtime ksigns_iq2xs
// table: all 128 sign bytes and all 128 widened pm_lut words. Host-only, no model, no special compiler flags.
#define GGML_COMMON_IMPL_CPP
#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-cuda/moe-iq-signs.h"

#include <cstdio>

int main() {
    int bad = 0;
    for (int i = 0; i < 128; ++i) {
        const uint8_t got = moe_iq::iq_sign_byte(i);
        if (got != ksigns_iq2xs[i]) {
            fprintf(stderr, "sign byte %d: got 0x%02x, ksigns_iq2xs 0x%02x\n", i, got, ksigns_iq2xs[i]);
            ++bad;
        }
        uint64_t w = 0;
        for (int bit = 0; bit < 8; ++bit) {
            w |= (uint64_t) (((ksigns_iq2xs[i] >> bit) & 1) ? 0xFF : 0x01) << (8 * bit);
        }
        if (moe_iq::k_pm.v[i] != w) {
            fprintf(stderr, "pm_lut %d: got 0x%016llx, expected 0x%016llx\n", i,
                    (unsigned long long) moe_iq::k_pm.v[i], (unsigned long long) w);
            ++bad;
        }
    }
    printf("test-moe-iq-signs: %s (%d mismatches over 128 entries)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
