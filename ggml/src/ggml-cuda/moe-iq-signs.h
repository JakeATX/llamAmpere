#pragma once
// Sign bytes of the IQ formats with 7-bit sign indices, computed at compile time.
//
// ggml-common.h declares ksigns_iq2xs as `static const`, not `constexpr`, so a constexpr constructor must not read it
// (g++ versions differ on whether that compiles; MSVC rejects it). Entry i of the table is the 7-bit index with an even
// parity bit in bit 7, so the byte is rebuilt from i. tests/test-moe-iq-signs.cpp compares every entry, and every
// pm_lut word, with the runtime ksigns_iq2xs table.

#include <cstdint>

namespace moe_iq {

// ksigns_iq2xs[i] for i in [0, 128): i | (parity of the low 7 bits of i) << 7
constexpr uint8_t iq_sign_byte(int i) {
    unsigned parity = 0;
    for (int b = 0; b < 7; ++b) { parity ^= ((unsigned) i >> b) & 1u; }
    return (uint8_t) ((unsigned) i | (parity << 7));
}

// Independent definition for the compile-time check: the low 7 bits are the index and the whole byte has an even
// number of set bits (counted by clearing the lowest set bit, not by xor-ing bits).
constexpr bool iq_sign_byte_valid(int i, uint8_t byte) {
    unsigned n = 0;
    for (unsigned v = byte; v; v &= v - 1) { ++n; }
    return (byte & 0x7F) == i && (n & 1u) == 0;
}

constexpr bool iq_sign_bytes_all_valid() {
    for (int i = 0; i < 128; ++i) {
        if (!iq_sign_byte_valid(i, iq_sign_byte(i))) { return false; }
    }
    return true;
}
static_assert(iq_sign_bytes_all_valid(), "iq_sign_byte must give even-parity bytes with the index in the low 7 bits");

// Widen each sign byte once to one byte per weight: 0x01 keeps the activation, 0xFF negates it (the second operand
// of _mm256_sign_epi8).
struct pm_lut {
    uint64_t v[128];
    constexpr pm_lut() : v() {
        for (int i = 0; i < 128; ++i) {
            const unsigned sign = iq_sign_byte(i);
            uint64_t w = 0;
            for (int bit = 0; bit < 8; ++bit) {
                w |= ((sign >> bit) & 1 ? 0xFFull : 0x01ull) << (8 * bit);
            }
            v[i] = w;
        }
    }
};
alignas(64) inline constexpr pm_lut k_pm{};
static_assert(k_pm.v[3] == 0x010101010101FFFFull && k_pm.v[127] == 0xFFFFFFFFFFFFFFFFull, "pm_lut");

}  // namespace moe_iq
