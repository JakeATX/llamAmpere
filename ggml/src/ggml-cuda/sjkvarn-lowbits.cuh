#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define SJKVARN_LOW_HD __host__ __device__ __forceinline__
#else
#define SJKVARN_LOW_HD inline
#endif

namespace sj_kvarn_lowbits {

// Read only bytes touched by a code, including the last code of a record.
template<int bits>
SJKVARN_LOW_HD uint32_t unpack(const uint8_t * payload, const int token, const int channel) {
    static_assert(bits == 2 || bits == 3, "Packed lower-bit code required");
    const unsigned bit = (token*256u + channel)*bits;
    const unsigned shift = bit & 7;
    const unsigned byte = bit >> 3;
    unsigned value = payload[byte];
    if (shift + bits > 8) {
        value |= unsigned(payload[byte + 1]) << 8;
    }
    return (value >> shift) & ((1u << bits) - 1);
}

// Reconstruct the existing MMA nibble word in registers without expanding the cache.
template<int bits, bool is_v>
SJKVARN_LOW_HD uint32_t fragment_word(const uint8_t * payload, const int word) {
    static_assert(bits >= 2 && bits <= 4, "Unsupported width");
    if constexpr (bits == 4) {
        return reinterpret_cast<const uint32_t *>(payload)[word];
    } else {
        const int lane = word & 31;
        const int tile = (word >> 5) & 15;
        const int strip = word >> 9;
        uint32_t result = 0;
        for (int nibble = 0; nibble < 8; ++nibble) {
            const int l = nibble & 3;
            const int e = nibble >> 2;
            const int row = (l & 1)*8 + (lane >> 2);
            const int col = 2*((l >> 1)*4 + (lane & 3)) + e;
            const int token = strip*16 + (is_v ? col : row);
            const int channel = tile*16 + (is_v ? row : col);
            result |= unpack<bits>(payload, token, channel) << (4*nibble);
        }
        return result;
    }
}

// Map fragment-ordered metadata slots to natural channel order for 2/3-bit records.
SJKVARN_LOW_HD int k_channel(const int slot) {
    const int lane_class = slot / 64;
    const int tile = (slot % 64) / 4;
    const int pair = (slot % 4) / 2;
    return tile*16 + 2*(lane_class + 4*pair) + (slot & 1);
}

SJKVARN_LOW_HD int v_channel(const int slot) {
    return ((slot % 32) / 2)*16 + slot/32 + (slot & 1)*8;
}

}

#undef SJKVARN_LOW_HD
