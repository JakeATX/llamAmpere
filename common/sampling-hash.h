#pragma once

// Counter-keyed target sampler (LLAMA_RTB_HASHSAMPLE=1, a TEST mode, default off).
//
// The final draw of the sampler chain becomes a Gumbel-max whose noise is a hash of (seed, output index, token id):
// token = argmax_i (logit_i + G(seed, n, id_i)), G = -log(-log(u)), u uniform in (0, 1) from the hash. This samples
// exactly softmax(logits), like the dist sampler, but the noise of output index n does not depend on how many random
// numbers were drawn before. With speculation verified by id match (the server forces that in this mode), spec on
// and spec off then emit the same tokens at any temperature, so a T = 1 speed gate can prove "no output change" by
// comparing ids. Not a production verify: p/q accepts more than id match.

#include <cmath>
#include <cstdint>

namespace rtb {

inline uint64_t hash_mix(uint64_t z) {
    z ^= z >> 33;
    z *= 0xff51afd7ed558ccdull;
    z ^= z >> 33;
    z *= 0xc4ceb9fe1a85ec53ull;
    z ^= z >> 33;
    return z;
}

// u in (0, 1), 53 bits, keyed by (seed, n, id)
inline double hash_uniform(uint64_t seed, uint64_t n, int32_t id) {
    const uint64_t h = hash_mix(hash_mix(seed * 0x9e3779b97f4a7c15ull + 0x632be59bd9b4e019ull) ^
                                hash_mix(n + 0x8cb92ba72f3d8dd7ull) ^ ((uint64_t) (uint32_t) id * 0xd1342543de82ef95ull));
    return ((double) (h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

inline double hash_gumbel(uint64_t seed, uint64_t n, int32_t id) {
    return -std::log(-std::log(hash_uniform(seed, n, id)));
}

} // namespace rtb
