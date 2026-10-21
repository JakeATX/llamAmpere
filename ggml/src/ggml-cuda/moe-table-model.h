#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace moe_tp {
struct ModelLayer {
    float scale;
    uint32_t training_rows, flags, top_n;
    std::array<uint64_t, 3> projection_bytes;
    std::vector<int16_t> bias, weights;
    bool active() const { return flags & 1; }
    bool positive_only() const { return flags & 2; }
};
struct Model {
    unsigned lead, method;
    std::array<uint8_t, 32> payload_sha256;
    std::vector<ModelLayer> layers;
};
// Throws on malformed dimensions, signed coefficients, metadata or payload hash.
Model load_model(const std::string & path, unsigned expected_layers = 0);
}
